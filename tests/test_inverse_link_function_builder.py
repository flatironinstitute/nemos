import jax.numpy as jnp
import numpy as np
import pytest

import nemos as nmo
from nemos._inverse_link_function_builder import (
    AVAILABLE_INVERSE_LINK_FUNCTIONS,
    LINK_NAME_TO_FUNC,
    instantiate_inverse_link_function,
)
from nemos.inverse_link_function_utils import (
    exp,
    expit,
    identity,
    log_softmax,
    logistic,
    norm_cdf,
    resolve_inverse_link_function,
    softplus,
)
from nemos.utils import _get_name, one_over_x

# one entry per distinct function (the registry maps many names to the same function)
UNIQUE_LINK_FUNCTIONS = list(dict.fromkeys(LINK_NAME_TO_FUNC.values()))

EXPECTED_FUNCTION = {
    "exp": exp,
    "expit": expit,
    "identity": identity,
    "log_softmax": log_softmax,
    "logistic": logistic,
    "norm_cdf": norm_cdf,
    "one_over_x": one_over_x,
    "softplus": softplus,
}


def assert_computes_same_values(link_function, expected_function):
    """Check two link functions agree numerically.

    Identity (``is``) is too strict: depending on the jax version, some saved names
    are shared by equivalent functions (e.g. ``logistic`` and ``expit``).
    """
    x = jnp.array([0.5, 1.0, 2.0])
    np.testing.assert_allclose(link_function(x), expected_function(x))


class TestInstantiateInverseLinkFunction:
    """String name -> inverse link function."""

    @pytest.mark.parametrize("link_name", AVAILABLE_INVERSE_LINK_FUNCTIONS)
    def test_every_available_name_resolves(self, link_name):
        link_function = instantiate_inverse_link_function(link_name)
        assert link_function is LINK_NAME_TO_FUNC[link_name]
        assert callable(link_function)

    @pytest.mark.parametrize("link_name", AVAILABLE_INVERSE_LINK_FUNCTIONS)
    def test_name_returns_the_right_function(self, link_name):
        assert (
            instantiate_inverse_link_function(link_name) is EXPECTED_FUNCTION[link_name]
        )

    @pytest.mark.parametrize("link_name", ["not_a_link", "", "EXP", "jax.numpy.log"])
    def test_unknown_name_raises(self, link_name):
        with pytest.raises(ValueError, match="Unknown link function"):
            instantiate_inverse_link_function(link_name)

    def test_error_message_lists_only_short_names(self):
        with pytest.raises(ValueError, match=r"('softplus'|\"softplus\")") as error:
            instantiate_inverse_link_function("not_a_link")
        assert "jax." not in str(error.value)
        assert "nemos." not in str(error.value)

    @pytest.mark.parametrize("link_name", AVAILABLE_INVERSE_LINK_FUNCTIONS)
    def test_resolve_uses_builder_for_strings(self, link_name):
        observation_model = nmo.observation_models.PoissonObservations()
        resolved = resolve_inverse_link_function(link_name, observation_model)
        assert resolved is instantiate_inverse_link_function(link_name)


class TestInverseLinkFunctionSaveLoad:
    """Function -> saved name -> function, as used by ``save_params``/``load_model``."""

    @pytest.mark.parametrize(
        "link_function", UNIQUE_LINK_FUNCTIONS, ids=lambda f: f.__name__
    )
    def test_saved_name_maps_back_to_same_function(self, link_function):
        """The name a model saves for its link function must be in the registry."""
        saved_name = _get_name(link_function)
        assert_computes_same_values(
            instantiate_inverse_link_function(saved_name), link_function
        )

    @pytest.mark.parametrize(
        "link_function", UNIQUE_LINK_FUNCTIONS, ids=lambda f: f.__name__
    )
    def test_save_load_model_round_trip(self, link_function, tmp_path):
        if link_function.__name__ == "log_softmax":
            model = nmo.glm.ClassifierGLM(
                classes=3, inverse_link_function=link_function
            )
        else:
            model = nmo.glm.GLM(inverse_link_function=link_function)
        save_path = tmp_path / "model.npz"
        model.save_params(save_path)
        loaded_model = nmo.load_model(save_path)
        assert_computes_same_values(loaded_model.inverse_link_function, link_function)
