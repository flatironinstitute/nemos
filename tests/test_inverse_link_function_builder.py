import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nemos as nmo
from nemos._inverse_link_function_builder import (
    AVAILABLE_INVERSE_LINK_FUNCTIONS,
    LINK_NAME_TO_FUNC,
    instantiate_inverse_link_function,
)
from nemos.inverse_link_function_utils import resolve_inverse_link_function
from nemos.utils import _get_name

# one entry per distinct function (the registry maps many names to the same function)
UNIQUE_LINK_FUNCTIONS = list(dict.fromkeys(LINK_NAME_TO_FUNC.values()))


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

    @pytest.mark.parametrize(
        "link_name, reference_function",
        [
            ("exp", jnp.exp),
            ("jax.numpy.exp", jnp.exp),
            ("softplus", jax.nn.softplus),
            ("jax.nn.softplus", jax.nn.softplus),
            ("logistic", jax.lax.logistic),
            ("expit", jax.scipy.special.expit),
            ("norm_cdf", jax.scipy.stats.norm.cdf),
            ("jax.scipy.stats.norm.cdf", jax.scipy.stats.norm.cdf),
            ("one_over_x", lambda x: 1.0 / x),
            ("identity", lambda x: x),
            ("log_softmax", jax.nn.log_softmax),
        ],
    )
    def test_name_returns_the_right_function(self, link_name, reference_function):
        """The returned function computes the same values as the function it names."""
        link_function = instantiate_inverse_link_function(link_name)
        assert_computes_same_values(link_function, reference_function)

    @pytest.mark.parametrize("link_name", ["not_a_link", "", "EXP", "jax.numpy.log"])
    def test_unknown_name_raises(self, link_name):
        with pytest.raises(ValueError, match="Unknown link function"):
            instantiate_inverse_link_function(link_name)

    def test_error_message_lists_available_names(self):
        with pytest.raises(ValueError, match="softplus"):
            instantiate_inverse_link_function("not_a_link")

    @pytest.mark.parametrize("link_name", ["exp", "softplus", "identity"])
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

    @pytest.mark.xfail(
        reason="Wrapped link functions copy jax's __module__/__qualname__, "
        "so the standard pickle module cannot find them. Pre-existing on development.",
        raises=pickle.PicklingError,
        strict=True,
    )
    def test_pickle_round_trip(self):
        link_function = instantiate_inverse_link_function("exp")
        assert pickle.loads(pickle.dumps(link_function)) is link_function
