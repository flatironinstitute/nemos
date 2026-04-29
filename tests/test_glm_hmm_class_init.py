"""Tests for GLMHMM class __init__, property setters, and setup method."""

import itertools
from unittest.mock import MagicMock, create_autospec, patch

import jax
import jax.numpy as jnp
import pytest

from nemos.glm_hmm.glm_hmm import GLMHMM
from nemos.glm_hmm.initialize_parameters import (
    DEFAULT_INIT_FUNCTIONS_GLMHMM,
    kmeans_glm_params_init,
    kmeans_scale_init,
)
from nemos.hmm.initialize_parameters import (
    random_initial_proba_init,
    uniform_transition_proba_init,
)

# =============================================================================
# Mock infrastructure
# =============================================================================


def _glm_mock_template(
    n_states, X, y, inverse_link_function, is_new_session, random_key, param1=None
):
    pass


def _hmm_mock_template(n_states, X, y, is_new_session, random_key, param1=None):
    pass


FUNC_NAMES = [
    "glm_params_init",
    "scale_init",
    "initial_proba_init",
    "transition_proba_init",
]
_GLM_FUNC_NAMES = {"glm_params_init", "scale_init"}
MOCK_VALID_KWARGS = {"param1": 0.5}

VALID_STRINGS = {
    "glm_params_init": "kmeans",
    "scale_init": "kmeans",
    "initial_proba_init": "random",
    "transition_proba_init": "uniform",
}

VALID_STRINGS_EXPECTED_FUNCS = {
    "glm_params_init": kmeans_glm_params_init,
    "scale_init": kmeans_scale_init,
    "initial_proba_init": random_initial_proba_init,
    "transition_proba_init": uniform_transition_proba_init,
}


def _get_mock_func(func_name):
    template = (
        _glm_mock_template if func_name in _GLM_FUNC_NAMES else _hmm_mock_template
    )
    return create_autospec(template, return_value=None)


def _get_mock_registry():
    return {fn: _get_mock_func(fn) for fn in FUNC_NAMES}


def _get_defining_class_and_prop(attr):
    for cls in GLMHMM.__mro__[1:]:
        if attr in cls.__dict__:
            return cls, cls.__dict__[attr]
    raise AttributeError(f"No defining class found for {attr!r}")


# =============================================================================
# TestGLMHMMInheritedSetters — one parametrized mock test per inherited attr
# =============================================================================


@pytest.mark.parametrize(
    "attr, init_kwargs",
    [
        ("n_states", {"n_states": 3}),
        ("maxiter", {"n_states": 2, "maxiter": 500}),
        ("tol", {"n_states": 2, "tol": 1e-6}),
        ("seed", {"n_states": 2, "seed": jax.random.PRNGKey(42)}),
        ("dirichlet_initial_proba", {"n_states": 2}),
        ("dirichlet_transition_proba", {"n_states": 2}),
    ],
)
def test_inherited_setter_called(attr, init_kwargs):
    """GLMHMM routes each inherited setter call to the base class setter."""
    defining_cls, original_prop = _get_defining_class_and_prop(attr)
    mock_fset = MagicMock(wraps=original_prop.fset)
    patched_prop = property(fget=original_prop.fget, fset=mock_fset)

    with patch.object(defining_cls, attr, new=patched_prop):
        GLMHMM(**init_kwargs)

    assert mock_fset.called


# =============================================================================
# TestGLMHMMInit — GLMHMM-specific setters and default values
# =============================================================================


class TestGLMHMMInit:

    # -------------------------------------------------------------------------
    # observation_model setter
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "obs_model",
        ["Poisson", "Bernoulli", "Gamma", "Gaussian", "NegativeBinomial"],
    )
    def test_observation_model_setter_string(self, obs_model):
        model = GLMHMM(n_states=2, observation_model=obs_model)
        assert obs_model in model.observation_model.__class__.__name__

    def test_observation_model_setter_instance(self):
        import nemos.observation_models as obs

        instance = obs.PoissonObservations()
        model = GLMHMM(n_states=2, observation_model=instance)
        assert model.observation_model is instance

    def test_observation_model_setter_invalid_string(self):
        with pytest.raises(ValueError, match="Unknown observation model"):
            GLMHMM(n_states=2, observation_model="InvalidModel")

    # -------------------------------------------------------------------------
    # inverse_link_function setter
    # -------------------------------------------------------------------------
    def test_inverse_link_function_none_uses_default(self):
        model = GLMHMM(
            n_states=2, observation_model="Poisson", inverse_link_function=None
        )
        assert model.inverse_link_function is not None

    def test_inverse_link_function_custom(self):
        custom_link = lambda x: x**2
        model = GLMHMM(n_states=2, inverse_link_function=custom_link)
        assert model.inverse_link_function is custom_link

    # -------------------------------------------------------------------------
    # initialization_funcs setter
    # -------------------------------------------------------------------------
    def test_initialization_funcs_none_uses_defaults(self):
        model = GLMHMM(n_states=2, initialization_funcs=None)
        assert model.initialization_funcs == DEFAULT_INIT_FUNCTIONS_GLMHMM

    def test_initialization_funcs_custom_callable_calls_setup(self):
        mock_func = _get_mock_func("glm_params_init")
        input_dict = {"glm_params_init": mock_func}
        with patch(
            "nemos.glm_hmm.glm_hmm.setup_glm_hmm_initialization"
        ) as mock_setup:
            mock_setup.return_value = DEFAULT_INIT_FUNCTIONS_GLMHMM
            GLMHMM(n_states=2, initialization_funcs=input_dict)
        mock_setup.assert_called_once()
        assert mock_setup.call_args.kwargs["init_funcs"] == input_dict

    # -------------------------------------------------------------------------
    # Default values and fit attributes
    # -------------------------------------------------------------------------
    def test_default_values(self):
        model = GLMHMM(n_states=3)
        assert model.n_states == 3
        assert model.maxiter == 1000
        assert model.tol == 1e-8
        assert "Bernoulli" in model.observation_model.__class__.__name__
        assert model.dirichlet_initial_proba is None
        assert model.dirichlet_transition_proba is None

    def test_fit_attributes_initialized_to_none(self):
        model = GLMHMM(n_states=3)
        assert model.coef_ is None
        assert model.intercept_ is None
        assert model.scale_ is None
        assert model.initial_prob_ is None
        assert model.transition_prob_ is None
        assert model.solver_state_ is None
        assert model.dof_resid_ is None

    # -------------------------------------------------------------------------
    # repr
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "obs_model", ["Poisson", "Bernoulli", "Gamma", "Gaussian", "NegativeBinomial"]
    )
    def test_repr(self, obs_model):
        model = GLMHMM(n_states=3, observation_model=obs_model)
        repr_str = repr(model)
        assert repr_str.startswith("GLMHMM(")
        assert obs_model in repr_str
        assert "n_states" in repr_str


# =============================================================================
# TestGLMHMMSetup — setup() method, parametrized over all four func names
# =============================================================================


@pytest.mark.parametrize("func_name", FUNC_NAMES)
class TestGLMHMMSetup:

    def test_setup_with_no_input_uses_defaults(self, func_name):
        model = GLMHMM(n_states=2)
        model.setup()
        assert model.initialization_funcs == DEFAULT_INIT_FUNCTIONS_GLMHMM

    def test_setup_function_by_string(self, func_name):
        model = GLMHMM(n_states=2)
        model.setup(**{func_name: VALID_STRINGS[func_name]})
        assert (
            model.initialization_funcs[func_name]
            is VALID_STRINGS_EXPECTED_FUNCS[func_name]
        )
        assert model.initialization_funcs[func_name + "_custom"] is False

    def test_setup_function_by_callable(self, func_name):
        mock_func = _get_mock_func(func_name)
        model = GLMHMM(n_states=2)
        model.setup(**{func_name: mock_func})
        assert model.initialization_funcs[func_name] is mock_func
        assert model.initialization_funcs[func_name + "_custom"] is True

    def test_setup_function_by_callable_with_kwargs(self, func_name):
        mock_func = _get_mock_func(func_name)
        model = GLMHMM(n_states=2)
        model.setup(**{func_name: mock_func, func_name + "_kwargs": MOCK_VALID_KWARGS})
        assert model.initialization_funcs[func_name] is mock_func
        assert model.initialization_funcs[func_name + "_kwargs"] == MOCK_VALID_KWARGS

    def test_setup_kwargs_only_validated_against_current_func(self, func_name):
        mock_func = _get_mock_func(func_name)
        model = GLMHMM(n_states=2)
        model.setup(**{func_name: mock_func})
        model.setup(**{func_name + "_kwargs": MOCK_VALID_KWARGS})
        assert model.initialization_funcs[func_name + "_kwargs"] == MOCK_VALID_KWARGS

    def test_setup_invalid_string_raises(self, func_name):
        model = GLMHMM(n_states=2)
        with pytest.raises(ValueError, match="Invalid initialization"):
            model.setup(**{func_name: "not_a_valid_string"})

    def test_setup_custom_function_missing_required_params_raises(self, func_name):
        model = GLMHMM(n_states=2)
        with pytest.raises(
            ValueError, match="Custom initialization function must have"
        ):
            model.setup(**{func_name: lambda x: x})

    def test_setup_invalid_kwargs_raises(self, func_name):
        mock_func = _get_mock_func(func_name)
        model = GLMHMM(n_states=2)
        with pytest.raises(ValueError, match="Invalid keyword argument"):
            model.setup(
                **{func_name: mock_func, func_name + "_kwargs": {"bad_param": 99}}
            )

    def test_setup_function_resets_kwargs(self, func_name):
        mock_func_a = _get_mock_func(func_name)
        mock_func_b = _get_mock_func(func_name)
        model = GLMHMM(n_states=2)
        model.setup(
            **{func_name: mock_func_a, func_name + "_kwargs": MOCK_VALID_KWARGS}
        )
        model.setup(**{func_name: mock_func_b})
        assert model.initialization_funcs[func_name + "_kwargs"] == {}

    def test_setup_does_not_affect_other_funcs(self, func_name):
        mock_func = _get_mock_func(func_name)
        model = GLMHMM(n_states=2)
        model.setup(**{func_name: mock_func})
        for other in FUNC_NAMES:
            if other != func_name:
                assert (
                    model.initialization_funcs[other]
                    is DEFAULT_INIT_FUNCTIONS_GLMHMM[other]
                )


# =============================================================================
# TestGLMHMMSetupMultiple — setup() with multiple functions at once
# =============================================================================


class TestGLMHMMSetupMultiple:

    def test_setup_all_funcs_at_once(self):
        mock_registry = _get_mock_registry()
        setup_kwargs = {}
        for fn in FUNC_NAMES:
            setup_kwargs[fn] = mock_registry[fn]
            setup_kwargs[fn + "_kwargs"] = MOCK_VALID_KWARGS
        model = GLMHMM(n_states=2)
        model.setup(**setup_kwargs)
        for fn in FUNC_NAMES:
            assert model.initialization_funcs[fn] is mock_registry[fn]
            assert model.initialization_funcs[fn + "_kwargs"] == MOCK_VALID_KWARGS

    def test_setup_all_pairs(self):
        for fn1, fn2 in itertools.combinations(FUNC_NAMES, 2):
            mock1, mock2 = _get_mock_func(fn1), _get_mock_func(fn2)
            model = GLMHMM(n_states=2)
            model.setup(
                **{
                    fn1: mock1,
                    fn1 + "_kwargs": MOCK_VALID_KWARGS,
                    fn2: mock2,
                    fn2 + "_kwargs": MOCK_VALID_KWARGS,
                }
            )
            assert model.initialization_funcs[fn1] is mock1
            assert model.initialization_funcs[fn2] is mock2
            for other in FUNC_NAMES:
                if other not in (fn1, fn2):
                    assert model.initialization_funcs[other + "_kwargs"] == {}

    def test_setup_consecutive_calls_accumulate(self):
        mock_glm = _get_mock_func("glm_params_init")
        mock_scale = _get_mock_func("scale_init")
        model = GLMHMM(n_states=2)
        model.setup(glm_params_init=mock_glm)
        model.setup(scale_init=mock_scale)
        assert model.initialization_funcs["glm_params_init"] is mock_glm
        assert model.initialization_funcs["scale_init"] is mock_scale
        assert (
            model.initialization_funcs["initial_proba_init"]
            is DEFAULT_INIT_FUNCTIONS_GLMHMM["initial_proba_init"]
        )


# =============================================================================
# TestGLMHMMCheckModelIsFit
# =============================================================================


class TestGLMHMMCheckModelIsFit:

    def _set_all_params(self, model):
        model.coef_ = jnp.zeros((3, 2))
        model.intercept_ = jnp.zeros((1, 2))
        model.scale_ = jnp.ones(2)

    def test_passes_when_all_params_set(self):
        model = GLMHMM(n_states=2)
        self._set_all_params(model)
        model._check_model_is_fit()

    @pytest.mark.parametrize("missing_attr", ["coef_", "intercept_", "scale_"])
    def test_raises_when_attr_is_none(self, missing_attr):
        model = GLMHMM(n_states=2)
        self._set_all_params(model)
        setattr(model, missing_attr, None)
        with pytest.raises(ValueError, match=missing_attr):
            model._check_model_is_fit()

    def test_raises_on_fresh_model(self):
        model = GLMHMM(n_states=2)
        with pytest.raises(ValueError):
            model._check_model_is_fit()
