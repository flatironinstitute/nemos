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
        with patch("nemos.glm_hmm.glm_hmm.setup_glm_hmm_initialization") as mock_setup:
            mock_setup.return_value = DEFAULT_INIT_FUNCTIONS_GLMHMM
            GLMHMM(n_states=2, initialization_funcs=input_dict)
        mock_setup.assert_called_once()
        assert mock_setup.call_args.kwargs["init_funcs"]["glm_params_init"] == mock_func

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
# TestGLMHMMSetup — setup() method: arg routing and result storage
# =============================================================================


class TestGLMHMMSetup:

    @pytest.mark.parametrize("func_name", FUNC_NAMES)
    def test_setup_forwards_args_and_stores_result(self, func_name):
        mock_func = _get_mock_func(func_name)
        mock_result = MagicMock()
        model = GLMHMM(n_states=2)
        with patch("nemos.glm_hmm.glm_hmm.setup_glm_hmm_initialization") as mock_setup:
            mock_setup.return_value = mock_result
            model.setup(
                **{func_name: mock_func, func_name + "_kwargs": MOCK_VALID_KWARGS}
            )
        assert mock_setup.call_args.kwargs[func_name] is mock_func
        assert mock_setup.call_args.kwargs[func_name + "_kwargs"] == MOCK_VALID_KWARGS
        assert model.initialization_funcs is mock_result

    @pytest.mark.parametrize(
        "provided_names",
        [
            list(combo)
            for r in range(2, len(FUNC_NAMES) + 1)
            for combo in itertools.combinations(FUNC_NAMES, r)
        ],
    )
    def test_setup_partial_args_forwarded(self, provided_names):
        mock_funcs = {fn: _get_mock_func(fn) for fn in provided_names}
        mock_result = MagicMock()
        model = GLMHMM(n_states=2)
        with patch("nemos.glm_hmm.glm_hmm.setup_glm_hmm_initialization") as mock_setup:
            mock_setup.return_value = mock_result
            model.setup(**{fn: mock_funcs[fn] for fn in provided_names})
        for fn in provided_names:
            assert mock_setup.call_args.kwargs[fn] is mock_funcs[fn]
        for fn in FUNC_NAMES:
            if fn not in provided_names:
                assert mock_setup.call_args.kwargs[fn] is None
        assert model.initialization_funcs is mock_result

    def test_setup_consecutive_calls_pass_accumulated_result_as_init_funcs(self):
        first_result = MagicMock()
        second_result = MagicMock()
        model = GLMHMM(n_states=2)
        with patch("nemos.glm_hmm.glm_hmm.setup_glm_hmm_initialization") as mock_setup:
            mock_setup.return_value = first_result
            model.setup(glm_params_init=_get_mock_func("glm_params_init"))
        with patch("nemos.glm_hmm.glm_hmm.setup_glm_hmm_initialization") as mock_setup:
            mock_setup.return_value = second_result
            model.setup(scale_init=_get_mock_func("scale_init"))
        assert mock_setup.call_args.kwargs["init_funcs"] is first_result


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
