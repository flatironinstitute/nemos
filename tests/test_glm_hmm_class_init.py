"""Tests for GLMHMM class __init__, property setters, and setup method."""

import itertools
from unittest.mock import MagicMock, create_autospec, patch

import jax
import jax.numpy as jnp
import pytest

from nemos._inspect_utils import extract_literal_options
from nemos.glm_hmm.glm_hmm import GLMHMM
from nemos.glm_hmm.initialize_parameters import (
    AVAIL_INIT_FUNCTIONS_GLM,
    DEFAULT_INIT_FUNCTIONS_GLMHMM,
    kmeans_glm_params_init,
    kmeans_scale_init,
)
from nemos.hmm.initialize_parameters import (
    AVAILABLE_INIT_FUNCTIONS,
    DEFAULT_INIT_FUNCTIONS,
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


_HMM_FUNC_NAMES = ["initial_proba_init", "transition_proba_init"]
_MODEL_FUNC_NAMES = ["glm_params_init", "scale_init"]
FUNC_NAMES = _MODEL_FUNC_NAMES + _HMM_FUNC_NAMES
_GLM_FUNC_NAMES = set(_MODEL_FUNC_NAMES)
MOCK_VALID_KWARGS = {"param1": 0.5}

_PATCH_PATH_HMM = "nemos.hmm.hmm.setup_hmm_initialization"
_PATCH_PATH_MODEL = "nemos.glm_hmm.glm_hmm.setup_glm_hmm_initialization"


def _patch_path_for(func_name):
    return _PATCH_PATH_MODEL if func_name in _GLM_FUNC_NAMES else _PATCH_PATH_HMM


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
    # initialization_funcs setters (HMM and model pipelines)
    # -------------------------------------------------------------------------
    def test_init_funcs_none_uses_defaults(self):
        """Without explicit init-funcs args, both pipelines load their defaults."""
        model = GLMHMM(n_states=2)
        assert dict(model.hmm_initialization_funcs) == dict(DEFAULT_INIT_FUNCTIONS)
        assert dict(model.model_initialization_funcs) == dict(
            DEFAULT_INIT_FUNCTIONS_GLMHMM
        )

    def test_hmm_initialization_funcs_passed_through_setter(self):
        """Custom callable in hmm_initialization_funcs reaches setup_hmm_initialization."""
        mock_func = _get_mock_func("initial_proba_init")
        with patch(_PATCH_PATH_HMM) as mock_setup:
            mock_setup.return_value = dict(DEFAULT_INIT_FUNCTIONS)
            GLMHMM(
                n_states=2,
                hmm_initialization_funcs={"initial_proba_init": mock_func},
            )
        # at least one call from the setter; the most recent one carries the custom func
        # in the merged init_funcs dict
        assert mock_setup.call_args.kwargs["init_funcs"]["initial_proba_init"] is (
            mock_func
        )

    def test_model_initialization_funcs_passed_through_setter(self):
        """Custom callable in model_initialization_funcs reaches setup_glm_hmm_initialization."""
        mock_func = _get_mock_func("glm_params_init")
        with patch(_PATCH_PATH_MODEL) as mock_setup:
            mock_setup.return_value = dict(DEFAULT_INIT_FUNCTIONS_GLMHMM)
            GLMHMM(
                n_states=2,
                model_initialization_funcs={"glm_params_init": mock_func},
            )
        assert mock_setup.call_args.kwargs["init_funcs"]["glm_params_init"] is mock_func

    # -------------------------------------------------------------------------
    # _model_use_kmeans flag — derived from function identity
    # -------------------------------------------------------------------------
    def test_model_use_kmeans_false_by_default(self):
        model = GLMHMM(n_states=2)
        assert model._model_use_kmeans == {
            "glm_params_init": False,
            "scale_init": False,
        }

    @pytest.mark.parametrize("key", _MODEL_FUNC_NAMES)
    def test_model_use_kmeans_true_via_string(self, key):
        model = GLMHMM(n_states=2)
        model.setup(**{key: "kmeans"})
        assert model._model_use_kmeans[key] is True

    @pytest.mark.parametrize(
        "key, kmeans_callable",
        [
            ("glm_params_init", kmeans_glm_params_init),
            ("scale_init", kmeans_scale_init),
        ],
    )
    def test_model_use_kmeans_true_via_callable_in_init_dict(
        self, key, kmeans_callable
    ):
        """Passing the kmeans callable directly in model_initialization_funcs sets the flag."""
        model = GLMHMM(n_states=2, model_initialization_funcs={key: kmeans_callable})
        assert model._model_use_kmeans[key] is True

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
    """``setup()`` splits HMM keys (-> setup_hmm_initialization) from
    model keys (-> setup_glm_hmm_initialization). Each pipeline is patched
    independently below."""

    @pytest.mark.parametrize("func_name", FUNC_NAMES)
    def test_setup_forwards_args_to_correct_pipeline(self, func_name):
        """Each key reaches the setup function for its own pipeline with its kwargs."""
        mock_func = _get_mock_func(func_name)
        mock_result = MagicMock()
        model = GLMHMM(n_states=2)
        with patch(_patch_path_for(func_name)) as mock_setup:
            mock_setup.return_value = mock_result
            model.setup(
                **{func_name: mock_func, func_name + "_kwargs": MOCK_VALID_KWARGS}
            )
        assert mock_setup.call_args.kwargs[func_name] is mock_func
        assert mock_setup.call_args.kwargs[func_name + "_kwargs"] == MOCK_VALID_KWARGS

    @pytest.mark.parametrize("func_name", FUNC_NAMES)
    def test_setup_stores_pipeline_result(self, func_name):
        """The return of the relevant setup_* fn is stored on the matching attr."""
        mock_func = _get_mock_func(func_name)
        mock_result = MagicMock()
        model = GLMHMM(n_states=2)
        with patch(_patch_path_for(func_name)) as mock_setup:
            mock_setup.return_value = mock_result
            model.setup(**{func_name: mock_func})
        if func_name in _GLM_FUNC_NAMES:
            assert model._model_initialization_funcs is mock_result
        else:
            assert model._hmm_initialization_funcs is mock_result

    @pytest.mark.parametrize(
        "provided_names",
        [
            list(combo)
            for r in range(2, len(FUNC_NAMES) + 1)
            for combo in itertools.combinations(FUNC_NAMES, r)
        ],
    )
    def test_setup_partial_args_forwarded(self, provided_names):
        """Each pipeline receives only its own keys; the rest stay at None."""
        mock_funcs = {fn: _get_mock_func(fn) for fn in provided_names}
        model = GLMHMM(n_states=2)
        with patch(_PATCH_PATH_HMM) as mock_hmm, patch(_PATCH_PATH_MODEL) as mock_model:
            mock_hmm.return_value = MagicMock()
            mock_model.return_value = MagicMock()
            model.setup(**{fn: mock_funcs[fn] for fn in provided_names})

        def assert_pipeline(mock_setup, pipeline_keys):
            for fn in pipeline_keys:
                expected = mock_funcs[fn] if fn in provided_names else None
                assert mock_setup.call_args.kwargs[fn] is expected

        assert_pipeline(mock_hmm, _HMM_FUNC_NAMES)
        assert_pipeline(mock_model, _MODEL_FUNC_NAMES)

    @pytest.mark.parametrize("func_name", FUNC_NAMES)
    def test_setup_consecutive_calls_thread_init_funcs(self, func_name):
        """A second setup() targeting a different key in the same pipeline sees
        the first call's result as ``init_funcs`` — so previously configured
        keys are preserved across calls."""
        pipeline = (
            _MODEL_FUNC_NAMES if func_name in _GLM_FUNC_NAMES else _HMM_FUNC_NAMES
        )
        other_key = next(k for k in pipeline if k != func_name)

        first_result = MagicMock()
        model = GLMHMM(n_states=2)
        patch_path = _patch_path_for(func_name)

        with patch(patch_path) as mock_setup:
            mock_setup.return_value = first_result
            model.setup(**{func_name: _get_mock_func(func_name)})

        with patch(patch_path) as mock_setup:
            mock_setup.return_value = MagicMock()
            model.setup(**{other_key: _get_mock_func(other_key)})
            assert mock_setup.call_args.kwargs["init_funcs"] is first_result

    # -------------------------------------------------------------------------
    # Literal-vs-registry consistency
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "param_name, registry",
        [
            ("initial_proba_init", AVAILABLE_INIT_FUNCTIONS["initial_proba_init"]),
            (
                "transition_proba_init",
                AVAILABLE_INIT_FUNCTIONS["transition_proba_init"],
            ),
            ("glm_params_init", AVAIL_INIT_FUNCTIONS_GLM["glm_params_init"]),
            ("scale_init", AVAIL_INIT_FUNCTIONS_GLM["scale_init"]),
        ],
    )
    def test_setup_literal_options_match_registry(self, param_name, registry):
        """``setup()`` Literal annotations must enumerate exactly the built-in
        string aliases declared in the init-function registries. If a new
        built-in is added (or one is removed) without updating the signature,
        this test fails — preventing silent drift."""
        literals = extract_literal_options(GLMHMM.setup, param_name)
        assert literals == set(registry.keys()), (
            f"Literal options for {param_name!r} in GLMHMM.setup ({literals}) "
            f"differ from registered keys ({set(registry.keys())})."
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
