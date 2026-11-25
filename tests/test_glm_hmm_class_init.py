"""Tests for GLMHMM class __init__ and property setters."""

from unittest.mock import MagicMock, create_autospec, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from sklearn.base import clone

from nemos._observation_model_builder import AVAILABLE_OBSERVATION_MODELS
from nemos.glm_hmm.glm_hmm import GLMHMM
from nemos.glm_hmm.initialize_parameters import (
    DEFAULT_INIT_FUNCTIONS_GLMHMM,
    kmeans_glm_params_init,
    kmeans_scale_init,
)
from nemos.hmm.initialize_parameters import (
    DEFAULT_INIT_FUNCTIONS,
    random_initial_proba_init,
    uniform_transition_proba_init,
)

# =============================================================================
# Mock infrastructure
# =============================================================================


def _glm_mock_template(
    n_states, X, y, inverse_link_function, session_starts, random_key, param1=None
):
    pass


def _hmm_mock_template(n_states, X, y, session_starts, random_key, param1=None):
    pass


_HMM_FUNC_NAMES = ["initial_proba_init", "transition_proba_init"]
_MODEL_FUNC_NAMES = ["glm_params_init", "scale_init"]
FUNC_NAMES = _MODEL_FUNC_NAMES + _HMM_FUNC_NAMES
_GLM_FUNC_NAMES = set(_MODEL_FUNC_NAMES)
MOCK_VALID_KWARGS = {"param1": 0.5}

_PATCH_PATH_HMM = "nemos.hmm.hmm.setup_hmm_initialization"
_PATCH_PATH_MODEL = "nemos.glm_hmm.glm_hmm.setup_glm_hmm_initialization"


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

    @pytest.mark.parametrize("obs_model", AVAILABLE_OBSERVATION_MODELS)
    def test_fit_attributes_initialized_to_none(self, obs_model):
        model = GLMHMM(n_states=3, observation_model=obs_model)
        assert model.coef_ is None
        assert model.intercept_ is None
        assert model.initial_prob_ is None
        assert model.transition_prob_ is None
        assert model.solver_state_ is None
        assert model.dof_resid_ is None
        for k in model.observation_model.extra_params:
            assert getattr(model, k + "_", 1) is None, (
                "Attribute `{}_` has not been set".format(k)
            )

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
# TestGLMHMMCheckModelIsFit
# =============================================================================


class TestGLMHMMCheckModelIsFit:
    def _set_all_params(self, model):
        model.coef_ = jnp.zeros((3, 2))
        model.intercept_ = jnp.zeros((1, 2))
        model.initial_prob_ = jnp.zeros((2,))
        model.transition_prob_ = jnp.zeros((2, 2))
        for k in model.observation_model.extra_params:
            setattr(model, k + "_", jnp.zeros((1, 2)))

    def test_passes_when_all_params_set(self):
        model = GLMHMM(n_states=2)
        self._set_all_params(model)
        model._check_model_is_fit()

    @pytest.mark.parametrize(
        "missing_attr", ["coef_", "intercept_", "scale_", "dispersion_"]
    )
    @pytest.mark.parametrize("obs_model", AVAILABLE_OBSERVATION_MODELS)
    def test_raises_when_attr_is_none(self, missing_attr, obs_model):
        model = GLMHMM(n_states=2, observation_model=obs_model)
        if not hasattr(model, missing_attr):
            pytest.skip(
                f"skipping attr {missing_attr} for observation {obs_model}, "
                f"since it is not required."
            )
        self._set_all_params(model)
        setattr(model, missing_attr, None)
        with pytest.raises(ValueError, match=missing_attr):
            model._check_model_is_fit()

    def test_raises_on_fresh_model(self):
        model = GLMHMM(n_states=2)
        with pytest.raises(ValueError):
            model._check_model_is_fit()


# =============================================================================
# TestGLMHMMSklearnClone — identity round-trip of the init-func dicts
# =============================================================================


def _get_custom_func(func_name):
    # plain functions, not autospec mocks: sklearn clone deepcopies params, and
    # deepcopy preserves identity for functions but returns a new object for mocks
    return _glm_mock_template if func_name in _GLM_FUNC_NAMES else _hmm_mock_template


def _init_funcs_attr(func_name):
    return (
        "model_initialization_funcs"
        if func_name in _GLM_FUNC_NAMES
        else "hmm_initialization_funcs"
    )


def _use_kmeans_attr(func_name):
    return "_model_use_kmeans" if func_name in _GLM_FUNC_NAMES else "_hmm_use_kmeans"


def _filter_attributes(obj, exclude_keys):
    return {k: v for k, v in obj.__dict__.items() if k not in exclude_keys}


# Attributes needing bespoke comparison in compare_hmm_models; everything else
# in __dict__ must compare equal under plain == (catch-all below).
_ARRAY_STATE_ATTRS = (
    "_seed",
    "_dirichlet_initial_proba",
    "_dirichlet_transition_proba",
)
_INSTANCE_STATE_ATTRS = ("_regularizer", "_observation_model")
_LAZY_STATE_ATTRS = ("_solver_spec",)


def compare_hmm_models(m1, m2):
    """Stringent whole-state comparison between two distinct HMM-based models.

    Default-deny: any ``__dict__`` attribute not explicitly handled below must
    compare equal under plain ``==`` (callables compare by identity), so newly
    added state is covered automatically and fails loudly if it needs bespoke
    handling.
    """
    assert m1 is not m2
    assert type(m1) is type(m2)
    # arrays (or None): == inside the catch-all dict comparison would raise
    for attr in _ARRAY_STATE_ATTRS:
        np.testing.assert_array_equal(m1.__dict__[attr], m2.__dict__[attr])
    # instances without __eq__: compare class and full state
    for attr in _INSTANCE_STATE_ATTRS:
        v1, v2 = m1.__dict__[attr], m2.__dict__[attr]
        assert type(v1) is type(v2)
        assert v1.__dict__ == v2.__dict__
    # _solver_spec is resolved lazily by the solver_name property, so one side
    # may hold None and the other a resolved SolverSpec; compare the resolved
    # name instead of the stored value
    assert m1.solver_name == m2.solver_name
    exclude = _ARRAY_STATE_ATTRS + _INSTANCE_STATE_ATTRS + _LAZY_STATE_ATTRS
    assert _filter_attributes(m1, exclude) == _filter_attributes(m2, exclude)


class TestGLMHMMSklearnClone:
    """``sklearn.base.clone`` rebuilds the model from ``get_params`` and requires
    each param of the rebuilt model to be the same object it passed to ``__init__``;
    the ``*_initialization_funcs`` setters must therefore round-trip an already
    resolved dict container by identity."""

    def test_clone_default(self):
        model = GLMHMM(n_states=2)
        compare_hmm_models(model, clone(model))

    @pytest.mark.parametrize("func_name", FUNC_NAMES)
    @pytest.mark.parametrize(
        "value_kind, kwargs",
        [
            ("string", None),
            ("string", {}),
            ("kmeans", None),
            ("custom", None),
            ("custom", {}),
            ("custom", MOCK_VALID_KWARGS),
        ],
    )
    def test_clone_single_func(self, func_name, value_kind, kwargs):
        if value_kind == "kmeans":
            value = "kmeans"
        elif value_kind == "string":
            value = VALID_STRINGS[func_name]
        else:
            value = _get_custom_func(func_name)
        config = {func_name: value}
        if kwargs is not None:
            config[f"{func_name}_kwargs"] = kwargs
        model = GLMHMM(n_states=2, **{_init_funcs_attr(func_name): config})
        cloned = clone(model)
        compare_hmm_models(model, cloned)
        if value_kind == "kmeans":
            # the use-kmeans flag must survive the get_params -> __init__ round
            # trip, not just the callable itself
            assert getattr(cloned, _use_kmeans_attr(func_name))[func_name]

    def test_clone_all_funcs_custom(self):
        hmm_config, model_config = {}, {}
        for func_name in FUNC_NAMES:
            config = model_config if func_name in _GLM_FUNC_NAMES else hmm_config
            config[func_name] = _get_custom_func(func_name)
            config[f"{func_name}_kwargs"] = dict(MOCK_VALID_KWARGS)
        model = GLMHMM(
            n_states=2,
            hmm_initialization_funcs=hmm_config,
            model_initialization_funcs=model_config,
        )
        compare_hmm_models(model, clone(model))
