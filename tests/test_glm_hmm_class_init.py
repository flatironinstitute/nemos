"""Tests for GLMHMM class __init__ and property setters."""

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from sklearn.base import clone

from nemos._observation_model_builder import AVAILABLE_OBSERVATION_MODELS
from nemos.glm_hmm.glm_hmm import GLMHMM

# =============================================================================
# Helpers
# =============================================================================


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
# TestGLMHMMSklearnClone — whole-state round-trip through sklearn clone
# =============================================================================


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
    """``sklearn.base.clone`` rebuilds the model from ``get_params``."""

    def test_clone_default(self):
        model = GLMHMM(n_states=2)
        compare_hmm_models(model, clone(model))
