"""Tests for GLMHMM.fit and related fit-path validation."""

import inspect
import math
import warnings
from contextlib import nullcontext as does_not_raise
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pynapple as nap
import pytest

import nemos as nmo
from nemos.glm.params import GLMParams
from nemos.glm_hmm.glm_hmm import GLMHMM
from nemos.glm_hmm.params import GLMHMMModelParams, GLMHMMParams
from nemos.glm_hmm.validation import GLMHMMValidator
from nemos.hmm.expectation_maximization import EMState
from nemos.hmm.hmm import BaseHMM
from nemos.hmm.params import HMMParams
from nemos.regularizer import Ridge, UnRegularized
from nemos.utils import _get_name
from tests.conftest import instantiate_glm_hmm_func

# ---------------------------------------------------------------------------
# Parametrize lists: GLMHMM only, single obs_model (Bernoulli).
# Extend obs_model list here when more observation models are implemented.
# ---------------------------------------------------------------------------

INSTANTIATE_MODEL_ONLY = [
    {"model": "GLMHMM", "obs_model": "Bernoulli", "simulate": False}
]
INSTANTIATE_MODEL_AND_SIMULATE = [
    {"model": "GLMHMM", "obs_model": "Bernoulli", "simulate": True}
]

DEFAULT_GLM_COEF_SHAPE = {
    "GLMHMM": (2, 3),  # (n_features, n_states)
}

N_STATES = 3
N_FEATURES = 2


# ---------------------------------------------------------------------------
# Shared helper: build a valid 5-tuple of init params from the fixture
# ---------------------------------------------------------------------------


def _init_params_from_fixture(fixture):
    return (
        fixture.params.model_params.coef,
        fixture.params.model_params.intercept,
        fixture.params.model_params.log_scale,
        jnp.exp(fixture.params.hmm_params.log_initial_prob),
        jnp.exp(fixture.params.hmm_params.log_transition_prob),
    )


# ---------------------------------------------------------------------------
# test_get_fit_attrs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "instantiate_base_regressor_subclass", INSTANTIATE_MODEL_ONLY, indirect=True
)
def test_get_fit_attrs(instantiate_base_regressor_subclass, mock_glm_hmm_optimizer_run):
    """_get_fit_state returns all-None before fit, all non-None after fit."""
    fixture = instantiate_base_regressor_subclass
    expected_state = {
        "coef_": None,
        "dof_resid_": None,
        "initial_prob_": None,
        "intercept_": None,
        "scale_": None,
        "solver_state_": None,
        "transition_prob_": None,
    }
    assert fixture.model._get_fit_state() == expected_state
    fixture.model.fit(fixture.X, fixture.y)
    state = fixture.model._get_fit_state()
    assert state.keys() == expected_state.keys()
    n_features = fixture.X.shape[1]
    assert state["coef_"].shape == (n_features, N_STATES)
    assert state["intercept_"].shape == (N_STATES,)
    assert state["scale_"].shape == (N_STATES,)
    assert jnp.all(state["scale_"] > 0)
    assert state["initial_prob_"].shape == (N_STATES,)
    assert jnp.allclose(state["initial_prob_"].sum(), 1.0)
    assert state["transition_prob_"].shape == (N_STATES, N_STATES)
    assert jnp.allclose(state["transition_prob_"].sum(axis=-1), jnp.ones(N_STATES))
    assert float(jnp.squeeze(state["dof_resid_"])) > 0
    # mock returns SimpleNamespace(iterations=1, converged=True)
    assert state["solver_state_"].iterations == 1
    assert state["solver_state_"].converged is True


# ---------------------------------------------------------------------------
# TestGLMHMM — fit-method tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "instantiate_base_regressor_subclass", INSTANTIATE_MODEL_AND_SIMULATE, indirect=True
)
class TestGLMHMM:
    """Unit tests for GLMHMM.fit that are observation-model-agnostic."""

    @pytest.mark.requires_x64
    def test_fit_accepts_bool_y(self, instantiate_base_regressor_subclass):
        """Test that bool array are valid inputs.

        Notes
        -----
        The test requires x64 to generate a corner case. The initialization function
        for the intercept calls a``jnp.nanmean``, which for bool arrays returns x32
        regardless of the jax configuration.
        If we do not cast y to float (inheriting the dtype from the configurations)
        the dtype change will raise at compilation time. This bug emerges only if the
        jax configuration is enabling x64, otherwise everything would remain x32 and
        the compilation would not raise.
        """
        fixture = instantiate_base_regressor_subclass
        n = 10
        X = fixture.X[:n]
        y = fixture.y[:n]
        y = y.astype(bool)
        fixture.model.fit(X, y)

    def test_fit_pynapple_tsd(self, instantiate_base_regressor_subclass):
        """Pynapple TSD/TsdFrame accepted; session boundaries affect the fit."""
        fixture = instantiate_base_regressor_subclass

        n = 50
        X_np = fixture.X[:n]
        y_np = fixture.y[:n]
        X_nap = nap.TsdFrame(t=np.arange(n, dtype=float), d=X_np)
        y_nap = nap.Tsd(t=np.arange(n, dtype=float), d=y_np)

        init_params = _init_params_from_fixture(fixture)

        # Pynapple TSD/TsdFrame input accepted; all fit attributes set afterwards
        model_pynap = fixture.model
        model_pynap.fit(X_nap, y_nap, init_params=init_params)
        state = model_pynap._get_fit_state()
        n_features = X_np.shape[1]
        assert state["coef_"].shape == (n_features, N_STATES)
        assert state["intercept_"].shape == (N_STATES,)
        assert state["scale_"].shape == (N_STATES,)
        assert jnp.all(state["scale_"] > 0)
        assert state["initial_prob_"].shape == (N_STATES,)
        assert jnp.allclose(state["initial_prob_"].sum(), 1.0)
        assert state["transition_prob_"].shape == (N_STATES, N_STATES)
        assert jnp.allclose(state["transition_prob_"].sum(axis=-1), jnp.ones(N_STATES))
        assert float(jnp.squeeze(state["dof_resid_"])) > 0
        assert isinstance(state["solver_state_"], EMState)

        # Providing an extra session boundary changes the M-step for initial_prob.
        model_single = deepcopy(fixture.model)
        model_multi = deepcopy(fixture.model)
        is_ns_single = np.zeros(n, dtype=bool)
        is_ns_single[0] = True
        is_ns_multi = np.zeros(n, dtype=bool)
        is_ns_multi[0] = True
        is_ns_multi[n // 2] = True
        model_single.fit(
            X_np, y_np, session_starts=is_ns_single, init_params=init_params
        )
        model_multi.fit(X_np, y_np, session_starts=is_ns_multi, init_params=init_params)
        assert not jnp.array_equal(
            model_single.initial_prob_, model_multi.initial_prob_
        )

    def test_kmeans_initialized_once(
        self, mock_glm_hmm_optimizer_run, instantiate_base_regressor_subclass
    ):
        fixture = instantiate_base_regressor_subclass
        fixture.model.setup(
            scale_init="kmeans",
            glm_params_init="kmeans",
            initial_proba_init="kmeans",
            transition_proba_init="kmeans",
        )
        with patch.object(GLMHMM, "_kmeans_init_class") as MockClass:
            inst = MockClass.return_value
            inst.glm_params.return_value = (
                fixture.params.model_params.coef,
                fixture.params.model_params.intercept,
            )
            inst.scale.return_value = jnp.exp(fixture.params.model_params.log_scale)
            inst.initial_probability.return_value = jnp.exp(
                fixture.params.hmm_params.log_initial_prob
            )
            inst.transition_probability.return_value = jnp.exp(
                fixture.params.hmm_params.log_transition_prob
            )
            fixture.model.fit(fixture.X, fixture.y)
            assert MockClass.call_count == 1

    def test_fit_calls_validate_consistency(
        self,
        instantiate_base_regressor_subclass,
        mock_glm_hmm_optimizer_run,
        monkeypatch,
    ):
        """fit() delegates X-vs-params consistency checking to validate_consistency.

        The exhaustive cases (feature count mismatch, shape mismatches) are
        tested directly against the validator in test_glm_hmm_validator.py.
        Here we only verify the delegation contract so that fit() cannot silently
        bypass the consistency check.
        """
        fixture = instantiate_base_regressor_subclass
        calls = _spy_calls(monkeypatch, GLMHMMValidator, "validate_consistency")
        fixture.model.fit(
            fixture.X,
            fixture.y,
            init_params=_init_params_from_fixture(fixture),
        )
        assert len(calls) == 1


@pytest.fixture
def glm_hmm_data():
    """Minimal X, y and valid init_params for GLMHMM with n_states=3, n_features=2."""
    rng = np.random.default_rng(0)
    n, k, s = 100, 2, 3
    X = np.ones((n, k))
    y = np.zeros(n)
    session_starts = jnp.zeros(n, dtype=bool).at[0].set(True)
    y[rng.choice(n, n // 3, replace=False)] = 1.0
    coef = jnp.zeros((k, s))
    intercept = jnp.zeros((s,))
    scale = jnp.ones((s,))
    init_prob = jnp.ones(s) / s
    trans_prob = jnp.ones((s, s)) / s
    return dict(
        X=X,
        y=y,
        session_starts=session_starts,
        init_params=(coef, intercept, scale, init_prob, trans_prob),
        n_states=s,
    )


class TestFitInterceptDeferred:
    """Parameter freezing (``fit_intercept``) is not extended to GLM-HMM; the base
    signature change (``frozen_params``) must stay compatible and inert."""

    def test_fit_intercept_not_exposed(self, glm_hmm_data):
        """GLM-HMM does not expose ``fit_intercept`` (freezing is deferred)."""
        model = GLMHMM(n_states=glm_hmm_data["n_states"], observation_model="Bernoulli")
        assert "fit_intercept" not in model.get_params()
        assert not hasattr(model, "fit_intercept")

    def test_initialize_optimizer_accepts_frozen_params(self):
        """``_initialize_optimizer_and_state`` accepts ``frozen_params`` for signature
        compatibility with the base entry points."""
        sig = inspect.signature(GLMHMM._initialize_optimizer_and_state)
        assert "frozen_params" in sig.parameters

    def test_frozen_params_is_ignored(self, glm_hmm_data):
        """Passing a non-None ``frozen_params`` yields the same optimizer state as
        passing None: the GLM-HMM ignores it."""
        model = GLMHMM(n_states=glm_hmm_data["n_states"], observation_model="Bernoulli")
        X, y = glm_hmm_data["X"], glm_hmm_data["y"]
        casted = model._validator.validate_and_cast_params(glm_hmm_data["init_params"])

        state_none = model._initialize_optimizer_and_state(
            casted, X, y, frozen_params=None
        )
        state_frozen = model._initialize_optimizer_and_state(
            casted, X, y, frozen_params="ignored-sentinel"
        )

        assert isinstance(state_none, EMState) and isinstance(state_frozen, EMState)
        assert state_none.iterations == state_frozen.iterations == 0
        assert bool(state_none.converged) is bool(state_frozen.converged) is False


def _spy_instantiate_solver(monkeypatch):
    """Wrap GLMHMM._instantiate_solver to capture each call's resolved args.

    Returns a list; each entry has resolved ``solver_name``, ``regularizer``,
    and the ``init_params`` passed to the call.
    """
    calls = []
    real = GLMHMM._instantiate_solver

    def spy(self, loss, init_params, solver_name=None, regularizer=None, **kw):
        calls.append(
            {
                "solver_name": (
                    solver_name
                    if solver_name is not None
                    else self.solver_spec.full_name
                ),
                "regularizer": (
                    regularizer if regularizer is not None else self.regularizer
                ),
                "init_params": init_params,
            }
        )
        return real(
            self,
            loss,
            init_params,
            solver_name=solver_name,
            regularizer=regularizer,
            **kw,
        )

    monkeypatch.setattr(GLMHMM, "_instantiate_solver", spy)
    return calls


def _spy_calls(monkeypatch, container, attr_name):
    """Wrap ``container.attr_name`` with a spy that forwards to the real callable.

    Returns a list of ``(args, kwargs)`` tuples per call. Use for any module-level
    function or class method we want to assert was called with given arguments
    without disturbing the real flow.
    """
    calls = []
    real = getattr(container, attr_name)

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(container, attr_name, spy)
    return calls


def _spy_call_order(monkeypatch, container, *attr_names):
    """Wrap several attributes of ``container`` with spies sharing one call log.

    Returns a list holding the name of each attribute in the order it was called,
    forwarding to the real callables. Complements ``_spy_calls``, which logs the
    arguments of a single attribute and so cannot capture the relative order of two.
    """
    order = []

    def make_spy(attr_name, real):
        def spy(*args, **kwargs):
            order.append(attr_name)
            return real(*args, **kwargs)

        return spy

    for attr_name in attr_names:
        monkeypatch.setattr(
            container, attr_name, make_spy(attr_name, getattr(container, attr_name))
        )
    return order


def _make_model_params(n_features, n_states, n_neurons=None):
    """Build a minimal GLMHMMModelParams (zeros). ``n_neurons=None`` for single-neuron."""
    if n_neurons is None:
        return GLMHMMModelParams(
            coef=jnp.zeros((n_features, n_states)),
            intercept=jnp.zeros((n_states,)),
            log_scale=jnp.zeros((n_states,)),
        )
    return GLMHMMModelParams(
        coef=jnp.zeros((n_features, n_neurons, n_states)),
        intercept=jnp.zeros((n_neurons, n_states)),
        log_scale=jnp.zeros((n_neurons, n_states)),
    )


# Custom init callables exercising every slot's *_custom path. Each returns a
# shape-correct, valid output so fit() can continue without extra mocking.


def _custom_glm_params_init(
    n_states,
    X,
    y,
    inverse_link_function,
    observation_model,
    session_starts,
    random_key,
):
    return jnp.zeros((X.shape[1], n_states)), jnp.zeros((n_states,))


def _custom_scale_init(
    n_states,
    X,
    y,
    inverse_link_function,
    observation_model,
    session_starts,
    random_key,
):
    return jnp.ones((n_states,))


def _custom_initial_proba_init(n_states, X, y, session_starts, random_key):
    return jnp.ones((n_states,)) / n_states


def _custom_transition_proba_init(n_states, X, y, session_starts, random_key):
    return jnp.full((n_states, n_states), 1.0 / n_states)


class TestSolverConfiguration:
    """Verify that _initialize_optimizer_and_state wires EM correctly."""

    @pytest.mark.parametrize(
        "regularizer, solver_name, expected_regularizer_type",
        [
            ("UnRegularized", "LBFGS", UnRegularized),
            ("Ridge", "LBFGS", Ridge),
        ],
    )
    def test_valid_solver_regularizer_combo_smoke(
        self,
        regularizer,
        solver_name,
        expected_regularizer_type,
        glm_hmm_data,
        mock_glm_hmm_optimizer_run,
        monkeypatch,
    ):
        """Valid regularizer+solver pairs wire _instantiate_solver with the right type and name."""
        model = GLMHMM(
            n_states=glm_hmm_data["n_states"],
            regularizer=regularizer,
            solver_name=solver_name,
        )
        calls = _spy_instantiate_solver(monkeypatch)
        model.fit(
            glm_hmm_data["X"],
            glm_hmm_data["y"],
            init_params=glm_hmm_data["init_params"],
        )
        assert len(calls) == 1
        primary = calls[0]
        assert isinstance(primary["regularizer"], expected_regularizer_type)
        assert solver_name in primary["solver_name"]

    def test_bernoulli_single_solver_smoke(
        self, glm_hmm_data, mock_glm_hmm_optimizer_run, monkeypatch
    ):
        """Bernoulli (fixed scale) routes through the joint path: one solver call with full params."""
        model = GLMHMM(n_states=glm_hmm_data["n_states"], observation_model="Bernoulli")
        calls = _spy_instantiate_solver(monkeypatch)
        model.fit(
            glm_hmm_data["X"],
            glm_hmm_data["y"],
            init_params=glm_hmm_data["init_params"],
        )
        # Fixed scale → else branch in prepare_mstep_update_fn → exactly one solver call.
        assert len(calls) == 1
        # Joint path passes full GLMHMMModelParams (coef, intercept, log_scale) to the solver.
        assert isinstance(calls[0]["init_params"], GLMHMMModelParams)

    def test_gaussian_separable_scale_smoke(
        self, glm_hmm_data, mock_glm_hmm_optimizer_run, monkeypatch
    ):
        """Gaussian (separable scale + analytical update) routes through the separable path."""
        rng = np.random.default_rng(1)
        n, k, s = 100, 2, 3
        X = np.ones((n, k))
        y = rng.standard_normal(n)
        init_params = glm_hmm_data["init_params"]

        model = GLMHMM(n_states=s, observation_model="Gaussian")
        calls = _spy_instantiate_solver(monkeypatch)
        model.fit(X, y, init_params=init_params)
        # Separable path with analytical scale update → one solver call (params only, no scale).
        assert len(calls) == 1
        # Separable path strips scale from init_params → GLMParams(coef, intercept).
        assert isinstance(calls[0]["init_params"], GLMParams)

    @pytest.mark.parametrize(
        "instantiate_base_regressor_subclass",
        INSTANTIATE_MODEL_AND_SIMULATE,
        indirect=True,
    )
    def test_session_starts_forwarded_to_initialization(
        self,
        instantiate_base_regressor_subclass,
        mock_glm_hmm_optimizer_run,
        monkeypatch,
    ):
        """session_starts passed to fit() reaches _model_specific_initialization."""
        fixture = instantiate_base_regressor_subclass
        n = 50
        X = fixture.X[:n]
        y = fixture.y[:n]
        is_ns = np.zeros(n, dtype=bool)
        is_ns[0] = True
        is_ns[n // 2] = True

        calls = _spy_calls(monkeypatch, GLMHMM, "_model_specific_initialization")
        fixture.model.fit(X, y, session_starts=is_ns)

        assert len(calls) == 1
        args, _ = calls[0]
        # signature: (self, X, y, session_starts)
        forwarded_session_starts = args[3]
        assert forwarded_session_starts is not None
        assert bool(forwarded_session_starts[n // 2])

    @pytest.mark.parametrize(
        "instantiate_base_regressor_subclass",
        INSTANTIATE_MODEL_AND_SIMULATE,
        indirect=True,
    )
    def test_convergence_warning_emitted_when_maxiter_reached(
        self, instantiate_base_regressor_subclass, monkeypatch
    ):
        """RuntimeWarning emitted when solver_state_.iterations == maxiter."""
        fixture = instantiate_base_regressor_subclass
        maxiter = 3
        fixture.model.maxiter = maxiter

        real_init = GLMHMM._initialize_optimizer_and_state

        def patched(self, init_params, data, y):
            result = real_init(self, init_params, data, y)
            self._optimizer_run = lambda p, *_, **__: (
                p,
                SimpleNamespace(iterations=maxiter, converged=False),
            )
            return result

        monkeypatch.setattr(GLMHMM, "_initialize_optimizer_and_state", patched)

        with pytest.warns(RuntimeWarning, match="did not converge"):
            fixture.model.fit(
                fixture.X,
                fixture.y,
                init_params=_init_params_from_fixture(fixture),
            )
        assert fixture.model.solver_state_.iterations == maxiter
        assert fixture.model.solver_state_.converged is False

    @pytest.mark.parametrize(
        "instantiate_base_regressor_subclass",
        INSTANTIATE_MODEL_AND_SIMULATE,
        indirect=True,
    )
    def test_no_convergence_warning_when_converged(
        self, instantiate_base_regressor_subclass, monkeypatch
    ):
        """No RuntimeWarning when solver_state_.iterations < maxiter."""
        fixture = instantiate_base_regressor_subclass
        maxiter = 3
        fixture.model.maxiter = maxiter

        real_init = GLMHMM._initialize_optimizer_and_state

        def patched(self, init_params, data, y):
            result = real_init(self, init_params, data, y)
            self._optimizer_run = lambda p, *_, **__: (
                p,
                SimpleNamespace(iterations=maxiter - 1, converged=True),
            )
            return result

        monkeypatch.setattr(GLMHMM, "_initialize_optimizer_and_state", patched)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fixture.model.fit(
                fixture.X,
                fixture.y,
                init_params=_init_params_from_fixture(fixture),
            )

        convergence_warns = [
            w
            for w in caught
            if issubclass(w.category, RuntimeWarning)
            and "did not converge" in str(w.message)
        ]
        assert len(convergence_warns) == 0
        assert fixture.model.solver_state_.iterations == maxiter - 1
        assert fixture.model.solver_state_.converged is True

    @pytest.mark.parametrize(
        "obs_model, y_factory",
        [
            (
                "Bernoulli",
                lambda n: np.random.default_rng(0).integers(0, 2, n).astype(float),
            ),
            ("Gaussian", lambda n: np.random.default_rng(0).standard_normal(n)),
        ],
        ids=["non-separable-scale", "separable-scale"],
    )
    def test_group_lasso_pytree_x_runs(self, obs_model, y_factory, monkeypatch):
        """GroupLasso with a dict pytree X auto-initializes a pytree mask from coef leaves.

        No explicit mask is provided. GroupLasso.initialize_mask receives the
        solver params with dict-structured coef (because X is a dict) and must
        return a mask whose .coef has the same keys and each leaf has shape
        (n_groups, *coef_leaf_shape) where n_groups == len(coef_leaves).
        """
        n, s = 100, 2
        rng = np.random.default_rng(0)
        X = {"p1": rng.standard_normal((n, 2)), "p2": rng.standard_normal((n, 3))}
        y = y_factory(n)

        captured = {}
        real_initialize_mask = nmo.regularizer.GroupLasso.initialize_mask

        def capturing_initialize_mask(self, params):
            result = real_initialize_mask(self, params)
            captured["mask"] = result
            return result

        monkeypatch.setattr(
            nmo.regularizer.GroupLasso, "initialize_mask", capturing_initialize_mask
        )

        model = GLMHMM(
            n_states=s,
            observation_model=obs_model,
            regularizer="GroupLasso",
            solver_name="ProximalGradient",
            regularizer_strength=0.1,
            maxiter=3,
        )
        model.fit(X, y)

        assert model.coef_ is not None
        assert "mask" in captured
        coef_leaves = jax.tree_util.tree_leaves(model.coef_)
        mask_coef_leaves = jax.tree_util.tree_leaves(captured["mask"].coef)
        # _initialize_subtree_mask groups by trailing dims of each coef leaf:
        # n_groups = sum(math.prod(leaf.shape[1:]) for each leaf).
        # For GLMHMM coef shape (n_states, n_features), shape[1:] = (n_features,),
        # so each feature gets its own group across all states.
        # If the grouping assumption changes (e.g. one group per leaf), update here.
        n_groups = sum(math.prod(c_leaf.shape[1:]) for c_leaf in coef_leaves)
        assert len(mask_coef_leaves) == len(coef_leaves)
        for m_leaf, c_leaf in zip(mask_coef_leaves, coef_leaves):
            assert m_leaf.shape == (n_groups, *c_leaf.shape)

    @pytest.mark.parametrize(
        "obs_model, y_factory",
        [
            (
                "Bernoulli",
                lambda n: np.random.default_rng(0).integers(0, 2, n).astype(float),
            ),
            ("Gaussian", lambda n: np.random.default_rng(0).standard_normal(n)),
        ],
        ids=["non-separable-scale", "separable-scale"],
    )
    def test_group_lasso_pytree_x_update_runs(self, obs_model, y_factory, monkeypatch):
        """GroupLasso with a dict pytree X auto-initializes a pytree mask via update()."""
        n, s = 100, 2
        rng = np.random.default_rng(0)
        X = {"p1": rng.standard_normal((n, 2)), "p2": rng.standard_normal((n, 3))}
        y = y_factory(n)
        session_starts = jnp.zeros(y.shape[0], dtype=bool).at[0].set(True)

        captured = {}
        real_initialize_mask = nmo.regularizer.GroupLasso.initialize_mask

        def capturing_initialize_mask(self, params):
            result = real_initialize_mask(self, params)
            captured["mask"] = result
            return result

        monkeypatch.setattr(
            nmo.regularizer.GroupLasso, "initialize_mask", capturing_initialize_mask
        )

        model = GLMHMM(
            n_states=s,
            observation_model=obs_model,
            regularizer="GroupLasso",
            solver_name="ProximalGradient",
            regularizer_strength=0.1,
        )
        init_params = model.initialize_params(X, y)
        opt_state = model.initialize_optimizer_and_state(init_params, X, y)
        params = init_params
        for _ in range(3):
            params, opt_state = model.update(params, opt_state, X, y, session_starts)

        assert model.coef_ is not None
        assert "mask" in captured
        coef_leaves = jax.tree_util.tree_leaves(model.coef_)
        mask_coef_leaves = jax.tree_util.tree_leaves(captured["mask"].coef)
        n_groups = sum(math.prod(c_leaf.shape[1:]) for c_leaf in coef_leaves)
        assert len(mask_coef_leaves) == len(coef_leaves)
        for m_leaf, c_leaf in zip(mask_coef_leaves, coef_leaves):
            assert m_leaf.shape == (n_groups, *c_leaf.shape)


# ---------------------------------------------------------------------------
# Shared stub: replace GLMHMM._simulate with a no-op returning dummy arrays.
# Used by delegation tests that need simulate() to complete without running EM.
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_glmhmm_simulate(monkeypatch):
    """Return a factory that patches GLMHMM._simulate to return dummy arrays of size n."""

    def _stub(n):
        dummy = jnp.zeros(n)
        monkeypatch.setattr(GLMHMM, "_simulate", lambda *_, **__: (dummy, dummy, dummy))

    return _stub


# ---------------------------------------------------------------------------
# TestFitDelegation — fit() calls validate_and_cast_session_starts
# ---------------------------------------------------------------------------


class TestFitDelegation:
    """fit() and simulate() must delegate session-boundary handling to validate_and_cast_session_starts."""

    def test_fit_calls_validate_and_cast_session_starts(
        self, glm_hmm_data, mock_glm_hmm_optimizer_run, monkeypatch
    ):
        n = glm_hmm_data["X"].shape[0]
        default_session_starts = jnp.zeros(n, dtype=bool).at[0].set(True)
        mock = MagicMock(return_value=default_session_starts)
        monkeypatch.setattr(GLMHMMValidator, "validate_and_cast_session_starts", mock)

        model = GLMHMM(n_states=glm_hmm_data["n_states"])
        model.fit(
            glm_hmm_data["X"],
            glm_hmm_data["y"],
            init_params=glm_hmm_data["init_params"],
        )

        mock.assert_called_once()

    def test_simulate_calls_validate_and_cast_session_starts(
        self,
        glm_hmm_data,
        mock_glm_hmm_optimizer_run,
        stub_glmhmm_simulate,
        monkeypatch,
    ):
        n = glm_hmm_data["X"].shape[0]
        X = glm_hmm_data["X"]

        model = GLMHMM(n_states=glm_hmm_data["n_states"])
        model.fit(X, glm_hmm_data["y"], init_params=glm_hmm_data["init_params"])

        default_session_starts = jnp.zeros(n, dtype=bool).at[0].set(True)
        mock = MagicMock(return_value=default_session_starts)
        monkeypatch.setattr(GLMHMMValidator, "validate_and_cast_session_starts", mock)
        stub_glmhmm_simulate(n)

        model.simulate(jax.random.key(0), X)

        mock.assert_called_once()

    def test_fit_calls_check_is_continuous(
        self, glm_hmm_data, mock_glm_hmm_optimizer_run, monkeypatch
    ):
        """fit() forwards X, y and the cast boundaries to check_is_continuous."""
        calls = _spy_calls(monkeypatch, GLMHMMValidator, "check_is_continuous")

        model = GLMHMM(n_states=glm_hmm_data["n_states"])
        model.fit(
            glm_hmm_data["X"],
            glm_hmm_data["y"],
            init_params=glm_hmm_data["init_params"],
        )

        assert len(calls) == 1
        _, kwargs = calls[0]
        assert kwargs["X"] is glm_hmm_data["X"]
        assert kwargs["y"] is glm_hmm_data["y"]
        assert kwargs["session_starts"].shape == (glm_hmm_data["X"].shape[0],)

    def test_simulate_calls_check_is_continuous(
        self,
        glm_hmm_data,
        mock_glm_hmm_optimizer_run,
        stub_glmhmm_simulate,
        monkeypatch,
    ):
        """simulate() checks the feedforward input, with no observations to pair it to."""
        n = glm_hmm_data["X"].shape[0]
        X = glm_hmm_data["X"]

        model = GLMHMM(n_states=glm_hmm_data["n_states"])
        model.fit(X, glm_hmm_data["y"], init_params=glm_hmm_data["init_params"])

        calls = _spy_calls(monkeypatch, GLMHMMValidator, "check_is_continuous")
        stub_glmhmm_simulate(n)

        model.simulate(jax.random.key(0), X)

        assert len(calls) == 1
        _, kwargs = calls[0]
        assert kwargs["X"] is X
        assert kwargs["y"] is None

    def test_fit_calls_validate_inputs(
        self, glm_hmm_data, mock_glm_hmm_optimizer_run, monkeypatch
    ):
        """fit() forwards X and y to GLMHMMValidator.validate_inputs."""
        calls = _spy_calls(monkeypatch, GLMHMMValidator, "validate_inputs")
        model = GLMHMM(n_states=glm_hmm_data["n_states"])
        model.fit(
            glm_hmm_data["X"],
            glm_hmm_data["y"],
            init_params=glm_hmm_data["init_params"],
        )
        assert len(calls) == 1
        _, kwargs = calls[0]
        assert kwargs["X"] is glm_hmm_data["X"]
        assert kwargs["y"] is glm_hmm_data["y"]

    def test_fit_with_init_params_calls_validate_and_cast_params(
        self, glm_hmm_data, mock_glm_hmm_optimizer_run, monkeypatch
    ):
        """fit(init_params=...) validates the user-provided params via the validator."""
        calls = _spy_calls(monkeypatch, GLMHMMValidator, "validate_and_cast_params")
        model = GLMHMM(n_states=glm_hmm_data["n_states"])
        model.fit(
            glm_hmm_data["X"],
            glm_hmm_data["y"],
            init_params=glm_hmm_data["init_params"],
        )
        assert len(calls) == 1
        # args = (validator_self, init_params) when patched on the class
        args, _ = calls[0]
        assert args[1] is glm_hmm_data["init_params"]

    def test_fit_default_init_skips_validate_and_cast_params(
        self, glm_hmm_data, mock_glm_hmm_optimizer_run, monkeypatch
    ):
        """fit(init_params=None) with default initializers bypasses validate_and_cast_params."""
        calls = _spy_calls(monkeypatch, GLMHMMValidator, "validate_and_cast_params")
        model = GLMHMM(n_states=glm_hmm_data["n_states"])
        model.fit(glm_hmm_data["X"], glm_hmm_data["y"])
        assert calls == []

    def test_fit_default_init_calls_generate_glm_hmm_initial_model_params(
        self, glm_hmm_data, mock_glm_hmm_optimizer_run, monkeypatch
    ):
        """fit(init_params=None) routes init through generate_glm_hmm_initial_model_params
        with all positional and keyword arguments propagated from the model."""
        from nemos.glm_hmm import glm_hmm as glm_hmm_module

        calls = _spy_calls(
            monkeypatch, glm_hmm_module, "generate_glm_hmm_initial_model_params"
        )
        model = GLMHMM(n_states=glm_hmm_data["n_states"])
        model.fit(glm_hmm_data["X"], glm_hmm_data["y"])

        assert len(calls) == 1
        args, kwargs = calls[0]
        # positional args: (n_states, X, y)
        assert args[0] == glm_hmm_data["n_states"]
        np.testing.assert_array_equal(args[1], glm_hmm_data["X"])
        np.testing.assert_array_equal(args[2], glm_hmm_data["y"])
        # keyword args: every entry the call site supplies
        assert set(kwargs) == {
            "inverse_link_function",
            "observation_model",
            "session_starts",
            "random_key",
            "init_funcs",
        }
        assert kwargs["inverse_link_function"] is model._inverse_link_function
        assert kwargs["init_funcs"] is model._model_initialization_funcs
        # session_starts: default is a length-n boolean array with first entry True
        n = glm_hmm_data["X"].shape[0]
        expected_session_starts = jnp.zeros(n, dtype=bool).at[0].set(True)
        np.testing.assert_array_equal(kwargs["session_starts"], expected_session_starts)
        # random_key: a jax PRNG key
        assert isinstance(kwargs["random_key"], jax.Array)
        assert kwargs["random_key"].dtype == jax.random.PRNGKey(0).dtype

    @pytest.mark.parametrize(
        "setup_kwargs",
        [
            pytest.param(
                {"glm_params_init": _custom_glm_params_init}, id="glm_params_init"
            ),
            pytest.param({"scale_init": _custom_scale_init}, id="scale_init"),
            pytest.param(
                {"initial_proba_init": _custom_initial_proba_init},
                id="initial_proba_init",
            ),
            pytest.param(
                {"transition_proba_init": _custom_transition_proba_init},
                id="transition_proba_init",
            ),
        ],
    )
    def test_fit_custom_init_func_triggers_validate_and_cast_params(
        self, glm_hmm_data, mock_glm_hmm_optimizer_run, monkeypatch, setup_kwargs
    ):
        """Any custom init function on any of the four slots makes ``validate_params``
        True, so fit(init_params=None) routes through validate_and_cast_params."""
        model = GLMHMM(n_states=glm_hmm_data["n_states"])
        model.setup(**setup_kwargs)
        calls = _spy_calls(monkeypatch, GLMHMMValidator, "validate_and_cast_params")
        model.fit(glm_hmm_data["X"], glm_hmm_data["y"])
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# TestValidationSequence — order of the input validation steps
# ---------------------------------------------------------------------------


VALIDATION_SEQUENCE = [
    method_name for method_name, _ in GLMHMMValidator.inputs_validation_sequence
]


class TestValidationSequence:
    """Every entry point runs the declared inputs_validation_sequence, in order. The
    order is load bearing: check_is_continuous reads the boundaries that
    validate_and_cast_session_starts produces, NaN shifts included, so running it
    earlier would check a session partition other than the one inference uses."""

    def test_declared_sequence(self):
        """The sequence the entry points are expected to run, spelled out."""
        assert VALIDATION_SEQUENCE == [
            "validate_inputs",
            "validate_and_cast_session_starts",
            "check_is_continuous",
        ]

    @pytest.fixture
    def fitted_model(self, glm_hmm_data, mock_glm_hmm_optimizer_run):
        model = GLMHMM(n_states=glm_hmm_data["n_states"])
        model.fit(
            glm_hmm_data["X"],
            glm_hmm_data["y"],
            init_params=glm_hmm_data["init_params"],
        )
        return model

    def test_fit(self, glm_hmm_data, mock_glm_hmm_optimizer_run, monkeypatch):
        order = _spy_call_order(monkeypatch, GLMHMMValidator, *VALIDATION_SEQUENCE)

        GLMHMM(n_states=glm_hmm_data["n_states"]).fit(
            glm_hmm_data["X"],
            glm_hmm_data["y"],
            init_params=glm_hmm_data["init_params"],
        )

        assert order == VALIDATION_SEQUENCE

    def test_update(self, glm_hmm_data, patch_optimizer_update, monkeypatch):
        patch_optimizer_update(GLMHMM)
        X, y, session_starts, init_params = (
            glm_hmm_data["X"],
            glm_hmm_data["y"],
            glm_hmm_data["session_starts"],
            glm_hmm_data["init_params"],
        )
        model = GLMHMM(n_states=glm_hmm_data["n_states"])

        # spy before initialize_optimizer_and_state, where validate_inputs is called
        order = _spy_call_order(monkeypatch, GLMHMMValidator, *VALIDATION_SEQUENCE)
        opt_state = model.initialize_optimizer_and_state(init_params, X, y)
        assert order == ["validate_inputs"]

        # spy again after the setup calls, where no validation is expected
        order = _spy_call_order(monkeypatch, GLMHMMValidator, *VALIDATION_SEQUENCE)
        model.update(init_params, opt_state, X, y, session_starts)
        assert order == VALIDATION_SEQUENCE

    @pytest.mark.parametrize(
        "method_name", ["score", "smooth_proba", "filter_proba", "decode_state"]
    )
    def test_inference_methods(
        self, method_name, fitted_model, glm_hmm_data, monkeypatch
    ):
        order = _spy_call_order(monkeypatch, GLMHMMValidator, *VALIDATION_SEQUENCE)

        getattr(fitted_model, method_name)(glm_hmm_data["X"], glm_hmm_data["y"])

        assert order == VALIDATION_SEQUENCE

    def test_simulate(
        self, fitted_model, glm_hmm_data, stub_glmhmm_simulate, monkeypatch
    ):
        stub_glmhmm_simulate(glm_hmm_data["X"].shape[0])
        order = _spy_call_order(monkeypatch, GLMHMMValidator, *VALIDATION_SEQUENCE)

        fitted_model.simulate(jax.random.key(0), glm_hmm_data["X"])

        assert order == VALIDATION_SEQUENCE


# ---------------------------------------------------------------------------
# TestEstimateResidDegreesOfFreedom — exceptions and Lasso branch
# ---------------------------------------------------------------------------


class TestEstimateResidDegreesOfFreedom:
    """Cover the exception path and Lasso dof branch of _estimate_resid_degrees_of_freedom."""

    def test_non_int_n_samples_raises(self, glm_hmm_data, mock_glm_hmm_optimizer_run):
        model = GLMHMM(n_states=glm_hmm_data["n_states"])
        model.fit(
            glm_hmm_data["X"],
            glm_hmm_data["y"],
            init_params=glm_hmm_data["init_params"],
        )
        with pytest.raises(
            TypeError, match="`n_samples` must either `None` or of type `int`"
        ):
            model._estimate_resid_degrees_of_freedom(glm_hmm_data["X"], n_samples="bad")

    def test_lasso_dof_uses_nonzero_coef(
        self, glm_hmm_data, mock_glm_hmm_optimizer_run
    ):
        """Lasso branch estimates dof from non-zero coef entries."""
        model = GLMHMM(
            n_states=glm_hmm_data["n_states"],
            regularizer="Lasso",
            solver_name="ProximalGradient",
            regularizer_strength=1.0,
        )
        model.fit(
            glm_hmm_data["X"],
            glm_hmm_data["y"],
            init_params=glm_hmm_data["init_params"],
        )
        # Noop optimizer leaves coef at zeros → resid_dof=0.
        # dof_intercept_and_hmm = n_states*1 + (n_states-1) + (n_states-1)*n_states = 3+2+6 = 11
        n_states = glm_hmm_data["n_states"]
        n_samples = glm_hmm_data["X"].shape[0]
        dof_intercept_and_hmm = n_states + (n_states - 1) + (n_states - 1) * n_states
        expected = n_samples - 0 - dof_intercept_and_hmm
        assert model.dof_resid_ == expected


# ---------------------------------------------------------------------------
# TestSimulate — GLMHMM.simulate() method tests
# ---------------------------------------------------------------------------


class TestSimulate:
    """Tests for GLMHMM.simulate()."""

    def test_simulate_calls_validate_and_prepare_inputs_with_y_none(
        self,
        glm_hmm_data,
        mock_glm_hmm_optimizer_run,
        stub_glmhmm_simulate,
        monkeypatch,
    ):
        """simulate() delegates to _validate_and_prepare_inputs with y=None."""
        X = glm_hmm_data["X"]
        n = X.shape[0]
        model = GLMHMM(n_states=glm_hmm_data["n_states"])
        model.fit(X, glm_hmm_data["y"], init_params=glm_hmm_data["init_params"])

        mock_vapi = MagicMock(
            return_value=(
                model._get_model_params(),
                jnp.asarray(X),
                None,
                glm_hmm_data["session_starts"],
            )
        )
        monkeypatch.setattr(GLMHMM, "_validate_and_prepare_inputs", mock_vapi)
        stub_glmhmm_simulate(n)

        model.simulate(jax.random.key(0), X)

        mock_vapi.assert_called_once()
        assert (
            mock_vapi.call_args.args[1] is None
        )  # y must be None, not a fake zeros array

    @pytest.mark.parametrize(
        "state_format, expectation",
        [
            ("index", does_not_raise()),
            ("one-hot", does_not_raise()),
            ("invalid", pytest.raises(ValueError, match="Invalid state_format")),
        ],
    )
    def test_simulate_state_format(
        self, state_format, expectation, glm_hmm_data, mock_glm_hmm_optimizer_run
    ):
        """Valid state formats run cleanly; invalid raises ValueError before any computation."""
        model = GLMHMM(n_states=glm_hmm_data["n_states"])
        model.fit(
            glm_hmm_data["X"],
            glm_hmm_data["y"],
            init_params=glm_hmm_data["init_params"],
        )
        with expectation:
            model.simulate(
                jax.random.key(0), glm_hmm_data["X"], state_format=state_format
            )

    def test_simulate_output_shapes(self, glm_hmm_data, mock_glm_hmm_optimizer_run):
        """simulate() returns arrays with correct shapes for both state formats."""
        model = GLMHMM(n_states=glm_hmm_data["n_states"])
        model.fit(
            glm_hmm_data["X"],
            glm_hmm_data["y"],
            init_params=glm_hmm_data["init_params"],
        )
        X = glm_hmm_data["X"]
        n, n_states = X.shape[0], glm_hmm_data["n_states"]

        activity, rates, states = model.simulate(
            jax.random.key(0), X, state_format="index"
        )
        assert activity.shape[0] == n
        assert rates.shape[0] == n
        assert states.shape == (n,)

        _, _, states_oh = model.simulate(jax.random.key(0), X, state_format="one-hot")
        assert states_oh.shape == (n, n_states)
        assert jnp.all(states_oh.sum(axis=1) == 1)

    def test_simulate_is_deterministic(self, glm_hmm_data, mock_glm_hmm_optimizer_run):
        """Same random key produces identical outputs."""
        model = GLMHMM(n_states=glm_hmm_data["n_states"])
        model.fit(
            glm_hmm_data["X"],
            glm_hmm_data["y"],
            init_params=glm_hmm_data["init_params"],
        )
        X = glm_hmm_data["X"]
        key = jax.random.key(42)
        act1, rate1, state1 = model.simulate(key, X)
        act2, rate2, state2 = model.simulate(key, X)
        assert jnp.array_equal(state1, state2)
        assert jnp.array_equal(rate1, rate2)
        assert jnp.array_equal(act1, act2)

    def test_simulate_forces_first_bin_new_session(
        self, glm_hmm_data, mock_glm_hmm_optimizer_run, monkeypatch
    ):
        """session_starts[0] is always True regardless of what the user passes."""
        model = GLMHMM(n_states=glm_hmm_data["n_states"])
        model.fit(
            glm_hmm_data["X"],
            glm_hmm_data["y"],
            init_params=glm_hmm_data["init_params"],
        )
        X = glm_hmm_data["X"]
        n = X.shape[0]
        captured = {}

        def capturing_simulate(self_inner, key, params, data, session_starts):
            captured["session_starts"] = session_starts
            return jnp.zeros(n), jnp.zeros(n), jnp.zeros(n, dtype=int)

        monkeypatch.setattr(GLMHMM, "_simulate", capturing_simulate)
        model.simulate(jax.random.key(0), X, session_starts=jnp.zeros(n, dtype=bool))

        assert bool(captured["session_starts"][0]) is True

    def test_simulate_sample_generator_receives_key_rate_scale(
        self, glm_hmm_data, mock_glm_hmm_optimizer_run, monkeypatch
    ):
        """_simulate calls obs model sample_generator with scalar key, rate, and scale per step.

        all_rates has shape (n_time_bins, n_states) before the scan; inside simulate_step
        the selected-state slice reduces each to a scalar.  Disable JIT so the scan
        body runs as a Python loop and the spy sees concrete values.
        """
        model = GLMHMM(n_states=glm_hmm_data["n_states"])
        model.fit(
            glm_hmm_data["X"],
            glm_hmm_data["y"],
            init_params=glm_hmm_data["init_params"],
        )
        X_short = jnp.ones((2, glm_hmm_data["X"].shape[1]))

        obs_model = model._observation_model
        original = obs_model.sample_generator
        calls = []

        def spy(key, predicted_rate, scale):
            calls.append(dict(key=key, predicted_rate=predicted_rate, scale=scale))
            return original(key=key, predicted_rate=predicted_rate, scale=scale)

        monkeypatch.setattr(obs_model, "sample_generator", spy)

        with jax.disable_jit():
            model.simulate(jax.random.key(0), X_short)

        assert len(calls) == 2
        for call in calls:
            # after state selection inside simulate_step, each is a scalar
            assert call["key"].ndim == 0
            assert call["predicted_rate"].ndim == 0
            assert call["scale"].ndim == 0


# ---------------------------------------------------------------------------
# TestLogLikelihood — _log_likelihood input shapes and cache behavior
# ---------------------------------------------------------------------------


class TestLogLikelihood:
    """GLMHMM._log_likelihood input shapes and cache behavior."""

    @pytest.mark.parametrize(
        "n_samples, n_features, n_neurons",
        [
            pytest.param(1, 1, None, id="1-sample-1-feature"),
            pytest.param(1, 4, None, id="1-sample-multi-feature"),
            pytest.param(50, 1, None, id="multi-sample-1-feature"),
            pytest.param(50, 4, None, id="multi-sample-multi-feature"),
            pytest.param(30, 3, 2, id="population-2-neurons"),
        ],
    )
    def test_output_shape_finite(self, n_samples, n_features, n_neurons):
        """Output is per-(sample, state), finite, across single-neuron and population y."""
        rng = np.random.default_rng(0)
        X = jnp.asarray(rng.standard_normal((n_samples, n_features)))
        if n_neurons is None:
            y = jnp.asarray(rng.integers(0, 2, size=n_samples))
        else:
            y = jnp.asarray(rng.integers(0, 2, size=(n_samples, n_neurons)))
        model = GLMHMM(n_states=2)
        ll = model._log_likelihood(_make_model_params(n_features, 2, n_neurons), X, y)
        assert ll.shape == (n_samples, 2)
        assert jnp.all(jnp.isfinite(ll))

    def test_cache_reuses_func_for_repeated_call(self, monkeypatch):
        """Second call with matching (y.ndim, obs_model, inv_link) reuses cached ll_func."""
        from nemos.glm_hmm import glm_hmm as glm_hmm_module

        calls = _spy_calls(monkeypatch, glm_hmm_module, "prepare_estep_log_likelihood")
        model = GLMHMM(n_states=2)
        params = _make_model_params(2, 2)
        X = jnp.zeros((10, 2))
        y = jnp.zeros(10)

        model._log_likelihood(params, X, y)
        model._log_likelihood(params, X, y)

        assert len(calls) == 1, (
            "prepare_estep_log_likelihood should be called once and reused from cache "
            "on the second call with matching key."
        )

    def test_cache_invalidates_when_y_ndim_changes(self, monkeypatch):
        """Switching between 1-d and 2-d y yields a different cache key."""
        from nemos.glm_hmm import glm_hmm as glm_hmm_module

        calls = _spy_calls(monkeypatch, glm_hmm_module, "prepare_estep_log_likelihood")
        model = GLMHMM(n_states=2)
        model._log_likelihood(
            _make_model_params(2, 2), jnp.zeros((10, 2)), jnp.zeros(10)
        )
        model._log_likelihood(
            _make_model_params(2, 2, n_neurons=3),
            jnp.zeros((10, 2)),
            jnp.zeros((10, 3)),
        )

        # is_population_glm (first positional arg) toggles False -> True
        assert [args[0][0] for args in calls] == [False, True]

    @pytest.mark.parametrize(
        "attr, new_value",
        [
            pytest.param("observation_model", "Gaussian", id="observation_model"),
            pytest.param(
                "inverse_link_function",
                lambda x: jnp.exp(x),
                id="inverse_link_function",
            ),
        ],
    )
    def test_cache_invalidates_when_model_attr_changes(
        self, monkeypatch, attr, new_value
    ):
        """Mutating observation_model or inverse_link_function between calls invalidates the cache."""
        from nemos.glm_hmm import glm_hmm as glm_hmm_module

        calls = _spy_calls(monkeypatch, glm_hmm_module, "prepare_estep_log_likelihood")
        model = GLMHMM(n_states=2)
        params = _make_model_params(2, 2)
        X = jnp.zeros((10, 2))
        y = jnp.zeros(10)

        model._log_likelihood(params, X, y)
        setattr(model, attr, new_value)
        model._log_likelihood(params, X, y)

        assert len(calls) == 2


# ---------------------------------------------------------------------------
# TestEMConfiguration — verify _initialize_optimizer_and_state wires EM correctly
# ---------------------------------------------------------------------------


class TestEMConfiguration:
    """Verify that _initialize_optimizer_and_state passes the right objects to each EM component.

    The solver-configuration tests (TestSolverConfiguration) already check that
    _instantiate_solver receives the right regularizer/solver name and init_params type.
    These tests cover the complementary side: observation_model, inverse_link_function,
    is_population_glm, maxiter, and tol all reach the correct callables.
    """

    def test_mstep_receives_params(
        self, glm_hmm_data, mock_glm_hmm_optimizer_run, monkeypatch
    ):
        """prepare_mstep_update_fn is called with the model's _observation_model."""
        from nemos.glm_hmm import glm_hmm as glm_hmm_module

        calls = _spy_calls(monkeypatch, glm_hmm_module, "prepare_mstep_update_fn")
        model = GLMHMM(
            n_states=glm_hmm_data["n_states"],
            observation_model="Poisson",
            inverse_link_function=jax.nn.softplus,
        )
        model.fit(
            glm_hmm_data["X"],
            glm_hmm_data["y"],
            init_params=glm_hmm_data["init_params"],
        )
        assert len(calls) == 1
        _, kwargs = calls[0]
        assert kwargs["observation_model"] is model._observation_model
        p = model._validator.to_model_params(glm_hmm_data["init_params"])
        assert eqx.tree_equal(p.model_params, kwargs["init_params"])
        assert kwargs["inverse_link_function"] is model._inverse_link_function

    def test_estep_ll_receives_observation_config(self, monkeypatch):
        """prepare_estep_log_likelihood receives the model's observation_model and inverse_link_function.

        Uses non-default Poisson + softplus so identity checks fail if the code
        always forwards the Bernoulli/sigmoid defaults.
        """
        from nemos.glm_hmm import glm_hmm as glm_hmm_module

        calls = _spy_calls(monkeypatch, glm_hmm_module, "prepare_estep_log_likelihood")
        model = GLMHMM(
            n_states=2,
            observation_model="Poisson",
            inverse_link_function=jax.nn.softplus,
        )
        model._log_likelihood(
            _make_model_params(2, 2), jnp.zeros((10, 2)), jnp.zeros(10)
        )
        assert len(calls) == 1
        args, _ = calls[0]
        assert args[1] is model._observation_model
        assert args[2] is model._inverse_link_function

    def test_is_population_false_for_1d_y(
        self, glm_hmm_data, mock_glm_hmm_optimizer_run, monkeypatch
    ):
        """is_population_glm=False for 1-D y (population not yet supported by GLMHMMValidator)."""
        from nemos.glm_hmm import glm_hmm as glm_hmm_module

        calls = _spy_calls(monkeypatch, glm_hmm_module, "prepare_mstep_update_fn")
        model = GLMHMM(n_states=glm_hmm_data["n_states"])
        model.fit(
            glm_hmm_data["X"],
            glm_hmm_data["y"],
            init_params=glm_hmm_data["init_params"],
        )
        assert len(calls) == 1
        _, kwargs = calls[0]
        assert kwargs["is_population_glm"] is False

    def test_maxiter_and_tol_threaded_into_em(self, glm_hmm_data, monkeypatch):
        """_optimizer_run partial binds maxiter and tol from the model."""
        from types import SimpleNamespace

        custom_maxiter = 17
        custom_tol = 3e-6
        captured = {}

        real_init = GLMHMM._initialize_optimizer_and_state

        def capturing_init(self, init_params, data, y):
            result = real_init(self, init_params, data, y)
            captured["optimizer_run"] = self._optimizer_run
            self._optimizer_run = lambda p, *a, **kw: (
                p,
                SimpleNamespace(iterations=1, converged=True),
            )
            return result

        monkeypatch.setattr(GLMHMM, "_initialize_optimizer_and_state", capturing_init)

        model = GLMHMM(
            n_states=glm_hmm_data["n_states"],
            maxiter=custom_maxiter,
            tol=custom_tol,
        )
        model.fit(
            glm_hmm_data["X"],
            glm_hmm_data["y"],
            init_params=glm_hmm_data["init_params"],
        )

        opt_run = captured["optimizer_run"]
        assert opt_run.keywords["maxiter"] == custom_maxiter
        assert opt_run.keywords["tol"] == custom_tol

    def test_optimizer_run_and_update_share_mstep_fn(self, glm_hmm_data, monkeypatch):
        """_optimizer_run and _optimizer_update receive the same m_step_fn_model_params closure."""
        from types import SimpleNamespace

        captured = {}

        real_init = GLMHMM._initialize_optimizer_and_state

        def capturing_init(self, init_params, data, y):
            result = real_init(self, init_params, data, y)
            captured["optimizer_run"] = self._optimizer_run
            captured["optimizer_update"] = self._optimizer_update
            self._optimizer_run = lambda p, *a, **kw: (
                p,
                SimpleNamespace(iterations=1, converged=True),
            )
            return result

        monkeypatch.setattr(GLMHMM, "_initialize_optimizer_and_state", capturing_init)

        model = GLMHMM(n_states=glm_hmm_data["n_states"])
        model.fit(
            glm_hmm_data["X"],
            glm_hmm_data["y"],
            init_params=glm_hmm_data["init_params"],
        )

        assert (
            captured["optimizer_run"].keywords["m_step_fn_model_params"]
            is captured["optimizer_update"].keywords["m_step_fn_model_params"]
        )


# ---------------------------------------------------------------------------
# save_params / load_model round-trip
# ---------------------------------------------------------------------------


def _assert_params_equal(a, b, path=""):
    """Recursively assert two param trees are equal, dispatching by type."""
    if a is None:
        assert b is None, f"{path}: expected None, got {b!r}"
    elif isinstance(a, dict):
        for (ka, va), (kb, vb) in zip(a.items(), b.items(), strict=True):
            assert ka == kb, f"{path}: key mismatch {ka!r} != {kb!r}"
            _assert_params_equal(va, vb, f"{path}.{ka}" if path else ka)
    elif isinstance(a, (np.ndarray, jnp.ndarray)):
        np.testing.assert_allclose(
            np.array(a), np.array(b), err_msg=f"array mismatch at {path}"
        )
    elif isinstance(a, (bool, int, float, str)):
        assert a == b, f"{path}: {a!r} != {b!r}"
    else:
        # callables and class instances (e.g. observation model objects): compare by name
        assert _get_name(a) == _get_name(
            b
        ), f"{path}: name mismatch {_get_name(a)} != {_get_name(b)}"


def _collect_params(model, *, skip_solver_state=True):
    """Merge get_params() and _get_fit_state() into a single flat dict."""
    params = model.get_params()
    fit_state = model._get_fit_state()
    if skip_solver_state:
        fit_state.pop("solver_state_", None)
    params.update(fit_state)
    return params


def _custom_glm_init(
    n_states,
    X,
    y,
    inverse_link_function,
    observation_model,
    session_starts,
    random_key,
):
    """Custom GLM-params initializer used to exercise the custom-callable path."""
    return jnp.zeros((X.shape[1], n_states)), jnp.zeros((n_states,))


@pytest.fixture(scope="module")
def fitted_glmhmm():
    f = instantiate_glm_hmm_func(simulate=False)
    f.model.fit(f.X, f.y, init_params=_init_params_from_fixture(f))
    return f.model


@pytest.fixture(scope="module")
def fitted_glmhmm_custom():
    f = instantiate_glm_hmm_func(simulate=False)
    f.model.setup(glm_params_init=_custom_glm_init)
    f.model.fit(f.X, f.y, init_params=_init_params_from_fixture(f))
    return f.model


class TestGLMHMMSaveLoad:
    """save_params / load_model round-trip with native and custom initializers."""

    def test_native_round_trip(self, fitted_glmhmm, tmp_path):
        """All params and fit-state survive a save/load cycle with default initializers."""
        save_path = tmp_path / "glmhmm.npz"
        fitted_glmhmm.save_params(save_path)
        loaded = nmo.load_model(save_path)

        _assert_params_equal(_collect_params(fitted_glmhmm), _collect_params(loaded))

    def test_custom_callable_requires_mapping(self, fitted_glmhmm_custom, tmp_path):
        """Custom callable can't be resolved from name alone; load raises a clear error."""
        save_path = tmp_path / "glmhmm.npz"
        fitted_glmhmm_custom.save_params(save_path)

        with pytest.raises(
            ValueError, match="Failed to instantiate model class"
        ) as excinfo:
            nmo.load_model(save_path)
        assert "glm_params_init" in str(excinfo.value.__cause__)

    @pytest.mark.parametrize(
        "mapping_dict",
        [
            {"model_initialization_funcs": {"glm_params_init": _custom_glm_init}},
            {"model_initialization_funcs__glm_params_init": _custom_glm_init},
        ],
        ids=["nested-dict", "sklearn-key"],
    )
    def test_custom_callable_mapping(
        self, fitted_glmhmm_custom, tmp_path, mapping_dict
    ):
        """Both mapping_dict formats resolve the custom callable and preserve all params."""
        save_path = tmp_path / "glmhmm.npz"
        fitted_glmhmm_custom.save_params(save_path)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            loaded = nmo.load_model(save_path, mapping_dict=mapping_dict)

        assert loaded.model_initialization_funcs["glm_params_init"] is _custom_glm_init
        _assert_params_equal(
            _collect_params(fitted_glmhmm_custom), _collect_params(loaded)
        )


# ---------------------------------------------------------------------------
# TestUpdateFitEquivalence — N update steps == fit with maxiter=N
# ---------------------------------------------------------------------------


class TestUpdateFitEquivalence:
    """Verify that calling update() N times produces the same result as fit(maxiter=N)."""

    @pytest.mark.requires_x64
    def test_n_update_steps_matches_fit(self):
        """fit(maxiter=N) and N manual update() calls from the same init produce identical params."""
        rng = np.random.default_rng(0)
        n, k, s = 80, 3, 2
        X = rng.standard_normal((n, k))
        y = rng.binomial(1, 0.4, size=n)
        session_starts = jnp.zeros(n, dtype=bool).at[0].set(True)

        # shared init params so both paths start from exactly the same point
        seed = jax.random.PRNGKey(7)
        init_coef = jnp.zeros((k, s))
        init_intercept = jnp.zeros((s,))
        init_scale = jnp.ones((s,))
        init_initial_prob = jnp.ones(s) / s
        init_transition_prob = jnp.ones((s, s)) / s
        init_params = (
            init_coef,
            init_intercept,
            init_scale,
            init_initial_prob,
            init_transition_prob,
        )

        n_steps = 3

        # --- path A: fit with maxiter=n_steps ---
        model_fit = GLMHMM(n_states=s, maxiter=n_steps, tol=1e-300, seed=seed)
        model_fit.fit(X, y, init_params=init_params)

        # --- path B: manual update loop ---
        model_update = GLMHMM(n_states=s, maxiter=n_steps, tol=1e-300, seed=seed)
        opt_state = model_update.initialize_optimizer_and_state(init_params, X, y)
        params = init_params
        for _ in range(n_steps):
            params, opt_state = model_update.update(
                params, opt_state, X, y, session_starts
            )

        np.testing.assert_allclose(model_fit.coef_, model_update.coef_)
        np.testing.assert_allclose(model_fit.intercept_, model_update.intercept_)
        np.testing.assert_allclose(model_fit.initial_prob_, model_update.initial_prob_)
        np.testing.assert_allclose(
            model_fit.transition_prob_, model_update.transition_prob_
        )


# ---------------------------------------------------------------------------
# TestUpdate — single EM iteration via the public update() method
# ---------------------------------------------------------------------------


class TestUpdate:
    """GLMHMM.update() runs one EM iteration and persists the fitted attributes.

    Pytree-X coverage of update() is provided by the shared
    test_base_regressor_subclasses.py::TestModelVsPytree::test_update_pytree_x.
    """

    @staticmethod
    def _prepare(model, X, y):
        """Run the init handshake update() documents as its precondition."""
        init_params = model.initialize_params(X, y)
        opt_state = model.initialize_optimizer_and_state(init_params, X, y)
        return init_params, opt_state

    def test_update_runs_and_sets_attrs(self, glm_hmm_data):
        """One update() call returns a 5-tuple and sets every fit attribute."""
        d = glm_hmm_data
        model = GLMHMM(n_states=d["n_states"], solver_kwargs={"maxiter": 1})
        init_params, opt_state = self._prepare(model, d["X"], d["y"])

        params, _ = model.update(
            init_params, opt_state, d["X"], d["y"], d["session_starts"]
        )

        assert len(params) == 5
        assert all(v is not None for v in model._get_fit_state().values())

    def test_update_safe_calls_validate_inputs(self, glm_hmm_data, monkeypatch):
        """update() forwards X and y to GLMHMMValidator.validate_inputs exactly zero times."""
        d = glm_hmm_data
        model = GLMHMM(n_states=d["n_states"], solver_kwargs={"maxiter": 1})
        init_params, opt_state = self._prepare(model, d["X"], d["y"])

        calls = _spy_calls(monkeypatch, GLMHMMValidator, "validate_inputs")
        model.update(init_params, opt_state, d["X"], d["y"], d["session_starts"])

        assert len(calls) == 1
        _, kwargs = calls[0]
        assert kwargs["X"] is d["X"]
        assert kwargs["y"] is d["y"]

    def test_update_unsafe_no_calls_validate_inputs(self, glm_hmm_data, monkeypatch):
        """update() forwards X and y to GLMHMMValidator.validate_inputs exactly zero times."""
        d = glm_hmm_data
        model = GLMHMM(n_states=d["n_states"], solver_kwargs={"maxiter": 1})
        init_params, opt_state = self._prepare(model, d["X"], d["y"])

        calls = _spy_calls(monkeypatch, GLMHMMValidator, "validate_inputs")
        model.update(init_params, opt_state, d["X"], d["y"], d["session_starts"], safe=False)

        assert len(calls) == 0

    def test_update_forces_first_bin_new_session(self, glm_hmm_data, monkeypatch):
        """update() marks the first sample as a session start before the EM step,
        regardless of the session_starts passed in."""
        d = glm_hmm_data
        n = d["X"].shape[0]
        model = GLMHMM(n_states=d["n_states"], solver_kwargs={"maxiter": 1})
        init_params, opt_state = self._prepare(model, d["X"], d["y"])

        captured = {}
        real_update = model._optimizer_update

        def capturing(params, state, data, y, *, session_starts, **kwargs):
            captured["session_starts"] = session_starts
            return real_update(
                params, state, data, y, session_starts=session_starts, **kwargs
            )

        monkeypatch.setattr(model, "_optimizer_update", capturing)
        model.update(
            init_params,
            opt_state,
            d["X"],
            d["y"],
            session_starts=np.zeros(n, dtype=bool),
        )

        assert bool(captured["session_starts"][0]) is True

    def test_update_drops_leading_nan_and_preserves_session_boundaries(
        self, glm_hmm_data, monkeypatch
    ):
        """Leading NaN rows are dropped before the EM step and session boundaries at
        non-NaN rows land at the correct post-drop indices."""
        d = glm_hmm_data
        n = d["X"].shape[0]
        n_leading_nan = 3
        second_session_row = 30  # a non-NaN row with an explicit session boundary
        model = GLMHMM(n_states=d["n_states"], solver_kwargs={"maxiter": 1})
        init_params, opt_state = self._prepare(model, d["X"], d["y"])

        X_nan = d["X"].copy().astype(float)
        X_nan[:n_leading_nan, :] = np.nan
        session_starts = np.zeros(n, dtype=bool)
        session_starts[0] = True
        session_starts[second_session_row] = True

        captured = {}
        real_update = model._optimizer_update

        def capturing(params, state, data, y, *, session_starts, **kwargs):
            captured["session_starts"] = session_starts
            captured["n_samples"] = data.shape[0]
            return real_update(
                params, state, data, y, session_starts=session_starts, **kwargs
            )

        monkeypatch.setattr(model, "_optimizer_update", capturing)
        model.update(
            init_params, opt_state, X_nan, d["y"], session_starts=session_starts
        )

        assert captured["n_samples"] == n - n_leading_nan
        assert bool(captured["session_starts"][0]) is True
        # second_session_row shifts left by n_leading_nan after the drop
        assert (
            bool(captured["session_starts"][second_session_row - n_leading_nan]) is True
        )

    # a boundary far enough from the edges
    SECOND_SESSION_ROW = 30

    @pytest.mark.parametrize(
        "nan_rows, expected_second_start, expectation",
        [
            # NaN before either boundary: the marker on it moves to the next valid
            # sample, and every later row shifts left by one
            (0, SECOND_SESSION_ROW - 1, does_not_raise()),
            # NaN on the last row of the first session, i.e. a trailing border
            (SECOND_SESSION_ROW - 1, SECOND_SESSION_ROW - 1, does_not_raise()),
            # NaN on the boundary row itself: the marker must move to the next valid
            # sample rather than be dropped along with the row it sits on
            (SECOND_SESSION_ROW, SECOND_SESSION_ROW, does_not_raise()),
            # NaN strictly inside the second session: dropping it would splice bins
            # that are not adjacent in time
            (
                SECOND_SESSION_ROW + 10,
                None,
                pytest.raises(ValueError, match="requires continuous time-series data"),
            ),
            # no valid sample left to run the EM step on
            (
                slice(None),
                None,
                pytest.raises(ValueError, match="At least a NaN or an Inf"),
            ),
        ],
    )
    def test_update_nan_placement_preserves_session_structure(
        self,
        nan_rows,
        expected_second_start,
        expectation,
        glm_hmm_data,
        monkeypatch,
    ):
        """update() hands the EM step the session partition fit() would hand it.

        Each NaN placement is a distinct behaviour of the preamble: a NaN at a session
        border is dropped and the boundary re-indexed, a NaN inside a session is
        refused because dropping it would make two non-adjacent bins consecutive, and
        an all-NaN input leaves nothing to iterate on.
        """
        d = glm_hmm_data
        n = d["X"].shape[0]
        model = GLMHMM(n_states=d["n_states"], solver_kwargs={"maxiter": 1})
        init_params, opt_state = self._prepare(model, d["X"], d["y"])

        X_nan = d["X"].copy().astype(float)
        X_nan[nan_rows] = np.nan
        session_starts = np.zeros(n, dtype=bool)
        session_starts[0] = True
        session_starts[self.SECOND_SESSION_ROW] = True

        calls = _spy_calls(monkeypatch, model, "_optimizer_update")
        with expectation:
            model.update(init_params, opt_state, X_nan, d["y"], session_starts)

        if expected_second_start is not None:
            data, passed_starts = calls[0][0][2], calls[0][1]["session_starts"]
            assert data.shape[0] == n - 1
            np.testing.assert_array_equal(
                np.flatnonzero(np.asarray(passed_starts)), [0, expected_second_start]
            )


# ---------------------------------------------------------------------------
# TestGLMHMMParamsContainer — static helpers on the GLMHMMParams class itself
# ---------------------------------------------------------------------------


class TestGLMHMMParamsContainer:
    """Static helpers on GLMHMMParams that are not exercised via the model path."""

    def test_regularizable_subtrees_extracts_coef(self):
        """The single subtree accessor returns model_params.coef (params.py:27-31)."""
        n_features, n_states = 4, 3
        coef = jnp.arange(n_features * n_states, dtype=float).reshape(
            n_features, n_states
        )
        params = GLMHMMParams(
            model_params=GLMHMMModelParams(
                coef=coef,
                intercept=jnp.zeros(n_states),
                log_scale=jnp.zeros(n_states),
            ),
            hmm_params=HMMParams(
                log_initial_prob=jnp.log(jnp.ones(n_states) / n_states),
                log_transition_prob=jnp.log(jnp.ones((n_states, n_states)) / n_states),
            ),
        )
        wheres = GLMHMMParams.regularizable_subtrees()
        assert len(wheres) == 1
        extracted = wheres[0](params)
        # eqx.tree_at uses ``is`` identity on the where-accessor, so the lambda
        # must return the exact leaf — not a copy.
        assert extracted is params.model_params.coef
        assert jnp.array_equal(extracted, coef)


# ---------------------------------------------------------------------------
# TestInferenceMethods — smooth_proba / filter_proba / decode_state overrides
# ---------------------------------------------------------------------------


class TestInferenceMethods:
    """GLMHMM.smooth_proba/filter_proba/decode_state are thin overrides that add a
    GLM-HMM-specific docstring example and delegate to BaseHMM. These tests
    confirm the override wires X/y/session_starts/state_format through and that
    the public surface returns the expected shapes."""

    @pytest.fixture
    def fitted_model(self, glm_hmm_data, mock_glm_hmm_optimizer_run):
        model = GLMHMM(n_states=glm_hmm_data["n_states"])
        model.fit(
            glm_hmm_data["X"],
            glm_hmm_data["y"],
            init_params=glm_hmm_data["init_params"],
        )
        return model

    @pytest.mark.parametrize("method_name", ["smooth_proba", "filter_proba"])
    def test_returns_normalized_posteriors(
        self, method_name, fitted_model, glm_hmm_data
    ):
        out = getattr(fitted_model, method_name)(glm_hmm_data["X"], glm_hmm_data["y"])
        n, n_states = glm_hmm_data["X"].shape[0], glm_hmm_data["n_states"]
        assert out.shape == (n, n_states)
        assert jnp.allclose(jnp.asarray(out).sum(axis=1), 1.0)

    @pytest.mark.parametrize(
        "state_format, expected_shape",
        [
            ("one-hot", lambda n, s: (n, s)),
            ("index", lambda n, s: (n,)),
        ],
    )
    def test_decode_state_output_shape(
        self, state_format, expected_shape, fitted_model, glm_hmm_data
    ):
        out = fitted_model.decode_state(
            glm_hmm_data["X"], glm_hmm_data["y"], state_format=state_format
        )
        n, n_states = glm_hmm_data["X"].shape[0], glm_hmm_data["n_states"]
        assert out.shape == expected_shape(n, n_states)
        if state_format == "one-hot":
            assert jnp.all(jnp.asarray(out).sum(axis=1) == 1)

    @pytest.mark.parametrize(
        "method_name", ["smooth_proba", "filter_proba", "decode_state"]
    )
    def test_session_starts_forwarded_to_base(
        self, method_name, fitted_model, glm_hmm_data, monkeypatch
    ):
        """The override forwards session_starts verbatim to BaseHMM."""
        captured = {}
        real = getattr(BaseHMM, method_name)

        def spy(self_inner, X, y, **kw):
            captured.update(kw)
            return real(self_inner, X, y, **kw)

        monkeypatch.setattr(BaseHMM, method_name, spy)
        n = glm_hmm_data["X"].shape[0]
        is_ns = jnp.zeros(n, dtype=bool).at[0].set(True).at[n // 2].set(True)
        getattr(fitted_model, method_name)(
            glm_hmm_data["X"], glm_hmm_data["y"], session_starts=is_ns
        )
        np.testing.assert_array_equal(captured["session_starts"], is_ns)

    def test_decode_state_forwards_state_format(
        self, fitted_model, glm_hmm_data, monkeypatch
    ):
        """state_format reaches BaseHMM.decode_state unchanged."""
        captured = {}
        real = BaseHMM.decode_state

        def spy(self_inner, X, y, **kw):
            captured.update(kw)
            return real(self_inner, X, y, **kw)

        monkeypatch.setattr(BaseHMM, "decode_state", spy)
        fitted_model.decode_state(
            glm_hmm_data["X"], glm_hmm_data["y"], state_format="index"
        )
        assert captured["state_format"] == "index"

    def test_pynapple_input_returns_tsdframe(self, fitted_model, glm_hmm_data):
        n = glm_hmm_data["X"].shape[0]
        X_tsd = nap.TsdFrame(t=np.arange(n, dtype=float), d=glm_hmm_data["X"])
        y_tsd = nap.Tsd(t=np.arange(n, dtype=float), d=glm_hmm_data["y"])
        assert isinstance(fitted_model.smooth_proba(X_tsd, y_tsd), nap.TsdFrame)


def test_get_optimal_solver_params_config():
    """GLMHMM has no SVRG optimal-params config: returns (None, None, None)
    (glm_hmm.py:1258-1260)."""
    assert GLMHMM(n_states=2)._get_optimal_solver_params_config() == (
        None,
        None,
        None,
    )
