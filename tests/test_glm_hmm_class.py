"""Tests for GLMHMM.fit and related fit-path validation."""

import warnings
from contextlib import nullcontext as does_not_raise
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pynapple as nap
import pytest

from nemos.glm.params import GLMParams
from nemos.glm_hmm.glm_hmm import GLMHMM
from nemos.glm_hmm.initialize_parameters import (
    kmeans_glm_params_init,
)
from nemos.glm_hmm.params import GLMHMMModelParams
from nemos.glm_hmm.validation import GLMHMMValidator
from nemos.hmm.expectation_maximization import EMState
from nemos.pytrees import FeaturePytree
from nemos.regularizer import Ridge, UnRegularized

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
            X_np, y_np, is_new_session=is_ns_single, init_params=init_params
        )
        model_multi.fit(X_np, y_np, is_new_session=is_ns_multi, init_params=init_params)
        assert not jnp.array_equal(
            model_single.initial_prob_, model_multi.initial_prob_
        )

    @pytest.mark.parametrize(
        "dim_weights, expectation",
        [
            (0, pytest.raises(ValueError, match=r"dimensionality")),
            (1, pytest.raises(ValueError, match=r"dimensionality")),
            (2, does_not_raise()),
            (3, pytest.raises(ValueError, match=r"dimensionality")),
        ],
    )
    def test_fit_weights_dimensionality(
        self,
        dim_weights,
        expectation,
        instantiate_base_regressor_subclass,
    ):
        """Coef with wrong ndim raises via validator; correct ndim=2 does not."""
        fixture = instantiate_base_regressor_subclass
        n_features = fixture.X.shape[1]
        coef_shape = DEFAULT_GLM_COEF_SHAPE[fixture.model.__class__.__name__]
        if dim_weights == 0:
            init_w = jnp.array([])
        elif dim_weights == 1:
            init_w = jnp.zeros((n_features,))
        elif dim_weights == 2:
            init_w = jnp.zeros(coef_shape)
        else:
            init_w = jnp.zeros(coef_shape + (1,) * (dim_weights - 2))

        validator = GLMHMMValidator(n_states=N_STATES)
        params = (
            init_w,
            fixture.params.model_params.intercept,
            fixture.params.model_params.log_scale,
            jnp.exp(fixture.params.hmm_params.log_initial_prob),
            jnp.exp(fixture.params.hmm_params.log_transition_prob),
        )
        with expectation:
            validator.validate_and_cast_params(params)

    @pytest.mark.parametrize(
        "dim_intercepts, expectation",
        [
            (0, pytest.raises(ValueError, match=r"Unexpected array dimensionality")),
            (1, does_not_raise()),
            (2, pytest.raises(ValueError, match=r"Unexpected array dimensionality")),
            (3, pytest.raises(ValueError, match=r"Unexpected array dimensionality")),
        ],
    )
    def test_fit_intercepts_dimensionality(
        self,
        dim_intercepts,
        expectation,
        instantiate_base_regressor_subclass,
    ):
        """Intercept with wrong ndim raises via validator; ndim=1 does not."""
        fixture = instantiate_base_regressor_subclass
        n_states = DEFAULT_GLM_COEF_SHAPE[fixture.model.__class__.__name__][1]
        if dim_intercepts == 0:
            init_b = jnp.array(1.0)
        else:
            init_b = jnp.ones((n_states,) + (1,) * (dim_intercepts - 1))

        validator = GLMHMMValidator(n_states=N_STATES)
        params = (
            fixture.params.model_params.coef,
            init_b,
            fixture.params.model_params.log_scale,
            jnp.exp(fixture.params.hmm_params.log_initial_prob),
            jnp.exp(fixture.params.hmm_params.log_transition_prob),
        )
        with expectation:
            validator.validate_and_cast_params(params)

    # Parametrize table for test_fit_init_glm_params_type.
    _fit_init_params_type_cases = (
        "expectation, init_params",
        [
            # Valid: correct shapes for all five params
            (
                does_not_raise(),
                (
                    jnp.zeros((2, 3)),
                    jnp.zeros((3,)),
                    jnp.ones((3,)),
                    jnp.ones(3) / 3,
                    jnp.ones((3, 3)) / 3,
                ),
            ),
            # Wrong tuple length (not 5)
            (
                pytest.raises(ValueError, match="Params must have length 5"),
                (jnp.zeros((1, 2, 3)), jnp.zeros((3,))),
            ),
            # Dict coef while X is a plain array — tested at consistency level
            (
                pytest.raises((AttributeError, TypeError)),
                (
                    dict(p1=jnp.zeros((1, 3)), p2=jnp.zeros((1, 3))),
                    jnp.zeros((3,)),
                    jnp.ones((3,)),
                    jnp.ones(3) / 3,
                    jnp.ones((3, 3)) / 3,
                ),
            ),
            # FeaturePytree coef while X is a plain array
            (
                pytest.raises(TypeError, match=r"X and coef have mismatched structure"),
                (
                    FeaturePytree(p1=jnp.zeros((1, 3)), p2=jnp.zeros((1, 3))),
                    jnp.zeros((3,)),
                    jnp.ones((3,)),
                    jnp.ones(3) / 3,
                    jnp.ones((3, 3)) / 3,
                ),
            ),
            # Scalar instead of tuple
            (pytest.raises(ValueError, match="Params must have length 5"), 0),
            # Set instead of tuple
            (pytest.raises(ValueError, match="Params must have length 5"), {0, 1}),
            # String intercept
            (
                pytest.raises(
                    TypeError, match="Failed to convert parameters to JAX arrays"
                ),
                (
                    jnp.zeros((2, 3)),
                    "",
                    jnp.ones((3,)),
                    jnp.ones(3) / 3,
                    jnp.ones((3, 3)) / 3,
                ),
            ),
            # String coef
            (
                pytest.raises(
                    TypeError, match="Failed to convert parameters to JAX arrays"
                ),
                (
                    "",
                    jnp.zeros((3,)),
                    jnp.ones((3,)),
                    jnp.ones(3) / 3,
                    jnp.ones((3, 3)) / 3,
                ),
            ),
        ],
    )

    @pytest.mark.parametrize(*_fit_init_params_type_cases)
    def test_fit_init_glm_params_type(
        self,
        instantiate_base_regressor_subclass,
        expectation,
        init_params,
        mock_glm_hmm_optimizer_run,
    ):
        """Valid init_params accepted; invalid types/lengths rejected with clear errors.

        The dict-coef and FeaturePytree cases require the consistency check that
        runs inside fit(), so model.fit() is still called for those.  All other
        cases only need the validation pipeline and use the validator directly.
        """
        fixture = instantiate_base_regressor_subclass

        # Cases that require X-vs-coef structure check inside fit()
        needs_fit = (
            isinstance(init_params, tuple)
            and len(init_params) == 5
            and (isinstance(init_params[0], (dict, FeaturePytree)))
        )

        if needs_fit:
            with expectation:
                fixture.model.fit(fixture.X, fixture.y, init_params=init_params)
        else:
            validator = GLMHMMValidator(n_states=N_STATES)
            with expectation:
                validated = validator.validate_and_cast_params(init_params)
                # For the valid case, also run consistency check
                if isinstance(init_params, tuple) and len(init_params) == 5:
                    validator.validate_consistency(validated, X=fixture.X, y=fixture.y)

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

    @pytest.mark.parametrize(
        "delta_n_features, expectation",
        [
            (-1, pytest.raises(ValueError, match="Inconsistent number of features")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="Inconsistent number of features")),
        ],
    )
    def test_fit_n_feature_consistency_weights(
        self,
        delta_n_features,
        expectation,
        instantiate_base_regressor_subclass,
    ):
        """Coef feature count must match X.shape[1]; mismatch raises ValueError."""
        fixture = instantiate_base_regressor_subclass
        init_w = jnp.zeros((fixture.X.shape[1] + delta_n_features, N_STATES))
        init_b = jnp.ones(N_STATES)
        init_scale = fixture.params.model_params.log_scale
        init_ip = jnp.exp(fixture.params.hmm_params.log_initial_prob)
        init_tp = jnp.exp(fixture.params.hmm_params.log_transition_prob)

        validator = GLMHMMValidator(n_states=N_STATES)
        params = (init_w, init_b, init_scale, init_ip, init_tp)
        with expectation:
            model_params = validator.validate_and_cast_params(params)
            validator.validate_consistency(model_params, X=fixture.X, y=fixture.y)


@pytest.fixture
def glm_hmm_data():
    """Minimal X, y and valid init_params for GLMHMM with n_states=3, n_features=2."""
    rng = np.random.default_rng(0)
    n, k, s = 100, 2, 3
    X = np.ones((n, k))
    y = np.zeros(n)
    y[rng.choice(n, n // 3, replace=False)] = 1.0
    coef = jnp.zeros((k, s))
    intercept = jnp.zeros((s,))
    scale = jnp.ones((s,))
    init_prob = jnp.ones(s) / s
    trans_prob = jnp.ones((s, s)) / s
    return dict(
        X=X,
        y=y,
        init_params=(coef, intercept, scale, init_prob, trans_prob),
        n_states=s,
    )


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
        assert len(calls) >= 1
        primary = calls[0]
        assert isinstance(primary["regularizer"], expected_regularizer_type)
        assert solver_name in primary["solver_name"]

    def test_mismatched_solver_raises(self, glm_hmm_data):
        """Lasso regularizer with LBFGS solver raises ValueError."""
        with pytest.raises(ValueError):
            GLMHMM(
                n_states=glm_hmm_data["n_states"],
                regularizer="Lasso",
                solver_name="LBFGS",
                regularizer_strength=1.0,
            )

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
    def test_is_new_session_forwarded_to_initialization(
        self,
        instantiate_base_regressor_subclass,
        mock_glm_hmm_optimizer_run,
        monkeypatch,
    ):
        """is_new_session passed to fit() reaches _model_specific_initialization."""
        fixture = instantiate_base_regressor_subclass
        n = 50
        X = fixture.X[:n]
        y = fixture.y[:n]
        is_ns = np.zeros(n, dtype=bool)
        is_ns[0] = True
        is_ns[n // 2] = True

        captured = {}
        original_model_init = GLMHMM._model_specific_initialization

        def capturing_model_init(self, X, y, is_new_session=None):
            captured["is_new_session"] = is_new_session
            return original_model_init(self, X, y, is_new_session)

        monkeypatch.setattr(
            GLMHMM, "_model_specific_initialization", capturing_model_init
        )

        fixture.model.fit(X, y, is_new_session=is_ns)

        assert "is_new_session" in captured
        assert captured["is_new_session"] is not None
        # The session boundary at n//2 should be present in the forwarded array.
        assert bool(captured["is_new_session"][n // 2])

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
# TestFitDelegation — fit() calls validate_and_cast_is_new_session
# ---------------------------------------------------------------------------


class TestFitDelegation:
    """fit() and simulate() must delegate session-boundary handling to validate_and_cast_is_new_session."""

    def test_fit_calls_validate_and_cast_is_new_session(
        self, glm_hmm_data, mock_glm_hmm_optimizer_run, monkeypatch
    ):
        n = glm_hmm_data["X"].shape[0]
        default_is_new_session = jnp.zeros(n, dtype=bool).at[0].set(True)
        mock = MagicMock(return_value=default_is_new_session)
        monkeypatch.setattr(GLMHMMValidator, "validate_and_cast_is_new_session", mock)

        model = GLMHMM(n_states=glm_hmm_data["n_states"])
        model.fit(
            glm_hmm_data["X"],
            glm_hmm_data["y"],
            init_params=glm_hmm_data["init_params"],
        )

        mock.assert_called_once()

    def test_simulate_calls_validate_and_cast_is_new_session(
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

        default_is_new_session = jnp.zeros(n, dtype=bool).at[0].set(True)
        mock = MagicMock(return_value=default_is_new_session)
        monkeypatch.setattr(GLMHMMValidator, "validate_and_cast_is_new_session", mock)
        stub_glmhmm_simulate(n)

        model.simulate(jax.random.key(0), X)

        mock.assert_called_once()


# ---------------------------------------------------------------------------
# TestSetParams — set_params joint-initialization path
# ---------------------------------------------------------------------------


class TestSetParams:
    """Cover the set_params override that orders initialization_funcs before initialization_kwargs."""

    def test_joint_initialization_sets_funcs_first(self, glm_hmm_data):
        """initialization_funcs is applied before initialization_kwargs when both are passed.

        The second call raises because initialization_kwargs is not a valid sklearn param;
        but initialization_funcs must already have been updated at that point.
        """
        model = GLMHMM(n_states=glm_hmm_data["n_states"])
        # Build funcs that are provably different from the current ones
        new_funcs = dict(**model.initialization_funcs)
        new_funcs["glm_params_init"] = kmeans_glm_params_init
        assert (
            model.initialization_funcs["glm_params_init"] is not kmeans_glm_params_init
        )

        with pytest.raises(
            ValueError, match="Invalid parameter 'initialization_kwargs'"
        ):
            model.set_params(
                initialization_funcs=new_funcs, initialization_kwargs="unused"
            )

        # initialization_funcs was updated before the error
        assert model.initialization_funcs["glm_params_init"] is kmeans_glm_params_init


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

        is_new_session = jnp.zeros(n, dtype=bool).at[0].set(True)
        mock_vapi = MagicMock(
            return_value=(
                model._get_model_params(),
                jnp.asarray(X),
                None,
                is_new_session,
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

    def test_simulate_requires_fit(self, glm_hmm_data):
        """simulate() raises ValueError if called before fit."""
        model = GLMHMM(n_states=glm_hmm_data["n_states"])
        with pytest.raises(ValueError, match="not fitted"):
            model.simulate(jax.random.key(0), glm_hmm_data["X"])

    def test_simulate_forces_first_bin_new_session(
        self, glm_hmm_data, mock_glm_hmm_optimizer_run, monkeypatch
    ):
        """is_new_session[0] is always True regardless of what the user passes."""
        model = GLMHMM(n_states=glm_hmm_data["n_states"])
        model.fit(
            glm_hmm_data["X"],
            glm_hmm_data["y"],
            init_params=glm_hmm_data["init_params"],
        )
        X = glm_hmm_data["X"]
        n = X.shape[0]
        captured = {}

        def capturing_simulate(self_inner, key, params, data, is_new_session):
            captured["is_new_session"] = is_new_session
            return jnp.zeros(n), jnp.zeros(n), jnp.zeros(n, dtype=int)

        monkeypatch.setattr(GLMHMM, "_simulate", capturing_simulate)
        model.simulate(jax.random.key(0), X, is_new_session=jnp.zeros(n, dtype=bool))

        assert bool(captured["is_new_session"][0]) is True

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
