"""Tests for GLMHMM.fit and related fit-path validation."""

import warnings
from contextlib import nullcontext as does_not_raise
from copy import deepcopy
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pynapple as nap
import pytest

from nemos.glm_hmm.glm_hmm import GLMHMM
from nemos.glm_hmm.validation import GLMHMMValidator
from nemos.pytrees import FeaturePytree

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
    assert all(val is not None for val in fixture.model._get_fit_state().values())
    assert fixture.model._get_fit_state().keys() == expected_state.keys()


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
        assert all(v is not None for v in model_pynap._get_fit_state().values())

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


class TestSolverConfiguration:
    """Verify that _initialize_optimizer_and_state wires EM correctly."""

    @pytest.mark.parametrize(
        "regularizer, solver_name",
        [
            ("UnRegularized", "LBFGS"),
            ("Ridge", "LBFGS"),
        ],
    )
    def test_valid_solver_regularizer_combo_smoke(
        self, regularizer, solver_name, glm_hmm_data, monkeypatch
    ):
        """Valid regularizer+solver pairs initialise without error."""
        model = GLMHMM(
            n_states=glm_hmm_data["n_states"],
            regularizer=regularizer,
            solver_name=solver_name,
        )
        # Patch so the real _initialize_optimizer_and_state runs but EM is skipped.
        real_init = GLMHMM._initialize_optimizer_and_state
        noop = lambda p, *_, **__: (
            p,
            SimpleNamespace(iterations=1, converged=True),
        )  # noqa: E731

        def patched(self, init_params, data, y):
            result = real_init(self, init_params, data, y)
            self._optimizer_run = noop
            return result

        monkeypatch.setattr(GLMHMM, "_initialize_optimizer_and_state", patched)
        model.fit(
            glm_hmm_data["X"],
            glm_hmm_data["y"],
            init_params=glm_hmm_data["init_params"],
        )
        assert callable(model._optimizer_run)

    def test_mismatched_solver_raises(self, glm_hmm_data):
        """Lasso regularizer with LBFGS solver raises ValueError."""
        with pytest.raises(ValueError):
            GLMHMM(
                n_states=glm_hmm_data["n_states"],
                regularizer="Lasso",
                solver_name="LBFGS",
                regularizer_strength=1.0,
            )

    def test_bernoulli_single_solver_smoke(self, glm_hmm_data, monkeypatch):
        """Bernoulli (fixed scale) initialises with a single solver path."""
        model = GLMHMM(n_states=glm_hmm_data["n_states"], observation_model="Bernoulli")
        real_init = GLMHMM._initialize_optimizer_and_state
        noop = lambda p, *_, **__: (
            p,
            SimpleNamespace(iterations=1, converged=True),
        )  # noqa: E731

        def patched(self, init_params, data, y):
            result = real_init(self, init_params, data, y)
            self._optimizer_run = noop
            return result

        monkeypatch.setattr(GLMHMM, "_initialize_optimizer_and_state", patched)
        model.fit(
            glm_hmm_data["X"],
            glm_hmm_data["y"],
            init_params=glm_hmm_data["init_params"],
        )
        assert callable(model._optimizer_run)

    def test_gaussian_separable_scale_smoke(self, glm_hmm_data, monkeypatch):
        """Gaussian (separable scale + analytical update) initialises without error."""
        rng = np.random.default_rng(1)
        n, k, s = 100, 2, 3
        X = np.ones((n, k))
        y = rng.standard_normal(n)
        init_params = glm_hmm_data["init_params"]

        model = GLMHMM(n_states=s, observation_model="Gaussian")
        real_init = GLMHMM._initialize_optimizer_and_state
        noop = lambda p, *_, **__: (
            p,
            SimpleNamespace(iterations=1, converged=True),
        )  # noqa: E731

        def patched(self, init_params, data, yy):
            result = real_init(self, init_params, data, yy)
            self._optimizer_run = noop
            return result

        monkeypatch.setattr(GLMHMM, "_initialize_optimizer_and_state", patched)
        model.fit(X, y, init_params=init_params)
        assert callable(model._optimizer_run)

    @pytest.mark.parametrize(
        "instantiate_base_regressor_subclass",
        INSTANTIATE_MODEL_AND_SIMULATE,
        indirect=True,
    )
    def test_is_new_session_forwarded_to_initialization(
        self, instantiate_base_regressor_subclass, monkeypatch
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

        real_init_opt = GLMHMM._initialize_optimizer_and_state
        noop = lambda p, *_, **__: (
            p,
            SimpleNamespace(iterations=1, converged=True),
        )  # noqa: E731

        def patched_init_opt(self, init_params, data, yy):
            result = real_init_opt(self, init_params, data, yy)
            self._optimizer_run = noop
            return result

        monkeypatch.setattr(
            GLMHMM, "_model_specific_initialization", capturing_model_init
        )
        monkeypatch.setattr(GLMHMM, "_initialize_optimizer_and_state", patched_init_opt)

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
