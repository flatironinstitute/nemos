"""Tests for GLMHMM.fit and related fit-path validation."""

from contextlib import nullcontext as does_not_raise
from copy import deepcopy

import jax.numpy as jnp
import numpy as np
import pynapple as nap
import pytest

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


# ---------------------------------------------------------------------------
# test_get_fit_attrs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "instantiate_base_regressor_subclass", INSTANTIATE_MODEL_ONLY, indirect=True
)
def test_get_fit_attrs(instantiate_base_regressor_subclass):
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
    fixture.model.solver_kwargs = {"maxiter": 1}
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

    @pytest.fixture
    def fit_weights_dimensionality_expectation(
        self, instantiate_base_regressor_subclass
    ):
        """Expected error/no-error for each coef dimensionality."""
        return {
            0: pytest.raises(ValueError, match=r"dimensionality"),
            1: pytest.raises(ValueError, match=r"dimensionality"),
            2: does_not_raise(),
            3: pytest.raises(ValueError, match=r"dimensionality"),
        }

    def test_fit_pynapple_tsd(self, instantiate_base_regressor_subclass):
        """Pynapple TSD/TsdFrame accepted; session boundaries encoded via is_new_session affect the fit."""
        fixture = instantiate_base_regressor_subclass

        n = 50
        X_np = fixture.X[:n]
        y_np = fixture.y[:n]
        X_nap = nap.TsdFrame(t=np.arange(n, dtype=float), d=X_np)
        y_nap = nap.Tsd(t=np.arange(n, dtype=float), d=y_np)

        init_params = (
            fixture.params.model_params.coef,
            fixture.params.model_params.intercept,
            fixture.params.model_params.log_scale,
            jnp.exp(fixture.params.hmm_params.log_initial_prob),
            jnp.exp(fixture.params.hmm_params.log_transition_prob),
        )

        # Pynapple TSD/TsdFrame input accepted; all fit attributes set afterwards
        model_pynap = fixture.model
        model_pynap.fit(X_nap, y_nap, init_params=init_params)
        assert all(v is not None for v in model_pynap._get_fit_state().values())

        # Providing an extra session boundary changes the M-step for initial_prob.
        # Use explicit is_new_session arrays so the test is not sensitive to fixture maxiter.
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

    @pytest.mark.parametrize("dim_weights", [0, 1, 2, 3])
    @pytest.mark.requires_x64
    def test_fit_weights_dimensionality(
        self,
        dim_weights,
        instantiate_base_regressor_subclass,
        fit_weights_dimensionality_expectation,
        mock_glm_hmm_optimizer_run,
    ):
        """Coef with wrong number of dimensions raises; correct ndim=2 does not."""
        fixture = instantiate_base_regressor_subclass
        expectation = fit_weights_dimensionality_expectation[dim_weights]
        n_samples, n_features = fixture.X.shape

        coef_shape = DEFAULT_GLM_COEF_SHAPE[fixture.model.__class__.__name__]
        if dim_weights == 0:
            init_w = jnp.array([])
        elif dim_weights == 1:
            init_w = jnp.zeros((n_features,))
        elif dim_weights == 2:
            init_w = jnp.zeros(coef_shape)
        else:
            init_w = jnp.zeros(coef_shape + (1,) * (dim_weights - 2))

        with expectation:
            fixture.model.fit(
                fixture.X,
                fixture.y,
                init_params=(
                    init_w,
                    fixture.params.model_params.intercept,
                    fixture.params.model_params.log_scale,
                    jnp.exp(fixture.params.hmm_params.log_initial_prob),
                    jnp.exp(fixture.params.hmm_params.log_transition_prob),
                ),
            )

    @pytest.mark.parametrize(
        "dim_intercepts, expectation",
        [
            (0, pytest.raises(ValueError, match=r"Unexpected array dimensionality")),
            (1, does_not_raise()),
            (2, pytest.raises(ValueError, match=r"Unexpected array dimensionality")),
            (3, pytest.raises(ValueError, match=r"Unexpected array dimensionality")),
        ],
    )
    @pytest.mark.requires_x64
    def test_fit_intercepts_dimensionality(
        self,
        dim_intercepts,
        expectation,
        instantiate_base_regressor_subclass,
        mock_glm_hmm_optimizer_run,
    ):
        """Intercept with wrong ndim raises; ndim=1 (shape (n_states,)) does not."""
        fixture = instantiate_base_regressor_subclass
        n_states = DEFAULT_GLM_COEF_SHAPE[fixture.model.__class__.__name__][1]
        if dim_intercepts == 0:
            init_b = jnp.array(1.0)
        else:
            init_b = jnp.ones((n_states,) + (1,) * (dim_intercepts - 1))

        with expectation:
            fixture.model.fit(
                fixture.X,
                fixture.y,
                init_params=(
                    fixture.params.model_params.coef,
                    init_b,
                    fixture.params.model_params.log_scale,
                    jnp.exp(fixture.params.hmm_params.log_initial_prob),
                    jnp.exp(fixture.params.hmm_params.log_transition_prob),
                ),
            )

    # Parametrize table shared between test_fit_init_glm_params_type.
    # init_params is a 5-tuple (coef, intercept, scale, init_prob, trans_prob).
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
            # Dict coef while X is a plain array
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
    @pytest.mark.requires_x64
    def test_fit_init_glm_params_type(
        self,
        instantiate_base_regressor_subclass,
        expectation,
        init_params,
        mock_glm_hmm_optimizer_run,
    ):
        """Valid init_params accepted; invalid types/lengths rejected with clear errors."""
        fixture = instantiate_base_regressor_subclass
        with expectation:
            fixture.model.fit(fixture.X, fixture.y, init_params=init_params)

    @pytest.mark.parametrize(
        "delta_n_features, expectation",
        [
            (-1, pytest.raises(ValueError, match="Inconsistent number of features")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="Inconsistent number of features")),
        ],
    )
    @pytest.mark.requires_x64
    def test_fit_n_feature_consistency_weights(
        self,
        delta_n_features,
        expectation,
        instantiate_base_regressor_subclass,
        mock_glm_hmm_optimizer_run,
    ):
        """Coef feature count must match X.shape[1]; mismatch raises ValueError."""
        fixture = instantiate_base_regressor_subclass
        init_w = jnp.zeros((fixture.X.shape[1] + delta_n_features, 3))
        init_b = jnp.ones(3)
        with expectation:
            fixture.model.fit(
                fixture.X,
                fixture.y,
                init_params=(
                    init_w,
                    init_b,
                    fixture.params.model_params.log_scale,
                    jnp.exp(fixture.params.hmm_params.log_initial_prob),
                    jnp.exp(fixture.params.hmm_params.log_transition_prob),
                ),
            )

    @pytest.mark.parametrize(
        "X, y, expectation",
        [
            # NaN at start/end of array — allowed (epoch boundary)
            (np.array([[np.nan], [0]]), np.array([0, 1]), does_not_raise()),
            (np.array([[0], [np.nan]]), np.array([0, 1]), does_not_raise()),
            (np.array([[0], [0]]), np.array([np.nan, 1]), does_not_raise()),
            (np.array([[0], [0]]), np.array([0, np.nan]), does_not_raise()),
            # NaN in the middle of data — rejected
            (
                np.array([[0], [np.nan], [0]]),
                np.array([0, 1, 2]),
                pytest.raises(ValueError, match="requires continuous time-series data"),
            ),
            (
                np.array([[0], [0], [0]]),
                np.array([0, np.nan, 2]),
                pytest.raises(ValueError, match="requires continuous time-series data"),
            ),
            # Pynapple: NaN inside an epoch — rejected
            (
                nap.TsdFrame(
                    t=np.arange(5),
                    d=np.array([[0], [np.nan], [0], [0], [0]]),
                    time_support=nap.IntervalSet([0, 3], [2, 5]),
                ),
                np.array([0, 1, 2, 4, 5]),
                pytest.raises(ValueError, match="requires continuous time-series data"),
            ),
            # Pynapple: NaN at epoch boundary — allowed
            (
                nap.TsdFrame(
                    t=np.arange(5),
                    d=np.array([[0], [0], [np.nan], [0], [0]]),
                    time_support=nap.IntervalSet([0, 3], [2, 5]),
                ),
                np.array([0, 1, 2, 4, 5]),
                does_not_raise(),
            ),
            # Pynapple y: NaN inside epoch — rejected
            (
                np.zeros((5, 1)),
                nap.Tsd(
                    t=np.arange(5),
                    d=np.array([0, np.nan, 2, 4, 5]),
                    time_support=nap.IntervalSet([0, 3], [2, 5]),
                ),
                pytest.raises(ValueError, match="requires continuous time-series data"),
            ),
            # Pynapple y: NaN at epoch boundary — allowed
            (
                np.zeros((5, 1)),
                nap.Tsd(
                    t=np.arange(5),
                    d=np.array([0, 1, np.nan, 4, 5]),
                    time_support=nap.IntervalSet([0, 3], [2, 5]),
                ),
                does_not_raise(),
            ),
            # Multiple consecutive NaNs in the middle — rejected
            (
                np.array([[0], [np.nan], [np.nan], [0]]),
                np.array([0, 1, 2, 3]),
                pytest.raises(ValueError, match="requires continuous time-series data"),
            ),
            # Multiple consecutive NaNs at start — allowed
            (
                np.array([[np.nan], [np.nan], [0]]),
                np.array([0, 1, 2]),
                does_not_raise(),
            ),
            # Multiple consecutive NaNs at end — allowed
            (
                np.array([[0], [np.nan], [np.nan]]),
                np.array([0, 1, 2]),
                does_not_raise(),
            ),
            # All NaN — rejected (caught by parent validation)
            (
                np.array([[np.nan], [np.nan]]),
                np.array([np.nan, np.nan]),
                pytest.raises(ValueError),
            ),
            # No NaN — allowed
            (np.array([[0], [1]]), np.array([0, 1]), does_not_raise()),
            # Pynapple: NaN at start of second epoch — allowed
            (
                nap.TsdFrame(
                    t=np.arange(5),
                    d=np.array([[0], [0], [np.nan], [0], [0]]),
                    time_support=nap.IntervalSet([0, 2], [2, 5]),
                ),
                np.zeros(5),
                does_not_raise(),
            ),
            # Both X and y NaN in middle at different positions — rejected
            (
                np.array([[0], [np.nan], [0], [0]]),
                np.array([0, 1, np.nan, 3]),
                pytest.raises(ValueError, match="requires continuous time-series data"),
            ),
            # Both X and y NaN in middle at same position — rejected
            (
                np.array([[0], [np.nan], [0]]),
                np.array([0, np.nan, 2]),
                pytest.raises(ValueError, match="requires continuous time-series data"),
            ),
        ],
    )
    def test_nan_between_epochs(
        self, instantiate_base_regressor_subclass, X, y, expectation
    ):
        """NaN values are allowed only at epoch boundaries, never in the middle of a session."""
        fixture = instantiate_base_regressor_subclass
        with expectation:
            fixture.model._validator.validate_inputs(X, y)
