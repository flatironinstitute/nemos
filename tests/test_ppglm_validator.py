from contextlib import nullcontext as does_not_raise

import jax
import jax.numpy as jnp
import numpy as np
import pynapple as nap
import pytest

from nemos.pp_glm.mock_ppglm import MockPopulationPPGLM, MockPPGLM
from nemos.pp_glm.data import X_ppglm, y_ppglm
from nemos.pp_glm.validation import PopulationPPGLMValidator, PPGLMValidator

@pytest.fixture
def eval_fn():
    def fn(pts):
        return jnp.ones((pts.shape[0], 5))

    return fn


@pytest.fixture
def validator(eval_fn) -> PPGLMValidator:
    return MockPPGLM(
        n_basis_funcs=5,
    )._validator


@pytest.fixture
def population_validator(eval_fn) -> PopulationPPGLMValidator:
    return MockPopulationPPGLM(
        n_basis_funcs=5,
    )._validator


@pytest.fixture
def recording_time():
    return nap.IntervalSet(start=0.0, end=10.0)


@pytest.fixture
def valid_X_y(recording_time):
    rng = np.random.default_rng(0)
    X = nap.TsGroup(
        {
            0: nap.Ts(np.sort(rng.uniform(0, 10, 50))),
            1: nap.Ts(np.sort(rng.uniform(0, 10, 40))),
        },
        time_support=recording_time,
    )
    y = nap.TsGroup(
        {0: nap.Ts(np.sort(rng.uniform(0, 10, 30)))},
        time_support=recording_time,
    )
    return X, y


@pytest.fixture
def processed_X_y(eval_fn, recording_time, valid_X_y):
    """Fresh model per test — _preprocess_inputs mutates instance state."""
    X, y = valid_X_y

    X, y = X.to_tsd(), y.to_tsd()
    event_times = jnp.asarray(X.t)
    spike_times = jnp.asarray(y.t)
    y_idx = jnp.searchsorted(event_times, spike_times)

    X = X_ppglm(
        times=event_times,
        ids=jnp.asarray(X.d, dtype=int),
    )

    y = y_ppglm(
        times=spike_times,
        ids=jnp.asarray(y.d, dtype=int),
        idx=y_idx,
    )

    return X, y


class TestPPGLMValidator:
    """Test suite for input validation logic in PPGLMValidator."""

    def _times_to_tsd(self, times, support):
        return nap.TsGroup({0: nap.Ts(times)}, time_support=support).to_tsd()

    @pytest.mark.parametrize(
        "X_times, X_support, y_times, y_support, rec_time, expectation",
        [
            # valid: data fits within recording_time
            (
                np.sort(np.random.uniform(0, 10, 30)),
                nap.IntervalSet(0, 10),
                np.sort(np.random.uniform(0, 10, 20)),
                nap.IntervalSet(0, 10),
                nap.IntervalSet(0, 10),
                does_not_raise(),
            ),
            # X extends beyond recording_time
            (
                np.array([1.0, 11.0]),
                nap.IntervalSet(0, 15),
                np.sort(np.random.uniform(0, 10, 20)),
                nap.IntervalSet(0, 10),
                nap.IntervalSet(0, 10),
                pytest.raises(ValueError, match="outside recording_time"),
            ),
            # y extends beyond recording_time
            (
                np.sort(np.random.uniform(0, 10, 30)),
                nap.IntervalSet(0, 10),
                np.array([1.0, 11.0]),
                nap.IntervalSet(0, 15),
                nap.IntervalSet(0, 10),
                pytest.raises(ValueError, match="outside recording_time"),
            ),
            # recording_time extends far beyond X (> tol=1s)
            (
                np.sort(np.random.uniform(0, 5, 30)),
                nap.IntervalSet(0, 5),
                np.sort(np.random.uniform(0, 5, 20)),
                nap.IntervalSet(0, 5),
                nap.IntervalSet(0, 10),  # 5s beyond data
                pytest.raises(ValueError, match="extends more than"),
            ),
            # recording_time extends just within tol (< 1s beyond data)
            (
                np.sort(np.random.uniform(0, 10, 30)),
                nap.IntervalSet(0, 10),
                np.sort(np.random.uniform(0, 10, 20)),
                nap.IntervalSet(0, 10),
                nap.IntervalSet(0, 10.5),  # 0.5s beyond — within tol
                does_not_raise(),
            ),
        ],
    )
    def test_validate_span(
        self, validator, X_times, X_support, y_times, y_support, rec_time, expectation
    ):
        """
        This validation is agnostic to gaps in either the data or recording_time
        because it's designed to just test the span.
        """
        X_tsd = self._times_to_tsd(X_times, X_support)
        y_tsd = self._times_to_tsd(y_times, y_support)
        with expectation:
            validator.validate_span(X_tsd, y_tsd, rec_time)

    def test_validate_consistency(self, validator, processed_X_y):
        # valid inputs, no error
        X, y = processed_X_y
        # n_features is 10
        n_features = int(jnp.unique(X.ids).size * 5)
        params = validator.to_model_params((jnp.zeros(n_features), jnp.zeros(1)))
        validator.validate_consistency(params, X=X)

        # incorrect number of features
        params = validator.to_model_params((jnp.zeros(5), jnp.zeros(1)))
        with pytest.raises(ValueError, match="Inconsistent number of features"):
            validator.validate_consistency(params, X=X)

        # test that no neuron ids are skipped
        X_skipped = X_ppglm(
            times=jnp.array([1.0, 2.0, 3.0]),
            ids=jnp.array([0, 2, 2]),  # skips id 1
        )
        params = validator.to_model_params((jnp.zeros(10), jnp.zeros(1)))
        with pytest.raises(ValueError, match="consecutive"):
            validator.validate_consistency(params, X=X_skipped)

        # should not raise if X is None
        params = validator.to_model_params((jnp.zeros(10), jnp.zeros(1)))
        validator.validate_consistency(params, X=None)

    @pytest.mark.parametrize(
        "key, dtype, expectation",
        [
            # valid uint32 PRNGKey
            (
                jax.random.PRNGKey(0),
                jnp.uint32,
                does_not_raise(),
            ),
            # valid float64 (pre-solver form)
            (
                None,
                jnp.float64,
                does_not_raise(),
            ),
            # wrong dtype for uint32 check: passing float64 when uint32 expected
            (
                None,
                jnp.uint32,
                pytest.raises(ValueError, match="uint32"),
            ),
            # wrong dtype for float64 check: passing uint32 when float64 expected
            (
                jax.random.PRNGKey(0),
                jnp.float64,
                pytest.raises(ValueError, match="float64"),
            ),
            # wrong shape
            (
                jnp.zeros(3, dtype=jnp.uint32),
                jnp.uint32,
                pytest.raises(ValueError, match="shape"),
            ),
            # scalar — wrong shape
            (
                jnp.array(0, dtype=jnp.uint32),
                jnp.uint32,
                pytest.raises(ValueError, match="shape"),
            ),
        ],
    )
    @pytest.mark.requires_x64
    def test_validate_random_key(self, validator, key, dtype, expectation):
        if key is None:
            key = jax.random.PRNGKey(0).astype(jnp.float64)
        with expectation:
            validator.validate_random_key(key, dtype=dtype)


class TestPopulationPPGLMValidator:
    """Test suite for input validation logic in PopulationPPGLMValidator. Only needed to test consistency for y"""

    def test_validate_consistency(self, population_validator, processed_X_y):
        # valid inputs, no error
        X, y = processed_X_y
        n_predictors = int(jnp.unique(X.ids).size)
        # n_neurons is 1
        params = population_validator.to_model_params(
            (jnp.zeros((n_predictors * 5, 1)), jnp.zeros(1))
        )
        population_validator.validate_consistency(params, X=X, y=y)

        # incorrect number of neurons in coef
        params = population_validator.to_model_params(
            (jnp.zeros((n_predictors * 5, 2)), jnp.zeros(2))
        )
        with pytest.raises(ValueError, match="Inconsistent number of neurons"):
            population_validator.validate_consistency(params, X=X, y=y)

        # non consecutive neuron ids in y
        X, _ = processed_X_y
        y_skipped = y_ppglm(
            times=jnp.array([1.0, 2.0, 3.0]),
            ids=jnp.array([0, 2, 2]),  # skips id 1
            idx=jnp.array([0, 1, 2]),
        )
        n_predictors = int(jnp.unique(X.ids).size)
        params = population_validator.to_model_params(
            (jnp.zeros((n_predictors * 5, 2)), jnp.zeros(2))
        )
        with pytest.raises(ValueError, match="Inconsistent neuron IDs"):
            population_validator.validate_consistency(params, X=X, y=y_skipped)

        # should not raise if y is None
        params = population_validator.to_model_params(
            (jnp.zeros((n_predictors * 5, 1)), jnp.zeros(1))
        )
        population_validator.validate_consistency(params, X=X, y=None)
