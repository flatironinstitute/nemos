from contextlib import nullcontext as does_not_raise
from typing import Any, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pynapple as nap
import pytest

from nemos.base_regressor import BaseRegressor
from nemos.glm.params import GLMParams
from nemos.pp_glm.data import PredictorsPPGLM, SpikesPPGLM
from nemos.pp_glm.params import PPGLMParamsWithKey
from nemos.pp_glm.validation import PopulationPPGLMValidator, PPGLMValidator
from nemos.regularizer import Regularizer
from nemos.typing import UserProvidedParamsT


class MockPPGLM(BaseRegressor):
    """
    Minimal PP-GLM stand-in for testing PPGLMValidator logic.

    """

    def __init__(
        self,
        n_basis_funcs: int,
        regularizer: Optional[Union[str, Regularizer]] = None,
        regularizer_strength: Any = None,
        solver_name: str = None,
        solver_kwargs: dict = None,
    ):
        super().__init__(
            regularizer=regularizer,
            regularizer_strength=regularizer_strength,
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
        )

        self.n_basis_funcs = n_basis_funcs

        # validator is a frozen dataclass — constructed explicitly with n_basis_funcs
        self._validator = PPGLMValidator(n_basis_funcs=self.n_basis_funcs)

        self.coef_: Optional[jnp.ndarray] = None
        self.intercept_: Optional[jnp.ndarray] = None

    def _get_model_params(self) -> GLMParams:
        return self._validator.to_model_params(
            (
                self.coef_,
                self.intercept_,
            )
        )

    def _set_model_params(self, params):
        coef, intercept = self._validator.from_model_params(params)
        self.coef_ = coef
        self.intercept_ = intercept

    def _check_model_is_fit(self):
        if self.coef_ is None:
            raise ValueError("Model is not fitted yet.")

    def _model_specific_initialization(
        self,
        X: PredictorsPPGLM,
        y: SpikesPPGLM,
        **kwargs,
    ) -> GLMParams:

        empty_params = self._validator.get_empty_params(X, y)

        n_neurons = jnp.unique(X.ids).size
        n_features = int(n_neurons * self.n_basis_funcs)

        init_params = eqx.tree_at(
            lambda p: (p.coef, p.intercept),
            empty_params,
            (jnp.zeros(n_features), jnp.ones(n_neurons)),
        )

        return init_params

    def initialize_params(
        self,
        X: PredictorsPPGLM,
        y: SpikesPPGLM,
    ) -> UserProvidedParamsT:
        pass

    def fit(self, X: PredictorsPPGLM, y: SpikesPPGLM, init_params=None):
        """Validate inputs and set zero params; no actual optimization."""
        fit_params = self._model_specific_initialization(X, y)
        self._set_model_params(fit_params)

    def predict(self, X):
        self._check_model_is_fit()
        return jnp.array(0.0)

    def score(self, X, y, **kwargs):
        self._check_model_is_fit()
        return jnp.array(0.0)

    def simulate(self, *args, **kwargs):
        pass

    def save_params(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def _initialize_optimizer_and_state(self, *args, **kwargs):
        pass

    def _log_likelihood(self, params, X, y):
        return jnp.zeros(0.0)

    def _compute_loss(
        self,
        params: PPGLMParamsWithKey,
        X: PredictorsPPGLM,
        y: SpikesPPGLM,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        return jnp.array(0.0)

    def _get_optimal_solver_params_config(self, *args, **kwargs):
        pass


class MockPopulationPPGLM(MockPPGLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._validator = PopulationPPGLMValidator(n_basis_funcs=self.n_basis_funcs)


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

    X = PredictorsPPGLM(
        times=event_times,
        predictor_ids=jnp.asarray(X.d, dtype=int),
    )

    y = SpikesPPGLM(
        times=spike_times,
        neuron_ids=jnp.asarray(y.d, dtype=int),
        timestamp_idx=y_idx,
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
        n_features = int(jnp.unique(X.predictor_ids).size * 5)
        params = validator.to_model_params((jnp.zeros(n_features), jnp.zeros(1)))
        validator.validate_consistency(params, X=X)

        # incorrect number of features
        params = validator.to_model_params((jnp.zeros(5), jnp.zeros(1)))
        with pytest.raises(ValueError, match="Inconsistent number of features"):
            validator.validate_consistency(params, X=X)

        # test that no neuron ids are skipped
        X_skipped = PredictorsPPGLM(
            times=jnp.array([1.0, 2.0, 3.0]),
            predictor_ids=jnp.array([0, 2, 2]),  # skips id 1
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

    @pytest.mark.parametrize(
        "time_series, expectation",
        [
            # --- TsGroup ---
            (
                nap.TsGroup(
                    {
                        0: nap.Ts(np.array([1.0, 2.0, 3.0])),
                        1: nap.Ts(np.array([1.5, 2.5])),
                    }
                ),
                does_not_raise(),
            ),
            (
                nap.TsGroup(
                    {
                        0: nap.Ts(np.array([1.0, 2.0])),
                        1: nap.Ts(np.array([])),
                    }
                ),
                pytest.raises(ValueError, match=r"index\(es\) \[1\]"),
            ),
            # --- single Ts ---
            (
                nap.Ts(np.array([1.0, 2.0, 3.0])),
                does_not_raise(),
            ),
            (
                nap.Ts(np.array([])),
                pytest.raises(ValueError, match="is empty"),
            ),
            # --- dict ---
            (
                {0: np.array([1.0, 2.0]), 1: np.array([3.0])},
                does_not_raise(),
            ),
            (
                {0: np.array([1.0, 2.0]), 1: np.array([])},
                pytest.raises(ValueError, match=r"key\(s\) \[1\]"),
            ),
            # --- np array ---
            (
                np.array([1.0, 2.0, 3.0]),
                does_not_raise(),
            ),
            (
                np.array([]),
                pytest.raises(ValueError, match="is empty"),
            ),
            # --- list ---
            (
                [1.0, 2.0, 3.0],
                does_not_raise(),
            ),
            # --- list of arrays ---
            (
                [np.array([1.0, 2.0]), np.array([3.0])],
                does_not_raise(),
            ),
            (
                [np.array([1.0, 2.0]), np.array([])],
                pytest.raises(ValueError, match=r"index\(es\) \[1\]"),
            ),
            # --- jnp array ---
            (
                jnp.array([1.0, 2.0, 3.0]),
                does_not_raise(),
            ),
            (
                jnp.array([]),
                pytest.raises(ValueError, match="is empty"),
            ),
            # --- unsupported type ---
            (
                "not_a_valid_input",
                pytest.raises(TypeError, match="Unsupported type"),
            ),
            (
                2.16,
                pytest.raises(TypeError, match="Unsupported type"),
            ),
        ],
    )
    def test_validate_time_series(self, validator, time_series, expectation):
        with expectation:
            validator._validate_time_series(time_series, name="X")

    def test_validate_inputs_valid(self, validator, valid_X_y):
        """Valid X and y (as TsGroups) should not raise."""
        X, y = valid_X_y
        validator.validate_inputs(X=X, y=y)

    def test_validate_inputs_only_X(self, validator, valid_X_y):
        X, _ = valid_X_y
        validator.validate_inputs(X=X, y=None)

    def test_validate_inputs_only_y(self, validator, valid_X_y):
        _, y = valid_X_y
        validator.validate_inputs(X=None, y=y)

    def test_validate_inputs_invalid_X(self, validator, valid_X_y):
        """An invalid X should raise with 'X' in the message, even if y is valid."""
        _, y = valid_X_y
        with pytest.raises(ValueError, match="X is empty"):
            validator.validate_inputs(X=np.array([]), y=y)

    def test_validate_inputs_invalid_y(self, validator, valid_X_y):
        """An invalid y should raise with 'y' in the message, even if X is valid."""
        X, _ = valid_X_y
        with pytest.raises(ValueError, match="y is empty"):
            validator.validate_inputs(X=X, y=np.array([]))

    @pytest.mark.parametrize(
        "n_series, expectation",
        [
            (1, does_not_raise()),
            (5, pytest.raises(ValueError, match="must contain exactly 1")),
        ],
    )
    def test_validate_y_dimensionality(self, validator, n_series, expectation):
        with expectation:
            validator._validate_y_dimensionality(n_series)

    def test_validate_y_dimensionality_single(self, validator, valid_X_y):
        """PPGLMValidator requires y to have exactly 1 time series."""
        X, _ = valid_X_y

        # valid y
        y_single = nap.Ts(np.sort(np.random.uniform(0, 10, 20)))
        validator.validate_inputs(X=X, y=y_single)

        # invalid y
        y_pop = nap.TsGroup(
            {
                0: nap.Ts(np.sort(np.random.uniform(0, 10, 20))),
                1: nap.Ts(np.sort(np.random.uniform(0, 10, 15))),
            }
        )
        with pytest.raises(ValueError, match="must contain exactly 1"):
            validator.validate_inputs(X=X, y=y_pop)

    def test_get_empty_params(self, validator, processed_X_y):
        X, y = processed_X_y
        n_features = int(jnp.unique(X.predictor_ids).size * validator.n_basis_funcs)

        params = validator.get_empty_params(X, y)

        assert params.coef.shape == (n_features,)
        assert params.intercept.shape == (1,)
        assert isinstance(params, GLMParams)


class TestPopulationPPGLMValidator:
    """Test suite for input validation logic in PopulationPPGLMValidator. Only needed to test consistency for y"""

    def test_validate_consistency(self, population_validator, processed_X_y):
        # valid inputs, no error
        X, y = processed_X_y
        n_predictors = int(jnp.unique(X.predictor_ids).size)
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
        y_skipped = SpikesPPGLM(
            times=jnp.array([1.0, 2.0, 3.0]),
            neuron_ids=jnp.array([0, 2, 2]),  # skips id 1
            timestamp_idx=jnp.array([0, 1, 2]),
        )
        n_predictors = int(jnp.unique(X.predictor_ids).size)
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

    @pytest.mark.parametrize(
        "n_series, expectation",
        [
            (5, does_not_raise()),
            (1, pytest.raises(ValueError, match="must contain more than 1")),
        ],
    )
    def test_validate_y_dimensionality(
        self, population_validator, n_series, expectation
    ):
        with expectation:
            population_validator._validate_y_dimensionality(n_series)

    def test_validate_y_dimensionality_population(
        self, population_validator, valid_X_y
    ):
        """PopulationPPGLMValidator requires y to have more than 1 time series."""
        X, _ = valid_X_y

        # valid y
        y_pop = nap.TsGroup(
            {
                0: nap.Ts(np.sort(np.random.uniform(0, 10, 20))),
                1: nap.Ts(np.sort(np.random.uniform(0, 10, 15))),
            }
        )
        population_validator.validate_inputs(X=X, y=y_pop)

        # invalid y
        y_single = nap.Ts(np.sort(np.random.uniform(0, 10, 20)))
        with pytest.raises(ValueError, match="must contain more than 1"):
            population_validator.validate_inputs(X=X, y=y_single)

    def test_get_empty_params(self, validator, processed_X_y):
        X, y = processed_X_y
        n_features = int(jnp.unique(X.predictor_ids).size * validator.n_basis_funcs)
        n_neurons = jnp.unique(y.neuron_ids).size

        params = validator.get_empty_params(X, y)

        assert params.coef.shape == (n_features,)
        assert params.intercept.shape == (n_neurons,)
        assert isinstance(params, GLMParams)
