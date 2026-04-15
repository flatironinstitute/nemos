from nemos.hmm.hmm import BaseHMM
from typing import Union, Optional, Callable, Tuple
import jax
import jax.numpy as jnp
from nemos.hmm.initialize_parameters import (
    INITIALIZATION_FN_DICT,
    DEFAULT_INIT_FUNCTIONS,
)
from nemos.params import ModelParams
from nemos.hmm.params import HMMParams
from nemos.base_regressor import BaseRegressor
from nemos.hmm.validation import HMMValidator
from nemos.base_validator import RegressorValidator
from nemos.hmm.validation import to_hmm_params, from_hmm_params
import nemos as nmo
import pytest
from contextlib import nullcontext as does_not_raise
from typing import Any, Callable, Tuple, Union
import numpy as np


class MockHMMModelParams(ModelParams):
    param: jnp.ndarray


class MockHMMParams(ModelParams):
    model_params: MockHMMModelParams
    hmm_params: HMMParams


MockHMMUserParams = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]


def to_mock_params(user_params: MockHMMUserParams) -> MockHMMParams:
    return MockHMMParams(
        model_params=MockHMMModelParams(user_params[0]),
        hmm_params=to_hmm_params(user_params[1:]),
    )


def from_mock_params(params: MockHMMParams) -> MockHMMUserParams:
    initial_prob, transition_prob = from_hmm_params(params.hmm_params)
    return (
        params.model_params.param,
        initial_prob,
        transition_prob,
    )


class MockHMMValidator(HMMValidator[MockHMMUserParams, MockHMMParams]):
    expected_param_dims: Tuple[int] = (
        1,
        *HMMValidator.expected_param_dims,
    )  # (coef.ndim, intercept.ndim, scale.ndim, init_prob.ndim, transition_prob.ndim)
    initial_prob_ind: int = 2
    transition_prob_ind: int = 3
    model_param_names: Tuple[str] = (
        "param",
        *HMMValidator.model_param_names,
    )
    to_model_params: Callable[[MockHMMUserParams], MockHMMParams] = to_mock_params
    from_model_params: Callable[[MockHMMParams], MockHMMUserParams] = from_mock_params
    model_class: str = "MockHMM"
    X_dimensionality: int = 2
    y_dimensionality: int = 1
    params_validation_sequence: Tuple[Tuple[str, None] | Tuple[str, dict[str, Any]]] = (
        *RegressorValidator.params_validation_sequence[:2],
        (
            "check_array_dimensions",
            dict(
                err_message_format="Invalid parameter dimensionality.\n- param must be of shape "
                "``(n_states,)``.\n- intercept must be of shape ``(n_states,)``.\n"
                "- initial_prob must be of shape ``(n_states,)``.\n"
                "- transition_prob must be of shape ``(n_states, n_states)``.\n"
                "\nThe provided param, initial_prob and transition_prob "
                "have shape ``{}``, ``{}`` and ``{}`` "
                "instead."
            ),
        ),
        *HMMValidator.params_validation_sequence,
        *RegressorValidator.params_validation_sequence[3:],
    )

    def validate_consistency(self, params: MockHMMParams) -> None:
        pass


class MockHMM(BaseHMM[MockHMMParams, MockHMMUserParams]):
    _validator_class = MockHMMValidator

    def __init__(
        self,
        n_states: int,
        dirichlet_prior_alphas_init_prob: Union[
            jnp.ndarray, None
        ] = None,  # (n_state, )
        dirichlet_prior_alphas_transition: Union[
            jnp.ndarray | None
        ] = None,  # (n_state, n_state):
        maxiter: int = 1000,
        tol: float = 1e-8,
        seed=jax.random.PRNGKey(123),
        hmm_initialization_funcs: INITIALIZATION_FN_DICT = {},
    ):
        BaseHMM.__init__(
            self,
            n_states=n_states,
            dirichlet_prior_alphas_init_prob=dirichlet_prior_alphas_init_prob,
            dirichlet_prior_alphas_transition=dirichlet_prior_alphas_transition,
            maxiter=maxiter,
            tol=tol,
            seed=seed,
            hmm_initialization_funcs=hmm_initialization_funcs,
        )
        self.param_: jnp.ndarray | None = None

    def setup(
        self,
        initial_proba_init: Optional[str | Callable] = None,
        initial_proba_init_kwargs: Optional[dict] = None,
        transition_proba_init: Optional[str | Callable] = None,
        transition_proba_init_kwargs: Optional[dict] = None,
        param_init: Optional[str | Callable] = None,
        param_init_kwargs: Optional[dict] = None,
    ):
        BaseHMM.setup(
            self,
            initial_proba_init=initial_proba_init,
            initial_proba_init_kwargs=initial_proba_init_kwargs,
            transition_proba_init=transition_proba_init,
            transition_proba_init_kwargs=transition_proba_init_kwargs,
        )

    def _check_is_fit(self):
        BaseHMM._check_is_fit(self)
        if self.param_ is None:
            raise ValueError("Model is not fitted yet.")

    def _get_model_params(self) -> MockHMMParams:
        return self._validator.to_model_params(
            self.param_,
            self.log_initial_prob_,
            self.log_transition_prob_,
        )

    def _set_model_params(self, params):
        param, initial_prob, transition_prob = self._validator.from_model_params(params)
        self.param_ = param
        self.initial_prob_ = initial_prob
        self.transition_prob_ = transition_prob

    def _log_likelihood(self, params, X, y):
        pass

    def _initialize_optimizer_and_state(self, *args, **kwargs):
        pass

    def _compute_loss(self, *args, **kwargs):
        pass

    def _get_optimal_solver_params_config(self, *args, **kwargs):
        pass

    def _model_specific_initialization(self, X, y):
        pass

    def fit(self, X, y, is_new_session=None, init_params=None):
        pass

    def predict(self, *args, **kwargs):
        pass

    def simulate(self, *args, **kwargs):
        pass

    def save_params(self, filename):
        pass

    def update(self):
        pass


class TestHMMInit:

    # -------------------------------------------------------------------------
    # n_states setter tests
    # -------------------------------------------------------------------------
    def test_n_states_creates_validator(self):
        """Test that setting n_states creates a GLMHMMValidator."""
        model = MockHMM(n_states=3)
        assert hasattr(model, "_validator")
        assert model._validator.n_states == 3

    @pytest.mark.parametrize(
        "n_states, expectation",
        [
            (1, does_not_raise()),
            (2, does_not_raise()),
            (10, does_not_raise()),
            (3.0, does_not_raise()),  # float with no decimals is allowed
            (0, pytest.raises(ValueError, match="must be a positive integer")),
            (-1, pytest.raises(ValueError, match="must be a positive integer")),
            (2.5, pytest.raises(TypeError, match="must be a positive integer")),
            ("3", pytest.raises(TypeError, match="must be a positive integer")),
            (None, pytest.raises(TypeError, match="must be a positive integer")),
            ([3], pytest.raises(TypeError, match="must be a positive integer")),
        ],
    )
    def test_n_states_setter(self, n_states, expectation):
        """Test n_states validation accepts positive integers only."""
        with expectation:
            model = MockHMM(n_states=n_states)
            assert model.n_states == int(n_states)

    # -------------------------------------------------------------------------
    # maxiter setter tests
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "maxiter, expectation",
        [
            (1, does_not_raise()),
            (100, does_not_raise()),
            (1000, does_not_raise()),
            (10.0, does_not_raise()),  # float with no decimals is allowed
            (0, pytest.raises(ValueError, match="must be a strictly positive integer")),
            (
                -1,
                pytest.raises(ValueError, match="must be a strictly positive integer"),
            ),
            (
                10.5,
                pytest.raises(ValueError, match="must be a strictly positive integer"),
            ),
            (
                "100",
                pytest.raises(ValueError, match="must be a strictly positive integer"),
            ),
            (
                None,
                pytest.raises(ValueError, match="must be a strictly positive integer"),
            ),
        ],
    )
    def test_maxiter_setter(self, maxiter, expectation):
        """Test maxiter validation accepts positive integers only."""
        with expectation:
            model = MockHMM(n_states=2, maxiter=maxiter)
            assert model.maxiter == int(maxiter)

    # -------------------------------------------------------------------------
    # tol setter tests
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "tol, expectation",
        [
            (1e-8, does_not_raise()),
            (0.001, does_not_raise()),
            (1.0, does_not_raise()),
            (1, does_not_raise()),  # int is allowed (converted to float)
            (0, pytest.raises(ValueError, match="must be a strictly positive float")),
            (
                -1e-8,
                pytest.raises(ValueError, match="must be a strictly positive float"),
            ),
            (
                "0.001",
                pytest.raises(ValueError, match="must be a strictly positive float"),
            ),
            (
                None,
                pytest.raises(ValueError, match="must be a strictly positive float"),
            ),
        ],
    )
    def test_tol_setter(self, tol, expectation):
        """Test tol validation accepts positive numbers only."""
        with expectation:
            model = MockHMM(n_states=2, tol=tol)
            assert model.tol == float(tol)

    # -------------------------------------------------------------------------
    # seed setter tests
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "seed, expectation",
        [
            (jax.random.PRNGKey(0), does_not_raise()),
            (jax.random.PRNGKey(123), does_not_raise()),
            (jax.random.PRNGKey(999999), does_not_raise()),
        ],
    )
    def test_seed_setter_valid(self, seed, expectation):
        """Test seed validation accepts valid JAX PRNG keys."""
        with expectation:
            model = MockHMM(n_states=2, seed=seed)
            assert jnp.array_equal(model.seed, seed)

    @pytest.mark.parametrize(
        "seed",
        [
            123,  # plain int
            np.array([1, 2, 3]),  # wrong shape
            jnp.array([1.0, 2.0]),  # wrong dtype
            "seed",  # string
            None,  # None
        ],
    )
    def test_seed_setter_invalid(self, seed):
        """Test seed validation rejects invalid inputs."""
        with pytest.raises(TypeError, match="seed must be a JAX PRNG key"):
            MockHMM(n_states=2, seed=seed)

    # -------------------------------------------------------------------------
    # dirichlet_prior_alphas_init_prob setter tests
    # -------------------------------------------------------------------------
    def test_dirichlet_prior_init_prob_none(self):
        """Test that None is accepted for dirichlet prior."""
        model = MockHMM(n_states=3, dirichlet_prior_alphas_init_prob=None)
        assert model.dirichlet_prior_alphas_init_prob is None

    def test_dirichlet_prior_init_prob_valid(self):
        """Test valid dirichlet prior alphas."""
        alphas = jnp.array([1.0, 2.0, 3.0])
        model = MockHMM(n_states=3, dirichlet_prior_alphas_init_prob=alphas)
        assert jnp.array_equal(model.dirichlet_prior_alphas_init_prob, alphas)

    def test_dirichlet_prior_init_prob_wrong_shape(self):
        """Test that wrong shape raises ValueError."""
        alphas = jnp.array([1.0, 2.0])  # n_states=3 but only 2 elements
        with pytest.raises(ValueError, match="must have shape"):
            MockHMM(n_states=3, dirichlet_prior_alphas_init_prob=alphas)

    def test_dirichlet_prior_init_prob_values_less_than_one(self):
        """Test that alpha values < 1 raise ValueError."""
        alphas = jnp.array([1.0, 0.5, 2.0])
        with pytest.raises(ValueError, match="must be >= 1"):
            MockHMM(n_states=3, dirichlet_prior_alphas_init_prob=alphas)

    # -------------------------------------------------------------------------
    # dirichlet_prior_alphas_transition setter tests
    # -------------------------------------------------------------------------
    def test_dirichlet_prior_transition_none(self):
        """Test that None is accepted for dirichlet prior."""
        model = MockHMM(n_states=3, dirichlet_prior_alphas_transition=None)
        assert model.dirichlet_prior_alphas_transition is None

    def test_dirichlet_prior_transition_valid(self):
        """Test valid dirichlet prior alphas for transitions."""
        alphas = jnp.ones((3, 3))
        model = MockHMM(n_states=3, dirichlet_prior_alphas_transition=alphas)
        assert jnp.array_equal(model.dirichlet_prior_alphas_transition, alphas)

    def test_dirichlet_prior_transition_wrong_shape(self):
        """Test that wrong shape raises ValueError."""
        alphas = jnp.ones((2, 3))  # n_states=3 but wrong shape
        with pytest.raises(ValueError, match="must have shape"):
            MockHMM(n_states=3, dirichlet_prior_alphas_transition=alphas)

    # -------------------------------------------------------------------------
    # hmm_initialization_funcs setter tests
    # -------------------------------------------------------------------------
    def test_initialization_funcs_none_uses_defaults(self):
        """Test that no setting uses default initialization functions."""
        model = MockHMM(n_states=2)
        assert model.hmm_initialization_funcs == DEFAULT_INIT_FUNCTIONS

    def test_initialization_funcs_modified(self):
        """Test that modified initialization funcs are preserved."""

        def custom_func(n_states, X, y, random_key, extra_arg):
            return jnp.full((n_states,), 1.0) / n_states

        init_funcs = {
            "initial_proba_init": custom_func,
            "initial_proba_init_kwargs": {"extra_arg": "value"},
        }
        model = MockHMM(n_states=2, hmm_initialization_funcs=init_funcs)
        assert model.hmm_initialization_funcs == DEFAULT_INIT_FUNCTIONS | init_funcs

    def test_initialization_funcs_invalid_key(self):
        """Test that invalid registry keys raise KeyError."""
        with pytest.raises(KeyError, match="unknown key"):
            MockHMM(n_states=2, hmm_initialization_funcs={"invalid_key": lambda: None})

    # -------------------------------------------------------------------------
    # Default values tests
    # -------------------------------------------------------------------------
    def test_default_values(self):
        """Test that default values are set correctly."""
        model = MockHMM(n_states=3)

        assert model.n_states == 3
        assert model.maxiter == 1000
        assert model.tol == 1e-8
        assert model.dirichlet_prior_alphas_init_prob is None
        assert model.dirichlet_prior_alphas_transition is None

    def test_fit_attributes_initialized_to_none(self):
        """Test that fit attributes are initialized to None."""
        model = MockHMM(n_states=3)

        assert model.initial_prob_ is None
        assert model.transition_prob_ is None


class TestHMMSetup:
    def test_setup_initializes_hmm_parameters(self):
        """Test that setup initializes HMM parameters."""
        model = MockHMM(n_states=3)
        model.setup()
        assert model.initial_prob_ is not None
        assert model.transition_prob_ is not None
