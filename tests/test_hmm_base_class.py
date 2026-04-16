from nemos.hmm.hmm import BaseHMM
from typing import Union, Optional, Callable, Tuple
import jax
import jax.numpy as jnp
from nemos.hmm.initialize_parameters import (
    INITIALIZATION_FN_DICT,
    DEFAULT_INIT_FUNCTIONS,
    uniform_initial_proba_init,
    random_initial_proba_init,
    kmeans_initial_proba_init,
    uniform_transition_proba_init,
    random_transition_proba_init,
    sticky_transition_proba_init,
    kmeans_transition_proba_init,
)
from nemos.params import ModelParams
from nemos.hmm.params import HMMParams
from nemos.hmm.validation import HMMValidator
from nemos.base_validator import RegressorValidator
from nemos.hmm.validation import to_hmm_params, from_hmm_params
import nemos as nmo
import pytest
from contextlib import nullcontext as does_not_raise
from typing import Any, Callable, Tuple, Union
import numpy as np
import pynapple as nap
from nemos import tree_utils
from dataclasses import dataclass


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


@dataclass(frozen=True, repr=False)
class MockHMMValidator(HMMValidator[MockHMMUserParams, MockHMMParams]):
    model_param_names: Tuple[str] = (
        "param",
        *HMMValidator.model_param_names,
    )
    to_model_params: Callable[[MockHMMUserParams], MockHMMParams] = to_mock_params
    from_model_params: Callable[[MockHMMParams], MockHMMUserParams] = from_mock_params
    model_class: str = "MockHMM"
    params_validation_sequence: Tuple[Tuple[str, None] | Tuple[str, dict[str, Any]]] = (
        *RegressorValidator.params_validation_sequence[:2],
        *HMMValidator.params_validation_sequence,
        *RegressorValidator.params_validation_sequence[3:],
    )

    def validate_consistency(self, params: MockHMMParams) -> None:
        return True


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
        model_initialization_funcs: INITIALIZATION_FN_DICT = {},
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
        self.model_initialization_funcs = model_initialization_funcs

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

    def _check_model_is_fit(self):
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

    def _model_params_initialization(self, X, y, is_new_session):
        return (
            jnp.zeros(self._n_states),
            False,
        )

    def fit(self, X, y, is_new_session=None, init_params=None):
        pass

    def _initialize_optimizer_and_state(self, *args, **kwargs):
        pass

    def _compute_loss(self, *args, **kwargs):
        pass

    def _get_optimal_solver_params_config(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass

    def simulate(self, *args, **kwargs):
        pass

    def save_params(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
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

    def test_setup_with_no_input(self):
        """Test that setup leaves everything as default."""
        model = MockHMM(n_states=3)
        assert model.hmm_initialization_funcs == DEFAULT_INIT_FUNCTIONS
        model.setup()
        assert model.hmm_initialization_funcs == DEFAULT_INIT_FUNCTIONS

    @pytest.mark.parametrize(
        "func_name, func, kwargs, expectation",
        [
            ("uniform", uniform_initial_proba_init, None, does_not_raise()),
            ("random", random_initial_proba_init, None, does_not_raise()),
            (
                "kmeans",
                kmeans_initial_proba_init,
                {"is_new_session": jnp.zeros(10)},
                does_not_raise(),
            ),
            (
                "random",
                None,
                {"extra_arg": "value"},
                pytest.raises(ValueError, match="Invalid keyword argument"),
            ),
            (
                "invalid",
                None,
                None,
                pytest.raises(ValueError, match="Invalid initialization"),
            ),
            (
                "custom",
                lambda n_states, X, y, random_key, extra_arg: jnp.full((n_states,), 1.0)
                / n_states,
                {"extra_arg": "value"},
                does_not_raise(),
            ),
            (
                "custom",
                lambda n_states, X, y, random_key: jnp.full((n_states,), 1.0)
                / n_states,
                {"extra_arg": "value"},
                pytest.raises(ValueError, match="Invalid keyword argument"),
            ),
        ],
    )
    def test_setup_initial_proba_init(self, func_name, func, kwargs, expectation):
        """Test setup arguments for initial_proba_init and kwargs."""
        model = MockHMM(n_states=3)
        with expectation:
            if func_name == "custom":
                model.setup(initial_proba_init=func, initial_proba_init_kwargs=kwargs)
                assert (
                    model.hmm_initialization_funcs["initial_proba_init_custom"] is True
                )
            else:
                model.setup(
                    initial_proba_init=func_name, initial_proba_init_kwargs=kwargs
                )
                assert (
                    model.hmm_initialization_funcs["initial_proba_init_custom"] is False
                )

            assert model.hmm_initialization_funcs["initial_proba_init"] == func

            if kwargs is None:
                assert model.hmm_initialization_funcs["initial_proba_init_kwargs"] == {}
            else:
                assert (
                    model.hmm_initialization_funcs["initial_proba_init_kwargs"]
                    == kwargs
                )

    @pytest.mark.parametrize(
        "func_name, func, kwargs, expectation",
        [
            ("sticky", sticky_transition_proba_init, None, does_not_raise()),
            ("uniform", uniform_transition_proba_init, None, does_not_raise()),
            ("random", random_transition_proba_init, None, does_not_raise()),
            (
                "kmeans",
                kmeans_transition_proba_init,
                {"is_new_session": jnp.zeros(10)},
                does_not_raise(),
            ),
            (
                "random",
                None,
                {"extra_arg": "value"},
                pytest.raises(ValueError, match="Invalid keyword argument"),
            ),
            (
                "invalid",
                None,
                None,
                pytest.raises(ValueError, match="Invalid initialization"),
            ),
            (
                "custom",
                lambda n_states, X, y, random_key, extra_arg: jnp.full(
                    (n_states, n_states), 1.0
                )
                / n_states,
                {"extra_arg": "value"},
                does_not_raise(),
            ),
            (
                "custom",
                lambda n_states, X, y, random_key: jnp.full((n_states, n_states), 1.0)
                / n_states,
                {"extra_arg": "value"},
                pytest.raises(ValueError, match="Invalid keyword argument"),
            ),
        ],
    )
    def test_setup_transition_proba_init(self, func_name, func, kwargs, expectation):
        """Test setup arguments for transition_proba_init and kwargs."""
        model = MockHMM(n_states=3)
        with expectation:
            if func_name == "custom":
                model.setup(
                    transition_proba_init=func, transition_proba_init_kwargs=kwargs
                )
                assert (
                    model.hmm_initialization_funcs["transition_proba_init_custom"]
                    is True
                )
            else:
                model.setup(
                    transition_proba_init=func_name, transition_proba_init_kwargs=kwargs
                )
                assert (
                    model.hmm_initialization_funcs["transition_proba_init_custom"]
                    is False
                )

            assert model.hmm_initialization_funcs["transition_proba_init"] == func

            if kwargs is None:
                assert (
                    model.hmm_initialization_funcs["transition_proba_init_kwargs"] == {}
                )
            else:
                assert (
                    model.hmm_initialization_funcs["transition_proba_init_kwargs"]
                    == kwargs
                )

    def test_setup_set_all(self):
        init_funcs = {
            "initial_proba_init": kmeans_initial_proba_init,
            "initial_proba_init_kwargs": {"is_new_session": jnp.zeros(10)},
            "transition_proba_init": kmeans_transition_proba_init,
            "transition_proba_init_kwargs": {"is_new_session": jnp.zeros(10)},
        }

        model = MockHMM(n_states=3)
        model.setup(
            initial_proba_init="kmeans",
            initial_proba_init_kwargs={"is_new_session": jnp.zeros(10)},
            transition_proba_init="kmeans",
            transition_proba_init_kwargs={"is_new_session": jnp.zeros(10)},
        )
        expected_funcs = DEFAULT_INIT_FUNCTIONS | init_funcs
        assert (
            model.hmm_initialization_funcs[key] == expected_funcs[key]
            for key in expected_funcs
        )

    def test_setup_consecutive_calls(self):
        init_funcs = {
            "initial_proba_init": kmeans_initial_proba_init,
            "initial_proba_init_kwargs": {"is_new_session": jnp.zeros(10)},
            "transition_proba_init": kmeans_transition_proba_init,
            "transition_proba_init_kwargs": {"is_new_session": jnp.zeros(10)},
        }

        model = MockHMM(n_states=3)
        model.setup(initial_proba_init="kmeans")
        # updated
        assert (
            model.hmm_initialization_funcs["initial_proba_init"]
            == init_funcs["initial_proba_init"]
        )
        # default
        assert (
            model.hmm_initialization_funcs[key] == DEFAULT_INIT_FUNCTIONS[key]
            for key in [
                "initial_proba_init_kwargs",
                "transition_proba_init",
                "transition_proba_init_kwargs",
            ]
        )

        model.setup(initial_proba_init_kwargs={"is_new_session": jnp.ones(10)})
        # updated
        assert (
            model.hmm_initialization_funcs[key] == init_funcs[key]
            for key in [
                "initial_proba_init",
                "initial_proba_init_kwargs",
            ]
        )
        # default
        assert (
            model.hmm_initialization_funcs[key] == DEFAULT_INIT_FUNCTIONS[key]
            for key in [
                "transition_proba_init",
                "transition_proba_init_kwargs",
            ]
        )

        model.setup(transition_proba_init="kmeans")
        # updated
        assert (
            model.hmm_initialization_funcs[key] == init_funcs[key]
            for key in [
                "initial_proba_init",
                "initial_proba_init_kwargs",
                "transition_proba_init",
            ]
        )
        # default
        assert (
            model.hmm_initialization_funcs["transition_proba_init_kwargs"]
            == DEFAULT_INIT_FUNCTIONS["transition_proba_init_kwargs"]
        )

        model.setup(transition_proba_init_kwargs={"is_new_session": jnp.zeros(10)})
        # updated
        assert (
            model.hmm_initialization_funcs[key] == init_funcs[key]
            for key in [
                "initial_proba_init",
                "initial_proba_init_kwargs",
                "transition_proba_init",
                "transition_proba_init_kwargs",
            ]
        )

    @pytest.mark.parametrize("key", ["initial_proba_init", "transition_proba_init"])
    def test_setup_reset_kwargs(self, key):
        """Test that kwargs are reset if method is set to something else."""
        model = MockHMM(n_states=3)
        model.setup(**{key: "kmeans", key + "_kwargs": {"is_new_session": 1.0}})
        assert model.hmm_initialization_funcs[key + "_kwargs"] == {
            "is_new_session": 1.0
        }
        model.setup(**{key: "random"})
        assert model.hmm_initialization_funcs[key + "_kwargs"] == {}


class TestHMMInitialParams:

    def test__hmm_params_initialization_defaults(self):
        """Test that _hmm_params_initialization returns expected default parameters and validation flag."""
        model = MockHMM(n_states=3)

        (initial_prob, transition_prob), validate_params = (
            model._hmm_params_initialization(None, None, None)
        )

        assert jnp.allclose(
            initial_prob, DEFAULT_INIT_FUNCTIONS["initial_proba_init"](3)
        )
        assert jnp.allclose(
            transition_prob, DEFAULT_INIT_FUNCTIONS["transition_proba_init"](3)
        )
        assert validate_params is False

    def test__hmm_params_initialization_custom_validation(self):
        model = MockHMM(n_states=3)
        model.setup(
            initial_proba_init=lambda n_states, X, y, random_key: jnp.full(
                (n_states,), 1.0
            )
            / n_states
        )
        (initial_prob, _), validate_params = model._hmm_params_initialization(
            None, None, None
        )
        assert jnp.allclose(initial_prob, jnp.full((3,), 1.0) / 3)
        assert validate_params is True

        model = MockHMM(n_states=3)
        model.setup(
            transition_proba_init=lambda n_states, X, y, random_key: jnp.full(
                (n_states, n_states), 1.0
            )
            / n_states
        )
        (_, transition_prob), validate_params = model._hmm_params_initialization(
            None, None, None
        )
        assert jnp.allclose(transition_prob, jnp.full((3, 3), 1.0) / 3)
        assert validate_params is True

    def test__model_specific_initialization_defaults(self):
        model = MockHMM(n_states=3)
        model_params = model._model_specific_initialization(None, None, None)
        assert jnp.allclose(
            model_params.hmm_params.log_initial_prob,
            jnp.log(DEFAULT_INIT_FUNCTIONS["initial_proba_init"](3)),
        )
        assert jnp.allclose(
            model_params.hmm_params.log_transition_prob,
            jnp.log(DEFAULT_INIT_FUNCTIONS["transition_proba_init"](3)),
        )


class TestHMMNewSession:

    @pytest.mark.parametrize(
        "X, y, is_new_session, expected_new_session",
        [
            ### No is_new_session provided
            (np.ones((3, 1)), np.ones((3,)), None, jnp.array([1, 0, 0])),
            # NaN at start in y
            (np.ones((3, 1)), np.array([np.nan, 0, 0]), None, jnp.array([0, 1, 0])),
            # X and y both have NaNs at different positions
            (
                np.array([[np.nan], [2], [3], [4]]),
                np.array([0, np.nan, 3, 4]),
                None,
                jnp.array([0, 0, 1, 0]),
            ),
            # X and y both have NaNs at same position
            (
                np.array([[np.nan], [1], [2], [3]]),
                np.array([np.nan, 1, 2, 3]),
                None,
                jnp.array([0, 1, 0, 0]),
            ),
            # NaN at the very end of data
            (
                np.ones((4, 1)),
                np.array([0, 1, 2, np.nan]),
                None,
                jnp.array([1, 0, 0, 0]),
            ),
            # Multiple NaNs at the start
            (
                np.ones((5, 1)),
                np.array([np.nan, np.nan, 1, 2, 3]),
                None,
                jnp.array([0, 0, 1, 0, 0]),
            ),
            # Multiple NaNs spaced
            (
                np.ones((5, 1)),
                np.array([np.nan, 1, np.nan, 2, 3]),
                None,
                jnp.array([0, 1, 0, 0, 0]),
            ),
            ### Explicit is_new_session provided by user
            ## boolean array or integer array with 1s and 0s
            (
                np.ones((3, 1)),
                np.ones((3,)),
                jnp.array([1, 0, 0]),
                jnp.array([1, 0, 0]),
            ),
            (
                np.ones((3, 1)),
                np.ones((3,)),
                jnp.array([True, False, False]),
                jnp.array([1, 0, 0]),
            ),
            (
                np.ones((3, 1)),
                np.ones((3,)),
                jnp.array([0, 1, 0]),
                jnp.array([1, 1, 0]),
            ),
            # new session added at beginning
            (
                np.ones((3, 1)),
                np.ones((3,)),
                jnp.array([0, 1, 0], dtype=bool),
                jnp.array([1, 1, 0]),
            ),
            (
                np.ones((3, 1)),
                np.array([np.nan, 0, 0]),
                jnp.array([1, 0, 1]),
                jnp.array([0, 1, 1]),
            ),
            # beginning new session dropped
            (
                np.array([[np.nan], [2], [3], [4]]),
                np.array([0, np.nan, 3, 4]),
                jnp.array([1, 0, 1, 0]),
                jnp.array([0, 0, 1, 0]),
            ),
            # middle new session moved
            (
                np.ones((5, 1)),
                np.array([0, 1, np.nan, np.nan, 3]),
                jnp.array([1, 0, 1, 0, 0]),
                jnp.array([1, 0, 0, 0, 1]),
            ),
            ## integer array with indices of new sessions
            (
                np.ones((5, 1)),
                np.array([0, 1, 2, 3, 4]),
                jnp.array([0]),
                jnp.array([1, 0, 0, 0, 0]),
            ),
            (
                np.ones((5, 1)),
                np.array([0, 1, 2, 3, 4]),
                jnp.array([0, 1]),
                jnp.array([1, 1, 0, 0, 0]),
            ),
            (
                np.ones((5, 1)),
                np.array([0, 1, 2, 3, 4]),
                jnp.array([2, 4]),
                jnp.array([1, 0, 1, 0, 1]),
            ),
            # nan moving is independent of input type so I won't add more cases
        ],
    )
    def test_initialize_new_session(self, X, y, is_new_session, expected_new_session):
        """Test that session boundaries are correctly initialized and moved when there are NaN values."""
        model = MockHMM(n_states=3)
        is_new_session = model._validator.validate_and_cast_is_new_session(
            X, y, is_new_session
        )
        assert jnp.all(is_new_session == expected_new_session)

    @pytest.mark.parametrize(
        "X, y, is_new_session, expected_new_session",
        [
            ## no is_new_session provided
            # inferred time support should only find one new session at start
            (
                nap.TsdFrame(
                    t=np.arange(3),
                    d=np.zeros((3, 3)),
                ),
                nap.Tsd(
                    t=np.arange(3),
                    d=np.zeros((3,)),
                ),
                None,
                jnp.array([1, 0, 0]),
            ),
            # inferred time support finds 2 new sessions
            (
                nap.TsdFrame(
                    t=np.arange(3),
                    d=np.zeros((3, 3)),
                    time_support=nap.IntervalSet([0, 1.5], [1.0, 2.0]),
                ),
                nap.Tsd(
                    t=np.arange(3),
                    d=np.zeros((3,)),
                    time_support=nap.IntervalSet([0, 1.5], [1.0, 2.0]),
                ),
                None,
                jnp.array([1, 0, 1]),
            ),
            # two new sessions where second is moved from nans
            (
                nap.TsdFrame(
                    t=np.arange(5),
                    d=np.zeros((5, 3)),
                    time_support=nap.IntervalSet([0, 1.5], [1.0, 5.0]),
                ),
                nap.Tsd(
                    t=np.arange(5),
                    d=np.array([0, 0, np.nan, np.nan, 0]),
                    time_support=nap.IntervalSet([0, 1.5], [1.0, 5.0]),
                ),
                None,
                jnp.array([1, 0, 0, 0, 1]),
            ),
            # time support prioritized from y
            (
                nap.TsdFrame(
                    t=np.arange(6),
                    d=np.zeros((6, 3)),
                    time_support=nap.IntervalSet([0, 1.5], [1.0, 5.0]),
                ),
                nap.Tsd(
                    t=np.arange(6),
                    d=np.array([0, 0, np.nan, np.nan, 0, 0]),
                ),
                None,
                jnp.array([1, 0, 0, 0, 0, 0]),
            ),
            # time support taken from x
            (
                nap.TsdFrame(
                    t=np.arange(6),
                    d=np.zeros((6, 3)),
                    time_support=nap.IntervalSet([0, 1.5], [1.0, 5.0]),
                ),
                np.array([0, 0, np.nan, np.nan, 0, 0]),
                None,
                jnp.array([1, 0, 0, 0, 1, 0]),
            ),
            # x not required for pynnaple support
            (
                np.array([[np.nan], [0], [np.nan], [0], [0]]),
                nap.Tsd(
                    t=np.arange(5),
                    d=np.zeros(5),
                    time_support=nap.IntervalSet([0, 1.5], [1.0, 5.0]),
                ),
                None,
                jnp.array([0, 1, 0, 1, 0]),
            ),
            # Entire epoch is NaN (with pynapple)
            (
                nap.TsdFrame(
                    t=np.arange(6),
                    d=np.zeros((6, 1)),
                    time_support=nap.IntervalSet([0, 2, 4], [1, 3, 5]),
                ),
                nap.Tsd(
                    t=np.arange(6),
                    d=np.array([0, 0, np.nan, np.nan, 3, 4]),
                    time_support=nap.IntervalSet([0, 2, 4], [1, 3, 5]),
                ),
                None,
                jnp.array([1, 0, 0, 0, 1, 0]),
            ),
            ## intervalset passed as is_new_session
            (
                nap.TsdFrame(
                    t=np.arange(3),
                    d=np.zeros((3, 3)),
                ),
                nap.Tsd(
                    t=np.arange(3),
                    d=np.zeros((3,)),
                ),
                nap.IntervalSet([0, 3]),
                jnp.array([1, 0, 0]),
            ),
            # forces first time point to be new session
            (
                nap.TsdFrame(
                    t=np.arange(3),
                    d=np.zeros((3, 3)),
                ),
                nap.Tsd(
                    t=np.arange(3),
                    d=np.zeros((3,)),
                ),
                nap.IntervalSet([2, 3]),
                jnp.array([1, 0, 1]),
            ),
            # time support finds 2 new sessions
            (
                nap.TsdFrame(
                    t=np.arange(3),
                    d=np.zeros((3, 3)),
                ),
                nap.Tsd(
                    t=np.arange(3),
                    d=np.zeros((3,)),
                ),
                nap.IntervalSet([0, 1.5], [1.0, 2.0]),
                jnp.array([1, 0, 1]),
            ),
            # ignore time support when intervalset is provided
            (
                nap.TsdFrame(
                    t=np.arange(3),
                    d=np.zeros((3, 3)),
                    time_support=nap.IntervalSet([0, 1.5], [1.0, 2.0]),
                ),
                nap.Tsd(
                    t=np.arange(3),
                    d=np.zeros((3,)),
                    time_support=nap.IntervalSet([0, 1.5], [1.0, 2.0]),
                ),
                nap.IntervalSet([0, 3]),
                jnp.array([1, 0, 0]),
            ),
            # ignore time support when intervalset is provided
            (
                nap.TsdFrame(
                    t=np.arange(3),
                    d=np.zeros((3, 3)),
                    time_support=nap.IntervalSet([0, 1.5], [1.0, 2.0]),
                ),
                nap.Tsd(
                    t=np.arange(3),
                    d=np.zeros((3,)),
                    time_support=nap.IntervalSet([0, 1.5], [1.0, 2.0]),
                ),
                jnp.array([1]),
                jnp.array([1, 1, 0]),
            ),
        ],
    )
    def test_compute_is_new_session_from_pynapple(
        self, X, y, is_new_session, expected_new_session
    ):
        """Test that is_new_session is correctly computed from pynapple time support and interval sets"""
        model = MockHMM(n_states=3)
        is_new_session = model._validator.validate_and_cast_is_new_session(
            X, y, is_new_session
        )
        assert jnp.all(is_new_session == expected_new_session)

    @pytest.mark.parametrize(
        "X, y, is_new_session, expectation",
        [
            # wrong shape for boolean array
            (
                np.ones((3, 1)),
                np.ones((3,)),
                jnp.array([True, False, False, False]),
                pytest.raises(ValueError, match="is_new_session must have shape"),
            ),
            # wrong length for integer array
            (
                np.ones((3, 1)),
                np.ones((3,)),
                jnp.array([1, 0, 0, 0]),
                pytest.raises(
                    ValueError, match="is_new_session array must have length"
                ),
            ),
            # integer out of bounds
            (
                np.ones((3, 1)),
                np.ones((3,)),
                jnp.array([0, 3]),
                pytest.raises(ValueError, match="is_new_session values must be"),
            ),
            # negative integer
            (
                np.ones((3, 1)),
                np.ones((3,)),
                jnp.array([-1]),
                pytest.raises(ValueError, match="is_new_session values must be"),
            ),
            # wrong dtype
            (
                np.ones((3, 1)),
                np.ones((3,)),
                jnp.array([0.0, 3.0]),
                pytest.raises(
                    TypeError,
                    match="is_new_session must be a boolean or integer array",
                ),
            ),
            # wrong dtype
            (
                np.ones((3, 1)),
                np.ones((3,)),
                "is_new_session",
                pytest.raises(
                    TypeError,
                    match="is_new_session must be a boolean or integer array",
                ),
            ),
            # interval set when no pynapple objects are used
            (
                np.ones((3, 1)),
                np.ones((3,)),
                nap.IntervalSet([0, 3]),
                pytest.raises(
                    TypeError,
                    match="X or y must be a pynapple",
                ),
            ),
        ],
    )
    def test_initialize_and_compute_new_session_errors(
        self, X, y, is_new_session, expectation
    ):
        """Test that session boundaries are correctly initialized and moved when there are NaN values."""
        model = MockHMM(n_states=3)
        with expectation:
            is_new_session = model._validator.validate_and_cast_is_new_session(
                X, y, is_new_session
            )
