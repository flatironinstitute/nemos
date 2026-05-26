from contextlib import nullcontext as does_not_raise
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pynapple as nap
import pytest
import sklearn.cluster

from nemos._inspect_utils import extract_literal_options
from nemos.base_validator import RegressorValidator
from nemos.hmm.hmm import BaseHMM
from nemos.hmm.initialize_parameters import (
    AVAILABLE_INIT_FUNCTIONS,
    DEFAULT_INIT_FUNCTIONS,
    HMM_INITIALIZATION_FN_DICT,
    kmeans_initial_proba_init,
    kmeans_transition_proba_init,
    random_initial_proba_init,
    random_transition_proba_init,
    sticky_transition_proba_init,
    uniform_initial_proba_init,
    uniform_transition_proba_init,
)
from nemos.hmm.params import HMMParams
from nemos.hmm.utils import initialize_session_starts
from nemos.hmm.validation import HMMValidator, from_hmm_params, to_hmm_params
from nemos.params import ModelParams


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
    X_dimensionality: int = 2
    y_dimensionality: int = 1
    params_validation_sequence: Tuple[Tuple[str, None] | Tuple[str, dict[str, Any]]] = (
        *RegressorValidator.params_validation_sequence[:2],
        *HMMValidator.params_validation_sequence,
        *RegressorValidator.params_validation_sequence[3:],
    )

    def validate_consistency(self, *args, **kwargs) -> None:
        return True


class MockHMM(
    BaseHMM[
        MockHMMParams, MockHMMUserParams, HMM_INITIALIZATION_FN_DICT, MockHMMValidator
    ]
):
    _validator_class = MockHMMValidator
    _model_default_init_dict = {
        "param_init": None,
        "param_init_kwargs": {},
        "param_init_custom": False,
    }

    def __init__(
        self,
        n_states: int,
        dirichlet_initial_proba: Union[jnp.ndarray, None] = None,  # (n_state, )
        dirichlet_transition_proba: Union[
            jnp.ndarray | None
        ] = None,  # (n_state, n_state):
        maxiter: int = 1000,
        tol: float = 1e-8,
        seed=jax.random.PRNGKey(123),
        hmm_initialization_funcs: HMM_INITIALIZATION_FN_DICT = None,
        model_initialization_funcs: HMM_INITIALIZATION_FN_DICT = None,
    ):
        BaseHMM.__init__(
            self,
            n_states=n_states,
            dirichlet_initial_proba=dirichlet_initial_proba,
            dirichlet_transition_proba=dirichlet_transition_proba,
            maxiter=maxiter,
            tol=tol,
            seed=seed,
            hmm_initialization_funcs=hmm_initialization_funcs,
        )
        self.param_: jnp.ndarray | None = None
        self.model_initialization_funcs = model_initialization_funcs

    def _model_setup(
        self,
        param_init: Optional[str | Callable] = None,
        param_init_kwargs: Optional[dict] = None,
    ):
        self._model_use_kmeans = {"param_init": param_init == "kmeans"}

    def _check_model_is_fit(self):
        if self.param_ is None:
            raise ValueError("Model is not fitted yet.")

    def _get_model_params(self) -> MockHMMParams:
        return self._validator.to_model_params(
            (
                self.param_,
                self.initial_prob_,
                self.transition_prob_,
            )
        )

    def _set_model_params(self, params):
        param, initial_prob, transition_prob = self._validator.from_model_params(params)
        self.param_ = param
        self.initial_prob_ = initial_prob
        self.transition_prob_ = transition_prob

    def _log_likelihood(self, params, X, y):
        return jnp.zeros((y.shape[0], self.n_states))

    def _model_params_initialization(self, X, y, session_starts, random_key=None):
        return (
            jnp.arange(self._n_states),
            False,
        )

    def fit(self, X, y, session_starts=None, init_params=None):
        session_starts = initialize_session_starts(X, y, session_starts)
        fit_params = self._model_specific_initialization(X, y, session_starts)
        self._set_model_params(fit_params)

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

    def score(self, *args, **kwargs):
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
    # dirichlet_initial_proba setter tests
    # -------------------------------------------------------------------------
    def test_dirichlet_prior_init_prob_none(self):
        """Test that None is accepted for dirichlet prior."""
        model = MockHMM(n_states=3, dirichlet_initial_proba=None)
        assert model.dirichlet_initial_proba is None

    def test_dirichlet_prior_init_prob_valid(self):
        """Test valid dirichlet prior alphas."""
        alphas = jnp.array([1.0, 2.0, 3.0])
        model = MockHMM(n_states=3, dirichlet_initial_proba=alphas)
        assert jnp.array_equal(model.dirichlet_initial_proba, alphas)

    def test_dirichlet_prior_init_prob_wrong_shape(self):
        """Test that wrong shape raises ValueError."""
        alphas = jnp.array([1.0, 2.0])  # n_states=3 but only 2 elements
        with pytest.raises(ValueError, match="must have shape"):
            MockHMM(n_states=3, dirichlet_initial_proba=alphas)

    def test_dirichlet_prior_init_prob_values_less_than_one(self):
        """Test that alpha values < 1 raise ValueError."""
        alphas = jnp.array([1.0, 0.5, 2.0])
        with pytest.raises(ValueError, match="must be >= 1"):
            MockHMM(n_states=3, dirichlet_initial_proba=alphas)

    # -------------------------------------------------------------------------
    # dirichlet_transition_proba setter tests
    # -------------------------------------------------------------------------
    def test_dirichlet_prior_transition_none(self):
        """Test that None is accepted for dirichlet prior."""
        model = MockHMM(n_states=3, dirichlet_transition_proba=None)
        assert model.dirichlet_transition_proba is None

    def test_dirichlet_prior_transition_valid(self):
        """Test valid dirichlet prior alphas for transitions."""
        alphas = jnp.ones((3, 3))
        model = MockHMM(n_states=3, dirichlet_transition_proba=alphas)
        assert jnp.array_equal(model.dirichlet_transition_proba, alphas)

    def test_dirichlet_prior_transition_wrong_shape(self):
        """Test that wrong shape raises ValueError."""
        alphas = jnp.ones((2, 3))  # n_states=3 but wrong shape
        with pytest.raises(ValueError, match="must have shape"):
            MockHMM(n_states=3, dirichlet_transition_proba=alphas)

    # -------------------------------------------------------------------------
    # initialization_funcs setter tests
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

    @pytest.mark.parametrize(
        "init_funcs, expectation",
        [
            (
                {"invalid_key": None},
                pytest.raises(KeyError, match="[Uu]nknown key"),
            ),
            (
                {"initial_prob_init": None},
                pytest.raises(KeyError, match="Did you mean"),
            ),
        ],
    )
    def test_initialization_funcs_invalid_key_via_init(self, init_funcs, expectation):
        """Test that invalid/misspelled keys raise KeyError when passed via constructor."""
        with expectation:
            MockHMM(n_states=2, hmm_initialization_funcs=init_funcs)

    @pytest.mark.parametrize(
        "init_funcs, expectation",
        [
            (
                {"invalid_key": None},
                pytest.raises(KeyError, match="[Uu]nknown key"),
            ),
            (
                {"initial_prob_init": None},
                pytest.raises(KeyError, match="Did you mean"),
            ),
        ],
    )
    def test_initialization_funcs_invalid_key_via_setter(self, init_funcs, expectation):
        """Test that invalid/misspelled keys raise KeyError when assigned via setter."""
        model = MockHMM(n_states=2)
        with expectation:
            model.hmm_initialization_funcs = init_funcs

    # -------------------------------------------------------------------------
    # Default values tests
    # -------------------------------------------------------------------------
    def test_default_values(self):
        """Test that default values are set correctly."""
        model = MockHMM(n_states=3)

        assert model.n_states == 3
        assert model.maxiter == 1000
        assert model.tol == 1e-8
        assert model.dirichlet_initial_proba is None
        assert model.dirichlet_transition_proba is None

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
                None,
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
                lambda n_states, X, y, session_starts, random_key, extra_arg: jnp.full(
                    (n_states,), 1.0
                )
                / n_states,
                {"extra_arg": "value"},
                does_not_raise(),
            ),
            (
                "custom",
                lambda n_states, X, y, session_starts, random_key: jnp.full(
                    (n_states,), 1.0
                )
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
                None,
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
                lambda n_states, X, y, session_starts, random_key, extra_arg: jnp.full(
                    (n_states, n_states), 1.0
                )
                / n_states,
                {"extra_arg": "value"},
                does_not_raise(),
            ),
            (
                "custom",
                lambda n_states, X, y, session_starts, random_key: jnp.full(
                    (n_states, n_states), 1.0
                )
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
            "initial_proba_init_kwargs": {"minimum_prob": 0.01},
            "transition_proba_init": kmeans_transition_proba_init,
            "transition_proba_init_kwargs": {"minimum_prob": 0.01},
        }

        model = MockHMM(n_states=3)
        model.setup(
            initial_proba_init="kmeans",
            initial_proba_init_kwargs={"minimum_prob": 0.01},
            transition_proba_init="kmeans",
            transition_proba_init_kwargs={"minimum_prob": 0.01},
        )
        expected_funcs = DEFAULT_INIT_FUNCTIONS | init_funcs
        assert all(
            model.hmm_initialization_funcs[key] == expected_funcs[key]
            for key in expected_funcs
        )

    def test_setup_consecutive_calls(self):
        init_funcs = {
            "initial_proba_init": kmeans_initial_proba_init,
            "initial_proba_init_kwargs": {"minimum_prob": 0.01},
            "transition_proba_init": kmeans_transition_proba_init,
            "transition_proba_init_kwargs": {"minimum_prob": 0.01},
        }

        model = MockHMM(n_states=3)
        model.setup(initial_proba_init="kmeans")
        # updated
        assert (
            model.hmm_initialization_funcs["initial_proba_init"]
            == init_funcs["initial_proba_init"]
        )
        # default
        assert all(
            model.hmm_initialization_funcs[key] == DEFAULT_INIT_FUNCTIONS[key]
            for key in [
                "initial_proba_init_kwargs",
                "transition_proba_init",
                "transition_proba_init_kwargs",
            ]
        )

        model.setup(initial_proba_init_kwargs={"minimum_prob": 0.01})
        # updated
        assert all(
            model.hmm_initialization_funcs[key] == init_funcs[key]
            for key in [
                "initial_proba_init",
                "initial_proba_init_kwargs",
            ]
        )
        # default
        assert all(
            model.hmm_initialization_funcs[key] == DEFAULT_INIT_FUNCTIONS[key]
            for key in [
                "transition_proba_init",
                "transition_proba_init_kwargs",
            ]
        )

        model.setup(transition_proba_init="kmeans")
        # updated
        assert all(
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

        model.setup(transition_proba_init_kwargs={"minimum_prob": 0.01})
        # updated
        assert all(
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
        model.setup(**{key: "kmeans", key + "_kwargs": {"minimum_prob": 0.01}})
        assert model.hmm_initialization_funcs[key + "_kwargs"] == {"minimum_prob": 0.01}
        model.setup(**{key: "random"})
        assert model.hmm_initialization_funcs[key + "_kwargs"] == {}

    @pytest.mark.parametrize(
        "param_name", ["initial_proba_init", "transition_proba_init"]
    )
    def test_setup_literal_options_match_registry(self, param_name):
        """``BaseHMM.setup`` Literal annotations must enumerate exactly the
        built-in string aliases declared in ``AVAILABLE_INIT_FUNCTIONS``. If a
        new built-in is added (or one is removed) without updating the
        signature, this test fails — preventing silent drift."""
        literals = extract_literal_options(BaseHMM.setup, param_name)
        registry = AVAILABLE_INIT_FUNCTIONS[param_name]
        assert literals == set(registry.keys()), (
            f"Literal options for {param_name!r} in BaseHMM.setup ({literals}) "
            f"differ from registered keys ({set(registry.keys())})."
        )


class TestHMMInitialParams:

    def test__hmm_params_initialization_defaults(self):
        """Test that _hmm_params_initialization returns expected default parameters and validation flag."""
        model = MockHMM(n_states=3)

        (initial_prob, transition_prob), validate_params = (
            model._hmm_params_initialization(
                None,
                None,
                None,
                random_key_pair=jax.random.split(jax.random.PRNGKey(0), 2),
            )
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
            initial_proba_init=lambda n_states, X, y, session_starts, random_key: jnp.full(
                (n_states,), 1.0
            )
            / n_states
        )
        (initial_prob, _), validate_params = model._hmm_params_initialization(
            None, None, None, random_key_pair=jax.random.split(jax.random.PRNGKey(0), 2)
        )
        assert jnp.allclose(initial_prob, jnp.full((3,), 1.0) / 3)
        assert validate_params is True

        model = MockHMM(n_states=3)
        model.setup(
            transition_proba_init=lambda n_states, X, y, session_starts, random_key: jnp.full(
                (n_states, n_states), 1.0
            )
            / n_states
        )
        (_, transition_prob), validate_params = model._hmm_params_initialization(
            None, None, None, random_key_pair=jax.random.split(jax.random.PRNGKey(0), 2)
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

    def test__kmeans_setup_initializer(self):
        """Verify setter is stored and is the same for hmm and model params."""
        model = MockHMM(n_states=3)
        model.setup(
            initial_proba_init="kmeans",
            transition_proba_init="kmeans",
            param_init="kmeans",
        )
        original_fit = sklearn.cluster.KMeans.fit
        # use mock fit to assert that kmeans is only called once at initializer construction
        with patch.object(
            sklearn.cluster.KMeans, "fit", autospec=True, side_effect=original_fit
        ) as mock_fit:
            model._model_specific_initialization(
                jnp.zeros((10, 10)), jnp.zeros(10), None
            )
            assert id(
                model.hmm_initialization_funcs["initial_proba_init_kwargs"][
                    "initializer"
                ]
            ) == id(
                model.hmm_initialization_funcs["transition_proba_init_kwargs"][
                    "initializer"
                ]
            )
            assert id(
                model.hmm_initialization_funcs["initial_proba_init_kwargs"][
                    "initializer"
                ]
            ) == id(
                model.model_initialization_funcs["param_init_kwargs"]["initializer"]
            )
            assert mock_fit.call_count == 1

    def test_kmeans_inconsistent_kwargs_raises(self):
        """_kmeans_resolve_model_kwargs raises when the same kwarg has conflicting values."""
        model = MockHMM(n_states=3)
        use_kmeans = {"param_a": True, "param_b": True}
        init_funcs = {
            "param_a_kwargs": {"minimum_prob": 0.01},
            "param_b_kwargs": {"minimum_prob": 0.05},
        }
        with pytest.raises(ValueError, match="Inconsistent KMeans init arg"):
            model._kmeans_resolve_model_kwargs(use_kmeans, init_funcs)

    @pytest.mark.parametrize(
        "key, value, expectation",
        [
            (
                "initial_proba_init",
                lambda n_states, X, y, session_starts, random_key: jnp.ones((n_states,))
                / n_states,
                does_not_raise(),
            ),
            (
                "transition_proba_init",
                lambda n_states, X, y, session_starts, random_key: jnp.ones(
                    (n_states, n_states)
                )
                / n_states,
                does_not_raise(),
            ),
            (
                "initial_proba_init",
                lambda n_states, X, y, session_starts, random_key: jnp.ones(
                    (n_states - 1,)
                ),
                pytest.raises(ValueError, match="initial_prob must be"),
            ),
            (
                "transition_proba_init",
                lambda n_states, X, y, session_starts, random_key: jnp.ones(
                    (n_states - 1, n_states - 1)
                ),
                pytest.raises(ValueError, match="transition_prob must be"),
            ),
            (
                "initial_proba_init",
                lambda n_states, X, y, session_starts, random_key: jnp.ones(
                    (n_states,)
                ),
                pytest.raises(ValueError, match="must sum to 1"),
            ),
            (
                "transition_proba_init",
                lambda n_states, X, y, session_starts, random_key: jnp.ones(
                    (n_states, n_states)
                ),
                pytest.raises(ValueError, match="rows must sum to 1"),
            ),
        ],
    )
    def test_custom_init_and_transition_prob_sum_and_shape(
        self, key, value, expectation
    ):
        model = MockHMM(n_states=3)
        model.setup(**{key: value})
        with expectation:
            model._model_specific_initialization(None, None, None)


class TestHMMNewSession:

    @pytest.mark.parametrize(
        "X, y, session_starts, expected_new_session",
        [
            # No session_starts provided
            (np.ones((3, 1)), np.ones((3,)), None, jnp.array([1, 0, 0])),
            # Explicit session_starts provided by user
            # boolean array or integer array with 1s and 0s
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
            # integer array with indices of new sessions
            # 1 session
            (
                np.ones((5, 1)),
                np.array([0, 1, 2, 3, 4]),
                jnp.array([0]),
                jnp.array([1, 0, 0, 0, 0]),
            ),
            # repeated values
            (
                np.ones((5, 1)),
                np.array([0, 1, 2, 3, 4]),
                jnp.array([0, 0]),
                jnp.array([1, 0, 0, 0, 0]),
            ),
            # all 0s and 1s with length < n_samples
            (
                np.ones((5, 1)),
                np.array([0, 1, 2, 3, 4]),
                jnp.array([0, 1]),
                jnp.array([1, 1, 0, 0, 0]),
            ),
            # higher int values, adding first session
            (
                np.ones((5, 1)),
                np.array([0, 1, 2, 3, 4]),
                jnp.array([2, 4]),
                jnp.array([1, 0, 1, 0, 1]),
            ),
        ],
    )
    def test_initialize_new_session(self, X, y, session_starts, expected_new_session):
        """Test that session boundaries are correctly initialized."""
        model = MockHMM(n_states=3)
        session_starts = model._validator.validate_and_cast_session_starts(
            X, y, session_starts
        )
        assert jnp.all(session_starts == expected_new_session)

    @pytest.mark.parametrize(
        "X, y, session_starts, expected_new_session",
        [
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
            # Explicit session_starts provided by user
            # beginning session shifted
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
            # middle session moved
            (
                np.ones((5, 1)),
                np.array([0, 1, np.nan, np.nan, 3]),
                jnp.array([1, 0, 1, 0, 0]),
                jnp.array([1, 0, 0, 0, 1]),
            ),
            # nan moving is independent of input type so I won't add casses with different types
        ],
    )
    def test_initialize_new_session_with_nan_shift(
        self, X, y, session_starts, expected_new_session
    ):
        """Test that session boundaries are correctly moved when there are NaN values."""
        model = MockHMM(n_states=3)
        session_starts = model._validator.validate_and_cast_session_starts(
            X, y, session_starts
        )
        assert jnp.all(session_starts == expected_new_session)

    @pytest.mark.parametrize(
        "X, y, session_starts, expected_new_session",
        [
            # no session_starts provided
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
            # intervalset passed as session_starts
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
    def test_compute_session_starts_from_pynapple(
        self, X, y, session_starts, expected_new_session
    ):
        """Test that session_starts is correctly computed from pynapple time support and interval sets"""
        model = MockHMM(n_states=3)
        session_starts = model._validator.validate_and_cast_session_starts(
            X, y, session_starts
        )
        assert jnp.all(session_starts == expected_new_session)

    @pytest.mark.parametrize(
        "X, y, session_starts, expected",
        [
            # Use provided iset
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
                nap.IntervalSet(1, 10),
                jnp.array([True, True, False]),
            ),
            # Use iset from Tsds
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
                jnp.array([True, False, True]),
            ),
        ],
    )
    def test_session_starts_resolution_hierarchy(self, X, y, session_starts, expected):
        model = MockHMM(n_states=3)
        session_starts = model._validator.validate_and_cast_session_starts(
            X, y, session_starts
        )
        np.testing.assert_array_equal(session_starts, expected)

    @pytest.mark.parametrize(
        "X, y, session_starts, expectation",
        [
            # wrong shape for boolean array
            (
                np.ones((3, 1)),
                np.ones((3,)),
                jnp.array([True, False, False, False]),
                pytest.raises(ValueError, match="session_starts must have shape"),
            ),
            # wrong length for integer array
            (
                np.ones((3, 1)),
                np.ones((3,)),
                jnp.array([1, 0, 0, 0]),
                pytest.raises(
                    ValueError, match="session_starts array must have length"
                ),
            ),
            # integer out of bounds
            (
                np.ones((3, 1)),
                np.ones((3,)),
                jnp.array([0, 3]),
                pytest.raises(ValueError, match="session_starts values must be"),
            ),
            # negative integer
            (
                np.ones((3, 1)),
                np.ones((3,)),
                jnp.array([-1]),
                pytest.raises(ValueError, match="session_starts values must be"),
            ),
            # wrong dtype
            (
                np.ones((3, 1)),
                np.ones((3,)),
                jnp.array([0.0, 3.0]),
                pytest.raises(
                    TypeError,
                    match="session_starts must be a boolean or integer array",
                ),
            ),
            # wrong dtype
            (
                np.ones((3, 1)),
                np.ones((3,)),
                "session_starts",
                pytest.raises(
                    TypeError,
                    match="session_starts must be a boolean or integer array",
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
        self, X, y, session_starts, expectation
    ):
        """Test that session boundaries are correctly initialized and moved when there are NaN values."""
        model = MockHMM(n_states=3)
        with expectation:
            session_starts = model._validator.validate_and_cast_session_starts(
                X, y, session_starts
            )


def all_subclasses(cls):
    seen = set()
    stack = list(cls.__subclasses__())
    while stack:
        sub = stack.pop()
        if sub in seen:
            continue
        seen.add(sub)
        stack.extend(sub.__subclasses__())
    return seen


class TestHMMValidator:
    """Test suite for input validation logic in HMMValidator."""

    def test_user_param_order(self) -> None:
        """Meta-test.

        Tests that any subclasses of HMMValidator have the correct user parameter order
        """
        import importlib
        import pkgutil

        import nemos

        # Import every submodule so all HMMValidator subclasses get registered.
        for _, modname, _ in pkgutil.walk_packages(nemos.__path__, prefix="nemos."):
            importlib.import_module(modname)

        # Filter the classes that are subclasses of 'SuperClass'.
        subclasses = all_subclasses(HMMValidator)

        for validator in subclasses:
            n_params = len(validator.model_param_names)
            user_par = [0.0] * (n_params - 2) + [1.0, 1.0]
            params = validator.to_model_params(user_par)
            assert np.all(params.hmm_params.log_initial_prob == 0.0)
            assert np.all(params.hmm_params.log_transition_prob == 0.0)

    @pytest.mark.parametrize(
        "X, y, expectation",
        [
            (
                np.random.rand(10, 2),
                np.random.rand(10),
                does_not_raise(),
            ),
            (
                np.random.rand(10, 2),
                np.random.rand(9),
                pytest.raises(ValueError, match="X and y must have"),
            ),
            (
                nap.TsdFrame(
                    t=np.arange(10),
                    d=np.random.rand(10, 2),
                ),
                nap.Tsd(
                    t=np.arange(10) + 1,
                    d=np.random.rand(10),
                ),
                pytest.raises(ValueError, match="Time axis mismatch"),
            ),
        ],
    )
    def test_validate_inputs(self, X, y, expectation):
        """Test that validate_inputs correctly validates X and y."""
        model = MockHMM(n_states=3)
        with expectation:
            model._validator.validate_inputs(X, y)

    @pytest.mark.parametrize(
        "X, y, expectation",
        [
            # nan border y
            (
                np.ones((5, 1)),
                np.array([np.nan, 1, 2, 3, np.nan]),
                does_not_raise(),
            ),
            # nan border x
            (
                np.array([[np.nan], [2], [3], [np.nan]]),
                np.array([0, 1, 3, 4]),
                does_not_raise(),
            ),
            # nan middle y
            (
                np.ones((5, 1)),
                np.array([np.nan, 1, np.nan, 2, 3]),
                pytest.raises(ValueError, match="HMM requires continuous"),
            ),
            # nan middle x
            (
                np.array([[np.nan], [2], [np.nan], [3]]),
                np.array([0, 1, 3, 4]),
                pytest.raises(ValueError, match="HMM requires continuous"),
            ),
        ],
    )
    def test_nans_only_at_border(self, X, y, expectation):
        """Test that validate_inputs allows NaNs only at the borders of the data."""
        model = MockHMM(n_states=3)
        with expectation:
            model._validator.validate_inputs(X, y)


class TestHMMInference:
    """Test suite for inference methods (smooth_proba, filter_proba, decode_state)."""

    @staticmethod
    def _get_expected_shape(method_name, kwargs, n_samples, n_states):
        """Helper to compute expected output shape based on method and kwargs."""
        if method_name in ["smooth_proba", "filter_proba"]:
            return (n_samples, n_states)
        elif method_name == "decode_state":
            if kwargs.get("state_format") == "index":
                return (n_samples,)
            else:  # one-hot (default)
                return (n_samples, n_states)
        else:
            raise ValueError(f"Unknown method: {method_name}")

    @pytest.mark.parametrize(
        "drop_attr",
        ["initial_prob_", "transition_prob_"],
    )
    @pytest.mark.parametrize(
        "method_config",
        [
            pytest.param(("smooth_proba", {}), id="smooth_proba"),
            pytest.param(("filter_proba", {}), id="filter_proba"),
            pytest.param(("decode_state", {}), id="decode_state-onehot"),
            pytest.param(
                ("decode_state", {"state_format": "index"}), id="decode_state-index"
            ),
        ],
    )
    def test_not_fitted_raises_error(self, drop_attr, method_config):
        """Test that inference methods raise an error when model is not fitted."""
        method_name, kwargs = method_config
        model = MockHMM(n_states=3)
        model.fit(np.random.rand(10, 2), np.random.rand(10))
        setattr(model, drop_attr, None)
        with pytest.raises(
            ValueError,
            match=rf"This MockHMM instance is not fitted yet. .+ \['{drop_attr}'\]",
        ):
            getattr(model, method_name)(None, None, **kwargs)

    @pytest.mark.parametrize(
        "method_config",
        [
            pytest.param(("smooth_proba", {}), id="smooth_proba"),
            pytest.param(("filter_proba", {}), id="filter_proba"),
            pytest.param(("decode_state", {}), id="decode_state-onehot"),
            pytest.param(
                ("decode_state", {"state_format": "index"}), id="decode_state-index"
            ),
        ],
    )
    def test_returns_correct_shape(self, method_config):
        """Test that inference methods return arrays with correct shapes."""
        method_name, kwargs = method_config
        model = MockHMM(n_states=3)

        # Get output
        X = np.random.rand(10, 2)
        y = np.random.rand(10)
        model.fit(X, y)
        out = getattr(model, method_name)(X, y, **kwargs)

        # Check shape
        n_samples = (~np.isnan(np.sum(y, axis=tuple(range(1, y.ndim))))).sum()
        n_states = model.n_states
        expected_shape = self._get_expected_shape(
            method_name, kwargs, n_samples, n_states
        )
        assert (
            out.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {out.shape}"

    @pytest.mark.parametrize("method_name", ["smooth_proba", "filter_proba"])
    def test_posterior_proba_returns_valid_probabilities(self, method_name):
        """Test that smooth_proba returns valid probabilities (between 0 and 1, summing to 1)."""
        model = MockHMM(n_states=3)
        X = np.random.rand(10, 2)
        y = np.random.rand(10)
        model.fit(X, y)

        # Get posteriors
        posteriors = getattr(model, method_name)(X, y)

        # Check all values are between 0 and 1
        assert jnp.all(posteriors >= 0), "Some posteriors are negative"
        assert jnp.all(posteriors <= 1), "Some posteriors are greater than 1"

        # Check sum across states
        row_sums = jnp.sum(posteriors, axis=1)
        assert jnp.allclose(
            row_sums, 1.0, rtol=1e-5
        ), f"Probabilities don't sum to 1. Min: {row_sums.min()}, Max: {row_sums.max()}"

    @pytest.mark.parametrize(
        "method_config",
        [
            pytest.param(("smooth_proba", {}), id="smooth_proba"),
            pytest.param(("filter_proba", {}), id="filter_proba"),
            pytest.param(("decode_state", {}), id="decode_state-onehot"),
            pytest.param(
                ("decode_state", {"state_format": "index"}), id="decode_state-index"
            ),
        ],
    )
    def test_with_arrays(self, method_config):
        """Test inference methods with numpy/jax arrays return jax array."""
        method_name, kwargs = method_config
        model = MockHMM(n_states=3)
        X = np.random.rand(10, 2)
        y = np.random.rand(10)
        model.fit(X, y)

        # Test with numpy array
        out = getattr(model, method_name)(X, y, **kwargs)
        assert isinstance(out, jnp.ndarray), f"Expected jnp.ndarray, got {type(out)}"

    @pytest.mark.parametrize("input_type", ["X", "y", "both"])
    @pytest.mark.parametrize(
        "method_config",
        [
            pytest.param(("smooth_proba", {}), id="smooth_proba"),
            pytest.param(("filter_proba", {}), id="filter_proba"),
            pytest.param(("decode_state", {}), id="decode_state-onehot"),
            pytest.param(
                ("decode_state", {"state_format": "index"}), id="decode_state-index"
            ),
        ],
    )
    def test_with_pynapple_returns_tsdframe(self, input_type, method_config):
        """Test that inference methods return TsdFrame/Tsd when input is pynapple."""
        method_name, kwargs = method_config
        model = MockHMM(n_states=3)
        X = np.random.rand(10, 2)
        y = np.random.rand(10)
        model.fit(X, y)

        # Convert to pynapple
        n_samples = X.shape[0]
        time = np.linspace(0, n_samples / 100, n_samples)

        if input_type in ["X", "both"]:
            X = nap.TsdFrame(t=time, d=X)
        if input_type in ["y", "both"]:
            y = nap.Tsd(t=time, d=y)

        # Get output
        out = getattr(model, method_name)(X, y, **kwargs)

        # Check return type - decode_state with index format returns Tsd, others return TsdFrame
        if method_name == "decode_state" and kwargs.get("state_format") == "index":
            assert isinstance(out, nap.Tsd), f"Expected nap.Tsd, got {type(out)}"
            assert out.shape == (n_samples,)
        else:
            assert isinstance(
                out, nap.TsdFrame
            ), f"Expected nap.TsdFrame, got {type(out)}"
            assert out.shape == (n_samples, model.n_states)
        assert jnp.allclose(out.t, time)

    @pytest.mark.parametrize(
        "method_config",
        [
            pytest.param(("smooth_proba", {}), id="smooth_proba"),
            pytest.param(("filter_proba", {}), id="filter_proba"),
            pytest.param(("decode_state", {}), id="decode_state-onehot"),
            pytest.param(
                ("decode_state", {"state_format": "index"}), id="decode_state-index"
            ),
        ],
    )
    def test_with_multiple_sessions(self, method_config):
        """Test inference methods with multiple sessions (pynapple epochs)."""
        method_name, kwargs = method_config
        model = MockHMM(n_states=3)
        X = np.random.rand(10, 2)
        y = np.random.rand(10)
        model.fit(X, y)

        # Create multi-session data
        n_samples = X.shape[0]
        session_1_end = n_samples // 2

        time = np.linspace(0, n_samples / 100, n_samples)
        epochs = nap.IntervalSet(
            start=[time[0], time[session_1_end]],
            end=[time[session_1_end - 1], time[-1]],
        )

        X_tsd = nap.TsdFrame(t=time, d=X, time_support=epochs)
        y_tsd = nap.Tsd(t=time, d=y, time_support=epochs)

        # Get output
        out = getattr(model, method_name)(X_tsd, y_tsd, **kwargs)

        # Check shape and type
        if method_name == "decode_state" and kwargs.get("state_format") == "index":
            assert isinstance(out, nap.Tsd)
            assert out.shape == (n_samples,)
        else:
            assert isinstance(out, nap.TsdFrame)
            assert out.shape == (n_samples, model.n_states)

        # Check probabilities are valid for proba methods
        if method_name in ["smooth_proba", "filter_proba"]:
            assert jnp.all(out.values >= 0)
            assert jnp.all(out.values <= 1)
            row_sums = jnp.sum(out.values, axis=1)
            assert jnp.allclose(row_sums, 1.0, rtol=1e-5)

    @pytest.mark.parametrize(
        "method_config",
        [
            pytest.param(("smooth_proba", {}), id="smooth_proba"),
            pytest.param(("filter_proba", {}), id="filter_proba"),
            pytest.param(("decode_state", {}), id="decode_state-onehot"),
            pytest.param(
                ("decode_state", {"state_format": "index"}), id="decode_state-index"
            ),
        ],
    )
    def test_consistency_across_calls(self, method_config):
        """Test that inference methods return consistent results across multiple calls."""
        method_name, kwargs = method_config
        model = MockHMM(n_states=3)
        X = np.random.rand(10, 2)
        y = np.random.rand(10)
        model.fit(X, y)

        # Get output twice
        out_1 = getattr(model, method_name)(X, y, **kwargs)
        out_2 = getattr(model, method_name)(X, y, **kwargs)

        # Check consistency
        assert jnp.allclose(
            out_1, out_2
        ), f"{method_name} returns different results on consecutive calls"

    @pytest.mark.parametrize(
        "method_name", ["smooth_proba", "filter_proba", "decode_state"]
    )
    def test_single_sample(self, method_name):
        """Test smooth_proba with a single sample."""
        model = MockHMM(n_states=3)
        X = np.random.rand(1, 2)
        y = np.random.rand(1)
        model.fit(X, y)

        # Get posteriors for single sample
        out = getattr(model, method_name)(X, y)

        # Check shape
        assert out.shape == (1, model.n_states)

        if method_name != "decode_state":
            # Check probabilities are valid
            assert jnp.all(out >= 0)
            assert jnp.all(out <= 1)
            assert jnp.allclose(jnp.sum(out), 1.0, rtol=1e-5)

    @pytest.mark.parametrize(
        "method_name", ["smooth_proba", "filter_proba", "decode_state"]
    )
    def test_with_nans_filtered(self, method_name):
        """Test that smooth_proba handles NaNs properly by filtering them."""
        model = MockHMM(n_states=3)
        X = np.random.rand(10, 2)
        y = np.random.rand(10)
        model.fit(X, y)

        # Create data with NaNs
        X_with_nan = X.copy()
        y_with_nan = y.copy()

        # Add NaNs at specific indices
        nan_indices = [0, 1, 2]
        X_with_nan[nan_indices] = np.nan

        # This should work - NaNs get filtered internally
        posteriors = getattr(model, method_name)(X_with_nan, y_with_nan)

        # Check that we get valid output (NaN rows filtered)
        assert posteriors.shape[1] == model.n_states
        assert posteriors.shape[0] == X.shape[0]

    @pytest.mark.parametrize(
        "method_name", ["smooth_proba", "filter_proba", "decode_state"]
    )
    @pytest.mark.parametrize("nan_location", [[], [0, 1, 10, 11, 12]])
    def test_pynapple_in_pynapple_out_X(self, method_name, nan_location):
        model = MockHMM(n_states=3)
        X = np.random.rand(100, 2)
        y = np.random.rand(100)
        model.fit(X, y)
        X[nan_location] = np.nan
        ep = nap.IntervalSet([0, 10], [9, 500])
        X = nap.TsdFrame(t=np.arange(X.shape[0]), d=X, time_support=ep)
        out = getattr(model, method_name)(X, y)
        assert isinstance(out, nap.TsdFrame), "Did not return pynapple!"
        assert np.all(
            np.isnan(out[nan_location])
        ), "Not returning NaNs in the expected location!"

    @pytest.mark.parametrize(
        "method_name", ["smooth_proba", "filter_proba", "decode_state"]
    )
    @pytest.mark.parametrize("nan_location", [[], [0, 1, 10, 11, 12]])
    def test_pynapple_in_pynapple_out_y(self, method_name, nan_location):
        model = MockHMM(n_states=3)
        X = np.random.rand(100, 2)
        y = np.random.rand(100)
        model.fit(X, y)
        y[nan_location] = np.nan
        ep = nap.IntervalSet([0, 10], [9, 500])
        y = nap.Tsd(t=np.arange(y.shape[0]), d=y, time_support=ep)
        posteriors = getattr(model, method_name)(X, y)
        assert isinstance(posteriors, nap.TsdFrame), "Did not return pynapple!"
        assert np.all(
            np.isnan(posteriors[nan_location])
        ), "Not returning NaNs in the expected location!"

    @pytest.mark.parametrize(
        "method_name", ["smooth_proba", "filter_proba", "decode_state"]
    )
    def test_int_vs_float_y(self, method_name):
        """Test that integer and float y with same values give same posteriors.

        This is a regression test for a bug where y.dtype was used to cast params
        before preprocessing, causing integer y to round float params to integers.
        """
        model = MockHMM(n_states=3)
        X = np.random.rand(100, 2)
        y = np.random.rand(100)
        model.fit(X, y)
        y = np.round(y)
        y_float = y.astype(float)
        y_int = y.astype(int)

        # Get posteriors with float y
        out_float = getattr(model, method_name)(X, y_float)

        # Get posteriors with int y (same values)
        out_int = getattr(model, method_name)(X, y_int)

        # Posteriors should be identical regardless of y dtype
        np.testing.assert_allclose(
            out_float,
            out_int,
            rtol=1e-10,
            err_msg=f"{method_name} gives different results for int vs float y with same values",
        )

    def test_onehot_vs_index_decode(self):
        model = MockHMM(n_states=3)
        X = np.random.rand(100, 2)
        y = np.random.rand(100)
        model.fit(X, y)
        out_onehot = model.decode_state(X, y, state_format="one-hot")
        out_index = model.decode_state(X, y, state_format="index")
        assert jnp.all(
            jnp.where(out_onehot == 1)[1] == out_index
        ), "index and one-hot do not match!"
        assert jnp.all(
            out_onehot.sum(axis=1) == 1
        ), "more than one hot value in one-hot array!"

    def test_decode_state_invalid_state_format(self):
        """Test that decode_state raises ValueError for invalid state_format."""
        model = MockHMM(n_states=3)
        X = np.random.rand(100, 2)
        y = np.random.rand(100)
        model.fit(X, y)
        with pytest.raises(ValueError, match="Invalid state_format"):
            model.decode_state(X, y, state_format="invalid")

    @pytest.mark.parametrize("n_states", [2, 3, 5])
    @pytest.mark.parametrize(
        "method_name", ["smooth_proba", "filter_proba", "decode_state"]
    )
    def test_different_n_states(self, n_states, method_name):
        """Test smooth_proba with different numbers of states."""
        model = MockHMM(n_states=n_states)
        n_samples, n_features = 100, 2
        X = np.random.rand(n_samples, n_features)
        y = np.random.rand(n_samples)
        model.fit(X, y)

        out = getattr(model, method_name)(X, y)

        # Check shape
        assert out.shape == (
            n_samples,
            n_states,
        ), f"Expected shape ({n_samples}, {n_states}), got {out.shape}"

        # Check probabilities are valid
        assert jnp.all(out >= 0)
        assert jnp.all(out <= 1)
        row_sums = jnp.sum(out, axis=1)
        assert jnp.allclose(row_sums, 1.0)

    @pytest.mark.parametrize(
        "method_name", ["smooth_proba", "filter_proba", "decode_state"]
    )
    def test_session_starts_is_used(self, method_name):
        model = MockHMM(n_states=3)
        X = np.random.rand(10, 2)
        y = np.random.rand(10)
        model.param_ = jnp.ones((3,))
        model.initial_prob_ = jnp.array([0.8, 0.1, 0.1])
        model.transition_prob_ = jnp.array(
            [[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]]
        )
        out_all_new_sess = getattr(model, method_name)(
            X, y, session_starts=np.ones(10, dtype=bool)
        )
        assert np.all(
            out_all_new_sess == out_all_new_sess[0]
        ), "Output should be the same for all time points"
        out_default = getattr(model, method_name)(X, y)
        assert not np.allclose(
            out_all_new_sess, out_default
        ), "Output with all new sessions should not match default output"
        out_no_new_sess = getattr(model, method_name)(
            X, y, session_starts=np.zeros(10, dtype=bool)
        )
        assert jnp.allclose(
            out_no_new_sess, out_default
        ), "Output with no new sessions should match default output"
