from nemos.hmm.hmm import BaseHMM
from typing import Union, Optional, Callable, Tuple
import jax
import jax.numpy as jnp
from nemos.hmm.initialize_parameters import INITIALIZATION_FN_DICT
from nemos.params import ModelParams
from nemos.hmm.params import HMMParams
from nemos.base_regressor import BaseRegressor
from nemos.hmm.validation import HMMValidator
from nemos.base_validator import RegressorValidator
from nemos.hmm.validation import to_hmm_params, from_hmm_params


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


class MockHMMValidator(HMMValidator):
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
    model_class: str = "GLMHMM"
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


class MockHMM(BaseHMM, BaseRegressor):
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
        super(BaseHMM).__init__(
            n_states=n_states,
            dirichlet_prior_alphas_init_prob=dirichlet_prior_alphas_init_prob,
            dirichlet_prior_alphas_transition=dirichlet_prior_alphas_transition,
            maxiter=maxiter,
            tol=tol,
            seed=seed,
            hmm_initialization_funcs=hmm_initialization_funcs,
        )
        self.param_: jnp.ndarray | None = None
        self._validator_class = MockHMMValidator

    def setup(
        self,
        initial_proba_init: Optional[str | Callable] = None,
        initial_proba_init_kwargs: Optional[dict] = None,
        transition_proba_init: Optional[str | Callable] = None,
        transition_proba_init_kwargs: Optional[dict] = None,
        param_init: Optional[str | Callable] = None,
        param_init_kwargs: Optional[dict] = None,
    ):
        super(BaseHMM).setup(
            initial_proba_init=initial_proba_init,
            initial_proba_init_kwargs=initial_proba_init_kwargs,
            transition_proba_init=transition_proba_init,
            transition_proba_init_kwargs=transition_proba_init_kwargs,
        )

    def _check_is_fit(self):
        super(BaseHMM)._check_is_fit()
        if self.param_ is None:
            raise ValueError("Model is not fitted yet.")

    def _log_likelihood(self, params, X, y):
        pass

    def fit(self, X, y, is_new_session=None, init_params=None):
        pass
