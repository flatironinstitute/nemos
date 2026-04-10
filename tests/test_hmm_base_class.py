from nemos.hmm.hmm import BaseHMM
from typing import Union, Optional, Callable
import jax
import jax.numpy as jnp
from nemos.hmm.initialize_parameters import INITIALIZATION_FN_DICT
from nemos.params import ModelParams
from nemos.hmm.params import HMMParams
from nemos.base_regressor import BaseRegressor
from nemos.hmm.validation import HMMValidator


class MockHMMValidator(HMMValidator):
    pass


class MockHMMModelParams(ModelParams):
    param: jnp.ndarray


class MockHMMParams(ModelParams):
    model_params: MockHMMModelParams
    hmm_params: HMMParams


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
