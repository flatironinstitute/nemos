"""API for the GLM-HMM model."""

from numbers import Number
from pathlib import Path
from typing import Callable, Literal, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import pynapple as nap
from numpy.typing import ArrayLike, NDArray

from .. import observation_models as obs
from .._observation_model_builder import instantiate_observation_model
from ..base_regressor import BaseRegressor
from ..inverse_link_function_utils import resolve_inverse_link_function
from ..observation_models import Observations
from ..regularizer import Regularizer
from ..typing import (
    DESIGN_INPUT_TYPE,
    RegularizerStrength,
    SolverState,
    StepResult,
)
from ..utils import format_repr
from .initialize_parameters import (
    INITIALIZATION_FN_DICT,
    _is_native_init_registry,
    _resolve_init_funcs_registry,
    glm_hmm_initialization,
    resolve_dirichlet_priors,
)
from .params import GLMHMMParams, GLMHMMUserParams
from .validation import GLMHMMValidator


class GLMHMM(BaseRegressor[GLMHMMUserParams, GLMHMMParams]):
    """GLM-HMM model."""

    def __init__(
        self,
        n_states: int,
        observation_model: (
            Observations
            | Literal["Poisson", "Gamma", "Bernoulli", "NegativeBinomial", "Gaussian"]
        ) = "Bernoulli",
        inverse_link_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        regularizer: Optional[Union[str, Regularizer]] = None,
        regularizer_strength: Optional[
            RegularizerStrength
        ] = None,  # this is used to regularize GLM coef.
        # prior to regularize init prob and transition
        dirichlet_prior_alphas_init_prob: Union[
            jnp.ndarray, None
        ] = None,  # (n_state, )
        dirichlet_prior_alphas_transition: Union[
            jnp.ndarray | None
        ] = None,  # (n_state, n_state)
        solver_name: str = None,
        solver_kwargs: Optional[dict] = None,
        initialization_funcs: Optional[INITIALIZATION_FN_DICT] = None,
        maxiter: int = 1000,
        tol: float = 1e-8,
        seed=jax.random.PRNGKey(123),
    ):
        super().__init__(
            regularizer=regularizer,
            regularizer_strength=regularizer_strength,
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
        )
        self.n_states = n_states
        self.observation_model = observation_model
        self.inverse_link_function = inverse_link_function

        # assign defaults to initialization functions
        self.initialization_funcs = initialization_funcs

        # set the prior params
        self.dirichlet_prior_alphas_init_prob = dirichlet_prior_alphas_init_prob
        self.dirichlet_prior_alphas_transition = dirichlet_prior_alphas_transition

        self.seed = seed
        self.maxiter = maxiter
        self.tol = tol
        # Model log-likelihood
        # inputs: (y, firing_rate)
        # input shapes:
        #     - y: (n_samples,)
        #     - firing_rate: (n_samples,)
        # returns: a 0-dim jnp.ndarray
        self._likelihood_func: (
            Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None
        ) = None
        # expected negative log-likelihood as function of the GLM parameters.
        # inputs: ((coef, intercept), X, y, posteriors)
        # input shapes:
        #     - (coef, intercept): ( (n_features, n_states), (n_states,) )
        #     - X: (n_samples, n_features)
        #     - y: (n_samples,)
        #     - posteriors: (n_samples, n_states)
        self._expected_negative_log_likelihood: (
            Callable[
                [
                    Tuple[DESIGN_INPUT_TYPE, jnp.ndarray],
                    jnp.ndarray,
                    jnp.ndarray,
                    jnp.ndarray,
                ],
                jnp.ndarray,
            ]
            | None
        ) = None

        # fit attributes
        self.transition_prob_: jnp.ndarray | None = None
        self.initial_prob_: jnp.ndarray | None = None
        self.coef_: jnp.ndarray | None = None
        self.intercept_: jnp.ndarray | None = None
        self.solver_state_: NamedTuple | None = None
        self.scale_: jnp.ndarray | None = None
        self.dof_resid_: int | None = None
        self.aux_ = None

    @property
    def n_states(self) -> int:
        """Number of hidden states of the HMM."""
        return self._n_states

    @n_states.setter
    def n_states(self, n_states: int):
        # quick sanity check and assignment
        if isinstance(n_states, int) and n_states > 0:
            self._n_states = n_states
            self._validator = GLMHMMValidator(n_states=n_states)
            return

        # further checks for other valid numeric types (like non-negative float with no-decimals)
        if not isinstance(n_states, Number):
            raise TypeError(
                f"n_states must be a positive integer. "
                f"n_states is of type ``{type(n_states)}`` instead."
            )

        # provided a non-integer number (check that has no decimals)
        int_n_states = int(n_states)
        if int_n_states != int(n_states):
            raise TypeError(
                f"n_states must be a positive integer. ``{n_states}`` provided instead."
            )
        elif int_n_states < 1:
            raise ValueError(
                f"n_states must be a positive integer. ``{n_states}`` provided instead."
            )
        self._n_states = int_n_states
        self._validator = GLMHMMValidator(n_states=n_states)

    @property
    def maxiter(self):
        """EM maximum number of iterations."""
        return self._maxiter

    @maxiter.setter
    def maxiter(self, maxiter: int):

        if not isinstance(maxiter, Number) or maxiter != int(maxiter) or maxiter <= 0:
            raise ValueError(
                f"``maxiter`` must be a strictly positive integer. {maxiter} provided."
            )
        self._maxiter = int(maxiter)

    @property
    def tol(self):
        """Tolerance for the EM algorithm convergence criterion.

        The algorithm stops when the absolute change in log-likelihood between
        consecutive iterations falls below this threshold:
        |log_likelihood_current - log_likelihood_previous| < tol

        Returns
        -------
            float: Convergence tolerance value.
        """
        return self._tol

    @tol.setter
    def tol(self, tol: float):

        if not isinstance(tol, Number) or tol <= 0:
            raise ValueError(
                f"``tol`` must be a strictly positive float. {tol} provided."
            )
        self._tol = float(tol)

    @property
    def initialization_funcs(self):
        """Dictionary of initialization functions for model parameters."""
        return self._initialization_funcs

    @initialization_funcs.setter
    def initialization_funcs(self, initialization_funcs: INITIALIZATION_FN_DICT):
        self._initialization_funcs = _resolve_init_funcs_registry(initialization_funcs)

    @property
    def dirichlet_prior_alphas_init_prob(self) -> jnp.ndarray | None:
        """Alpha parameters of the Dirichlet prior over the initial probabilities of HMM states.

        If ``None``, a flat prior is assumed.
        """
        return self._dirichlet_prior_alphas_init_prob

    @dirichlet_prior_alphas_init_prob.setter
    def dirichlet_prior_alphas_init_prob(self, value: jnp.ndarray | None):
        self._dirichlet_prior_alphas_init_prob = resolve_dirichlet_priors(
            value, (self._n_states,)
        )

    @property
    def dirichlet_prior_alphas_transition(self) -> jnp.ndarray | None:
        """Alpha parameters of the Dirichlet prior over the initial probabilities of HMM states.

        If ``None``, a flat prior is assumed.
        """
        return self._dirichlet_prior_alphas_transition

    @dirichlet_prior_alphas_transition.setter
    def dirichlet_prior_alphas_transition(self, value: jnp.ndarray | None):
        self._dirichlet_prior_alphas_transition = resolve_dirichlet_priors(
            value, (self._n_states, self._n_states)
        )

    @property
    def observation_model(self) -> obs.Observations:
        """Getter for the ``observation_model`` attribute."""
        return self._observation_model

    @observation_model.setter
    def observation_model(self, observation: obs.Observations):
        if isinstance(observation, str):
            self._observation_model = instantiate_observation_model(observation)
            return
        # check that the model has the required attributes
        # and that the attribute can be called
        obs.check_observation_model(observation)
        self._observation_model = observation

    @property
    def inverse_link_function(self):
        """Getter for the inverse link function for the model."""
        return self._inverse_link_function

    @inverse_link_function.setter
    def inverse_link_function(self, inverse_link_function: Callable):
        """Setter for the inverse link function for the model."""
        self._inverse_link_function = resolve_inverse_link_function(
            inverse_link_function, self._observation_model
        )

    @property
    def seed(self):
        """Random seed as a jax PRNG key."""
        return self._seed

    @seed.setter
    def seed(self, value):
        value = jnp.asarray(value)
        # Validate it's a JAX PRNG key
        if value.shape != (2,) or value.dtype != jnp.uint32:
            raise TypeError(
                f"seed must be a JAX PRNG key (jax.random.PRNGKey). "
                f"Got {type(value)} with shape {getattr(value, 'shape', 'N/A')}"
            )
        self._seed = value

    def _model_specific_initialization(
        self,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
    ) -> GLMHMMParams:
        """GLM-HMM initialization."""
        user_params = glm_hmm_initialization(self._n_states, X, y, self._seed)

        # check if registry uses NeMoS init funcs
        is_nemos_init = _is_native_init_registry(self._initialization_funcs)
        if is_nemos_init:
            # skip validation and just cast
            return self._validator.to_model_params(user_params)

        # params casting with validation
        model_params = self._validator.validate_and_cast_params(user_params)
        self._validator.validate_consistency(model_params, X=X, y=y)
        return model_params

    def _initialize_optimization_and_state(
        self,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        init_params: GLMHMMParams,
    ) -> SolverState:
        """Initialize the EM functions."""
        pass

    def fit(
        self,
        X: DESIGN_INPUT_TYPE,
        y: Union[NDArray, jnp.ndarray, nap.Tsd],
        init_params: Optional[GLMHMMUserParams] = None,
    ) -> "GLMHMM":
        """Fit the GLM-HMM model to the data."""
        pass

    def _estimate_resid_degrees_of_freedom(
        self, X: DESIGN_INPUT_TYPE, n_samples: Optional[int] = None
    ):
        """
        Estimate the degrees of freedom of the residuals.

        Parameters
        ----------
        self :
            A fitted GLM model.
        X :
            The design matrix.
        n_samples :
            The number of samples observed. If not provided, n_samples is set to ``X.shape[0]``. If the fit is
            batched, the n_samples could be larger than ``X.shape[0]``.

        Returns
        -------
        :
            An estimate of the degrees of freedom of the residuals.
        """
        pass

    def score(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: ArrayLike,
        score_type: Literal[
            "log-likelihood", "pseudo-r2-McFadden", "pseudo-r2-Cohen"
        ] = "log-likelihood",
        aggregate_sample_scores: Callable = jnp.mean,
    ) -> jnp.ndarray:
        """Compute the model score."""
        pass

    def simulate(
        self,
        random_key: jax.Array,
        feedforward_input: DESIGN_INPUT_TYPE,
    ) -> Tuple[jnp.ndarray, jnp.ndarray] | Tuple[nap.Tsd, nap.Tsd]:
        """Simulate spikes from the model, returns neural activity and states."""
        pass

    def smooth_proba(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: NDArray,
    ) -> jnp.ndarray | nap.TsdFrame:
        """Compute the smoothing posteriors over-states."""
        pass

    def filter_proba(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: NDArray,
    ):
        """Compute the filtering posteriors over-states."""
        pass

    def decode_state(
        self, X: Union[DESIGN_INPUT_TYPE, ArrayLike], y: ArrayLike
    ) -> jnp.ndarray | nap.TsdFrame:
        """Compute the most likely states over samples."""
        pass

    def _compute_loss(
        self,
        params: GLMHMMParams,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        """Loss function (expected HMM log-likelihood) without validation."""
        pass

    def save_params(
        self,
        filename: Union[str, Path],
    ):
        """Save model params."""
        pass

    # SVRG specific optimization not available.
    def _get_optimal_solver_params_config(self):
        """No optimal parameters known for SVRG in HMMGLM."""
        return None, None, None

    def _get_model_params(self) -> GLMHMMParams:
        pass

    def _set_model_params(self, params: GLMHMMParams):
        pass

    def update(
        self,
        params: Tuple[jnp.ndarray, jnp.ndarray],
        opt_state: NamedTuple,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        *args,
        n_samples: Optional[int] = None,
        **kwargs,
    ) -> StepResult:
        """Run a single update step of the jaxopt solver."""
        pass

    def __repr__(self) -> str:
        """Hierarchical repr for the GLMHMM class."""
        return format_repr(
            self, multiline=True, use_name_keys=["inverse_link_function"]
        )
