"""API for the GLM-HMM model."""

from pathlib import Path
from typing import Any, Callable, Literal, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import pynapple as nap
from numpy.typing import ArrayLike, NDArray

from .. import observation_models as obs
from .._observation_model_builder import instantiate_observation_model
from ..hmm.hmm import BaseHMM
from ..inverse_link_function_utils import resolve_inverse_link_function
from ..observation_models import Observations
from ..regularizer import Regularizer
from ..type_casting import support_pynapple
from ..typing import (
    DESIGN_INPUT_TYPE,
    ModelParamsT,
    SolverState,
    StepResult,
)
from ..utils import format_repr
from .initialize_parameters import (
    DEFAULT_INIT_FUNCTIONS_GLMHMM,
    GLMHMM_INITIALIZATION_FN_DICT,
    setup_glm_hmm_initialization,
)
from .params import GLMHMMParams, GLMHMMUserParams
from .validation import GLMHMMValidator


def _check_state_format(state_format: str) -> None:
    """Validate state_format parameter.

    Parameters
    ----------
    state_format :
        Format for state output, must be "one-hot" or "index".

    Raises
    ------
    ValueError
        If state_format is not "one-hot" or "index".
    """
    valid_formats = ("one-hot", "index")
    if state_format not in valid_formats:
        raise ValueError(
            f"Invalid state_format '{state_format}'. "
            f"Must be one of {valid_formats}."
        )


class GLMHMM(BaseHMM[GLMHMMUserParams, GLMHMMParams, GLMHMM_INITIALIZATION_FN_DICT]):
    r"""Generalized Linear Model with Hidden Markov Model (GLM-HMM).

    This model combines a Generalized Linear Model (GLM) with a Hidden Markov Model (HMM) to capture
    state-dependent relationships between predictors and neural or behavioral responses. The GLM-HMM
    is suitable for modeling time series data where the relationship between inputs and outputs
    varies according to an underlying latent state that evolves over time following Markovian dynamics.

    The model assumes that at each time step, the system is in one of ``n_states`` discrete hidden states.
    Each state has its own GLM parameters (coefficients and intercept), and transitions between states
    are governed by a transition probability matrix. The model is fitted using the Expectation-Maximization
    (EM) algorithm.

    Below is a table of the default inverse link function for the available observation models.

    +---------------------+---------------------------------+
    | Observation Model   | Default Inverse Link Function   |
    +=====================+=================================+
    | Poisson             | :math:`e^x`                     |
    +---------------------+---------------------------------+
    | Gamma               | :math:`1/x`                     |
    +---------------------+---------------------------------+
    | Bernoulli           | :math:`1 / (1 + e^{-x})`        |
    +---------------------+---------------------------------+
    | NegativeBinomial    | :math:`e^x`                     |
    +---------------------+---------------------------------+
    | Gaussian            | :math:`x`                       |
    +---------------------+---------------------------------+

    Below is a table listing the default and available solvers for each regularizer.

    +---------------+------------------+-------------------------------------------------------------+
    | Regularizer   | Default Solver   | Available Solvers                                           |
    +===============+==================+=============================================================+
    | UnRegularized | GradientDescent  | GradientDescent, BFGS, LBFGS, NonlinearCG, ProximalGradient |
    +---------------+------------------+-------------------------------------------------------------+
    | Ridge         | GradientDescent  | GradientDescent, BFGS, LBFGS, NonlinearCG, ProximalGradient |
    +---------------+------------------+-------------------------------------------------------------+
    | Lasso         | ProximalGradient | ProximalGradient                                            |
    +---------------+------------------+-------------------------------------------------------------+
    | ElasticNet    | ProximalGradient | ProximalGradient                                            |
    +---------------+------------------+-------------------------------------------------------------+
    | GroupLasso    | ProximalGradient | ProximalGradient                                            |
    +---------------+------------------+-------------------------------------------------------------+

    Parameters
    ----------
    n_states :
        The number of hidden states in the HMM. Must be a positive integer.
    observation_model :
        Observation model to use. The model describes the distribution of the response variable.
        Default is the Bernoulli model. Alternatives are "Poisson", "Gamma", "NegativeBinomial",
        and "Gaussian".
    inverse_link_function :
        A function that maps the linear combination of predictors into a rate or probability.
        The default depends on the observation model, see the table above.
    regularizer :
        Regularization to use for GLM parameter optimization. Defines the regularization scheme
        and related parameters. Default is UnRegularized regression.
    regularizer_strength :
        Typically a float. Default is None. Sets the regularizer strength for the GLM coefficients.
        If a user does not pass a value, and it is needed for regularization,
        a warning will be raised and the strength will default to 1.0.
    dirichlet_initial_proba :
        Alpha parameters for the Dirichlet prior over the initial state probabilities.
        Shape ``(n_states,)``. If None, a flat (uninformative) prior is assumed.
    dirichlet_transition_proba :
        Alpha parameters for the Dirichlet prior over the transition probabilities.
        Shape ``(n_states, n_states)``. If None, a flat (uninformative) prior is assumed.
    solver_name :
        Solver to use for GLM optimization within the M-step. Defines the optimization scheme
        and related parameters. The solver must be an appropriate match for the chosen regularizer.
        Default is None. If no solver specified, one will be chosen based on the regularizer.
        See the table above for regularizer/optimizer pairings.
    solver_kwargs :
        Optional dictionary for keyword arguments that are passed to the solver when instantiated.
        E.g., stepsize, tol, acceleration, etc.
    maxiter :
        Maximum number of EM iterations. Default is 1000.
    tol :
        Convergence tolerance for the EM algorithm. The algorithm stops when the absolute change
        in log-likelihood between consecutive iterations falls below this threshold. Default is 1e-8.
    seed :
        JAX PRNG key for random number generation during initialization. Default is
        ``jax.random.PRNGKey(123)``.
    initialization_funcs : dict, optional
        TODO: Update description

    Attributes
    ----------
    transition_prob_ :
        Transition probability matrix of shape ``(n_states, n_states)``. Entry ``[i, j]`` represents
        the probability of transitioning from state ``i`` to state ``j``.
    initial_prob_ :
        Initial state probability vector of shape ``(n_states,)``. Entry ``[i]`` represents
        the probability of starting in state ``i``.
    coef_ :
        GLM coefficients for each state, shape ``(n_features, n_states)``.
    intercept_ :
        GLM intercepts (bias terms) for each state, shape ``(n_states,)``.
    solver_state_ :
        State of the solver after fitting. May include details like optimization error.
    scale_ :
        Scale parameter for the observation model, shape ``(n_states,)``.
    dof_resid_ :
        Degrees of freedom for the residuals.

    Notes
    -----
    To bypass the initialization functions entirely and provide parameter arrays
    directly, pass them to the ``fit()`` method::

        model.fit(X, y, init_params=my_params)

    Raises
    ------
    TypeError
        If ``n_states`` is not a positive integer.
    TypeError
        If provided ``regularizer`` or ``observation_model`` are not valid.
    TypeError
        If ``seed`` is not a valid JAX PRNG key.
    KeyError
        If ``initialization_funcs`` contains invalid keys (not one of the four
        valid initialization function names).
    ValueError
        If ``initialization_kwargs`` contains keyword arguments that don't match
        the signature of the corresponding initialization function.
    ValueError
        If ``maxiter`` is not a positive integer.
    ValueError
        If ``tol`` is not a positive float.
    """

    _validator_class = GLMHMMValidator
    _default_init_dict = DEFAULT_INIT_FUNCTIONS_GLMHMM

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
            Any
        ] = None,  # this is used to regularize GLM coef.
        # prior to regularize init prob and transition
        dirichlet_initial_proba: Union[jnp.ndarray, None] = None,  # (n_state, )
        dirichlet_transition_proba: Union[
            jnp.ndarray | None
        ] = None,  # (n_state, n_state)
        solver_name: str = None,
        solver_kwargs: Optional[dict] = None,
        maxiter: int = 1000,
        tol: float = 1e-8,
        seed=jax.random.PRNGKey(123),
        initialization_funcs: Optional[GLMHMM_INITIALIZATION_FN_DICT] = None,
    ):
        super().__init__(
            n_states=n_states,
            dirichlet_initial_proba=dirichlet_initial_proba,
            dirichlet_transition_proba=dirichlet_transition_proba,
            regularizer=regularizer,
            regularizer_strength=regularizer_strength,
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
            maxiter=maxiter,
            tol=tol,
            seed=seed,
            initialization_funcs=initialization_funcs,
        )
        self.observation_model = observation_model
        self.inverse_link_function = inverse_link_function

        # fit attributes
        self.coef_: jnp.ndarray | None = None
        self.intercept_: jnp.ndarray | None = None
        self.solver_state_: NamedTuple | None = None
        self.scale_: jnp.ndarray | None = None
        self.dof_resid_: int | None = None

    def _log_likelihood(
        self, params: GLMHMMParams, X: DESIGN_INPUT_TYPE, y: ArrayLike
    ) -> jnp.ndarray:
        """Compute the log-likelihood of the data given the model parameters."""
        pass

    def setup(
        self,
        initial_proba_init: Optional[str | Callable] = None,
        initial_proba_init_kwargs: Optional[dict] = None,
        transition_proba_init: Optional[str | Callable] = None,
        transition_proba_init_kwargs: Optional[dict] = None,
        glm_params_init: Optional[str | Callable] = None,
        glm_params_init_kwargs: Optional[dict] = None,
        scale_init: Optional[str | Callable] = None,
        scale_init_kwargs: Optional[dict] = None,
    ):
        """Set the initialization functions."""
        self._initialization_funcs = setup_glm_hmm_initialization(
            initial_proba_init=initial_proba_init,
            initial_proba_init_kwargs=initial_proba_init_kwargs,
            transition_proba_init=transition_proba_init,
            transition_proba_init_kwargs=transition_proba_init_kwargs,
            glm_params_init=glm_params_init,
            glm_params_init_kwargs=glm_params_init_kwargs,
            scale_init=scale_init,
            scale_init_kwargs=scale_init_kwargs,
            init_funcs=self._initialization_funcs,
        )

    def set_params(self, **kwargs):
        """Set model parameters, ensuring initialization functions are set before their kwargs.

        This override ensures that when both ``initialization_funcs`` and
        ``initialization_kwargs`` are provided, the functions are set first so
        that kwargs are validated against the new functions, not the old ones.
        """
        if "initialization_funcs" in kwargs and "initialization_kwargs" in kwargs:
            super().set_params(initialization_funcs=kwargs.pop("initialization_funcs"))
        return super().set_params(**kwargs)

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

    def _check_model_is_fit(self):
        """Ensure the instance has been fitted."""
        flat_params = [
            self.coef_,
            self.intercept_,
            self.scale_,
        ]
        is_missing = [x is None for x in flat_params]
        if any(is_missing):
            param_labels = [
                "coef_",
                "intercept_",
                "scale_",
            ]
            missing_params = [
                p for p, missing in zip(param_labels, is_missing) if missing
            ]
            raise ValueError(
                "This GLMHMM instance is not fitted yet. The following attributes are not set:"
                f" {missing_params}.\nPlease fit the GLM-HMM model first or "
                "set the missing attributes."
            )

    def _model_specific_initialization(
        self,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        is_new_session: jnp.ndarray,
    ) -> GLMHMMParams:
        """GLM-HMM initialization."""
        pass

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
        """Estimate the degrees of freedom of the residuals."""
        pass

    def score(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: ArrayLike,
        score_type: Literal[
            "log-likelihood", "pseudo-r2-McFadden", "pseudo-r2-Cohen"
        ] = "log-likelihood",
        null_model: Optional[Literal["constant", "glm"]] = None,
    ) -> jnp.ndarray:
        """Compute the model score."""
        pass

    @support_pynapple(conv_type="jax")
    def simulate(
        self,
        random_key: jax.Array,
        feedforward_input: DESIGN_INPUT_TYPE,
        state_format: Literal["one-hot", "index"] = "index",
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Simulate neural activity and hidden states from the model."""
        pass

    def smooth_proba(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: Union[NDArray, jnp.ndarray, nap.Tsd],
    ) -> jnp.ndarray | nap.TsdFrame:
        """Compute smoothing posterior probabilities over hidden states."""
        pass

    def filter_proba(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: Union[NDArray, jnp.ndarray, nap.Tsd],
    ) -> jnp.ndarray | nap.TsdFrame:
        """Compute filtering posterior probabilities over hidden states."""
        pass

    def decode_state(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: ArrayLike,
        state_format: Literal["one-hot", "index"] = "one-hot",
    ) -> jnp.ndarray | nap.TsdFrame:
        """Compute the most likely hidden state sequence (Viterbi decoding)."""
        pass

    def save_params(
        self,
        filename: Union[str, Path],
    ):
        """Save model params."""
        pass

    # SVRG specific optimization not available.
    def _get_optimal_solver_params_config(self):
        """No optimal parameters known for SVRG in GLMHMM."""
        return None, None, None

    def _get_model_params(self) -> GLMHMMParams:
        return self._validator.to_model_params(
            (
                self.coef_,
                self.intercept_,
                self.scale_,
                self.initial_prob_,
                self.transition_prob_,
            )
        )

    def _set_model_params(self, params: GLMHMMParams):
        coef, intercept, scale, initial_prob, transition_prob = (
            self._validator.from_model_params(params)
        )
        self.coef_ = coef
        self.intercept_ = intercept
        self.scale_ = scale
        self.initial_prob_ = initial_prob
        self.transition_prob_ = transition_prob

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

    def _compute_loss(
        self,
        params: ModelParamsT,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        *args,
        **kwargs,
    ):
        pass

    def _initialize_optimizer_and_state(
        self,
        init_params: ModelParamsT,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
    ) -> SolverState:
        """Initialize the optimizer and state of the model."""
        pass

    def _model_params_initialization(self, X, y, is_new_session):
        """Initialize the model parameters."""
        pass
