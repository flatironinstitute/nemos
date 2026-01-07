"""API for the GLM-HMM model."""

import warnings
from numbers import Number
from pathlib import Path
from typing import Any, Callable, Literal, NamedTuple, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import pynapple as nap
from numpy.typing import ArrayLike, NDArray

from .. import observation_models as obs
from .. import tree_utils
from .._observation_model_builder import instantiate_observation_model
from ..base_regressor import BaseRegressor
from ..inverse_link_function_utils import resolve_inverse_link_function
from ..observation_models import Observations
from ..pytrees import FeaturePytree
from ..regularizer import GroupLasso, Lasso, Regularizer, Ridge
from ..type_casting import cast_to_jax, is_pynapple_tsd, support_pynapple
from ..typing import (
    DESIGN_INPUT_TYPE,
    SolverState,
    StepResult,
)
from ..utils import format_repr
from . import forward_backward
from .algorithm_configs import (
    get_analytical_scale_update,
    prepare_estep_log_likelihood,
    prepare_mstep_nll_objective_param,
    prepare_mstep_nll_objective_scale,
)
from .expectation_maximization import (
    GLMHMMState,
    em_glm_hmm,
    em_step,
)
from .initialize_parameters import (
    INITIALIZATION_FN_DICT,
    _is_native_init_registry,
    _resolve_dirichlet_priors,
    _resolve_init_funcs_registry,
    _resolve_init_kwargs_registry,
    glm_hmm_initialization,
)
from .params import GLMHMMParams, GLMHMMUserParams
from .validation import GLMHMMValidator


def compute_is_new_session(
    time: NDArray | jnp.ndarray,
    start: NDArray | jnp.ndarray,
    is_nan: Optional[NDArray | jnp.ndarray] = None,
) -> jnp.ndarray:
    """Compute indicator vector marking the start of new sessions.

    This function identifies session boundaries in time-series data by marking positions
    where new epochs begin or where data resumes after NaN values. When NaN values are
    present, the first valid sample immediately following each NaN is marked as a new
    session start.

    Parameters
    ----------
    time :
        Timestamps for each sample in the time series, shape ``(n_time_points,)``.
        Must be monotonically increasing.
    start :
        Start times marking the beginning of each epoch or session, shape ``(n_epochs,)``.
        Each value should correspond to a timestamp in ``time``.
    is_nan :
        Boolean array indicating NaN positions, shape ``(n_time_points,)``.
        If provided, positions immediately after NaNs will be marked as new session starts.

    Returns
    -------
    is_new_session :
        Binary indicator array of shape ``(n_time_points,)`` where 1 indicates the start
        of a new session and 0 otherwise.

    Notes
    -----
    The function marks positions as new sessions in two cases:
    1. Positions matching epoch start times (from ``start`` parameter)
    2. Positions immediately following NaN values (when ``is_nan`` is provided)

    This ensures that after dropping NaN values, session boundaries are preserved.
    """
    is_new_session = (
        jax.numpy.zeros_like(time).at[jax.numpy.searchsorted(time, start)].set(1)
    )
    if is_nan is not None:
        # set the first element after nan as new session beginning
        is_new_session = is_new_session.at[1:].set(
            jnp.where(is_nan[:-1], 1, is_new_session[1:])
        )
    return is_new_session


class GLMHMM(BaseRegressor[GLMHMMUserParams, GLMHMMParams]):
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
    dirichlet_prior_alphas_init_prob :
        Alpha parameters for the Dirichlet prior over the initial state probabilities.
        Shape ``(n_states,)``. If None, a flat (uninformative) prior is assumed.
    dirichlet_prior_alphas_transition :
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
    initialization_funcs : dict, optional
        Dictionary specifying custom initialization functions for model components.
        Keys must be a subset of: ``"glm_params_init"``, ``"scale_init"``,
        ``"initial_proba_init"``, ``"transition_proba_init"``. Values can be:

        - A callable with the appropriate signature (see :mod:`nemos.glm_hmm.initialize_parameters`)
        - A string alias: ``"random"`` (GLM params), ``"constant"`` (scale),
          ``"sticky"``/``"uniform"`` (transition), ``"uniform"`` (initial)

        Missing keys use NeMoS defaults:

        - ``"glm_params_init"``: ``"random"`` - small random coefficients, mean-rate intercept
        - ``"scale_init"``: ``"constant"`` - scale initialized to 1.0
        - ``"initial_proba_init"``: ``"uniform"`` - equal probability for all states
        - ``"transition_proba_init"``: ``"sticky"`` - high self-transition probability (0.95)

        If None, all defaults are used.
    initialization_kwargs : dict, optional
        Keyword arguments passed to each initialization function. Keys must match
        those in ``initialization_funcs``. For example::

            initialization_kwargs={
                "transition_proba_init": {"prob_stay": 0.99},
                "glm_params_init": {"std_dev": 0.01},
            }

        Each kwarg is validated against the corresponding function's signature.
        If an initialization function is changed via ``initialization_funcs``, its
        kwargs are reset to defaults (with a warning) since the new function may
        have a different signature.
    maxiter :
        Maximum number of EM iterations. Default is 1000.
    tol :
        Convergence tolerance for the EM algorithm. The algorithm stops when the absolute change
        in log-likelihood between consecutive iterations falls below this threshold. Default is 1e-8.
    seed :
        JAX PRNG key for random number generation during initialization. Default is
        ``jax.random.PRNGKey(123)``.

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

    Examples
    --------
    **Generate Synthetic Data**

    Generate synthetic data with state-dependent responses:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>> import nemos as nmo
    >>> np.random.seed(123)
    >>> n_samples, n_features, n_states = 500, 3, 2
    >>> X = np.random.normal(size=(n_samples, n_features))
    >>> # State-dependent weights
    >>> weights = np.array([[[0.5], [-0.3], [0.2]], [[-0.5], [0.4], [-0.1]]])
    >>> # Simulate state sequence
    >>> states = np.random.choice(n_states, size=n_samples)
    >>> # Generate responses based on current state
    >>> logits = np.array([X[t] @ weights[states[t]] for t in range(n_samples)]).squeeze()
    >>> y = np.random.binomial(1, 1 / (1 + np.exp(-logits)))

    **Fit a GLM-HMM**

    Fit the model to the synthetic data:

    >>> model = nmo.glm.GLMHMM(n_states=2, observation_model="Bernoulli")
    >>> model = model.fit(X, y)
    >>> model.coef_.shape
    (3, 2)

    **Customize the Observation Model**

    Use a Poisson observation model for count data:

    >>> model = nmo.glm.GLMHMM(n_states=3, observation_model="Poisson")
    >>> model.observation_model
    PoissonObservations()

    **Use Regularization**

    Fit with Ridge regularization on the GLM coefficients:

    >>> model = nmo.glm.GLMHMM(
    ...     n_states=2,
    ...     observation_model="Bernoulli",
    ...     regularizer="Ridge",
    ...     regularizer_strength=0.1
    ... )
    >>> model.regularizer
    Ridge()
    """

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
        dirichlet_prior_alphas_init_prob: Union[
            jnp.ndarray, None
        ] = None,  # (n_state, )
        dirichlet_prior_alphas_transition: Union[
            jnp.ndarray | None
        ] = None,  # (n_state, n_state)
        solver_name: str = None,
        solver_kwargs: Optional[dict] = None,
        initialization_funcs: Optional[INITIALIZATION_FN_DICT] = None,
        initialization_kwargs: Optional[dict] = None,
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
        self.initialization_kwargs = initialization_kwargs

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

    @property
    def n_states(self) -> int:
        """Number of hidden states of the HMM."""
        return self._n_states

    @n_states.setter
    def n_states(self, n_states: int):
        # quick sanity check and assignment
        if isinstance(n_states, int) and n_states > 0:
            self._n_states = n_states
            self._validator: GLMHMMValidator = GLMHMMValidator(n_states=n_states)
            return

        # further checks for other valid numeric types (like non-negative float with no-decimals)
        if not isinstance(n_states, Number):
            raise TypeError(
                f"n_states must be a positive integer. "
                f"n_states is of type ``{type(n_states)}`` instead."
            )

        # provided a non-integer number (check that has no decimals)
        int_n_states = int(n_states)
        if int_n_states != n_states:
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
        current_initialization_funcs = getattr(self, "_initialization_funcs", None)
        initialization_funcs = _resolve_init_funcs_registry(initialization_funcs)

        # at initialization
        if current_initialization_funcs is None:
            self._initialization_funcs = initialization_funcs
            return

        # invalidate the kwargs if old func != new func
        for func_name, func_new in initialization_funcs.items():
            if func_name not in self._initialization_kwargs:
                continue
            # after init current_initialization_funcs should have all
            # the functions. If this assumption is not match, the line
            # below would fail immediately, which is what we want for
            # maintainability.
            func_old = current_initialization_funcs[func_name]
            kwargs = self._initialization_kwargs.get(func_name, None)
            if func_new != func_old and (kwargs is not None and kwargs != {}):
                warnings.warn(
                    f"The initialization function for '{func_name}' changed. "
                    f"Resetting its kwargs to defaults since the previous kwargs "
                    f"may not be compatible with the new function.",
                    UserWarning,
                    stacklevel=2,
                )
                self._initialization_kwargs[func_name] = {}

        self._initialization_funcs = initialization_funcs

    @property
    def initialization_kwargs(self):
        """Dictionary of initialization kwargs for model parameters.

        Dictionary of kwargs that will be provided as keyword argument to
        the initialization functions in ``self.initialization_funcs``.
        """
        return self._initialization_kwargs

    def set_params(self, **kwargs):
        """Set model parameters, ensuring initialization functions are set before their kwargs.

        This override ensures that when both ``initialization_funcs`` and
        ``initialization_kwargs`` are provided, the functions are set first so
        that kwargs are validated against the new functions, not the old ones.
        """
        if "initialization_funcs" in kwargs and "initialization_kwargs" in kwargs:
            super().set_params(initialization_funcs=kwargs.pop("initialization_funcs"))
        return super().set_params(**kwargs)

    @initialization_kwargs.setter
    def initialization_kwargs(self, initialization_kwargs: dict | None):
        self._initialization_kwargs = _resolve_init_kwargs_registry(
            initialization_kwargs, self._initialization_funcs
        )

    @property
    def dirichlet_prior_alphas_init_prob(self) -> jnp.ndarray | None:
        """Alpha parameters of the Dirichlet prior over the initial probabilities of HMM states.

        If ``None``, a flat prior is assumed.
        """
        return self._dirichlet_prior_alphas_init_prob

    @dirichlet_prior_alphas_init_prob.setter
    def dirichlet_prior_alphas_init_prob(self, value: jnp.ndarray | None):
        self._dirichlet_prior_alphas_init_prob = _resolve_dirichlet_priors(
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
        self._dirichlet_prior_alphas_transition = _resolve_dirichlet_priors(
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
        try:
            value = jnp.asarray(value)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"seed must be a JAX PRNG key (jax.random.PRNGKey). "
                f"Got {type(value).__name__} instead."
            ) from e
        # Validate it's a JAX PRNG key
        if value.shape != (2,) or value.dtype != jnp.uint32:
            raise TypeError(
                f"seed must be a JAX PRNG key (jax.random.PRNGKey). "
                f"Got {type(value).__name__} with shape {getattr(value, 'shape', 'N/A')}"
            )
        self._seed = value

    def _check_is_fit(self):
        """Ensure the instance has been fitted."""
        flat_params = [
            self.coef_,
            self.intercept_,
            self.scale_,
            self.initial_prob_,
            self.transition_prob_,
        ]
        is_missing = [x is None for x in flat_params]
        if any(is_missing):
            param_labels = [
                "coef_",
                "intercept_",
                "scale_",
                "initial_prob_",
                "transition_prob_",
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
    ) -> GLMHMMParams:
        """GLM-HMM initialization."""
        user_params = glm_hmm_initialization(
            self._n_states,
            X,
            y,
            inverse_link_function=self._inverse_link_function,
            random_key=self._seed,
            initialization_funcs=self._initialization_funcs,
            initialization_kwargs=self._initialization_kwargs,
        )

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
        # glm params m-step setup
        is_population = y.ndim > 1
        log_likelihood = prepare_estep_log_likelihood(
            is_population_glm=is_population,
            observation_model=self.observation_model,
        )
        objective = prepare_mstep_nll_objective_param(
            is_population_glm=is_population,
            observation_model=self._observation_model,
            inverse_link_function=self._inverse_link_function,
        )
        _, _, glm_params_update_fn = self._instantiate_solver(
            objective, init_params.glm_params
        )

        scale_update_fn = get_analytical_scale_update(
            is_population_glm=is_population, observation_model=self._observation_model
        )
        if scale_update_fn is None:
            objective_scale = prepare_mstep_nll_objective_scale(
                is_population_glm=is_population,
                observation_model=self._observation_model,
            )
            _, _, scale_update_fn = self._instantiate_solver(
                objective_scale, init_params.glm_scale
            )

        # cannot wrap is_new_session, that's to be calculated at each update form the provided X and y.
        # for consistency, do not make a partial of that argument in run as well.
        self._optimization_run = eqx.Partial(
            em_glm_hmm,
            inverse_link_function=self.inverse_link_function,
            log_likelihood_func=log_likelihood,
            m_step_fn_glm_params=glm_params_update_fn,
            m_step_fn_glm_scale=scale_update_fn,
            maxiter=self.maxiter,
            tol=self.tol,
        )

        self._optimization_update = eqx.Partial(
            em_step,
            inverse_link_function=self.inverse_link_function,
            log_likelihood_func=log_likelihood,
            m_step_fn_glm_params=glm_params_update_fn,
            m_step_fn_glm_scale=scale_update_fn,
        )

        def init_state_fn(*args, **kwargs) -> SolverState:
            state = GLMHMMState(
                data_log_likelihood=-jnp.array(jnp.inf),
                previous_data_log_likelihood=-jnp.array(jnp.inf),
                log_likelihood_history=jnp.full(self.maxiter, jnp.nan),
                iterations=0,
            )
            return state

        self._optimization_init_state = init_state_fn
        return init_state_fn()

    @staticmethod
    def _get_is_new_session(
        X: DESIGN_INPUT_TYPE, y: ArrayLike | nap.Tsd | nap.TsdFrame
    ) -> jnp.ndarray | None:
        """Compute session boundary indicators for GLM-HMM time-series data.

        Identifies session boundaries by detecting epoch starts and gaps in the data
        (represented by NaN values in either predictors or response). This is essential
        for GLM-HMM models to properly segment time series data and reset the hidden
        state between discontinuous recordings.

        Parameters
        ----------
        X :
            Design matrix or predictor time series. Can be a pynapple Tsd/TsdFrame or
            array-like of shape ``(n_time_points, n_features)``.
        y :
            Response variable time series of shape ``(n_time_points,)`` or
            ``(n_time_points, n_neurons)``.

        Returns
        -------
        is_new_session :
            Binary indicator array of shape ``(n_time_points,)`` marking session starts
            with 1s. Returns None if unable to compute session boundaries.

        Notes
        -----
        Session boundaries are identified from:
        - Epoch start times (when using pynapple Tsd objects with time_support)
        - Positions immediately following NaN values in either X or y

        When both X and y are pynapple objects, y's time information takes precedence.

        For non-pynapple inputs, a default session structure is initialized based on
        the length of y.

        See Also
        --------
        compute_is_new_session : Core function for computing session indicators.
        """
        # compute the nan location along the sample axis
        nan_y = jnp.any(jnp.isnan(jnp.asarray(y)).reshape(y.shape[0], -1), axis=1)
        nan_x = jnp.any(jnp.isnan(jnp.asarray(X)).reshape(X.shape[0], -1), axis=1)
        combined_nans = nan_y | nan_x

        # define new session array
        if is_pynapple_tsd(y):
            is_new_session = compute_is_new_session(
                y.t, y.time_support.start, combined_nans
            )
        elif is_pynapple_tsd(X):
            is_new_session = compute_is_new_session(
                X.t, X.time_support.start, combined_nans
            )
        else:
            is_new_session = compute_is_new_session(
                jnp.arange(X.shape[0]), jnp.array([0.0]), combined_nans
            )
        return is_new_session

    def fit(
        self,
        X: DESIGN_INPUT_TYPE,
        y: Union[NDArray, jnp.ndarray, nap.Tsd],
        init_params: Optional[GLMHMMUserParams] = None,
    ) -> "GLMHMM":
        """Fit the GLM-HMM model to the data."""
        self._validator.validate_inputs(X=X, y=y)
        is_new_session = self._get_is_new_session(X, y)

        # validate the inputs & initialize solver
        # initialize params if no params are provided
        if init_params is None:
            init_params = self._model_specific_initialization(X, y)
        else:
            init_params = self._validator.validate_and_cast_params(init_params)
            self._validator.validate_consistency(init_params, X=X, y=y)

        self._validator.feature_mask_consistency(
            getattr(self, "_feature_mask", None), init_params
        )

        # filter for non-nans, grab data if needed
        data, y, is_new_session = self._preprocess_inputs(X, y, is_new_session)

        # make sure is_new_session starts with a 1
        is_new_session = is_new_session.at[0].set(True)

        # set up optimization
        self._initialize_optimization_and_state(data, y, init_params)

        # run EM
        (
            fit_params,
            self.solver_state_,
        ) = self._optimization_run(
            init_params, X=data, y=y, is_new_session=is_new_session
        )

        if self.solver_state_.iterations == self.maxiter:
            warnings.warn(
                "The fit did not converge. "
                "Consider the following:"
                "\n1) Enable float64 with ``jax.config.update('jax_enable_x64', True)``"
                "\n2) Increase the ``maxiter`` parameter (max number of iterations of the EM) "
                "or increase the ``tol`` parameter (tolerance).",
                RuntimeWarning,
            )

        # assign fit attributes
        self._set_model_params(fit_params)
        self.dof_resid_ = self._estimate_resid_degrees_of_freedom(data)
        return self

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
        # Convert a pytree to a design-matrix with pytrees
        X = jnp.hstack(jax.tree_util.tree_leaves(X))

        if n_samples is None:
            n_samples = X.shape[0]
        else:
            if not isinstance(n_samples, int):
                raise TypeError(
                    "`n_samples` must either `None` or of type `int`. Type {type(n_sample)} provided "
                    "instead!"
                )

        params = self._get_model_params()
        coef = params.glm_params.coef
        if coef.ndim == 3:
            n_neurons = coef.shape[1]
        else:
            n_neurons = 1

        dof_intercept_and_hmm = (
            self._n_states * n_neurons  # intercept
            + (
                self._n_states - 1
            )  # init prob (n values but sum to 1, so n-1 free values)
            + (self._n_states - 1) * self._n_states
        )  # transition n n-dim vectors that sum to 1

        # if the regularizer is lasso use the non-zero
        # coef as an estimate of the dof
        # see https://arxiv.org/abs/0712.0881
        if isinstance(self.regularizer, (GroupLasso, Lasso)):
            resid_dof = sum(
                tree_utils.pytree_map_and_reduce(
                    lambda x: ~jnp.isclose(x, jnp.zeros_like(x)),
                    lambda x: sum([jnp.sum(i, axis=0) for i in x]),
                    coef,
                )
            )
            return n_samples - resid_dof - dof_intercept_and_hmm
        elif isinstance(self.regularizer, Ridge):
            # for Ridge, use the tot parameters (X.shape[1] + intercept)
            return (
                n_samples - (X.shape[1] * self.n_states) - dof_intercept_and_hmm
            ) * jnp.ones(n_neurons)
        else:
            # for UnRegularized, use the rank
            rank = jnp.linalg.matrix_rank(X)
            return (n_samples - rank - dof_intercept_and_hmm) * jnp.ones(n_neurons)

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

    @support_pynapple(conv_type="jax")
    def _smooth_proba(
        self,
        params: GLMHMMParams,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: Union[NDArray, jnp.ndarray, nap.Tsd],
        is_new_session: jnp.ndarray,
    ) -> jnp.ndarray:
        # filter for non-nans, grab data if needed
        valid = tree_utils.get_valid_multitree(X, y)
        data, y, is_new_session = self._preprocess_inputs(X, y, is_new_session)

        # safe conversion to jax arrays of float
        params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, y.dtype), params)

        # make sure is_new_session starts with a 1
        is_new_session = is_new_session.at[0].set(True)

        # smooth with forward backward
        log_posteriors, _, _, _, _, _ = forward_backward(
            params=params,
            X=data,
            y=y,
            is_new_session=is_new_session,
            log_likelihood_func=prepare_estep_log_likelihood(
                y.ndim > 1, self.observation_model
            ),
            inverse_link_function=self._inverse_link_function,
        )
        proba = jnp.exp(log_posteriors)
        # renormalize (numerical precision due to exponentiation)
        proba /= proba.sum(axis=1, keepdims=True)
        # re-attach nans
        proba = jnp.full((valid.shape[0], proba.shape[1]), jnp.nan).at[valid].set(proba)
        return proba

    def _validate_and_prepare_inputs(self, X, y):
        """Validate and prepare inputs."""
        # check if the model was fit
        self._check_is_fit()
        params = self._get_model_params()

        # validate inputs
        self._validator.validate_inputs(X=X, y=y)
        self._validator.validate_consistency(params, X=X, y=y)

        # compute new session indicator
        is_new_session = self._get_is_new_session(X, y)
        return params, X, y, is_new_session

    def smooth_proba(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: Union[NDArray, jnp.ndarray, nap.Tsd],
    ) -> jnp.ndarray | nap.TsdFrame:
        """Compute smoothing posterior probabilities over hidden states.

        Computes the probability of being in each hidden state at each time point,
        conditioned on the entire observed sequence. This method uses the forward-backward
        algorithm to incorporate information from both past and future observations,
        providing optimal state estimates given all available data.

        The smoothing posteriors answer: "Given all observations, what is the probability
        that the system was in state k at time t?"

        Parameters
        ----------
        X
            Predictors, shape ``(n_time_points, n_features)``.
        y
            Observed neural activity, shape ``(n_time_points,)`` for single neuron or
            ``(n_time_points, n_neurons)`` for population.

        Returns
        -------
        posteriors
            Smoothing posterior probabilities, shape ``(n_time_points, n_states)``.
            Each row sums to 1 and represents the probability distribution over states
            at that time point.

        Raises
        ------
        ValueError
            If the model has not been fit (``fit()`` must be called first).
        ValueError
            If inputs contain NaN values in the middle of epochs (only boundary NaNs allowed).
        ValueError
            If X and y have inconsistent shapes or features.

        See Also
        --------
        filter_proba : Compute filtering posteriors (conditioned on past observations only).
        decode_state : Compute most likely state sequence (Viterbi decoding).

        Notes
        -----
        - Smoothing provides better state estimates than filtering because it uses all data
        - The algorithm properly handles session boundaries and NaN values at epoch borders

        Examples
        --------
        Fit a GLM-HMM and compute smoothing posteriors:

        >>> import numpy as np
        >>> import nemos as nmo
        >>> # Generate example data
        >>> np.random.seed(123)
        >>> X = np.random.randn(100, 5)  # 100 time points, 5 features
        >>> y = np.random.poisson(2, size=100)  # Poisson spike counts
        >>>
        >>> # Fit model with 3 hidden states
        >>> model = nmo.glm_hmm.GLMHMM(n_states=3, observation_model="Poisson")
        >>> model = model.fit(X, y)
        >>> # Compute smoothing posteriors
        >>> posteriors = model.smooth_proba(X, y)
        >>> print(posteriors.shape)
        (100, 3)
        >>> # Each row sums to 1
        >>> print(np.allclose(posteriors.sum(axis=1), 1.0))
        True

        Using with pynapple for time-series analysis:

        >>> import pynapple as nap
        >>> # Create time-indexed data
        >>> t = np.arange(100) * 0.01  # 10ms bins
        >>> X_tsd = nap.TsdFrame(t=t, d=X)
        >>> y_tsd = nap.Tsd(t=t, d=y)
        >>>
        >>> # Posteriors returned as TsdFrame
        >>> posteriors_tsd = model.smooth_proba(X_tsd, y_tsd)
        >>> print(type(posteriors_tsd))
        <class 'pynapple.core.time_series.TsdFrame'>
        """
        params, X, y, is_new_session = self._validate_and_prepare_inputs(X, y)
        return self._smooth_proba(params, X, y, is_new_session)

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
        # initialize saving dictionary
        fit_attrs = self._get_fit_state()
        fit_attrs.pop("solver_state_")
        string_attrs = ["inverse_link_function"]
        self._save_params(filename, fit_attrs, string_attrs)

    # SVRG specific optimization not available.
    def _get_optimal_solver_params_config(self):
        """No optimal parameters known for SVRG in HMMGLM."""
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
        is_new_session = self._get_is_new_session(X, y)

        # cast to jax and find non-nans
        X, y, is_new_session = tree_utils.drop_nans(X, y, is_new_session)
        X, y = cast_to_jax(lambda *x: x)(X, y)
        is_new_session = is_new_session.at[0].set(True)

        # grab the data
        data = X.data if isinstance(X, FeaturePytree) else X

        # wrap into GLM params, this assumes params are well-structured,
        # if initialization is done via `initialize_solver_and_state` it
        # should be fine
        params = self._validator.to_model_params(params)

        # perform a one-step update
        updated_params, updated_state = self._optimization_update(
            params, opt_state, data, y, *args, is_new_session=is_new_session, **kwargs
        )

        # store params and state
        self._set_model_params(updated_params)
        self.solver_state_ = updated_state

        # estimate the scale
        self.dof_resid_ = self._estimate_resid_degrees_of_freedom(
            X, n_samples=n_samples
        )
        return self._validator.from_model_params(updated_params), updated_state

    def __repr__(self) -> str:
        """Hierarchical repr for the GLMHMM class."""
        return format_repr(
            self, multiline=True, use_name_keys=["inverse_link_function"]
        )
