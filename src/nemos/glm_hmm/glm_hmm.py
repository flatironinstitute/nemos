"""API for the GLM-HMM model."""

import warnings
from pathlib import Path
from typing import Any, Callable, Literal, NamedTuple, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import pynapple as nap
from numpy.typing import ArrayLike, NDArray

from .. import observation_models as obs
from .._observation_model_builder import instantiate_observation_model
from ..hmm.expectation_maximization import EMState, em_hmm, em_step
from ..hmm.hmm import BaseHMM
from ..hmm.utils import initialize_is_new_session
from ..inverse_link_function_utils import resolve_inverse_link_function
from ..observation_models import Observations
from ..regularizer import GroupLasso, Lasso, Regularizer, Ridge
from ..tree_utils import pytree_map_and_reduce
from ..type_casting import support_pynapple
from ..typing import (
    DESIGN_INPUT_TYPE,
    ModelParamsT,
    SolverState,
    StepResult,
)
from ..utils import format_repr
from .algorithm_configs import prepare_estep_log_likelihood, prepare_mstep_update_fn
from .initialize_parameters import (
    DEFAULT_INIT_FUNCTIONS_GLMHMM,
    GLMHMM_INITIALIZATION_FN_DICT,
    generate_glm_hmm_initial_model_params,
    KMeansInitializerGLM,
    kmeans_glm_params_init,
    kmeans_scale_init,
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

        # please the type checkers
        self._validator: GLMHMMValidator = (
            self._validator
        )  # Why is this not automatically recognized?
        # cache the log-like
        self._log_like_cache = {}

    def _log_likelihood(
        self, params: GLMHMMParams, X: DESIGN_INPUT_TYPE, y: ArrayLike
    ) -> jnp.ndarray:
        """Compute the log-likelihood of the data given the model parameters.

        Use cached values to avoid unnecessary computations.
        """
        ll_func = self._log_like_cache.get(
            (y.ndim > 1, self._observation_model, self._inverse_link_function)
        )
        if ll_func is None:
            ll_func = prepare_estep_log_likelihood(
                y.ndim > 1, self._observation_model, self._inverse_link_function
            )
        return ll_func(params, X, y)

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

    def _kmeans_extra_kwargs(self) -> dict:
        return {"inverse_link_function": self.inverse_link_function}

    def _model_params_initialization(
        self,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        is_new_session: jnp.ndarray,
        random_key: jax.Array,
    ) -> Tuple[GLMHMMUserParams, bool]:
        """GLM-HMM initialization."""
        user_params = generate_glm_hmm_initial_params(
            self._n_states,
            X,
            y,
            inverse_link_function=self._inverse_link_function,
            is_new_session=is_new_session,
            random_key=random_key,
            init_funcs=self._initialization_funcs,
        )
        validate_params = any(
            self._initialization_funcs.get(s, True)
            for s in [
                "initial_proba_init_custom",
                "transition_proba_init_custom",
                "glm_params_init_custom",
                "glm_scale_init_custom",
            ]
        )
        return user_params, validate_params

    def fit(
        self,
        X: DESIGN_INPUT_TYPE,
        y: Union[NDArray, jnp.ndarray, nap.Tsd],
        init_params: Optional[GLMHMMUserParams] = None,
        is_new_session: Optional[jnp.ndarray] = None,
    ) -> "GLMHMM":
        """Fit the GLM-HMM model to the data."""
        self._validator.validate_inputs(X=X, y=y)
        # this validates or initialize the new session
        is_new_session = initialize_is_new_session(X, y, is_new_session)

        # validate the inputs & initialize solver
        # initialize params if no params are provided
        if init_params is None:
            init_params = self._model_specific_initialization(X, y, is_new_session)
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
        self._initialize_optimizer_and_state(data, y, init_params)

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
                pytree_map_and_reduce(
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
        # glm params m-step setup
        is_population = y.ndim > 1
        m_step_update = prepare_mstep_update_fn(
            is_population_glm=is_population,
            observation_model=self._observation_model,
            inverse_link_function=self._inverse_link_function,
            setup_solver=self._instantiate_solver,
            init_params=init_params,
        )

        # cannot wrap is_new_session, that's to be calculated at each update form the provided X and y.
        # for consistency, do not make a partial of that argument in run as well.
        self._optimization_run = eqx.Partial(
            em_hmm,
            inverse_link_function=self.inverse_link_function,
            log_likelihood_func=self._log_likelihood,
            m_step_fn_model_params=m_step_update,
            maxiter=self.maxiter,
            tol=self.tol,
        )

        self._optimization_update = eqx.Partial(
            em_step,
            inverse_link_function=self.inverse_link_function,
            log_likelihood_func=self._log_likelihood,
            m_step_fn_model_params=m_step_update,
        )

        def init_state_fn(*args, **kwargs) -> SolverState:
            state = EMState(
                data_log_likelihood=-jnp.array(jnp.inf),
                previous_data_log_likelihood=-jnp.array(jnp.inf),
                log_likelihood_history=jnp.full(self.maxiter, jnp.nan),
                iterations=0,
                converged=False,
            )
            return state

        self._optimization_init_state = init_state_fn
        return init_state_fn()
