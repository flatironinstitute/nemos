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
from ..hmm.initialize_parameters import HMM_INITIALIZATION_FN_DICT
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
    hmm_initialization_funcs : dict, optional
        Dictionary of initialization functions for HMM probabilities (initial and
        transition). Included for scikit-learn compatibility; prefer configuring via the
        :meth:`setup` method after construction. If ``None``, defaults from
        ``DEFAULT_INIT_FUNCTIONS`` are used.
    model_initialization_funcs : dict, optional
        Dictionary of initialization functions for the GLM-specific parameters
        (coefficients, intercept, and scale). Included for scikit-learn compatibility;
        prefer configuring via the :meth:`setup` method after construction. If ``None``,
        defaults from ``DEFAULT_INIT_FUNCTIONS_GLMHMM`` are used.

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
        If ``hmm_initialization_funcs`` or ``model_initialization_funcs`` contains keys
        that are not valid for their respective default dictionary.
    ValueError
        If any ``*_kwargs`` entry in either initialization-funcs dictionary contains
        keyword arguments that don't match the signature of the corresponding
        initialization function.
    ValueError
        If ``maxiter`` is not a positive integer.
    ValueError
        If ``tol`` is not a positive float.
    """

    _validator_class = GLMHMMValidator
    _model_default_init_dict = DEFAULT_INIT_FUNCTIONS_GLMHMM
    _kmeans_init_class = KMeansInitializerGLM

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
        hmm_initialization_funcs: Optional[HMM_INITIALIZATION_FN_DICT] = None,
        model_initialization_funcs: Optional[GLMHMM_INITIALIZATION_FN_DICT] = None,
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
            hmm_initialization_funcs=hmm_initialization_funcs,
        )
        self.observation_model = observation_model
        self.inverse_link_function = inverse_link_function

        self.model_initialization_funcs = model_initialization_funcs

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
        initial_proba_init: Optional[
            Literal["uniform", "random", "dirichlet", "kmeans"] | Callable
        ] = None,
        initial_proba_init_kwargs: Optional[dict] = None,
        transition_proba_init: Optional[
            Literal["sticky", "uniform", "random", "dirichlet", "kmeans"] | Callable
        ] = None,
        transition_proba_init_kwargs: Optional[dict] = None,
        glm_params_init: Optional[Literal["random", "kmeans"] | Callable] = None,
        glm_params_init_kwargs: Optional[dict] = None,
        scale_init: Optional[Literal["constant", "kmeans"] | Callable] = None,
        scale_init_kwargs: Optional[dict] = None,
    ):
        """Configure how :meth:`fit` initializes each model parameter.

        Calling :meth:`setup` is optional: if it is never called, fitting starts from
        the default initializers listed below. Use it to change the initialization
        strategy by providing either the name of a built-in initialization function
        or a custom callable. Each argument left as ``None`` keeps the previously
        configured value; only the parameters you supply are updated.

        Available built-in initialization functions:

        - ``initial_proba_init``: ``"uniform"`` (default), ``"random"``,
          ``"dirichlet"``, ``"kmeans"``.
        - ``transition_proba_init``: ``"sticky"`` (default), ``"uniform"``,
          ``"random"``, ``"dirichlet"``, ``"kmeans"``.
        - ``glm_params_init``: ``"random"`` (default), ``"kmeans"``.
        - ``scale_init``: ``"constant"`` (default), ``"kmeans"``.

        Custom callables must follow one of two protocols depending on the parameter
        they initialize:

        - HMM probability initializers (``initial_proba_init``,
          ``transition_proba_init``) take ``(n_states, X, y, is_new_session,
          random_key, **kwargs)`` and return a ``jnp.ndarray`` of shape
          ``(n_states,)`` for the initial probabilities or ``(n_states, n_states)``
          for the transition matrix.
        - GLM parameter initializers (``glm_params_init``, ``scale_init``) take
          ``(n_states, X, y, inverse_link_function, is_new_session, random_key,
          **kwargs)``. ``glm_params_init`` returns ``(coef, intercept)`` shaped to
          match the design and ``n_states``; ``scale_init`` returns the scale array
          for the observation model.

        All arguments must appear in the function signature even when unused, so the
        framework can supply them uniformly.

        Parameters
        ----------
        initial_proba_init :
            Built-in name or custom callable used to initialize the initial-state
            probabilities (shape ``(n_states,)``).
        initial_proba_init_kwargs :
            Extra keyword arguments forwarded to ``initial_proba_init``.
        transition_proba_init :
            Built-in name or custom callable used to initialize the transition matrix
            (shape ``(n_states, n_states)``).
        transition_proba_init_kwargs :
            Extra keyword arguments forwarded to ``transition_proba_init``.
        glm_params_init :
            Built-in name or custom callable used to initialize the per-state GLM
            coefficients and intercepts.
        glm_params_init_kwargs :
            Extra keyword arguments forwarded to ``glm_params_init``.
        scale_init :
            Built-in name or custom callable used to initialize the scale parameter
            of the observation model (e.g. variance for Gaussian, dispersion for
            NegativeBinomial). Ignored by observation models without a scale.
        scale_init_kwargs :
            Extra keyword arguments forwarded to ``scale_init``.

        Raises
        ------
        ValueError
            If a custom callable's signature is incompatible with the protocol
            above, or if a ``*_kwargs`` entry contains keys that don't match the
            corresponding initializer's signature.

        Examples
        --------
        Switch a parameter to a different built-in scheme by passing its label:

        >>> from nemos.glm_hmm import GLMHMM
        >>> model = GLMHMM(n_states=3)
        >>> model.setup(initial_proba_init="random", glm_params_init="kmeans")

        Plug in a custom callable matching the GLM-side protocol:

        >>> import jax.numpy as jnp
        >>> def my_glm_init(
        ...     n_states, X, y, inverse_link_function, is_new_session, random_key,
        ... ):
        ...     coef = jnp.zeros((X.shape[1], n_states))
        ...     intercept = jnp.zeros((n_states,))
        ...     return coef, intercept
        >>> model.setup(glm_params_init=my_glm_init)
        """
        super().setup(
            initial_proba_init=initial_proba_init,
            initial_proba_init_kwargs=initial_proba_init_kwargs,
            transition_proba_init=transition_proba_init,
            transition_proba_init_kwargs=transition_proba_init_kwargs,
            glm_params_init=glm_params_init,
            glm_params_init_kwargs=glm_params_init_kwargs,
            scale_init=scale_init,
            scale_init_kwargs=scale_init_kwargs,
        )

    def _model_setup(
        self,
        glm_params_init: Optional[str | Callable] = None,
        glm_params_init_kwargs=None,
        scale_init: Optional[str | Callable] = None,
        scale_init_kwargs=None,
    ):
        """Validate and set GLM-side initialization functions.

        Derives ``_model_use_kmeans`` from the identity of the stored callables so the
        flag stays accurate regardless of whether the user passed the string ``"kmeans"``
        or the kmeans callable directly (e.g. when set via the
        ``model_initialization_funcs`` property).
        """
        self._model_initialization_funcs = setup_glm_hmm_initialization(
            glm_params_init=glm_params_init,
            glm_params_init_kwargs=glm_params_init_kwargs,
            scale_init=scale_init,
            scale_init_kwargs=scale_init_kwargs,
            init_funcs=self._model_initialization_funcs,
        )
        self._model_use_kmeans = {
            "glm_params_init": (
                self._model_initialization_funcs["glm_params_init"]
                is kmeans_glm_params_init
            ),
            "scale_init": (
                self._model_initialization_funcs["scale_init"] is kmeans_scale_init
            ),
        }

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
        return {
            "inverse_link_function": self.inverse_link_function,
            "observation_model": self.observation_model,
        }

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
            self._initialization_funcs[s]
            for s in [
                "initial_proba_init_custom",
                "transition_proba_init_custom",
                "glm_params_init_custom",
                "scale_init_custom",
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
        self._initialize_optimizer_and_state(init_params, data, y)

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
        coef = params.model_params.coef
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
            init_params=init_params.model_params,
        )

        # cannot wrap is_new_session, that's to be calculated at each update form the provided X and y.
        # for consistency, do not make a partial of that argument in run as well.
        self._optimization_run = eqx.Partial(
            em_hmm,
            log_likelihood_func=self._log_likelihood,
            m_step_fn_model_params=m_step_update,
            maxiter=self.maxiter,
            tol=self.tol,
        )

        self._optimization_update = eqx.Partial(
            em_step,
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
