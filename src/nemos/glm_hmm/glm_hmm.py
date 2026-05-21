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
from ..hmm.initialize_parameters import HMM_INITIALIZATION_FN_DICT, InitFunctionHMM
from ..hmm.utils import _check_state_format
from ..inverse_link_function_utils import resolve_inverse_link_function
from ..observation_models import Observations
from ..regularizer import GroupLasso, Lasso, Regularizer, Ridge
from ..tree_utils import pytree_map_and_reduce
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
    InitFunctionGLM,
    KMeansInitializerGLM,
    generate_glm_hmm_initial_model_params,
    kmeans_glm_params_init,
    kmeans_scale_init,
    setup_glm_hmm_initialization,
)
from .params import GLMHMMParams, GLMHMMUserParams
from .utils import compute_rate_per_state
from .validation import GLMHMMValidator


class GLMHMM(
    BaseHMM[
        GLMHMMUserParams, GLMHMMParams, GLMHMM_INITIALIZATION_FN_DICT, GLMHMMValidator
    ]
):
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
    | UnRegularized | LBFGS            | GradientDescent, BFGS, LBFGS, NonlinearCG, ProximalGradient |
    +---------------+------------------+-------------------------------------------------------------+
    | Ridge         | LBFGS            | GradientDescent, BFGS, LBFGS, NonlinearCG, ProximalGradient |
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
        Regularization scheme used in the M-step for the per-state GLM coefficients.
        Default is ``"Ridge"``. Pass ``"UnRegularized"`` to disable regularization.
    regularizer_strength :
        Strength of the regularization applied to the GLM coefficients. Default is
        ``1.0``. Ignored when ``regularizer="UnRegularized"``.
    dirichlet_initial_proba :
        Alpha parameters for the Dirichlet prior over the initial state probabilities.
        Shape ``(n_states,)``. If None, a flat (uninformative) prior is assumed.
    dirichlet_transition_proba :
        Alpha parameters for the Dirichlet prior over the transition probabilities.
        Shape ``(n_states, n_states)``. If None, a flat (uninformative) prior is assumed.
    solver_name :
        Solver used for the GLM M-step. The solver must be valid for the chosen
        regularizer (see table above). Default is ``None``, in which case the
        regularizer's default solver is selected (``"LBFGS"`` for Ridge /
        UnRegularized, ``"ProximalGradient"`` for Lasso / ElasticNet /
        GroupLasso).
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

    Examples
    --------
    **Fit a GLM-HMM**

    Basic model fitting with the default Bernoulli observation model. The number
    of hidden states is the only required argument; ``coef_`` carries one column
    per state, and the HMM transition matrix and initial distribution are exposed
    as fitted attributes.

    >>> import jax
    >>> import numpy as np
    >>> import nemos as nmo
    >>> np.random.seed(123)
    >>> X = np.random.normal(size=(200, 4))
    >>> y = np.random.binomial(n=1, p=0.5, size=200)
    >>> model = nmo.glm_hmm.GLMHMM(n_states=2).fit(X, y)
    >>> model.coef_.shape
    (4, 2)
    >>> model.transition_prob_.shape
    (2, 2)
    >>> model.initial_prob_.shape
    (2,)

    **Customize the Observation Model**

    Specify the observation model as a string:

    >>> model = nmo.glm_hmm.GLMHMM(n_states=2, observation_model="Poisson")
    >>> model.observation_model
    PoissonObservations()

    Or pass the observation model object directly:

    >>> model = nmo.glm_hmm.GLMHMM(
    ...     n_states=2, observation_model=nmo.observation_models.PoissonObservations()
    ... )
    >>> model.observation_model
    PoissonObservations()

    **Customize the Inverse Link Function**

    Use a soft-plus inverse link function instead of the observation-model default:

    >>> model = nmo.glm_hmm.GLMHMM(n_states=2, inverse_link_function=jax.nn.softplus)
    >>> model.inverse_link_function.__name__
    'softplus'

    **Change the Regularization**

    Regularization applies to the per-state GLM coefficients. The default is
    Ridge with strength ``1.0``. Tune the strength:

    >>> model = nmo.glm_hmm.GLMHMM(n_states=2, regularizer_strength=0.1).fit(X, y)
    >>> model.regularizer, float(model.regularizer_strength)
    (Ridge(), 0.1)

    Or switch to Lasso for sparse per-state coefficients (Lasso requires a
    proximal solver):

    >>> model = nmo.glm_hmm.GLMHMM(
    ...     n_states=2,
    ...     regularizer="Lasso",
    ...     regularizer_strength=0.01,
    ...     solver_name="ProximalGradient",
    ... ).fit(X, y)
    >>> model.regularizer
    Lasso()

    **Select a Solver**

    The solver is used for the M-step inside EM. Pick LBFGS for potentially
    faster convergence on smooth losses:

    >>> model = nmo.glm_hmm.GLMHMM(n_states=2, solver_name="LBFGS").fit(X, y)
    >>> model.solver_name
    'LBFGS'

    **Fit Across Multiple Sessions**

    Mark session boundaries with ``session_starts`` so the HMM resets at each
    new session start instead of treating the data as a single chain. Pass
    either a boolean mask of shape ``(n_time_bins,)`` with ``True`` at each
    session start, or an integer array of session-start indices — the two
    are equivalent:

    >>> is_new_mask = np.zeros(200, dtype=bool)
    >>> is_new_mask[0] = True
    >>> is_new_mask[100] = True
    >>> model = nmo.glm_hmm.GLMHMM(n_states=2).fit(X, y, session_starts=is_new_mask)
    >>> # Equivalent: pass the starts as integer indices.
    >>> model = nmo.glm_hmm.GLMHMM(n_states=2).fit(X, y, session_starts=np.array([0, 100]))

    **Decode Hidden States**

    Recover the most-likely state sequence (Viterbi-style) or the smoothed
    posterior probabilities from the forward-backward pass:

    >>> states = model.decode_state(X, y, session_starts=is_new_mask)
    >>> states.shape
    (200, 2)
    >>> post = model.smooth_proba(X, y, session_starts=is_new_mask)
    >>> post.shape
    (200, 2)

    **Simulate from the Fitted Model**

    Sample a hidden-state trajectory and observations conditioned on inputs:

    >>> activity, rates, sim_states = model.simulate(
    ...     jax.random.key(0), X, state_format="index"
    ... )
    >>> activity.shape, sim_states.shape
    ((200,), (200,))


    **Use a Dict of Arrays as Input**

    Features can be passed as any JAX pytree of 2-D arrays; the fitted ``coef_``
    will share the same pytree structure, with the trailing axis indexing states:

    >>> X_dict = {"input_1": X[:, :2], "input_2": X[:, 2:]}
    >>> model = nmo.glm_hmm.GLMHMM(n_states=2).fit(X_dict, y)
    >>> type(model.coef_)
    <class 'dict'>
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
        regularizer: Union[str, Regularizer] = "Ridge",
        regularizer_strength: Any = 1.0,  # this is used to regularize GLM coef.
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

        # cache the log-like
        self._log_like_cache = {}

    def _log_likelihood(
        self, params: GLMHMMParams, X: DESIGN_INPUT_TYPE, y: ArrayLike
    ) -> jnp.ndarray:
        """Compute the log-likelihood of the data given the model parameters.

        Use cached values to avoid unnecessary computations.
        """
        cache_key = (
            y.ndim > 1,
            self._observation_model,
            self._inverse_link_function,
        )
        ll_func = self._log_like_cache.get(cache_key)
        if ll_func is None:
            ll_func = prepare_estep_log_likelihood(
                y.ndim > 1, self._observation_model, self._inverse_link_function
            )
            self._log_like_cache[cache_key] = ll_func
        return ll_func(params, X, y)

    def setup(
        self,
        initial_proba_init: Optional[
            Literal["uniform", "random", "dirichlet", "kmeans"] | InitFunctionHMM
        ] = None,
        initial_proba_init_kwargs: Optional[dict] = None,
        transition_proba_init: Optional[
            Literal["sticky", "uniform", "random", "dirichlet", "kmeans"]
            | InitFunctionHMM
        ] = None,
        transition_proba_init_kwargs: Optional[dict] = None,
        glm_params_init: Optional[Literal["random", "kmeans"] | InitFunctionGLM] = None,
        glm_params_init_kwargs: Optional[dict] = None,
        scale_init: Optional[Literal["constant", "kmeans"] | InitFunctionGLM] = None,
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

        Custom callables must satisfy one of two ``typing.Protocol`` classes:

        - ``initial_proba_init`` and ``transition_proba_init`` must satisfy
          :class:`~nemos.hmm.initialize_parameters.InitFunctionHMM` and return a
          ``jnp.ndarray`` of shape ``(n_states,)`` or ``(n_states, n_states)``
          respectively.
        - ``glm_params_init`` and ``scale_init`` must satisfy
          :class:`~nemos.glm_hmm.initialize_parameters.InitFunctionGLM`.
          ``glm_params_init`` returns ``(coef, intercept)`` matched to the design
          and ``n_states``; ``scale_init`` returns the scale array for the
          observation model.

        To inspect a protocol's signature, import and ``help()`` it::

            from nemos.hmm.initialize_parameters import InitFunctionHMM
            from nemos.glm_hmm.initialize_parameters import InitFunctionGLM
            help(InitFunctionHMM)  # or help(InitFunctionGLM)

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
        ...     n_states, X, y, inverse_link_function, session_starts, random_key,
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
        """The observation model governing the emission distribution at each state.

        Always an instance of an :class:`~nemos.observation_models.Observations`
        subclass. The same distribution is used across all hidden states (per-state
        differences come from the state-specific coefficients/intercept/scale, not
        from the family). If a string alias was passed at construction time it is
        resolved to the corresponding instance here.
        """
        return self._observation_model

    @observation_model.setter
    def observation_model(self, observation: obs.Observations):
        """Validate and set the observation model.

        Parameters
        ----------
        observation :
            Either an :class:`~nemos.observation_models.Observations` instance,
            or a string alias from
            ``{"Poisson", "Gamma", "Bernoulli", "NegativeBinomial", "Gaussian"}``.
            String aliases are instantiated via
            :func:`nemos.observation_models.instantiate_observation_model`.

        Raises
        ------
        AttributeError, TypeError
            If the instance does not implement the
            :class:`~nemos.observation_models.Observations` interface (checked
            via :func:`nemos.observation_models.check_observation_model`).
        """
        if isinstance(observation, str):
            self._observation_model = instantiate_observation_model(observation)
            return
        # check that the model has the required attributes
        # and that the attribute can be called
        obs.check_observation_model(observation)
        self._observation_model = observation

    @property
    def inverse_link_function(self):
        """Inverse link function mapping the linear predictor to the emission space.

        Always a callable. If ``None`` was passed at construction time, this is
        resolved to the observation model's default (e.g. ``jnp.exp`` for Poisson,
        ``1 / x`` for Gamma, ``jax.nn.sigmoid`` for Bernoulli). Shared across all
        hidden states.
        """
        return self._inverse_link_function

    @inverse_link_function.setter
    def inverse_link_function(self, inverse_link_function: Callable):
        """Validate and set the inverse link function.

        Parameters
        ----------
        inverse_link_function :
            One of:

            - ``None`` — use the observation model's default inverse link.
            - ``str`` — name of a built-in (e.g. ``"identity"``, ``"log"``,
              ``"logit"``); resolved by
              :func:`nemos.inverse_link_function_utils.resolve_inverse_link_function`.
            - ``Callable`` — a custom function. Must be JAX-traceable
              (differentiable) and return a ``jax.numpy.ndarray`` or scalar
              when called on a JAX array.

        Raises
        ------
        TypeError
            If the value is neither callable nor a string.
        ValueError
            If a callable is non-differentiable or returns an unsupported type.
        """
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

    def _kmeans_init_func_kwargs(self) -> dict:
        # observation_model is a required arg of kmeans_glm_params_init /
        # kmeans_scale_init; surface it alongside the injected initializer so the
        # upstream guard and the function signature are both satisfied.
        return {"observation_model": self.observation_model}

    def _model_params_initialization(
        self,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        session_starts: jnp.ndarray,
        random_key: jax.Array,
    ) -> Tuple[GLMHMMUserParams, bool]:
        """GLM-HMM initialization."""
        user_params = generate_glm_hmm_initial_model_params(
            self._n_states,
            X,
            y,
            inverse_link_function=self._inverse_link_function,
            session_starts=session_starts,
            random_key=random_key,
            init_funcs=self._model_initialization_funcs,
        )
        validate_params = any(
            self._model_initialization_funcs[s]
            for s in ("glm_params_init_custom", "scale_init_custom")
        )
        return user_params, validate_params

    def fit(
        self,
        X: DESIGN_INPUT_TYPE,
        y: Union[NDArray, jnp.ndarray, nap.Tsd],
        init_params: Optional[GLMHMMUserParams] = None,
        session_starts: Optional[jnp.ndarray] = None,
    ) -> "GLMHMM":
        """Fit the GLM-HMM via Expectation-Maximization.

        Runs the EM algorithm until the absolute change in log-likelihood between
        consecutive iterations falls below ``tol`` or ``maxiter`` is reached.
        Fitted parameters are exposed on the instance as ``coef_``, ``intercept_``,
        ``scale_``, ``initial_prob_``, ``transition_prob_``, plus
        ``solver_state_`` (EM trace) and ``dof_resid_``.

        How parameters are initialized:

        - If ``init_params`` is ``None`` (typical), the per-state GLM parameters
          and HMM probabilities are produced by the initializers configured via
          :meth:`setup` (or the package defaults when :meth:`setup` was never
          called).
        - If ``init_params`` is provided, it bypasses the initializers entirely.
          It must be a 5-tuple ``(coef, intercept, scale, initial_prob,
          transition_prob)`` whose shapes are consistent with ``X``, ``y``, and
          ``n_states``.

        Parameters
        ----------
        X :
            Predictors, shape ``(n_time_bins, n_features)``. A pytree of arrays
            sharing leading dimension is also accepted; the fitted ``coef_``
            mirrors the pytree structure (with a trailing state axis). A pynapple
            ``TsdFrame`` is accepted.
        y :
            Observations, shape ``(n_time_bins,)`` for single neuron or
            ``(n_time_bins, n_neurons)`` for population models. A pynapple
            ``Tsd``/``TsdFrame`` is accepted.
        init_params :
            Optional explicit initial parameters as a 5-tuple
            ``(coef, intercept, scale, initial_prob, transition_prob)``. When
            ``None`` (default), the initializers configured by :meth:`setup`
            (or the defaults) are used.
        session_starts :
            Optional session boundaries for the HMM. Accepts:

            - a boolean array of shape ``(n_time_bins,)`` with ``True`` at each
              session start,
            - an integer array of session-start indices,
            - a pynapple ``IntervalSet`` (when ``X`` or ``y`` is a pynapple
              object).

            If ``X`` or ``y`` is a pynapple object and ``session_starts`` is
            ``None``, the (unique, enforced) ``time_support`` of the pynapple
            input determines the session starts. With no pynapple input and
            ``session_starts=None``, the whole input is treated as a single
            session.

        Returns
        -------
        self :
            The fitted estimator.

        Raises
        ------
        ValueError
            If inputs fail dimensionality, shape, or consistency checks (e.g.
            ``coef`` features do not match ``X.shape[1]``, or NaNs appear
            mid-epoch).
        TypeError
            If ``init_params`` is not a 5-tuple or has incompatible leaf types.

        Warns
        -----
        RuntimeWarning
            Emitted when EM runs out of iterations without satisfying the ``tol``
            criterion (``solver_state_.iterations == maxiter``). Consider
            enabling float64, raising ``maxiter``, or loosening ``tol``.

        Examples
        --------
        Basic fit with default Bernoulli observations:

        >>> import numpy as np
        >>> import nemos as nmo
        >>> np.random.seed(0)
        >>> X = np.random.normal(size=(200, 4))
        >>> y = np.random.binomial(n=1, p=0.5, size=200)
        >>> model = nmo.glm_hmm.GLMHMM(n_states=2).fit(X, y)
        >>> model.coef_.shape, model.transition_prob_.shape
        ((4, 2), (2, 2))

        Multiple sessions via explicit ``session_starts``:

        >>> session_starts = np.array([0, 100])
        >>> model = nmo.glm_hmm.GLMHMM(n_states=2).fit(X, y, session_starts=session_starts)

        See Also
        --------
        setup : Configure the initializers used when ``init_params is None``.
        update : Run a single EM iteration (advanced, manual loop).
        """
        self._validator.validate_inputs(X=X, y=y)
        # validate and cast session boundaries, shifting markers off NaN samples
        session_starts = self._validator.validate_and_cast_session_starts(
            X, y, session_starts=session_starts
        )

        # validate the inputs & initialize solver
        # initialize params if no params are provided
        if init_params is None:
            init_params = self._model_specific_initialization(X, y, session_starts)
        else:
            init_params = self._validator.validate_and_cast_params(init_params)
            self._validator.validate_consistency(init_params, X=X, y=y)

        self._validator.feature_mask_consistency(
            getattr(self, "_feature_mask", None), init_params
        )

        # filter for non-nans, grab data if needed
        data, y, session_starts = self._preprocess_inputs(X, y, session_starts)

        # make sure session_starts starts with a 1
        session_starts = session_starts.at[0].set(True)

        # set up optimization
        self._initialize_optimizer_and_state(init_params, data, y)

        # run EM
        (
            fit_params,
            self.solver_state_,
        ) = self._optimizer_run(init_params, X=data, y=y, session_starts=session_starts)

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
        coef_leaf = jax.tree_util.tree_leaves(coef)[0]
        if coef_leaf.ndim == 3:
            n_neurons = coef_leaf.shape[1]
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

    def simulate(
        self,
        random_key: jax.Array,
        feedforward_input: DESIGN_INPUT_TYPE,
        state_format: Literal["one-hot", "index"] = "index",
        session_starts: Optional[jax.Array] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Simulate neural activity and hidden states from the model.

        Simulates a trajectory through the hidden state space according to the
        HMM dynamics, then generates observations from the GLM emission model
        conditioned on each state.

        Parameters
        ----------
        random_key :
            JAX random key for reproducible simulation.
        feedforward_input :
            Design matrix of shape ``(n_time_bins, n_features)``. If a pynapple
            Tsd/TsdFrame is provided, session boundaries are detected from
            ``time_support`` and the hidden state chain is reset at each session start.
        state_format :
            Format for the returned states:

            - ``"index"``: Integer array of shape ``(n_time_bins,)`` with state indices.
            - ``"one-hot"``: Binary array of shape ``(n_time_bins, n_states)``.
        session_starts :
            Boolean array of shape ``(n_time_bins,)`` marking session starts with
            ``True``. If ``None``, the entire input is treated as a single session.
            Ignored when ``feedforward_input`` is a pynapple object (boundaries are
            inferred from ``time_support``).

        Returns
        -------
        simulated_activity :
            Simulated observations from the emission model. Shape ``(n_time_bins,)``
            for single neuron or ``(n_time_bins, n_neurons)`` for population models.
        firing_rates :
            Predicted firing rates conditioned on the simulated states.
            Shape ``(n_time_bins,)`` or ``(n_time_bins, n_neurons)``.
        simulated_states :
            Simulated hidden state trajectory. Shape depends on ``state_format``.

        Raises
        ------
        ValueError
            If the model has not been fit.

        Examples
        --------
        >>> import jax
        >>> import numpy as np
        >>> import nemos as nmo
        >>> np.random.seed(123)
        >>> X = np.random.randn(100, 3)
        >>> y = np.random.binomial(1, 0.5, 100)
        >>> model = nmo.glm_hmm.GLMHMM(n_states=2, observation_model="Bernoulli")
        >>> model = model.fit(X, y)
        >>> key = jax.random.key(0)
        >>> X_new = np.random.randn(50, 3)
        >>> activity, rates, states = model.simulate(key, X_new)
        >>> activity.shape
        (50,)
        >>> states.shape
        (50,)

        See Also
        --------
        decode_state : Infer most likely state sequence from observations.
        smooth_proba : Compute posterior state probabilities.
        """
        _check_state_format(state_format)

        params, feedforward_input, _, session_starts = (
            self._validate_and_prepare_inputs(feedforward_input, None, session_starts)
        )

        # preprocess inputs (drop nans, extract data)
        data, _, session_starts = self._preprocess_inputs(
            feedforward_input, None, session_starts
        )

        # ensure first time point is a session start
        session_starts = session_starts.at[0].set(True)

        # run simulation
        simulated_activity, firing_rates, simulated_states = self._simulate(
            random_key, params, data, session_starts
        )

        # format state output
        if state_format == "one-hot":
            simulated_states = jax.nn.one_hot(
                simulated_states, self._n_states, dtype=jnp.int32
            )

        return simulated_activity, firing_rates, simulated_states

    def smooth_proba(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: Union[NDArray, jnp.ndarray, nap.Tsd],
        session_starts: Optional[ArrayLike] = None,
    ) -> jnp.ndarray | nap.TsdFrame:
        """Compute smoothing posterior probabilities for the GLM-HMM.

        Thin override of :meth:`nemos.hmm.BaseHMM.smooth_proba` carrying a
        GLM-HMM-specific Example. The full Parameters/Returns/Raises/Notes
        documentation lives on the base method.

        Examples
        --------
        Fit a GLM-HMM and compute smoothing posteriors:

        >>> import numpy as np
        >>> import nemos as nmo
        >>> np.random.seed(123)
        >>> X = np.random.randn(100, 5)
        >>> y = np.random.poisson(2, size=100)
        >>> model = nmo.glm_hmm.GLMHMM(n_states=3, observation_model="Poisson").fit(X, y)
        >>> posteriors = model.smooth_proba(X, y)
        >>> posteriors.shape
        (100, 3)
        >>> bool(np.allclose(posteriors.sum(axis=1), 1.0))
        True

        With pynapple inputs the result is returned as a ``TsdFrame``:

        >>> import pynapple as nap
        >>> t = np.arange(100) * 0.01
        >>> X_tsd = nap.TsdFrame(t=t, d=X)
        >>> y_tsd = nap.Tsd(t=t, d=y.astype(float))
        >>> type(model.smooth_proba(X_tsd, y_tsd)).__name__
        'TsdFrame'
        """
        return super().smooth_proba(X, y, session_starts=session_starts)

    def filter_proba(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: Union[NDArray, jnp.ndarray, nap.Tsd],
        session_starts: Optional[ArrayLike] = None,
    ) -> jnp.ndarray | nap.TsdFrame:
        """Compute filtering posterior probabilities for the GLM-HMM.

        Thin override of :meth:`nemos.hmm.BaseHMM.filter_proba` carrying a
        GLM-HMM-specific Example. The full Parameters/Returns/Raises/Notes
        documentation lives on the base method.

        Examples
        --------
        Fit a GLM-HMM and compute filtering posteriors (causal/online):

        >>> import numpy as np
        >>> import nemos as nmo
        >>> np.random.seed(123)
        >>> X = np.random.randn(100, 5)
        >>> y = np.random.poisson(2, size=100)
        >>> model = nmo.glm_hmm.GLMHMM(n_states=3, observation_model="Poisson").fit(X, y)
        >>> filt = model.filter_proba(X, y)
        >>> filt.shape
        (100, 3)
        >>> bool(np.allclose(filt.sum(axis=1), 1.0))
        True
        """
        return super().filter_proba(X, y, session_starts=session_starts)

    def decode_state(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: ArrayLike,
        session_starts: Optional[ArrayLike] = None,
        state_format: Literal["one-hot", "index"] = "one-hot",
    ) -> jnp.ndarray | nap.TsdFrame:
        """Viterbi-decode the most likely hidden state sequence for the GLM-HMM.

        Thin override of :meth:`nemos.hmm.BaseHMM.decode_state` carrying a
        GLM-HMM-specific Example. The full Parameters/Returns/Raises/Notes
        documentation lives on the base method.

        Examples
        --------
        Decode the most likely state sequence as integer indices:

        >>> import numpy as np
        >>> import nemos as nmo
        >>> np.random.seed(123)
        >>> X = np.random.randn(100, 5)
        >>> y = np.random.poisson(2, size=100)
        >>> model = nmo.glm_hmm.GLMHMM(n_states=3, observation_model="Poisson").fit(X, y)
        >>> states = model.decode_state(X, y, state_format="index")
        >>> states.shape
        (100,)

        One-hot output (default):

        >>> states_onehot = model.decode_state(X, y)
        >>> states_onehot.shape
        (100, 3)
        >>> bool(np.all(states_onehot.sum(axis=1) == 1))
        True
        """
        return super().decode_state(
            X, y, session_starts=session_starts, state_format=state_format
        )

    def _simulate(
        self,
        random_key: jax.Array,
        params: GLMHMMParams,
        X: jnp.ndarray,
        session_starts: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Simulate activity vis jax.lax.scan.

        Parameters
        ----------
        random_key :
            JAX random key.
        params :
            Model parameters.
        X :
            Design matrix of shape ``(n_time_bins, n_features)``.
        session_starts :
            Boolean array marking session starts.

        Returns
        -------
        simulated_activity :
            Simulated observations.
        firing_rates :
            Predicted rates conditioned on simulated states.
        simulated_states :
            State indices at each time point.
        """
        # unpack log probabilities directly (avoid exp then log in categorical)
        log_initial_prob = params.hmm_params.log_initial_prob
        log_transition_prob = params.hmm_params.log_transition_prob
        scale = jnp.exp(params.model_params.log_scale)

        # pre-compute rates for all states: (n_time_bins, n_states) or (n_time_bins, n_neurons, n_states)
        all_rates = compute_rate_per_state(
            X, params.model_params, self._inverse_link_function
        )

        # pre-generate random keys for all time steps
        n_time_bins = X.shape[0]
        all_keys = jax.random.split(random_key, n_time_bins * 2)
        state_keys = all_keys[:n_time_bins]
        obs_keys = all_keys[n_time_bins:]

        def simulate_step(carry, inputs):
            """Single simulation step."""
            prev_state_idx = carry
            rates_t, is_new_sess, state_key, obs_key = inputs

            # sample state: log_initial_prob if new session, else log transition from prev
            log_state_probs = jax.lax.cond(
                is_new_sess,
                lambda: log_initial_prob,
                lambda: log_transition_prob[prev_state_idx],
            )
            state_idx = jax.random.categorical(state_key, log_state_probs)

            # get rate and scale for sampled state
            # handles both (n_states,) and (n_neurons, n_states)
            rate = rates_t[..., state_idx]
            state_scale = scale[..., state_idx]

            # sample observation
            y_t = self._observation_model.sample_generator(
                key=obs_key, predicted_rate=rate, scale=state_scale
            )

            return state_idx, (y_t, rate, state_idx)

        # initialize carry (state will be overwritten at first step since session_starts[0]=True)
        init_carry = jnp.array(0)

        # run scan
        _, (simulated_activity, firing_rates, simulated_states) = jax.lax.scan(
            simulate_step, init_carry, (all_rates, session_starts, state_keys, obs_keys)
        )

        return simulated_activity, firing_rates, simulated_states

    def save_params(
        self,
        filename: Union[str, Path],
    ):
        """Save GLM-HMM model parameters and fit state to a .npz file.

        Persists hyperparameters returned by :meth:`get_params` together with the
        fitted attributes (``coef_``, ``intercept_``, ``scale_``, ``initial_prob_``,
        ``transition_prob_``, ``dof_resid_``). The ``solver_state_`` is intentionally
        excluded as it is solver-specific and not needed to reuse the fitted model.
        The file can be reloaded with :func:`nemos.load_model`.

        Initialization functions are serialized by their fully-qualified name when
        they are built-ins; :func:`nemos.load_model` resolves them via the registry.
        Custom callables are also stored by name, which means a custom callable
        must be supplied at load time. Because the io path consumes the
        ``model_initialization_funcs`` / ``hmm_initialization_funcs`` constructor
        argument (not :meth:`setup`), the override is passed as a (partial) dict of
        slot → callable, and the setter fills in the remaining slots from the saved
        names.

        Parameters
        ----------
        filename :
            Path of the output file (``.npz`` format).

        Examples
        --------
        Default round-trip — built-in initializers are resolved automatically on
        load:

        >>> import os, tempfile
        >>> import numpy as np
        >>> import nemos as nmo
        >>> np.random.seed(0)
        >>> X = np.random.normal(size=(80, 3))
        >>> y = np.random.binomial(n=1, p=0.5, size=80)
        >>> model = nmo.glm_hmm.GLMHMM(n_states=2).fit(X, y)
        >>> with tempfile.TemporaryDirectory() as d:
        ...     path = os.path.join(d, "glmhmm.npz")
        ...     model.save_params(path)
        ...     loaded = nmo.load_model(path)
        >>> bool(np.allclose(model.coef_, loaded.coef_))
        True

        Round-trip with a custom GLM-params initializer. Pass it back as a partial
        dict under ``model_initialization_funcs``; remaining slots fall back to the
        saved (built-in) names:

        >>> import jax.numpy as jnp
        >>> def my_glm_init(
        ...     n_states, X, y, inverse_link_function, session_starts, random_key,
        ... ):
        ...     return jnp.zeros((X.shape[1], n_states)), jnp.zeros((n_states,))
        >>> model = nmo.glm_hmm.GLMHMM(n_states=2)
        >>> model.setup(glm_params_init=my_glm_init)
        >>> _ = model.fit(X, y)
        >>> with tempfile.TemporaryDirectory() as d:
        ...     path = os.path.join(d, "glmhmm.npz")
        ...     model.save_params(path)
        ...     loaded = nmo.load_model(
        ...         path,
        ...         mapping_dict={
        ...             "model_initialization_funcs": {"glm_params_init": my_glm_init},
        ...         },
        ...     )
        >>> loaded.model_initialization_funcs["glm_params_init"] is my_glm_init
        True
        """
        # initialize saving dictionary
        fit_attrs = self._get_fit_state()
        fit_attrs.pop("solver_state_", None)
        string_attrs = ["inverse_link_function"]
        self._save_params(filename, fit_attrs, string_attrs)

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
        params: GLMHMMUserParams,
        opt_state: NamedTuple,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        *args,
        session_starts: Optional[jnp.ndarray] = None,
        n_samples: Optional[int] = None,
        **kwargs,
    ) -> StepResult:
        """Run a single EM iteration on the GLM-HMM.

        Performs one E-step / M-step pair starting from the supplied parameters and
        EM state, updates the model's fitted attributes (``coef_``, ``intercept_``,
        ``scale_``, ``initial_prob_``, ``transition_prob_``, ``solver_state_``,
        ``dof_resid_``) in place, and returns the updated parameter tuple and EM
        state. Intended for callers that need fine-grained control over EM
        iteration (e.g. checkpointing, custom convergence criteria) instead of the
        bundled :meth:`fit` loop.

        :meth:`initialize_optimizer_and_state` must be called first so that the EM
        step function and initial ``opt_state`` are available.

        Parameters
        ----------
        params :
            Current model parameters as a 5-tuple
            ``(coef, intercept, scale, initial_prob, transition_prob)`` matching
            the structure produced by :meth:`initialize_params`.
        opt_state :
            EM state returned by :meth:`initialize_optimizer_and_state` or by the
            previous call to :meth:`update`.
        X :
            Predictors, shape ``(n_time_bins, n_features)`` (or a pytree of arrays
            of the same shape).
        y :
            Observations, shape ``(n_time_bins,)`` or ``(n_time_bins, n_neurons)``.
        session_starts :
            Optional session-boundary spec. Accepts a boolean mask of shape
            ``(n_time_bins,)``, an integer array of session-start indices, or a
            pynapple ``IntervalSet`` (when X or y is a pynapple object). ``None``
            treats all samples as a single session.
        n_samples :
            Total sample count to use when estimating the residual degrees of
            freedom. Defaults to ``X.shape[0]``.

        Returns
        -------
        params :
            Updated user-facing parameter tuple.
        state :
            Updated EM state.

        Raises
        ------
        ValueError
            If inputs fail shape/consistency validation.

        Examples
        --------
        >>> import numpy as np
        >>> import nemos as nmo
        >>> np.random.seed(0)
        >>> X = np.random.normal(size=(80, 3))
        >>> y = np.random.binomial(n=1, p=0.5, size=80)
        >>> model = nmo.glm_hmm.GLMHMM(n_states=2)
        >>> init_params = model.initialize_params(X, y)
        >>> opt_state = model.initialize_optimizer_and_state(init_params, X, y)
        >>> new_params, new_state = model.update(init_params, opt_state, X, y)
        """
        # validate inputs and session boundaries
        self._validator.validate_inputs(X=X, y=y)
        session_starts = self._validator.validate_and_cast_session_starts(
            X, y, session_starts=session_starts
        )

        # drop nans and pull pytree data
        data, y, session_starts = self._preprocess_inputs(X, y, session_starts)

        # ensure first sample is a session start
        session_starts = session_starts.at[0].set(True)

        # wrap into model params (assumes init was done via
        # `initialize_optimizer_and_state` so the EM step function is in place)
        params = self._validator.to_model_params(params)

        # one EM step
        updated_params, updated_state = self._optimizer_update(
            params, opt_state, data, y, session_starts=session_starts
        )

        # persist
        self._set_model_params(updated_params)
        self.solver_state_ = updated_state
        self.dof_resid_ = self._estimate_resid_degrees_of_freedom(
            data, n_samples=n_samples
        )

        return self._validator.from_model_params(updated_params), updated_state

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

        # cannot wrap session_starts, that's to be calculated at each update form the provided X and y.
        # for consistency, do not make a partial of that argument in run as well.
        self._optimizer_run = eqx.Partial(
            em_hmm,
            log_likelihood_func=self._log_likelihood,
            m_step_fn_model_params=m_step_update,
            maxiter=self.maxiter,
            tol=self.tol,
        )

        self._optimizer_update = eqx.Partial(
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

        self._optimizer_init_state = init_state_fn
        return init_state_fn()
