"""Base class for HMM models."""

from __future__ import annotations

import abc
from numbers import Number
from typing import Any, Callable, Generic, Literal, Optional, Tuple, TypeVar, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import lazy_loader as lazy
from numpy.typing import ArrayLike, NDArray

from .. import tree_utils
from ..base_regressor import BaseRegressor
from ..hmm.expectation_maximization import (
    forward_backward,
    forward_pass,
    max_sum,
)
from ..regularizer import Regularizer
from ..type_casting import support_pynapple
from ..typing import (
    DESIGN_INPUT_TYPE,
)
from .initialize_parameters import (
    DEFAULT_INIT_FUNCTIONS,
    HMM_INITIALIZATION_FN_DICT,
    KMeansInitializer,
    _resolve_dirichlet_priors,
    _validate_init_funcs_keys,
    generate_hmm_initial_params,
    setup_hmm_initialization,
)
from .params import HMMModelParamsT, HMMUserParams, HMMUserProvidedParamsT
from .utils import _check_state_format
from .validation import HMMValidator

nap = lazy.load("pynapple")

MODEL_INITIALIZATION_FN_DICT_T = TypeVar("MODEL_INITIALIZATION_FN_DICT_T")
HMMValidatorT = TypeVar("HMMValidatorT", bound="HMMValidator")


class BaseHMM(
    BaseRegressor[HMMModelParamsT, HMMUserProvidedParamsT, HMMValidatorT],
    Generic[
        HMMModelParamsT,
        HMMUserProvidedParamsT,
        MODEL_INITIALIZATION_FN_DICT_T,
        HMMValidatorT,
    ],
):
    """Base class for HMM models.

    This class implements the core functionality for HMMs, handling tasks related to HMM parameters that are common
    across HMM-based models. Model-specific parameters and methods should be implemented in subclasses, where required
    abstract methods are defined.

    Parameters
    ----------
    n_states :
        The number of hidden states in the HMM. Must be a positive integer.
    dirichlet_initial_proba :
        Alpha parameters for the Dirichlet prior over the initial state probabilities.
        Shape ``(n_states,)``. If None, a flat (uninformative) prior is assumed.
    dirichlet_transition_proba :
        Alpha parameters for the Dirichlet prior over the transition probabilities.
        Shape ``(n_states, n_states)``. If None, a flat (uninformative) prior is assumed.
    regularizer :
        Regularization to use for model parameter optimization. Defines the regularization scheme
        and related parameters. Default is UnRegularized.
    regularizer_strength :
        Typically a float. Default is None. Sets the regularizer strength for the model parameters.
        If a user does not pass a value, and it is needed for regularization,
        a warning will be raised and the strength will default to 1.0.
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
    hmm_initialization_funcs :
        Dictionary specifying the initialization functions for the HMM parameters. This parameter is
        included at initialization for scikit-learn compatibility; however, users should set up the
        initialization functions using the :meth:`~nemos.hmm.hmm.BaseHMM.setup` method after model
        instantiation.
    """

    _validator_class: type[HMMValidatorT]
    _model_default_init_dict: MODEL_INITIALIZATION_FN_DICT_T
    _kmeans_init_class = KMeansInitializer

    def __init__(
        self,
        n_states: int,
        dirichlet_initial_proba: Union[jnp.ndarray, None] = None,  # (n_state, )
        dirichlet_transition_proba: Union[
            jnp.ndarray | None
        ] = None,  # (n_state, n_state):
        regularizer: Optional[Union[str, Regularizer]] = None,
        regularizer_strength: Optional[
            Any
        ] = None,  # this is used to regularize model params
        solver_name: str = None,
        solver_kwargs: Optional[dict] = None,
        maxiter: int = 1000,
        tol: float = 1e-8,
        seed=jax.random.PRNGKey(123),
        hmm_initialization_funcs: Optional[HMM_INITIALIZATION_FN_DICT] = None,
    ):
        super().__init__(
            regularizer=regularizer,
            regularizer_strength=regularizer_strength,
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
        )
        self.n_states = n_states
        # set the prior params
        self.dirichlet_initial_proba = dirichlet_initial_proba
        self.dirichlet_transition_proba = dirichlet_transition_proba

        self.seed = seed
        self.maxiter = maxiter
        self.tol = tol

        # fit attributes
        self.transition_prob_: Optional[jnp.ndarray] = None
        self.initial_prob_: Optional[jnp.ndarray] = None

        self.hmm_initialization_funcs = hmm_initialization_funcs

    def _hmm_setup(
        self,
        initial_proba_init: Optional[
            Literal["uniform", "random", "dirichlet", "kmeans"] | Callable
        ] = None,
        initial_proba_init_kwargs: Optional[dict] = None,
        transition_proba_init: Optional[
            Literal["sticky", "uniform", "random", "dirichlet", "kmeans"] | Callable
        ] = None,
        transition_proba_init_kwargs: Optional[dict] = None,
    ):
        """
        Set up the HMM model with specified initialization functions for the initial and transition probabilities.

        An optional initialization step that allows for users to specify initialization functions other than the
        defaults for initial and transition probabilities. The user can specify other built-in initialization functions
        or provide custom ones. If no initialization functions are provided, default initialization will be used.

        Available built-in initialization functions include:
        - For initial probabilities: "uniform" (default), "random", "kmeans"
        - For transition probabilities: "sticky" (default), "uniform", "random", "kmeans"

        Any custom initialization function provided by the user should be a callable that matches the following input
        (n_states, X, y, session_starts, random_key, **kwargs) and returns an array of the appropriate shape for the
        parameters it initializes (initial probabilities should return shape (n_states,) and transition probabilities
        should return shape (n_states, n_states)). Even if the function does not use all the inputs, they should be
        included in the function signature to ensure compatibility with the setup process.

        An example of a custom initialization function for initial probabilities could be:
        ```
        def custom_initial_proba_init(n_states, X, y, session_starts, random_key, min_prob=0.05):
            init_prob = jax.random.uniform(random_key, (n_states,), dtype=float)
            init_prob = init_prob / init_prob.sum()  # normalize to sum to 1
            init_prob = jnp.clip(init_prob, a_min=min_prob)  # enforce minimum probability
            return init_prob / init_prob.sum()  # renormalize after clipping
        ```

        Parameters
        ----------
        initial_proba_init :
            A string identifier for a built-in initialization function or a custom callable for initializing the
            initial probabilities of the HMM states.
        initial_proba_init_kwargs :
            A dictionary of keyword arguments to pass to the initial probability initialization function.
        transition_proba_init :
            A string identifier for a built-in initialization function or a custom callable for initializing the
            transition probabilities between HMM states.
        transition_proba_init_kwargs :
            A dictionary of keyword arguments to pass to the transition probability initialization function.
        """
        # flag for initializing kmeans model at parameter initialization
        self._hmm_use_kmeans = {
            "initial_proba_init": initial_proba_init == "kmeans",
            "transition_proba_init": transition_proba_init == "kmeans",
        }
        self._hmm_initialization_funcs = setup_hmm_initialization(
            initial_proba_init=initial_proba_init,
            initial_proba_init_kwargs=initial_proba_init_kwargs,
            transition_proba_init=transition_proba_init,
            transition_proba_init_kwargs=transition_proba_init_kwargs,
            init_funcs=self._hmm_initialization_funcs,
        )

    @abc.abstractmethod
    def _model_setup(self, **kwargs):
        """Model-specific setup of initialization functions."""
        # self._model_use_kmeans and self._model_initialization_funcs must be set here
        pass

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
        **kwargs,
    ):
        """
        Set up the HMM-based model, including HMM and model-specific setup.

        This can be overwritten by the child class for custom docstrings.
        """
        self._hmm_setup(
            initial_proba_init=initial_proba_init,
            initial_proba_init_kwargs=initial_proba_init_kwargs,
            transition_proba_init=transition_proba_init,
            transition_proba_init_kwargs=transition_proba_init_kwargs,
        )
        self._model_setup(**kwargs)

    @property
    def n_states(self) -> int:
        """Number of hidden states of the HMM."""
        return self._n_states

    @n_states.setter
    def n_states(self, n_states: int):
        """Set the number of hidden states and validator."""
        # quick sanity check and assignment
        if isinstance(n_states, int) and n_states > 0:
            self._n_states = n_states
            self._validator = self._validator_class(n_states=n_states)
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
        self._validator = self._validator_class(n_states=n_states)

    @property
    def maxiter(self):
        """EM maximum number of iterations."""
        return self._maxiter

    @maxiter.setter
    def maxiter(self, maxiter: int):
        """Validate and set the maximum number of iterations for the EM algorithm."""
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
        """Validate and set the tolerance for the EM algorithm convergence criterion."""
        if not isinstance(tol, Number) or tol <= 0:
            raise ValueError(
                f"``tol`` must be a strictly positive float. {tol} provided."
            )
        self._tol = float(tol)

    @property
    def dirichlet_initial_proba(self) -> jnp.ndarray | None:
        """Alpha parameters of the Dirichlet prior over the initial probabilities of HMM states.

        If ``None``, a flat prior is assumed.
        """
        return self._dirichlet_initial_proba

    @dirichlet_initial_proba.setter
    def dirichlet_initial_proba(self, value: jnp.ndarray | None):
        """Validate and set the alpha parameters of the Dirichlet prior over the initial probabilities."""
        self._dirichlet_initial_proba = _resolve_dirichlet_priors(
            value, (self._n_states,)
        )

    @property
    def dirichlet_transition_proba(self) -> jnp.ndarray | None:
        """Alpha parameters of the Dirichlet prior over the initial probabilities of HMM states.

        If ``None``, a flat prior is assumed.
        """
        return self._dirichlet_transition_proba

    @dirichlet_transition_proba.setter
    def dirichlet_transition_proba(self, value: jnp.ndarray | None):
        """Validate and set the alpha parameters of the Dirichlet prior over the transition probabilities."""
        self._dirichlet_transition_proba = _resolve_dirichlet_priors(
            value, (self._n_states, self._n_states)
        )

    @property
    def seed(self):
        """Random seed as a jax PRNG key."""
        return self._seed

    @seed.setter
    def seed(self, value):
        """Validate and set the random seed as a JAX PRNG key."""
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

    @property
    def hmm_initialization_funcs(self) -> HMM_INITIALIZATION_FN_DICT | None:
        """Dictionary of initialization functions for HMM parameters."""
        return self._hmm_initialization_funcs

    @hmm_initialization_funcs.setter
    def hmm_initialization_funcs(self, value: HMM_INITIALIZATION_FN_DICT | None):
        """
        Set the dictionary of initialization functions for HMM parameters.

        Validates keys against the HMM-specific DEFAULT_INIT_FUNCTIONS and merges
        with defaults before storing. Calls :meth:`_hmm_setup` afterward to apply any
        function/kwargs updates. May be called directly or via ``__init__``.
        """
        # always key validated in a model dependent way
        self._hmm_initialization_funcs = _validate_init_funcs_keys(
            value, DEFAULT_INIT_FUNCTIONS
        )
        self._hmm_setup()

    @property
    def model_initialization_funcs(self) -> MODEL_INITIALIZATION_FN_DICT_T | None:
        """Dictionary of initialization functions for model parameters."""
        return self._model_initialization_funcs

    @model_initialization_funcs.setter
    def model_initialization_funcs(self, value: MODEL_INITIALIZATION_FN_DICT_T | None):
        """
        Set the dictionary of initialization functions for model parameters.

        Validates keys against the model-specific ``_model_default_init_dict`` and merges
        with defaults before storing. Calls :meth:`_model_setup` afterward to apply any
        function/kwargs updates. May be called directly or via ``__init__``.
        """
        # always key validated in a model dependent way
        self._model_initialization_funcs = _validate_init_funcs_keys(
            value, self._model_default_init_dict
        )
        self._model_setup()

    def _hmm_params_initialization(
        self,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        session_starts: jnp.ndarray,
        random_key_pair: jax.Array,
    ) -> Tuple[HMMUserParams, bool]:
        """
        Initialize HMM parameters (initial and transition probabilities).

        Parameters
        ----------
        X :
            Design matrix.
        y :
            Target observations.
        session_starts :
            Boolean array marking the start of new sessions.
        random_key_pair :
            Pair of JAX PRNG keys passed to the initial-prob and transition-prob
            initialization functions respectively.

        Returns
        -------
        :
            Tuple of ``(hmm_params, validate_params)`` where ``hmm_params`` is an
            ``HMMUserParams`` and ``validate_params`` is True when custom initialization
            functions were used (requiring downstream parameter validation).
        """
        hmm_params = generate_hmm_initial_params(
            self._n_states,
            X,
            y,
            session_starts,
            random_key_pair=random_key_pair,
            init_funcs=self._hmm_initialization_funcs,
        )
        validate_params = self._hmm_initialization_funcs.get(
            "initial_proba_init_custom", True
        ) or self._hmm_initialization_funcs.get("transition_proba_init_custom", True)
        return hmm_params, validate_params

    @abc.abstractmethod
    def _model_params_initialization(self, X, y, session_starts, random_key: jax.Array):
        """
        Model-specific parameter initialization.

        Parameters
        ----------
        X :
            Design matrix.
        y :
            Target observations.
        session_starts :
            Boolean array marking the start of new sessions.
        random_key :
            JAX PRNG key for any stochastic initialization.

        Returns
        -------
        :
            Tuple of ``(model_params, validate_model)`` where ``validate_model`` is True
            when the initialized parameters require downstream validation.
        """
        pass

    @property
    def _use_kmeans(self) -> bool:
        """Check if kmeans initialization is needed for any HMM parameters."""
        return (
            (self._hmm_use_kmeans is not None) and any(self._hmm_use_kmeans.values())
        ) or (
            (self._model_use_kmeans is not None)
            and any(self._model_use_kmeans.values())
        )

    def _kmeans_extra_kwargs(self) -> dict:
        """Extra kwargs to initialize the kmeans model. Can be overridden by child classes if needed.

        Any key returned here must also appear as a reserved parameter in the relevant init-function
        protocol (e.g. ``InitFunctionGLM`` declares ``observation_model``/``inverse_link_function``).
        That keeps the same value from arriving twice at the initializer constructor.
        """
        return {}

    def _kmeans_resolve_model_kwargs(self, use_kmeans, init_funcs) -> dict:
        """
        Extract model kmeans kwargs and ensure consistency across relevant init funcs.

        This is not relevant for HMM init funcs, whose kwargs are not needed for the kmeans initializer
        """
        kmeans_kwargs = {}
        for param, use in use_kmeans.items():
            if use:
                kwargs = init_funcs.get(f"{param}_kwargs", {})
                for k, v in kwargs.items():
                    if k in kmeans_kwargs:
                        eq = eqx.tree_equal(kmeans_kwargs[k], v)
                        if eq is None or not bool(eq):
                            raise ValueError(
                                f"Inconsistent KMeans init arg '{k}': "
                                f"{kmeans_kwargs[k]} != {v}"
                            )
                    else:
                        kmeans_kwargs[k] = v
        return kmeans_kwargs

    def _kmeans_inject_initializer(self, initializer) -> None:
        """Inject ``initializer`` into every kmeans-using init-func kwargs dict.

        Other constructor-only inputs the model needs (e.g. ``observation_model`` for GLMHMM)
        reach the init funcs as protocol-required positional arguments at call time, so nothing
        beyond ``initializer`` is injected here.
        """
        if self._hmm_use_kmeans is not None:
            for param, use_kmeans in self._hmm_use_kmeans.items():
                if use_kmeans:
                    self._hmm_initialization_funcs[f"{param}_kwargs"][
                        "initializer"
                    ] = initializer
        if self._model_use_kmeans is not None:
            for param, use_kmeans in self._model_use_kmeans.items():
                if use_kmeans:
                    self._model_initialization_funcs[f"{param}_kwargs"][
                        "initializer"
                    ] = initializer

    def _kmeans_setup_initializer(
        self, X, y, session_starts=None, random_key: Optional[jax.Array] = None
    ):
        """Set up the kmeans initializer if any HMM or model parameters require kmeans initialization."""
        if self._model_use_kmeans is not None:
            kmeans_kwargs = self._kmeans_resolve_model_kwargs(
                self._model_use_kmeans, self._model_initialization_funcs
            )
        else:
            kmeans_kwargs = {}

        model_kwargs = self._kmeans_extra_kwargs()

        initializer = kmeans_kwargs.get(
            "initializer",
            self._kmeans_init_class(
                self.n_states,
                X,
                y,
                session_starts=session_starts,
                random_key=random_key,
                **model_kwargs,
                **kmeans_kwargs,
            ),
        )
        self._kmeans_inject_initializer(initializer)

    def _model_specific_initialization(self, X, y, session_starts=None):
        """Model-specific initialization."""
        keys = jax.random.split(self._seed, 3)
        hmm_keys = keys[:2]
        # the model needs to figure out how to split the key internally
        model_key = keys[2]

        # check kmeans kwargs and setup initializer.
        # fold_in derives an independent key so hmm_keys and model_key are unaffected.
        if self._use_kmeans:
            kmeans_key = jax.random.fold_in(self._seed, 0)
            self._kmeans_setup_initializer(
                X, y, session_starts=session_starts, random_key=kmeans_key
            )

        hmm_params, validate_hmm = self._hmm_params_initialization(
            X,
            y,
            session_starts,
            random_key_pair=hmm_keys,
        )
        model_params, validate_model = self._model_params_initialization(
            X,
            y,
            session_starts,
            random_key=model_key,
        )
        user_params = self._validator.wrap_user_params(
            model_params
        ) + self._validator.wrap_user_params(hmm_params)
        if validate_hmm or validate_model:
            model_params = self._validator.validate_and_cast_params(user_params)
            self._validator.validate_consistency(model_params, X=X, y=y)
            return model_params
        else:
            return self._validator.to_model_params(user_params)

    def _check_hmm_is_fit(self):
        """Ensure the HMM parameters have been fitted."""
        flat_params = [
            self.initial_prob_,
            self.transition_prob_,
        ]
        is_missing = [x is None for x in flat_params]
        if any(is_missing):
            param_labels = [
                "initial_prob_",
                "transition_prob_",
            ]
            missing_params = [
                p for p, missing in zip(param_labels, is_missing) if missing
            ]
            raise ValueError(
                f"This {self._validator.model_class} instance is not fitted yet. The following attributes are not set:"
                f" {missing_params}.\nPlease fit the HMM model first or "
                "set the missing attributes."
            )

    @abc.abstractmethod
    def _check_model_is_fit(self):
        """Ensure the model-specific parameters have been fitted."""
        pass

    def _check_is_fit(self):
        """Ensure the model has been fitted."""
        self._check_hmm_is_fit()
        self._check_model_is_fit()

    def _validate_and_prepare_inputs(self, X, y, session_starts=None):
        """Validate and prepare inputs."""
        # check if the model was fit
        self._check_is_fit()
        params = self._get_model_params()

        # validate inputs
        self._validator.validate_inputs(X=X, y=y)
        self._validator.validate_consistency(params, X=X, y=y)
        session_starts = self._validator.validate_and_cast_session_starts(
            X=X, y=y, session_starts=session_starts
        )
        return params, X, y, session_starts

    @abc.abstractmethod
    def _log_likelihood(
        self, params: HMMModelParamsT, X: DESIGN_INPUT_TYPE, y: ArrayLike
    ) -> jnp.ndarray:
        """Compute the log-likelihood of the data given the model parameters."""
        pass

    def _compute_loss(
        self,
        params: HMMModelParamsT,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: Union[NDArray, jnp.ndarray, nap.Tsd],
        session_starts: jnp.ndarray,
    ) -> jnp.ndarray:
        """Negative marginal log-likelihood via the forward pass.

        Implements the BaseRegressor ``_compute_loss`` contract for HMM-family
        models: returns the unpenalized scalar loss given parameters and data.
        ``score`` negates this to recover the log-likelihood.
        """
        # filter for non-nans, grab data if needed
        data, y, session_starts = self._preprocess_inputs(X, y, session_starts)
        # safe conversion to jax arrays of float
        params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, y.dtype), params)

        # make sure session_starts starts with a 1
        session_starts = session_starts.at[0].set(True)

        _, log_norm = forward_pass(
            params=params,
            X=data,
            y=y,
            session_starts=session_starts,
            log_likelihood_func=self._log_likelihood,
        )
        return -jnp.sum(log_norm)

    @cast_to_jax
    def compute_loss(
        self,
        params: HMMUserProvidedParamsT,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        session_starts: jnp.array,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        """Compute the loss function for the model.

        This method validates inputs and converts user-provided parameters to the internal
        representation before computing the loss.

        Parameters
        ----------
        params
            Parameter tuple of (coefficients, intercept).
        X
            Input data, array of shape ``(n_time_bins, n_features)`` or pytree of same.
        y
            Target data, array of shape ``(n_time_bins,)`` for single neuron models or
            ``(n_time_bins, n_neurons)`` for population models.
        session_starts
            Array indicating start indices of new sessions, shape ``(n_time_bins,)`` for
            boolean array or array of 0s and 1s or ``(n_sessions,)`` for integer array.
        *args
            Additional positional arguments passed to the model-specific loss function.
        **kwargs
            Additional keyword arguments passed to the model-specific loss function.

        Returns
        -------
        loss
            The loss value (negative log-likelihood).

        Raises
        ------
        ValueError
            If inputs or parameters have incompatible shapes or invalid values.
        """
        params = self._validator.validate_and_cast_params(params)
        self._validator.validate_inputs(X, y)
        self._validator.validate_consistency(params, X, y)
        session_starts = self._validator.validate_and_cast_session_starts(
            X=X, y=y, session_starts=session_starts
        )
        X, y = self._preprocess_inputs(X, y, session_starts)
        return self._compute_loss(params, X, y, session_starts, *args, **kwargs)

    def score(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: ArrayLike,
        session_starts: Optional[ArrayLike] = None,
    ) -> jnp.ndarray:
        """
        Marginal log-likelihood of the data under the fitted HMM.

        HMM-family models score only by log-likelihood. Variance-based or
        deviance-based pseudo-R² metrics are not implemented because they
        depend on a null/saturated-model construction that has no clean
        analogue for latent-state sequence models. Compute AIC/BIC or
        held-out log-likelihood externally if needed.

        Parameters
        ----------
        X :
            Input data/design matrix, shape ``(n_samples, n_features)``.
        y :
            Output data/observations, shape ``(n_samples, n_observations)``.
        session_starts :
            Optional array indicating user-provided session boundaries. Can be:
            - a boolean array indicating session starts, shape ``(n_samples,)``
            - an integer array of indices marking session starts, shape ``(n_sessions,)``
            - a pynapple.IntervalSet marking session epochs (requires either X or y to be a
            pynapple Tsd or TsdFrame to get timestamps)
            If None, creates a default array treating all data as one session.

        Returns
        -------
        :
            The marginal log-likelihood (summed over time).
        """
        params, X, y, session_starts = self._validate_and_prepare_inputs(
            X, y, session_starts
        )
        return -self._compute_loss(params, X, y, session_starts)

    @support_pynapple(conv_type="jax")
    def _smooth_proba(
        self,
        params: HMMModelParamsT,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: Union[NDArray, jnp.ndarray, nap.Tsd],
        session_starts: jnp.ndarray,
    ) -> jnp.ndarray:
        """Private smooth_proba compute."""
        # filter for non-nans, grab data if needed
        valid = tree_utils.get_valid_multitree(X, y)
        data, y, session_starts = self._preprocess_inputs(X, y, session_starts)

        # safe conversion to jax arrays of float
        params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, y.dtype), params)

        # make sure session_starts starts with a 1
        session_starts = session_starts.at[0].set(True)

        # smooth with forward backward
        log_posteriors, _, _, _, _, _ = forward_backward(
            params=params,
            X=data,
            y=y,
            session_starts=session_starts,
            log_likelihood_func=self._log_likelihood,
        )
        proba = jnp.exp(log_posteriors)
        # renormalize (numerical precision due to exponentiation)
        proba /= proba.sum(axis=1, keepdims=True)
        # re-attach nans
        proba = jnp.full((valid.shape[0], proba.shape[1]), jnp.nan).at[valid].set(proba)
        return proba

    def smooth_proba(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: Union[NDArray, jnp.ndarray, nap.Tsd],
        session_starts: Optional[ArrayLike] = None,
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
        X :
            Predictors, shape ``(n_time_points, n_features)``.
        y :
            Observations, shape ``(n_time_points,)`` for single observation or
            ``(n_time_points, n_observations)`` for population.
        session_starts :
            Optional array indicating user-provided session boundaries. Can be:
            - a boolean array indicating session starts, shape ``(n_time_points,)``
            - an integer array of indices marking session starts, shape ``(n_sessions,)``
            - a pynapple.IntervalSet marking session epochs (requires either X or y to be a
            pynapple Tsd or TsdFrame to get timestamps)
            If None, creates a default array treating all data as one session.

        Returns
        -------
        posteriors :
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
        :meth:`~nemos.hmm.BaseHMM.filter_proba`
            Compute filtering posteriors (conditioned on past observations only).
        :meth:`~nemos.hmm.BaseHMM.decode_state`
            Compute most likely state sequence (Viterbi decoding).

        Notes
        -----
        - Smoothing provides better state estimates than filtering because it uses all data
        - The algorithm properly handles session boundaries and NaN values at epoch borders
        """
        params, X, y, session_starts = self._validate_and_prepare_inputs(
            X, y, session_starts
        )
        return self._smooth_proba(params, X, y, session_starts)

    @support_pynapple(conv_type="jax")
    def _filter_proba(
        self,
        params: HMMModelParamsT,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: Union[NDArray, jnp.ndarray, nap.Tsd],
        session_starts: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute filtering probabilities without validation (internal method)."""
        # filter for non-nans, grab data if needed
        valid = tree_utils.get_valid_multitree(X, y)
        data, y, session_starts = self._preprocess_inputs(X, y, session_starts)

        # safe conversion to jax arrays of float
        params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, y.dtype), params)

        # make sure session_starts starts with a 1
        session_starts = session_starts.at[0].set(True)
        log_proba, _ = forward_pass(
            params,
            data,
            y,
            session_starts=session_starts,
            log_likelihood_func=self._log_likelihood,
        )
        proba = jnp.exp(log_proba)
        # renormalize (numerical errors due to exponentiating)
        proba /= proba.sum(axis=1, keepdims=True)
        # re-attach nans
        proba = jnp.full((valid.shape[0], proba.shape[1]), jnp.nan).at[valid].set(proba)
        return proba

    def filter_proba(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: Union[NDArray, jnp.ndarray, nap.Tsd],
        session_starts: Optional[ArrayLike] = None,
    ) -> jnp.ndarray | nap.TsdFrame:
        """Compute filtering posterior probabilities over hidden states.

        Computes the probability of being in each hidden state at each time point,
        conditioned only on observations up to that time point. This method uses the
        forward pass of the forward-backward algorithm, providing causal (online) state
        estimates that only use past and current observations.

        The filtering posteriors answer: "Given observations up to time t, what is the
        probability that the system is in state k at time t?"

        Parameters
        ----------
        X
            Predictors, shape ``(n_time_points, n_features)``.
        y
            Observations, shape ``(n_time_points,)`` for single observation or
            ``(n_time_points, n_observations)`` for population.
        session_starts :
            Optional array indicating user-provided session boundaries. Can be:
            - a boolean array indicating session starts, shape ``(n_time_points,)``
            - an integer array of indices marking session starts, shape ``(n_sessions,)``
            - a pynapple.IntervalSet marking session epochs (requires either X or y to be a
            pynapple Tsd or TsdFrame to get timestamps)
            If None, creates a default array treating all data as one session.

        Returns
        -------
        posteriors
            Filtering posterior probabilities, shape ``(n_time_points, n_states)``.
            Each row sums to 1 and represents the probability distribution over states
            at that time point conditioned on past observations.

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
        :meth:`~nemos.hmm.BaseHMM.smooth_proba`
            Compute smoothing posteriors (conditioned on all observations).
        :meth:`~nemos.hmm.BaseHMM.decode_state`
            Compute most likely state sequence (Viterbi decoding).

        Notes
        -----
        - Filtering provides causal state estimates suitable for online/real-time applications
        - Smoothing provides better estimates but requires all data (non-causal)
        - The algorithm properly handles session boundaries and NaN values at epoch borders
        - NaN values are removed before inference, but session markers are preserved
        - For pynapple inputs, the output TsdFrame has columns named "state_0", "state_1", etc.
        """
        params, X, y, session_starts = self._validate_and_prepare_inputs(
            X, y, session_starts
        )
        return self._filter_proba(params, X, y, session_starts)

    @support_pynapple(conv_type="jax")
    def _decode_state(
        self,
        params: HMMModelParamsT,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: Union[NDArray, jnp.ndarray, nap.Tsd],
        session_starts: jnp.ndarray,
        return_index: bool,
    ) -> jnp.ndarray:
        """Decode most likely state sequence without validation (internal method)."""
        # filter for non-nans, grab data if needed
        valid = tree_utils.get_valid_multitree(X, y)
        data, y, session_starts = self._preprocess_inputs(X, y, session_starts)

        # safe conversion to jax arrays of float
        params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, y.dtype), params)

        # make sure session_starts starts with a 1
        session_starts = session_starts.at[0].set(True)

        decoded_states = max_sum(
            params,
            data,
            y,
            session_starts=session_starts,
            log_likelihood_func=self._log_likelihood,
            return_index=return_index,
        )

        # reattach nans
        decoded_states = (
            jnp.full((valid.shape[0], *decoded_states.shape[1:]), jnp.nan)
            .at[valid]
            .set(decoded_states)
        )
        return decoded_states

    def decode_state(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: ArrayLike,
        session_starts: Optional[ArrayLike] = None,
        state_format: Literal["one-hot", "index"] = "one-hot",
    ) -> jnp.ndarray | nap.TsdFrame:
        """Compute the most likely hidden state sequence (Viterbi decoding).

        Finds the single most likely sequence of hidden states that best explains
        the observed data. This method uses the Viterbi (max-sum) algorithm to
        compute the state sequence that maximizes the joint probability of states
        and observations.

        Unlike ``smooth_proba()`` and ``filter_proba()`` which return probability
        distributions over states at each time point, this method makes a deterministic
        choice of the single best state sequence.

        The decoded states answer: "What is the most likely sequence of states that
        generated the observed data?"

        Parameters
        ----------
        X
            Predictors, shape ``(n_time_points, n_features)``.
        y
            Observations, shape ``(n_time_points,)`` for single observation or
            ``(n_time_points, n_observations)`` for population.
        session_starts :
            Optional array indicating user-provided session boundaries. Can be:
            - a boolean array indicating session starts, shape ``(n_time_points,)``
            - an integer array of indices marking session starts, shape ``(n_sessions,)``
            - a pynapple.IntervalSet marking session epochs (requires either X or y to be a
            pynapple Tsd or TsdFrame to get timestamps)
            If None, creates a default array treating all data as one session.
        state_format
            Format of the returned states:

            - ``"one-hot"``: Binary matrix of shape ``(n_time_points, n_states)`` where
              each row has a single 1 indicating the decoded state.
            - ``"index"``: Integer array of shape ``(n_time_points,)`` with values
              in ``[0, n_states-1]`` indicating the decoded state.

        Returns
        -------
        decoded_states
            Most likely state sequence:

            - If ``state_format="one-hot"``: shape ``(n_time_points, n_states)``.
              Each row is a one-hot vector with 1 in the position of the decoded state.
            - If ``state_format="index"``: shape ``(n_time_points,)``.
              Integer indices of the decoded states.

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
        :meth:`~nemos.hmm.BaseHMM.smooth_proba`
            Compute smoothing posteriors (conditioned on all observations).
        :meth:`~nemos.hmm.BaseHMM.filter_proba`
            Compute filtering posteriors (conditioned on past observations only).

        Notes
        -----
        - Viterbi decoding finds the globally optimal state sequence, not the sequence
          of individually most likely states from ``smooth_proba()``
        - This is a hard assignment (single best path) unlike probabilistic posteriors
        - The algorithm properly handles session boundaries and NaN values at epoch borders
        - Decoding is useful for segmenting continuous data into discrete behavioral states
        - For uncertainty estimates about states, use ``smooth_proba()`` instead
        """
        params, X, y, session_starts = self._validate_and_prepare_inputs(
            X, y, session_starts
        )
        # validate state_format
        _check_state_format(state_format)
        # define the return type for the max-sum
        return_index = False if state_format == "one-hot" else True
        return self._decode_state(params, X, y, session_starts, return_index)
