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
from ..hmm.parallel_expectation import (
    forward_backward_assoc,
    forward_pass_assoc,
    max_sum_assoc,
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
    _resolution_was_noop,
    _resolve_dirichlet_priors,
    _validate_init_funcs_keys,
    generate_hmm_initial_params,
    kmeans_initial_proba_init,
    kmeans_transition_proba_init,
    setup_hmm_initialization,
)
from .params import HMMModelParamsT, HMMUserParams, HMMUserProvidedParamsT
from .utils import _check_state_format
from .validation import HMMValidator

nap = lazy.load("pynapple")

MODEL_INITIALIZATION_FN_DICT_T = TypeVar("MODEL_INITIALIZATION_FN_DICT_T")
HMMValidatorT = TypeVar("HMMValidatorT", bound="HMMValidator")

FORWARD_PASS = {
    "sequential": forward_pass,
    "associative": forward_pass_assoc,
}

FORWARD_BACKWARD = {
    "sequential": forward_backward,
    "associative": forward_backward_assoc,
}

MAX_SUM = {
    "sequential": max_sum,
    "associative": max_sum_assoc,
}


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
        Any array-like (list, tuple, NumPy or JAX array) of shape ``(n_states,)``, cast
        to a JAX array on assignment. All values must be >= 1. If None, a flat
        (uninformative) prior is assumed.
    dirichlet_transition_proba :
        Alpha parameters for the Dirichlet prior over the transition probabilities.
        Any array-like (list, tuple, NumPy or JAX array) of shape
        ``(n_states, n_states)``, cast to a JAX array on assignment. All values must be
        >= 1. If None, a flat (uninformative) prior is assumed.
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
    estep_type:
        How the forward-backward recursions of the E-step are evaluated. ``"sequential"``
        steps through the time bins one at a time; ``"associative"`` uses
        ``jax.lax.associative_scan``, whose depth grows like ``log(n_time_bins)`` rather
        than ``n_time_bins``.

        Which is faster depends on the hardware. On GPU and TPU the ``"associative"``
        can be orders of magnitude faster; on CPU it is usually slower, a CPU core being
        well suited to a tight sequential loop. ``"associative"`` also holds its whole scan
        in memory, roughly ``4 * n_time_bins * n_states ** 2`` floats against
        ``n_time_bins * n_states``, which becomes the binding constraint for large
        ``n_states``.
    """

    _validator_class: type[HMMValidatorT]
    _model_default_init_dict: MODEL_INITIALIZATION_FN_DICT_T
    _kmeans_init_class = KMeansInitializer

    def __init__(
        self,
        n_states: int,
        dirichlet_initial_proba: Optional[ArrayLike] = None,  # (n_state, )
        dirichlet_transition_proba: Optional[ArrayLike] = None,  # (n_state, n_state):
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
        estep_type: Literal["sequential", "associative"] = "sequential",
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
        self.estep_type = estep_type

    @property
    def estep_type(self):
        """Expectation type: sequential or associative."""
        return self._estep_type

    @estep_type.setter
    def estep_type(self, value):
        """Setter for expectation type: sequential or associative."""
        if value not in ["sequential", "associative"]:
            raise ValueError(
                "estep_type must be either ``'sequential'`` or ``'associative'``. "
                f"{value} provided instead."
            )
        self._estep_type = value
        # the E-step is bound into the optimizer partials when the solver is
        # instantiated, so an already-built solver would keep running the previous
        # algorithm. The inference paths read self._estep_type live and are unaffected.
        self._invalidate_solver()

    def _hmm_setup(
        self,
        initial_proba_init: Optional[
            Literal["uniform", "random", "dirichlet"]
        ] = None,
        transition_proba_init: Optional[
            Literal["sticky", "uniform", "random", "dirichlet"]
        ] = None,
    ):
        """
        Set which initialization methods to use for the initial and transition probabilities.

        An optional initialization step that allows for users to specify initialization methods other than the
        defaults for initial and transition probabilities. The user can specify one of the built-in initialization
        functions listed below.
        - For initial probabilities: "uniform" (default), "random", "dirichet"
        - For transition probabilities: "sticky" (default), "uniform", "random", "dirichlet"

        For any custom changes to the built-in methods, e.g. overriding default optional arguments, for using
        kmeans initialization, or for any other custom initialization method, the user should generate the 
        initial parameters externally and provide them as `init_params` to the `fit` method.

        Parameters
        ----------
        initial_proba_init :
            A string identifier for a built-in initialization function or a custom callable for initializing the
            initial probabilities of the HMM states.
        transition_proba_init :
            A string identifier for a built-in initialization function or a custom callable for initializing the
            transition probabilities between HMM states.
        """
        self._hmm_initialization_funcs = setup_hmm_initialization(
            initial_proba_init=initial_proba_init,
            transition_proba_init=transition_proba_init,
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
            Literal["uniform", "random", "dirichlet"] 
        ] = None,
        transition_proba_init: Optional[
            Literal["sticky", "uniform", "random", "dirichlet"]
        ] = None,
        **kwargs,
    ):
        """
        Set up the HMM-based model, including HMM and model-specific setup.

        This can be overwritten by the child class for custom docstrings.
        """
        if initial_proba_init not in [None, "uniform", "random", "dirichlet"]:
            raise ValueError(
                f"Invalid value for initial_proba_init: {initial_proba_init}. "
                f"Expected one of [None, 'uniform', 'random', 'dirichlet']."
            )
        if transition_proba_init not in [None, "sticky", "uniform", "random", "dirichlet"]:
            raise ValueError(
                f"Invalid value for transition_proba_init: {transition_proba_init}. "
                f"Expected one of [None, 'sticky', 'uniform', 'random', 'dirichlet']."
            )
        self._hmm_setup(
            initial_proba_init=initial_proba_init,
            transition_proba_init=transition_proba_init,
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
         ``|log_likelihood_current - log_likelihood_previous| < tol``

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
    def dirichlet_initial_proba(self, value: Optional[ArrayLike]):
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
    def dirichlet_transition_proba(self, value: Optional[ArrayLike]):
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
        if value is not None and _resolution_was_noop(
            value, self._hmm_initialization_funcs
        ):
            self._hmm_initialization_funcs = value

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
        if value is not None and _resolution_was_noop(
            value, self._model_initialization_funcs
        ):
            self._model_initialization_funcs = value

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

    def _model_specific_initialization(self, X, y, session_starts=None):
        """Model-specific initialization."""
        keys = jax.random.split(self._seed, 3)
        hmm_keys = keys[:2]
        # the model needs to figure out how to split the key internally
        model_key = keys[2]

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

        # validate inputs and session boundaries
        session_starts = self._validator.validate_and_cast_inputs(
            X=X, y=y, session_starts=session_starts
        )
        self._validator.validate_consistency(params, X=X, y=y)
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
        (params,) = tree_utils.tree_astype(params, dtype=y.dtype)

        # make sure session_starts starts with a 1
        session_starts = session_starts.at[0].set(True)

        _, log_norm = FORWARD_PASS[self._estep_type](
            params=params,
            X=data,
            y=y,
            session_starts=session_starts,
            log_likelihood_func=self._log_likelihood,
        )
        return -jnp.sum(log_norm)

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
        (params,) = tree_utils.tree_astype(params, dtype=y.dtype)

        # make sure session_starts starts with a 1
        session_starts = session_starts.at[0].set(True)

        # smooth with forward backward
        log_posteriors, _, _, _, _, _ = FORWARD_BACKWARD[self._estep_type](
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
        (params,) = tree_utils.tree_astype(params, dtype=y.dtype)

        # make sure session_starts starts with a 1
        session_starts = session_starts.at[0].set(True)
        log_proba, _ = FORWARD_PASS[self._estep_type](
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
        (params,) = tree_utils.tree_astype(params, dtype=y.dtype)

        # make sure session_starts starts with a 1
        session_starts = session_starts.at[0].set(True)

        decoded_states = MAX_SUM[self._estep_type](
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

        Decoding walks the time bins one at a time, which suits a CPU core and does not
        suit a GPU: each step is a tiny kernel launch of a few microseconds, so the cost
        is set by the number of bins rather than by the arithmetic, and a long recording
        can take minutes on an accelerator that a CPU finishes in seconds. Unlike the
        E-step, which :class:`~nemos.glm_hmm.GLMHMM`'s ``estep_type`` can switch to a
        parallel scan, this method has no such option yet.

        On a long recording it is therefore often faster to decode on the CPU, even from
        a model fit on the GPU. Placing the inputs on a CPU device is enough, the
        computation following its data:

        >>> import jax  # doctest: +SKIP
        >>> cpu = jax.devices("cpu")[0]  # doctest: +SKIP
        >>> states = model.decode_state(  # doctest: +SKIP
        ...     jax.device_put(X, cpu), jax.device_put(y, cpu)
        ... )

        The transfers cost a fraction of what the loop saves: the inputs are one
        ``(n_samples, n_features)`` array and one of observations, against a per-bin
        launch overhead paid ``n_samples`` times. ``jax.default_device`` is the
        alternative if several calls should run there:

        >>> with jax.default_device(cpu):  # doctest: +SKIP
        ...     states = model.decode_state(X, y)
        """
        params, X, y, session_starts = self._validate_and_prepare_inputs(
            X, y, session_starts
        )
        # validate state_format
        _check_state_format(state_format)
        # define the return type for the max-sum
        return_index = False if state_format == "one-hot" else True
        return self._decode_state(params, X, y, session_starts, return_index)
