from numbers import Number
from typing import Union

import jax.numpy as jnp
import jax

from ..glm_hmm.initialize_parameters import (
    _resolve_dirichlet_priors,
)
from .validation import HMMValidator
from .initialize_parameters import setup_hmm_initialization

# from .params import HMMParams


class BaseHMM:
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
        hmm_initialization_funcs: dict | None = None,
    ):
        self._validator_class = None  # to be set by inherited class
        self.n_states = n_states
        # set the prior params
        self.dirichlet_prior_alphas_init_prob = dirichlet_prior_alphas_init_prob
        self.dirichlet_prior_alphas_transition = dirichlet_prior_alphas_transition

        self.seed = seed
        self.maxiter = maxiter
        self.tol = tol

        # fit attributes
        self.transition_prob_: jnp.ndarray | None = None
        self.initial_prob_: jnp.ndarray | None = None

        self.hmm_initialization_funcs = hmm_initialization_funcs
        # self.setup()

    def setup(
        self,
        initial_proba_init: Optional[str | Callable] = None,
        initial_proba_init_kwargs: Optional[dict] = None,
        transition_proba_init: Optional[str | Callable] = None,
        transition_proba_init_kwargs: Optional[dict] = None,
    ):
        self._hmm_initialization_funcs = setup_hmm_initialization(
            initial_proba_init=initial_proba_init,
            initial_proba_init_kwargs=initial_proba_init_kwargs,
            transition_proba_init=transition_proba_init,
            transition_proba_init_kwargs=transition_proba_init_kwargs,
            init_funcs=self._hmm_initialization_funcs,
        )

    @property
    def n_states(self) -> int:
        """Number of hidden states of the HMM."""
        return self._n_states

    @n_states.setter
    def n_states(self, n_states: int):
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
        self._validator = HMMValidator(n_states=n_states)

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
                f" {missing_params}.\nPlease fit the GLM-HMM model first or "
                "set the missing attributes."
            )

    @staticmethod
    def _get_is_new_session(
        X: DESIGN_INPUT_TYPE, y: ArrayLike | nap.Tsd | nap.TsdFrame | None = None
    ) -> jnp.ndarray | None:
        """Compute session boundary indicators for HMM time-series data.

        Identifies session boundaries by detecting epoch starts and gaps in the data
        (represented by NaN values in either predictors or response). This is essential
        for HMM models to properly segment time series data and reset the hidden
        state between discontinuous recordings.

        Parameters
        ----------
        X :
            Design matrix or predictor time series. Can be a pynapple Tsd/TsdFrame or
            array-like of shape ``(n_time_points, n_features)``.
        y :
            Response variable time series of shape ``(n_time_points,)`` or
            ``(n_time_points, n_neurons)``. If None, NaN detection is based on X only
            (useful for simulation where y is not available).

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
        the length of X (or y if provided).

        See Also
        --------
        compute_is_new_session : Core function for computing session indicators.
        """
        # compute the nan location along the sample axis
        nan_x = jnp.any(jnp.isnan(jnp.asarray(X)).reshape(X.shape[0], -1), axis=1)
        if y is not None:
            nan_y = jnp.any(jnp.isnan(jnp.asarray(y)).reshape(y.shape[0], -1), axis=1)
            combined_nans = nan_y | nan_x
        else:
            combined_nans = nan_x

        # define new session array
        if y is not None and is_pynapple_tsd(y):
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

    def _score(
        self,
        params: GLMHMMParams,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: Union[NDArray, jnp.ndarray, nap.Tsd],
        is_new_session: jnp.ndarray,
    ) -> jnp.ndarray:
        """Private score compute."""
        # filter for non-nans, grab data if needed
        data, y, is_new_session = self._preprocess_inputs(X, y, is_new_session)
        # safe conversion to jax arrays of float
        params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, y.dtype), params)

        # make sure is_new_session starts with a 1
        is_new_session = is_new_session.at[0].set(True)

        # smooth with forward backward
        _, log_norm = forward_pass(
            params=params,
            X=data,
            y=y,
            is_new_session=is_new_session,
            log_likelihood_func=prepare_estep_log_likelihood(
                y.ndim > 1, self.observation_model
            ),
            inverse_link_function=self._inverse_link_function,
        )
        return jnp.sum(log_norm)

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
        if score_type == "log-likelihood" and null_model is not None:
            warnings.warn(
                "The null model is not used for the log-likelihood computation.",
                UserWarning,
                stacklevel=2,
            )
        if score_type != "log-likelihood":
            raise NotImplementedError(
                f"score of type {score_type} not implemented yet!"
            )
        params, X, y, is_new_session = self._validate_and_prepare_inputs(X, y)
        return self._score(params, X, y, is_new_session)

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

    @support_pynapple(conv_type="jax")
    def _filter_proba(
        self,
        params: GLMHMMParams,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: Union[NDArray, jnp.ndarray, nap.Tsd],
        is_new_session: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute filtering probabilities without validation (internal method)."""
        # filter for non-nans, grab data if needed
        valid = tree_utils.get_valid_multitree(X, y)
        data, y, is_new_session = self._preprocess_inputs(X, y, is_new_session)

        # safe conversion to jax arrays of float
        params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, y.dtype), params)

        # make sure is_new_session starts with a 1
        is_new_session = is_new_session.at[0].set(True)
        log_proba, _ = forward_pass(
            params,
            data,
            y,
            inverse_link_function=self._inverse_link_function,
            is_new_session=is_new_session,
            log_likelihood_func=prepare_estep_log_likelihood(
                y.ndim > 1, self.observation_model
            ),
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
            Observed neural activity, shape ``(n_time_points,)`` for single neuron or
            ``(n_time_points, n_neurons)`` for population.

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
        smooth_proba : Compute smoothing posteriors (conditioned on all observations).
        decode_state : Compute most likely state sequence (Viterbi decoding).

        Notes
        -----
        - Filtering provides causal state estimates suitable for online/real-time applications
        - Smoothing provides better estimates but requires all data (non-causal)
        - The algorithm properly handles session boundaries and NaN values at epoch borders
        - NaN values are removed before inference, but session markers are preserved
        - For pynapple inputs, the output TsdFrame has columns named "state_0", "state_1", etc.

        Examples
        --------
        Fit a GLM-HMM and compute filtering posteriors:

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
        >>>
        >>> # Compute filtering posteriors (causal/online)
        >>> filter_posteriors = model.filter_proba(X, y)
        >>> print(filter_posteriors.shape)
        (100, 3)
        >>> # Each row sums to 1
        >>> print(np.allclose(filter_posteriors.sum(axis=1), 1.0))
        True

        Using with pynapple for real-time state estimation:

        >>> import pynapple as nap
        >>> # Create time-indexed data
        >>> t = np.arange(100) * 0.01  # 10ms bins
        >>> X_tsd = nap.TsdFrame(t=t, d=X)
        >>> y_tsd = nap.Tsd(t=t, d=y)
        >>>
        >>> # Filtering posteriors returned as TsdFrame
        >>> filter_tsd = model.filter_proba(X_tsd, y_tsd)
        >>> print(type(filter_tsd))
        <class 'pynapple.core.time_series.TsdFrame'>
        """
        params, X, y, is_new_session = self._validate_and_prepare_inputs(X, y)
        return self._filter_proba(params, X, y, is_new_session)

    @support_pynapple(conv_type="jax")
    def _decode_state(
        self,
        params: GLMHMMParams,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: Union[NDArray, jnp.ndarray, nap.Tsd],
        is_new_session: jnp.ndarray,
        return_index: bool,
    ) -> jnp.ndarray:
        """Decode most likely state sequence without validation (internal method)."""
        # filter for non-nans, grab data if needed
        valid = tree_utils.get_valid_multitree(X, y)
        data, y, is_new_session = self._preprocess_inputs(X, y, is_new_session)

        # safe conversion to jax arrays of float
        params = jax.tree_util.tree_map(lambda x: jnp.asarray(x, y.dtype), params)

        # make sure is_new_session starts with a 1
        is_new_session = is_new_session.at[0].set(True)

        decoded_states = max_sum(
            params,
            data,
            y,
            inverse_link_function=self._inverse_link_function,
            is_new_session=is_new_session,
            log_likelihood_func=prepare_estep_log_likelihood(
                y.ndim > 1, self.observation_model
            ),
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
            Observed neural activity, shape ``(n_time_points,)`` for single neuron or
            ``(n_time_points, n_neurons)`` for population.
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
        smooth_proba : Compute smoothing posteriors (conditioned on all observations).
        filter_proba : Compute filtering posteriors (conditioned on past observations only).

        Notes
        -----
        - Viterbi decoding finds the globally optimal state sequence, not the sequence
          of individually most likely states from ``smooth_proba()``
        - This is a hard assignment (single best path) unlike probabilistic posteriors
        - The algorithm properly handles session boundaries and NaN values at epoch borders
        - Decoding is useful for segmenting continuous data into discrete behavioral states
        - For uncertainty estimates about states, use ``smooth_proba()`` instead

        Examples
        --------
        Fit a GLM-HMM and decode the most likely state sequence:

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
        >>>
        >>> # Decode states as one-hot encoding
        >>> states_onehot = model.decode_state(X, y, state_format="one-hot")
        >>> print(states_onehot.shape)
        (100, 3)
        >>> # Each row has exactly one 1
        >>> print(np.all(states_onehot.sum(axis=1) == 1))
        True
        >>>
        >>> # Decode states as integer indices
        >>> states_idx = model.decode_state(X, y, state_format="index")
        >>> print(states_idx.shape)
        (100,)
        >>> print(states_idx[:10])  # First 10 decoded states
        [...]

        Using with pynapple for time-series state decoding:

        >>> import pynapple as nap
        >>> # suppress jax to numpy conversion warning
        >>> nap.nap_config.suppress_conversion_warnings = True
        >>> # Create time-indexed data
        >>> t = np.arange(100) * 0.01  # 10ms bins
        >>> X_tsd = nap.TsdFrame(t=t, d=X)
        >>> y_tsd = nap.Tsd(t=t, d=y)
        >>>
        >>> # Decoded states returned as TsdFrame or Tsd
        >>> states_tsd = model.decode_state(X_tsd, y_tsd, state_format="one-hot")
        >>> print(type(states_tsd))
        <class 'pynapple.core.time_series.TsdFrame'>
        >>> # With index format, returns Tsd
        >>> states_idx_tsd = model.decode_state(X_tsd, y_tsd, state_format="index")
        >>> print(type(states_idx_tsd))
        <class 'pynapple.core.time_series.Tsd'>
        """
        params, X, y, is_new_session = self._validate_and_prepare_inputs(X, y)
        # validate state_format
        _check_state_format(state_format)
        # define the return type for the max-sum
        if state_format == "one-hot":
            return_index = False
        else:
            return_index = True
        return self._decode_state(params, X, y, is_new_session, return_index)
