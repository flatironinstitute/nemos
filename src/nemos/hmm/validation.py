"""Validation mixin class for HMM-based models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import lazy_loader as lazy
import numpy as np

from .. import validation
from ..base_validator import RegressorValidator
from ..type_casting import is_pynapple_tsd
from ..typing import DESIGN_INPUT_TYPE, ArrayLike
from .params import HMMModelParamsT, HMMParams, HMMUserParams, HMMUserProvidedParamsT
from .utils import (
    initialize_session_starts,
    shift_nan_session_starts,
)

nap = lazy.load("pynapple")


def nan_sample_mask(X: DESIGN_INPUT_TYPE, y: Optional[ArrayLike] = None) -> jnp.ndarray:
    """Flag the samples containing NaNs in any leaf of ``X`` or in ``y``.

    Parameters
    ----------
    X :
        Input data/design matrix, shape ``(n_samples, n_features)``, or a pytree of
        such arrays. Pynapple time series are accepted.
    y :
        Output data/observations, shape ``(n_samples, ...)``. If None (e.g. during
        simulation), only ``X`` is inspected.

    Returns
    -------
    :
        Boolean array of shape ``(n_samples,)``, True where any feature or
        observation of that sample is NaN.
    """

    def _is_nan(x):
        x = jnp.asarray(x)
        return jnp.any(jnp.isnan(x.reshape(x.shape[0], -1)), axis=1)

    is_nan = jax.tree_util.tree_reduce(jnp.logical_or, jax.tree.map(_is_nan, X))
    if y is not None:
        is_nan = is_nan | _is_nan(y)
    return is_nan


def has_interior_nans(is_nan: ArrayLike, session_starts: ArrayLike) -> bool:
    """Check if any session holds a NaN sample between two valid samples.

    A session is acceptable when its valid samples form a single contiguous run:
    dropping NaNs at its head or tail shortens the session without reordering it, while
    dropping an interior NaN would splice together bins that are not adjacent in time.

    Counting the runs over the whole recording answers this without a per-session scan.
    A session holding valid samples contributes one run when they are contiguous and at
    least two when a NaN splits them; an all-NaN session contributes none. The total
    number of runs therefore exceeds the number of sessions holding a valid sample if
    and only if some session is split.

    The two counts run on the host: they reduce one boolean per sample, which costs less
    than the ``jax`` dispatch needed to place them on a device.

    Parameters
    ----------
    is_nan :
        Boolean array flagging the NaN samples, shape ``(n_samples,)``, as returned by
        :func:`nan_sample_mask`.
    session_starts :
        Boolean array of session-start indicators, shape ``(n_samples,)``. The first
        element must be True.

    Returns
    -------
    :
        True if at least one session holds a NaN between two valid samples. Sessions
        that are entirely NaN, and sessions holding a single valid sample, hold none.
    """
    is_nan = np.asarray(is_nan)
    session_starts = np.asarray(session_starts)
    is_valid = ~is_nan

    # a run opens at a valid sample with no valid predecessor inside its session, that
    # is one preceded by a NaN or one starting a session. The slices pair sample i with
    # its predecessor i - 1, which leaves out sample 0: it opens a run when it is valid.
    opens_run = is_valid[1:] & (is_nan[:-1] | session_starts[1:])
    n_runs = np.count_nonzero(opens_run) + is_valid[0]

    # reduceat sums is_valid between consecutive session starts, giving one valid-sample
    # count per session, so its non-zero entries are the sessions holding valid samples
    n_valid_per_session = np.add.reduceat(is_valid, np.flatnonzero(session_starts))
    return n_runs > np.count_nonzero(n_valid_per_session)


def to_hmm_params(user_params: HMMUserParams) -> HMMParams:
    """Map from HMMUserParams to HMMParams.

    Converts user-provided parameters (scale and probabilities in regular space)
    to internal model parameters (log_scale and log probabilities).
    """
    return HMMParams(*(jnp.log(p) for p in user_params))


def from_hmm_params(params: HMMParams) -> HMMUserParams:
    """Map from HMMParams to HMMUserParams.

    Converts internal model parameters (log_scale and log probabilities)
    to user-facing parameters (scale and probabilities in regular space).
    """
    # exponentiate and re-normalize
    initial_prob = jnp.exp(params.log_initial_prob)
    initial_prob /= initial_prob.sum()
    transition_prob = jnp.exp(params.log_transition_prob)
    transition_prob /= transition_prob.sum(axis=1, keepdims=True)
    return (
        initial_prob,
        transition_prob,
    )


@dataclass(frozen=True, repr=False)
class HMMValidator(RegressorValidator[HMMUserProvidedParamsT, HMMModelParamsT]):
    """Validate HMM parameters. Meant to be used as a mixin class for models that use HMMs."""

    n_states: int = field(kw_only=True)  # keyword only and required.
    model_param_names: Tuple[str] = ("initial_prob", "transition_prob")
    model_class: str = "HMM"
    params_validation_sequence: Tuple[Tuple[str, None] | Tuple[str, dict[str, Any]]] = (
        ("check_init_and_transition_prob_shape", None),
        ("check_init_and_transition_prob_sum_to_1", None),
    )
    # tuples [(meth, kwargs), ...]; see validate_and_cast_inputs for the step contract
    inputs_validation_sequence: Tuple[
        Tuple[str, None] | Tuple[str, dict[str, Any]], ...
    ] = (
        ("validate_and_cast_session_starts", None),
        ("validate_inputs", None),
        ("check_is_continuous", None),
    )

    def check_user_params_structure(
        self, params: HMMUserProvidedParamsT, **kwargs
    ) -> HMMUserProvidedParamsT:
        """
        Validate that user parameters are a two-element structure.

        Parameters
        ----------
        params :
            User-provided parameters (should be a tuple/list of length 2).
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        :
            The validated parameters.

        Raises
        ------
        ValueError
            If parameters do not have length two.
        """
        validation.check_length(
            params,
            len(self.model_param_names),
            f"Params must have length {len(self.model_param_names)}: "
            f"({', '.join(self.model_param_names)}).",
        )
        if not isinstance(params, (tuple, list)):
            raise TypeError(
                f"{self.model_class} params must be a tuple/list of length {len(self.model_param_names)}, "
                f"({', '.join(self.model_param_names)})."
            )
        return params

    def check_init_and_transition_prob_shape(
        self, params: HMMUserProvidedParamsT
    ) -> HMMUserProvidedParamsT:
        """Check initial and transition probabilities shape."""
        initial_prob, transition_prob = self.wrap_user_params(params)[-2:]
        if initial_prob.shape != (self.n_states,):
            raise ValueError(
                f"initial_prob must be a 1-dimensional array of shape ``({self.n_states},)``. "
                f"Provided initial_prob is of shape ``{initial_prob.shape}`` instead."
            )
        if transition_prob.shape != (self.n_states, self.n_states):
            raise ValueError(
                f"transition_prob must be a 2-dimensional array of shape ``({self.n_states}, {self.n_states})``."
                f"Provided transition_prob is of shape ``{transition_prob.shape}`` instead."
            )
        return params

    def check_init_and_transition_prob_sum_to_1(
        self, params: HMMUserProvidedParamsT
    ) -> HMMUserProvidedParamsT:
        """Check that initial and transition probability sum to 1."""
        initial_prob, transition_prob = self.wrap_user_params(params)[-2:]

        if not jnp.allclose(initial_prob.sum(), 1):
            raise ValueError(
                f"initial_prob must sum to 1, but got sum = {initial_prob.sum()}. "
            )
        if not jnp.allclose(jnp.sum(transition_prob, axis=1), 1):
            row_sums = jnp.sum(transition_prob, axis=1)
            raise ValueError(
                f"transition_prob matrix rows must sum to 1 over columns, but got sum = {row_sums}. "
                f"Each row i represents the probability distribution of transitioning from state i"
                f"and must sum to 1. "
            )
        return params

    def validate_inputs(
        self,
        X: Optional[DESIGN_INPUT_TYPE] = None,
        y: Optional[ArrayLike] = None,
        session_starts: Optional[jnp.ndarray] = None,
    ):
        """
        Validate input data dimensions and sample consistency.

        Checks that X and y have the expected dimensionality (as specified by
        X_dimensionality and y_dimensionality) and that they have the same
        number of samples along axis 0. Also verifies that session_starts has
        the same number of samples as X and y, if provided. This check is redundant
        with validate_and_cast_session_starts, but is added for validation of the
        inputs in custom solver pipelines that bypasses casting session_starts.

        Parameters
        ----------
        X :
            Input data/design matrix, shape ``(n_samples, n_features)``, or a pytree
            of such arrays.
        y :
            Output data/observations, shape ``(n_samples, ...)``. If None (e.g. during
            simulation), only ``X`` is checked.
        session_starts :
            Boolean array of session-start indicators, shape ``(n_samples,)``, as
            returned by :meth:`validate_and_cast_session_starts`.

        Raises
        ------
        ValueError
            If X or y don't have the expected dimensionality.
        ValueError
            If X and y have different number of samples along axis 0.
        ValueError
            If all samples are invalid (contain only NaN/Inf values).
        """
        super().validate_inputs(X=X, y=y)
        # redundant check for session_starts shape for public initialize_optimizer_and_state
        if session_starts is not None:
            n_samples = y.shape[0] if y is not None else jax.tree_util.tree_leaves(X)[0].shape[0]
            if session_starts.shape[0] != n_samples:
                raise ValueError(
                    "session_starts must have the same number of samples as input. "
                    f"input has {n_samples} samples, "
                    f"and session_starts has {session_starts.shape[0]} samples."
                )
            validation.error_all_invalid(session_starts)

    def validate_and_cast_inputs(
        self,
        X: DESIGN_INPUT_TYPE,
        y: Optional[ArrayLike] = None,
        session_starts: Optional[ArrayLike | nap.IntervalSet] = None,
        **validation_kwargs,
    ) -> jnp.ndarray:
        """Run ``inputs_validation_sequence`` in order, returning the cast boundaries.

        Every step is called with the inputs and the session boundaries known so far. A
        step returning a value replaces the boundaries for the steps that follow, which
        is how :meth:`validate_and_cast_session_starts` hands the cast indicators to
        :meth:`check_is_continuous`; a step returning None leaves them untouched.

        Parameters
        ----------
        X :
            Input data/design matrix, shape ``(n_samples, n_features)``, or a pytree
            of such arrays.
        y :
            Output data/observations, shape ``(n_samples, ...)``. If None (e.g. during
            simulation), the checks that need it are skipped.
        session_starts :
            User-provided session boundaries, see
            :func:`~nemos.hmm.utils.initialize_session_starts`.
        **validation_kwargs
            Extra keyword arguments forwarded to every step of the sequence.

        Returns
        -------
        :
            Boolean array of session-start indicators, shape ``(n_samples,)``.
        """
        for method_name, method_kwargs in self.inputs_validation_sequence:
            method_kwargs = {} if method_kwargs is None else method_kwargs
            # Merge default kwargs with any user-provided kwargs
            merged_kwargs = {**method_kwargs, **validation_kwargs}
            out = getattr(self, method_name)(
                X=X, y=y, session_starts=session_starts, **merged_kwargs
            )
            if out is not None:
                session_starts = out

        return session_starts

    def check_is_continuous(
        self,
        X: DESIGN_INPUT_TYPE,
        y: Optional[ArrayLike],
        session_starts: jnp.ndarray,
    ) -> None:
        """Check that each session is a contiguous stretch of valid samples.

        The forward-backward recursions run over the samples of a session in order, so
        a NaN in the middle of a session would break the message passing. NaNs at the
        borders of a session are dropped before inference without altering the
        ordering, and are therefore allowed.

        Sessions are delimited by ``session_starts``, not by the epochs of a pynapple
        input: the boundaries that inference will use are the ones that matter, and the
        two differ whenever the caller passes explicit boundaries alongside a pynapple
        time series.

        Parameters
        ----------
        X :
            Input data/design matrix, shape ``(n_samples, n_features)``, or a pytree
            of such arrays.
        y :
            Output data/observations, shape ``(n_samples, ...)``. If None (e.g. during
            simulation), only ``X`` is checked.
        session_starts :
            Boolean array of session-start indicators, shape ``(n_samples,)``, as
            returned by :meth:`validate_and_cast_session_starts`.

        Raises
        ------
        ValueError
            If any session holds a NaN sample between two valid samples.
        """
        is_nan = np.asarray(nan_sample_mask(X, y))

        # nothing to check when the data holds no NaN
        if not is_nan.any():
            return

        if has_interior_nans(is_nan, session_starts):
            raise ValueError(
                f"{self.model_class} requires continuous time-series data. NaN values must only "
                "appear at the beginning or end of the data, not in the middle. "
                "Found NaN values within the time series, which would break the "
                "forward-backward algorithm. Please ensure your data is continuous "
                "or split it into separate epochs at the gaps."
            )

    def validate_and_cast_session_starts(
        self, X, y, session_starts: Optional[ArrayLike | nap.IntervalSet] = None
    ) -> jnp.ndarray:
        """Validate and cast session_starts to a binary array of shape (n_samples,).

        Parameters
        ----------
        X :
            Input data/design matrix, shape ``(n_samples, n_features)``, or a pytree
            of such arrays.
        y :
            Output data/observations, shape ``(n_samples, ...)``. If None (e.g. during
            simulation), the sample count is read off ``X``.
        session_starts :
            User-provided session boundaries, see
            :func:`~nemos.hmm.utils.initialize_session_starts`. If None, the pynapple
            time support of ``y`` or ``X`` is used when available, otherwise the data
            is treated as a single session.

        Returns
        -------
        :
            Boolean array of session-start indicators, shape ``(n_samples,)``, with the
            markers falling on NaN samples shifted to the next valid sample.
        """
        if session_starts is None:
            if is_pynapple_tsd(y):
                session_starts = y.time_support
            elif is_pynapple_tsd(X):
                session_starts = X.time_support

        session_starts = initialize_session_starts(X, y, session_starts)

        # shift any True values that fall on NaN samples to the next valid sample
        return shift_nan_session_starts(session_starts, nan_sample_mask(X, y))

    def get_empty_params(self, X, y) -> HMMModelParamsT:
        """Return the param shape given the input data."""
        return HMMParams(
            log_initial_prob=jnp.empty((self.n_states,)),
            log_transition_prob=jnp.empty((self.n_states, self.n_states)),
        )
