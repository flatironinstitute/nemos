"""Utilities for HMM."""

import jax
import jax.numpy as jnp
from numpy.typing import NDArray
from ..type_casting import is_pynapple_tsd
import pynapple as nap

Array = NDArray | jax.numpy.ndarray


def initialize_is_new_session(n_samples, is_new_session):
    """
    Initialize and validate session boundary indicators for HMM inference.

    Ensures the session boundary array is properly formatted for HMM algorithms.
    If no session information is provided, treats all samples as a single session.
    Always enforces that the first time bin is marked as a session start.

    Parameters
    ----------
    n_samples :
        Total number of time bins in the data.
    is_new_session :
        Boolean array indicating session starts, shape ``(n_samples,)``.
        If None, creates a default array treating all data as one session.

    Returns
    -------
    is_new_session :
        Validated boolean array with session boundaries, shape ``(n_samples,)``.
        The first element is always True.

    Notes
    -----
    Session boundaries are critical for proper HMM inference as they reset
    the forward and backward message passing at session starts, preventing
    information leakage between independent experimental sessions or trials.
    """
    # Revise if the data is one single session or multiple sessions.
    # If new_sess is not provided, assume one session
    if is_new_session is None:
        # default: all False, but first time bin must be True
        is_new_session = jax.lax.dynamic_update_index_in_dim(
            jnp.zeros(n_samples, dtype=bool), True, 0, axis=0
        )
    elif hasattr(is_new_session, "dtype"):
        if jnp.issubdtype(is_new_session.dtype, jnp.bool_):
            if is_new_session.shape != (n_samples,):
                raise ValueError(
                    f"Boolean is_new_session must have shape (n_samples,), but got shape {is_new_session.shape}."
                )
            # use the user-provided tree, but force the first time bin to be True
            is_new_session = jax.lax.dynamic_update_index_in_dim(
                jnp.asarray(is_new_session, dtype=bool), True, 0, axis=0
            )
        elif jnp.issubdtype(is_new_session.dtype, jnp.integer):
            if (jnp.max(is_new_session) >= n_samples) or (jnp.min(is_new_session) < 0):
                raise ValueError(
                    "Integer is_new_session values must be between 0 and n_samples-1, "
                    f"but got min {jnp.min(is_new_session)} and max {jnp.max(is_new_session)}."
                )
            elif len(is_new_session) > n_samples:
                raise ValueError(
                    f"Integer is_new_session array must have length <= n_samples, but got length {len(is_new_session)} "
                    f"and n_samples {n_samples}."
                )

            if (
                (is_new_session == 0) | (is_new_session == 1)
            ).all() and is_new_session.shape == (n_samples,):
                # should be treated as a boolean array
                is_new_session = jax.lax.dynamic_update_index_in_dim(
                    jnp.asarray(is_new_session, dtype=bool), True, 0, axis=0
                )
            else:
                is_new_session = (
                    jnp.zeros(n_samples, dtype=bool)
                    .at[jnp.asarray(is_new_session, dtype=int)]
                    .set(True)
                )
                is_new_session = jax.lax.dynamic_update_index_in_dim(
                    is_new_session, True, 0, axis=0
                )
        else:
            raise TypeError(
                "is_new_session must be a boolean or integer array, but got dtype "
                f"{is_new_session.dtype}."
            )
    else:
        raise TypeError(
            "is_new_session must be a boolean or integer array, but got type "
            f"{type(is_new_session)}."
        )

    return is_new_session


def compute_is_new_session(
    X: nap.Tsd | nap.TsdFrame,
    y: nap.Tsd | nap.TsdFrame,
    is_new_session: nap.IntervalSet,
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
    if not (is_pynapple_tsd(X) or is_pynapple_tsd(y)):
        raise TypeError(
            "Either X or y must be a pynapple Tsd or TsdFrame to compute session boundaries from "
            "a pynapple.IntervalSet object."
        )
    time = y.t if is_pynapple_tsd(y) else X.t
    start = is_new_session.start
    is_new_session = (
        jax.numpy.zeros_like(time, dtype=bool)
        .at[jax.numpy.searchsorted(time, start)]
        .set(True)
    )
    is_new_session = jax.lax.dynamic_update_index_in_dim(
        is_new_session, True, 0, axis=0
    )
    return is_new_session


def shift_nan_is_new_session(
    is_new_session: jnp.ndarray, is_nan: jnp.ndarray
) -> jnp.ndarray:
    """
    Shift session-start markers off NaN samples to the next valid sample.

    Any ``True`` in ``is_new_session`` that falls on an index marked by ``is_nan``
    is moved forward to the first subsequent index where ``is_nan`` is ``False``.
    Markers already on valid samples are preserved. If no valid sample follows a
    misplaced marker, it is dropped.

    Parameters
    ----------
    is_new_session :
        Boolean array of session-start indicators, shape ``(n_samples,)``.
    is_nan :
        Boolean mask of NaN samples, shape ``(n_samples,)``.

    Returns
    -------
    :
        Updated boolean session-start indicators, shape ``(n_samples,)``.
    """
    n_samples = is_new_session.shape[0]
    indices = jnp.arange(n_samples)

    def body(carry, x):
        is_nan_i, idx = x
        next_valid = jnp.where(is_nan_i, carry, idx)
        return next_valid, next_valid

    # next_valid will contain the index of the next valid sample for each position, or the position itself if it's valid
    _, next_valid = jax.lax.scan(body, n_samples, (is_nan, indices), reverse=True)
    new_idx = next_valid[is_new_session]
    return jnp.zeros(n_samples, dtype=bool).at[new_idx].set(True, mode="drop")


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
