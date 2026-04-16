"""Utilities for HMM."""

import jax
import jax.numpy as jnp
import pynapple as nap
from numpy.typing import ArrayLike, NDArray

from ..type_casting import is_pynapple_tsd
from ..typing import DESIGN_INPUT_TYPE

Array = NDArray | jax.numpy.ndarray


def initialize_is_new_session(
    X: DESIGN_INPUT_TYPE,
    y: ArrayLike,
    is_new_session: ArrayLike | nap.IntervalSet | None = None,
):
    """
    Initialize and validate session boundary indicators for HMM inference.

    Ensures the session boundary array is properly formatted for HMM algorithms.
    If no session information is provided, treats all samples as a single session.
    Always enforces that the first time bin is marked as a session start.

    Parameters
    ----------
    X :
        Input data/design matrix, shape ``(n_samples, n_features)``. Used to infer
        session boundaries if is_new_session is a pynapple.IntervalSet and y is not
        a pynapple Tsd or TsdFrame.
    y :
        Output data/observations, shape ``(n_samples, n_observations)``. Used to get
        n_samples and to infer session boundaries if is_new_session is a pynapple.IntervalSet.
    is_new_session :
        Optional array indicating user-provided session boundaries. Can be:
        - a boolean array indicating session starts, shape ``(n_samples,)``
        - an integer array of indices marking session starts, shape ``(n_sessions,)``
        - a pynapple.IntervalSet marking session epochs (requires either X or y to be a
          pynapple Tsd or TsdFrame to get timestamps)
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
    n_samples = y.shape[0]
    # If new_sess is not provided, assume one session
    if is_new_session is None:
        # default: all False, but first time bin must be True (set at end)
        is_new_session = jnp.zeros(n_samples, dtype=bool)
    elif isinstance(is_new_session, nap.IntervalSet):
        is_new_session = compute_is_new_session_from_pynapple(X, y, is_new_session)
    elif hasattr(is_new_session, "dtype"):
        if jnp.issubdtype(is_new_session.dtype, jnp.bool_):
            if is_new_session.shape != (n_samples,):
                raise ValueError(
                    f"Boolean is_new_session must have shape (n_samples,), but got shape {is_new_session.shape}."
                )
            # force jax boolean array
            is_new_session = jnp.asarray(is_new_session, dtype=bool)

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
                is_new_session = jnp.asarray(is_new_session, dtype=bool)
            else:
                is_new_session = (
                    jnp.zeros(n_samples, dtype=bool)
                    .at[jnp.asarray(is_new_session, dtype=int)]
                    .set(True)
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

    # force first time bin to be True
    is_new_session = jax.lax.dynamic_update_index_in_dim(
        jnp.asarray(is_new_session, dtype=bool), True, 0, axis=0
    )
    return is_new_session


def compute_is_new_session_from_pynapple(
    X: nap.Tsd | nap.TsdFrame,
    y: nap.Tsd | nap.TsdFrame,
    is_new_session: nap.IntervalSet,
) -> jnp.ndarray:
    """Compute indicator vector marking the start of new sessions.

    This function identifies session boundaries in Pynapple time-series data by marking
    positions where new epochs begin.

    Parameters
    ----------
    X :
        Input data/design matrix, shape ``(n_samples, n_features)``. Used for sample time
        stamps if y is not a pynapple Tsd or TsdFrame.
    y :
        Output data/observations, shape ``(n_samples, n_observations)``. Used primarily when
        retrieving sample time stamps.
    is_new_session :
        pynapple.IntervalSet marking the start and end times of sessions, where start times
        are used to identify session boundaries.

    Returns
    -------
    is_new_session :
        Binary indicator array of shape ``(n_time_points,)`` where 1 indicates the start
        of a new session and 0 otherwise.
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
    return is_new_session


def shift_nan_is_new_session(
    is_new_session: jnp.ndarray, is_nan: jnp.ndarray
) -> jnp.ndarray:
    """
    Shift session-start markers off NaN samples to the next valid sample.

    Any ``True`` in ``is_new_session`` that falls on an index marked by ``is_nan``
    is moved forward to the first subsequent index where ``is_nan`` is ``False``.
    Markers already on valid samples are preserved. If no valid sample follows a
    misplaced marker, it is dropped. This ensures that after dropping NaN values,
    session boundaries are preserved.

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
