"""Utilities for HMM."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import lazy_loader as lazy
from numpy.typing import ArrayLike, NDArray

from ..type_casting import is_pynapple_tsd
from ..typing import DESIGN_INPUT_TYPE

Array = NDArray | jax.numpy.ndarray

nap = lazy.load("pynapple")

if TYPE_CHECKING:
    import pynapple as nap


def initialize_session_starts(
    X: DESIGN_INPUT_TYPE,
    y: ArrayLike | None,
    session_starts: ArrayLike | nap.IntervalSet | None = None,
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
        session boundaries if session_starts is a pynapple.IntervalSet and y is not
        a pynapple Tsd or TsdFrame.
    y :
        Output data/observations, shape ``(n_samples, n_observations)``. Used to get
        n_samples and to infer session boundaries if session_starts is a pynapple.IntervalSet.
        If None (e.g. during simulation), n_samples is inferred from X.
    session_starts :
        Optional array indicating user-provided session boundaries. Can be:
        - a boolean array or integer array of 1s and 0s indicating session starts, shape ``(n_samples,)``
        - an integer array of indices marking session starts, shape ``(n_sessions,)``
        - a pynapple.IntervalSet marking session epochs (requires either X or y to be a
          pynapple Tsd or TsdFrame to get timestamps)
        If None, creates a default array treating all data as one session.

    Returns
    -------
    session_starts :
        Validated boolean array with session boundaries, shape ``(n_samples,)``.
        The first element is always True.

    Notes
    -----
    Session boundaries are critical for proper HMM inference as they reset
    the forward and backward message passing at session starts, preventing
    information leakage between independent experimental sessions or trials.
    """
    n_samples = (
        y.shape[0] if y is not None else jax.tree_util.tree_leaves(X)[0].shape[0]
    )
    # If new_sess is not provided, assume one session
    if session_starts is None:
        # default: all False, but first time bin must be True (set at end)
        session_starts = jnp.zeros(n_samples, dtype=bool)
    elif hasattr(session_starts, "start") and hasattr(session_starts, "end"):
        session_starts = compute_session_starts_from_pynapple(X, y, session_starts)
    elif hasattr(session_starts, "dtype"):
        if jnp.issubdtype(session_starts.dtype, jnp.bool_):
            if session_starts.shape != (n_samples,):
                raise ValueError(
                    f"Boolean session_starts must have shape (n_samples,), but got shape {session_starts.shape}."
                )
            # force jax boolean array
            session_starts = jnp.asarray(session_starts, dtype=bool)

        elif jnp.issubdtype(session_starts.dtype, jnp.integer):
            if (jnp.max(session_starts) >= n_samples) or (jnp.min(session_starts) < 0):
                raise ValueError(
                    "Integer session_starts values must be between 0 and n_samples-1, "
                    f"but got min {jnp.min(session_starts)} and max {jnp.max(session_starts)}."
                )
            elif len(session_starts) > n_samples:
                raise ValueError(
                    f"Integer session_starts array must have length <= n_samples, but got length {len(session_starts)} "
                    f"and n_samples {n_samples}."
                )

            if (
                (session_starts == 0) | (session_starts == 1)
            ).all() and session_starts.shape == (n_samples,):
                # should be treated as a boolean array
                session_starts = jnp.asarray(session_starts, dtype=bool)
            else:
                session_starts = (
                    jnp.zeros(n_samples, dtype=bool)
                    .at[jnp.asarray(session_starts, dtype=int)]
                    .set(True)
                )
        else:
            raise TypeError(
                "session_starts must be a boolean or integer array, but got dtype "
                f"{session_starts.dtype}."
            )
    else:
        raise TypeError(
            "session_starts must be a boolean or integer array, but got type "
            f"{type(session_starts)}."
        )

    # force first time bin to be True
    session_starts = jax.lax.dynamic_update_index_in_dim(
        jnp.asarray(session_starts, dtype=bool), True, 0, axis=0
    )
    return session_starts


def compute_session_starts_from_pynapple(
    X: nap.Tsd | nap.TsdFrame,
    y: nap.Tsd | nap.TsdFrame,
    session_starts: nap.IntervalSet,
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
    session_starts :
        pynapple.IntervalSet marking the start and end times of sessions, where start times
        are used to identify session boundaries.

    Returns
    -------
    session_starts :
        Binary indicator array of shape ``(n_time_points,)`` where 1 indicates the start
        of a new session and 0 otherwise.
    """
    if not (is_pynapple_tsd(X) or is_pynapple_tsd(y)):
        raise TypeError(
            "Either X or y must be a pynapple Tsd or TsdFrame to compute session boundaries from "
            "a pynapple.IntervalSet object."
        )
    time = y.t if is_pynapple_tsd(y) else X.t
    start = session_starts.start
    session_starts = (
        jax.numpy.zeros_like(time, dtype=bool)
        .at[jax.numpy.searchsorted(time, start)]
        .set(True)
    )
    return session_starts


def shift_nan_session_starts(
    session_starts: jnp.ndarray, is_nan: jnp.ndarray
) -> jnp.ndarray:
    """
    Shift session-start markers off NaN samples to the next valid sample.

    Any ``True`` in ``session_starts`` that falls on an index marked by ``is_nan``
    is moved forward to the first subsequent index where ``is_nan`` is ``False``.
    Markers already on valid samples are preserved. If no valid sample follows a
    misplaced marker, it is dropped. This ensures that after dropping NaN values,
    session boundaries are preserved.

    Parameters
    ----------
    session_starts :
        Boolean array of session-start indicators, shape ``(n_samples,)``.
    is_nan :
        Boolean mask of NaN samples, shape ``(n_samples,)``.

    Returns
    -------
    :
        Updated boolean session-start indicators, shape ``(n_samples,)``.
    """
    n = session_starts.shape[0]
    valid = jnp.arange(n)[~is_nan]  # non-NaN indices
    invalid_session_idx = jnp.arange(n)[is_nan & session_starts]  # misplaced markers

    # Append sentinel n: searchsorted returning len(valid) (no valid sample after
    # the marker) then maps to n, which mode="drop" discards. The append copies
    # valid but avoids any clip/where logic and handles empty valid correctly.
    pos = jnp.searchsorted(valid, invalid_session_idx, side="left")
    shifted_idx = jnp.append(valid, n)[pos]

    return (
        session_starts.at[invalid_session_idx]
        .set(False)
        .at[shifted_idx]
        .set(True, mode="drop")
    )


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
