"""Utilities for HMM."""

import jax
import jax.numpy as jnp
from numpy.typing import NDArray
from typing import Optional
from ..typing import DESIGN_INPUT_TYPE, ArrayLike
from ..type_casting import is_pynapple_tsd
import pynapple as nap

Array = NDArray | jax.numpy.ndarray


def initialize_new_session(n_samples, is_new_session):
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
    else:
        # use the user-provided tree, but force the first time bin to be True
        is_new_session = jax.lax.dynamic_update_index_in_dim(
            jnp.asarray(is_new_session, dtype=bool), True, 0, axis=0
        )

    return is_new_session


def compute_is_new_session(
    time: NDArray | jnp.ndarray,
    start: NDArray | jnp.ndarray,
    is_nan: Optional[NDArray | jnp.ndarray] = None,
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
    is_new_session = (
        jax.numpy.zeros_like(time).at[jax.numpy.searchsorted(time, start)].set(1)
    )
    if is_nan is not None:
        # set the first element after nan as new session beginning
        is_new_session = is_new_session.at[1:].set(
            jnp.where(is_nan[:-1], 1, is_new_session[1:])
        )
    return is_new_session


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
