"""Utilities for HMM."""

import jax
import jax.numpy as jnp
from numpy.typing import NDArray

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
