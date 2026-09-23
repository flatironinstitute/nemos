"""PP-GLM core log-likelihood computation."""

from functools import partial
from typing import List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike

from .data import MCSamplePPGLM, PredictorsPPGLM, SpikesPPGLM


# SCAN UTILS
@partial(jax.jit, static_argnums=2)
def slice_array(array: jnp.ndarray, i: int, window_size: int):
    """
    Select events within the history window.

    Parameters
    ----------
    array :
        array to slice. Shape (n_events,).
    i :
        index where the reference time point falls within array.
    window_size :
        the number of preceding events to select.

    Returns
    -------
    :
        A slice of recent events. Shape (n_channels, window_size).
    """
    return jax.lax.dynamic_slice(array, (i - window_size,), (window_size,))


def _reshape_2d_coef(coef: jnp.ndarray) -> jnp.ndarray:
    """
    Ensure that coef is a 2d array corresponding to the number of features and neurons.

    If the coef vector is 1d, adds a trailing dimension to it.

    Parameters
    ----------
    coef :
        Flat or 2d weight array. Shape (n_predictors * n_basis_funcs,) or
        (n_predictors * n_basis_funcs, n_neurons).

    Returns
    -------
    :
        Weight matrix. Shape (n_predictors * n_basis_funcs, n_neurons).
    """
    if coef.ndim == 1:
        return coef.reshape(-1, 1)
    elif coef.ndim == 2:
        return coef
    else:
        raise ValueError(
            f"Weights must be either 1d or 2d array, the provided weights have shape {coef.shape}"
        )


def _reshape_and_pad_eval_points(
    eval_pts: SpikesPPGLM | MCSamplePPGLM,
    chunk_size: int,
) -> Tuple[SpikesPPGLM | MCSamplePPGLM, jnp.ndarray]:
    """
    Pad evaluation point time series and reshape into scan chunks of equal size.

    Each field is padded with copies of its last entry. The returned validity mask
    is False on those padded entries, so that their contribution to
    the log-likelihood is dropped.

    Parameters
    ----------
    eval_pts :
        Preprocessed spike / sample times to scan over.
    chunk_size :
        Number of evaluation points processed per scan.

    Returns
    -------
    chunked :
        The padded time series, with every field reshaped to (n_chunks, chunk_size).
    valid :
        False on padded entries, True elsewhere. Shape (n_chunks, chunk_size).
    """
    n_points = eval_pts.times.shape[0]
    pad_len = -n_points % chunk_size

    valid = jnp.ones(n_points, dtype=bool)
    if pad_len:
        eval_pts = jax.tree_util.tree_map(
            lambda arr: jnp.concatenate(
                [arr, jnp.full(pad_len, arr[-1], dtype=arr.dtype)]
            ),
            eval_pts,
        )
        valid = jnp.concatenate([valid, jnp.zeros(pad_len, dtype=bool)])

    chunked = jax.tree_util.tree_map(lambda arr: arr.reshape(-1, chunk_size), eval_pts)
    return chunked, valid.reshape(-1, chunk_size)


def _allocate_samples(lengths: jnp.ndarray, M_samples: int) -> jnp.ndarray:
    """Apportion M_samples over the epochs, one each and the rest by largest remainder."""
    remaining = M_samples - lengths.shape[0]  # after reserving one sample per epoch
    proportional_share = remaining * lengths / lengths.sum()
    whole_share = jnp.floor(proportional_share).astype(int)
    # rank the epochs by the fraction of a sample they were docked, largest first
    remainder_rank = jnp.argsort(jnp.argsort(whole_share - proportional_share))
    leftover = remaining - whole_share.sum()
    # the reserved sample, the whole ones, and one more for each of the epochs that
    # lost the most to the floor; leftover is below n_epochs, so it is always exhausted
    return 1 + whole_share + jnp.where(remainder_rank < leftover, 1, 0)


def build_mc_sampling_grid(epochs: jnp.ndarray, M_samples: int):
    """
    Build a stratified sampling grid for Monte Carlo integration.

    Subdivides each recording epoch into equal-width bins proportionally to its
    length and ensures the total grid size equals M_samples exactly.

    Parameters
    ----------
    epochs :
        Epoch boundaries, ``nap.IntervalSet.values``. Shape (n_epochs, 2).
    M_samples :
        Total number of Monte Carlo sample points.

    Returns
    -------
    grid :
        Left edge of every stratum across all epochs. Shape (M_samples,).
    widths :
        Stratum width at each grid point. Shape (M_samples,).
    """
    if M_samples < epochs.shape[0]:
        raise ValueError(
            f"The number of MC samples ({M_samples}) must be equal or greater than the number of recording "
            f"epochs {epochs.shape[0]})."
        )
    lengths = jnp.diff(epochs, axis=-1).ravel()
    M_sub = _allocate_samples(lengths, M_samples)
    widths = lengths / M_sub  # per-epoch stratum width
    epoch_of_point = jnp.repeat(
        jnp.arange(epochs.shape[0]), M_sub, total_repeat_length=M_samples
    )
    offset = jnp.concatenate([jnp.zeros(1, int), jnp.cumsum(M_sub)[:-1]])
    within = jnp.arange(M_samples) - offset[epoch_of_point]
    grid = epochs[epoch_of_point, 0] + within * widths[epoch_of_point]  # left edges
    return grid, widths[epoch_of_point]


# DATA PREPROCESSING UTILS
@jax.jit
def compute_max_window_size(
    bounds: Union[ArrayLike, List, Tuple],
    ref_spike_times: jnp.ndarray,
    event_times: jnp.ndarray,
):
    """
    Pre-compute the maximum number of events that fall within the history window across all reference spike times.

    Parameters
    ----------
    bounds :
        Two-element array [lower_bound, upper_bound] defining the history window
        relative to a reference spike. Shape (2,).
    ref_spike_times :
        Reference spike times for the target neuron. Shape (n_spikes,).
    event_times :
        Sorted array of all events. Shape (n_events,).

    Returns
    -------
    :
        Maximum number of events within a history window.
    """
    idxs_plus = jnp.searchsorted(event_times, ref_spike_times + bounds[1])
    idxs_minus = jnp.searchsorted(event_times, ref_spike_times + bounds[0])
    within_windows = idxs_plus - idxs_minus
    return jnp.max(within_windows)


@partial(jax.jit, static_argnums=(1, 2))
def adjust_indices_and_spike_times(
    X: PredictorsPPGLM,
    history_window: float,
    max_window: int,
    y: Optional[SpikesPPGLM] = None,
) -> tuple[PredictorsPPGLM, Optional[SpikesPPGLM]]:
    """
    Add padding to the events array so that the history window scan over never goes out of bounds.

    Adds max_window out-of-bound dummy events before the real event times
    and shifts indexing of y spikes to account for this offset (if provided).

    Parameters
    ----------
    X :
        Preprocessed predictor time series to be padded.
    history_window : float
        Duration of the history window (s).
    max_window : int
        The maximum number of events in the history window.
    y :
        Preprocessed postsynaptic spike train.

    Returns
    -------
    shifted_X :
        Padded predictor time series with max_window dummy events prepended.
    shifted_y :
        Spike train with idx shifted by max_window. Only returned if y is not None.
    """
    shifted_X = PredictorsPPGLM(
        times=jnp.concatenate([jnp.full(max_window, -history_window - 1), X.times]),
        predictor_ids=jnp.concatenate(
            [jnp.zeros(max_window, dtype=jnp.int32), X.predictor_ids]
        ),
    )

    shifted_y = None
    if y is not None:
        shifted_y = SpikesPPGLM(
            times=y.times,
            neuron_ids=y.neuron_ids,
            timestamp_idx=y.timestamp_idx + max_window,
        )
    return shifted_X, shifted_y
