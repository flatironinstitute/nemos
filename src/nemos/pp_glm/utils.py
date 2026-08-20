"""PP-GLM core log-likelihood computation."""

from functools import partial
from typing import List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import pynapple as nap
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


def reshape_coef_for_scan(weights: jnp.ndarray, n_basis_funcs: int):
    """
    Reshape weight array into (n_predictors, n_basis_funcs, n_neurons) format expected by the scan loop.

    Parameters
    ----------
    weights :
        Flat or 2d weight array. Shape (n_predictors * n_basis_funcs,) or
        (n_predictors * n_basis_funcs, n_neurons).
    n_basis_funcs :
        Number of basis functions per source neuron.

    Returns
    -------
    :
        Reshaped weights. Shape (n_predictors, n_basis_funcs, n_neurons).
    """
    if len(weights.shape) == 1:
        return weights.reshape(-1, n_basis_funcs, 1)
    elif len(weights.shape) == 2:
        n_target_neurons = weights.shape[1]
        return weights.reshape(-1, n_basis_funcs, n_target_neurons)
    else:
        raise ValueError(
            f"Weights must be either 1d or 2d array, the provided weights have shape {weights.shape}"
        )


@partial(jax.jit, static_argnums=1)
def reshape_input_for_scan(data: SpikesPPGLM | MCSamplePPGLM, scan_size: int):
    """
    Reshape time series into scan inputs of equal size. Pad the last input with copies of the last time point if needed.

    Parameters
    ----------
    data :
        Preprocessed spike / sample times to scan over.
    scan_size :
        the number of time points to process in each scan

    Returns
    -------
    reshaped :
        Reshaped padded input. Each field has shape (n_scans, scan_size).
    padding_values :
        The last value of each field.
    padding_len :
        Number of padding time points appended to make n_points divisible by scan_size.
    """

    def reshape_one(arr):
        padding_len = -arr.shape[0] % scan_size
        padded = jnp.concatenate([arr, jnp.full((padding_len,), arr[-1])])
        return padded.reshape(-1, scan_size)

    padding_len = -data.times.shape[0] % scan_size
    padding_values = jax.tree_util.tree_map(lambda arr: arr[-1], data)
    reshaped = jax.tree_util.tree_map(reshape_one, data)

    return reshaped, padding_values, padding_len


def build_mc_sampling_grid(recording_time: nap.IntervalSet, M_samples: int):
    """
    Build a stratified sampling grid for Monte Carlo integration.

    Subdivides each recording epoch into equal-width bins proportionally to its
    length and ensures the total grid size equals M_samples exactly.

    Parameters
    ----------
    recording_time :
        pynapple IntervalSet defining the recording epochs.
    M_samples :
        Total number of Monte Carlo sample points.

    Returns
    -------
    :
        Concatenated grid of bin midpoints across all epochs. Shape (M_samples,).
    """
    if M_samples < len(recording_time.start):
        raise ValueError(
            f"The number of MC samples ({M_samples}) must be equal or greater than the number of recording "
            f"epochs {len(recording_time.start)})."
        )
    dt = recording_time.tot_length() / M_samples
    starts, ends = recording_time.start, recording_time.end
    M_sub = jnp.floor((ends - starts) / dt).astype(int)
    M_sub = M_sub.at[-1].set(M_samples - jnp.sum(M_sub[:-1]))
    return jnp.concatenate(
        [jnp.linspace(s + dt, e, m) - dt / 2 for s, e, m in zip(starts, ends, M_sub)]
    )


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
