from functools import partial
from typing import List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike
from pynapple import IntervalSet

from ..typing import DESIGN_INPUT_TYPE


### SCAN UTILS
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
    return jax.lax.dynamic_slice(
        array,
        (i - window_size,),
        (window_size,)
    )


def reshape_coef_for_scan(weights: jnp.ndarray, n_basis_funcs: int):
    """
    Reshape weight array into (n_predictors, n_basis_funcs, n_neurons)
    format expected by the scan loop.

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
def reshape_input_for_scan(times: tuple, scan_size: int):
    """
    Reshape time series into scan inputs of equal size. Pad the last input with copies of
    the last time point if needed.

    Parameters
    ----------
    times : tuple of jnp.ndarray, length n_channels
        Marked time series to scan over. Each array has shape (n_time_points,).
    scan_size :
        the number of time points to process in each scan

    Returns
    -------
    padded_times_reshaped : tuple of jnp.ndarray, length n_channels
        Reshaped padded input. Each array has shape (n_scans, scan_size).
    padding_value: tuple of jnp.ndarray, length n_channels
        The last time point.
    padding_len :
        Number of padding time points appended to make n_points divisible by scan_size.
    """
    def reshape_one(arr):
        padding_len = -arr.shape[0] % scan_size
        padded = jnp.concatenate([arr, jnp.full((padding_len,), arr[-1])])
        return padded.reshape(-1, scan_size)  # (n_scans, scan_size)

    padding_len = -times[0].shape[0] % scan_size
    padding_values = tuple(arr[-1] for arr in times)
    reshaped = tuple(reshape_one(arr) for arr in times)

    return reshaped, padding_values, padding_len


def build_mc_sampling_grid(recording_time: IntervalSet, M_samples: int):
    """
    Build a stratified sampling grid of bin midpoints for Monte Carlo integration
    of the conditional intensity function over the recording.

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


### DATA PREPROCESSING UTILS
@jax.jit
def compute_max_window_size(
    bounds: Union[ArrayLike, List, Tuple],
    ref_spike_times: jnp.ndarray,
    event_times: jnp.ndarray,
):
    """
    Pre-compute the maximum number of events that fall within the history window
    across all reference spike times.

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
    X: DESIGN_INPUT_TYPE,
    history_window: float,
    max_window: int,
    y: Optional[tuple] = None,
):
    """
    Add padding to the events array so that history window selection near
    the start of the recording never goes out of bounds.

    Adds max_window out-of-bound dummy events before the real event times
    and shifts indexing of y spikes to  account for this offset (if provided).

    Parameters
    ----------
    X :
        Event time array to be padded. Shape (2, n_events).
    history_window :
        Duration of the history window (s).
    max_window :
        Number of dummy events to prepend.
    y :
        Spike time series: (times: float (n_spikes,), neuron_ids: int (n_spikes,),
        event_indices: int (n_spikes,)).

    Returns
    -------
    shifted_X :
        Padded event array. Shape (2, max_window + n_events).
    shifted_y :
        Index-corrected spike time array (only returned if y is not None).
        Shape (3, n_spikes).
    """
    shifted_X = (
        jnp.concatenate([jnp.full(max_window, -history_window - 1), X[0]]),
        jnp.concatenate([jnp.zeros(max_window, dtype=jnp.int32), X[1]]),
    )
    if y is not None:
        shifted_y = (
            y[0],
            y[1],
            y[2] + max_window,
        )
        return shifted_X, shifted_y
    else:
        return shifted_X
