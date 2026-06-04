from functools import partial
from typing import List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import pynapple as nap
from numpy.typing import ArrayLike

from .log_likelihood import X_ppglm, mc_sample_ppglm, y_ppglm


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
    return jax.lax.dynamic_slice(array, (i - window_size,), (window_size,))


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
def reshape_input_for_scan(data: y_ppglm | mc_sample_ppglm, scan_size: int):
    """
    Reshape time series into scan inputs of equal size. Pad the last input with copies of
    the last time point if needed.

    Parameters
    ----------
    data :
        Preprocessed spike / sample times to scan over.
    scan_size :
        the number of time points to process in each scan

    Returns
    -------
    reshaped : y_ppglm
        Reshaped padded input. Each field has shape (n_scans, scan_size).
    padding_values : y_ppglm
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
def to_tsgroup(time_series) -> nap.TsGroup:
    """Convert various spike timestamp formats to a re-indexed TsGroup.

    If time_series is a pynapple object, the output will preserve its time support.
    """

    error_message = "All time series must be non-empty. "

    # --- TsGroup ---
    if isinstance(time_series, nap.TsGroup):
        if any(len(ts) == 0 for ts in time_series.values()):
            raise ValueError(error_message)
        return nap.TsGroup(
            {i: ts for i, ts in enumerate(time_series.values())},
            time_support=time_series.time_support,
        )

    # --- single Ts ---
    if isinstance(time_series, nap.Ts):
        if len(time_series) == 0:
            raise ValueError(error_message)
        return nap.TsGroup({0: time_series}, time_support=time_series.time_support)

    # --- dict ---
    if isinstance(time_series, dict):
        if any(len(arr) == 0 for arr in time_series.values()):
            raise ValueError(error_message)
        return nap.TsGroup(
            {i: nap.Ts(np.asarray(arr)) for i, arr in enumerate(time_series.values())}
        )

    # --- np.ndarray or list ---
    if isinstance(time_series, (np.ndarray, list)):
        if len(time_series) > 0 and np.isscalar(time_series[0]):
            times = np.asarray(time_series, dtype=float).ravel()
            if len(times) == 0:
                raise ValueError(error_message)
            return nap.TsGroup({0: nap.Ts(times)})
        else:
            if any(len(s) == 0 for s in time_series):
                raise ValueError(error_message)
            return nap.TsGroup(
                {
                    i: nap.Ts(np.asarray(s, dtype=float))
                    for i, s in enumerate(time_series)
                }
            )

    raise TypeError(
        f"Unsupported type for input: {type(time_series)}. "
        "Expected np.ndarray, list, dict, pynapple.Ts, or pynapple.TsGroup."
    )


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
    X: X_ppglm,
    history_window: float,
    max_window: int,
    y: Optional[y_ppglm] = None,
) -> tuple[X_ppglm, Optional[y_ppglm]]:
    """
    Add padding to the events array so that history window selection near
    the start of the recording never goes out of bounds.

    Adds max_window out-of-bound dummy events before the real event times
    and shifts indexing of y spikes to account for this offset (if provided).

    Parameters
    ----------
    X : X_ppglm
        Preprocessed predictor time series to be padded.
    history_window : float
        Duration of the history window (s).
    max_window : int
        The maximum number of events in the history window.
    y : y_ppglm, optional
        Preprocessed postsynaptic spike train.

    Returns
    -------
    shifted_X : X_ppglm
        Padded predictor time series with max_window dummy events prepended.
    shifted_y : y_ppglm, optional
        Spike train with idx shifted by max_window. Only returned if y is not None.
    """
    shifted_X = X_ppglm(
        times=jnp.concatenate([jnp.full(max_window, -history_window - 1), X.times]),
        ids=jnp.concatenate([jnp.zeros(max_window, dtype=jnp.int32), X.ids]),
    )
    if y is not None:
        shifted_y = y_ppglm(
            times=y.times,
            ids=y.ids,
            idx=y.idx + max_window,
        )
        return shifted_X, shifted_y
    return shifted_X, None
