from typing import Optional
from numpy.typing import ArrayLike

from functools import partial

import jax
import jax.numpy as jnp

### SCAN UTILS
@partial(jax.jit, static_argnums=2)
def slice_array(array, i, window_size):
    """
         Select events within the history window.

         Parameters
         ----------
         array :
             event time series. Shape (n_channels, n_events).
         i :
             index where the reference time point falls within array.
         window_size :
             the number of preceding events to select.

         Returns
         -------
         :
             A slice of recent events. Shape (n_channels, window_size).
     """
    n_channels = array.shape[0]
    return jax.lax.dynamic_slice(array, (0, i - window_size), (n_channels, window_size,))

def reshape_coef_for_scan(weights, n_basis_funcs):
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
def reshape_input_for_scan(times, scan_size):
    """
        Reshape time series into scan inputs of equal size. Pad the last input with copies of
        the last time point if needed.

        Parameters
        ----------
        times :
            time series to scan over. Shape (n_channels, n_time_points)
        scan_size :
            the number of time points to process in each scan

        Returns
        -------
        padded_times_reshaped :
            Reshaped padded input. Shape (n_scans, scan_size, n_channels).
        padding_value:
            The last time point. Shape (n_channels,)
        padding_len :
            Number of padding time points appended to make n_points divisible by scan_size.
    """
    n_channels = times.shape[0]
    padding_value = times[:, -1]
    padding_len = -times.shape[1] % scan_size
    padding = jnp.full((n_channels,) + (padding_len,), padding_value[:,None])
    padded_spikes = jnp.hstack(
        (times, padding)
    )
    padded_times_reshaped = padded_spikes.reshape(times.shape[0], scan_size,-1).transpose(2,1,0)

    return padded_times_reshaped, padding_value, padding_len

def build_mc_sampling_grid(recording_time, M_samples):
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
    dt = recording_time.tot_length() / M_samples
    starts, ends = recording_time.start, recording_time.end
    M_sub = jnp.floor((ends - starts) / dt).astype(int)
    M_sub = M_sub.at[-1].set(M_samples - jnp.sum(M_sub[:-1]))
    return jnp.concatenate([jnp.linspace(s + dt, e, m) - dt / 2 for s, e, m in zip(starts, ends, M_sub)])


### DATA PREPROCESSING UTILS
@jax.jit
def compute_max_window_size(bounds, ref_spike_times, event_times):
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

@partial(jax.jit, static_argnums=(1,2))
def adjust_indices_and_spike_times(
        X: ArrayLike,
        history_window: float,
        max_window: int,
        y: Optional=None,
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
            Spike time array whose last row contains integer indices into X.
            Shape (3, n_spikes). If provided, indices are shifted by max_window.

        Returns
        -------
        shifted_X :
            Padded event array. Shape (2, max_window + n_events).
        shifted_y :
            Index-corrected spike time array (only returned if y is not None).
            Shape (3, n_spikes).
    """
    shift = jnp.vstack((jnp.full(max_window, -history_window - 1), jnp.full(max_window, 0)))
    shifted_X = jnp.hstack((shift, X))
    if y is not None:
        shifted_idx = y[-1].astype(int) + max_window
        shifted_y = jnp.vstack((y[:-1], shifted_idx))
        return shifted_X, shifted_y
    else:
        return shifted_X