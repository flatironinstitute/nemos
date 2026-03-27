from typing import Optional
from numpy.typing import ArrayLike

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from scipy.special import genlaguerre
from itertools import combinations

@partial(jax.jit, static_argnums=2)
def slice_array(array, i, window_size):
    """
        Select events within the history window.

        Parameters
        ----------
        array :
            time series of marked events
        i :
            index where the reference event falls within array
        window_size :
            the number of preceding events to select

        Returns
        -------
        :
            A slice of recent events. Shape (marked event, ).
    """
    return jax.lax.dynamic_slice(array, (0,i - window_size), (2,window_size,))

def reshape_w(weights, n_basis_funcs):
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
def reshape_for_vmap(spikes, n_batches_scan):
    padding_shape = (-spikes.shape[1] % n_batches_scan,)
    padding = jnp.full((spikes.shape[0],) + padding_shape, spikes[:, :1])
    shifted_spikes = jnp.hstack(
        (spikes, padding)
    )
    shifted_spikes_array = shifted_spikes.reshape(spikes.shape[0],n_batches_scan,-1).transpose(2,1,0)

    return shifted_spikes_array, padding.transpose(1,0)

def build_sampling_grid(recording_time, M_samples):
    dt = recording_time.tot_length() / M_samples
    starts, ends = recording_time.start, recording_time.end
    M_sub = jnp.floor((ends - starts) / dt).astype(int)
    M_sub = M_sub.at[-1].set(M_samples - jnp.sum(M_sub[:-1]))
    return jnp.concatenate([jnp.linspace(s + dt, e, m) - dt / 2 for s, e, m in zip(starts, ends, M_sub)])

@jax.jit
def compute_max_window_size(bounds, spike_times, all_spikes):
    """Pre-compute window size for a single neuron"""
    idxs_plus = jnp.searchsorted(all_spikes, spike_times + bounds[1])
    idxs_minus = jnp.searchsorted(all_spikes, spike_times + bounds[0])
    within_windows = idxs_plus - idxs_minus
    return jnp.max(within_windows)

@partial(jax.jit, static_argnums=(1,2))
def adjust_indices_and_spike_times(
        X: ArrayLike,
        history_window: float,
        max_window: int,
        y: Optional=None,
):
    shift = jnp.vstack((jnp.full(max_window, -history_window - 1), jnp.full(max_window, 0)))
    shifted_X = jnp.hstack((shift, X))
    if y is not None:
        shifted_idx = y[-1].astype(int) + max_window
        shifted_y = jnp.vstack((y[:-1], shifted_idx))
        return shifted_X, shifted_y
    else:
        return shifted_X