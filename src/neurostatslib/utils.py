# required to get ArrayLike to render correctly, unnecessary as of python 3.10
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
from typing import Union, Optional
from functools import partial
import matplotlib.pyplot as plt

# Broadcasted 1d convolution operations.
# [[n x t],[w]] -> [n x (t - w + 1)]
_CORR1 = jax.vmap(partial(jnp.convolve, mode='valid'), (0, None), 0)
# [[n x t],[p x w]] -> [n x p x (t - w + 1)]
_CORR2 = jax.vmap(_CORR1, (None, 0), 0)


def convolve_1d_basis(basis_matrix: ArrayLike,
                      time_series: ArrayLike) -> ArrayLike:
    """Parameters
    ----------
    basis_matrix : (n_basis_funcs, window_size)
    	Matrix holding 1d basis functions.
    time_series : (n_neurons, n_timebins)
        Matrix holding multivariate time series.

    Returns
    -------
    convolved_series : (n_neurons, n_basis_funcs, n_timebins - window_size + 1)
        Result of convolution between all pairs of
        features and basis functions.

    Notes
    -----
    For example, ``time_series`` could be a matrix of spike counts with
    ``n_neurons`` neurons and ``n_timebins`` timebins, and ``basis_matrix``
    could be a matrix of ``n_basis_funcs`` temporal basis functions with a
    window size of ``window_size``.

    """
    return _CORR2(
    	jnp.atleast_2d(basis_matrix),
    	jnp.atleast_2d(time_series)
    )


def plot_spike_raster(spike_data: Union[jnp.ndarray, np.ndarray],
                      lineoffsets: Union[None, float, ArrayLike] = None,
                      linelengths: Union[float, ArrayLike] = .2,
                      linewidths: Union[float, ArrayLike] = .5,
                      ax: Optional[plt.Axes] = None,
                      **kwargs) -> plt.Axes:
    """Plot decent looking spike raster.

    We set ``yticks=[]`` and ``xlim=[0, spike_data.shape[1]]``.

    Parameters
    ----------
    spike_data
        2d array of spikes, (neurons, time_bins). Should contain integers where
        spikes occur.
    lineoffsets
        The offset of the center of the lines from the origin, in the direction
        orthogonal to orientation. This can be a sequence with length matching
        the length of positions. If None, we use np.arange(spike_data.shape[0])
    linelengths
        The total height of the lines (i.e. the lines stretches from lineoffset
        - linelength/2 to lineoffset + linelength/2). This can be a sequence
        with length matching the length of positions.
    linewidths
        The line width(s) of the event lines, in points. This can be a sequence
        with length matching the length of positions.
    ax
        Axes to plot on. If None, we create a new one-axis figure with
        ``figsize=(spike_data.shape[1]/100, spike_data.shape[0]/5)``
    kwargs
        Passed to ``ax.eventplot``

    Returns
    -------
    ax
        Axis containing the raster plot

    Raises
    ------
    ValueError
        If ``spike_data.ndim!=2``

    """
    if spike_data.ndim != 2:
        raise ValueError(f"spike_data should be 2d, but got {spike_data.ndim}d instead!")
    events = [d.nonzero()[0] for d in spike_data]
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(spike_data.shape[1]/100, spike_data.shape[0]/5))
    if lineoffsets is None:
        lineoffsets = jnp.arange(len(events))
    ax.eventplot(events, lineoffsets=lineoffsets, linelengths=linelengths, linewidths=linewidths, **kwargs)
    ax.set(yticks=[], xlim=[0, spike_data.shape[1]])
    return ax
