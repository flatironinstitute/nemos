"""Utility functions for data pre-processing
"""
# required to get ArrayLike to render correctly, unnecessary as of python 3.10
from __future__ import annotations

from functools import partial
from typing import Optional, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray


def convolve_1d_trials(basis_matrix: ArrayLike, trials_time_series: ArrayLike[NDArray]) -> ArrayLike[jnp.ndarray]:
    """
    Convolve trial time series with a basis matrix.

    This function checks if all trials have the same duration. If they do, it uses a fast method
    to convolve all trials with the basis matrix at once. If they do not, it falls back to convolving
    each trial individually.

    Parameters
    ----------
    basis_matrix :
        The basis matrix with which to convolve the trials. Shape (n_basis_funcs, window_size).
    trials_time_series :
        The time series of trials to convolve with the basis matrix. It should be a list of 2D arrays,
        where each array represents a trial and its second dimension matches the first dimension
        of the basis_matrix. Each trial has shape (n_neurons, n_timebins_trial).

    Returns
    -------
    :
        The convolved trials. It is a list of 3D arrays, where each array represents a convolved trial.
        Each element of the list will have shape (n_neurons, n_basis_funcs, n_timebins_trial - window_size - 1).

    Raises
    ------
    ValueError
        - If basis_matrix is not a 2D array-like object.
        - If trials_time_series is not an iterable of 2D array-like objects.
        - If trials_time_series contains empty trials.
        - If the number of time points in each trial is less than the window size.
    """

    basis_matrix = jnp.asarray(basis_matrix)
    # check input size
    if basis_matrix.ndim != 2:
        raise ValueError("basis_matrix must be a 2 dimensional array-like object.")

    try:
        # this should fail for variable trial length
        trials_time_series = jnp.asarray(trials_time_series)
        if trials_time_series.ndim != 3:
            raise ValueError


    except ValueError:
        # convert each trial two array
        trials_time_series = [jnp.asarray(trial) for trial in trials_time_series]
        if not all(trial.ndim == 2 for trial in trials_time_series):
            raise ValueError("trials_time_series must be an iterable of 2 dimensional array-like objects.")

    if any(k == 0 for trial in trials_time_series for k in trial.shape) | (len(trials_time_series) == 0):
        raise ValueError("trials_time_series should not contain empty trials!")

    # Broadcasted 1d convolution operations
    _CORR1 = jax.vmap(partial(jnp.convolve, mode="valid"), (0, None), 0)

    # Check window size
    ws = len(basis_matrix[0])
    if any(trial.shape[1] < ws for trial in trials_time_series):
        raise ValueError("Insufficient trial duration. The number of time points in each trial must "
                         "be greater or equal to the window size.")

    # Same trial duration
    # [[r x n x t], [w]] -> [r x n x (t - w + 1)]
    _CORR2 = jax.vmap(_CORR1, (1, None), 1)
    _CORR_SAME_TRIAL_DUR = jax.vmap(_CORR2, (None, 0), 2)

    # Variable trial dur
    # [[n x t],[p x w]] -> [n x p x (t - w + 1)]
    _CORR_VARIABLE_TRIAL_DUR = jax.vmap(_CORR1, (None, 0), 1)

    # Check if all trials have the same duration
    same_dur = trials_time_series.ndim == 3 if isinstance(trials_time_series, jnp.ndarray) else False

    if same_dur:
        print('All trials have the same duration.')
        conv_trials = list(_CORR_SAME_TRIAL_DUR(trials_time_series, basis_matrix))
    else:
        print('Trials have variable durations.')
        conv_trials = [_CORR_VARIABLE_TRIAL_DUR(jnp.atleast_2d(trial), basis_matrix) for trial in trials_time_series]

    return conv_trials


def nan_pad_conv(conv_trials: ArrayLike, window_size: int, convolution_type: Optional[str] = 'causal') -> ArrayLike:
    """
    Add NaN padding to convolution trials based on the convolution type.

    This function adds NaN padding to the left and/or right of the trial based on the specified convolution type.
    It also removes the first or last time point of every trial based on the convolution type. The trials with
    the same duration are processed together for efficiency.

    Parameters
    ----------
    conv_trials :
        A 4D array-like of trials to be padded. Each trial has shape (n_neurons, n_basis_funcs, n_timebins_trial).
    window_size :
        The window size to determine the padding.
    convolution_type :
        The type of convolution, by default 'causal'. It must be one of 'causal', 'acausal', or 'anti-causal'.

    Returns
    -------
    ArrayLike
        A 4D array-like of padded trials. Each trial has shape (n_neurons, n_basis_funcs, n_timebins_trial + padding).

    Raises
    ------
    ValueError
        If the convolution_type is not one of 'causal', 'acausal', or 'anti-causal'.
    """

    padding_settings = {
        "causal": (window_size, 0),
        "acausal": ((window_size - 1)//2, window_size - 1 - (window_size - 1)//2),
        "anti-causal": (0, window_size)
    }

    if convolution_type not in padding_settings:
        raise ValueError(f'convolution_type must be "causal", "acausal", or "anti-causal". {convolution_type} provided instead!')

    pad_left, pad_right = padding_settings[convolution_type]

    # Check if all trials have the same duration
    same_dur = len(set(len(trial[-1]) for trial in conv_trials)) == 1

    if same_dur:
        conv_trials = jnp.asarray(conv_trials)
        # Adjust time points
        if convolution_type in ("causal", "anti-causal"):
            start, end = (1, None) if convolution_type == "anti-causal" else (None, -1)
            conv_trials = conv_trials[:, :, :, start:end]

        conv_trials = list(jnp.pad(conv_trials,
                                   ((0, 0), (0, 0), (0, 0), (pad_left, pad_right)),
                                   constant_values=jnp.nan))
    else:
        # Adjust time points
        if convolution_type in ("causal", "anti-causal"):
            start, end = (1, None) if convolution_type == "anti-causal" else (None, -1)
            conv_trials = [trial[:, :, :, start:end] for trial in conv_trials]

        conv_trials = [jnp.pad(x, ((0, 0), (0, 0), (pad_left, pad_right)),
                               constant_values=jnp.nan) for x in conv_trials]

    return conv_trials



def plot_spike_raster(
    spike_data: Union[jnp.ndarray, np.ndarray],
    lineoffsets: Union[None, float, ArrayLike] = None,
    linelengths: Union[float, ArrayLike] = 0.2,
    linewidths: Union[float, ArrayLike] = 0.5,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
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
        raise ValueError(
            f"spike_data should be 2d, but got {spike_data.ndim}d instead!"
        )
    events = [d.nonzero()[0] for d in spike_data]
    if ax is None:
        _, ax = plt.subplots(
            1, 1, figsize=(spike_data.shape[1] / 100, spike_data.shape[0] / 5)
        )
    if lineoffsets is None:
        lineoffsets = jnp.arange(len(events))
    ax.eventplot(
        events,
        lineoffsets=lineoffsets,
        linelengths=linelengths,
        linewidths=linewidths,
        **kwargs,
    )
    ax.set(yticks=[], xlim=[0, spike_data.shape[1]])
    return ax
