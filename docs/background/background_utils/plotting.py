#!/usr/bin/env python3

import jax
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap


def lnp_schematic(input_feature: nap.Tsd,
                  weights: np.ndarray,
                  intercepts: np.ndarray,
                  plot_nonlinear: bool = False,
                  plot_spikes: bool = False):
    """Create LNP schematic.

    - Works best with len(weights)==3.

    - Requires len(weights)==len(intercepts)

    - plot_nonlinear=False and plot_spikes=True will look weird

    """
    assert len(weights) == len(intercepts), "weights and intercepts must have same length!"
    fig, axes = plt.subplots(len(weights), 4, sharex=True,
                             sharey='col',
                             gridspec_kw={'wspace': .65})
    for ax in axes.flatten():
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(1))
    weights = np.expand_dims(weights, -1)
    intercepts = np.expand_dims(intercepts, -1)
    times = input_feature.t
    # only need to do this once, since they share x
    axes[0, 0].set_xticks([times.min(), times.max()])
    input_feature = np.expand_dims(input_feature, 0)
    linear = weights * input_feature + intercepts
    axes[0, 0].set_visible(False)
    axes[2, 0].set_visible(False)
    axes[1, 0].plot(times, input_feature[0], 'gray')
    axes[1, 0].set_title('$x$', fontsize=10)
    axes[1, 0].tick_params('x', labelbottom=True)
    axes[0, 1].tick_params('y', labelleft=True)
    axes[2, 1].tick_params('y', labelleft=True)
    arrowkwargs = {'xycoords': 'axes fraction', 'textcoords': 'axes fraction',
                   'ha': 'center', 'va': 'center'}
    arrowprops = {'color': '0', 'arrowstyle': '->', 'lw': 1,
                  'connectionstyle': 'arc,angleA=0,angleB=180,armA=20,armB=25,rad=5'}
    y_vals = [1.7, .5, -.7]
    for y in y_vals:
        axes[1, 0].annotate('', (1.5, y), (1, .5), arrowprops=arrowprops, **arrowkwargs)
    titles = []
    for i, l in enumerate(linear):
        axes[i, 1].plot(times, l)
        if intercepts[i, 0] < 0:
            s = '-'
        else:
            s = '+'
        titles.append(f"{weights[i, 0]}x {s} {abs(intercepts[i, 0])}")
        axes[i, 1].set_title(f"${titles[-1]}$", y=.95, fontsize=10)
    nonlinear = np.exp(linear)
    if plot_nonlinear:
        for i, l in enumerate(nonlinear):
            axes[i, 2].plot(times, l)
            axes[i, 2].set_title(f"$\\exp({titles[i]})$", y=.95, fontsize=10)
            axes[i, 1].annotate('', (1.5, .5), (1, .5), arrowprops=arrowprops, **arrowkwargs)
    else:
        for i, _ in enumerate(nonlinear):
            axes[i, 2].set_visible(False)
    if plot_spikes:
        for i, l in enumerate(nonlinear):
            gs = axes[i, 3].get_subplotspec().subgridspec(3, 1)
            axes[i, 3].set_frame_on(False)
            axes[i, 3].xaxis.set_visible(False)
            axes[i, 3].yaxis.set_visible(False)
            ax = None
            for j in range(3):
                ax = fig.add_subplot(gs[j, 0], sharey=ax)
                spikes = jax.random.poisson(jax.random.PRNGKey(j*i + j + i), l)
                spike_times = np.where(spikes)
                spike_heights = spikes[spike_times]
                ax.vlines(times[spike_times], 0, spike_heights, color='k')
                ax.yaxis.set_visible(False)
                if j != 2 or i != len(nonlinear) - 1:
                    ax.xaxis.set_visible(False)
                else:
                    ax.set_xticks([times.min(), times.max()])
            axes[i, 2].annotate('', (1.5, .5), (1, .5), arrowprops=arrowprops, **arrowkwargs)
    else:
        for i, _ in enumerate(nonlinear):
            axes[i, 3].set_visible(False)
    suptitles = ["Input", "Linear", "Nonlinear", "Poisson samples\n(spike histogram)"]
    suptitles_to_add = [True, True, plot_nonlinear, plot_spikes]
    for b, ax, t in zip(suptitles_to_add, axes[0, :], suptitles):
        if b:
            axes[0, 1].text(.5, 1.4, t, transform=ax.transAxes,
                            horizontalalignment='center',
                            verticalalignment='top', fontsize=12)
    return fig

