#!/usr/bin/env python3

from typing import Optional, Union

import jax

try:
    import matplotlib as mpl
except ImportError:
    raise ImportError(
        "Missing optional dependency 'matplotlib'."
        " Please use pip or "
        "conda to install 'matplotlib'."
    )

try:
    import seaborn as sns
except ImportError:
    raise ImportError(
        "Missing optional dependency 'seaborn'."
        " Please use pip or "
        "conda to install 'seaborn'."
    )


import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

from ..basis import RaisedCosineLogEval

warnings.warn(
    "plotting functions contained within `_documentation_utils` are intended for nemos's documentation. "
    "Feel free to use them, but they will probably not work as intended with other datasets / in other contexts."
)


def lnp_schematic(
    input_feature: nap.Tsd,
    weights: np.ndarray,
    intercepts: np.ndarray,
    plot_nonlinear: bool = False,
    plot_spikes: bool = False,
):
    """Create LNP schematic.

    - Works best with len(weights) == 3.
    - Requires len(weights) == len(intercepts).
    - plot_nonlinear=False and plot_spikes=True will look weird.
    """
    assert len(weights) == len(
        intercepts
    ), "weights and intercepts must have same length!"

    n_weights = len(weights)
    fig, axes = plt.subplots(
        n_weights, 4, sharex=True, sharey="col", gridspec_kw={"wspace": 0.65}
    )

    times = input_feature.t
    input_feature = np.expand_dims(input_feature, 0)
    weights = np.expand_dims(weights, -1)
    intercepts = np.expand_dims(intercepts, -1)
    linear = weights * input_feature + intercepts
    nonlinear = np.exp(linear)

    # Set the input feature plot
    axes[1, 0].plot(times, input_feature[0], "gray")
    axes[1, 0].set_title("$x$", fontsize=10)
    axes[1, 0].tick_params("x", labelbottom=True)
    axes[0, 0].set_xticks([times.min(), times.max()])
    axes[0, 0].set_visible(False)
    axes[2, 0].set_visible(False)

    arrowprops = {
        "color": "0",
        "arrowstyle": "->",
        "lw": 1,
        "connectionstyle": "arc,angleA=0,angleB=180,armA=20,armB=25,rad=5",
    }

    # Draw arrows from input to linear layer
    y_vals = [1.7, 0.5, -0.7]
    for y in y_vals:
        axes[1, 0].annotate(
            "",
            (1.5, y),
            (1, 0.5),
            xycoords="axes fraction",
            textcoords="axes fraction",
            ha="center",
            va="center",
            arrowprops=arrowprops,
        )

    titles = []
    for i in range(n_weights):
        axes[i, 1].plot(times, linear[i])
        sign = "+" if intercepts[i, 0] >= 0 else "-"
        titles.append(f"{weights[i, 0]}x {sign} {abs(intercepts[i, 0])}")
        axes[i, 1].set_title(f"${titles[-1]}$", y=0.95, fontsize=10)

        if plot_nonlinear:
            axes[i, 2].plot(times, nonlinear[i])
            axes[i, 2].set_title(f"$\\exp({titles[i]})$", y=0.95, fontsize=10)
            axes[i, 1].annotate(
                "",
                (1.5, 0.5),
                (1, 0.5),
                xycoords="axes fraction",
                textcoords="axes fraction",
                ha="center",
                va="center",
                arrowprops=arrowprops,
            )
        else:
            axes[i, 2].set_visible(False)

        if plot_spikes:
            gs = axes[i, 3].get_subplotspec().subgridspec(3, 1)
            axes[i, 3].set_frame_on(False)
            axes[i, 3].xaxis.set_visible(False)
            axes[i, 3].yaxis.set_visible(False)
            ax = None
            for j in range(3):
                ax = fig.add_subplot(gs[j, 0], sharey=ax)
                spikes = jax.random.poisson(
                    jax.random.PRNGKey(j * i + j + i), nonlinear[i]
                )
                spike_times = np.where(spikes)
                ax.vlines(times[spike_times], 0, spikes[spike_times], color="k")
                ax.yaxis.set_visible(False)
                if j != 2 or i != n_weights - 1:
                    ax.xaxis.set_visible(False)
                else:
                    ax.set_xticks([times.min(), times.max()])
            axes[i, 2].annotate(
                "",
                (1.5, 0.5),
                (1, 0.5),
                xycoords="axes fraction",
                textcoords="axes fraction",
                ha="center",
                va="center",
                arrowprops=arrowprops,
            )
        else:
            axes[i, 3].set_visible(False)

    # Set section titles
    suptitles = ["Input", "Linear", "Nonlinear", "Poisson samples\n(spike histogram)"]
    suptitles_to_add = [True, True, plot_nonlinear, plot_spikes]
    for b, ax, t in zip(suptitles_to_add, axes[0, :], suptitles):
        if b:
            ax.text(
                0.5,
                1.4,
                t,
                transform=ax.transAxes,
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=12,
            )

    for ax in axes.flatten():
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(1))

    return fig


def tuning_curve_plot(tuning_curve: pd.DataFrame):
    fig, ax = plt.subplots(1, 1)
    tc_idx = tuning_curve.index.to_numpy()
    tc_val = tuning_curve.values.flatten()
    width = tc_idx[1] - tc_idx[0]
    ax.bar(
        tc_idx,
        tc_val,
        width,
        facecolor="grey",
        edgecolor="k",
        label="observed",
        alpha=0.4,
    )
    ax.set_xlabel("Current (pA)")
    ax.set_ylabel("Firing rate (Hz)")
    return fig


def current_injection_plot(
    current: nap.Tsd,
    spikes: nap.TsGroup,
    firing_rate: nap.TsdFrame,
    predicted_firing_rate: Optional[nap.TsdFrame] = None,
):
    ex_intervals = current.threshold(0.0).time_support

    # define plotting parameters
    # colormap, color levels and transparency level
    # for the current injection epochs
    cmap = plt.get_cmap("autumn")
    color_levs = [0.8, 0.5, 0.2]
    alpha = 0.4

    fig = plt.figure(figsize=(7, 7))
    # first row subplot: current
    ax = plt.subplot2grid((4, 3), loc=(0, 0), rowspan=1, colspan=3, fig=fig)
    ax.plot(current, color="grey")
    ax.set_ylabel("Current (pA)")
    ax.set_title("Injected Current")
    ax.set_xticklabels([])
    ax.axvspan(
        ex_intervals.loc[0, "start"],
        ex_intervals.loc[0, "end"],
        alpha=alpha,
        color=cmap(color_levs[0]),
    )
    ax.axvspan(
        ex_intervals.loc[1, "start"],
        ex_intervals.loc[1, "end"],
        alpha=alpha,
        color=cmap(color_levs[1]),
    )
    ax.axvspan(
        ex_intervals.loc[2, "start"],
        ex_intervals.loc[2, "end"],
        alpha=alpha,
        color=cmap(color_levs[2]),
    )

    # second row subplot: response
    resp_ax = plt.subplot2grid((4, 3), loc=(1, 0), rowspan=1, colspan=3, fig=fig)
    resp_ax.plot(firing_rate, color="k", label="Observed firing rate")
    if predicted_firing_rate:
        resp_ax.plot(
            predicted_firing_rate, color="tomato", label="Predicted firing rate"
        )
    resp_ax.plot(spikes.to_tsd([-1.5]), "|", color="k", ms=10, label="Observed spikes")
    resp_ax.set_ylabel("Firing rate (Hz)")
    resp_ax.set_xlabel("Time (s)")
    resp_ax.set_title("Neural response", y=0.95)
    resp_ax.axvspan(
        ex_intervals.loc[0, "start"],
        ex_intervals.loc[0, "end"],
        alpha=alpha,
        color=cmap(color_levs[0]),
    )
    resp_ax.axvspan(
        ex_intervals.loc[1, "start"],
        ex_intervals.loc[1, "end"],
        alpha=alpha,
        color=cmap(color_levs[1]),
    )
    resp_ax.axvspan(
        ex_intervals.loc[2, "start"],
        ex_intervals.loc[2, "end"],
        alpha=alpha,
        color=cmap(color_levs[2]),
    )
    ylim = resp_ax.get_ylim()

    # third subplot: zoomed responses
    zoom_axes = []
    for i in range(len(ex_intervals)):
        interval = ex_intervals.loc[[i]]
        ax = plt.subplot2grid((4, 3), loc=(2, i), rowspan=1, colspan=1, fig=fig)
        ax.plot(firing_rate.restrict(interval), color="k")
        ax.plot(spikes.restrict(interval).to_tsd([-1.5]), "|", color="k", ms=10)
        if predicted_firing_rate:
            ax.plot(predicted_firing_rate.restrict(interval), color="tomato")
        else:
            ax.set_ylim(ylim)
        if i == 0:
            ax.set_ylabel("Firing rate (Hz)")
        ax.set_xlabel("Time (s)")
        for spine in ["left", "right", "top", "bottom"]:
            color = cmap(color_levs[i])
            # add transparency
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_color(color)
            ax.spines[spine].set_linewidth(2)
        zoom_axes.append(ax)

    resp_ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.4),
        bbox_transform=zoom_axes[1].transAxes,
    )
    return fig


def plot_weighted_sum_basis(time, weights, basis_kernels, basis_coeff):
    """
    Plot weighted sum of basis.

    Parameters
    ----------
    time:
        Time axis.
    weights:
        GLM fitted weights (num_neuron, window_size).
    basis_kernels:
        Basis kernels (window_size, num_basis_funcs).
    basis_coeff:
        The basis coefficients.

    Returns
    -------
        The figure.

    """
    basis_coeff = np.squeeze(basis_coeff)
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))

    axs[0].set_title("Basis")
    lines = axs[0].plot(time, basis_kernels)
    axs[0].set_xlabel("Time from spike (sec)")
    axs[0].set_ylabel("a.u.")

    colors = [p.get_color() for p in lines]

    axs[1].set_title("Coefficients")
    for k in range(len(basis_coeff)):
        axs[1].bar([k], [basis_coeff[k]], width=1, color=colors[k])
    axs[1].set_xticks([0, 7])
    axs[1].set_xlabel("Basis ID")
    axs[1].set_ylabel("Coefficient")

    axs[2].set_title("Basis x Coefficients")
    # flip time plot how a spike affects the future rate
    for k in range(basis_kernels.shape[1]):
        axs[2].plot(time, basis_kernels[:, k] * basis_coeff[k], color=colors[k])
    axs[2].set_xlabel("Time from spike (sec)")
    axs[2].set_ylabel("Weight")

    axs[3].set_title("Spike History Effect")
    axs[3].plot(time, np.squeeze(weights), alpha=0.3)
    axs[3].plot(time, basis_kernels @ basis_coeff, "--k")
    axs[3].set_xlabel("Time from spike (sec)")
    axs[3].set_ylabel("Weight")
    axs[3].axhline(0, color="k", lw=0.5)
    plt.tight_layout()
    return fig


class PlotSlidingWindow:
    def __init__(
        self,
        counts: nap.Tsd,
        start: float,
        n_shift: int = 20,
        history_window: float = 0.8,
        bin_size: float = 0.01,
        ylim: tuple[float, float] = (0, 3),
        plot_every: int = 1,
        figsize: tuple[float, float] = (8, 8),
        interval: int = 200,
        add_before: float = 0.0,
        add_after: float = 0.0,
    ):
        self.counts = counts
        self.n_shift = n_shift
        self.history_window = history_window
        self.plot_every = plot_every
        self.bin_size = bin_size
        self.start = start
        self.ylim = ylim
        self.add_before = add_before
        self.add_after = add_after
        self.fig, self.rect_obs, self.rect_hist, self.line_tree, self.rect_hist_ax2 = (
            self.set_up(figsize)
        )
        self.interval = interval
        self.count_frame_0 = -1

    def set_up(self, figsize):
        fig = plt.figure(figsize=figsize)

        # set up the plot for the sliding history window
        ax = plt.subplot2grid((5, 1), (0, 0), rowspan=1, colspan=1, fig=fig)
        # create the two rectangles, prediction and current observation
        rect_hist = plt.Rectangle(
            (self.start, 0),
            self.history_window,
            self.ylim[1] - self.ylim[0],
            alpha=0.3,
            color="orange",
        )
        rect_obs = plt.Rectangle(
            (self.start + self.history_window, 0),
            self.bin_size,
            self.ylim[1] - self.ylim[0],
            alpha=0.3,
            color="tomato",
        )
        plot_ep = nap.IntervalSet(
            -self.add_before + self.start,
            self.start
            + self.history_window
            + (self.n_shift - 1) * self.bin_size * self.plot_every
            + self.add_after,
        )
        color = ax.step(
            self.counts.restrict(plot_ep).t,
            self.counts.restrict(plot_ep).d,
            where="post",
        )[0].get_color()

        ax.add_patch(rect_obs)
        ax.add_patch(rect_hist)
        ax.set_xlim(*plot_ep.values)

        # set up the feature matrix plot
        ax = plt.subplot2grid((5, 1), (1, 0), rowspan=4, colspan=1, fig=fig)

        line_tree = []
        for frame in range(self.n_shift):
            iset = nap.IntervalSet(
                start=rect_hist.get_x() + self.bin_size * self.plot_every * frame,
                end=rect_hist.get_x()
                + rect_hist.get_width()
                + self.bin_size * self.plot_every * frame,
            )
            cnt = self.counts.restrict(iset).d
            line_tree.append(
                fig.axes[1].step(
                    np.arange(cnt.shape[0]) * self.bin_size,
                    np.diff(self.ylim) * (self.n_shift - frame - 1) + cnt,
                    where="post",
                    color=color,
                )
            )
            if frame == 0:
                rect_hist_ax2 = plt.Rectangle(
                    (0, (self.ylim[1] - self.ylim[0]) * (self.n_shift - 1)),
                    (cnt.shape[0] - 1) * self.bin_size,
                    self.ylim[1] - self.ylim[0],
                    alpha=0.3,
                    color="orange",
                )
                ax.add_patch(rect_hist_ax2)

        # revert tick labels
        yticks = ax.get_yticks()
        original_ytick_labels = ax.get_yticklabels()
        ax.set_yticks(yticks + self.ylim[1] - self.ylim[0])
        reverse_ytick_labels = [
            label.get_text() for label in reversed(original_ytick_labels)
        ]
        ax.set_yticklabels(reverse_ytick_labels)

        ax.set_ylabel("Sample Index")
        ax.set_xlabel("Time From Spike (sec)")
        ax.set_ylim(-1, self.n_shift * np.diff(self.ylim) + 1)
        self.set_lines_visible(line_tree[1:], False)

        plt.tight_layout()
        return fig, rect_obs, rect_hist, line_tree, rect_hist_ax2

    def update_fig(self, frame):

        if frame == 0:
            self.rect_hist.set_x(self.start)
            self.rect_obs.set_x(self.start + self.rect_hist.get_width())
            self.set_lines_visible(self.line_tree, False)
            self.rect_hist_ax2.set_y((self.ylim[1] - self.ylim[0]) * (self.n_shift - 1))
        else:
            self.rect_obs.set_x(self.rect_obs.get_x() + self.bin_size * self.plot_every)
            self.rect_hist.set_x(
                self.rect_hist.get_x() + self.bin_size * self.plot_every
            )
            self.rect_hist_ax2.set_y(
                self.rect_hist_ax2.get_y() - (self.ylim[1] - self.ylim[0])
            )
            self.rect_hist_ax2.set_height(self.ylim[1] - self.ylim[0])

        self.set_lines_visible(self.line_tree[frame], True)

    @staticmethod
    def set_lines_visible(line_tree, visible: bool):
        jax.tree_util.tree_map(lambda line: line.set_visible(visible), line_tree)

    def run(self):
        anim = FuncAnimation(
            self.fig, self.update_fig, self.n_shift, interval=self.interval, repeat=True
        )
        plt.close(self.fig)
        return anim


def run_animation(counts: nap.Tsd, start: float):
    return PlotSlidingWindow(counts, start).run()


def plot_coupling(
    responses,
    tuning,
    cmap_name="seismic",
    figsize=(10, 8),
    fontsize=15,
    alpha=0.5,
    cmap_label="hsv",
):
    pref_ang = tuning.idxmax()
    cmap_tun = plt.get_cmap(cmap_label)
    color_tun = (pref_ang.values - pref_ang.values.min()) / (
        pref_ang.values.max() - pref_ang.values.min()
    )

    # plot heatmap
    sum_resp = np.sum(responses, axis=2)
    # normalize by cols (for fixed receiver neuron, scale all responses
    # so that the strongest peaks to 1)
    sum_resp_n = (sum_resp.T / sum_resp.max(axis=1)).T

    # scale to 0,1
    color = -0.5 * (sum_resp_n - sum_resp_n.min()) / sum_resp_n.min()

    cmap = plt.get_cmap(cmap_name)
    n_row, n_col, n_tp = responses.shape
    time = np.arange(n_tp)
    fig, axs = plt.subplots(n_row + 1, n_col + 1, figsize=figsize, sharey="row")
    for rec, rec_resp in enumerate(responses):
        for send, resp in enumerate(rec_resp):
            axs[rec, send].plot(time, responses[rec, send], color="k")
            axs[rec, send].spines["left"].set_visible(False)
            axs[rec, send].spines["bottom"].set_visible(False)
            axs[rec, send].set_xticks([])
            axs[rec, send].set_yticks([])
            axs[rec, send].axhline(0, color="k", lw=0.5)
            if rec == n_row - 1:
                axs[n_row, send].remove()  # Remove the original axis
                axs[n_row, send] = fig.add_subplot(
                    n_row + 1,
                    n_col + 1,
                    np.ravel_multi_index((n_row, send + 1), (n_row + 1, n_col + 1)),
                    polar=True,
                )  # Add new polar axis

                axs[n_row, send].fill_between(
                    tuning.iloc[:, send].index,
                    np.zeros(len(tuning)),
                    tuning.iloc[:, send].values,
                    color=cmap_tun(color_tun[send]),
                    alpha=0.5,
                )
                axs[n_row, send].set_xticks([])
                axs[n_row, send].set_yticks([])

        axs[rec, send + 1].remove()  # Remove the original axis
        axs[rec, send + 1] = fig.add_subplot(
            n_row + 1,
            n_col + 1,
            np.ravel_multi_index((rec, send + 1), (n_row + 1, n_col + 1)) + 1,
            polar=True,
        )  # Add new polar axis

        axs[rec, send + 1].fill_between(
            tuning.iloc[:, rec].index,
            np.zeros(len(tuning)),
            tuning.iloc[:, rec].values,
            color=cmap_tun(color_tun[rec]),
            alpha=0.5,
        )
        axs[rec, send + 1].set_xticks([])
        axs[rec, send + 1].set_yticks([])
    axs[rec + 1, send + 1].set_xticks([])
    axs[rec + 1, send + 1].set_yticks([])
    axs[rec + 1, send + 1].spines["left"].set_visible(False)
    axs[rec + 1, send + 1].spines["bottom"].set_visible(False)
    for rec, rec_resp in enumerate(responses):
        for send, resp in enumerate(rec_resp):
            xlim = axs[rec, send].get_xlim()
            ylim = axs[rec, send].get_ylim()
            rect = plt.Rectangle(
                (xlim[0], ylim[0]),
                xlim[1] - xlim[0],
                ylim[1] - ylim[0],
                alpha=alpha,
                color=cmap(color[rec, send]),
                zorder=1,
            )
            axs[rec, send].add_patch(rect)
            axs[rec, send].set_xlim(xlim)
            axs[rec, send].set_ylim(ylim)
    axs[n_row // 2, 0].set_ylabel("receiver\n", fontsize=fontsize)
    axs[n_row, n_col // 2].set_xlabel("\nsender", fontsize=fontsize)

    plt.suptitle("Pairwise Interaction", fontsize=fontsize)
    return fig


def plot_history_window(neuron_count, interval, window_size_sec):
    bin_size = 1 / neuron_count.rate
    # define the count history window used for prediction
    history_interval = nap.IntervalSet(
        start=interval["start"][0], end=window_size_sec + interval["start"][0] - 0.001
    )

    # define the observed counts bin (the bin right after the history window)
    observed_count_interval = nap.IntervalSet(
        start=history_interval["end"], end=history_interval["end"] + bin_size
    )

    fig, _ = plt.subplots(1, 1, figsize=(8, 3.5))
    plt.step(
        neuron_count.restrict(interval).t,
        neuron_count.restrict(interval).d,
        where="post",
    )
    ylim = plt.ylim()
    plt.axvspan(
        history_interval["start"][0],
        history_interval["end"][0],
        *ylim,
        alpha=0.4,
        color="orange",
        label="history",
    )
    plt.axvspan(
        observed_count_interval["start"][0],
        observed_count_interval["end"][0],
        *ylim,
        alpha=0.4,
        color="tomato",
        label="predicted",
    )
    plt.ylim(ylim)
    plt.title("Spike Count Time Series")
    plt.xlabel("Time (sec)")
    plt.ylabel("Counts")
    plt.legend()
    plt.tight_layout()
    return fig


def plot_convolved_counts(counts, conv_spk, *epochs, figsize=(6.5, 4.5)):
    n_rows = len(epochs)
    conv_spk = np.squeeze(conv_spk)
    fig, axs = plt.subplots(n_rows, 1, sharey="all", figsize=figsize)
    for row, ep in enumerate(epochs):
        axs[row].plot(conv_spk.restrict(ep))
        cnt_ep = counts.restrict(ep)
        axs[row].vlines(cnt_ep.t[cnt_ep.d > 0], -1, -0.1, "k", lw=2, label="spikes")

        if row == 0:
            axs[0].set_title("Convolved Counts")
            axs[0].legend()
        elif row == n_rows - 1:
            axs[row].set_xlabel("Time (sec)")
    return fig


def plot_rates_and_smoothed_counts(
    counts, rate_dict, start=8819.4, end=8821, smooth_std=0.05, smooth_ws_scale=20
):
    ep = nap.IntervalSet(start=start, end=end)
    fig = plt.figure()
    for key in rate_dict:
        plt.plot(np.squeeze(rate_dict[key].restrict(ep)), label=key)

    idx_spikes = np.where(counts.restrict(ep).d > 0)[0]
    plt.vlines(counts.restrict(ep).t[idx_spikes], -8, -1, color="k")
    plt.plot(
        counts.smooth(smooth_std, size_factor=smooth_ws_scale).restrict(ep)
        * counts.rate,
        color="k",
        label="Smoothed spikes",
    )
    plt.axhline(0, color="k")
    plt.xlabel("Time (sec)")
    plt.ylabel("Firing Rate (Hz)")
    plt.legend()
    return fig


def plot_basis(n_basis_funcs=8, window_size_sec=0.8):
    fig = plt.figure()
    basis = RaisedCosineLogEval(n_basis_funcs=n_basis_funcs)
    time, basis_kernels = basis.evaluate_on_grid(1000)
    time *= window_size_sec
    plt.plot(time, basis_kernels)
    plt.xlabel("time (sec)")
    return fig


def plot_position_phase_speed_tuning(
    pf, glm_pf, tc_speed, glm_speed, tc_pos_theta, glm_pos_theta, xybins
):
    fig = plt.figure()
    gs = plt.GridSpec(2, 2)
    plt.subplot(gs[0, 0])
    plt.plot(pf)
    plt.plot(glm_pf, label="GLM")
    plt.xlabel("Position (cm)")
    plt.ylabel("Firing rate (Hz)")
    plt.legend()

    plt.subplot(gs[0, 1])
    plt.plot(tc_speed)
    plt.plot(glm_speed, label="GLM")
    plt.xlabel("Speed (cm/s)")
    plt.ylabel("Firing rate (Hz)")
    plt.legend()

    plt.subplot(gs[1, 0])
    extent = (xybins[0][0], xybins[0][-1], xybins[1][0], xybins[1][-1])
    plt.imshow(tc_pos_theta.T, aspect="auto", origin="lower", extent=extent)
    plt.xlabel("Position (cm)")
    plt.ylabel("Theta Phase (rad)")

    plt.subplot(gs[1, 1])
    plt.imshow(glm_pos_theta.T, aspect="auto", origin="lower", extent=extent)
    plt.xlabel("Position (cm)")
    plt.ylabel("Theta Phase (rad)")
    plt.title("GLM")

    plt.tight_layout()

    return fig


def plot_position_speed_tuning(axis, tc, pf, tc_speed, m):
    gs = axis.subgridspec(1, 2)
    plt.subplot(gs[0, 0])
    plt.plot(pf, "--", label="Observed")
    plt.plot(tc["position"][0])
    plt.xlabel("Position (cm)")
    plt.ylabel("Firing rate (Hz)")
    plt.title("Model : {}".format(m))
    plt.legend()

    plt.subplot(gs[0, 1])
    plt.plot(tc_speed, "--")
    plt.plot(tc["speed"][0])
    plt.xlabel("Speed (cm/s)")


def highlight_max_cell(cvdf_wide, ax):
    max_col = cvdf_wide.max().idxmax()
    max_col_index = cvdf_wide.columns.get_loc(max_col)
    max_row = cvdf_wide[max_col].idxmax()
    max_row_index = cvdf_wide.index.get_loc(max_row)

    ax.add_patch(
        Rectangle(
            (max_col_index, max_row_index), 1, 1, fill=False, lw=3, color="skyblue"
        )
    )


def plot_heatmap_cv_results(cvdf_wide, label=None):
    plt.figure()
    ax = sns.heatmap(
        cvdf_wide,
        annot=True,
        square=True,
        linecolor="white",
        linewidth=0.5,
    )

    # Labeling the colorbar
    colorbar = ax.collections[0].colorbar
    if not label:
        colorbar.set_label("log-likelihood")
    else:
        colorbar.set_label(label)

    ax.set_xlabel("ridge regularization strength")
    ax.set_ylabel("number of basis functions")

    highlight_max_cell(cvdf_wide, ax)


def plot_head_direction_tuning(
    tuning_curves: pd.DataFrame,
    spikes: nap.TsGroup,
    angle: nap.Tsd,
    threshold_hz: int = 1,
    start: float = 8910,
    end: float = 8960,
    cmap_label="hsv",
    figsize=(12, 6),
):
    """
    Plot head direction tuning.

    Parameters
    ----------
    tuning_curves:

    spikes:
        The spike times.
    angle:
        The heading angles.
    threshold_hz:
        Minimum firing rate for neuron to be plotted.,
    start:
        Start time
    end:
        End time
    cmap_label:
        cmap label ("hsv", "rainbow", "Reds", ...)
    figsize:
        Figure size in inches.

    Returns
    -------

    """
    plot_ep = nap.IntervalSet(start, end)
    index_keep = spikes.restrict(plot_ep).getby_threshold("rate", threshold_hz).index

    # filter neurons
    tuning_curves = tuning_curves.loc[:, index_keep]
    pref_ang = tuning_curves.idxmax().loc[index_keep]
    spike_tsd = (
        spikes.restrict(plot_ep).getby_threshold("rate", threshold_hz).to_tsd(pref_ang)
    )

    # plot raster and heading
    cmap = plt.get_cmap(cmap_label)
    unq_angles = np.unique(pref_ang.values)
    n_subplots = len(unq_angles)
    relative_color_levs = (unq_angles - unq_angles[0]) / (
        unq_angles[-1] - unq_angles[0]
    )
    fig = plt.figure(figsize=figsize)
    # plot head direction angle
    ax = plt.subplot2grid(
        (3, n_subplots), loc=(0, 0), rowspan=1, colspan=n_subplots, fig=fig
    )
    ax.plot(angle.restrict(plot_ep), color="k", lw=2)
    ax.set_ylabel("Angle (rad)")
    ax.set_title("Animal's Head Direction")

    ax = plt.subplot2grid(
        (3, n_subplots), loc=(1, 0), rowspan=1, colspan=n_subplots, fig=fig
    )
    ax.set_title("Neural Activity")
    for i, ang in enumerate(unq_angles):
        sel = spike_tsd.d == ang
        ax.plot(
            spike_tsd[sel].t,
            np.ones(sel.sum()) * i,
            "|",
            color=cmap(relative_color_levs[i]),
            alpha=0.5,
        )
    ax.set_ylabel("Sorted Neurons")
    ax.set_xlabel("Time (s)")

    for i, ang in enumerate(unq_angles):
        neu_idx = np.argsort(pref_ang.values)[i]
        ax = plt.subplot2grid(
            (3, n_subplots),
            loc=(2 + i // n_subplots, i % n_subplots),
            rowspan=1,
            colspan=1,
            fig=fig,
            projection="polar",
        )
        ax.fill_between(
            tuning_curves.iloc[:, neu_idx].index,
            np.zeros(len(tuning_curves)),
            tuning_curves.iloc[:, neu_idx].values,
            color=cmap(relative_color_levs[i]),
            alpha=0.5,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    return fig


def plot_head_direction_tuning_model(
    tuning_curves: pd.DataFrame,
    predicted_firing_rate: nap.TsdFrame,
    spikes: nap.TsGroup,
    angle: nap.Tsd,
    threshold_hz: int = 1,
    start: float = 8910,
    end: float = 8960,
    cmap_label="hsv",
    figsize=(12, 6),
):
    """
    Plot head direction tuning.

    Parameters
    ----------
    tuning_curves:
        The tuning curve dataframe.
    predicted_firing_rate:
        The time series of the predicted rate.
    spikes:
        The spike times.
    angle:
        The heading angles.
    threshold_hz:
        Minimum firing rate for neuron to be plotted.,
    start:
        Start time
    end:
        End time
    cmap_label:
        cmap label ("hsv", "rainbow", "Reds", ...)
    figsize:
        Figure size in inches.

    Returns
    -------
    fig:
        The figure.
    """
    plot_ep = nap.IntervalSet(start, end)
    index_keep = spikes.restrict(plot_ep).getby_threshold("rate", threshold_hz).index

    # filter neurons
    tuning_curves = tuning_curves.loc[:, index_keep]
    pref_ang = tuning_curves.idxmax().loc[index_keep]
    spike_tsd = (
        spikes.restrict(plot_ep).getby_threshold("rate", threshold_hz).to_tsd(pref_ang)
    )

    # plot raster and heading
    cmap = plt.get_cmap(cmap_label)
    unq_angles = np.unique(pref_ang.values)
    n_subplots = len(unq_angles)
    relative_color_levs = (unq_angles - unq_angles[0]) / (
        unq_angles[-1] - unq_angles[0]
    )
    fig = plt.figure(figsize=figsize)
    # plot head direction angle
    ax = plt.subplot2grid(
        (4, n_subplots), loc=(0, 0), rowspan=1, colspan=n_subplots, fig=fig
    )
    ax.plot(angle.restrict(plot_ep), color="k", lw=2)
    ax.set_ylabel("Angle (rad)")
    ax.set_title("Animal's Head Direction")

    ax = plt.subplot2grid(
        (4, n_subplots), loc=(1, 0), rowspan=1, colspan=n_subplots, fig=fig
    )
    ax.set_title("Neural Activity")
    for i, ang in enumerate(unq_angles):
        sel = spike_tsd.d == ang
        ax.plot(
            spike_tsd[sel].t,
            np.ones(sel.sum()) * i,
            "|",
            color=cmap(relative_color_levs[i]),
            alpha=0.5,
        )
    ax.set_ylabel("Sorted Neurons")
    ax.set_xlabel("Time (s)")

    ax = plt.subplot2grid(
        (4, n_subplots), loc=(2, 0), rowspan=1, colspan=n_subplots, fig=fig
    )
    ax.set_title("Neural Firing Rate")

    fr = predicted_firing_rate.restrict(plot_ep).d
    fr = fr.T / np.max(fr, axis=1)
    ax.imshow(fr[::-1], cmap="Blues", aspect="auto")
    ax.set_ylabel("Sorted Neurons")
    ax.set_xlabel("Time (s)")

    for i, ang in enumerate(unq_angles):
        neu_idx = np.argsort(pref_ang.values)[i]
        ax = plt.subplot2grid(
            (4, n_subplots),
            loc=(3 + i // n_subplots, i % n_subplots),
            rowspan=1,
            colspan=1,
            fig=fig,
            projection="polar",
        )
        ax.fill_between(
            tuning_curves.iloc[:, neu_idx].index,
            np.zeros(len(tuning_curves)),
            tuning_curves.iloc[:, neu_idx].values,
            color=cmap(relative_color_levs[i]),
            alpha=0.5,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    return fig


def plot_count_history_window(
    counts: nap.Tsd,
    n_shift: int,
    history_window: float,
    bin_size: float,
    start: float,
    ylim: tuple[float, float],
    plot_every: int,
):
    """
    Plot the count history rolling window.

    Parameters
    ----------
    counts:
        The spike counts of a neuron.
    n_shift:
        Number of rolling windows to plot.
    history_window:
        Size of the history window in seconds.
    bin_size:
        Bin size of the counts in seconds.
    start:
        Start time for the first plotted window
    ylim:
        y limits for axes.
    plot_every:
        Plot a window series every "plot_every" bins

    Returns
    -------

    """
    interval = nap.IntervalSet(
        start, start + history_window + bin_size * n_shift * plot_every
    )
    fig, axs = plt.subplots(n_shift, 1, figsize=(8, 8))
    for shift_bin in range(0, n_shift * plot_every, plot_every):
        ax = axs[shift_bin // plot_every]

        shift_sec = shift_bin * bin_size
        # select the first bin after one sec
        input_interval = nap.IntervalSet(
            start=interval["start"][0] + shift_sec,
            end=history_window + interval["start"][0] + shift_sec - 0.001,
        )
        predicted_interval = nap.IntervalSet(
            start=history_window + interval["start"][0] + shift_sec,
            end=history_window + interval["start"][0] + bin_size + shift_sec,
        )

        ax.step(counts.restrict(interval).t, counts.restrict(interval).d, where="post")

        ax.axvspan(
            input_interval["start"][0],
            input_interval["end"][0],
            *ylim,
            alpha=0.4,
            color="orange",
        )
        ax.axvspan(
            predicted_interval["start"][0],
            predicted_interval["end"][0],
            *ylim,
            alpha=0.4,
            color="tomato",
        )

        plt.ylim(ylim)
        if shift_bin == 0:
            ax.set_title("Spike Count Time Series")
        elif shift_bin == n_shift - 1:
            ax.set_xlabel("Time (sec)")
        if shift_bin != n_shift - 1:
            ax.set_xticks([])
        ax.set_yticks([])
        if shift_bin == 0:
            for spine in ["top", "right", "left", "bottom"]:
                ax.spines[spine].set_color("tomato")
                ax.spines[spine].set_linewidth(2)
        else:
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)

    plt.tight_layout()
    return fig


def plot_features(
    input_feature: Union[nap.Tsd, nap.TsdFrame, nap.TsdTensor, NDArray],
    sampling_rate: float,
    suptitle: str,
    n_rows: int = 20,
):
    """
    Plot feature matrix.

    Parameters
    ----------
    input_feature:
        The (num_samples, n_neurons, num_feature) feature array.
    sampling_rate:
        Sampling rate in hz.
    n_rows:
        Number of rows to plot.
    suptitle:
        Suptitle of the plot.

    Returns
    -------

    """
    input_feature = np.squeeze(input_feature).dropna()
    window_size = input_feature.shape[1]
    fig = plt.figure(figsize=(8, 8))
    plt.suptitle(suptitle)
    time = np.arange(0, window_size) / sampling_rate
    for k in range(n_rows):
        ax = plt.subplot(n_rows, 1, k + 1)
        plt.step(time, np.squeeze(input_feature[k]), where="post")

        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.axvspan(0, time[-1], alpha=0.4, color="orange")
        ax.set_yticks([])
        if k != n_rows - 1:
            ax.set_xticks([])
        else:
            ax.set_xlabel("lag (sec)")
        if k in [0, n_rows - 1]:
            ax.set_ylabel("$t_{%d}$" % (window_size + k), rotation=0)

    plt.tight_layout()
    return fig
