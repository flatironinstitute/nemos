"""Figures for the GLM-HMM behavioral-states tutorial.

Run this module to regenerate the committed assets::

    python docs/scripts/glmhmm_figs.py

The Ashwood panels are drawn from the values reported in the paper rather than
reproduced from the published page, so the output is real vector art (with real
``<text>``) instead of a screenshot.
"""

import pathlib
import re

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# Three-state palette, shared with the GLM-HMM graphical model asset.
STATE_COLORS = ["#E07B39", "#4F9E4F", "#4C82C3"]

# Transition-matrix palette, matching the paper: low probabilities are a muted
# slate, high ones near-white, so the diagonal reads as the light band.
TRANSITION_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "ashwood_transitions", ["#6F7A8E", "#F7FAFB"]
)

# Values reported in Ashwood et al. (2022), Nature Neuroscience 25:201-212.
TRANSITIONS = np.array([[0.98, 0.01, 0.01], [0.04, 0.94, 0.02], [0.04, 0.01, 0.96]])
GLM_WEIGHTS = np.array(
    [
        [6.6, -0.1, 0.35, 0.15],  # state 1
        [1.5, -2.55, -0.15, 0.1],  # state 2
        [1.6, 2.2, 0.55, 0.35],  # state 3
    ]
)
PREDICTORS = ["Stimulus", "Bias", "Prev.\nchoice", "Win-stay,\nlose-switch"]
ACCURACY = [80, 90, 60, 58]
OCCUPANCY = [0.69, 0.15, 0.16]


def _panel_label(ax, label):
    ax.text(
        -0.24,
        1.12,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _transition_panel(ax):
    # Drawn as patches rather than with imshow: imshow embeds a base64 PNG in the
    # SVG, which is the raster-in-vector pattern this figure exists to avoid.
    for i in range(3):
        for j in range(3):
            value = TRANSITIONS[i, j]
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, facecolor=TRANSITION_CMAP(value), lw=0
                )
            )
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                # Sits on a coloured cell, so it must not follow the page theme.
                # Every cell in this palette is light enough for black text.
                color="#111111",
            )
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(2.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(3), [str(i + 1) for i in range(3)])
    ax.set_yticks(range(3), [str(i + 1) for i in range(3)])
    ax.set_xlabel("State $t$")
    ax.set_ylabel("State $t-1$")
    _panel_label(ax, "2d")


def _weights_panel(ax):
    x = np.arange(len(PREDICTORS))
    for state, weights in enumerate(GLM_WEIGHTS):
        ax.plot(
            x,
            weights,
            "-o",
            ms=5,
            color=STATE_COLORS[state],
            label=f"State {state + 1}",
        )
    ax.axhline(0, ls=":", lw=1, color="grey")
    ax.set_xticks(x, PREDICTORS, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("GLM weight")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    _panel_label(ax, "2e")


def _accuracy_panel(ax):
    labels = ["All", "1", "2", "3"]
    colors = ["#9A9A9A"] + STATE_COLORS
    bars = ax.bar(labels, ACCURACY, color=colors)
    ax.bar_label(bars, fmt="%d", fontsize=9, padding=2)
    ax.set_ylim(50, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("State")
    _panel_label(ax, "2f")


def _occupancy_panel(ax):
    labels = ["1", "2", "3"]
    bars = ax.bar(labels, OCCUPANCY, color=STATE_COLORS)
    ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=2)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Frac. occupancy")
    ax.set_xlabel("State")
    _panel_label(ax, "3d")


def plot_ashwood_targets():
    """The four Ashwood et al. (2022) panels this tutorial reproduces."""
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.1))
    _transition_panel(axes[0])
    _weights_panel(axes[1])
    _accuracy_panel(axes[2])
    _occupancy_panel(axes[3])
    for ax in axes[1:]:
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].spines[:].set_visible(False)
    fig.tight_layout()
    return fig


# Colour for axis labels, tick labels, spines, panel letters, legend and bar
# labels. Black, and fixed: an embedded SVG cannot see the page's light/dark
# theme, only the reader's OS preference, so anything that reacted to the latter
# ended up light-on-white whenever the two disagreed.
INK = "#000000"


def _save_svg(fig, path):
    """Save as SVG, reproducibly.

    The RDF block is dropped because matplotlib stamps a creation timestamp in
    it, which would otherwise make every regeneration show up as a diff.
    """
    fig.savefig(path, transparent=True, bbox_inches="tight")
    text = pathlib.Path(path).read_text()
    text = re.sub(r"<metadata>.*?</metadata>\s*", "", text, flags=re.DOTALL)
    pathlib.Path(path).write_text(text)


if __name__ == "__main__":
    # "none" keeps glyphs as real <text> instead of outlining them into paths,
    # which is the difference between a ~40 KB asset and a ~300 KB one. The font
    # stack matches the hand-authored assets and keeps the per-<text> style short.
    plt.rcParams["svg.fonttype"] = "none"
    # Fixes the salt matplotlib uses for generated element ids, so regenerating
    # the asset does not churn every clip-path reference.
    plt.rcParams["svg.hashsalt"] = "nemos-glmhmm"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    # Route all themable ink through the sentinel colour (see _save_svg).
    plt.rcParams.update(
        {
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )
    _save_svg(plot_ashwood_targets(), "docs/assets/ashwood_targets.svg")
