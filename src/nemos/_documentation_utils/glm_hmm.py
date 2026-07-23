#!/usr/bin/env python3
"""Helper functions for the GLM-HMM behavioral-states tutorial.

These functions handle the IBL-specific preprocessing and plotting used in
``docs/tutorials/plot_07_behavioral_states``. They are documentation-only
utilities and are not part of the public NeMoS API.
"""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

__all__ = [
    "select_sessions",
    "relabel",
    "plot_proba_left",
    "plot_glm_weights",
    "plot_transition_matrix",
    "plot_design_matrix",
    "plot_posteriors",
    "plot_accuracy_and_occupancy",
]


def select_sessions(
    trials,
    max_violations=10,
    violation_value=0,
    probability_left=0.5,
):
    """Select sessions matching the Ashwood et al. (2022) preprocessing.

    Keeps sessions that contain 0.2, 0.5 and 0.8 probability blocks and have
    fewer than ``max_violations`` invalid trials in the specified block, then
    restricts the returned trials to that block.

    Parameters
    ----------
    trials : pd.DataFrame
        Trial-level dataframe with columns 'session', 'probabilityLeft', and 'choice'.
    max_violations : int, optional
        Maximum number of invalid trials allowed in the specified block. Default is 10.
    violation_value : int, optional
        Value in the 'choice' column that indicates an invalid trial. Default is 0.
    probability_left : float, optional
        The probability block to filter trials on. Default is 0.5.

    Returns
    -------
    df_trials : pd.DataFrame
        Trials from valid sessions restricted to the selected block.
    valid_sessions : pd.Index
        Session identifiers that passed the selection criteria.
    """
    has_three_blocks = (
        trials.groupby("session")["probabilityLeft"]
        .agg(lambda s: set(s.unique()) == {0.2, 0.5, 0.8})
    )

    violations = (
        trials.query("probabilityLeft == @probability_left")
        .groupby("session")["choice"]
        .agg(lambda s: s.eq(violation_value).sum())
    )

    valid_sessions = has_three_blocks.index[
        has_three_blocks & (violations < max_violations)
    ]

    df_trials = trials.query(
        "session in @valid_sessions and probabilityLeft == @probability_left"
    )

    return df_trials, valid_sessions


def relabel(model):
    """Relabel the states of a fitted GLM-HMM to match Ashwood et al. (2022).

    Reorders the latent states so that the engaged state comes first, followed
    by the biased-left and biased-right states, using the sign and magnitude of
    the per-state bias (intercept).

    Parameters
    ----------
    model :
        Fitted GLM-HMM model.

    Returns
    -------
    model :
        GLM-HMM model with states reordered in-place.
    """
    # the intercept is the bias term
    bias_right = np.flatnonzero(model.intercept_ < -2)
    bias_left = np.flatnonzero(model.intercept_ > 2)
    # less biased (smallest |intercept|) is the engaged state
    engaged_state = np.argmin(np.abs(model.intercept_))
    relabel = np.concatenate([[engaged_state], bias_left, bias_right])

    # apply re-labeling
    model.coef_ = model.coef_[:, relabel]
    model.intercept_ = model.intercept_[relabel]
    model.initial_prob_ = model.initial_prob_[relabel]
    model.transition_prob_ = model.transition_prob_[relabel][:, relabel]
    return model


def plot_proba_left(trials):
    """Plot how the probability of a left stimulus evolves within a session.

    Highlights the first 90 trials, during which the stimulus appears on either
    side with equal probability.

    Parameters
    ----------
    trials : pd.DataFrame
        Trial-level dataframe with columns 'session' and 'probabilityLeft'.
    """
    # Choose example session
    sess_ex = "726b6915-e7de-4b55-a38e-ff4c461211d3"
    # Subset session trials
    trials_sess = trials[trials.session == sess_ex].reset_index()

    # Plot
    plt.plot(trials_sess["probabilityLeft"][:300])
    plt.axvspan(0, 90, color="skyblue", alpha=0.3, label="first 90 trials")
    plt.axvline(90, color="skyblue", linestyle="--")

    plt.ylabel("P(stimulus on the left)")
    plt.xlabel("Trial number")
    plt.show()
    return None


def plot_glm_weights(model, n_states=3):
    """Plot the per-state GLM weights of a fitted GLM-HMM.

    One line per latent state shows how strongly each covariate (stimulus,
    bias, previous choice, win-stay-lose-shift) drives the animal's choice in
    that state. Reproduces the weight panel of Ashwood et al. (2022), Fig. 2e.

    Parameters
    ----------
    model :
        A fitted GLM-HMM whose ``coef_`` (shape ``(n_features, n_states)``) and
        ``intercept_`` (shape ``(n_states,)``) hold the per-state weights.
    n_states :
        Number of latent states to plot. Default 3.

    Returns
    -------
    matplotlib.figure.Figure
        The figure with the weight traces.
    """
    fig = plt.figure(figsize=(6, 5))
    colors = ["#ff7f00", "#4daf4a", "#377eb8"]

    n_features = model.coef_.shape[0] + 1  # add 1 for the intercept

    # Change order of weights so output matches Ashwood et al. (2022) 2e plot
    recovered_weights = np.zeros((n_features, n_states))
    recovered_weights[0, :] = model.coef_[0, :]  # stimulus
    recovered_weights[1, :] = model.intercept_   # bias
    recovered_weights[2, :] = model.coef_[2, :]  # prev choice
    recovered_weights[3, :] = model.coef_[1, :]  # wsls

    # Labels
    X_labels = ["Stimulus", "Bias", "Prev.choice", "WSLS"]

    state_labels = [
        'State 1: "engaged"',
        'State 2: "biased left"',
        'State 3: "biased right"',
    ]

    for state in range(n_states):
        plt.plot(
            range(n_features),
            recovered_weights[:, state],
            color=colors[state],
            marker="o",
            lw=1.5,
            label=state_labels[state],
            linestyle="-",
        )

    plt.yticks([-2.5, 0, 2.5, 5])
    plt.ylabel("GLM weight")
    plt.xlabel("Covariate")
    plt.xticks([i for i in range(n_features)], X_labels, fontsize=12, rotation=45)
    plt.axhline(y=0, color="k", alpha=0.5, ls="--")

    plt.legend()
    plt.tight_layout()

    # save image for thumbnail
    import os

    root = os.environ.get("READTHEDOCS_OUTPUT")
    if root:
        path = pathlib.Path(root) / "html/_static/thumbnails/tutorials"
    # if local store in assets
    else:
        path = pathlib.Path("../_build/html/_static/thumbnails/tutorials")

    # make sure the folder exists if run from build
    if root or pathlib.Path("../assets/stylesheets").exists():
        path.mkdir(parents=True, exist_ok=True)

    if path.exists():
        fig.savefig(path / "plot_07_behavioral_states.svg")

    plt.show()
    return fig


def plot_transition_matrix(model, n_states=3):
    """Plot the state transition matrix of a fitted GLM-HMM as a heatmap.

    Entry ``(i, j)`` is the probability of moving from state ``i`` at trial
    ``t-1`` to state ``j`` at trial ``t``. Large diagonal values indicate the
    animal tends to persist in the same state across consecutive trials.

    Parameters
    ----------
    model :
        A fitted GLM-HMM exposing ``transition_prob_`` of shape
        ``(n_states, n_states)``.
    n_states :
        Number of latent states. Default 3.

    Returns
    -------
    matplotlib.figure.Figure
        The figure with the transition-matrix heatmap.
    """
    fig = plt.figure(figsize=(8, 3))
    n_decimals = 3
    # Plot matrix colors
    plt.imshow(model.transition_prob_, vmin=-0.8, vmax=1, cmap="bone")

    # Write probabilities
    for i in range(n_states):
        for j in range(n_states):
            plt.text(
                j,
                i,
                str(np.around(model.transition_prob_[i, j], decimals=n_decimals))[
                    : n_decimals + 2
                ],
                ha="center",
                va="center",
                color="k",
            )
    plt.xlim(-0.5, n_states - 0.5)
    plt.xticks(range(0, n_states), ("1", "2", "3"))
    plt.xlabel("State t")

    plt.yticks(range(0, n_states), ("1", "2", "3"))
    plt.ylim(n_states - 0.5, -0.5)
    plt.ylabel("State t-1")

    plt.title("Transition matrix")
    plt.subplots_adjust(0, 0, 1, 1)
    plt.show()
    return fig


# coin-flip icon used as the stochastic-emission marker in plot_design_matrix
_COINFLIP_ICON = pathlib.Path(__file__).parent / "coinflip.png"


def plot_design_matrix(X, choices, n_trials=20):
    """Plot the GLM-HMM design matrix and the associated choices side by side.

    The left heatmap shows the first ``n_trials`` rows of the design matrix
    (one column per predictor); the right heatmap shows the corresponding
    choices. Blue/green encode the left/right extremes, white the neutral
    middle, so you can eyeball how predictors line up with the animal's choice.

    Parameters
    ----------
    X :
        Design matrix of shape ``(n_trials_total, n_features)``.
    choices :
        Choice values (e.g. a ``pynapple.Tsd`` or array).
    n_trials :
        Number of trials (rows) to display. Default 20.

    Returns
    -------
    matplotlib.figure.Figure
        The figure with the two heatmaps.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.image as mpimg

    # width ratio 3:1 so the 3-column design matrix and 1-column choices end
    # up with the same (square) cell size, hence the same height and alignment.
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6, 8),
        sharey=True,
        gridspec_kw={"width_ratios": [3, 1], "wspace": 2.0},
    )
    # wide gap between the two heatmaps to fit the design colorbar + "x beta"
    # + arrow; right margin leaves room for the choice colorbar and its labels
    fig.subplots_adjust(left=0.08, right=0.78, top=0.92, bottom=0.08)

    # continuous left → neutral → right map for the real-valued design matrix
    cmap_design = LinearSegmentedColormap.from_list(
        "bias_map", ["#377eb8", "white", "#4daf4a"]
    )
    # the choice is binary, so it gets its own two-colour map:
    # -1 (right) → blue, +1 (left) → green
    cmap_choice = ListedColormap(["#377eb8", "#4daf4a"])
    norm_choice = BoundaryNorm([-2, 0, 2], cmap_choice.N)

    # dedicated colorbar axes (repositioned after the first draw)
    cbar_design = fig.add_axes([0.70, 0.3, 0.022, 0.4])
    cbar_choice = fig.add_axes([0.82, 0.3, 0.022, 0.4])

    # ---- heatmap 1: full design matrix ----
    sns.heatmap(
        np.asarray(X)[:n_trials, :],
        ax=axes[0],
        square=True,
        cmap=cmap_design,
        cbar=True,
        cbar_ax=cbar_design,
        vmin=-2.4,
        vmax=2.4,
        linewidths=0.5,
        linecolor="black",
    )
    axes[0].set_xticks(
        [0.5, 1.5, 2.5],
        ["Sign. contr.", "WSLS", "Prev. choice"],
        rotation=90,
    )
    axes[0].set_yticks([])
    axes[0].set_ylabel("Trials")
    axes[0].set_title("Design \nmatrix")

    # ---- heatmap 2: choices (separate bi-colour scale) ----
    sns.heatmap(
        np.asarray(choices).reshape(-1, 1)[:n_trials],
        ax=axes[1],
        square=True,
        cmap=cmap_choice,
        norm=norm_choice,
        cbar=True,
        cbar_ax=cbar_choice,
        linewidths=0.5,
        linecolor="black",
    )
    axes[1].set_xticks([0.5], ["Choices"], rotation=90)
    # the choices heatmap re-adds y-ticks via sharey; hide them again
    axes[1].set_yticks([])
    cbar_choice.set_yticks([-1, 1])
    cbar_choice.set_yticklabels(["Right", "Left"])

    # seaborn hides the spines, so the outer cell borders look clipped; re-enable
    # them to close the black grid around each heatmap
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color("black")

    # square=True shrinks the axes to fit; redraw so we read their true extents
    fig.canvas.draw()
    pos0 = axes[0].get_position()
    pos1 = axes[1].get_position()

    # each colorbar sits right next to its own heatmap, matching its height
    cbar_design.set_position([pos0.x1 + 0.012, pos0.y0, 0.022, pos0.height])
    cbar_choice.set_position([pos1.x1 + 0.012, pos1.y0, 0.022, pos1.height])

    mid_y = (pos0.y0 + pos0.y1) / 2

    # "x beta" sits in the gap after the design colorbar...
    fig.text(
        pos0.x1 + 0.08,
        mid_y,
        r"$\times\,\beta$",
        ha="left",
        va="center",
        fontsize=15,
    )

    # ...then an arrow to the choices. The arrow (rather than "=") leaves room
    # for the implied nonlinearity + Bernoulli emission, illustrated by the
    # coin-flip icon centred on top of the arrow. Keep a clear gap after beta.
    x_s = pos0.x1 + 0.175
    x_e = pos1.x0 - 0.008
    fig.add_artist(
        FancyArrowPatch(
            (x_s, mid_y),
            (x_e, mid_y),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=16,
            lw=1.6,
            color="black",
        )
    )

    # Put the coin on its own axes so placement is independent of dpi/bbox:
    # a square box (in inches) centred on the arrow, sitting just above it.
    if _COINFLIP_ICON.exists():
        fig_w, fig_h = fig.get_size_inches()
        coin_w = 0.11  # figure-fraction width
        coin_h = coin_w * fig_w / fig_h  # keep the box square in inches
        coin_ax = fig.add_axes(
            [(x_s + x_e) / 2 - coin_w / 2, mid_y + 0.025, coin_w, coin_h]
        )
        coin_ax.imshow(mpimg.imread(str(_COINFLIP_ICON)))
        coin_ax.axis("off")

    plt.show()
    return fig


def plot_posteriors(posteriors, session, n_states=3, sess_to_plot=None):
    """Plot the posterior state probabilities over trials for example sessions.

    For each chosen session, plots ``P(state)`` at every trial (one colored
    line per latent state). Sustained, near-1 probabilities reveal the model
    committing to a single state for long stretches of trials.

    Parameters
    ----------
    posteriors :
        Array of shape ``(n_trials_total, n_states)`` with the per-trial
        posterior probability of each state.
    session :
        Array of session ids aligned with ``posteriors`` rows, used to slice out
        each example session.
    n_states :
        Number of latent states. Default 3.
    sess_to_plot :
        Session ids to display (one subplot each). If ``None``, uses three
        representative example sessions.

    Returns
    -------
    matplotlib.figure.Figure
        The figure with one subplot per example session.
    """
    if sess_to_plot is None:
        sess_to_plot = [
            "0ccee376-2873-47dd-9293-c19e424c1bee",
            "66f20f92-171f-4cc5-aca9-69fc3cb6370f",
            "19f4acbd-aeac-4f83-9f30-85a8aa002820",
        ]

    # Get these sessions' indexes
    sess_examples = [np.where(session == s)[0] for s in sess_to_plot]

    colors = ["#ff7f00", "#4daf4a", "#377eb8"]
    fig, ax = plt.subplots(1, len(sess_examples), figsize=(20, 4))

    for i, sess_ex in enumerate(sess_examples):
        for state in range(n_states):
            # Plot all trials for a given session and state
            ax[i].plot(
                posteriors[sess_ex][:, state],
                label="State " + str(state + 1),
                lw=3,
                color=colors[state],
            )
            ax[i].set_title("Example session " + str(i + 1))
            if i == 0:
                ax[i].set_xticks([0, 45, 90], ["0", "45", "90"])
                ax[i].set_ylabel("P(state)")
                ax[i].set_xlabel("Trial #")
                ax[i].set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
            else:
                ax[i].set_xticks([0, 45, 90], [" ", " ", " "])
                ax[i].set_yticks([0, 0.5, 1], [" ", " ", " "])
    return fig


def plot_accuracy_and_occupancy(frac_occupancy, accuracies_to_plot):
    """Plot per-state fraction of occupancy and predictive accuracy as bar charts.

    Left panel: the fraction of trials spent in each latent state. Right panel:
    the model's choice-prediction accuracy overall ("All") and within each
    state, so you can see whether some states are more predictable than others.

    Parameters
    ----------
    frac_occupancy :
        Per-state occupancy fractions, length ``n_states`` (should sum to ~1).
    accuracies_to_plot :
        Accuracies in ``[0, 1]``, length ``n_states + 1``; the first entry is
        the overall accuracy and the rest are per-state accuracies.

    Returns
    -------
    matplotlib.figure.Figure
        The figure with the occupancy and accuracy bar charts.
    """
    cols = [
        "#ff7f00", "#4daf4a", "#377eb8", "#f781bf", "#a65628", "#984ea3",
        "#999999", "#e41a1c", "#dede00",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: state occupancies
    ax = axes[0]
    for z, occ in enumerate(frac_occupancy):
        ax.bar(z, occ, width=0.8, color=cols[z])
        ax.text(z, occ, f"{occ:.2f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylim(0, 1)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["1", "2", "3"])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("state")
    ax.set_ylabel("frac. occupancy")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # Right: accuracies
    ax = axes[1]
    for z, acc in enumerate(accuracies_to_plot):
        col = "grey" if z == 0 else cols[z - 1]
        ax.bar(z, acc * 100, width=0.8, color=col)
        ax.text(z, acc * 100 + 1, f"{acc*100:.2f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylim(50, 100)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["All", "1", "2", "3"])
    ax.set_yticks([50, 75, 100])
    ax.set_xlabel("state")
    ax.set_ylabel("accuracy (%)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig
