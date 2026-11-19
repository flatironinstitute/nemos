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
    "relabel_by_bias",
    "simulate_glm_hmm",
    "plot_proba_left",
    "plot_glm_weights",
    "plot_transition_matrix",
    "plot_design_matrix",
    "plot_posteriors",
    "plot_accuracy_and_occupancy",
    "plot_simulated_session",
    "plot_state_weights",
    "plot_param_recovery",
    "plot_state_inference",
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
    has_three_blocks = trials.groupby("session")["probabilityLeft"].agg(
        lambda s: set(s.unique()) == {0.2, 0.5, 0.8}
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


def relabel_by_bias(model):
    """Reorder the states of a fitted GLM-HMM by their bias, engaged state first.

    The latent-state labels found by EM are arbitrary, so two fits of the same
    data can agree on everything but the ordering of the states. This puts them
    in a canonical order: the state with the smallest absolute bias (the
    engaged state, whose choices track the stimulus) comes first, followed by
    the remaining states sorted by decreasing bias (biased-left, then
    biased-right).

    Unlike :func:`relabel`, this does not assume a threshold on the bias
    magnitude, so it is safe to use on simulated data with arbitrary bias
    values.

    Parameters
    ----------
    model :
        Fitted GLM-HMM model.

    Returns
    -------
    model :
        GLM-HMM model with states reordered in-place. Because ``smooth_proba``
        and ``decode_state`` read the fitted attributes, their outputs follow
        the new ordering too.
    """
    intercept = np.asarray(model.intercept_)
    engaged = int(np.argmin(np.abs(intercept)))
    biased = sorted(
        (k for k in range(intercept.shape[0]) if k != engaged),
        key=lambda k: -intercept[k],
    )
    order = np.array([engaged, *biased])

    model.coef_ = model.coef_[:, order]
    model.intercept_ = model.intercept_[order]
    model.initial_prob_ = model.initial_prob_[order]
    model.transition_prob_ = model.transition_prob_[order][:, order]
    return model


def simulate_glm_hmm(X, coef, intercepts, transition_prob, initial_prob, seed=0):
    """Sample choices and hidden states from a GLM-HMM with known parameters.

    :meth:`GLMHMM.simulate <nemos.glm_hmm.GLMHMM.simulate>` draws from a
    *fitted* model, so to sample from parameters we choose ourselves we fit a
    throwaway model on placeholder choices only to populate its fitted
    attributes, then overwrite those attributes with the ground truth before
    sampling.

    Parameters
    ----------
    X :
        Design matrix of shape ``(n_samples, n_features)``.
    coef :
        Ground-truth GLM weights of shape ``(n_features, n_states)``.
    intercepts :
        Ground-truth per-state intercepts (biases) of shape ``(n_states,)``.
    transition_prob :
        Ground-truth transition matrix of shape ``(n_states, n_states)``.
    initial_prob :
        Ground-truth initial state distribution of shape ``(n_states,)``.
    seed :
        Integer seed for the placeholder choices and the sampling.

    Returns
    -------
    y :
        Simulated binary choices of shape ``(n_samples,)``, as floats.
    z :
        The (normally hidden) state sequence of shape ``(n_samples,)``, as ints.
    """
    import jax
    import jax.numpy as jnp

    import nemos as nmo

    intercepts = np.asarray(intercepts)
    n_states = intercepts.shape[0]
    rng = np.random.default_rng(seed)

    model = nmo.glm_hmm.GLMHMM(
        n_states=n_states,
        regularizer="UnRegularized",
        seed=jax.random.PRNGKey(seed),
    )
    # placeholder fit, only to populate the fitted attributes
    model.fit(X, rng.integers(0, 2, size=X.shape[0]))

    model.coef_ = jnp.asarray(coef, dtype=jnp.float64)
    model.intercept_ = jnp.asarray(intercepts, dtype=jnp.float64)
    model.transition_prob_ = jnp.asarray(transition_prob, dtype=jnp.float64)
    model.initial_prob_ = jnp.asarray(initial_prob, dtype=jnp.float64)

    y, _, z = model.simulate(
        jax.random.PRNGKey(seed + 1), X, state_format="index"
    )
    return np.asarray(y, dtype=float), np.asarray(z).astype(int)


def plot_simulated_session(X, y, z_true, state_labels=None, window=(0, 200)):
    """Show a stretch of simulated trials: the hidden state and the choices.

    The top panel shades each trial by the state that generated it, making the
    multi-trial "blocks" produced by a sticky transition matrix visible. The
    bottom panel plots the stimulus against the resulting choice, so the
    difference between an engaged state (choices track the stimulus) and a
    biased state (choices ignore it) is visible directly in the data.

    Parameters
    ----------
    X :
        Design matrix of shape ``(n_samples, 1)``: a single signed-evidence
        predictor.
    y :
        Simulated binary choices of shape ``(n_samples,)``.
    z_true :
        Ground-truth state sequence of shape ``(n_samples,)``.
    state_labels :
        Optional list of ``n_states`` names. Defaults to
        ``["state 1", "state 2", ...]``.
    window :
        ``(start, stop)`` trial range to display. Default ``(0, 200)``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure with the simulated session.
    """
    X = np.asarray(X).reshape(-1)
    y = np.asarray(y).reshape(-1)
    z_true = np.asarray(z_true).reshape(-1)

    start, stop = window
    stop = min(stop, z_true.shape[0])
    trials = np.arange(start, stop)
    n_states = int(z_true.max()) + 1
    if state_labels is None:
        state_labels = [f"state {k + 1}" for k in range(n_states)]

    # same ordering as plot_param_recovery and plot_state_inference, so a given
    # state keeps its color across every figure on the page
    colors = ["#377eb8", "#ff7f00", "#4daf4a", "#984ea3", "#a65628"]

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 4.5), sharex=True, height_ratios=[1, 3]
    )

    # ---- panel 1: the hidden state, as contiguous blocks ----
    ax = axes[0]
    for k in range(n_states):
        ax.fill_between(
            trials,
            0,
            1,
            where=(z_true[start:stop] == k),
            color=colors[k % len(colors)],
            step="post",
            label=state_labels[k],
        )
    ax.set_yticks([])
    ax.set_ylabel("state", rotation=0, ha="right", va="center")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 2.0), ncol=n_states, frameon=False)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)

    # ---- panel 2: the stimulus and the choice it produced ----
    ax = axes[1]
    # stems rather than a line: at this scale a connected trace is unreadable,
    # and what matters is the sign of the evidence on each individual trial
    ax.vlines(trials, 0, X[start:stop], color="0.65", lw=1.0)
    ax.scatter(
        trials,
        np.where(y[start:stop] == 1, 1.35, -1.35),
        c=[colors[k % len(colors)] for k in z_true[start:stop]],
        s=9,
    )
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_yticks([-1.35, 0, 1.35])
    ax.set_yticklabels(["choice: right", "signed contrast", "choice: left"])
    ax.set_xlabel("trial")
    ax.set_xlim(start, stop)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig


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
    recovered_weights[1, :] = model.intercept_  # bias
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


def plot_state_weights(coef, intercepts, state_labels=None):
    """Plot the per-state GLM weights of a two-covariate GLM-HMM.

    One line per latent state shows the weight the state puts on the stimulus
    and its bias (intercept), making the difference between a state whose
    choices follow the stimulus and one that leans to a fixed side visible at a
    glance. This is the two-covariate counterpart of
    :func:`plot_glm_weights`, which plots the four covariates of the applied
    tutorial.

    Parameters
    ----------
    coef :
        Per-state stimulus weights of shape ``(1, n_states)``.
    intercepts :
        Per-state intercepts of shape ``(n_states,)``.
    state_labels :
        Optional list of ``n_states`` names. Defaults to
        ``["state 1", "state 2", ...]``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure with the weight traces.
    """
    coef = np.asarray(coef).reshape(-1)
    intercepts = np.asarray(intercepts).reshape(-1)
    n_states = intercepts.shape[0]
    if state_labels is None:
        state_labels = [f"state {k + 1}" for k in range(n_states)]

    # the palette of plot_simulated_session, so states keep their color
    colors = ["#377eb8", "#ff7f00", "#4daf4a", "#984ea3", "#a65628"]
    weights = np.stack([coef, intercepts])  # (2, n_states)

    fig, ax = plt.subplots(figsize=(5, 4))
    for state in range(n_states):
        ax.plot(
            range(2),
            weights[:, state],
            color=colors[state % len(colors)],
            marker="o",
            lw=1.5,
            label=state_labels[state],
        )
    ax.axhline(y=0, color="k", alpha=0.5, ls="--")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["signed contrast", "bias"])
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylabel("GLM weight")
    ax.set_xlabel("covariate")
    ax.legend()
    fig.tight_layout()
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
    import matplotlib.image as mpimg
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import FancyArrowPatch

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
        "#ff7f00",
        "#4daf4a",
        "#377eb8",
        "#f781bf",
        "#a65628",
        "#984ea3",
        "#999999",
        "#e41a1c",
        "#dede00",
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
        ax.text(
            z, acc * 100 + 1, f"{acc * 100:.2f}", ha="center", va="bottom", fontsize=10
        )

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


def plot_param_recovery(
    true_coef,
    true_intercept,
    rec_coef,
    rec_intercept,
    true_trans,
    rec_trans,
    state_labels=None,
):
    """Compare ground-truth and recovered GLM-HMM parameters side by side.

    The left panel overlays the true and recovered per-state weights (stimulus
    slope and intercept/bias) as grouped bars; the middle and right panels show
    the true and recovered transition matrices as heatmaps. States are assumed
    to already be aligned between the two parameter sets.

    Parameters
    ----------
    true_coef, rec_coef :
        Ground-truth and recovered GLM weights of shape ``(n_features, n_states)``.
    true_intercept, rec_intercept :
        Ground-truth and recovered per-state intercepts of shape ``(n_states,)``.
    true_trans, rec_trans :
        Ground-truth and recovered transition matrices of shape
        ``(n_states, n_states)``.
    state_labels :
        Optional list of ``n_states`` names used in the legend and on the
        transition-matrix ticks. Defaults to ``["state 1", "state 2", ...]``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure with the parameter-recovery comparison.
    """
    true_coef = np.asarray(true_coef)
    rec_coef = np.asarray(rec_coef)
    true_intercept = np.asarray(true_intercept)
    rec_intercept = np.asarray(rec_intercept)
    true_trans = np.asarray(true_trans)
    rec_trans = np.asarray(rec_trans)

    n_states = true_intercept.shape[0]
    # stack the stimulus weight(s) and the intercept into one "parameter" axis
    true_vals = np.vstack(
        [true_coef, true_intercept[None, :]]
    )  # (n_features+1, n_states)
    rec_vals = np.vstack([rec_coef, rec_intercept[None, :]])
    n_params = true_vals.shape[0]
    param_labels = (
        ["signed contrast"]
        if true_coef.shape[0] == 1
        else [f"stim w{i + 1}" for i in range(true_coef.shape[0])]
    ) + ["bias"]
    if state_labels is None:
        state_labels = [f"state {k + 1}" for k in range(n_states)]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # ---- panel 1: weights, true vs recovered ----
    ax = axes[0]
    x = np.arange(n_params)
    # same palette and per-state line style as plot_state_weights, so the true
    # values here can be read against the ground-truth figure of the overview
    palette = ["#377eb8", "#ff7f00", "#4daf4a", "#984ea3", "#a65628"]
    for k in range(n_states):
        color = palette[k % len(palette)]
        ax.plot(
            x,
            true_vals[:, k],
            color=color,
            marker="o",
            lw=1.5,
            label=state_labels[k],
        )
        ax.plot(
            x,
            rec_vals[:, k],
            color=color,
            marker="o",
            markerfacecolor="white",
            lw=1.5,
            ls=":",
        )
    # one neutral entry per line style, so the legend stays short
    ax.plot([], [], color="0.35", lw=1.5, label="true")
    ax.plot([], [], color="0.35", lw=1.5, ls=":", markerfacecolor="white",
            marker="o", label="recovered")
    ax.axhline(0, color="k", alpha=0.5, ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(param_labels)
    ax.set_xlim(-0.25, n_params - 0.75)
    ax.set_ylabel("GLM weight")
    ax.set_xlabel("covariate")
    ax.set_title("GLM weights")
    # headroom so the legend does not sit on top of the highest line
    low, high = ax.get_ylim()
    ax.set_ylim(low, high + 0.32 * (high - low))
    ax.legend(fontsize=11, ncol=2, loc="upper center", framealpha=0.9)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # ---- panels 2 and 3: transition matrices ----
    for ax, mat, title in (
        (axes[1], true_trans, "True transitions"),
        (axes[2], rec_trans, "Recovered transitions"),
    ):
        ax.imshow(mat, vmin=0, vmax=1, cmap="bone")
        for i in range(n_states):
            for j in range(n_states):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="k")
        ax.set_xticks(range(n_states))
        ax.set_xticklabels(state_labels, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(n_states))
        ax.set_yticklabels(state_labels, fontsize=9)
        ax.set_xlabel("state t")
        ax.set_ylabel("state t-1")
        ax.set_title(title)

    plt.tight_layout()
    plt.show()
    return fig


def plot_state_inference(
    z_true,
    posteriors,
    posterior_states,
    viterbi_states,
    state_labels=None,
    window=(0, 200),
    stimulus=None,
    choices=None,
):
    """Visualize forward-backward and Viterbi inference against the true states.

    Over a window of time steps, the top panel shows the ground-truth state sequence
    together with the Viterbi (:meth:`decode_state
    <nemos.glm_hmm.GLMHMM.decode_state>`) path, a single globally consistent
    sequence, and the second panel shows the smoothed per-state posteriors
    (:meth:`smooth_proba <nemos.glm_hmm.GLMHMM.smooth_proba>`), the marginals
    over states at each time step. If ``stimulus`` is given, a third panel shows
    the stimulus at each time step, colored by the choice when ``choices`` is
    also given.

    Parameters
    ----------
    z_true :
        Ground-truth latent-state sequence of shape ``(n_samples,)``.
    posteriors :
        Smoothed per-state posterior probabilities of shape
        ``(n_samples, n_states)``, aligned so that column ``k`` corresponds to
        state ``k`` in ``z_true`` and ``viterbi_states``.
    viterbi_states :
        Viterbi-decoded state at each time step, shape ``(n_samples,)``.
    state_labels :
        Optional list of ``n_states`` names for the y-axis and legend. Defaults
        to ``["state 1", "state 2", ...]``.
    window :
        ``(start, stop)`` range of time steps to display. Default ``(0, 200)``.
    stimulus :
        Optional signed stimulus at each time step, shape ``(n_samples,)`` or
        ``(n_samples, 1)``; positive values are stimuli on the left.
    choices :
        Optional choice at each time step, shape ``(n_samples,)``, with 1 for
        left and 0 for right. Used only together with ``stimulus``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure with the state-inference comparison.
    """
    a, b = window
    # the state sequences may arrive as JAX arrays, and decode_state returns its
    # indices as floats, so convert them to integer NumPy arrays here
    z_true_full = np.asarray(z_true, dtype=int)
    z_true = z_true_full[a:b]
    posteriors_full = np.asarray(posteriors)
    viterbi_full = np.asarray(viterbi_states, dtype=int)
    posterior_map_full = np.asarray(posterior_states, dtype=int)
    map_acc = (posterior_map_full == z_true_full).mean()
    viterbi_acc = (viterbi_full == z_true_full).mean()
    posteriors = posteriors_full[a:b]
    viterbi_states = viterbi_full[a:b]
    posterior_map = posterior_map_full[a:b]
    time_steps = np.arange(a, b)
    n_states = posteriors.shape[1]

    if state_labels is None:
        state_labels = [f"state {k + 1}" for k in range(n_states)]

    palette = ["#377eb8", "#ff7f00", "#4daf4a", "#984ea3", "#a65628"]
    colors = [palette[k % len(palette)] for k in range(n_states)]

    show_stimulus = stimulus is not None
    height_ratios = [1, 1.2, 0.9] if show_stimulus else [1, 1.2]
    fig, axes = plt.subplots(
        len(height_ratios),
        1,
        figsize=(11, 5.4 + 2.4 * show_stimulus),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )

    # ---- panel 1: the three state sequences, aligned ----
    ax = axes[0]
    rows = [
        ("ground truth", z_true),
        (f"smooth_proba\n(per-step MAP)\n{map_acc:.1%} correct", posterior_map),
        (f"decode_state\n(Viterbi)\n{viterbi_acc:.1%} correct", viterbi_states),
    ]
    # one bar per time step, centered on it like the posterior steps below, so a
    # state that lasts a single time step is still drawn
    for row, (_, seq) in enumerate(rows):
        ax.bar(
            time_steps,
            0.76,
            width=1.0,
            bottom=row - 0.38,
            color=[colors[k] for k in seq],
            linewidth=0,
        )
    # mark the time steps each readout gets wrong, so disagreements are countable
    for row, (_, seq) in enumerate(rows[1:], start=1):
        wrong = time_steps[seq != z_true]
        ax.plot(wrong, np.full(wrong.shape, row + 0.27), marker="|", ms=7,
                ls="none", color="k", alpha=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _ in rows], fontsize=9)
    # ground truth on top, with room above it for the legend
    ax.set_ylim(len(rows) - 0.5, -1.3)
    ax.tick_params(left=False)
    ax.set_title("Which state was active, by each readout")
    handles = [
        plt.Line2D([], [], color=colors[k], lw=6, label=state_labels[k])
        for k in range(n_states)
    ]
    handles.append(
        plt.Line2D([], [], color="k", marker="|", ms=7, ls="none", alpha=0.8,
                   label="wrong time step")
    )
    ax.legend(handles=handles, loc="upper right", fontsize=9, ncol=len(handles),
              framealpha=0.9)
    for side in ("right", "top", "left"):
        ax.spines[side].set_visible(False)

    # ---- panel 2: the posteriors those readouts come from ----
    ax = axes[1]
    for k in range(n_states):
        ax.step(
            time_steps,
            posteriors[:, k],
            where="mid",
            color=colors[k],
            lw=2,
            label=f"P({state_labels[k]})",
        )
    # room above 1 for the legend
    ax.set_ylim(-0.02, 1.3)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("posterior")
    ax.legend(loc="upper right", fontsize=9, ncol=n_states, framealpha=0.9)
    ax.set_title("Marginal posteriors at each time step (smooth_proba)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # ---- panel 3 (optional): the stimulus, colored by the choice ----
    if show_stimulus:
        ax = axes[2]
        stim = np.asarray(stimulus).reshape(len(z_true_full), -1)[a:b, 0]
        if choices is None:
            ax.vlines(time_steps, 0, stim, color="0.5", lw=2.5)
        else:
            chose_left = np.asarray(choices)[a:b] == 1
            ax.vlines(time_steps[chose_left], 0, stim[chose_left], color="0.15",
                      lw=2.5, label="chose left")
            ax.vlines(time_steps[~chose_left], 0, stim[~chose_left], color="0.7",
                      lw=2.5, label="chose right")
            ax.legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.9)
        ax.axhline(0, color="k", lw=0.8)
        # room above the stimuli for the legend
        lim = np.abs(stim).max()
        ax.set_ylim(-1.1 * lim, 1.7 * lim)
        ax.set_ylabel("stimulus\n(− right, + left)")
        ax.set_title("Stimulus at each time step")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    axes[-1].set_xlabel("time step")
    plt.tight_layout()
    plt.show()
    return fig
