import colorsys

import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Rectangle

import nemos as nmo

plt.rcParams.update(
    {
        "figure.dpi": 300,
    }
)

# The saturated tones of the GLM-HMM graphical model and of lnp_model_colscheme,
# so the thumbnails and the two diagrams above them read as one set. They are
# mid-tones, which also keeps them legible on both the light and the dark theme:
# an embedded figure cannot see which one the page is using, only the reader's OS
# preference, and the two do not have to agree.
ORANGE = "#E07B39"
GREEN = "#4F9E4F"
BLUE = "#4C82C3"

# The pastel each of those outlines in the diagrams, for shapes that are filled
# rather than drawn as a line.
PASTEL = {ORANGE: "#FAE0C8", GREEN: "#DDEFDA", BLUE: "#D8E4F4"}


FIGSIZE = (5, 3)

# The loaded NWB files are held here for the lifetime of the build. They own a
# lazy HDF5 handle, and letting the first one be collected while a second is
# open makes the reads on the survivor fail.
_FILES = {}


def _load(name):
    if name not in _FILES:
        _FILES[name] = nap.load_file(nmo.fetch.fetch_data(name))
    return _FILES[name]


def _blank_axes(figsize=FIGSIZE, keep=None):
    if keep is None:
        keep = set()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for side in {"left", "right", "top", "bottom"}.difference(keep):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    # Saved with no background, so the card shows through in either theme.
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    return fig, ax


def plot_counts_thumbnail():
    """Raster of head-direction cells, from the head-direction tutorial recording."""
    data = _load("Mouse32-140822.nwb")
    epochs = data["epochs"]
    wake_ep = epochs[epochs.tags == "wake"]
    adn = data["units"][data["units"].location == "adn"]

    # Ordering the cells by preferred direction groups the ones active together,
    # which is what gives the raster its band structure.
    tuning = nap.compute_tuning_curves(
        adn, data["ry"].restrict(wake_ep), 60, feature_names=["angle"]
    )
    order = list(tuning.idxmax(dim="angle").to_pandas().sort_values().index)
    # A handful of cells over a short window, so the ticks stay countable at
    # thumbnail size.

    # The window the tutorial uses to show individual spikes.
    window = nap.IntervalSet(start=8819.4, end=8821 + 3)
    max_selected = 8
    selected = [unit for unit in order if 3 < len(adn[unit].restrict(window).t) < 15]
    selected = selected[:max_selected]
    fig, ax = _blank_axes(keep={"left", "bottom"})
    for row, unit in enumerate(selected):
        ax.vlines(
            adn[unit].restrict(window).t, row + 0.25, row + 0.75, color=BLUE, lw=3
        )
    ax.set_xlim(window.start[0] - 0.1, window.end[0])
    ax.set_ylim(0.0, len(selected))
    # The card renders this 3-inch-tall figure at 100px, so type has to be set
    # far larger than the on-figure size suggests to survive the reduction.
    ax.set_xlabel("time", fontsize=24)
    ax.set_ylabel("unit", fontsize=24)
    fig.tight_layout()


def plot_continuous_thumbnail():
    """Calcium transients of one cell, from the calcium imaging tutorial recording."""
    transients = _load("A0670-221213.nwb")["RoiResponseSeries"]

    start = transients.t[0] + 300
    chunk = transients.restrict(nap.IntervalSet(start=start, end=start + 45))

    fig, ax = _blank_axes()
    ax.plot(chunk.t, np.asarray(chunk.d)[:, 54], color=BLUE, lw=2.4)
    ax.set_xlim(chunk.t[0], chunk.t[-1])
    fig.tight_layout()


def plot_binary_thumbnail():
    """Grid of ones and zeros, standing in for a binary response per time bin."""
    rng = np.random.default_rng(0)
    n_rows, n_cols = 7, 12
    bits = rng.integers(0, 2, size=(n_rows, n_cols))

    fig, ax = _blank_axes()
    for row in range(n_rows):
        for col in range(n_cols):
            ax.text(
                col,
                n_rows - row,
                str(bits[row, col]),
                color=GREEN,
                fontsize=15,
                family="monospace",
                ha="center",
                va="center",
            )
    ax.set_xlim(-0.6, n_cols - 0.4)
    ax.set_ylim(0.4, n_rows + 0.6)
    fig.tight_layout()


def plot_choices_thumbnail():
    """Class probabilities over the alternatives of a three-way choice task."""
    labels = ["left", "right", "no-go"]

    fig, ax = _blank_axes(keep={"left", "bottom"})
    edges = [ORANGE, GREEN, BLUE]
    ax.bar(
        labels,
        [0.28, 0.55, 0.17],
        color=[PASTEL[edge] for edge in edges],
        edgecolor=edges,
        linewidth=2.5,
        width=0.6,
    )
    ax.set_ylim(0, 0.62)
    # _blank_axes pins an empty tick locator, which the bar call does not undo,
    # so the category labels have to be put back explicitly.
    ax.set_xticks(range(len(labels)), labels)
    # The card renders this 3-inch-tall figure at 100px, so type has to be set
    # far larger than the on-figure size suggests to survive the reduction.
    ax.tick_params(axis="x", length=0, labelsize=26)
    ax.set_ylabel("p(category)", fontsize=24)
    fig.tight_layout()


def plot_addition_thumbnail():
    """Design matrix of three predictors, each occupying its own block of columns."""
    # Widths in the order of basis_processing.svg: a joint basis over two inputs,
    # then two one-dimensional ones.
    widths = [0.44, 0.28, 0.18]
    gap = 0.05

    fig, ax = _blank_axes(keep={"left", "bottom"})
    left = 0.0
    for index, (width, edge) in enumerate(zip(widths, (ORANGE, GREEN, BLUE)), start=1):
        ax.add_patch(
            Rectangle(
                (left, 0),
                width,
                1,
                facecolor=PASTEL[edge],
                edgecolor=edge,
                linewidth=2.5,
            )
        )
        ax.text(
            left + width / 2,
            -0.06,
            f"basis {index}",
            color=edge,
            fontsize=17,
            ha="center",
            va="bottom",
        )
        left += width + gap
    ax.set_xlim(-0.02, left - gap + 0.02)
    ax.set_ylim(-0.02, 1.03)
    ax.invert_yaxis()
    # The card renders this 3-inch-tall figure at 100px, so type has to be set
    # far larger than the on-figure size suggests to survive the reduction.
    ax.set_ylabel("time", fontsize=24)
    ax.set_xlabel("features", fontsize=24)
    fig.tight_layout()


def _cyclic_colours(count):
    """A hue wheel carrying the weight of the default cycle.

    ``hsv`` is cyclic but far louder than the cycle the straight bases use, and
    both twilight maps pass through a near-white that vanishes on a light page.
    Sweeping hue at the saturation and value of the default cycle keeps the
    periodic panel the same weight as the ones beside it.
    """
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"][:count]
    reference = [colorsys.rgb_to_hsv(*to_rgb(colour)) for colour in cycle]
    saturation = float(np.mean([s for _, s, _ in reference]))
    value = float(np.mean([v for _, _, v in reference]))
    return [
        colorsys.hsv_to_rgb(hue, saturation, value)
        for hue in np.linspace(0, 1, count, endpoint=False)
    ]


def _draw_kernels(ax, x, kernels, periodic=False):
    """One basis drawn into a panel.

    The straight bases take matplotlib's default cycle, the one the basis table
    in the user guide is drawn with. A basis that wraps takes a cyclic map
    instead, so its last element comes back to the colour of the first.
    """
    if periodic:
        count = kernels.shape[1]
        colours = _cyclic_colours(count)
        for index in range(count):
            ax.plot(x, kernels[:, index], color=colours[index], lw=1.6)
    else:
        ax.plot(x, kernels, lw=1.6)
    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_alpha(0)


def plot_zoo_thumbnail():
    """Four of the basis families, enough to show the shapes on offer differ."""
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    fig.patch.set_alpha(0)

    # The cyclic basis is drawn with the looping palette, the rest with the
    # straight one, so the panel that wraps is marked out by its colours too.
    bases = (
        (nmo.basis.BSplineEval(n_basis_funcs=6), False),
        (nmo.basis.RaisedCosineLogEval(n_basis_funcs=6), False),
        (nmo.basis.FourierEval(frequencies=3), False),
        (nmo.basis.CyclicBSplineEval(n_basis_funcs=6), True),
    )
    for ax, (basis, periodic) in zip(axes.ravel(), bases):
        _draw_kernels(ax, *basis.evaluate_on_grid(200), periodic=periodic)
    fig.subplots_adjust(
        left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.08, hspace=0.12
    )


def plot_product_thumbnail():
    """One element of a basis over two inputs, beside the pair it is a product of."""
    a_basis = nmo.basis.BSplineEval(n_basis_funcs=6)
    b_basis = nmo.basis.BSplineEval(n_basis_funcs=6)
    i, j = 2, 3

    x, a_kernels = a_basis.evaluate_on_grid(200)
    y, b_kernels = b_basis.evaluate_on_grid(200)
    _, _, Z = (a_basis * b_basis).evaluate_on_grid(150, 150)
    element = Z[:, :, i * b_basis.n_basis_funcs + j]

    fig = plt.figure(figsize=FIGSIZE)
    fig.patch.set_alpha(0)
    grid = fig.add_gridspec(
        2, 2, width_ratios=(1, 4), height_ratios=(1, 4), wspace=0.04, hspace=0.04
    )
    top, side, main = (
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    )

    for ax in (top, side, main):
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.patch.set_alpha(0)

    # The two factors, greyed except the element whose product is shown.
    top.plot(x, a_kernels, color=ORANGE, lw=1.4, alpha=0.4)
    top.plot(x, a_kernels[:, i], color=ORANGE, lw=3)
    side.plot(b_kernels, y, color=BLUE, lw=1.4, alpha=0.4)
    side.plot(b_kernels[:, j], y, color=BLUE, lw=3)
    side.invert_xaxis()

    # Levels start above zero so the flat surround is left unpainted rather than
    # washed with the bottom of the colour map.
    ceiling = element.max()
    main.contourf(
        element.T,
        levels=np.linspace(0.04 * ceiling, ceiling, 7),
        cmap=LinearSegmentedColormap.from_list("product", [PASTEL[GREEN], GREEN]),
        extend="neither",
    )

    # A gridspec figure cannot use tight_layout, so the margins are set directly;
    # the default ones leave the drawing small inside the card.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)


# White type and spines for the dark theme. The marks themselves keep the palette
# colours, which carry on either background.
_DARK_RC = {
    "text.color": "#ffffff",
    "axes.labelcolor": "#ffffff",
    "axes.edgecolor": "#ffffff",
    "xtick.color": "#ffffff",
    "ytick.color": "#ffffff",
}


def _dark_variant(draw):
    """Name a dark-theme twin of a thumbnail, drawn by the same function."""

    def variant():
        with plt.rc_context(_DARK_RC):
            draw()

    variant.__name__ = f"{draw.__name__}_dark"
    variant.__doc__ = f"Dark-theme variant of :func:`{draw.__name__}`."
    return variant


plot_counts_thumbnail_dark = _dark_variant(plot_counts_thumbnail)
plot_continuous_thumbnail_dark = _dark_variant(plot_continuous_thumbnail)
plot_binary_thumbnail_dark = _dark_variant(plot_binary_thumbnail)
plot_choices_thumbnail_dark = _dark_variant(plot_choices_thumbnail)
plot_addition_thumbnail_dark = _dark_variant(plot_addition_thumbnail)
plot_zoo_thumbnail_dark = _dark_variant(plot_zoo_thumbnail)
plot_product_thumbnail_dark = _dark_variant(plot_product_thumbnail)
