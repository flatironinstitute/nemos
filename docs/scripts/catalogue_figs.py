import colorsys

import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Rectangle
from scipy.stats import gamma, nbinom, norm, poisson

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


def plot_basis_scheme_thumbnail():
    """A weighted sum of raised cosines, and the two-peaked tuning it composes."""
    basis = nmo.basis.RaisedCosineLinearEval(n_basis_funcs=10)
    x, kernels = basis.evaluate_on_grid(400)
    # All the weight on the third basis function and, half as much, on the
    # seventh: a tall mode and a shorter one. Nothing on either end, so the
    # tails of the sum reach zero inside the panel.
    weights = np.array([0.0, 0.0, 1.0, 0.4, 0.0, 0.0, 0.55, 0.25, 0.0, 0.0])
    tuning = kernels @ weights

    # Wide and shallow: the card gives this the full width of a half band, and
    # the three panels are only legible there if they are not also tall.
    fig = plt.figure(figsize=(8.4, 1.5))
    fig.patch.set_alpha(0)
    grid = fig.add_gridspec(1, 3, width_ratios=(1, 0.55, 1.25), wspace=0.45)
    basis_ax, weight_ax, tuning_ax = (fig.add_subplot(grid[0, i]) for i in range(3))
    for ax in (basis_ax, weight_ax, tuning_ax):
        for side in ("right", "top"):
            ax.spines[side].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.patch.set_alpha(0)

    # Headroom over the basis, so the family does not fill its panel top to
    # bottom the way the tuning function it composes does.
    basis_ax.plot(x, kernels, lw=1.4)
    basis_ax.set_xlim(x[0], x[-1])
    basis_ax.set_ylim(0, float(kernels.max()) * 1.7)

    # A marker on a weight of zero would sit half under the axis, so only the
    # basis functions that carry weight are drawn.
    positions = np.arange(len(weights))
    carrying = weights > 0
    weight_ax.vlines(positions[carrying], 0, weights[carrying], color=ORANGE, lw=2.2)
    weight_ax.plot(positions[carrying], weights[carrying], "o", color=ORANGE, ms=5)
    weight_ax.set_xlim(-1, len(weights))
    weight_ax.set_ylim(0, float(weights.max()) * 1.35)

    tuning_ax.fill_between(x, tuning, color=PASTEL[GREEN])
    tuning_ax.plot(x, tuning, color=GREEN, lw=3)
    tuning_ax.set_xlim(x[0], x[-1])
    tuning_ax.set_ylim(0, float(tuning.max()) * 1.15)

    # The card renders this at a tenth of its size, so the operators have to be
    # set far larger than they look on the figure.
    ink = plt.rcParams["text.color"]
    for left, right, symbol in (
        (basis_ax, weight_ax, "\u00d7"),
        (weight_ax, tuning_ax, "\u2192"),
    ):
        gap = (left.get_position().x1 + right.get_position().x0) / 2
        fig.text(gap, 0.5, symbol, ha="center", va="center", fontsize=20, color=ink)


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

    # Which factor runs along which axis of the element below.
    top.set_title("basis 2", color=ORANGE, fontsize=17, pad=2)
    side.set_ylabel("basis 1", color=BLUE, fontsize=17, labelpad=2)

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
    # the default ones leave the drawing small inside the card. The left and top
    # edges keep room for the two labels.
    fig.subplots_adjust(left=0.07, right=0.98, top=0.91, bottom=0.02)


def _penalty_ball(radius, edges=(), figsize=(3, 3)):
    """The unit ball of a penalty, drawn as a surface in three dimensions.

    Every ball here is star-shaped about the origin, so the boundary is given by
    the distance to it along each direction ``u`` of the unit sphere. ``edges``
    names the coordinate planes the surface creases in, which are drawn as
    lines; the section in the xy plane is drawn dashed for all of them, as the
    floor the shape stands on.
    """
    polar, azimuth = np.meshgrid(
        np.linspace(0, np.pi, 80), np.linspace(0, 2 * np.pi, 160), indexing="ij"
    )
    u = np.stack(
        [
            np.sin(polar) * np.cos(azimuth),
            np.sin(polar) * np.sin(azimuth),
            np.cos(polar),
        ]
    )
    x, y, z = u * radius(u)

    fig = plt.figure(figsize=figsize)
    fig.patch.set_alpha(0)
    ax = fig.add_subplot(projection="3d")
    ax.set_axis_off()
    # mplot3d leaves a wide margin around the box; the zoom spends it on the
    # shape, which is what the card has room for.
    ax.set_box_aspect((1, 1, 1), zoom=1.65)
    ax.view_init(elev=14, azim=-58)

    ink = plt.rcParams["text.color"]

    # The three coordinate axes, drawn past the ball so the corners it has (or
    # does not have) on them are readable.
    span = 1.25
    line = dict(color=ink, lw=1.2, zorder=1)
    ax.plot([-span, span], [0, 0], [0, 0], **line)
    ax.plot([0, 0], [-span, span], [0, 0], **line)
    ax.plot([0, 0], [0, 0], [-span, span], **line)

    ax.plot_surface(
        x,
        y,
        z,
        # The fill the densities beside these are drawn with, so the two
        # grids on the landing page read as one set.
        color=PASTEL[BLUE],
        alpha=0.55,
        linewidth=0,
        shade=False,
        # The seam matplotlib antialiases between neighbouring quads shows
        # through a translucent surface as a crosshatch.
        antialiased=False,
        zorder=4,
    )

    angle = np.linspace(0, 2 * np.pi, 400)
    zero, cos, sin = np.zeros_like(angle), np.cos(angle), np.sin(angle)
    sections = {
        "xy": np.stack([cos, sin, zero]),
        "xz": np.stack([cos, zero, sin]),
        "yz": np.stack([zero, cos, sin]),
    }
    towards_viewer = np.array(
        [
            np.cos(np.radians(14)) * np.cos(np.radians(-58)),
            np.cos(np.radians(14)) * np.sin(np.radians(-58)),
            np.sin(np.radians(14)),
        ]
    )
    for plane, direction in sections.items():
        if plane != "xy" and plane not in edges:
            continue
        curve = direction * radius(direction)

        # Each ball is convex, so a point of its boundary faces the viewer
        # exactly when a short step towards them leaves the ball. The near half
        # of a section is drawn as the line it is, the far half dashed.
        stepped = curve + 0.02 * towards_viewer[:, None]
        length = np.linalg.norm(stepped, axis=0)
        near = length > radius(stepped / length)

        for keep, style in (
            (near, dict(lw=1.5)),
            (~near, dict(linestyle=(0, (4, 3)), lw=1.1)),
        ):
            # A gap is a break in the line, not a segment joining its ends.
            ax.plot(*np.where(keep, curve, np.nan), color=ink, zorder=6, **style)

    for set_lim in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        set_lim(-span, span)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)


def plot_ridge_thumbnail():
    """The L2 ball: a sphere, which has no corner to push a coefficient onto."""
    _penalty_ball(lambda u: 1.0)


def plot_lasso_thumbnail():
    """The L1 ball: an octahedron, with a corner on each coordinate axis."""
    _penalty_ball(lambda u: 1.0 / np.abs(u).sum(axis=0), edges=("xz", "yz"))


def plot_group_lasso_thumbnail():
    """The ball of a two-group penalty: an edge where the group of two enters."""
    _penalty_ball(lambda u: 1.0 / (np.linalg.norm(u[:2], axis=0) + np.abs(u[2])))


def plot_elastic_net_thumbnail():
    """The elastic-net ball: the octahedron, rounded everywhere but its corners.

    ``ratio`` weights the L1 part, so the boundary along ``u`` solves the
    quadratic ``(1 - ratio) r**2 + ratio * |u|_1 * r = 1``.
    """
    ratio = 0.55

    def radius(u):
        l1 = np.abs(u).sum(axis=0)
        return (-ratio * l1 + np.sqrt((ratio * l1) ** 2 + 4 * (1 - ratio))) / (
            2 * (1 - ratio)
        )

    _penalty_ball(radius, edges=("xz", "yz"))


# The observation models are drawn at a common width so the six read as one set:
# far enough out that the tail of each is visibly over, without so much empty
# axis that the shape shrinks.
_DENSITY_FIGSIZE = (4, 2.6)


def _plot_pmf(support, probabilities):
    """A discrete distribution, as a stem for each outcome it puts mass on."""
    fig, ax = _blank_axes(figsize=_DENSITY_FIGSIZE, keep={"bottom"})
    ax.vlines(support, 0, probabilities, color=BLUE, lw=3)
    ax.plot(support, probabilities, "o", color=BLUE, ms=7)
    # Half an outcome of margin, so the stems at either end of a short support
    # are not drawn on the edge of the axes.
    ax.set_xlim(support[0] - 0.6, support[-1] + 0.6)
    ax.set_ylim(0, probabilities.max() * 1.18)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.06)


def _plot_pdf(x, density):
    """A continuous distribution, as a filled curve."""
    fig, ax = _blank_axes(figsize=_DENSITY_FIGSIZE, keep={"bottom"})
    ax.fill_between(x, density, color=PASTEL[BLUE])
    ax.plot(x, density, color=BLUE, lw=3)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, density.max() * 1.18)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.06)


def plot_poisson_thumbnail():
    """Poisson counts at a rate of three per bin."""
    counts = np.arange(0, 13)
    _plot_pmf(counts, poisson.pmf(counts, 3.0))


def plot_neg_binomial_thumbnail():
    """Negative binomial counts, at the mean of the Poisson beside it and three times its spread."""
    counts = np.arange(0, 13)
    _plot_pmf(counts, nbinom.pmf(counts, 1.5, 1 / 3))


def plot_gamma_thumbnail():
    """A Gamma density: non-negative and right-skewed."""
    x = np.linspace(0, 8, 400)
    _plot_pdf(x, gamma.pdf(x, 2.0))


def plot_gaussian_thumbnail():
    """A Gaussian density, over the whole line and symmetric."""
    x = np.linspace(-4, 4, 400)
    _plot_pdf(x, norm.pdf(x))


def plot_bernoulli_thumbnail():
    """Bernoulli outcomes: mass on zero and one only."""
    _plot_pmf(np.arange(2), np.array([0.65, 0.35]))


def plot_categorical_thumbnail():
    """Categorical outcomes: mass on each of four unordered alternatives."""
    _plot_pmf(np.arange(4), np.array([0.18, 0.42, 0.25, 0.15]))


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
plot_basis_scheme_thumbnail_dark = _dark_variant(plot_basis_scheme_thumbnail)
plot_product_thumbnail_dark = _dark_variant(plot_product_thumbnail)
plot_ridge_thumbnail_dark = _dark_variant(plot_ridge_thumbnail)
plot_lasso_thumbnail_dark = _dark_variant(plot_lasso_thumbnail)
plot_group_lasso_thumbnail_dark = _dark_variant(plot_group_lasso_thumbnail)
plot_elastic_net_thumbnail_dark = _dark_variant(plot_elastic_net_thumbnail)
plot_poisson_thumbnail_dark = _dark_variant(plot_poisson_thumbnail)
plot_neg_binomial_thumbnail_dark = _dark_variant(plot_neg_binomial_thumbnail)
plot_gamma_thumbnail_dark = _dark_variant(plot_gamma_thumbnail)
plot_gaussian_thumbnail_dark = _dark_variant(plot_gaussian_thumbnail)
plot_bernoulli_thumbnail_dark = _dark_variant(plot_bernoulli_thumbnail)
plot_categorical_thumbnail_dark = _dark_variant(plot_categorical_thumbnail)
