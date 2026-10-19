"""Draw the observation-model diagram, light and dark, for the user guide.

An embedded figure cannot see which theme the page is using, so the two variants
are two files, shown by the ``only-light``/``only-dark`` classes. Run

    python docs/scripts/observation_model_scheme.py

after editing anything here, and commit both SVGs.

The colours are the landing page palette, and the themes differ the way
``make_dark_svgs.py`` makes them differ: the ink, the greys and the achromatic
panel fills flip, while the orange, green and blue stay as they are. The pastel
fills stay light too, so the labels sitting on them keep their dark ink in both
variants.
"""

import math
import pathlib
import random

ASSETS = pathlib.Path(__file__).resolve().parent.parent / "assets"

# The saturated tones of the landing page diagrams and thumbnails.
ORANGE = "#E07B39"
GREEN = "#4F9E4F"
BLUE = "#4C82C3"
PASTEL = {ORANGE: "#FAE0C8", GREEN: "#DDEFDA", BLUE: "#D8E4F4"}

# Ink dark enough to stay legible on the pastel fills, which stay light in both themes.
PASTEL_INK = "#1a1a1a"

THEMES = {
    "": dict(ink="#1a1a1a", muted="#666666", line="#9a9a9a", card="#f6f6f6"),
    "_dark": dict(ink="#e8e8e8", muted="#a8a8a8", line="#7a7a7a", card="#262626"),
}

# Smooth mean and Gamma-distributed observations for the input sparklines.
N = 40


def mean_fn(t):
    """Two bumps, so the trace looks like a tuning curve rather than noise."""
    return (
        1.0
        + 2.5 * math.exp(-((t - 0.35) ** 2) / 0.02)
        + 1.6 * math.exp(-((t - 0.8) ** 2) / 0.008)
    )


mean = [mean_fn(i / (N - 1)) for i in range(N)]
_rng = random.Random(3)
obs = [_rng.gammavariate(1 / 0.15, m * 0.15) for m in mean]


def spark_trace(x0, y0, w, h, vals, vmax):
    """Observations as dots on a faint connecting line."""
    dx = w / (len(vals) - 1)
    xy = [(x0 + i * dx, y0 + h - min(v, vmax) / vmax * h) for i, v in enumerate(vals)]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in xy)
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.3" class="data"/>' for x, y in xy
    )
    return f'<polyline points="{pts}" class="dataline"/>' + dots


def spark_line(x0, y0, w, h, vals, vmax):
    """The predicted mean as a single line."""
    dx = w / (len(vals) - 1)
    pts = " ".join(
        f"{x0 + i * dx:.1f},{y0 + h - v / vmax * h:.1f}" for i, v in enumerate(vals)
    )
    return f'<polyline points="{pts}" class="meanline"/>'


# Gamma densities at one mean, for a few scales: shape 1/phi, scale mu * phi.
MU = 2.0
PHIS = [(0.1, 1.0), (0.3, 0.6), (0.6, 0.3)]  # (phi, stroke opacity)
YMAX = 6.0


def gamma_pdf(y, mu, phi):
    """Gamma density at ``y`` with mean ``mu`` and scale ``phi``."""
    k, theta = 1 / phi, mu * phi
    return math.exp(
        (k - 1) * math.log(y) - y / theta - math.lgamma(k) - k * math.log(theta)
    )


def gamma_curves(x0, y0, w, h):
    """The family the instance models: one density per scale, at a fixed mean."""
    ys = [YMAX * (i + 1) / 200 for i in range(200)]
    dmax = max(gamma_pdf(y, MU, PHIS[0][0]) for y in ys) * 1.08

    def sx(y):
        return x0 + y / YMAX * w

    s = f'<line x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}" class="axis"/>'
    for phi, op in PHIS:
        pts = f"{x0:.1f},{y0 + h:.1f} " + " ".join(
            f"{sx(y):.1f},{y0 + h - gamma_pdf(y, MU, phi) / dmax * h:.1f}" for y in ys
        )
        s += f'<polyline points="{pts}" class="pdf" stroke-opacity="{op}"/>'
    xm = sx(MU)
    s += (
        f'<line x1="{xm:.1f}" y1="{y0 - 6}" x2="{xm:.1f}" y2="{y0 + h}" class="mu"/>'
        f'<text x="{xm:.1f}" y="{y0 - 11}" class="small acc" text-anchor="middle">μ</text>'
        f'<text x="{x0 + w}" y="{y0 + h + 18}" class="small pastelmuted" text-anchor="end">y</text>'
    )
    lx, ly = x0 + w - 62, y0 + 4
    for i, (phi, op) in enumerate(PHIS):
        yy = ly + i * 19
        s += (
            f'<line x1="{lx}" y1="{yy}" x2="{lx + 18}" y2="{yy}" class="pdf" stroke-opacity="{op}"/>'
            f'<text x="{lx + 24}" y="{yy + 4.5}" class="legend pastelmuted">φ = {phi}</text>'
        )
    return s


OUTPUTS = [
    ("log_likelihood", "log p(y | μ, φ)"),
    ("deviance", "distance to a perfect fit"),
    ("pseudo_r2", "goodness of fit"),
    ("estimate_scale", "estimate φ from y and μ"),
    ("sample_generator", "draw y ~ p(· | μ, φ)"),
]


def build(c):
    """Return the SVG source for one theme."""
    W, H = 960, 360
    s = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        'font-family="Helvetica, Arial, sans-serif" role="img" '
        'aria-label="An observation model takes observations and a predicted mean and returns '
        'log-likelihood, deviance, pseudo-R2, scale, and samples">',
        "<title>What an observation model does</title>",
        f"""<style>
  .ink {{ fill: {c["ink"]}; }} .muted {{ fill: {c["muted"]}; }} .acc {{ fill: {BLUE}; }}
  .pastelink {{ fill: {PASTEL_INK}; }} .pastelmuted {{ fill: #555555; }}
  .lbl {{ font-size: 17px; }} .small {{ font-size: 14px; }}
  .code {{ font-family: Menlo, Consolas, monospace; }}
  .title {{ font-size: 21px; font-weight: bold; }}
  .card {{ fill: {c["card"]}; stroke: {c["line"]}; stroke-width: 1; }}
  .box {{ fill: {PASTEL[BLUE]}; stroke: {BLUE}; stroke-width: 2.5; }}
  .outcard {{ fill: {PASTEL[GREEN]}; stroke: {GREEN}; stroke-width: 1.5; }}
  .outname {{ fill: {GREEN}; font-weight: bold; }}
  .data {{ fill: {ORANGE}; }}
  .dataline {{ fill: none; stroke: {ORANGE}; stroke-width: 1.2; stroke-opacity: .6; }}
  .pdf {{ fill: none; stroke: {BLUE}; stroke-width: 2.5; stroke-linejoin: round; }}
  .legend {{ font-size: 12.5px; }}
  .meanline {{ fill: none; stroke: {BLUE}; stroke-width: 2.5; stroke-linejoin: round; }}
  .axis {{ stroke: {BLUE}; stroke-width: 1.5; stroke-opacity: .45; }}
  .mu {{ stroke: {BLUE}; stroke-width: 2; stroke-dasharray: 4 3; }}
  .wire {{ fill: none; stroke: {c["line"]}; stroke-width: 2; }}
  .tip {{ fill: {c["line"]}; }}
  .group {{ fill: none; stroke: {c["muted"]}; stroke-width: 1.5; stroke-dasharray: 7 5; }}
  .grouplbl {{ font-size: 15px; font-weight: bold; letter-spacing: 1px; }}
</style>""",
        '<defs><marker id="tip" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse"><path class="tip" d="M0 0 L10 5 L0 10 z"/>'
        "</marker></defs>",
    ]

    s.append('<g transform="translate(0,20)">')

    # What the user brings, and what NeMoS does with it.
    s.append('<rect x="8" y="-12" width="254" height="180" rx="16" class="group"/>')
    s.append('<text x="22" y="14" class="grouplbl muted">DATA</text>')
    s.append('<rect x="8" y="178" width="254" height="152" rx="16" class="group"/>')
    s.append('<text x="22" y="203" class="grouplbl muted">MODEL</text>')
    s.append('<rect x="306" y="-12" width="646" height="342" rx="16" class="group"/>')
    s.append('<text x="320" y="14" class="grouplbl muted">NeMoS</text>')

    # Inputs.
    s.append('<rect x="20" y="40" width="230" height="115" rx="10" class="card"/>')
    s.append(
        '<text x="36" y="66" class="lbl ink">observations '
        '<tspan class="code" font-weight="bold">y</tspan></text>'
    )
    s.append(spark_trace(36, 80, 198, 60, obs, 5.5))
    s.append('<rect x="20" y="213" width="230" height="105" rx="10" class="card"/>')
    s.append(
        '<text x="36" y="238" class="ink" style="font-size: 15px">predicted mean '
        '<tspan class="acc" font-weight="bold">μ</tspan> and scale '
        '<tspan class="acc" font-weight="bold">φ</tspan></text>'
    )
    s.append(spark_line(36, 250, 198, 35, mean, 3.8))
    s.append(
        '<text x="36" y="308" class="small muted">from a model; φ defaults to 1</text>'
    )

    # The instance, and the distribution it models.
    bx, by, bw, bh = 330, 40, 290, 260
    s.append(
        f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="14" class="box"/>'
    )
    s.append(
        f'<text x="{bx + bw / 2}" y="{by + 34}" class="title pastelink code" '
        'text-anchor="middle">Observations</text>'
    )
    s.append(
        f'<text x="{bx + bw / 2}" y="{by + 58}" class="small pastelmuted" text-anchor="middle">'
        'e.g. <tspan class="code">GammaObservations</tspan></text>'
    )
    s.append(gamma_curves(bx + 40, by + 95, bw - 80, 110))
    s.append(
        f'<text x="{bx + bw / 2}" y="{by + bh - 14}" class="small pastelmuted" '
        'text-anchor="middle">distribution p(y | μ, φ)</text>'
    )

    # Inputs into the instance.
    s.append(
        f'<path d="M250 97 C 290 97, 290 140, {bx - 2} 140" class="wire" marker-end="url(#tip)"/>'
    )
    s.append(
        f'<path d="M250 265 C 290 265, 290 200, {bx - 2} 200" class="wire" marker-end="url(#tip)"/>'
    )

    # What it returns.
    ox, ow, oh = 710, 230, 46
    for i, (name, desc) in enumerate(OUTPUTS):
        oy = 25 + i * 60
        yc = oy + oh / 2
        s.append(
            f'<path d="M{bx + bw} 170 C {bx + bw + 45} 170, {ox - 45} {yc}, {ox - 2} {yc}" '
            'class="wire" marker-end="url(#tip)"/>'
        )
        s.append(
            f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" rx="8" class="outcard"/>'
        )
        s.append(
            f'<text x="{ox + 12}" y="{oy + 20}" class="small code outname">{name}</text>'
        )
        s.append(
            f'<text x="{ox + 12}" y="{oy + 38}" class="small pastelmuted">{desc}</text>'
        )

    s.append("</g>")
    s.append("</svg>")
    return "\n".join(s)


for suffix, theme in THEMES.items():
    (ASSETS / f"observation_model_scheme{suffix}.svg").write_text(build(theme))
