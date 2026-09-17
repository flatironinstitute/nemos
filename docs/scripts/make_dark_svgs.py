"""Derive the dark-theme variant of the landing page diagrams.

An embedded figure cannot see which theme the page is using, so the light and
dark variants are two files, shown by the ``only-light``/``only-dark`` classes.
This keeps the dark one reproducible instead of hand-maintained: run

    python docs/scripts/make_dark_svgs.py

after editing any of the light diagrams listed in ``DIAGRAMS``.

Only near-achromatic colours are touched. The ink, the greys and the panel fills
have their lightness flipped so they read against a dark page, while the orange,
green and blue of the palette are left exactly as they are.

Labels sitting on a pastel fill are the exception: the fill stays light in the
dark variant, so flipping their ink would leave white type on pale orange. Those
keep the colour they had.
"""

import pathlib
import re

ASSETS = pathlib.Path(__file__).resolve().parent.parent / "assets"

DIAGRAMS = ("glm_hmm_graphical_model.svg", "lnp_model_colscheme.svg")

# Largest spread between channels for a colour to count as a grey. The inks here
# are slightly tinted (#231f20, #1c2123), so an exact r == g == b test misses them.
CHROMA_TOLERANCE = 12

HEX = re.compile(r"#([0-9a-fA-F]{6})\b")
CIRCLE = re.compile(r"<circle\b[^>]*>")
TEXT = re.compile(r"<text\b[^>]*>")
ATTR = re.compile(r'(\w[\w-]*)="([^"]*)"')


def _channels(value):
    return [int(value[i : i + 2], 16) for i in (0, 2, 4)]


def _is_grey(value):
    channels = _channels(value)
    return max(channels) - min(channels) <= CHROMA_TOLERANCE


def _invert_if_grey(match):
    if not _is_grey(match.group(1)):
        return match.group(0)
    return "#" + "".join(f"{255 - v:02x}" for v in _channels(match.group(1)))


def _pastel_discs(svg):
    """Centre and radius of every circle carrying a light, saturated fill."""
    discs = []
    for tag in CIRCLE.findall(svg):
        attrs = dict(ATTR.findall(tag))
        fill = attrs.get("fill", "")
        if not HEX.fullmatch(fill) or _is_grey(fill[1:]):
            continue
        if min(_channels(fill[1:])) < 128:  # a saturated fill, not a pale one
            continue
        discs.append((float(attrs["cx"]), float(attrs["cy"]), float(attrs["r"])))
    return discs


def _restore_labels_on_pastel(light, dark):
    """Put back the original ink of the <text> anchored inside a pastel disc."""
    discs = _pastel_discs(light)
    if not discs:
        return dark

    def restore(match):
        attrs = dict(ATTR.findall(match.group(0)))
        if "x" not in attrs or "y" not in attrs or "fill" not in attrs:
            return match.group(0)
        x, y = float(attrs["x"]), float(attrs["y"])
        # The anchor sits on the text baseline, below the centre of the disc.
        if any((x - cx) ** 2 + (y - cy) ** 2 <= (r + 8) ** 2 for cx, cy, r in discs):
            return HEX.sub(
                lambda m: (
                    "#" + "".join(f"{255 - v:02x}" for v in _channels(m.group(1)))
                ),
                match.group(0),
            )
        return match.group(0)

    return TEXT.sub(restore, dark)


def main():
    for name in DIAGRAMS:
        source = ASSETS / name
        light = source.read_text()
        dark = HEX.sub(_invert_if_grey, light)
        dark = _restore_labels_on_pastel(light, dark)
        target = source.with_name(f"{source.stem}_dark.svg")
        target.write_text(dark)
        print(f"{source.name} -> {target.name}")


if __name__ == "__main__":
    main()
