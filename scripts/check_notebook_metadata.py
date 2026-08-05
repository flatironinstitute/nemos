"""Check the jupytext/kernelspec metadata of the myst-nb markdown notebooks.

The documentation notebooks are stored as myst-nb markdown. Whenever one of them
is opened and saved from Jupyter Lab, jupytext rewrites the YAML front matter: it
stamps the jupytext version that happened to be installed, and it replaces the
kernel display name with the one of the kernel that was used. Neither field
changes how the notebook is executed or rendered, but both produce diffs that
have nothing to do with the change under review.

The ``[tool.jupytext] notebook_metadata_filter`` setting in ``pyproject.toml``
keeps jupytext from writing those fields in the first place. This script is the
backstop for that setting: it reports any notebook that still carries them, for
instance one written before the setting existed or by a tool that did not read
the configuration. It only reports, it never rewrites, so the fix is to let
jupytext round-trip the file or to delete the offending lines by hand.
"""

import pathlib
import sys

import yaml

# Front matter is delimited by a line containing exactly three dashes.
DELIMITER = "---"

# The display name jupyter lab writes for a plain ipykernel, and the one already
# used by the vast majority of the notebooks.
CANONICAL_DISPLAY_NAME = "Python 3 (ipykernel)"

DOCS_ROOT = pathlib.Path(__file__).resolve().parent.parent / "docs"


def front_matter(text):
    """Return the parsed YAML front matter of a markdown file, or None if it has none."""
    lines = text.splitlines()
    if not lines or lines[0] != DELIMITER:
        return None
    for index, line in enumerate(lines[1:], start=1):
        if line == DELIMITER:
            return yaml.safe_load("\n".join(lines[1:index]))
    return None


def problems(path):
    """Return the list of metadata problems in ``path``, empty when it is clean."""
    meta = front_matter(path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or "jupytext" not in meta:
        return []  # not a notebook
    found = []
    text_repr = meta["jupytext"].get("text_representation", {})
    if "jupytext_version" in text_repr:
        found.append("carries jupytext_version")
    display_name = meta.get("kernelspec", {}).get("display_name")
    if display_name not in (None, CANONICAL_DISPLAY_NAME):
        found.append(f"display_name is {display_name!r}")
    return found


def main():
    offenders = []
    for path in sorted(DOCS_ROOT.rglob("*.md")):
        found = problems(path)
        if found:
            offenders.append((path, found))

    if not offenders:
        print("INFO: All notebooks have clean metadata.")
        return 0

    print("WARNING: Notebook metadata is not clean!")
    for path, found in offenders:
        print(f"\t- {path}: {', '.join(found)}")
    print(
        "\nThese fields are filtered by `[tool.jupytext] notebook_metadata_filter` in "
        "pyproject.toml. Re-saving the notebook through jupytext, or removing the lines "
        "by hand, will clear this."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
