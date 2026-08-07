"""Check the jupytext/kernelspec metadata of the myst-nb markdown notebooks.

The documentation notebooks are stored as myst-nb markdown. Whenever one of them
is opened and saved from Jupyter Lab, jupytext rewrites the YAML front matter: it
stamps the jupytext version that happened to be installed, and it replaces the
kernel display name with the one of the kernel that was used. Neither field
changes how the notebook is executed or rendered, but both produce diffs that
have nothing to do with the change under review.

The ``[tool.jupytext] notebook_metadata_filter`` setting in ``pyproject.toml``
keeps jupytext from writing the version stamp in the first place, so this script
is the backstop for it: it reports any notebook that still carries one, for
instance a file written before the setting existed or by a tool that did not
read the configuration. The display name cannot be handled that way, since
``kernelspec.display_name`` is required by the nbformat schema and filtering it
out leaves a notebook jupytext refuses to read, so this script is what keeps it
uniform.

With ``--fix`` the offending lines are rewritten in place. The rewrite is
deliberately line based rather than a jupytext round trip: a round trip drops
root level front matter, such as the ``hide:`` key of ``docs/quickstart.md``.
Everything outside the two lines is left byte for byte as it was.
"""

import argparse
import pathlib
import sys

import yaml

# Front matter is delimited by a line containing exactly three dashes.
DELIMITER = "---"

# The display name jupyter lab writes for a plain ipykernel, and the one already
# used by the vast majority of the notebooks.
CANONICAL_DISPLAY_NAME = "Python 3 (ipykernel)"

DOCS_ROOT = pathlib.Path(__file__).resolve().parent.parent / "docs"


def front_matter_end(lines):
    """Return the index of the line closing the front matter, or None if there is none."""
    if not lines or lines[0].strip() != DELIMITER:
        return None
    return next(
        (i for i in range(1, len(lines)) if lines[i].strip() == DELIMITER), None
    )


def front_matter(lines):
    """Return the parsed YAML front matter of a markdown file, or None if it has none."""
    end = front_matter_end(lines)
    return None if end is None else yaml.safe_load("".join(lines[1:end]))


def problems(path):
    """Return the list of metadata problems in ``path``, empty when it is clean."""
    meta = front_matter(path.read_text(encoding="utf-8").splitlines(keepends=True))
    if not isinstance(meta, dict) or "jupytext" not in meta:
        return []  # not a notebook
    found = []
    text_repr = meta["jupytext"].get("text_representation", {})
    if "jupytext_version" in text_repr:
        found.append("carries jupytext_version")
    display_name = meta.get("kernelspec", {}).get("display_name")
    if display_name != CANONICAL_DISPLAY_NAME:
        found.append(f"display_name is {display_name!r}")
    return found


def rewrite(path):
    """Drop the ``jupytext_version`` line and restore the canonical display name."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    end = front_matter_end(lines)
    header = []
    for line in lines[:end]:
        stripped = line.strip()
        if stripped.startswith("jupytext_version:"):
            continue
        if stripped.startswith("display_name:"):
            indent = line[: len(line) - len(line.lstrip())]
            line = f"{indent}display_name: {CANONICAL_DISPLAY_NAME}\n"
        header.append(line)
    if not any(line.strip().startswith("display_name:") for line in header):
        # A notebook saved while `-kernelspec.display_name` was still filtered has no
        # such line to replace, and nbformat requires the key, so put it back.
        kernelspec = next(
            i for i, line in enumerate(header) if line.strip() == "kernelspec:"
        )
        header.insert(kernelspec + 1, f"  display_name: {CANONICAL_DISPLAY_NAME}\n")
    path.write_text("".join(header + lines[end:]), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths",
        nargs="*",
        type=pathlib.Path,
        help="markdown files to inspect, defaulting to every markdown file under docs/",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="rewrite the offending lines instead of only reporting them",
    )
    args = parser.parse_args()

    offenders = []
    for path in args.paths or sorted(DOCS_ROOT.rglob("*.md")):
        found = problems(path)
        if found:
            offenders.append((path, found))

    if not offenders:
        print("INFO: All notebooks have clean metadata.")
        return 0

    if args.fix:
        # Exit non zero so that a pre-commit run stops and the rewritten files are
        # staged again rather than committed in their original state.
        print("INFO: Notebook metadata rewritten, stage the files again.")
        for path, found in offenders:
            rewrite(path)
            print(f"\t- {path}: {', '.join(found)}")
        return 1

    print("WARNING: Notebook metadata is not clean!")
    for path, found in offenders:
        print(f"\t- {path}: {', '.join(found)}")
    print(
        "\nRun `tox -e fix`, or `python scripts/check_notebook_metadata.py --fix`, to "
        "rewrite these lines. Note that re-saving the file through jupytext is not a "
        "substitute: it drops any root level front matter the page relies on, such as "
        "the `hide:` key of docs/quickstart.md."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
