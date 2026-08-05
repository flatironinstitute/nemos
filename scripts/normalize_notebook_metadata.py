"""Normalize the jupytext/kernelspec metadata of the myst-nb markdown notebooks.

The documentation notebooks are stored as myst-nb markdown. Whenever one of them
is opened and saved from Jupyter Lab, jupytext rewrites the YAML front matter:
it stamps the jupytext version that happened to be installed, and it replaces the
kernel display name with the one of the kernel that was used. Neither field
changes how the notebook is executed or rendered, but both produce diffs that
have nothing to do with the change under review.

This script rewrites the front matter to a canonical form:

* ``jupytext.text_representation.jupytext_version`` is dropped, since it only
  records the writer's local install.
* ``kernelspec.display_name`` is pinned to ``DEFAULT_DISPLAY_NAME``, so that a
  local environment name never leaks into the repository.

Run it with ``--check`` to report the files that are not normalized instead of
rewriting them, which is what the CI check does.
"""

import pathlib
import re
from typing import List, Tuple

# The display name jupyter lab writes for a plain ipykernel, and the one already
# used by the vast majority of the notebooks.
DEFAULT_DISPLAY_NAME = "Python 3 (ipykernel)"

# Front matter is delimited by a line containing exactly three dashes.
FRONT_MATTER_DELIMITER = "---"

JUPYTEXT_VERSION_PATTERN = re.compile(r"^\s*jupytext_version:\s*\S+\s*$")
DISPLAY_NAME_PATTERN = re.compile(r"^(?P<indent>\s*)display_name:\s*\S.*$")


def split_front_matter(lines: List[str]) -> Tuple[int, int]:
    """
    Locate the YAML front matter of a markdown file.

    Parameters
    ----------
    lines :
        The lines of the file, newline characters included.

    Returns
    -------
    :
        The start and end index of the front matter body, i.e. the lines between
        the two delimiters. ``(0, 0)`` is returned when the file has no front
        matter.
    """
    if not lines or lines[0].rstrip("\n") != FRONT_MATTER_DELIMITER:
        return 0, 0

    for index, line in enumerate(lines[1:], start=1):
        if line.rstrip("\n") == FRONT_MATTER_DELIMITER:
            return 1, index

    # unterminated front matter, leave the file alone
    return 0, 0


def normalize_front_matter(lines: List[str]) -> List[str]:
    """
    Return the file lines with a canonical jupytext front matter.

    Parameters
    ----------
    lines :
        The lines of the file, newline characters included.

    Returns
    -------
    :
        The normalized lines. Files without a jupytext front matter are returned
        unchanged.
    """
    start, end = split_front_matter(lines)
    if start == end:
        return lines

    front_matter = lines[start:end]
    if not any(line.startswith("jupytext:") for line in front_matter):
        return lines

    normalized = []
    for line in front_matter:
        if JUPYTEXT_VERSION_PATTERN.match(line):
            continue
        match = DISPLAY_NAME_PATTERN.match(line)
        if match:
            line = f"{match.group('indent')}display_name: {DEFAULT_DISPLAY_NAME}\n"
        normalized.append(line)

    return lines[:start] + normalized + lines[end:]


def collect_markdown_files(path: pathlib.Path) -> List[pathlib.Path]:
    """
    Collect every markdown file under ``path``.

    Parameters
    ----------
    path :
        A markdown file or a directory to search recursively.

    Returns
    -------
    :
        The sorted list of markdown files.
    """
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.md"))


def normalize_file(path: pathlib.Path, write: bool) -> bool:
    """
    Normalize a single markdown file.

    Parameters
    ----------
    path :
        The file to normalize.
    write :
        If True, the normalized content is written back to disk.

    Returns
    -------
    :
        True if the file was not already normalized.
    """
    with open(path, "r", encoding="utf-8", newline="") as fh:
        lines = fh.readlines()

    normalized = normalize_front_matter(lines)
    if normalized == lines:
        return False

    if write:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.writelines(normalized)

    return True


if __name__ == "__main__":
    import argparse
    import logging
    import sys

    default_path = pathlib.Path(__file__).parent.parent / "docs"

    parser = argparse.ArgumentParser(
        description="Normalize the front matter of the myst-nb markdown notebooks."
    )
    parser.add_argument(
        "--path",
        "-p",
        type=pathlib.Path,
        help="Markdown file or directory to normalize.",
        default=default_path,
    )
    parser.add_argument(
        "--check",
        "-c",
        action="store_true",
        help="Report the files that need normalizing instead of rewriting them.",
    )
    args = parser.parse_args()

    logger = logging.getLogger("normalize_notebook_metadata")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    changed = [
        path
        for path in collect_markdown_files(args.path)
        if normalize_file(path, write=not args.check)
    ]

    if changed and args.check:
        msg_lines = ["Notebook metadata is not normalized!\n"]
        for path in changed:
            msg_lines.append(f"\t- {path}\n")
        msg_lines.append(
            "\nRun `python scripts/normalize_notebook_metadata.py` to fix them.\n"
        )
        logger.warning("".join(msg_lines))
        sys.exit(1)
    elif changed:
        logger.info("Normalized %d notebook(s).", len(changed))
    else:
        logger.info("All notebooks are normalized.")
