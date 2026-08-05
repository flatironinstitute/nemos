"""Tests for the myst-nb markdown front matter normalizer."""

import importlib.util
import pathlib
import subprocess
import sys

SCRIPT_PATH = (
    pathlib.Path(__file__).parent.parent / "scripts" / "normalize_notebook_metadata.py"
)


def _load_script():
    """Import the normalizer, which lives outside the installed package."""
    spec = importlib.util.spec_from_file_location(
        "normalize_notebook_metadata", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NOTEBOOK = """---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: nemos (3.12.11)
  language: python
  name: python3
---

# A notebook
"""

NORMALIZED_NOTEBOOK = """---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# A notebook
"""

PLAIN_MARKDOWN = """---
title: not a notebook
---

# A page
"""


def test_normalize_front_matter():
    """The jupytext version is dropped and the display name is pinned."""
    module = _load_script()
    normalized = module.normalize_front_matter(NOTEBOOK.splitlines(keepends=True))
    assert "".join(normalized) == NORMALIZED_NOTEBOOK


def test_normalize_front_matter_is_idempotent():
    """Normalizing an already normalized notebook is a no-op."""
    module = _load_script()
    lines = NORMALIZED_NOTEBOOK.splitlines(keepends=True)
    assert module.normalize_front_matter(lines) == lines


def test_normalize_front_matter_ignores_plain_markdown():
    """Markdown without a jupytext front matter is left alone."""
    module = _load_script()
    lines = PLAIN_MARKDOWN.splitlines(keepends=True)
    assert module.normalize_front_matter(lines) == lines


def test_docs_notebooks_are_normalized():
    """Every notebook shipped in docs/ is already normalized."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
