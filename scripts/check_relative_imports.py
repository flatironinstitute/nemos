"""Check that the package never imports itself absolutely.

Inside ``src/nemos`` every intra-package import must be relative (``from .x import y``,
``from ..x import y``). An absolute self-import hard-codes the distribution name, so it
breaks under vendoring, on rename, and when a subpackage moves, while the relative form
keeps working. IDEs that treat ``src`` as a sources root emit the absolute form by
default, so it drifts back in unless checked.

Neither of the linters already in the dev dependencies can express this: ruff's
``TID251`` matches the *resolved* module name, so it flags relative and absolute imports
alike, ``TID252`` bans the opposite direction, and bare flake8 has no such check. Hence
this script, which needs no extra dependency.

Parsing is AST-based, so the ``>>> import nemos as nmo`` lines in docstrings are never
seen: only real import statements are visited.
"""

import ast
import pathlib
from typing import List, NamedTuple

PACKAGE = "nemos"

# Documentation helpers are demo code, where ``import nemos as nmo`` is the idiomatic
# form users are meant to copy. Excluded for the same reason ruff excludes it.
EXCLUDED_DIRS = ("_documentation_utils",)


class Violation(NamedTuple):
    """An absolute intra-package import and the relative form replacing it."""

    file: pathlib.Path
    lineno: int
    found: str
    expected: str


def _is_self_import(module: str) -> bool:
    """Whether a module path refers to the package itself."""
    return module == PACKAGE or module.startswith(f"{PACKAGE}.")


class SelfImportVisitor(ast.NodeVisitor):
    """Collect absolute imports of the package from a module inside the package."""

    def __init__(self, file: pathlib.Path, root: pathlib.Path):
        self.file = file
        # depth inside the package: nemos/glm/glm.py -> 1, so the prefix is ".."
        self.depth = len(file.relative_to(root).parts) - 1
        self.violations: List[Violation] = []

    def _relative_form(self, module: str) -> str:
        """Spell the relative import that should replace an absolute one."""
        target = module[len(PACKAGE) :].lstrip(".")
        return f"from {'.' * (self.depth + 1)}{target} import ..."

    def visit_Import(self, node: ast.Import) -> None:
        """Flag ``import nemos...``."""
        for alias in node.names:
            if _is_self_import(alias.name):
                self.violations.append(
                    Violation(
                        self.file,
                        node.lineno,
                        f"import {alias.name}",
                        self._relative_form(alias.name),
                    )
                )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Flag ``from nemos... import ...``; level 0 means absolute."""
        if node.level == 0 and node.module and _is_self_import(node.module):
            self.violations.append(
                Violation(
                    self.file,
                    node.lineno,
                    f"from {node.module} import ...",
                    self._relative_form(node.module),
                )
            )


def find_absolute_self_imports(root: pathlib.Path) -> List[Violation]:
    """Walk the package and collect every absolute self-import."""
    violations: List[Violation] = []
    for file in sorted(root.rglob("*.py*")):
        if file.suffix not in (".py", ".pyi"):
            continue
        if any(part in EXCLUDED_DIRS for part in file.parts):
            continue
        visitor = SelfImportVisitor(file, root)
        visitor.visit(ast.parse(file.read_text(), filename=str(file)))
        violations.extend(visitor.violations)
    return violations


if __name__ == "__main__":
    import argparse
    import logging
    import sys

    default_path = pathlib.Path(__file__).parent.parent / "src" / "nemos"

    parser = argparse.ArgumentParser(
        description="Check that intra-package imports are relative, using AST."
    )
    parser.add_argument(
        "--path",
        "-p",
        type=pathlib.Path,
        help="Root path to the package (source folder).",
        default=default_path,
    )
    args = parser.parse_args()

    logger = logging.getLogger("check_relative_imports")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    violations = find_absolute_self_imports(args.path)

    if violations:
        msg_lines = ["Absolute intra-package imports found, use relative imports:\n"]
        for violation in violations:
            msg_lines.append(f"\t{violation.file}:{violation.lineno}\n")
            msg_lines.append(f"\t\tfound:    {violation.found}\n")
            msg_lines.append(f"\t\texpected: {violation.expected}\n")
        logger.warning("".join(msg_lines))
        sys.exit(1)
    else:
        logger.info("No absolute intra-package imports found.")
