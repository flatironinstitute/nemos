import inspect
import re
from pydoc import render_doc
from typing import Type

from ._solver_registry import solver_registry


# TODO: Add a short how-to guide about this.
def get_solver_documentation(solver: str | Type, show_help: bool = False) -> str:
    """
    Get the documentation of a specified solver, including accepted arguments and the docstring of its `__init__`.

    Parameters
    ----------
    solver:
        `solver` can be a string or a type (e.g. `nemos.solvers.JaxoptGradientDescent`).
        If `solver` is a string, the corresponding solver will be read from the solver registry.
    show_help:
        Instead of the docstring, show the full output that be produced by `help(solver)` where `solver` is a type.
    """
    if isinstance(solver, str):
        solver = solver_registry[solver]

    if show_help:
        return render_doc(solver, title="Help on %s")

    solver_class_path = f"{solver.__module__}.{solver.__name__}"
    intro = inspect.cleandoc(
        f"""
        Showing docstring of {solver_class_path}.
        For potentially more info, use `show_help=True`.
        """
    )
    solver_doc = inspect.getdoc(solver) or f"No documentation found for {solver}."
    doc = intro + "\n\n" + solver_doc

    # if it doesn't already have it, then expand the docs
    # with the __init__'s docstring'
    pattern = r"More info from (.+)\.__init__"
    if re.search(pattern, doc) is None:
        solver_init_doc_header = inspect.cleandoc(
            f"More info from {solver.__name__}.__init__"
        )
        solver_init_doc_header += "\n" + "-" * len(solver_init_doc_header)
        solver_init_doc = inspect.cleandoc(
            inspect.getdoc(solver.__init__) or "No __init__ documentation found."
        )
        solver_init_doc = solver_init_doc_header + "\n" + solver_init_doc

        doc += "\n\n" + solver_init_doc

    return doc
