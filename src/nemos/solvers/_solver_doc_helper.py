from typing import Type
import inspect
import re

from ._solver_registry import solver_registry


# TODO: Add a short how-to guide about this.
# TODO: Add this to the API reference
def get_solver_documentation(solver: str | Type) -> str:
    """
    Get the documentation of a specified solver, including the docstring of its `__init__`.

    `solver` can be a string or a type (e.g. `nemos.solvers.JaxoptGradientDescent`).
    If `solver` is a string, the corresponding solver will be read from the solver registry.
    """
    if isinstance(solver, str):
        solver = solver_registry[solver]

    doc = inspect.getdoc(solver) or f"No documentation found for {solver}."

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
