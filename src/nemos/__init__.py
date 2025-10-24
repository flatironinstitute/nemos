"""Public available modules."""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _get_version

from . import (
    basis,
    convolve,
    exceptions,
    fetch,
    glm,
    identifiability_constraints,
    observation_models,
    pytrees,
    regularizer,
    simulation,
    solvers,
    styles,
    tree_utils,
    type_casting,
    utils,
)
from .io.io import inspect_npz, load_model

__all__ = [
    "basis",
    "convolve",
    "exceptions",
    "fetch",
    "glm",
    "identifiability_constraints",
    "observation_models",
    "pytrees",
    "regularizer",
    "simulation",
    "styles",
    "tree_utils",
    "type_casting",
    "utils",
    "load_model",
    "inspect_npz",
]


def __dir__() -> list[str]:
    return __all__


try:
    __version__ = _get_version("nemos")
except _PackageNotFoundError:
    # package is not installed
    pass
