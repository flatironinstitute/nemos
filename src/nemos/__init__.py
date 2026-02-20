"""Public available modules."""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _get_version

import lazy_loader as _lazy

# All submodules are lazy-loaded for faster import times
__getattr__, __dir__, __all__ = _lazy.attach(
    __name__,
    submodules=[
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
        "solvers",
        "styles",
        "tree_utils",
        "type_casting",
        "utils",
    ],
    submod_attrs={
        "io.io": ["inspect_npz", "load_model"],
    },
)


try:
    __version__ = _get_version("nemos")
except _PackageNotFoundError:
    # package is not installed
    pass
