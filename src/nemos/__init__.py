#!/usr/bin/env python3
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
    styles,
    tree_utils,
    type_casting,
    utils,
)

try:
    __version__ = _get_version("nemos")
except _PackageNotFoundError:
    # package is not installed
    pass
