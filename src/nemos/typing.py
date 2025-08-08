"""Collection of nemos typing."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Tuple, Union, TypeVar, TypeAlias

import jax.numpy as jnp
import pynapple as nap
from jax.typing import ArrayLike
from numpy.typing import NDArray
from .pytrees import FeaturePytree


Pytree: TypeAlias = Any
Params: TypeAlias = Pytree
SolverState = TypeVar("SolverState")
StepResult = TypeVar("StepResult")

DESIGN_INPUT_TYPE = Union[jnp.ndarray, FeaturePytree]

# copying jax.random's annotation
KeyArrayLike = ArrayLike

# TODO: Update the argument types of these methods
SolverRun = Callable[
    [
        Any,  # parameters, could be any pytree
        jnp.ndarray,  # Predictors (i.e. model design for GLM)
        jnp.ndarray,
    ],  # Output (neural activity)
    StepResult,
]

SolverInit = Callable[
    [
        Any,  # parameters, could be any pytree
        jnp.ndarray,  # Predictors (i.e. model design for GLM)
        jnp.ndarray,
    ],  # Output (neural activity)
    SolverState,
]

SolverUpdate = Callable[
    [
        Any,  # parameters, could be any pytree
        NamedTuple,
        jnp.ndarray,  # Predictors (i.e. model design for GLM)
        jnp.ndarray,
    ],  # Output (neural activity)
    StepResult,
]

ProximalOperator = Callable[
    [
        Any,  # parameters, could be any pytree
        float,  # Regularizer strength (for now float, eventually pytree)
        float,
    ],  # Step-size for optimization (must be a float)
    Tuple[jnp.ndarray, jnp.ndarray],
]

FeatureMatrix = nap.TsdFrame | NDArray

RegularizerStrength = float | Tuple[float, float]
