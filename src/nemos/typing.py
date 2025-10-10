"""Collection of nemos typing."""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Tuple, TypeAlias, TypeVar, Union

import jax.numpy as jnp
import pynapple as nap
from jax.typing import ArrayLike
from numpy.typing import NDArray

from .pytrees import FeaturePytree

Pytree: TypeAlias = Any
Params: TypeAlias = Pytree
SolverState = TypeVar("SolverState")
StepResult: TypeAlias = Tuple[Params, SolverState]
DESIGN_INPUT_TYPE = Union[jnp.ndarray, FeaturePytree, nap.TsdFrame]

# copying jax.random's annotation
KeyArrayLike = ArrayLike

SolverRun = Callable[
    [
        Params,  # parameters, could be any pytree
        jnp.ndarray,  # Predictors (i.e. model design for GLM)
        jnp.ndarray,
    ],  # Output (neural activity)
    StepResult,
]

SolverInit = Callable[
    [
        Params,  # parameters, could be any pytree
        jnp.ndarray,  # Predictors (i.e. model design for GLM)
        jnp.ndarray,
    ],  # Output (neural activity)
    SolverState,
]

SolverUpdate = Callable[
    [
        Params,  # parameters, could be any pytree
        NamedTuple,
        jnp.ndarray,  # Predictors (i.e. model design for GLM)
        jnp.ndarray,
    ],  # Output (neural activity)
    StepResult,
]

ProximalOperator = Callable[
    [
        Params,  # parameters, could be any pytree
        float,  # Regularizer strength (for now float, eventually pytree)
        float,
    ],  # Step-size for optimization (must be a float)
    Tuple[jnp.ndarray, jnp.ndarray],
]

FeatureMatrix = nap.TsdFrame | NDArray

RegularizerStrength = float | Tuple[float, float]
