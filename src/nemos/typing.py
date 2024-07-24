"""Collection of nemos typing."""

from typing import Any, Callable, NamedTuple, Tuple, Union

import jax.numpy as jnp
import jaxopt

from .pytrees import FeaturePytree

DESIGN_INPUT_TYPE = Union[jnp.ndarray, FeaturePytree]

SolverRun = Callable[
    [
        Any,  # parameters, could be any pytree
        jnp.ndarray,  # Predictors (i.e. model design for GLM)
        jnp.ndarray,
    ],  # Output (neural activity)
    jaxopt.OptStep,
]

SolverInit = Callable[
    [
        Any,  # parameters, could be any pytree
        jnp.ndarray,  # Predictors (i.e. model design for GLM)
        jnp.ndarray,
    ],  # Output (neural activity)
    NamedTuple,
]

SolverUpdate = Callable[
    [
        Any,  # parameters, could be any pytree
        NamedTuple,
        jnp.ndarray,  # Predictors (i.e. model design for GLM)
        jnp.ndarray,
    ],  # Output (neural activity)
    jaxopt.OptStep,
]

ProximalOperator = Callable[
    [
        Any,  # parameters, could be any pytree
        float,  # Regularizer strength (for now float, eventually pytree)
        float,
    ],  # Step-size for optimization (must be a float)
    Tuple[jnp.ndarray, jnp.ndarray],
]