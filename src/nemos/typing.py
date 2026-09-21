"""Collection of nemos typing."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    NamedTuple,
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
)

import jax.numpy as jnp
from jax.typing import ArrayLike
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pynapple as nap

    from .base_validator import RegressorValidator

Pytree: TypeAlias = Any
Params: TypeAlias = Pytree
Aux = TypeVar("Aux")
SolverState = TypeVar("SolverState")
StepResult: TypeAlias = Tuple[Params, SolverState, Aux]
DESIGN_INPUT_TYPE: TypeAlias = "Union[jnp.ndarray, Pytree, nap.TsdFrame]"

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

LogLikelihoodFn = Callable[
    [
        Params,  # model parameters, excluding the HMM ones
        jnp.ndarray,  # Predictors (i.e. model design for GLM)
        jnp.ndarray,
    ],  # Output (neural activity)
    jnp.ndarray,
]  # Elementwise log-likelihood, shape (n_time_bins, n_states)

EStepOutput = Tuple[
    jnp.ndarray,  # log_posteriors, (n_time_bins, n_states)
    jnp.ndarray,  # log_joint_posterior, summed over time, (n_states, n_states)
    jnp.ndarray,  # log_likelihood of the observations, scalar
    jnp.ndarray,  # likelihood_norm, the per-sample normalized likelihood, scalar
    jnp.ndarray,  # log_alphas, (n_time_bins, n_states)
    jnp.ndarray,  # log_betas, (n_time_bins, n_states)
]

EStepFn = Callable[
    [
        Params,  # HMM and model parameters
        jnp.ndarray,  # Predictors (i.e. model design for GLM)
        jnp.ndarray,  # Output (neural activity)
        LogLikelihoodFn,  # static, closes over the observation model
        jnp.ndarray,
    ],  # Boolean session starts
    EStepOutput,
]

FeatureMatrix: TypeAlias = "nap.TsdFrame | NDArray | jnp.ndarray"

# A concrete (non-pynapple) array, either NumPy or JAX.
Array: TypeAlias = "Union[NDArray, jnp.ndarray]"

# User provided init_params (e.g. for GLMs Tuple[array, array])
UserProvidedParamsT = TypeVar("UserProvidedParamsT")
# Model internal representation (e.g. for GLMs nemos.glm.glm.GLMParams)
ModelParamsT = TypeVar("ModelParamsT")
# Validator type associated with a regressor (e.g. GLMValidator for GLM)
ValidatorT = TypeVar("ValidatorT", bound="RegressorValidator")
