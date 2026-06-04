"""Validation classes for PPGLM and PopulationPPGLM models."""

import jax.numpy as jnp

from ..glm.params import GLMParams
from .params import PPGLMParamsWithKey


def to_pp_glm_params_with_key(params: GLMParams, random_key: jnp.array):
    """Map from PPGLMParams to PPGLMParamsWithKey by appending a jax random key.
    The key is converted from uint32 to float to avoid solver initialization error"""
    return PPGLMParamsWithKey(params, random_key.astype(float))
