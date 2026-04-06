"""Validation classes for PPGLM and PopulationPPGLM models."""

import jax.numpy as jnp

from .params import PPGLMParams, PPGLMParamsWithKey, PPGLMUserParams


def to_pp_glm_params_with_key(params: PPGLMParams, random_key: jnp.array):
    """Map from PPGLMParams to PPGLMParamsWithKey by appending a jax random key.
    The key is converted from uint32 to float to avoid solver initialization error"""
    return PPGLMParamsWithKey(params, random_key.astype(float))


def to_pp_glm_params(user_params: PPGLMUserParams):
    """Map from PPGLMUserParams to PPGLMParams"""
    return PPGLMParams(*user_params)


def from_pp_glm_params(params: PPGLMParams) -> PPGLMUserParams:
    """Map from PPGLMParams to PPGLMUserParams."""
    return params.coef, params.intercept
