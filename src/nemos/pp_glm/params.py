from typing import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ..glm.params import GLMParams
from ..params import ModelParams


class PPGLMParamsWithKey(ModelParams):
    """Wrapper around PPGLMParams that contains a PRNG key."""

    params: GLMParams
    random_key: jnp.ndarray

    @staticmethod
    def regularizable_subtrees() -> (
        list[Callable[["PPGLMParamsWithKey"], jnp.ndarray | dict]]
    ):
        return [lambda p: p.params.coef]


class X_ppglm(eqx.Module):
    """Preprocessed predictors for PP-GLM."""

    times: Float[Array, "n_events"]
    ids: Int[Array, "n_events"]


class y_ppglm(eqx.Module):
    """Preprocessed spikes for PP-GLM."""

    times: Float[Array, "n_spikes"]
    ids: Int[Array, "n_spikes"]
    idx: Int[Array, "n_spikes"]


class mc_sample_ppglm(eqx.Module):
    times: Float[Array, "n_samples"]
    idx: Int[Array, "n_samples"]
