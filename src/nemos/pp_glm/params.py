"""PP-GLM parameter definitions and type aliases."""

from typing import Callable

import jax.numpy as jnp

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
        """Filter regularizable subtrees."""
        return [lambda p: p.params.coef]
