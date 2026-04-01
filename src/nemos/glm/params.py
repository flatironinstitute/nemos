"""GLM parameter definitions and type aliases."""

from typing import Callable, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
from numpy.typing import ArrayLike

from ..params import ModelParams
from ..typing import DESIGN_INPUT_TYPE


class GLMParams(ModelParams):
    """Parameter container for GLM models."""

    coef: jnp.ndarray | dict
    intercept: jnp.ndarray

    @staticmethod
    def regularizable_subtrees() -> list[Callable[["GLMParams"], jnp.ndarray | dict]]:
        """Filter regularizable subtrees."""
        return [lambda p: p.coef]


class NBGLMParams(eqx.Module):
    """Parameter container for NBGLM models."""

    glm_params: GLMParams
    log_scale: jnp.ndarray


GLMUserParams = Tuple[Union[DESIGN_INPUT_TYPE, ArrayLike], ArrayLike]
NBGLMUserParams = Tuple[Union[DESIGN_INPUT_TYPE, ArrayLike], ArrayLike, float]
