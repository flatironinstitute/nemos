"""GLM parameter definitions and type aliases."""

from typing import Callable, Tuple, Union

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


class GLMScaleParams(GLMParams):
    """Add scale to GLMParams, specific for the GLM-HMM and NBGLM models."""

    log_scale: jnp.ndarray | None = None


GLMUserParams = Tuple[Union[DESIGN_INPUT_TYPE, ArrayLike], ArrayLike]
GLMScaleUserParams = Tuple[Union[DESIGN_INPUT_TYPE, ArrayLike], ArrayLike, ArrayLike]
