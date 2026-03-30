from typing import Callable, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
from numpy.typing import ArrayLike

from ..typing import DESIGN_INPUT_TYPE


class PPGLMParams(eqx.Module):
    """Paramter container for PP GLM models."""

    coef: jnp.ndarray | dict
    intercept: jnp.ndarray

    @staticmethod
    def regularizable_subtrees() -> list[Callable[["PPGLMParams"], jnp.ndarray | dict]]:
        """Filter regularizable subtrees."""
        return [lambda p: p.coef]


PPGLMUserParams = Tuple[Union[DESIGN_INPUT_TYPE, ArrayLike], ArrayLike]