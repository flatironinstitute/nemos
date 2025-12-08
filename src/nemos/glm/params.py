from typing import Tuple, Union

import equinox as eqx
import jax.numpy as jnp
from numpy.typing import ArrayLike

from ..typing import DESIGN_INPUT_TYPE


class GLMParams(eqx.Module):
    """Parameter container for GLM models."""

    coef: jnp.ndarray | dict
    intercept: jnp.ndarray


GLMUserParams = Tuple[Union[DESIGN_INPUT_TYPE, ArrayLike], ArrayLike]
