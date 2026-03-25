"""GLM-HMM parameter definitions and type aliases."""

from typing import Callable, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
from numpy.typing import ArrayLike

from ..glm.params import GLMParams
from ..hmm.params import HMMParams
from ..params import ModelParams
from ..typing import DESIGN_INPUT_TYPE


class GLMScale(ModelParams):
    """Scale parameter container."""

    log_scale: jnp.ndarray


class GLMHMMParams(ModelParams):
    """Parameter container for GLM-HMM models."""

    model_params: Tuple[GLMParams, GLMScale]
    # glm_scale: GLMScale
    # scale: jnp.ndarray | None = None
    hmm_params: HMMParams

    @staticmethod
    def regularizable_subtrees() -> (
        list[Callable[["GLMHMMParams"], jnp.ndarray | dict]]
    ):
        """Filter regularizable subtrees."""
        return [lambda p: p.model_params.coef]


# Tuple[coef, intercept, scale, init_proba, transition_proba]
GLMHMMUserParams = Tuple[
    Union[DESIGN_INPUT_TYPE, ArrayLike], ArrayLike, ArrayLike, ArrayLike, ArrayLike
]
# Tuple[init_proba, transition_proba]
HMMUserParams = Tuple[ArrayLike, ArrayLike]
