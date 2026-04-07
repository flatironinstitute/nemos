"""GLM-HMM parameter definitions and type aliases."""

from typing import Callable, Tuple, Union

import jax.numpy as jnp
from numpy.typing import ArrayLike

from ..glm.params import GLMScaleParams
from ..hmm.params import HMMParams
from ..params import ModelParams
from ..typing import DESIGN_INPUT_TYPE


class GLMHMMParams(ModelParams):
    """Parameter container for GLM-HMM models."""

    model_params: GLMScaleParams
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
