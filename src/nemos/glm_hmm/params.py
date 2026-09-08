"""GLM-HMM parameter definitions and type aliases."""

from typing import Callable, Tuple, Union

import jax.numpy as jnp
from numpy.typing import ArrayLike

from ..glm.params import GLMParams
from ..hmm.params import HMMParams
from ..params import ModelParams
from ..typing import DESIGN_INPUT_TYPE


class GLMHMMModelParams(GLMParams):
    """Add scale to GLMParams, which is specific for the GLM-HMM."""

    log_scale: jnp.ndarray | None = None


class GLMHMMParams(ModelParams):
    """Parameter container for GLM-HMM models."""

    model_params: GLMHMMModelParams
    hmm_params: HMMParams

    @staticmethod
    def regularizable_subtrees() -> (
        list[Callable[["GLMHMMParams"], jnp.ndarray | dict]]
    ):
        """Filter regularizable subtrees."""
        return [lambda p: p.model_params.coef]

    @staticmethod
    def solver_param_subtree() -> Callable[["GLMHMMParams"], GLMHMMModelParams]:
        """Accessor for the sub-pytree the numerical solver and regularizer operate on.

        GLM-HMM is a composite model: its EM M-step optimizes only the GLM-level
        params (coef, intercept, scale). The regularizer and its GroupLasso mask are
        interpreted at this level. Flat models have no such method — consumers must
        treat its absence as the identity accessor.
        """
        return lambda p: p.model_params


# Tuple[coef, intercept, scale, init_proba, transition_proba]
GLMHMMUserParams = Tuple[
    Union[DESIGN_INPUT_TYPE, ArrayLike], ArrayLike, ArrayLike, ArrayLike, ArrayLike
]
