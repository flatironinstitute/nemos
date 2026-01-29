"""GLM-HMM parameter definitions and type aliases."""

from typing import Callable, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
from numpy.typing import ArrayLike

from ..glm.params import GLMParams
from ..typing import DESIGN_INPUT_TYPE


class HMMParams(eqx.Module):
    """Parameter container for HMM models."""

    log_initial_prob: jnp.ndarray
    log_transition_prob: jnp.ndarray

    @staticmethod
    def regularizable_subtrees() -> list[Callable[["HMMParams"], jnp.ndarray | dict]]:
        """Filter regularizable subtrees."""
        return []


class GLMScale(eqx.Module):
    """Scale parameter container."""

    log_scale: jnp.ndarray

    @staticmethod
    def regularizable_subtrees() -> list[Callable[["HMMParams"], jnp.ndarray | dict]]:
        """Filter regularizable subtrees."""
        return []


class GLMHMMParams(eqx.Module):
    """Parameter container for GLM-HMM models."""

    glm_params: GLMParams
    glm_scale: GLMScale
    hmm_params: HMMParams

    @staticmethod
    def regularizable_subtrees() -> (
        list[Callable[["GLMHMMParams"], jnp.ndarray | dict]]
    ):
        """Filter regularizable subtrees."""
        return [lambda p: p.glm_params.coef]


# Tuple[coef, intercept, scale, init_proba, transition_proba]
GLMHMMUserParams = Tuple[
    Union[DESIGN_INPUT_TYPE, ArrayLike], ArrayLike, ArrayLike, ArrayLike, ArrayLike
]
# Tuple[init_proba, transition_proba]
HMMUserParams = Tuple[ArrayLike, ArrayLike]
