"""GLM-HMM parameter definitions and type aliases."""

from typing import Callable, Tuple

import equinox as eqx
import jax.numpy as jnp
from numpy.typing import ArrayLike


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


# Tuple[init_proba, transition_proba]
HMMUserParams = Tuple[ArrayLike, ArrayLike]
