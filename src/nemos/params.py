"""GLM parameter definitions and type aliases."""

from typing import Callable

import equinox as eqx
import jax.numpy as jnp


class ModelParams(eqx.Module):
    """Shared methods for model parameter containers."""

    @staticmethod
    def regularizable_subtrees() -> list[Callable[["ModelParams"], jnp.ndarray | dict]]:
        """
        Filter regularizable subtrees.

        Replace this function in subclasses to specify which subtrees of the parameter pytree should be regularized.
        """
        return []

    @classmethod
    def initialize_params(cls, *args, **kwargs) -> "ModelParams":
        """Initialize parameters."""
        return cls(*args, **kwargs)
