import warnings
from typing import Any

import jax
import jax.numpy as jnp

from .utils import pytree_map_and_reduce


def warn_invalid_entry(*pytree: Any):
    """
    Warns if any entry in the provided pytrees contains NaN or Infinite (Inf) values.

    Parameters
    ----------
    *pytree :
        Variable number of pytrees to check for invalid entries. A pytree is a nested structure of lists, tuples,
        dictionaries, or other containers, with leaves that are arrays.

    """
    any_infs = pytree_map_and_reduce(jnp.any, any, jax.tree_map(jnp.isinf, pytree))
    any_nans = pytree_map_and_reduce(jnp.any, any, jax.tree_map(jnp.isnan, pytree))
    if any_infs and any_nans:
        warnings.warn(
            message="The provided trees contain Infs and Nans!", category=UserWarning
        )
    elif any_infs:
        warnings.warn(message="The provided trees contain Infs!", category=UserWarning)
    elif any_nans:
        warnings.warn(message="The provided trees contain Nans!", category=UserWarning)


def error_invalid_entry(*pytree: Any):
    """
    Raises an error if any entry in the provided pytrees contains NaN or Infinite (Inf) values.

    Parameters
    ----------
    *pytree : Any
        Variable number of pytrees to be checked for invalid entries. A pytree is defined as a nested structure
        of lists, tuples, dictionaries, or other containers, with leaves that are arrays.

    Raises
    ------
    ValueError
        If any NaN or Inf values are found in the provided pytrees.
    """
    any_infs = pytree_map_and_reduce(jnp.any, any, jax.tree_map(jnp.isinf, pytree))
    any_nans = pytree_map_and_reduce(jnp.any, any, jax.tree_map(jnp.isnan, pytree))
    if any_infs and any_nans:
        raise ValueError("The provided trees contain Infs and Nans!")
    elif any_infs:
        raise ValueError("The provided trees contain Infs!")
    elif any_nans:
        raise ValueError("The provided trees contain Nans!")
