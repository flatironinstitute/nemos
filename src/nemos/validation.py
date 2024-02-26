import warnings
from typing import Any

import jax
import jax.numpy as jnp

from .utils import get_valid_multitree, pytree_map_and_reduce


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


def error_all_invalid(*pytree: Any):
    """
    Raises an error if all sample points across multiple pytrees are invalid.

    This function checks multiple pytrees with NDArrays as leaves to determine if all sample points are invalid.
    A sample point is considered invalid if it contains NaN or infinite values in at least one of the pytrees.
    The sample axis is the first dimension of the NDArray.

    Parameters
    ----------
    pytree :
        Variable number of pytrees to be evaluated. Each pytree is expected to have NDArrays as leaves with a
        consistent size for the first dimension (sample dimension).
        The function checks for the validity of sample points across these pytrees.

    Raises
    ------
    ValueError
        If all sample points across the provided pytrees are invalid (i.e., contain NaN or infinite values).
    """
    if all(~get_valid_multitree(*pytree)):
        raise ValueError("At least a NaN or an Inf at all sample points!")
