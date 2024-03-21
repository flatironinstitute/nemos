"""Collection of methods utilities."""

import warnings
from typing import Tuple, Any, Union

from numpy.typing import NDArray
import jax
import jax.numpy as jnp

from .tree_utils import get_valid_multitree, pytree_map_and_reduce
from .pytrees import FeaturePytree


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
    Raise an error if any entry in the provided pytrees contains NaN or Infinite (Inf) values.

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
    Raise an error if all sample points across multiple pytrees are invalid.

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


def check_length(x: Any, err_message: str):
    if not hasattr(x, "__len__") or len(x) != 2:
        raise ValueError(err_message)


def convert_tree_leaves_to_jax_array(tree: Any, data_type: jnp.dtype):
    try:
        tree = jax.tree_map(lambda x: jnp.asarray(x, dtype=data_type), tree)
    except (ValueError, TypeError):
        raise TypeError(
            "Initial parameters must be array-like objects (or pytrees of array-like objects) "
            "with numeric data-type!"
        )
    return tree


def check_tree_leaves_dimensionality(tree: Any, expected_dim: int, err_message: str):
    if pytree_map_and_reduce(lambda x: x.ndim != expected_dim, any, tree):
        raise ValueError(err_message)


def check_same_shape_on_axis(*arrays: NDArray, axis: int = 0, err_message: str):
    if len(arrays) > 1:
        if any(arr.shape[axis] != arrays[0].shape[axis] for arr in arrays[1:]):
            raise ValueError(err_message)


def check_array_shape_match_tree(
        tree: Any,
        array: NDArray,
        axis: int,
        err_message: str
):
    if pytree_map_and_reduce(lambda arr: arr.shape[axis] != array.shape[axis], any, tree):
        raise ValueError(err_message)


def array_axis_consistency(
    array_1: Union[FeaturePytree, jnp.ndarray, NDArray],
    array_2: Union[FeaturePytree, jnp.ndarray, NDArray],
    axis_1: int,
    axis_2: int,
):
    if array_1.shape[axis_1] != array_2.shape[axis_2]:
        return True
    else:
        return False


def check_tree_axis_consistency(tree_1: Any, tree_2: Any, axis_1: int, axis_2: int, err_message: str, ):
    if pytree_map_and_reduce(
        lambda x, y: array_axis_consistency(x, y, axis_1, axis_2), any, tree_1, tree_2
    ):
        raise ValueError(err_message)


def check_tree_structure(tree_1: Any, tree_2: Any, err_message: str):
    if jax.tree_util.tree_structure(tree_1) != jax.tree_util.tree_structure(tree_2):
        raise TypeError(err_message)
