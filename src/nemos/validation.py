"""Collection of methods utilities."""

import warnings
from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
from numpy.typing import DTypeLike, NDArray

from .pytrees import FeaturePytree
from .tree_utils import get_valid_multitree, pytree_map_and_reduce


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
    any_infs = pytree_map_and_reduce(
        jnp.any, any, jax.tree_util.tree_map(jnp.isinf, pytree)
    )
    any_nans = pytree_map_and_reduce(
        jnp.any, any, jax.tree_util.tree_map(jnp.isnan, pytree)
    )
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


def check_length(x: Any, expected_len: int, err_message: str):
    """
    Check if the provided object has a length of two.

    Parameters
    ----------
    x :
        Object to check the length of.
    expected_len :
        The expected length of the object.
    err_message :
        Error message to raise if the length is not two.

    Raises
    ------
    ValueError
        If the object does not have the specified length.
    """
    try:
        assert len(x) == expected_len
    except Exception:
        raise ValueError(err_message)


def convert_tree_leaves_to_jax_array(
    tree: Any, err_message: str, data_type: Optional[DTypeLike] = None
):
    """
    Convert the leaves of a given pytree to JAX arrays with the specified data type.

    Parameters
    ----------
    tree :
        Pytree with leaves that are array-like objects.
    data_type :
        Data type to convert the leaves to.

    Raises
    ------
    TypeError
        If conversion to JAX arrays fails due to incompatible types.

    Returns
    -------
    :
        A tree of the same structure as the original, with leaves converted
        to JAX arrays.
    """
    try:
        tree = jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=data_type), tree)
    except (ValueError, TypeError):
        raise TypeError(err_message)
    return tree


def check_tree_leaves_dimensionality(tree: Any, expected_dim: int, err_message: str):
    """
    Check if the leaves of the pytree have the specified dimensionality.

    Parameters
    ----------
    tree :
        Pytree to check the dimensionality of its leaves.
    expected_dim :
        Expected dimensionality of the leaves.
    err_message :
        Error message to raise if the dimensionality does not match.

    Raises
    ------
    ValueError
        If any leaf does not match the expected dimensionality.
    """
    if pytree_map_and_reduce(lambda x: x.ndim != expected_dim, any, tree):
        raise ValueError(err_message)


def check_same_shape_on_axis(*arrays: NDArray, axis: int = 0, err_message: str):
    """
    Check if the arrays have the same shape along a specified axis.

    Parameters
    ----------
    *arrays :
        Arrays to check shape consistency of.
    axis :
        Axis along which to check the shape consistency.
    err_message :
        Error message to raise if the shapes are inconsistent.

    Raises
    ------
    ValueError
        If the arrays do not have the same shape along the specified axis.
    """
    if len(arrays) > 1:
        if any(arr.shape[axis] != arrays[0].shape[axis] for arr in arrays[1:]):
            raise ValueError(err_message)


def check_array_shape_match_tree(
    tree: Any, array: NDArray, axis: int, err_message: str
):
    """
    Check if the shape of an array matches the shape of arrays in a pytree along a specified axis.

    Parameters
    ----------
    tree :
        Pytree with arrays as leaves.
    array :
        Array to compare the shape with.
    axis :
        Axis along which to compare the shape.
    err_message :
        Error message to raise if the shapes do not match.

    Raises
    ------
    ValueError
        If the array's shape does not match the pytree leaves' shapes along the specified axis.
    """
    if pytree_map_and_reduce(
        lambda arr: arr.shape[axis] != array.shape[axis], any, tree
    ):
        raise ValueError(err_message)


def array_axis_consistency(
    array_1: Union[FeaturePytree, jnp.ndarray, NDArray],
    array_2: Union[FeaturePytree, jnp.ndarray, NDArray],
    axis_1: int,
    axis_2: int,
):
    """
    Check if two arrays are consistent along specified axes.

    Parameters
    ----------
    array_1 :
        First array to check.
    array_2 :
        Second array to check.
    axis_1 :
        Axis to check in the first array.
    axis_2 :
        Axis to check in the second array.

    Returns
    -------
    bool
        True if inconsistent, otherwise False.
    """
    if array_1.shape[axis_1] != array_2.shape[axis_2]:
        return True
    else:
        return False


def check_tree_axis_consistency(
    tree_1: Any,
    tree_2: Any,
    axis_1: int,
    axis_2: int,
    err_message: str,
):
    """
    Check if two pytrees are consistent along specified axes for their respective leaves.

    Parameters
    ----------
    tree_1 :
        First pytree to check.
    tree_2 :
        Second pytree to check.
    axis_1 :
        Axis to check in the first pytree.
    axis_2 :
        Axis to check in the second pytree.
    err_message :
        Error message to raise if the pytrees' leaves are inconsistent along the given axes.

    Raises
    ------
    ValueError
        If the pytrees' leaves are inconsistent along the specified axes.
    """
    if pytree_map_and_reduce(
        lambda x, y: array_axis_consistency(x, y, axis_1, axis_2), any, tree_1, tree_2
    ):
        raise ValueError(err_message)


def check_tree_structure(tree_1: Any, tree_2: Any, err_message: str):
    """Check if two pytrees have the same structure.

    Parameters
    ----------
    tree_1 :
        First pytree to compare.
    tree_2 :
        Second pytree to compare.
    err_message :
        Error message to raise if the structures of the pytrees do not match.

    Raises
    ------
    TypeError
        If the structures of the pytrees do not match.
    """
    if jax.tree_util.tree_structure(tree_1) != jax.tree_util.tree_structure(tree_2):
        raise TypeError(err_message)


def check_fraction_valid_samples(*tree: Any, err_msg: str, warn_msg: str) -> None:
    """
    Check the fraction of entries that are not infinite or NaN.

    Parameters
    ----------
    *tree :
        Trees containing arrays with the same sample axis.
    err_msg :
        The exception message.
    warn_msg :
        The warning message.

    Raises
    ------
    ValueError
        If all the samples contain invalid entries (either NaN or Inf).

    Warns
    -----
    UserWarning
        If more than 90% of the sample points contain NaNs or Infs.
    """
    valid = get_valid_multitree(tree)
    if all(~valid):
        raise ValueError(err_msg)
    elif valid.mean() <= 0.1:
        warnings.warn(warn_msg, UserWarning)


def _warn_if_not_float64(feature_matrix: Any, message: str):
    """Warn if the feature matrix uses float32 precision."""
    all_float64 = pytree_map_and_reduce(
        lambda x: jnp.issubdtype(x.dtype, jnp.float64), all, feature_matrix
    )
    if not all_float64:
        warnings.warn(
            message,
            UserWarning,
        )
