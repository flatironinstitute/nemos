"""Utilities for manipulating and checking PyTrees."""

import operator
from functools import partial, reduce
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp


def _get_not_inf(array: jnp.ndarray) -> jnp.ndarray:
    """
    Identify non-infinite entries within an array.

    This function evaluates each element in the input array to determine whether it is finite (not infinite).
    It performs this check across all axes except the first one, aggregating results using a logical 'AND' operation.
    Thus, for a given element along the first axis, if all corresponding elements in other dimensions are finite,
    the result is True; otherwise, it is False.

    Parameters
    ----------
    array : jnp.ndarray
        Input array to check for infinite values.

    Returns
    -------
        A 1D boolean array of length equal to the size of the first dimension of the input array.
        Each entry in this array corresponds to an aggregation across all other dimensions of the input array,
        with True indicating all values are finite (not infinite) and False indicating at least one infinite value.
    """
    return jax.numpy.all(~jnp.isinf(array), axis=range(1, array.ndim))


def _get_not_nan(array: jnp.ndarray) -> jnp.ndarray:
    """
    Identify non-NaN (Not a Number) entries within an array.

    Similar to the _get_not_inf function, this function checks each element in the input array for being non-NaN across
    all axes except the first one. It aggregates these checks using a logical 'AND' operation. An element along the
    first axis is considered valid (True) if all its corresponding elements in other dimensions are not NaN.

    Parameters
    ----------
    array :
        The input array for which to check for NaN values.

    Returns
    -------
    :
        A 1D boolean array of length equal to the size of the first dimension of the input array.
        Each entry in this array is the result of an aggregation across all other dimensions,
        with True indicating all values are not NaN, and False indicating at least one NaN value.
    """
    return jax.numpy.all(~jnp.isnan(array), axis=range(1, array.ndim))


def _get_valid_tree(tree: Any) -> jnp.ndarray:
    """
    Filter valid entries across all leaves in a pytree.

    Processes a pytree to identify entries without NaN or infinite values across all its leaves. A leaf is considered
    valid if all its entries are finite and not NaN. The function assumes homogeneous first dimension across all leaves.

    Parameters
    ----------
    tree :
        A pytree with leaves as NDArrays sharing the same size for the first dimension.

    Returns
    -------
        A boolean array indicating validity of each entry across all leaves. True represents a valid entry,
        while False indicates an invalid (NaN or infinite) entry.
    """
    valid = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda x: _get_not_inf(x) & _get_not_nan(x), tree)
    )
    return reduce(jnp.logical_and, valid)


def get_valid_multitree(*tree: Any) -> jnp.ndarray:
    """
    Filter valid entries across multiple pytrees.

    Evaluates multiple pytrees to identify common entries that are valid (non-NaN and finite) across all of them.
    Assumes that all pytrees have leaves with NDArrays sharing the same size for the first dimension.

    Parameters
    ----------
    tree :
        Variable number of pytrees with NDArrays as leaves, each having a consistent first dimension size.

    Returns
    -------
    :
        A boolean array indicating the validity of each entry across all leaves in all pytrees. True for valid entries,
        False for invalid ones.
    """
    return reduce(jnp.logical_and, map(_get_valid_tree, tree))


def pytree_map_and_reduce(
    map_fn: Callable,
    reduce_fn: Callable,
    *pytrees: Any,
    is_leaf: Optional[Callable[[Any], bool]] = None,
):
    """
    Apply a mapping function to each leaf of the pytrees and then reduce the results.

    This function performs a map/reduce operation where a mapping function is applied
    to each leaf of the given pytrees, and then a reduction function is used to
    aggregate these results into a single output.

    Parameters
    ----------
    map_fn :
        A function to be applied to each leaf of the pytrees. This function should
        take a single argument and return a single value.
    reduce_fn :
        A function that reduces the mapped results. This function should take an
        iterable and return a single value.
    *pytrees :
        One or more pytrees to which the map and reduce functions are applied.
    is_leaf :
        Callable, returns true if sub-tree is a leaf.

    Returns
    -------
    The result of applying the reduce function to the mapped results. The type of the
    return value depends on the reduce function.

    Examples
    --------
    >>> import nemos as nmo
    >>> pytree1 = nmo.pytrees.FeaturePytree(a=jnp.array([0]), b=jnp.array([0]))
    >>> pytree2 = nmo.pytrees.FeaturePytree(a=jnp.array([10]), b=jnp.array([20]))
    >>> map_fn = lambda x, y: x > y
    >>> # Example usage
    >>> result_any = pytree_map_and_reduce(map_fn, any, pytree1, pytree2)
    """
    cond_tree = jax.tree_util.tree_map(map_fn, *pytrees, is_leaf=is_leaf)
    # for some reason, tree_reduce doesn't work well with any.
    return reduce_fn(jax.tree_util.tree_leaves(cond_tree))


def tree_slice(data: Any, idx, is_leaf: Optional[Callable] = None):
    """
    Apply an indexing operation to each array in a nested structure.

    Parameters
    ----------
    data :
        A nested structure containing arrays (e.g., a dictionary of arrays).
    idx :
        The indexing operation to apply. This can be an integer, slice,
        NumPy array (boolean or integer), tuple of indexing operations, ellipsis, or None.
    is_leaf :
        Callable, returns true if sub-tree is a leaf.

    Returns
    -------
    Any
        A nested structure with the same format as `data`, where each array has been sliced according to `idx`.
    """
    return jax.tree_util.tree_map(lambda x: x[idx], data, is_leaf=is_leaf)


# The following functions are adapted from jaxopt.tree_utils

tree_add = partial(jax.tree_util.tree_map, operator.add)
tree_add.__doc__ = "Tree addition."

tree_sub = partial(jax.tree_util.tree_map, operator.sub)
tree_sub.__doc__ = "Tree subtraction."


def tree_scalar_mul(scalar, tree_x):
    """Compute scalar * tree_x."""
    return jax.tree_util.tree_map(lambda x: scalar * x, tree_x)


def tree_add_scalar_mul(tree_x, scalar, tree_y):
    """Compute tree_x + scalar * tree_y."""
    return jax.tree_util.tree_map(lambda x, y: x + scalar * y, tree_x, tree_y)


def tree_sum(tree_x):
    """Compute sum(tree_x)."""
    sums = jax.tree_util.tree_map(jnp.sum, tree_x)
    return jax.tree_util.tree_reduce(operator.add, sums)


def tree_l2_norm(tree_x, squared=False):
    """Compute the l2 norm ||tree_x||."""
    squared_tree = jax.tree_util.tree_map(
        lambda leaf: jnp.square(leaf.real) + jnp.square(leaf.imag), tree_x
    )
    sqnorm = tree_sum(squared_tree)
    if squared:
        return sqnorm
    else:
        return jnp.sqrt(sqnorm)


def tree_zeros_like(tree_x):
    """Creates an all-zero tree with the same structure as tree_x."""
    return jax.tree_util.tree_map(jnp.zeros_like, tree_x)
