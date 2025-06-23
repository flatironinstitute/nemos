"""Collection of methods utilities."""

import warnings
from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
from numpy.typing import DTypeLike, NDArray

from . import utils
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


def _check_basis_matrix_shape(basis_matrix):
    basis_matrix = jnp.asarray(basis_matrix)
    if not utils.check_dimensionality(basis_matrix, 2):
        raise ValueError(
            "basis_matrix must be a 2 dimensional array! "
            f"{basis_matrix.ndim} dimensions provided instead."
        )
    if basis_matrix.shape[0] == 1:
        raise ValueError("`basis_matrix.shape[0]` should be at least 2!")
    return basis_matrix


def _check_non_empty_inputs(time_series, basis_matrix):
    utils.check_non_empty(basis_matrix, "basis_matrix")
    utils.check_non_empty(time_series, "time_series")


def _check_time_series_ndim(time_series, axis):
    if not utils.pytree_map_and_reduce(lambda x: x.ndim > axis, all, time_series):
        raise ValueError(
            "`time_series` should contain arrays of at least one-dimension. "
            "At least one 0-dimensional array provided."
        )


def _check_shift_causality_consistency(shift, predictor_causality):
    """Check shift causality consistency."""
    if shift and predictor_causality == "acausal":
        raise ValueError(
            "Cannot shift `predictor` when `predictor_causality` is `acausal`!"
        )


def _check_batch_size(batch_size, var_name):
    """Check if ``batch_size`` is a positive integer."""
    if batch_size is None:
        return
    elif not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(
            f"When provided ``{var_name}`` must be a strictly positive integer! "
            f"``{batch_size}`` provided instead."
        )


def _check_trials_longer_than_time_window(
    time_series: Any, window_size: int | Any, axis: int = 0
):
    """
    Check if the duration of each trial in the time series is at least as long as the window size.

    Parameters
    ----------
    time_series :
        A pytree of trial data.
    window_size :
        The size of the window to be used in convolution. Either an int or a pytree with the same
        struct as time_series.
    axis :
        The axis in the arrays representing the time dimension.

    Raises
    ------
    ValueError
        If any trial in the time series is shorter than the window size.
    """
    has_same_struct = jax.tree_util.tree_structure(
        time_series
    ) == jax.tree_util.tree_structure(window_size)
    if has_same_struct:
        insufficient_window_size = pytree_map_and_reduce(
            lambda x, w: x.shape[axis] < w, any, time_series, window_size
        )
    else:
        insufficient_window_size = pytree_map_and_reduce(
            lambda x: x.shape[axis] < window_size, any, time_series
        )
    # Check window size
    if insufficient_window_size:
        raise ValueError(
            "Insufficient trial duration. The number of time points in each trial must "
            "be greater or equal to the window size."
        )


def _check_batch_size_larger_than_convolution_window(
    batch_size: int | Any, window_size: int | Any
):
    """Check if the batch_size is larger than the window size."""
    has_same_struct = jax.tree_util.tree_structure(
        batch_size
    ) == jax.tree_util.tree_structure(window_size)
    if has_same_struct:
        insufficient_window_size = pytree_map_and_reduce(
            lambda x, w: x < w, any, batch_size, window_size
        )
    else:
        insufficient_window_size = pytree_map_and_reduce(
            lambda x: x < window_size, any, batch_size
        )
    if insufficient_window_size:
        bs = jax.tree_util.tree_leaves(batch_size)[0]
        ws = jax.tree_util.tree_leaves(window_size)[0]
        raise ValueError(
            "Batch size too small. Batch size must be larger than the convolution window size. "
            f"The provided batch size is ``{bs}``, while the window size for the convolution is ``{ws}``. "
            "Please increase the batch size."
        )
