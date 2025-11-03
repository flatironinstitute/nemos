"""Collection of methods utilities."""

import difflib
import warnings
from typing import Any, List, Optional, Union

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
    pytree: Any, err_message: str, data_type: Optional[DTypeLike] = None
):
    """
    Convert the leaves of a given pytree to JAX arrays with the specified data type.

    Parameters
    ----------
    pytree :
        Pytree with leaves that are array-like objects.
    err_message:
        The error message to raise if the leaves do not have the specified data type.
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
        pytree = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x, dtype=data_type), pytree
        )
    except (ValueError, TypeError):
        raise TypeError(err_message)
    return pytree


def check_tree_leaves_dimensionality(pytree: Any, expected_dim: int, err_message: str):
    """
    Check if the leaves of the pytree have the specified dimensionality.

    Parameters
    ----------
    pytree :
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
    if pytree_map_and_reduce(lambda x: x.ndim != expected_dim, any, pytree):
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
    pytree: Any, array: NDArray, axis: int, err_message: str
):
    """
    Check if the shape of an array matches the shape of arrays in a pytree along a specified axis.

    Parameters
    ----------
    pytree :
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
        lambda arr: arr.shape[axis] != array.shape[axis], any, pytree
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
    pytree_1: Any,
    pytree_2: Any,
    axis_1: int,
    axis_2: int,
    err_message: str,
):
    """
    Check if two pytrees are consistent along specified axes for their respective leaves.

    Parameters
    ----------
    pytree_1 :
        First pytree to check.
    pytree_2 :
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
        lambda x, y: array_axis_consistency(x, y, axis_1, axis_2),
        any,
        pytree_1,
        pytree_2,
    ):
        raise ValueError(err_message)


def check_tree_structure(pytree_1: Any, pytree_2: Any, err_message: str):
    """Check if two pytrees have the same structure.

    Parameters
    ----------
    pytree_1 :
        First pytree to compare.
    pytree_2 :
        Second pytree to compare.
    err_message :
        Error message to raise if the structures of the pytrees do not match.

    Raises
    ------
    TypeError
        If the structures of the pytrees do not match.
    """
    if jax.tree_util.tree_structure(pytree_1) != jax.tree_util.tree_structure(pytree_2):
        raise TypeError(err_message)


def check_fraction_valid_samples(*pytree: Any, err_msg: str, warn_msg: str) -> None:
    """
    Check the fraction of entries that are not infinite or NaN.

    Parameters
    ----------
    *pytree :
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
    valid = get_valid_multitree(pytree)
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
    all_empty_or_valid = pytree_map_and_reduce(
        lambda x: x.shape[axis] == 0 or x.shape[axis] >= window_size, all, time_series
    )
    if insufficient_window_size and not all_empty_or_valid:
        warnings.warn(
            f"One or more trials are shorter than the convolution window size "
            f"({window_size} samples). These trials will produce NaN values in the output.",
            category=UserWarning,
            stacklevel=2,
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


def _suggest_keys(
    unmatched_keys: List[str], valid_keys: List[str], cutoff: float = 0.6
):
    """
    Suggest the closest matching valid key for each unmatched key using fuzzy string matching.

    This function compares each unmatched key to a list of valid keys and returns a suggestion
    if a close match is found based on the similarity score.

    Parameters
    ----------
    unmatched_keys :
        Keys that were provided by the user but not found in the expected set.
    valid_keys :
        The list of valid/expected keys to compare against.
    cutoff :
        The minimum similarity ratio (between 0 and 1) required to consider a match.
        A higher value means stricter matching. Defaults to 0.6.

    Returns
    -------
    :
        A list of (provided_key, suggested_key) pairs. If no match is found,
        `suggested_key` will be `None`.

    Examples
    --------
    >>> _suggest_keys(["observaton_model"], ["observation_model", "regularization"])
    [('observaton_model', 'observation_model')]
    """
    key_paris = []  # format, (user_provided, similar key)
    for unmatched_key in unmatched_keys:
        suggestions = difflib.get_close_matches(
            unmatched_key, valid_keys, n=1, cutoff=cutoff
        )
        key_paris.append((unmatched_key, suggestions[0] if suggestions else None))
    return key_paris
