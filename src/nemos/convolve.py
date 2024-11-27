"""Convolution utilities."""

# required to get ArrayLike to render correctly
from __future__ import annotations

import re
import warnings
from functools import partial
from typing import Any, Literal, Optional

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike, NDArray

from . import type_casting, utils

_CORR_VEC_BASIS = jax.vmap(partial(jnp.convolve, mode="valid"), (None, 1), 1)

_CORR_VEC = jax.vmap(partial(jnp.convolve, mode="valid"), (1, None), 1)
_CORR_VEC = jax.vmap(_CORR_VEC, (None, 1), 2)


@jax.jit
def tensor_convolve(array: NDArray, eval_basis: NDArray):
    """
    Apply a convolution on the given array with the evaluation basis and reshapes the result.

    This function first flattens the input array across dimensions other than the first one, then
    performs a vectorized convolution with the evaluation basis. The result is reshaped back to the
    original array's shape (except for the first dimension, which is adjusted based on the convolution).

    Parameters
    ----------
    array :
        The input array to convolve. It is expected to be at least 1D. The first axis is expeted to be
        the sample axis, i.e. the shape of array is ``(num_samples, ...)``.
    eval_basis :
        The evaluation basis array for convolution. It should be 2D, where the first dimension
        represents the window size for convolution. Shape ``(window_size, n_basis_funcs)``.

    Returns
    -------
    :
        The convolved array, reshaped to maintain the original dimensions except for the first one,
        which is adjusted based on the window size of ``eval_basis``.

    Notes
    -----
    The convolution implemented here is in mode ``"valid"``. This implies that the time axis shrinks
    ``num_samples - window_size + 1``, where num_samples is the first size of the first axis of ``array``
    and ``window_size`` is the size of the first axis in ``eval_basis``.
    """
    # flatten over other dims & apply vectorized conv
    conv = _CORR_VEC(array.reshape(array.shape[0], -1), eval_basis)

    # unravel the dimensions
    window_size = eval_basis.shape[0]
    num_samples = array.shape[0]
    conv = conv.reshape(
        num_samples - window_size + 1, *array.shape[1:], eval_basis.shape[1]
    )
    return conv


def _shift_time_axis_and_convolve(array: NDArray, eval_basis: NDArray, axis: int):
    """
    Shifts the specified axis to the first position, applies convolution, and then reverses the shift.

    This applies a convolution along a specific axis of a multi-dimensional array. The process involves
    three steps: shifting the axis, convolving, and then reversing the shift to maintain the original
    axis order.

    Parameters
    ----------
    array : NDArray
        The input array for convolution.
    eval_basis : NDArray
        The evaluation basis array for convolution, should be 2D.
    axis : int
        The axis along which the convolution is applied. This axis is temporarily shifted
        to the first position for the convolution operation.

    Returns
    -------
    NDArray
        The array after applying the convolution along the specified axis, with the original
        axis order restored.

    Notes
    -----
    This function supports arrays of any dimensionality greater or equal than 1.
    """
    # move time axis to first
    new_axis = (jnp.arange(array.ndim) + axis) % array.ndim
    array = jnp.transpose(array, new_axis)

    # convolve
    if array.ndim > 1:
        conv = tensor_convolve(array, eval_basis)
    else:
        conv = _CORR_VEC_BASIS(array, eval_basis)

    # reverse transposition
    new_axis = (*((jnp.arange(array.ndim) - axis) % array.ndim), array.ndim)
    conv = jnp.transpose(conv, new_axis)
    return conv


def _list_epochs(tsd: Any):
    """List epochs from a time series with data object, supporting 'pynapple' Tsd formats.

    If the input is recognized as a 'pynapple' TSD, it extracts epochs based on the TSD's
    time support. Otherwise, it returns the input as is, assuming it's a single epoch.

    Parameters
    ----------
    tsd :
        The time series data, potentially in Pynapple TSD format or a generic data structure.

    Returns
    -------
    :
        A list of epochs extracted from the TSD. If the input is not a Pynapple TSD, returns
        a list containing the input itself.
    """
    if type_casting.is_pynapple_tsd(tsd):
        return [tsd.get(s, e) for s, e in tsd.time_support.values]
    return [tsd]


def _convolve_pad_and_shift(
    basis_matrix: ArrayLike,
    time_series: Any,
    predictor_causality: Literal["causal", "acausal", "anti-causal"] = "causal",
    axis: int = 0,
    shift: Optional[bool] = None,
):
    """Create predictor by convolving basis_matrix with time_series.

    To create the convolutional predictor, three steps are taken, each of
    calls a single function. See their docstrings for more details about
    each step. This function **preserves** the number of samples by
    NaN-padding appropriately.

    - Convolve `basis_matrix` with `time_series` (function: `convolve_1d_trials`)
    - Pad output with `basis_matrix.shape[0]-1` NaNs, with location based on
      causality of intended predictor (function: `nan_pad`).
    - (Optional) Shift predictor based on causality (function: `shift_time_series`)

    Parameters
    ----------
    basis_matrix :
        The basis matrix with which to convolve the trials. Shape
        `(window_size, n_basis_funcs)`.
    time_series :
        The time series to convolve with the basis matrix. This variable should
        be a pytree with arrays of at least one-dimension as leaves.
    predictor_causality:
        Causality of this predictor, which determines where padded values are
        added and how the predictor is shifted.
    axis:
        Axis containing samples.
    shift :
        Whether to shift predictor based on causality (only valid if
        `predictor_causality != 'acausal'`). Default is True for `causal` and
        `anti-causal`, False for `acausal`.

    Returns
    -------
    predictor :
        Predictor of with same shape and structure as `time_series`
    """

    # apply convolution
    def conv(x):
        return _shift_time_axis_and_convolve(x, basis_matrix, axis=axis)

    predictor = jax.tree_util.tree_map(conv, time_series)

    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter("always")
        predictor = utils.nan_pad(
            predictor, basis_matrix.shape[0] - 1, predictor_causality, axis=axis
        )

    for w in warns:
        if re.match("^With acausal filter", str(w.message)):
            warnings.warn(
                message="With `acausal` filter, `basis_matrix.shape[0] "
                "should probably be odd, so that we can place an equal number of NaNs on "
                "either side of input.",
                category=UserWarning,
            )
        else:
            warnings.warn(w.message, w.category)

    if shift:
        predictor = utils.shift_time_series(predictor, predictor_causality, axis=axis)
    return predictor


def create_convolutional_predictor(
    basis_matrix: ArrayLike,
    time_series: Any,
    predictor_causality: Literal["causal", "acausal", "anti-causal"] = "causal",
    shift: Optional[bool] = None,
    axis: int = 0,
):
    """Create a convolutional predictor by convolving a basis matrix with a time series.

    To create the convolutional predictor, three steps are taken, convolve, pad and shift.

    - Convolve `basis_matrix` with `time_series` (function: `_convolve_1d_trials`)
    - Pad output with `basis_matrix.shape[0]-1` NaNs, with location based on
      causality of intended predictor (function: `nemos.utils.nan_pad`).
    - (Optional) Shift predictor based on causality (function: `nemos.utils.shift_time_series`)

    The function is designed to handle both single arrays and pynapple time series with data
     (across multiple epochs), as well as their combinations. For pynapple time series, it
    treats each epoch separately, applies the convolution process, and then reassembles the time series.

    Parameters
    ----------
    basis_matrix :
        A 2D array representing the basis matrix to convolve with the time series.
        The first dimension should represent the window size for the convolution.
    time_series :
        The time series data to convolve with the basis matrix. Can be single arrays, pytree of arrays,
        pynapple time series, pytree of pynapple time series or a mix.
        In case of Pynapple time series data, each epoch will be convolved separately.
    predictor_causality :
        The causality of the predictor, determining how the padding and shifting
        should be applied to the convolution result.
        - 'causal': Pads and/or shifts the result to be causal with respect to the input.
        - 'acausal': Applies padding equally on both sides without shifting.
        - 'anti-causal': Pads and/or shifts the result to be anti-causal with respect to the input.
    shift :
        Determines whether to shift the convolution result based on the causality.
        If None, it defaults to True for 'causal' and 'anti-causal' and to False for 'acausal'.
    axis :
        The axis along which the convolution is applied.

    Returns
    -------
    Any
        The convolutional predictor, structured similarly to the input `time_series`, appropriately
        padded and shifted according to the specified causality.

    Raises
    ------
    ValueError:
        If `basis_matrix` is not a 2-dimensional array or has a singleton first dimension.
    ValueError:
        If `time_series` does not contain arrays of at least one dimension or contains
        arrays with a dimensionality less than `axis`.
    ValueError:
        If any array within `time_series` or `basis_matrix` is empty.
    ValueError:
        If the number of elements along the convolution axis in any array within `time_series`
        is less than the window size of the `basis_matrix`.
    ValueError:
        If shifting is attempted with 'acausal' causality.
    """
    # convert to jnp.ndarray
    basis_matrix = jnp.asarray(basis_matrix)
    if not utils.check_dimensionality(basis_matrix, 2):
        raise ValueError(
            "basis_matrix must be a 2 dimensional array! "
            f"{basis_matrix.ndim} dimensions provided instead."
        )

    if basis_matrix.shape[0] == 1:
        raise ValueError("`basis_matrix.shape[0]` should be at least 2!")

    # check for empty inputs
    utils.check_non_empty(basis_matrix, "basis_matrix")
    utils.check_non_empty(time_series, "time_series")

    # check sample_axis exists
    if not utils.pytree_map_and_reduce(lambda x: x.ndim > axis, all, time_series):
        raise ValueError(
            "`time_series` should contain arrays of at least one-dimension. "
            "At list one 0-dimensional array provided."
        )

    # assign defaults
    if shift is None:
        if predictor_causality == "acausal":
            shift = False
        else:
            shift = True

    if shift and predictor_causality == "acausal":
        raise ValueError(
            "Cannot shift `predictor` when `predictor_causality` is `acausal`!"
        )

    # flatten and grab tree struct
    flat_tree, struct = jax.tree_util.tree_flatten(time_series)

    # find pynapple
    is_nap = list(type_casting.is_pynapple_tsd(x) for x in flat_tree)

    # retrieve time info
    time_info = [
        type_casting.get_time_info(ts) if is_nap[i] else None
        for i, ts in enumerate(flat_tree)
    ]

    # split epochs (adds one layer to pytree)
    two_layer = jax.tree_util.tree_map(_list_epochs, flat_tree)

    # check trial size (after splitting)
    utils.check_trials_longer_than_time_window(two_layer, basis_matrix.shape[0], axis)

    # convert to array
    two_layer = jax.tree_util.tree_map(jnp.asarray, two_layer)

    # convolve
    conv = _convolve_pad_and_shift(
        basis_matrix,
        two_layer,
        predictor_causality=predictor_causality,
        axis=axis,
        shift=shift,
    )

    #  concatenate back
    flat_stack = [jnp.concatenate(x, axis=axis) for x in conv]

    # re-attach time axis
    flat_stack = [
        type_casting.cast_to_pynapple(x, *time_info[i]) if is_nap[i] else x
        for i, x in enumerate(flat_stack)
    ]

    # recreate tree
    return jax.tree_util.tree_unflatten(struct, flat_stack)
