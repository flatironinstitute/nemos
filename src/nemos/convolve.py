"""Convolution utilities."""

# required to get ArrayLike to render correctly
from __future__ import annotations

import re
import warnings
from functools import partial
from math import prod
from typing import Any, Literal, Optional

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import type_casting, utils, validation


def _resolve_shift_default(shift, predictor_causality):
    if shift is None:
        return predictor_causality != "acausal"
    return shift


_CORR_VEC_BASIS = jax.vmap(partial(jnp.convolve, mode="valid"), (None, 1), 1)

_CORR_VEC = jax.vmap(partial(jnp.convolve, mode="valid"), (1, None), 1)
_CORR_VEC = jax.vmap(_CORR_VEC, (None, 1), 2)


def _batched_convolve(array: NDArray, eval_basis: NDArray, batch_size: int):
    n_filters = eval_basis.shape[1]
    window_size = eval_basis.shape[0]

    leftover = n_filters % batch_size
    n_batches = n_filters // batch_size

    def conv_batch(_, batch_idx):
        start = batch_idx * batch_size
        chunk = jax.lax.dynamic_slice(eval_basis, (0, start), (window_size, batch_size))
        conv_out = _CORR_VEC(array, chunk)
        return None, conv_out

    _, batched_convs = jax.lax.scan(conv_batch, None, jnp.arange(n_batches))
    batched_convs = np.transpose(batched_convs, axes=(1, 2, 0, 3)).reshape(
        *batched_convs.shape[1:-1], -1
    )
    if leftover > 0:
        batched_convs = jnp.concatenate(
            [batched_convs, _CORR_VEC(array, eval_basis[:, n_batches * batch_size :])],
            axis=2,
        )
    return batched_convs


@partial(jax.jit, static_argnums=(2, 3))
def tensor_convolve(
    array: NDArray, eval_basis: NDArray, batch_time_series: int, batch_basis: int
):
    """
    Memory-efficient convolution using scan over batches of input channels.

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
    batch_time_series :
        Batch size over time series channels.
    batch_basis:
        Batch size over basis filters.

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
    n_samples, *feat_shape = array.shape
    vectorized_dimension = prod(feat_shape)  # total number of columns
    array_flat = array.reshape(n_samples, -1)

    window_size, n_basis_funcs = eval_basis.shape
    n_samples_out = n_samples - window_size + 1

    n_batches = vectorized_dimension // batch_time_series
    leftover = vectorized_dimension % batch_time_series

    def conv_batch(_, batch_idx):
        # Get start:end column indices
        start = batch_idx * batch_time_series

        # Slice and apply _CORR_VEC
        chunk = jax.lax.dynamic_slice(
            array_flat, (0, start), (n_samples, batch_time_series)
        )
        conv_out = _batched_convolve(
            chunk, eval_basis, batch_basis
        )  # shape: (T_out, batch_size, B)
        return None, conv_out

    _, batched_convs = jax.lax.scan(
        conv_batch, None, jnp.arange(n_batches)
    )  # shape: (n_batches, n_samples_out, batch_size, n_basis)

    # Reshape to (n_samples_out, batch_size * n_batches, n_basis)
    batched_convs = batched_convs.transpose(1, 0, 2, 3).reshape(
        n_samples_out, n_batches * batch_time_series, n_basis_funcs
    )

    def finish_conv():
        leftover_chunk = array_flat[
            :, n_batches * batch_time_series :
        ]  # shape: (T, leftover)
        leftover_conv = _CORR_VEC(
            leftover_chunk, eval_basis
        )  # shape: (T_out, leftover, B)
        full_conv = jnp.concatenate([batched_convs, leftover_conv], axis=1)
        return full_conv

    # Handle leftovers
    if leftover:
        leftover_conv = _CORR_VEC(
            array_flat[:, n_batches * batch_time_series :], eval_basis
        )  # shape: (T_out, leftover, B)
        batched_convs = jnp.concatenate([batched_convs, leftover_conv], axis=1)

    return batched_convs.reshape(n_samples_out, *feat_shape, n_basis_funcs)


def _shift_time_axis_and_convolve(
    array: NDArray,
    eval_basis: NDArray,
    axis: int,
    batches_time_series: int,
    batches_basis: int,
):
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
    batches_time_series:
        Number of batched input channels for the convolution.
    batches_basis:
        Number of batched basis filters for the convolution.

    Returns
    -------
    NDArray
        The array after applying the convolution along the specified axis, with the original
        axis order restored.

    Notes
    -----
    This function supports arrays of any dimensionality greater or equal than 1.
    """
    # convert axis
    axis = axis if axis >= 0 else array.ndim + axis
    # move time axis to first
    new_axis = (jnp.arange(array.ndim) + axis) % array.ndim
    array = jnp.transpose(array, new_axis)

    # convolve
    if array.ndim > 1:
        conv = tensor_convolve(array, eval_basis, batches_time_series, batches_basis)
    else:
        conv = tensor_convolve(
            array[:, jnp.newaxis], eval_basis, batches_time_series, batches_basis
        )[:, 0]

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
    batches_time_series: Any,
    batches_basis: int,
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
    batches_time_series :
        Pytree of batch sizes. The number of batched channels for the convolution.
    batches_basis :
        Number of batched basis filters for the convolution.
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
    def conv(x, bs):
        return _shift_time_axis_and_convolve(
            x,
            basis_matrix,
            axis=axis,
            batches_time_series=bs,
            batches_basis=batches_basis,
        )

    predictor = jax.tree_util.tree_map(conv, time_series, batches_time_series)

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
    batches_time_series: Optional[int] = None,
    batches_basis: Optional[int] = None,
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
    batches_time_series :
        Batch size for the time series channels.
    batches_basis :
        Batch size for the convolved basis filters.

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
    # apply checks
    validation.check_basis_matrix_shape(basis_matrix)
    validation.check_non_empty_inputs(time_series, basis_matrix)
    validation.check_time_series_ndim(time_series, axis)
    shift = _resolve_shift_default(shift, predictor_causality)
    validation.check_shift_causality_consistency(shift, predictor_causality)

    # flatten and grab tree struct
    time_series, struct = jax.tree_util.tree_flatten(time_series)

    if batches_time_series is None:
        batches_time_series = jax.tree_util.tree_map(
            lambda x: prod(x.shape[:axis] + x.shape[axis + 1 :]), time_series
        )
    elif not isinstance(batches_time_series, int) or batches_time_series < 1:
        raise ValueError(
            f"When provided `batches_time_series` must be a strictly positive integer! "
            f"{batches_time_series} provided instead."
        )
    else:
        batches_time_series = jax.tree_util.tree_map(
            lambda x: min(
                [prod(x.shape[:axis] + x.shape[axis + 1 :]), batches_time_series]
            ),
            time_series,
        )

    if batches_basis is None:
        batches_basis = basis_matrix.shape[1]
    elif not isinstance(batches_basis, int) or batches_basis < 1:
        raise ValueError(
            f"When provided `batches_basis` must be a strictly positive integer! "
            f"{batches_basis} provided instead."
        )
    else:
        batches_basis = min([batches_basis, basis_matrix.shape[1]])

    # find pynapple
    is_nap = list(type_casting.is_pynapple_tsd(x) for x in time_series)

    # retrieve time info
    time_info = [
        type_casting.get_time_info(ts) if is_nap[i] else None
        for i, ts in enumerate(time_series)
    ]

    # split epochs (adds one layer to pytree)
    # if pynapple one batch size per epoch to match tree-struct
    batches_time_series = jax.tree_util.tree_map(
        lambda x, y: (
            [x] * len(y.time_support) if type_casting.is_pynapple_tsd(y) else [x]
        ),
        batches_time_series,
        time_series,
    )
    time_series = jax.tree_util.tree_map(_list_epochs, time_series)

    # check trial size (after splitting)
    utils.check_trials_longer_than_time_window(time_series, basis_matrix.shape[0], axis)

    # convert to array
    time_series = jax.tree_util.tree_map(jnp.asarray, time_series)

    # convolve
    conv = _convolve_pad_and_shift(
        basis_matrix,
        time_series,
        batches_time_series=batches_time_series,
        batches_basis=batches_basis,
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
