"""Convolution utilities."""

# required to get ArrayLike to render correctly
from __future__ import annotations

import re
import warnings
from functools import partial
from math import prod
from typing import Any, Callable, List, Literal, Optional

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike, NDArray

from . import type_casting, utils, validation


def _resolve_shift_default(shift, predictor_causality):
    if shift is None:
        return predictor_causality != "acausal"
    return shift


_CORR_VEC_BASIS = jax.vmap(partial(jnp.convolve, mode="valid"), (None, 1), 1)

_CORR_VEC = jax.vmap(partial(jnp.convolve, mode="valid"), (1, None), 1)
_CORR_VEC = jax.vmap(_CORR_VEC, (None, 1), 2)


def _reorganize_scan_out(out, axis):
    out = jnp.transpose(
        out, [i + 1 for i in range(axis)] + [0] + [i for i in range(axis + 1, out.ndim)]
    )
    return out.reshape(*out.shape[:axis], -1, *out.shape[axis + 2 :])


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def _batch_binary_func(
    batched_array: NDArray,
    other_array: NDArray,
    binary_func: Callable[[NDArray, NDArray], NDArray],
    batch_size: int,
    axis: int = 1,
    out_axis: int = 1,
    pad_final_batch: bool = True,
):
    """
    Apply a binary function in memory-efficient batches along a specified axis.

    This function slices `batched_array` along the given `axis` into batches of size `batch_size`,
    and applies `binary_func(chunk, other_array)` to each batch. Results are concatenated
    along the same axis.

    Optionally, the final batch can be padded to avoid recompilation due to varying shapes.

    Parameters
    ----------
    batched_array :
        The array to be processed in chunks along the specified axis.
        This is typically the larger array in a binary operation.
    other_array :
        The second operand passed to `binary_func` (not batched).
        This is typically shared across batches (e.g. convolution kernel).
    binary_func :
        A function accepting two arrays (batch_chunk, other_array) and returning an array of the same shape
        as a batch result. Must be JAX-compatible and support broadcasting across batch dimensions.
    batch_size :
        Number of elements to process per batch along the batching axis.
    axis :
        The axis along which to batch `batched_array`.
    out_axis :
        The axis over which concatenating the scan output.
    pad_final_batch :
        If True, pads the final batch to `batch_size` with zeros (and slices off the padded output),
        ensuring uniform shape for all batches and preventing JAX recompilation.
        If False, the leftover portion is processed separately, which may incur compilation overhead.

    Returns
    -------
    batched_out :
        The output of applying `binary_func` to all slices of `batched_array`, concatenated along `out_axis`.

    Notes
    -----
    - This function is useful when applying memory-intensive operations (e.g., convolutions) on large arrays.
    - If `pad_final_batch` is False, JAX may compile multiple versions of `binary_func`, affecting performance.
    - If both operands must be batched, a separate batching loop should be implemented externally.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import jax
    >>>
    >>> jax.config.update("jax_enable_x64", True)
    >>> def func(x, y):
    ...     return x + y
    >>> x, y = jax.random.normal(jax.random.PRNGKey(123), (10, 11, 12)), jnp.array(1)
    >>> result = _batch_binary_func(x, y, func, batch_size=3, axis=1)
    >>> jnp.allclose(result, x + y)
    Array(True, dtype=bool)
    """
    n_batched_features = batched_array.shape[axis]
    leftover = n_batched_features % batch_size
    n_batches = (
        (n_batched_features + batch_size - 1) // batch_size
        if pad_final_batch
        else n_batched_features // batch_size
    )

    if pad_final_batch and leftover > 0:
        pad_width = [(0, 0)] * batched_array.ndim
        pad_width[axis] = (0, batch_size - leftover)
        batched_array = jnp.pad(batched_array, pad_width, constant_values=0)

    def scan_fn(_, batch_idx):
        start = batch_idx * batch_size
        # define slices
        start_indices = [0] * axis + [start] + [0] * (batched_array.ndim - axis - 1)
        slice_sizes = (
            *batched_array.shape[:axis],
            batch_size,
            *batched_array.shape[axis + 1 :],
        )
        chunk = jax.lax.dynamic_slice(batched_array, start_indices, slice_sizes)
        conv_out = binary_func(chunk, other_array)
        return None, conv_out

    _, batched_out = jax.lax.scan(scan_fn, None, jnp.arange(n_batches))

    # Reshape
    batched_out = _reorganize_scan_out(batched_out, out_axis)

    if pad_final_batch and leftover > 0:
        batched_out = batched_out[
            (slice(None),) * out_axis
            + (slice(0, n_batched_features),)
            + (slice(None),) * (batched_out.ndim - out_axis - 1)
        ]
    elif not pad_final_batch and leftover > 0:
        # process the leftover batch manually
        slices = (
            *[slice(None)] * axis,
            slice(n_batches * batch_size, n_batched_features),
            *[slice(None)] * (batched_array.ndim - axis - 1),
        )
        leftover_out = binary_func(batched_array[slices], other_array)
        batched_out = jnp.concatenate([batched_out, leftover_out], axis=out_axis)

    return batched_out


@partial(jax.jit, static_argnums=(2, 3, 4))
def _tensor_convolve(
    array: NDArray,
    eval_basis: NDArray,
    batch_size_samples: int,
    batch_size_channels: int,
    batch_size_basis: int,
):
    """
    Memory-efficient convolution using scan over batches of input channels.

    This function first flattens the input array across dimensions other than the first one, then
    performs a vectorized convolution with the evaluation basis. The result is reshaped back to the
    original array's shape (except for the first dimension, which is adjusted based on the convolution).

    Parameters
    ----------
    array :
        The input array to convolve. It is expected to be at least 1D. The first axis is expected to be
        the sample axis, i.e. the shape of array is ``(num_samples, ...)``.
    eval_basis :
        The evaluation basis array for convolution. It should be 2D, where the first dimension
        represents the window size for convolution. Shape ``(window_size, n_basis_funcs)``.
    batch_size_samples :
        The batch size for convolution as number of samples. Must be larger than ``eval_basis.shape[1]``.
    batch_size_channels :
        Batch size over time series channels.
    batch_size_basis :
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

    array = array.reshape(n_samples, -1)
    window_size, n_basis = eval_basis.shape
    n_features = array.shape[1]

    step = batch_size_samples - window_size + 1
    n_samples_padded = (
        ((n_samples - window_size + 1 + step - 1) // step) * step + window_size - 1
    )
    n_batches = (n_samples_padded - window_size + 1) // step

    pad_len = n_samples_padded - n_samples
    array = jnp.pad(array, ((0, pad_len), (0, 0)), constant_values=0.0)

    def conv_batch(concat_conv, batch_idx):
        start_chunk = batch_idx * step
        chunk = jax.lax.dynamic_slice(
            array, (start_chunk, 0), (batch_size_samples, n_features)
        )
        concat_conv = jax.lax.dynamic_update_slice(
            concat_conv,
            _batch_convolve_over_channels(
                chunk, eval_basis, batch_size_channels, batch_size_basis
            ),
            (start_chunk, 0, 0),
        )
        return concat_conv, None

    # initialize output of valid convolution
    conv_output = jnp.full(
        (n_samples_padded - window_size + 1, n_features, n_basis), jnp.nan
    )
    conv_output, _ = jax.lax.scan(conv_batch, conv_output, jnp.arange(n_batches))
    conv_output = conv_output[: n_samples - window_size + 1]
    return conv_output.reshape(n_samples - window_size + 1, *feat_shape, n_basis)


def _batch_convolve_over_channels(
    array: NDArray, eval_basis: NDArray, batch_size_channels: int, batch_size_basis: int
):
    def conv_over_basis(array_chunk, basis_matrix):
        return _batch_over_basis_convolve(array_chunk, basis_matrix, batch_size_basis)

    return _batch_binary_func(
        array, eval_basis, conv_over_basis, batch_size_channels, axis=1
    )


def _batch_over_basis_convolve(
    array: NDArray, eval_basis: NDArray, batch_size_basis: int
):
    out = _batch_binary_func(
        eval_basis,
        array,
        _CORR_VEC,
        batch_size_basis,
        axis=1,
        out_axis=1,
    )
    # move basis axis.
    return out.transpose(0, 2, 1)


def _shift_time_axis_and_convolve(
    array: jnp.ndarray,
    eval_basis: NDArray,
    axis: int,
    batch_size_samples: int,
    batch_size_channels: int,
    batch_size_basis: int,
) -> jnp.ndarray:
    """
    Shifts the specified axis to the first position, applies convolution, and then reverses the shift.

    This applies a convolution along a specific axis of a multi-dimensional array. The process involves
    three steps: shifting the axis, convolving, and then reversing the shift to maintain the original
    axis order.

    Parameters
    ----------
    array :
        The input array for convolution.
    eval_basis :
        The evaluation basis array for convolution, should be 2D.
    axis :
        The axis along which the convolution is applied. This axis is temporarily shifted
        to the first position for the convolution operation.
    batch_size_samples :
        Size of the batches in samples.
    batch_size_channels :
        Size of the batches in number of input channels.
    batch_size_basis :
        Size of the batches in number of basis filters.

    Returns
    -------
    conv :
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
    conv = _tensor_convolve(
        array, eval_basis, batch_size_samples, batch_size_channels, batch_size_basis
    )

    # reverse transposition
    new_axis = (*((jnp.arange(array.ndim) - axis) % array.ndim), array.ndim)
    conv = jnp.transpose(conv, new_axis)
    return conv


def _list_epochs(tsd: Any):
    """
    List epochs from a time series with data object, supporting 'pynapple' Tsd formats.

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
    time_series: jnp.ndarray,
    batch_size_samples: int,
    batch_size_channels: Any,
    batch_size_basis: int,
    predictor_causality: Literal["causal", "acausal", "anti-causal"] = "causal",
    axis: int = 0,
    shift: Optional[bool] = None,
) -> jnp.ndarray:
    """
    Create predictor by convolving basis_matrix with time_series.

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
        be an array of at least one-dimension.
    batch_size_samples:
        Size of the batches in samples.
    batch_size_channels :
        Pytree of batch sizes. The number of batched channels for the convolution.
    batch_size_basis :
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
    predictor = _shift_time_axis_and_convolve(
        time_series,
        basis_matrix,
        axis=axis,
        batch_size_samples=batch_size_samples,
        batch_size_channels=batch_size_channels,
        batch_size_basis=batch_size_basis,
    )

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
    batch_size_samples: Optional[int] = None,
    batch_size_channels: Optional[int] = None,
    batch_size_basis: Optional[int] = None,
):
    """
    Create a convolutional predictor by convolving a basis matrix with a time series.

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
    batch_size_samples :
        Batch size for the convolution in terms of number of sample points.
        If this parameter is set, the convolution will be applied sequentially
        over a fixed-length chunks of the time_series.
    batch_size_channels :
        Batch size for the convolution in terms of number of input channels.
        If this parameter is set, the convolution will be vectorized over batches
        of input channels. Default vectorizes over all channels.
    batch_size_basis :
        Batch size for the convolution in terms of number of basis kernels.
        If this parameter is set, the convolution will be vectorized over batches
        of kernels. Default vectorizes over all basis kernels.

    Returns
    -------
    :
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
    ValueError:
        Raised if any explicitly provided batch sizes—``batch_size_samples``,
        ``batch_size_channels``, or ``batch_size_basis``—are not positive integers.
        A value of ``None`` is allowed and treated as unspecified.

    """
    # apply checks
    validation._check_basis_matrix_shape(basis_matrix)
    validation._check_non_empty_inputs(time_series, basis_matrix)
    validation._check_time_series_ndim(time_series, axis)
    shift = _resolve_shift_default(shift, predictor_causality)
    validation._check_shift_causality_consistency(shift, predictor_causality)
    validation._check_batch_size(batch_size_samples, "batch_size_samples")
    validation._check_batch_size(batch_size_channels, "batch_size_channels")
    validation._check_batch_size(batch_size_basis, "batch_size_basis")

    # flatten and grab tree struct
    time_series, struct = jax.tree_util.tree_flatten(time_series)

    # run a single conv if batches provided, otherwise make sure batch size is smaller than
    # array shape.
    if batch_size_samples is None:
        batch_size_samples = jax.tree_util.tree_map(
            lambda x: x.shape[axis], time_series
        )
    else:
        batch_size_samples = jax.tree_util.tree_map(
            lambda x: min([x.shape[axis], batch_size_samples]), time_series
        )

        # check that the batch size is big enough
        validation._check_batch_size_larger_than_convolution_window(
            batch_size=batch_size_samples, window_size=basis_matrix.shape[0]
        )

    if batch_size_channels is None:
        batch_size_channels = jax.tree_util.tree_map(
            lambda x: prod(x.shape[:axis] + x.shape[axis + 1 :]), time_series
        )
    else:
        batch_size_channels = jax.tree_util.tree_map(
            lambda x: min(
                [prod(x.shape[:axis] + x.shape[axis + 1 :]), batch_size_channels]
            ),
            time_series,
        )

    # basis is always a 2D array, no tree-map needed for the same logic
    if batch_size_basis is None:
        batch_size_basis = basis_matrix.shape[1]
    else:
        batch_size_basis = min([batch_size_basis, basis_matrix.shape[1]])

    # find pynapple
    is_nap = list(type_casting.is_pynapple_tsd(x) for x in time_series)

    if not any(is_nap):
        # no pynapple, validate shape and run convolutions
        validation._check_trials_longer_than_time_window(
            time_series, basis_matrix.shape[0], axis
        )

        def apply_convolution(x, bs, bc):
            if x.shape[axis] < basis_matrix.shape[0]:
                return jnp.full((*x.shape, basis_matrix.shape[1]), jnp.nan)
            return _convolve_pad_and_shift(
                basis_matrix,
                x,
                batch_size_samples=bs,
                batch_size_channels=bc,
                batch_size_basis=batch_size_basis,
                predictor_causality=predictor_causality,
                axis=axis,
                shift=shift,
            )

        flat_stack = jax.tree_util.tree_map(
            apply_convolution, time_series, batch_size_samples, batch_size_channels
        )
    else:
        flat_stack = _convolve_pynapple(
            basis_matrix,
            time_series,
            is_nap,
            axis,
            shift,
            predictor_causality,
            batch_size_channels,
            batch_size_samples,
            batch_size_basis,
        )
    return jax.tree_util.tree_unflatten(struct, flat_stack)


def _convolve_pynapple(
    basis_matrix: NDArray | jnp.ndarray,
    time_series: Any,
    is_nap: List[bool],
    axis: int,
    shift: bool,
    predictor_causality: Literal["causal", "acausal", "anti-causal"],
    batch_size_channels: List[int],
    batch_size_samples: List[int],
    batch_size_basis: int,
):
    """
    Convolve a PyTree containing pynapple time series.

    Parameters
    ----------
    basis_matrix:
        The basis matrix for the convolution as a 2D array.
    time_series :
        The time series data to convolve with the basis matrix, after tree-flattening (i.e. it is a list of arrays).
    is_nap :
        A list of the same length of time_series of boolean, marking which time series is a pynapple object.
    axis :
        The axis along which the convolution is applied.
    shift :
        Determines whether to shift the convolution result based on the causality.
        If None, it defaults to True for 'causal' and 'anti-causal' and to False for 'acausal'.
    predictor_causality :
        The causality of the predictor, determining how the padding and shifting
        should be applied to the convolution result.
        - 'causal': Pads and/or shifts the result to be causal with respect to the input.
        - 'acausal': Applies padding equally on both sides without shifting.
        - 'anti-causal': Pads and/or shifts the result to be anti-causal with respect to the input.
    batch_size_samples :
        Batch size for the convolution in terms of number of sample points.
        If this parameter is set, the convolution will be applied sequentially
        over a fixed-length chunks of the time_series.
    batch_size_channels :
        Batch size for the convolution in terms of number of input channels.
        If this parameter is set, the convolution will be vectorized over batches
        of input channels. Default vectorizes over all channels.
    batch_size_basis :
        Batch size for the convolution in terms of number of basis kernels.
        If this parameter is set, the convolution will be vectorized over batches
        of kernels. Default vectorizes over all basis kernels.

    Returns
    -------
    :
        A list with the convolved time series.
    """
    # retrieve time info
    time_info = [
        type_casting.get_time_info(ts) if is_nap[i] else None
        for i, ts in enumerate(time_series)
    ]

    # split epochs (adds one layer to pytree)
    # if pynapple one batch size per epoch to match tree-struct
    def add_tree_level(x, y):
        return [x] * len(y.time_support) if type_casting.is_pynapple_tsd(y) else [x]

    batch_size_channels = jax.tree_util.tree_map(
        add_tree_level,
        batch_size_channels,
        time_series,
    )
    batch_size_samples = jax.tree_util.tree_map(
        add_tree_level,
        batch_size_samples,
        time_series,
    )
    time_series = jax.tree_util.tree_map(_list_epochs, time_series)

    # convert to array
    time_series = jax.tree_util.tree_map(jnp.asarray, time_series)

    # check trial size (after splitting)
    validation._check_trials_longer_than_time_window(
        time_series, basis_matrix.shape[0], axis
    )

    def apply_convolution(x, bs, bc):
        if x.shape[axis] < basis_matrix.shape[0]:
            return jnp.full((*x.shape, basis_matrix.shape[1]), jnp.nan)
        else:
            return _convolve_pad_and_shift(
                basis_matrix,
                x,
                batch_size_samples=bs,
                batch_size_channels=bc,
                batch_size_basis=batch_size_basis,
                predictor_causality=predictor_causality,
                axis=axis,
                shift=shift,
            )

    conv = jax.tree_util.tree_map(
        apply_convolution, time_series, batch_size_samples, batch_size_channels
    )

    #  concatenate back
    flat_stack = [jnp.concatenate(x, axis=axis) for x in conv]

    # re-attach time axis
    flat_stack = [
        type_casting.cast_to_pynapple(x, *time_info[i]) if is_nap[i] else x
        for i, x in enumerate(flat_stack)
    ]

    # recreate tree
    return flat_stack
