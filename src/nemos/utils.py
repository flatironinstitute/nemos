"""Utility functions for data pre-processing."""

# required to get ArrayLike to render correctly, unnecessary as of python 3.10
from __future__ import annotations

import warnings
from typing import Any, Callable, List, Literal, Optional, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
from pynapple import Tsd, TsdFrame, TsdTensor

from .tree_utils import pytree_map_and_reduce
from .type_casting import is_numpy_array_like, support_pynapple


def check_dimensionality(
    pytree: Any,
    expected_dim: int,
) -> bool:
    """
    Check the dimensionality of the arrays in a pytree.

    Check that all arrays in pytree have the expected dimensionality.

    Parameters
    ----------
    pytree :
        A pytree object.
    expected_dim :
        Number of expected dimension for the NDArrays.

    Returns
    -------
    True if all the arrays has the expected number of dimension, False otherwise.
    """
    return not pytree_map_and_reduce(lambda x: x.ndim != expected_dim, any, pytree)


def validate_axis(tree: Any, axis: int):
    """
    Validate the axis for each array in a given tree structure.

    This function checks if the specified axis exists in each array within the tree. It raises a ValueError
    if the specified axis is equal to or greater than the number of dimensions in any of the arrays.

    Parameters
    ----------
    tree :
        A tree containing arrays.
    axis :
        The axis that should be valid for each array in the tree. This means each array must have at least
        `axis + 1` dimensions.

    Raises
    ------
    ValueError
        - If the specified axis is equal to or greater than the number of dimensions (`ndim`) of any array
        within the tree. This ensures that operations intended for a specific axis can be safely performed
        on every array in the tree.
        - If the axis is negative or non-integer.
    """
    if not isinstance(axis, int) or axis < 0:
        raise ValueError("`axis` must be a non negative integer.")

    if pytree_map_and_reduce(lambda x: x.ndim <= axis, any, tree):
        raise ValueError(
            "'axis' must be smaller than the number of dimensions of any array in 'tree'."
        )


def check_convolve_input_dims(basis_matrix: jnp.ndarray, time_series: Any):
    """
    Check the dimensions of inputs for convolution operation.

    This function validates that the `basis_matrix` is 2-dimensional and the `time_series`
    is either a pytree of 2-dimensional arrays or a single 3-dimensional array.

    Parameters
    ----------
    basis_matrix :
        A 2-dimensional array representing the basis matrix.
    time_series :
        A pytree of 2-dimensional arrays or a single 3-dimensional array representing the time series.

    Raises
    ------
    ValueError
        If `basis_matrix` is not a 2-dimensional array or if `time_series` is not a pytree of
        2-dimensional arrays or a single 3-dimensional array.
    """
    # check input size
    if not check_dimensionality(basis_matrix, expected_dim=2):
        raise ValueError("basis_matrix must be a 2 dimensional array-like object.")

    try:
        if time_series.ndim != 3:
            raise AttributeError
    except AttributeError:
        if not check_dimensionality(time_series, 2):
            raise ValueError(
                "time_series must be a pytree of 2 dimensional array-like objects or a"
                " 3 dimensional array-like object."
            )


def check_non_empty(pytree: Any, pytree_name: str):
    """
    Check if any array in the pytree is empty.

    Parameters
    ----------
    pytree :
        A pytree object containing arrays.
    pytree_name :
        The name of the pytree variable for error message purposes.

    Raises
    ------
    ValueError
        If any array in the pytree is empty (i.e., has a zero dimension).
    """
    if pytree_map_and_reduce(lambda x: 0 in x.shape, any, pytree):
        raise ValueError(
            f"Empty array provided. At least one of dimension in {pytree_name} is empty."
        )


def check_trials_longer_than_time_window(
    time_series: Any, window_size: int, axis: int = 0
):
    """
    Check if the duration of each trial in the time series is at least as long as the window size.

    Parameters
    ----------
    time_series :
        A pytree of trial data.
    window_size :
        The size of the window to be used in convolution.
    axis :
        The axis in the arrays representing the time dimension.

    Raises
    ------
    ValueError
        If any trial in the time series is shorter than the window size.
    """
    # Check window size
    if pytree_map_and_reduce(lambda x: x.shape[axis] < window_size, any, time_series):
        raise ValueError(
            "Insufficient trial duration. The number of time points in each trial must "
            "be greater or equal to the window size."
        )


def _pad_dimension(
    array: jnp.ndarray,
    axis: int,
    pad_size: int,
    predictor_causality: Literal["causal", "acausal", "anti-causal"] = "causal",
    constant_values: float = jnp.nan,
) -> jnp.ndarray:
    """
    Add padding to the last dimension of an array based on the convolution type.

    This is a helper function used by `nan_pad_conv`, which is the function we expect the user will interact with.

    Parameters
    ----------
    array:
        The array to be padded.
    axis:
        The axis to be padded.
    pad_size:
        The number of NaNs to concatenate as padding.
    predictor_causality:
        Causality of this predictor, which determines where padded values are added.
    constant_values:
        The constant values for padding, default is jnp.nan.

    Returns
    -------
    :
        An array with padded last dimension.
    """
    padding_settings = {
        "causal": (pad_size, 0),
        "acausal": ((pad_size) // 2, pad_size - (pad_size) // 2),
        "anti-causal": (0, pad_size),
    }

    pad_width = (
        ((0, 0),) * axis
        + (padding_settings[predictor_causality],)
        + ((0, 0),) * (array.ndim - 1 - axis)
    )
    return jnp.pad(array, pad_width, constant_values=constant_values)


def nan_pad(
    conv_time_series: Any,
    pad_size: int,
    predictor_causality: Literal["causal", "acausal", "anti-causal"] = "causal",
    axis: int = 0,
) -> Any:
    """
    Add NaN padding to a convolved time series based on specified causality and axis.

    This function pads the convolved time series with NaNs along a specified axis. The amount
    and location of the padding are determined by the pad_size and predictor_causality parameters.

    Parameters
    ----------
    conv_time_series :
        The convolved time series to pad. This variable should be a pytree with arrays as leaves.
        The structure can be one of the following:
        1. A single array with ndim > axis.
        2. A pytree whose leaves are arrays.
    pad_size :
        The number of NaNs to concatenate as padding.
    predictor_causality : {'causal', 'acausal', 'anti-causal'}, default='causal'
        Causality of the predictor, which determines where padded values are added:
        - 'causal': Padding is added before the data.
        - 'acausal': Padding is evenly distributed before and after the data.
        - 'anti-causal': Padding is added after the data.
    axis : int, default=0
        The axis along which to add padding.

    Returns
    -------
    padded_conv_time_series : Any
        The convolved time series with NaN padding. The structure matches that of `conv_time_series`.

    Raises
    ------
    ValueError
        - If `pad_size` is not a positive integer.
        - If `predictor_causality` is not one of the expected values ('causal', 'acausal', 'anti-causal').
        - If `axis` is not a valid axis for any of the arrays in `conv_time_series`, specifically
          if `axis >= array.ndim` for any array.
        - If any array in `conv_time_series` does not have a floating-point data type.
    """
    if not isinstance(pad_size, int) or pad_size <= 0:
        raise ValueError(
            f"pad_size must be a positive integer! Pad size of {pad_size} provided instead!"
        )

    causality_choices = ["causal", "acausal", "anti-causal"]
    if predictor_causality not in causality_choices:
        raise ValueError(
            f"predictor_causality must be one of {causality_choices}. {predictor_causality} provided instead!"
        )
    if predictor_causality == "acausal" and (pad_size % 2 == 1):
        warnings.warn(
            "With acausal filter, pad_size should probably be even,"
            " so that we can place an equal number of NaNs on either side of input",
            UserWarning,
        )

    # convert to jax ndarray
    conv_time_series = jax.tree_map(jnp.asarray, conv_time_series)

    # validate the axis
    validate_axis(conv_time_series, axis)

    if is_numpy_array_like(conv_time_series):
        if not np.issubdtype(conv_time_series.dtype, np.floating):
            raise ValueError("conv_time_series must have a float dtype!")
        return _pad_dimension(
            conv_time_series,
            axis,
            pad_size,
            predictor_causality,
            constant_values=jnp.nan,
        )

    else:
        if pytree_map_and_reduce(
            lambda trial: not np.issubdtype(trial.dtype, np.floating),
            any,
            conv_time_series,
        ):
            raise ValueError("All leaves of conv_time_series must have a float dtype!")
        return jax.tree_map(
            lambda trial: _pad_dimension(
                trial, axis, pad_size, predictor_causality, constant_values=jnp.nan
            ),
            conv_time_series,
        )


def _compute_index_adjust(
    time_series: NDArray, causality: Literal["causal", "anti-causal"], axis: int
):
    """Compute index adjustment for shifting a time series."""
    adjust_indices = {
        "causal": (0, time_series.shape[axis] - 1),
        "anti-causal": (1, time_series.shape[axis]),
    }
    return adjust_indices[causality]


def shift_time_series(
    time_series: Any,
    predictor_causality: Literal["causal", "anti-causal"] = "causal",
    axis: int = 0,
):
    """Shift time series based on causality of predictor, adding NaNs as needed.

    Shift a time series based on the causality of the predictor and adds NaNs as needed,
    with the operation applied along a specified axis.

    Parameters
    ----------
    time_series :
        The time series to shift, which can be a single array or a pytree of arrays.
        Each array should have a floating-point data type.
    predictor_causality :
        Determines the direction of the shift:
        - 'causal': Shifts the series forward, inserting a NaN at the start.
        - 'anti-causal': Shifts the series backward, appending a NaN at the end.
    axis :
        The axis along which to perform the shift. Must be valid for all arrays in the time series.

    Returns
    -------
    shifted_time_series : Any
        The shifted time series. The structure matches that of `time_series`, with each element
        shifted according to `predictor_causality` and NaNs added accordingly.

    Raises
    ------
    ValueError
        - If `predictor_causality` is not 'causal' or 'anti-causal'.
        - If `axis` is invalid for any array within `time_series`.
        - If any array in `time_series` does not have a floating-point data type.

    Notes
    -----
    The direction of the shift depends on the value of `predictor_causality`:

    - `'causal'`: shift `time_series` one time bin forward and drop final time
      point, so that e.g., `[0, 1, 2]` becomes `[np.nan, 0, 1]`
    - `'anti-causal'`: shift `time_series` one time bin backwards and drop
      first time point, so that e.g., `[0, 1, 2]` becomes `[1, 2, np.nan]`

    """
    # validate axis
    validate_axis(time_series, axis)

    if predictor_causality not in ["causal", "anti-causal"]:
        raise ValueError(
            f"predictor_causality must be one of 'causal', 'anti-causal'. {predictor_causality} provided instead!"
        )

    # compute the start, end indices tree
    adjust_idx = jax.tree_map(
        lambda x: _compute_index_adjust(x, predictor_causality, axis), time_series
    )

    # convert to jax ndarray
    time_series = jax.tree_map(jnp.asarray, time_series)

    if is_numpy_array_like(time_series):

        if not np.issubdtype(time_series.dtype, np.floating):
            raise ValueError("time_series must have a float dtype!")
        return _pad_dimension(
            jnp.take(time_series, jnp.arange(*adjust_idx), axis=axis),
            axis,
            1,
            predictor_causality,
            jnp.nan,
        )
    else:
        if pytree_map_and_reduce(
            lambda trial: not np.issubdtype(trial.dtype, np.floating), any, time_series
        ):
            raise ValueError("All leaves of time_series must have a float dtype!")
        return jax.tree_map(
            lambda trial, idx: _pad_dimension(
                jnp.take(trial, jnp.arange(*idx), axis=axis),
                axis,
                1,
                predictor_causality,
                jnp.nan,
            ),
            time_series,
            adjust_idx,
        )


def plot_spike_raster(
    spike_data: Union[jnp.ndarray, np.ndarray],
    lineoffsets: Union[None, float, ArrayLike] = None,
    linelengths: Union[float, ArrayLike] = 0.2,
    linewidths: Union[float, ArrayLike] = 0.5,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """Plot decent looking spike raster.

    We set ``yticks=[]`` and ``xlim=[0, spike_data.shape[1]]``.

    Parameters
    ----------
    spike_data
        2d array of spikes, (neurons, time_bins). Should contain integers where
        spikes occur.
    lineoffsets
        The offset of the center of the lines from the origin, in the direction
        orthogonal to orientation. This can be a sequence with length matching
        the length of positions. If None, we use np.arange(spike_data.shape[0])
    linelengths
        The total height of the lines (i.e. the lines stretches from lineoffset
        - linelength/2 to lineoffset + linelength/2). This can be a sequence
        with length matching the length of positions.
    linewidths
        The line width(s) of the event lines, in points. This can be a sequence
        with length matching the length of positions.
    ax
        Axes to plot on. If None, we create a new one-axis figure with
        ``figsize=(spike_data.shape[1]/100, spike_data.shape[0]/5)``
    kwargs
        Passed to ``ax.eventplot``

    Returns
    -------
    ax
        Axis containing the raster plot

    Raises
    ------
    ValueError
        If ``spike_data.ndim!=2``

    """
    if spike_data.ndim != 2:
        raise ValueError(
            f"spike_data should be 2d, but got {spike_data.ndim}d instead!"
        )
    events = [d.nonzero()[0] for d in spike_data]
    if ax is None:
        _, ax = plt.subplots(
            1, 1, figsize=(spike_data.shape[1] / 100, spike_data.shape[0] / 5)
        )
    if lineoffsets is None:
        lineoffsets = jnp.arange(len(events))
    ax.eventplot(
        events,
        lineoffsets=lineoffsets,
        linelengths=linelengths,
        linewidths=linewidths,
        **kwargs,
    )
    ax.set(yticks=[], xlim=[0, spike_data.shape[1]])
    return ax


def row_wise_kron(A: jnp.array, C: jnp.array, jit=False, transpose=True) -> jnp.array:
    r"""Compute the row-wise Kronecker product.

    Compute the row-wise Kronecker product between two matrices using JAX.
    See [\[1\]](#references) for more details on the Kronecker product.

    Parameters
    ----------
    A : jax.numpy.ndarray
        The first matrix.
    C : jax.numpy.ndarray
        The second matrix.
    jit : bool, optional
        Activate Just-in-Time (JIT) compilation. Default is False.
    transpose : bool, optional
        Transpose matrices A and C before computation. Default is True.

    Returns
    -------
    K : jnp.nparray
        The resulting matrix with row-wise Kronecker product.

    Notes
    -----
    This function computes the row-wise Kronecker product between dense matrices A and C
    using JAX for automatic differentiation and GPU acceleration.

    References
    ----------
    1. Petersen, Kaare Brandt, and Michael Syskind Pedersen. "The matrix cookbook."
    Technical University of Denmark 7.15 (2008): 510.
    """
    if transpose:
        A = A.T
        C = C.T

    @jax.jit if jit else lambda x: x
    def row_wise_kron(a, c):
        return jnp.kron(a, c)

    K = jax.vmap(row_wise_kron)(A, C)

    if transpose:
        K = K.T

    return K


def assert_has_attribute(obj: Any, attr_name: str):
    """Ensure the object has the given attribute."""
    if not hasattr(obj, attr_name):
        raise AttributeError(
            f"The provided object does not have the required `{attr_name}` attribute!"
        )


def assert_is_callable(func: Callable, func_name: str):
    """Ensure the provided function is callable."""
    if not callable(func):
        raise TypeError(f"The `{func_name}` must be a Callable!")


def assert_returns_ndarray(
    func: Callable, inputs: Union[List[jnp.ndarray], List[float]], func_name: str
):
    """Ensure the function returns a jax.numpy.ndarray."""
    array_out = func(*inputs)
    if not isinstance(array_out, jnp.ndarray):
        raise TypeError(f"The `{func_name}` must return a jax.numpy.ndarray!")


def assert_differentiable(func: Callable, func_name: str):
    """Ensure the function is differentiable."""
    try:
        gradient_fn = jax.grad(func)
        gradient_fn(jnp.array(1.0))
    except Exception as e:
        raise TypeError(f"The `{func_name}` is not differentiable. Error: {str(e)}")


def assert_preserve_shape(
    func: Callable, inputs: List[jnp.ndarray], func_name: str, input_index: int
):
    """Check that the function preserve the input shape."""
    result = func(*inputs)
    if not result.shape == inputs[input_index].shape:
        raise ValueError(f"The `{func_name}` must preserve the input array shape!")


def assert_scalar_func(func: Callable, inputs: List[jnp.ndarray], func_name: str):
    """Check that `func` return an array containing a single scalar."""
    assert_returns_ndarray(func, inputs, func_name)
    array_out = func(*inputs)
    try:
        float(array_out)
    except TypeError:
        raise TypeError(
            f"The `{func_name}` should return a scalar! "
            f"Array of shape {array_out.shape} returned instead!"
        )


@support_pynapple(conv_type="jax")
def pynapple_concatenate(
    arrays: Union[Tsd, TsdFrame, TsdTensor, jnp.ndarray, NDArray],
    axis: int = 1,
    dtype: type = None,
):
    """Concatenation  for arrays and pynapple Tsd, TsdGroup, TsdFrame.

    Pynapple doesn't allow concatenation. With this function,
    we relax this, allowing concatenation when the time axis and support matches.

    Parameters
    ----------
    arrays:
        Sequence of ndarrays or pynapple time series with data.
        The arrays must have the same shape along all but the second axis,
        except 1-D arrays which can be any length.

    axis:
        Concatenation axis.

    dtype:
        The data type of the concatenated array.

    Returns
    -------
        The array/pynapple time series with data, stacked horizontally.
    """
    return jnp.concatenate(arrays, axis=axis, dtype=dtype)
