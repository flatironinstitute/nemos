"""Utility functions for data pre-processing."""
# required to get ArrayLike to render correctly, unnecessary as of python 3.10
from __future__ import annotations

from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    List,
    Literal,
    Optional,
    Union,
)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray

# to avoid circular imports
if TYPE_CHECKING:
    from .pytrees import FeaturePytree

# Same trial duration
# [[r , t , n], [w]] -> [r , (t - w + 1) , n]
# Broadcasted 1d convolution operations
_CORR1 = jax.vmap(partial(jnp.convolve, mode="valid"), (0, None), 0)
_CORR2 = jax.vmap(_CORR1, (2, None), 2)
_CORR_SAME_TRIAL_DUR = jax.vmap(_CORR2, (None, 1), 3)

# Variable trial dur
# [[n x t],[p x w]] -> [n x p x (t - w + 1)]
_CORR3 = jax.vmap(partial(jnp.convolve, mode="valid"), (1, None), 1)
_CORR_VARIABLE_TRIAL_DUR = jax.vmap(_CORR3, (None, 1), 2)


def check_dimensionality(
    iterable: Union[NDArray, Iterable[NDArray], jnp.ndarray, Iterable[jnp.ndarray]],
    expected_dim: int,
) -> bool:
    """
    Check the dimensionality of the arrays in iterable.

    Check that all arrays in iterable has the expected dimensionality.

    Parameters
    ----------
    iterable :
        Array-like object containing numpy or jax.numpy NDArrays.
    expected_dim :
        Number of expected dimension for the NDArrays.

    Returns
    -------
    True if all the arrays has the expected number of dimension, False otherwise.
    """
    return not any(array.ndim != expected_dim for array in iterable)


def convolve_1d_trials(
    basis_matrix: ArrayLike,
    time_series: Union[Iterable[NDArray], NDArray, Iterable[jnp.ndarray], jnp.ndarray],
) -> Union[Iterable[NDArray], NDArray, Iterable[jnp.ndarray], jnp.ndarray]:
    """Convolve trial time series with a basis matrix.

    This function checks if all trials have the same duration. If they do, it uses a fast method
    to convolve all trials with the basis matrix at once. If they do not, it falls back to convolving
    each trial individually.

    Parameters
    ----------
    basis_matrix :
        The basis matrix with which to convolve the trials. Shape (window_size, n_basis_funcs).
    time_series :
        The time series of trials to convolve with the basis matrix. It should be a list of 2D arrays,
        where each array represents a trial and its second dimension matches the first dimension
        of the basis_matrix. Each trial has shape (n_timebins_trial, n_neurons).

    Returns
    -------
    :
        The convolved trials. It is a list of 3D arrays, where each array represents a convolved trial.
        Each element of the list will have shape (n_timebins_trial - window_size - 1, n_neurons, n_basis_funcs).

    Raises
    ------
    ValueError
        - If basis_matrix is not a 2D array-like object.
        - If trials_time_series is not an iterable of 2D array-like objects.
        - If trials_time_series contains empty trials.
        - If the number of time points in each trial is less than the window size.
    """
    basis_matrix = jnp.asarray(basis_matrix)
    # check input size
    if basis_matrix.ndim != 2:
        raise ValueError("basis_matrix must be a 2 dimensional array-like object.")
    elif 0 in basis_matrix.shape:
        raise ValueError(
            "Empty basis_matrix provided. "
            f"The shape of basis_matrix is {basis_matrix.shape}!"
        )
    try:
        # this should fail for variable trial length
        time_series = jnp.asarray(time_series)
        if time_series.ndim != 3:
            raise ValueError

    except ValueError:
        # convert each trial to array
        time_series = jax.tree_map(jnp.asarray, time_series)
        if not check_dimensionality(time_series, 2):
            raise ValueError(
                "time_series must be an iterable of 2 dimensional array-like objects."
            )

    if any(k == 0 for trial in time_series for k in trial.shape) or (
        len(time_series) == 0
    ):
        raise ValueError("time_series should not contain empty trials!")

    # Check window size
    ws = len(basis_matrix)
    if pytree_map_and_reduce(lambda x: x.shape[0] < ws, any, list(time_series)):
        raise ValueError(
            "Insufficient trial duration. The number of time points in each trial must "
            "be greater or equal to the window size."
        )

    if isinstance(time_series, jnp.ndarray):
        # if the conversion to array went through, time_series have trials with equal size
        conv_trials = _CORR_SAME_TRIAL_DUR(time_series, basis_matrix)
    else:
        # trials have different length
        conv_trials = jax.tree_map(
            lambda x: _CORR_VARIABLE_TRIAL_DUR(jnp.atleast_2d(x), basis_matrix),
            time_series
        )

    return conv_trials


def _pad_dimension(
    array: jnp.ndarray,
    axis: int,
    window_size: int,
    filter_type: Literal["causal", "acausal", "anti-causal"] = "causal",
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
    window_size:
        The window size to determine the padding.
    filter_type:
        The type of convolution.
    constant_values:
        The constant values for padding, default is jnp.nan.

    Returns
    -------
    :
        An array with padded last dimension.
    """
    if axis < 0 or not isinstance(axis, int):
        raise ValueError("`axis` must be a non negative integer.")
    elif axis >= array.ndim:
        raise IndexError(
            "`axis` must be smaller than `array.ndim`. "
            f"array.ndim is {array.ndim}, axis = {axis} provided!"
        )

    padding_settings = {
        "causal": (window_size, 0),
        "acausal": ((window_size - 1) // 2, window_size - 1 - (window_size - 1) // 2),
        "anti-causal": (0, window_size),
    }

    if filter_type not in padding_settings:
        cases_string = " or ".join(padding_settings).replace(
            " or ", ", ", len(padding_settings) - 2
        )
        raise ValueError(
            f"filter_type must be {cases_string}. {filter_type} provided instead!"
        )

    pad_width = (
        ((0, 0),) * axis
        + (padding_settings[filter_type],)
        + ((0, 0),) * (array.ndim - 1 - axis)
    )
    return jnp.pad(array, pad_width, constant_values=constant_values)


def nan_pad_conv(
    conv_trials: Union[Iterable[jnp.ndarray], Iterable[NDArray], NDArray, jnp.ndarray],
    window_size: int,
    filter_type: Literal["causal", "acausal", "anti-causal"] = "causal",
) -> List[jnp.ndarray]:
    """
    Add NaN padding to convolution trials based on the convolution type.

    Parameters
    ----------
    conv_trials:
        A 4D array-like of trials to be padded. Each trial has shape (n_neurons, n_basis_funcs, n_timebins_trial).
    window_size:
        The window size to determine the padding.
    filter_type: str, optional
        The type of convolution.

    Returns
    -------
    :
        A 4D array-like of padded trials. Each trial has shape (n_neurons, n_basis_funcs, n_timebins_trial + padding).

    Raises
    ------
    ValueError
        If the window_size is not a positive integer, or if the filter_type is not one of 'causal',
        'acausal', or 'anti-causal'. Also raises ValueError if the dimensionality of conv_trials is not as expected.

    """
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError(
            f"window_size must be a positive integer! Window size of {window_size} provided instead!"
        )
    # add a note about this in docstrings
    adjust_indices = {
        "causal": (None, -1),
        "acausal": (None, None),
        "anti-causal": (1, None),
    }

    if filter_type not in adjust_indices:
        cases_string = " or ".join(adjust_indices).replace(
            " or ", ", ", len(adjust_indices) - 2
        )
        raise ValueError(
            f"filter_type must be {cases_string}. {filter_type} provided instead!"
        )

    start, end = adjust_indices[filter_type]

    try:
        conv_trials = jnp.asarray(conv_trials)
        if conv_trials.ndim != 4:
            raise ValueError(
                "conv_trials must be an iterable of 3D arrays or a 4D array!"
            )

        conv_trials = conv_trials[:, start:end]
        return list(
            _pad_dimension(
                conv_trials, 1, window_size, filter_type, constant_values=jnp.nan
            )
        )

    except (TypeError, ValueError):
        if not check_dimensionality(conv_trials, 3):
            raise ValueError(
                "conv_trials must be an iterable of 3D arrays or a 4D array!"
            )

        return [
            _pad_dimension(
                trial[start:end],
                0,
                window_size,
                filter_type,
                constant_values=jnp.nan,
            )
            for trial in conv_trials
        ]


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

    @ jax.jit if jit else lambda x: x
    def row_wise_kron(a, c):
        return jnp.kron(a, c)

    K = jax.vmap(row_wise_kron)(A, C)

    if transpose:
        K = K.T

    return K


def check_invalid_entry(
    pytree: Union[FeaturePytree, jnp.ndarray], pytree_name: str
) -> None:
    """Check if the array has nans or infs.

    Parameters
    ----------
    pytree:
        The pytree to be checked.
    pytree_name:
        The array name.

    Raises
    ------
        - ValueError: If any entry of `pytree` is either NaN or inf.

    """
    # this is a double-any situation: first, we check if each leaf have any
    # Infs, then we check whether any leaf has an Inf
    if pytree_map_and_reduce(jnp.any, any, jax.tree_map(jnp.isinf, pytree)):
        raise ValueError(f"Input '{pytree_name}' contains Infs!")
    if pytree_map_and_reduce(jnp.any, any, jax.tree_map(jnp.isnan, pytree)):
        raise ValueError(f"Input '{pytree_name}' contains NaNs!")


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


def pytree_map_and_reduce(
    map_fn: Callable,
    reduce_fn: Callable,
    *pytrees: Union[FeaturePytree, NDArray, jnp.ndarray],
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
    cond_tree = jax.tree_map(map_fn, *pytrees)
    # for some reason, tree_reduce doesn't work well with any.
    return reduce_fn(jax.tree_util.tree_leaves(cond_tree))
