"""Decorator for array casting.

This module provides utilities for seamless type casting between JAX, NumPy, and pynapple time series with data
objects.

It includes functions to check if objects are numpy array-like or pynapple time series data (TSD) objects, verify
consistency in time axes and supports among TSD objects, and perform conditional conversion to JAX arrays.
The key feature for this module is the `cast_jax` decorator, which automatically casts numpy-array like inputs
to JAX arrays and, where applicable, converts outputs back to pynapple TSD objects.
"""

from functools import wraps
from typing import Any, Callable, List, Literal, Optional, Type, Union

import jax
import jax.numpy as jnp
import numpy as np
import pynapple as nap
from numpy.typing import NDArray

from . import tree_utils

_NAP_TIME_PRECISION = 10 ** (-nap.nap_config.time_index_precision)


def is_numpy_array_like(obj) -> bool:
    """
    Check if an object is array-like of at least 1-dimension.

    This function determines if an object has array-like properties but isn't an _BaseTsd.
    An object is considered array-like if it has attributes typically associated with arrays
    (such as `.shape`, `.dtype`, and `.ndim`), supports indexing, and is iterable.

    Parameters
    ----------
    obj : object
        The object to check for array-like properties.

    Returns
    -------
    :
        True if the object is array-like, False otherwise.

    Notes
    -----
    This function uses a combination of checks for attributes (`shape`, `dtype`, `ndim`),
    indexability, and iterability to determine if the given object behaves like an array.
    It is designed to be flexible and work with various types of array-like objects, including
    but not limited to NumPy arrays and JAX arrays. However, it may not be foolproof for all
    possible array-like types or objects that mimic these properties without being suitable for
    numerical operations.

    """
    # if pandas check obj.value
    obj = (
        obj.values
        if all(hasattr(obj, name) for name in ("values", "index"))
        and not is_pynapple_tsd(obj)
        else obj
    )

    # Check for array-like attributes
    has_shape = hasattr(obj, "shape")
    has_dtype = hasattr(obj, "dtype")
    has_ndim = hasattr(obj, "ndim")

    # Check for indexability (try to access the first element)
    try:
        obj[0]
        is_indexable = True
    except (TypeError, IndexError):
        is_indexable = False

    # Check for iterable property
    try:
        iter(obj)
        is_iterable = True
    except TypeError:
        is_iterable = False

    return has_shape and has_dtype and has_ndim and is_indexable and is_iterable


def is_pynapple_tsd(x: Any) -> bool:
    """
    Verify if an object is a pynapple time series with data object.

    Examines the presence of specific attributes that are characteristic of pynapple time series data structures.

    Parameters
    ----------
    x
        Object to evaluate.

    Returns
    -------
    :
        Indicates the result of the evaluation.
    """
    return all(hasattr(x, attr) for attr in ["times", "data", "time_support"])


def _has_same_time_axis(*args, **kwargs) -> bool:
    """
    Check for matching time axes among pynapple objects.

    Evaluates whether the time axes of provided pynapple objects are close enough to be considered identical,
    using a predefined precision level.

    Parameters
    ----------
    args:
        Any positional argument.
    kwargs:
        Any keyword argument.


    Returns
    -------
    :
    Indicates whether all time axes match.
    """
    flat_tree, _ = jax.tree_util.tree_flatten((args, kwargs))

    if len(flat_tree) == 1:
        return True

    # get first pynapple
    is_nap = (is_pynapple_tsd(x) for x in flat_tree)
    time = [x.t for x, bl in zip(flat_tree, is_nap) if bl]

    # check time samples are close (using pynapple precision)
    return _check_all_close(time)


def _has_same_support(*args, **kwargs):
    """
    Verify matching time support intervals among pynapple objects.

    Evaluates whether the time support intervals of provided pynapple objects are close enough to be
    considered identical, using a predefined precision level.

    Parameters
    ----------
    args:
        Any positional argument.
    kwargs:
        Any keyword argument.

    Returns
    -------
    :
        Indicates the result of the verification.
    """
    flat_tree, _ = jax.tree_util.tree_flatten((args, kwargs))

    if len(flat_tree) == 1:
        return True

    # get first pynapple
    is_nap = (is_pynapple_tsd(x) for x in flat_tree)
    time_support = [x.time_support.values for x, bl in zip(flat_tree, is_nap) if bl]

    # check starts and ends are close (using pynapple precision)
    return _check_all_close(time_support)


def _check_all_close(arrays: List[NDArray]) -> bool:
    """
    Check that equality of two arrays up to numerical precision.

    Parameters
    ----------
    arrays:
        List of arrays to compare

    Returns
    -------
    :
        True if the array are equal up to numerical precision, False otherwise.

    """
    if not all(arrays[0].ndim == arr.ndim for arr in arrays[1:]):
        return False

    elif not all(arrays[0].shape == arr.shape for arr in arrays[1:]):
        return False

    return all(
        jnp.allclose(
            arrays[0],
            x,
            rtol=0,
            atol=_NAP_TIME_PRECISION,
        )
        for x in arrays[1:]
    )


def all_same_time_info(*args, **kwargs) -> bool:
    """
    Ensure consistent time axis and support information among pynapple objects.

    Combines checks for matching time axes and support intervals to verify consistency across all provided pynapple
    objects.

    Parameters
    ----------
    args:
        Any positional argument.
    kwargs:
        Any keyword argument.

    Returns
    -------
    :
        Indicates whether time-related information is consistent.
    """
    return _has_same_time_axis(*args, **kwargs) and _has_same_support(*args, **kwargs)


def get_time_info(*args, **kwargs):
    """
    Extract time axis and support information from the first pynapple object.

    Assumes the presence of at least one pynapple object among the inputs and retrieves its time-related information.

    Parameters
    ----------
    args:
        Any positional argument.
    kwargs:
        Any keyword argument.

    Returns
    -------
    :
        Time axis and support information of the first pynapple object detected.
    """
    # list of bool
    flat, _ = jax.tree_util.tree_flatten(
        jax.tree_util.tree_map(is_pynapple_tsd, (args, kwargs))
    )

    idx = flat.index(True)

    # get the corresponding tsd
    tsd, _ = jax.tree_util.tree_flatten((args, kwargs))

    return tsd[idx].t, tsd[idx].time_support


def cast_to_pynapple(
    array: jnp.ndarray, time: NDArray, time_support: nap.IntervalSet
) -> Union[nap.Tsd, nap.TsdFrame, nap.TsdTensor]:
    """
    Convert an array to a pynapple time series object.

    Depending on the dimensionality of the array, creates an appropriate pynapple time series data structure using
    the provided time and support information.

    Parameters
    ----------
    array:
        Data array to convert.
    time:
        Time axis for the pynapple object.
    time_support:
        Time support information for the pynapple object.

    Returns
    -------
    :
    A pynapple time series object based on the input array.
    """
    # keep time on CPU, pynapple numba operations on time are more efficient
    time = np.asarray(time)
    if time.shape[0] != array.shape[0]:
        return array
    elif array.ndim == 1:
        return nap.Tsd(t=time, d=array, time_support=time_support)
    elif array.ndim == 2:
        return nap.TsdFrame(t=time, d=array, time_support=time_support)
    else:
        return nap.TsdTensor(t=time, d=array, time_support=time_support)


def jnp_asarray_if(
    x: Any,
    condition: Callable[[Any], bool] = is_numpy_array_like,
    dtype: Optional[Type] = None,
) -> Any:
    """
    Conditionally convert an object to a JAX array.

    Applies the conversion if the specified condition is met. Allows for flexible handling of inputs that should
    be treated as arrays for numerical computations.

    Parameters
    ----------
    x:
        Object to potentially convert.
    condition:
        A callable that determines whether conversion should occur.
    dtype:
        dtype for the conversion.

    Returns
    -------
    :
        The original object or its conversion to a JAX array, based on the condition.
    """
    if condition(x):
        x = jnp.asarray(x, dtype=dtype)
    return x


def np_asarray_if(
    x: Any, condition: Callable[[Any], bool] = is_numpy_array_like
) -> Any:
    """
    Conditionally convert an object to a numpy array.

    Applies the conversion if the specified condition is met. Allows for flexible handling of inputs that should
    be treated as arrays for numerical computations.

    Parameters
    ----------
    x:
        Object to potentially convert.
    condition:
        A callable that determines whether conversion should occur.

    Returns
    -------
    :
        The original object or its conversion to a numpy array, based on the condition.
    """
    if condition(x):
        x = np.asarray(x)
    return x


def support_pynapple(conv_type: Literal["jax", "numpy"] = "jax") -> Callable:
    """
    Decorate a function to cast inputs between JAX arrays and pynapple objects.

    Automatically converts input arguments to JAX arrays for processing. If the original inputs include pynapple
    objects, attempts to convert the function's output back to pynapple format while ensuring consistency in the
    time axis.

    Parameters
    ----------
    conv_type
        The type of conversion. Either "numpy" or "jax".

    Returns
    -------
    :
        A wrapper function that applies the specified casting behavior.

    Raises
    ------
    NotImplementedError:
        If the conversion type is not implemented.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # check for the presence of any pynapple tsd/tsdFrame/tsdTensor
            any_nap = tree_utils.pytree_map_and_reduce(
                is_pynapple_tsd, any, (args, kwargs)
            )

            # type casting pynapple
            if any_nap:
                # check if the time axis is the same
                if not all_same_time_info(*args, **kwargs):
                    raise ValueError(
                        "Time axis mismatch. pynapple objects have mismatching time axis."
                    )
                time, time_support = get_time_info(*args, **kwargs)

                def cast_out(tree):
                    # cast back to pynapple
                    return jax.tree_util.tree_map(
                        lambda x: cast_to_pynapple(x, time, time_support), tree
                    )

            else:
                # if no pynapple time series is present, apply the function/method
                return func(*args, **kwargs)

            if conv_type == "jax":
                # cast to jax
                args, kwargs = jax.tree_util.tree_map(jnp_asarray_if, (args, kwargs))
            elif conv_type == "numpy":
                # cast to numpy
                args, kwargs = jax.tree_util.tree_map(np_asarray_if, (args, kwargs))
            else:
                raise NotImplementedError(
                    f"Conversion of type '{conv_type}' not implemented!"
                )
            # apply function/method
            res = func(*args, **kwargs)
            # revert casting if pynapple
            return cast_out(res)

        return wrapper

    return decorator
