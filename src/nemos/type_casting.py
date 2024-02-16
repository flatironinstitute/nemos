"""Decorator for array casting.

This module provides utilities for seamless type casting between JAX, NumPy, and pynapple time series with data
objects.

It includes functions to check if objects are numpy array-like or pynapple time series data (TSD) objects, verify
consistency in time axes and supports among TSD objects, and perform conditional conversion to JAX arrays.
The key feature for this module is the `cast_jax` decorator, which automatically casts numpy-array like inputs
to JAX arrays and, where applicable, converts outputs back to pynapple TSD objects.
"""

from functools import wraps
from typing import Any, Callable, Union

import jax
import jax.numpy as jnp
import numpy as np
import pynapple as nap
from numpy.typing import NDArray

from . import utils


def is_numpy_array_like(obj) -> bool:
    """
    Check if an object is array-like.

    This function determines if an object has array-like properties but isn't an _AstractTsd.
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
    return all(hasattr(x, attr) for attr in ["times", "data"])


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
    is_nap = list(is_pynapple_tsd(x) for x in flat_tree)
    first_nap = is_nap.index(True)

    # check time samples are close (10**-9 is hard-coded in pynapple)
    return all(
        jnp.allclose(flat_tree[first_nap].t, x.t, rtol=0, atol=10**-9)
        for i, x in enumerate(flat_tree)
        if is_nap[i] and i != first_nap
    )


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
    is_nap = list(is_pynapple_tsd(x) for x in flat_tree)
    first_nap = is_nap.index(True)

    # check starts and ends are close (10**-9 is hard-coded in pynapple)
    bool_support = all(
        jnp.allclose(
            flat_tree[first_nap].time_support.values,
            x.time_support.values,
            rtol=0,
            atol=10**-9,
        )
        for i, x in enumerate(flat_tree)
        if is_nap[i] and i != first_nap
    )
    return bool_support


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


def _get_time_info(*args, **kwargs):
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
    flat, _ = jax.tree_util.tree_flatten(jax.tree_map(is_pynapple_tsd, (args, kwargs)))

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
    time = np.asarray(time)
    array = np.asarray(array)
    if array.ndim == 1:
        return nap.Tsd(t=time, d=array, time_support=time_support)
    elif array.ndim == 2:
        return nap.TsdFrame(t=time, d=array, time_support=time_support)
    else:
        return nap.TsdTensor(t=time, d=array, time_support=time_support)


def jnp_asarray_if(
    x: Any, condition: Callable[[Any], bool] = is_numpy_array_like
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

    Returns
    -------
    :
        The original object or its conversion to a JAX array, based on the condition.
    """
    if condition(x):
        x = jnp.asarray(x)
    return x


def cast_jax(func: Callable) -> Callable:
    """
    Decorate a function to cast inputs between JAX arrays and pynapple objects.

    Automatically converts input arguments to JAX arrays for processing. If the original inputs include pynapple
    objects, attempts to convert the function's output back to pynapple format while ensuring consistency in the
    time axis.

    Parameters
    ----------
    func
        The function to decorate.

    Returns
    -------
    :
        A wrapper function that applies the specified casting behavior.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # check for the presence of any pynapple tsd/tsdFrame/tsdTensor
        any_nap = utils.pytree_map_and_reduce(is_pynapple_tsd, any, (args, kwargs))

        # type casting pynapple
        if any_nap:
            # check if the time axis is the same
            if not all_same_time_info(*args, **kwargs):
                raise ValueError(
                    "Time axis mismatch. pynapple objects have mismatching time axis."
                )
            time, support = _get_time_info(*args, **kwargs)

            def cast_out(tree):
                return jax.tree_map(lambda x: cast_to_pynapple(x, time, support), tree)

        else:

            def cast_out(tree):
                return tree

        args, kwargs = jax.tree_map(jnp_asarray_if, (args, kwargs))
        res = func(*args, **kwargs)
        return cast_out(res)

    return wrapper
