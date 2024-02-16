"""
Type casting decorator.
"""
from typing import Callable, Any
from functools import wraps
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from . import utils
import pynapple as nap


def is_array_like(obj):
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
    bool
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

    return (
        has_shape
        and has_dtype
        and has_ndim
        and is_indexable
        and is_iterable
    )

def is_pynapple_tsd(x):
    return all(hasattr(x, attr) for attr in ["times", "data"])


def _has_same_time_axis(*args, **kwargs):
    """Check that the time axis matches using pynapple precision.
    """

    flat_tree, _ = jax.tree_util.tree_flatten((args, kwargs))
    if len(flat_tree) == 1:
        return True

    # get first pynapple
    is_nap = list(is_pynapple_tsd(x) for x in flat_tree)
    first_nap = is_nap.index(True)

    # check time samples are close (10**-9 is hard-coded in pynapple)
    return all(
        jnp.allclose(flat_tree[first_nap].t, x.t, rtol=0, atol=10 ** -9)
        for i, x in enumerate(flat_tree) if is_nap[i] and i != first_nap
    )


def _has_same_support(*args, **kwargs):
    """Check that the time axis matches using pynapple precision.
    """
    flat_tree, _ = jax.tree_util.tree_flatten((args, kwargs))

    if len(flat_tree) == 1:
        return True

    # get first pynapple
    is_nap = list(is_pynapple_tsd(x) for x in flat_tree)
    first_nap = is_nap.index(True)

    # check starts and ends are close (10**-9 is hard-coded in pynapple)
    bool_support = all(
        jnp.allclose(flat_tree[first_nap].time_support.values, x.time_support.values, rtol=0, atol=10 ** -9)
        for i, x in enumerate(flat_tree) if is_nap[i] and i != first_nap
    )
    return bool_support


def all_same_time_info(*args, **kwargs):
    return _has_same_time_axis(*args, **kwargs) and _has_same_support(*args, **kwargs)


def _get_time_info(*args, **kwargs):
    """Get the time axis from the first pynapple object.

    This function assumes that there is at least one nap object
    """
    # list of bool
    flat, _ = jax.tree_util.tree_flatten(jax.tree_map(is_pynapple_tsd, (args, kwargs)))

    idx = flat.index(True)

    # get the corresponding tsd
    tsd, _ = jax.tree_util.tree_flatten((args, kwargs))

    return tsd[idx].t, tsd[idx].time_support


def cast_to_pynapple(array: jnp.ndarray, time: NDArray, time_support: nap.IntervalSet):
    time = np.asarray(time)
    array = np.asarray(array)
    if array.ndim == 1:
        return nap.Tsd(t=time, d=array, time_support=time_support)
    elif array.ndim == 2:
        return nap.TsdFrame(t=time, d=array, time_support=time_support)
    else:
        return nap.TsdTensor(t=time, d=array, time_support=time_support)


def jnp_asarray_if(x: Any, condition: Callable[[Any], bool] = is_array_like):
    if condition(x):
        x = jnp.asarray(x)
    return x


def cast_jax(func):
    """
    Decorator for casting between jax and pynapple.

    This decorator casts the input to func to jax arrays. If any pynapple Tsd/TsdFrame/TsdTensor is present,
    it will cast back the output of func to pynapple object. If time_math=False, and input to func are
    multiple pynapple object, it will use the time axis of the first pynapple object detected.


    Parameters
    ----------
    func
    time_match

    Returns
    -------

    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # check for the presence of any pynapple tsd/tsdFrame/tsdTensor
        any_nap = utils.pytree_map_and_reduce(is_pynapple_tsd, any, (args, kwargs))

        # type casting pynapple
        if any_nap:
            # check if the time axis is the same
            if not all_same_time_info(*args, **kwargs):
                raise ValueError("Time axis mismatch. pynapple objects have mismatching time axis.")
            time, support = _get_time_info(*args, **kwargs)
            cast_out = lambda tree: jax.tree_map(lambda x: cast_to_pynapple(x, time, support), tree)
        else:
            cast_out = lambda x: x

        args, kwargs = jax.tree_map(jnp_asarray_if, (args, kwargs))
        res = func(*args, **kwargs)
        return cast_out(res)

    return wrapper


