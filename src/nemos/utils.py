"""Utility functions for data pre-processing."""

import inspect
import os
import warnings
from importlib.metadata import version
from typing import Any, Callable, List, Literal, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from .base_class import Base
from .tree_utils import pytree_map_and_reduce
from .type_casting import support_pynapple
from .typing import Pytree

__all__ = [
    "check_dimensionality",
    "validate_axis",
    "nan_pad",
    "shift_time_series",
    "row_wise_kron",
    "one_over_x",
    "get_flattener_unflattener",
]


def __dir__() -> list[str]:
    return __all__


SPECIAL_KEY_NAMES = {
    jax.scipy.stats.norm.cdf: "norm.cdf",
}


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


def validate_axis(pytree: Any, axis: int):
    """
    Validate the axis for each array in a given tree structure.

    This function checks if the specified axis exists in each array within the tree. It raises a ValueError
    if the specified axis is equal to or greater than the number of dimensions in any of the arrays.

    Parameters
    ----------
    pytree :
        A tree containing arrays.
    axis :
        The axis that should be valid for each array in the tree. This means each array must have at least
        `axis + 1` dimensions.

    Raises
    ------
    ValueError
        If the specified axis is equal to or greater than the number of dimensions (`ndim`) of any array
        within the tree. This ensures that operations intended for a specific axis can be safely performed
        on every array in the tree.
    ValueError
        If the axis is negative or non-integer.
    """
    if not isinstance(axis, int) or axis < 0:
        raise ValueError("`axis` must be a non negative integer.")

    if pytree_map_and_reduce(lambda x: x.ndim <= axis, any, pytree):
        raise ValueError(
            "'axis' must be smaller than the number of dimensions of any array in 'tree'."
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
    # convert negative axis in jax jit compilable way
    axis = axis * (axis >= 0) + (array.ndim + axis) * (axis < 0)

    pad_width = (
        ((0, 0),) * axis
        + (padding_settings[predictor_causality],)
        + ((0, 0),) * (array.ndim - 1 - axis)
    )
    return jnp.pad(array, pad_width, constant_values=constant_values)


def nan_pad(
    conv_time_series: NDArray | jnp.ndarray,
    pad_size: int,
    predictor_causality: Literal["causal", "acausal", "anti-causal"] = "causal",
    axis: int = 0,
) -> jnp.ndarray:
    """
    Add NaN padding to a convolved time series based on specified causality and axis.

    This function pads the convolved time series with NaNs along a specified axis. The amount
    and location of the padding are determined by the pad_size and predictor_causality parameters.

    Parameters
    ----------
    conv_time_series :
        The convolved time series to pad. This variable should be an array
        with ndim > axis.
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
    padded_conv_time_series :
        The convolved time series with NaN padding.

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
    conv_time_series = jnp.asarray(conv_time_series)

    # validate the axis
    validate_axis(conv_time_series, axis)

    if not np.issubdtype(conv_time_series.dtype, np.floating):
        raise ValueError("All leaves of conv_time_series must have a float dtype!")
    return _pad_dimension(
        conv_time_series, axis, pad_size, predictor_causality, constant_values=jnp.nan
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
    time_series: NDArray | jnp.ndarray,
    predictor_causality: Literal["causal", "anti-causal"] = "causal",
    axis: int = 0,
):
    """Shift time series based on causality of predictor, adding NaNs as needed.

    Shift a time series based on the causality of the predictor and adds NaNs as needed,
    with the operation applied along a specified axis.

    Parameters
    ----------
    time_series :
        The time series to shift, must be a single array and should have a floating-point
        data type.
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
    adjust_idx = _compute_index_adjust(time_series, predictor_causality, axis)

    # convert to jax ndarray
    time_series = jnp.asarray(time_series)
    if not np.issubdtype(time_series.dtype, np.floating):
        raise ValueError("time_series must have a float dtype!")
    return _pad_dimension(
        jnp.take(time_series, jnp.arange(*adjust_idx), axis=axis),
        axis,
        1,
        predictor_causality,
        jnp.nan,
    )


def row_wise_kron(
    A: jnp.ndarray, C: jnp.ndarray, jit: bool = False, transpose: bool = True
) -> jnp.ndarray:
    r"""Compute the row-wise Kronecker product.

    Compute the row-wise Kronecker product between two matrices using JAX.
    See [\[1\]](#references) for more details on the Kronecker product.

    Parameters
    ----------
    A : jax.numpy.ndarray
        The first matrix.
    C : jax.numpy.ndarray
        The second matrix.
    jit :
        Activate Just-in-Time (JIT) compilation. Default is False.
    transpose :
        Transpose matrices A and C before computation. Default is True.

    Returns
    -------
    K : jnp.nparray
        The resulting matrix with row-wise Kronecker product.

    Notes
    -----
    This function computes the row-wise Kronecker product between dense matrices A and C
    using JAX for automatic differentiation and GPU acceleration.

    # References
    ------------
    [1] Petersen, Kaare Brandt, and Michael Syskind Pedersen. "The matrix cookbook."
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


def format_repr(
    obj: Base,
    exclude_keys: Optional[List[str]] = None,
    use_name_keys: Optional[List[str]] = None,
    multiline=False,
):
    """
    Format the representation string of an object (`__repr__`).

    This function generates a string representation of an object, including
    the parameters returned by `obj.get_params(deep=False)`. The output excludes
    specified keys and formats certain keys using their class names instead of
    their default string representations.

    Parameters
    ----------
    obj :
        The object inheriting from  Base whose representation string is being formatted.
    exclude_keys :
        List of parameter keys to exclude from the representation. If `None`, no
        keys are excluded. Defaults to `None`.
    use_name_keys :
        List of keys for which the value's `__name__` attribute is used instead
        of the default `__repr__` output. Defaults to an empty list.
    multiline:
        If True, add each parameter on a new line, if False (default), list all parameters
        in a single line.

    Returns
    -------
    :
        A formatted representation string for the object, including its class
        name and relevant parameters.

    Notes
    -----
    - The function includes only non-empty or falsey values (except 0 and False) in the
      representation. The falsey values include empty list or dict, or similar.
    - Parameters are displayed in the order defined in the `obj.__init__`
      method.
    - Parameters with a `shape` attribute (arrays) are excluded from the representation.

    Examples
    --------
    >>> import jax
    >>> from nemos.base_class import Base
    >>> from nemos.regularizer import Ridge
    >>> from nemos.utils import format_repr
    >>> class Example(Base):
    ...     def __init__(self, a, b, c, d):
    ...         self.a = a
    ...         self.b = b
    ...         self.c = c
    ...         self.d = d
    >>> obj = Example(1, jax.numpy.exp, c="hi", d=None)
    >>> format_repr(obj, exclude_keys=["c"], use_name_keys=["b"])
    'Example(a=1, b=exp)'
    """
    exclude_keys = [] if exclude_keys is None else exclude_keys
    use_name_keys = [] if use_name_keys is None else use_name_keys

    init_params = list(inspect.signature(obj.__init__).parameters.keys())
    disp_params = []
    # use special method for basis
    all_params = obj.get_params(deep=False)
    label = all_params.pop("label", None)
    for k, v in all_params.items():
        repr_param = (
            k not in exclude_keys and not hasattr(v, "shape") and (v or v in (0, False))
        )
        if repr_param:
            if k in use_name_keys:
                if v in SPECIAL_KEY_NAMES:
                    v = SPECIAL_KEY_NAMES[v]
                else:
                    v = getattr(v, "__name__", repr(v))
            elif isinstance(v, str):
                v = repr(v)
            disp_params.append(f"{k}={v}")
    disp_params = sorted(disp_params, key=lambda x: init_params.index(x.split("=")[0]))
    cls_name = obj.__class__.__name__
    # if label doesn't exist or is the same as the class name (as is the default for
    # basis), then don't use it
    disp_label = (label is not None) and (label != cls_name)
    if multiline:
        tab = "    "
        disp_params = "\n" + tab + f",\n{tab}".join(disp_params) + "\n" + tab[:-4]
        repr_str = (
            f"{repr(label)}: {cls_name}({disp_params})"
            if disp_label
            else f"{cls_name}({disp_params})"
        )
    else:
        disp_params = ", ".join(disp_params)
        repr_str = (
            f"{repr(label)}: {cls_name}({disp_params})"
            if disp_label
            else f"{cls_name}({disp_params})"
        )
    return repr_str


# enable concatenation for pynapple objects.
pynapple_concatenate_jax = support_pynapple(conv_type="jax")(jnp.concatenate)
pynapple_concatenate_numpy = support_pynapple(conv_type="numpy")(np.concatenate)


def _get_terminal_size():
    """Get the terminal size for __repr__.

    Returns
    -------
    rows:
        Number of rows in the terminal window.
    cols:
        Number of columns in the terminal window.

    """
    cols = 100  # Default
    rows = 2
    try:
        cols, rows = os.get_terminal_size()
    except Exception:
        import shutil

        cols, rows = shutil.get_terminal_size()

    return cols, rows


def one_over_x(x: NDArray):
    """Implement 1/x."""
    return jnp.power(x, -1)


def _flatten_dict(nested_dict: dict, parent_key: str = "") -> dict:
    """
    Flatten a nested dictionary into a single-level dictionary with keys representing the hierarchy.

    Parameters
    ----------
    nested_dict :
        The dictionary to flatten.
    parent_key :
        This key starts blank, but recursively it will be filled with the parent key,
        which is used to create the hierarchy in the flattened dictionary.

    Returns
    -------
    dict :
        A flattened dictionary where the hierarchy is represented by concatenated keys (using __ as a separator).
    """

    sep = "__"
    items = []
    # Iterate over key-value pairs in the dictionary
    for k, v in nested_dict.items():
        # containers (tuple, dict, list for our purposes are labeled)
        if isinstance(v, dict):
            container_label = "dict"
        elif isinstance(v, list):
            container_label = "list"
        elif isinstance(v, tuple):
            container_label = "tuple"
        else:
            container_label = "item"

        new_key = (
            f"{parent_key}{sep}{container_label}{sep}{k}"
            if parent_key
            else f"{container_label}{sep}{k}"
        )

        # Recursively flatten if the value is a dictionary
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key).items())
        # if tuple or list, convert to dict using item index and recursively flatten
        elif isinstance(v, (list, tuple)):
            items.extend(
                _flatten_dict({f"{i}": vi for i, vi in enumerate(v)}, new_key).items()
            )
        else:
            # None values and non-standard types are converted to numpy
            if v is None:
                v = np.nan
            elif not isinstance(v, (str, int, float, bool)):
                v = np.array(v)
            items.append((new_key, v))
    return dict(items)


def _unflatten_dict(flat_dict: dict) -> dict:
    """
    Unflatten a dictionary that was flattened using _flatten_dict, and saved as npz.

    Note that all leaves are expected to be numpy arrays.

    Parameters
    ----------
    flat_dict :
        The flattened dictionary with keys containing type information and hierarchy and leaves numpy arrays.

    Returns
    -------
    dict :
        The reconstructed nested dictionary with original structure and types.
    """
    sep = "__"
    root = {}

    for key, value in flat_dict.items():
        # Replace np.nan back to None
        # Convert numpy string, int, float or nan to their respective types
        if value.ndim == 0:
            if value.dtype.type is np.str_:
                value = str(value)
            elif value.dtype.type is np.int_:
                value = int(value)
            elif issubclass(value.dtype.type, np.floating):
                value = None if np.isnan(value) else float(value)

        # split correctly `varname_` type of params
        parts = [part[::-1] for part in key[::-1].split(sep)[::-1]]

        current = root

        # Walk the path - but handle the last pair differently if it's an 'item'
        pairs = list(zip(parts[::2], parts[1::2]))

        # Navigate through all pairs except potentially the last one
        for i, (container_type, container_key) in enumerate(pairs):
            is_last = i == len(pairs) - 1

            # If this is the last pair and it's an 'item', just set the value directly
            if is_last and container_type == "item":
                current[container_key] = value
                break

            # Otherwise, create/navigate into the container
            if container_key not in current:
                current[container_key] = {"__container_type__": container_type}

            current = current[container_key]
        else:
            # This else clause runs if we didn't break (i.e., no 'item' leaf)
            # This shouldn't happen with your flatten structure, but just in case
            final_key = parts[-1]
            current[final_key] = value

    return reconstruct_object_from_structured_dict(root)


# Convert to proper types
def reconstruct_object_from_structured_dict(obj):
    if not isinstance(obj, dict):
        return obj

    container_type = obj.pop("__container_type__", None)

    # Check if numeric keys (list/tuple)
    keys = [k for k in obj.keys() if k != "__container_type__"]
    if keys and all(k.isdigit() for k in keys):
        sorted_items = sorted(obj.items(), key=lambda x: int(x[0]))
        items = [reconstruct_object_from_structured_dict(v) for k, v in sorted_items]
        return tuple(items) if container_type == "tuple" else items

    # Regular dict
    return {k: reconstruct_object_from_structured_dict(v) for k, v in obj.items()}


def _get_name(x: object) -> str:
    """
    Get the name of an object ``x``, for saving/loading purposes.

    Parameters
    ----------
    x :
        A python object or function.

    Returns
    -------
    name :
        The name of the object, with full module path (e.g.,
        ``nemos.observation_models.PoissonObservations``).
    """
    if x is None:
        return None
    if hasattr(x, "__module__") and hasattr(x, "__name__"):
        # x is a function or class
        return f"{x.__module__}.{x.__name__}"
    elif hasattr(x, "__class__"):
        # x is an instance of a class
        cls = x.__class__
        return f"{cls.__module__}.{cls.__name__}"
    else:
        raise TypeError(f"Cannot retrieve name of variable {x} of type {type(x)}.")


def _is_callable_or_class(obj):
    """Check if obj is callable or class."""
    return callable(obj) or inspect.isclass(obj)


def _unpack_params(params_dict: dict, string_attrs: list = None) -> dict:
    """
    Convert a parameter dictionary into serializable format.

    For objects with `get_params`/`set_params`, extracts the class name and
    parameters. Some attributes are converted to strings to facilitate saving and loading.

    Parameters
    ----------
    params_dict :
        Dictionary of parameters, possibly containing objects.
    string_attrs :
        List of attributes that should be converted to strings (e.g., `inverse_link_function`).

    Returns
    -------
    dict :
        Serializable dictionary with class names and nested parameters.
    """

    out = dict()
    for key, value in params_dict.items():
        # if the parameter is an objet with get_params/set_params,
        # extract its class name and parameters
        if hasattr(value, "get_params") and hasattr(value, "set_params"):
            cls_name = _get_name(value)
            params = _unpack_params(value.get_params(deep=False), string_attrs)
            out[key] = {"class": cls_name, "params": params}
        else:
            # if the parameter is in string_attrs, store its name
            if string_attrs is not None and (
                key in string_attrs or _is_callable_or_class(value)
            ):
                out[key] = _get_name(value)
            else:
                out[key] = value
    return out


def get_env_metadata() -> dict[str, str]:
    """Get environment metadata.

    Get the environment metadata relevant to model fitting.


    Notes
    -----
    ``jax`` and ``jaxlib`` for arrays and linear algebra, ``scipy`` is used at model
    initialization to find numerical inverse for custom link functions, ``scikit-learn``
    is used for pipelines and cross-validation.
    """
    return {
        "jax": version("jax"),
        "jaxlib": version("jaxlib"),
        "scipy": version("scipy"),
        "scikit-learn": version("scikit-learn"),
        "nemos": version("nemos"),
    }


def get_flattener_unflattener(parameter_tree: Pytree):
    """
    Create functions for flattening parameter pytrees and reshaping them to their original shape.

    Parameters
    ----------
    parameter_tree :
        Pytree to flatten. Usually model parameters.

    Returns
    -------
    (flattener, unflattener):
        Tuple of two functions: first one flattens the parameters, second one resshapes the flat ones.
    """
    flat, struct = jax.tree_util.tree_flatten(parameter_tree)
    shapes = [x.shape for x in flat]
    sizes = jnp.array([x.size for x in flat], dtype=int)
    split_indices = jnp.cumsum(sizes[:-1])

    def flattener(parameter_tree):
        flat = jax.tree_util.tree_leaves(parameter_tree)
        return jnp.concatenate([x.flatten() for x in flat])

    def unflattener(flat_params):
        split_params = jnp.split(flat_params, split_indices)
        split_params = [x.reshape(s) for x, s in zip(split_params, shapes)]
        return jax.tree_util.tree_unflatten(struct, split_params)

    return flattener, unflattener
