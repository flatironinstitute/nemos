"""Provides functionality to load a previously saved nemos model from a `.npz` file."""

import inspect
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from ..base_regressor import BaseRegressor
    from ..observation_models import Observations
    from ..regularizer import Regularizer

from .._observation_model_builder import (
    AVAILABLE_OBSERVATION_MODELS,
    instantiate_observation_model,
)
from .._regularizer_builder import AVAILABLE_REGULARIZERS, instantiate_regularizer
from ..glm import GLM, PopulationGLM
from ..utils import _get_name, _unflatten_dict, get_env_metadata
from ..validation import _suggest_keys

MODEL_REGISTRY = {
    "nemos.glm.glm.GLM": GLM,
    "nemos.glm.glm.PopulationGLM": PopulationGLM,
}

ERROR_MSG_OVERRIDE_NOT_ALLOWED = (
    "Cannot override the parameter {key}. "
    "NeMoS only allows overriding parameters that cannot be directly saved, "
    "such as callables, custom classes, or other objects that require pickling. "
    "If you really want to override the parameter, load the model without mapping "
    "it and then call ``set_params`` to set it afterwards."
)


def load_model(filename: Union[str, Path], mapping_dict: dict = None):
    """
    Load a previously saved nemos model from a .npz file.

    This will read the model parameters from the specified file and instantiate
    the model class with those parameters. It allows for custom mapping of
    attribute names to their actual objects using a mapping dictionary.

    Parameters
    ----------
    filename :
        Path to the saved .npz file.
    mapping_dict :
        Optional dictionary to map custom attribute names to their actual objects.

    Returns
    -------
    model :
        An instance of the model class with the loaded parameters.

    Examples
    --------
    >>> import nemos as nmo
    >>> # Create a GLM model with specified parameters
    >>> solver_args = {"stepsize": 0.1, "maxiter": 1000, "tol": 1e-6}
    >>> model = nmo.glm.GLM(
    ...     regularizer="Ridge",
    ...     regularizer_strength=0.1,
    ...     observation_model="Gamma",
    ...     solver_name="BFGS",
    ...     solver_kwargs=solver_args,
    ... )
    >>> for key, value in model.get_params().items():
    ...     print(f"{key}: {value}")
    inverse_link_function: <function one_over_x at ...>
    observation_model: GammaObservations()
    regularizer: Ridge()
    regularizer_strength: 0.1
    solver_kwargs: {'stepsize': 0.1, 'maxiter': 1000, 'tol': 1e-06}
    solver_name: BFGS
    >>> # Save the model parameters to a file
    >>> model.save_params("model_params.npz")
    >>> # Load the model from the saved file
    >>> model = nmo.load_model("model_params.npz")
    >>> # Model has the same parameters before and after load
    >>> for key, value in model.get_params().items():
    ...     print(f"{key}: {value}")
    inverse_link_function: <function one_over_x at ...>
    observation_model: GammaObservations()
    regularizer: Ridge()
    regularizer_strength: 0.1
    solver_kwargs: {'stepsize': 0.1, 'maxiter': 1000, 'tol': 1e-06}
    solver_name: BFGS

    >>> # Loading a custom inverse link function
    >>> model = nmo.glm.GLM(inverse_link_function=lambda x: x**2)
    >>> model.save_params("model_params.npz")
    >>> # Provide a mapping for the custom link function when loading.
    >>> mapping_dict = {
    ...     "inverse_link_function": lambda x: x**2,
    ... }
    >>> loaded_model = nmo.load_model("model_params.npz", mapping_dict=mapping_dict)
    >>> # Now the loaded model will have the updated solver_name and solver_kwargs
    >>> for key, value in loaded_model.get_params().items():
    ...     print(f"{key}: {value}")
    inverse_link_function: <function <lambda> at ...>
    observation_model: PoissonObservations()
    regularizer: UnRegularized()
    regularizer_strength: None
    solver_kwargs: {}
    solver_name: GradientDescent
    """
    # load the model from a .npz file
    filename = Path(filename)
    data = np.load(filename, allow_pickle=False)

    # unflatten dictionary
    saved_params = _unflatten_dict(data)
    # "save_metadata" is used to store versions of Nemos and Jax, not needed for loading
    saved_params.pop("save_metadata")
    # unflatten user map
    nested_map_dict, key_not_found = _unflattened_user_map(mapping_dict, saved_params)

    invalid_keys = _get_invalid_mappings(nested_map_dict)
    if len(invalid_keys) > 0:
        raise ValueError(
            "Invalid map parameter types detected. "
            f"The following parameters are non mappable:\n\t{invalid_keys}\n"
            "Only callables and classes can be mapped."
        )

    # backtrack all errors
    if key_not_found:
        available_keys = get_user_keys_from_nested_dict(saved_params)
        requested_keys = get_user_keys_from_nested_dict(mapping_dict)
        not_available = sorted(set(requested_keys).difference(available_keys))
        suggested_pairs = _suggest_keys(not_available, available_keys)
        suggestions = "".join(
            [
                (
                    f"\t- '{provided}', did you mean '{suggested}'?\n"
                    if suggested is not None
                    else f"\t- '{provided}'\n"
                )
                for provided, suggested in suggested_pairs
            ]
        )
        raise ValueError(
            "The following keys in your mapping do not match any parameters in the loaded model:\n\n"
            f"{suggestions}\n"
            "Please double-check your mapping dictionary."
        )
    # if any value from saved_params is a key in mapping_dict,
    # replace it with the corresponding value from mapping_dict
    saved_params, updated_keys = _apply_custom_map(saved_params, nested_map_dict)

    if len(updated_keys) > 0:
        warnings.warn(
            f"The following keys have been replaced in the model parameters: {updated_keys}.",
            UserWarning,
        )

    # Extract the model class from the saved attributes
    model_name = str(saved_params.pop("model_class"))
    model_class = MODEL_REGISTRY[model_name]

    config_params, fit_params = _split_model_params(saved_params, model_class)

    # Create the model instance
    try:
        model = model_class(**config_params)
    except Exception:
        raise ValueError(
            f"Failed to instantiate model class '{model_name}' with parameters: {config_params}. "
            f"Use `nmo.inspect_npz('{filename}')` to inspect the saved object."
        )

    # Set the rest of the parameters as attributes if recognized
    _set_fit_params(model, fit_params, filename)

    return model


def _is_param(par):
    if not isinstance(par, dict):
        return True
    return "class" not in par


def _safe_instantiate(
    parameter_name: str, class_name: str, **kwargs
) -> "Regularizer | Observations":
    if not isinstance(class_name, str):
        # this should not be hit, if it does the saved params had been modified.
        raise ValueError(
            f"Parameter ``{parameter_name}`` cannot be initialized. "
            "When a parameter specifies a class, the class name must be a string. "
            f"Class name for the loaded parameter is {class_name}."
        )
    class_basename = class_name.split(".")[-1]
    if class_basename in AVAILABLE_REGULARIZERS:
        return instantiate_regularizer(class_name, **kwargs)
    elif any(class_basename.startswith(obs) for obs in AVAILABLE_OBSERVATION_MODELS):
        return instantiate_observation_model(class_name, **kwargs)
    else:
        # Hit when loading a custom class without mapping
        if parameter_name == "observation_model":
            class_type = "observation"
        else:
            class_type = "regularization"
        raise ValueError(
            f"The class '{class_basename}' is not a native NeMoS class.\n"
            f"To load a custom {class_type} class, please provide the following mapping:\n\n"
            f" - nemos.load_model(save_path, mapping_dict={{'{parameter_name}': {class_basename}}})"
        )


def _apply_custom_map(
    params: dict, mapping_dict: dict, updated_keys: List | None = None
) -> Tuple[dict, List]:
    """
    Recursively apply user-defined mappings to a saved parameter structure.

    This function processes the nested parameter dictionary produced by `_unflatten_dict`
    and applies user-specified overrides where allowed. It does the following:

    - For leaf parameters stored as [value, [is_mapped, mapped_value]]:
        * If `is_mapped` is True, the parameter is replaced with `mapped_value`.
          Only callables or classes are allowed; other types raise an error.
        * If `is_mapped` is False, the original saved value is kept.
    - For nested dictionaries of parameters (e.g., solver kwargs):
        * These cannot be overridden because they are not callables or classes.
        * All leaves are recursively unwrapped to extract the original saved values,
          discarding any mapping info.
    - For parameters representing classes:
        * If not mapped, the original class name is checked and instantiated safely using `_safe_instantiate`.
        * If mapped, the mapping must be an actual Python class object (not a string or an instance).
          This invariant is enforced with an internal assertion for developer safety.

    This function also keeps track of which keys were overridden by the user-supplied mapping,
    returning this list alongside the reconstructed parameter dictionary.

    Parameters
    ----------
    params :
        The nested saved parameters to process. Each entry is either:
          - A leaf in the form [value, [is_mapped, mapped_value]], or
          - A nested dict representing classes.
    mapping_dict:
        A dict of mappings following the same keypath that params follows (a nested dict).
    updated_keys :
        List of keys that have already been updated, used for accumulating changes
        across recursive calls.

    Returns
    -------
    updated_params :
        The new parameter dictionary with mappings applied and wrappers removed.
    updated_keys :
        List of all keys that were actually overridden.

    Raises
    ------
    ValueError
        If a user tries to override a parameter with an unsupported type (non-callable, non-class),
        or provides a mapped class as a string instead of a Python class object. This is triggered
        in `_safe_instantiate`.

    Notes
    -----
    This function enforces strict override safety: only callables and classes may be
    mapped at load time. Directly serializable values and nested dictionaries cannot
    be overridden and must be changed later using `set_params` if needed.

    Internal invariants are checked with `assert` to ensure that only valid class mappings
    reach instantiation. If these assertions fail, it indicates a bug in the input validation
    logic and should never occur in normal use.
    """
    updated_params = {}

    if updated_keys is None:
        updated_keys = []

    for key, val in params.items():
        # handle classes and params separately
        if _is_param(val):
            if isinstance(val, dict):
                # dict cannot be mapped, so store original params
                updated_params[key] = val
            else:
                mapped_val = mapping_dict.get(key, None)
                is_mapped = mapped_val is not None
                if is_mapped:
                    updated_params[key] = mapped_val
                    updated_keys.append(key)
                else:
                    updated_params[key] = val

        else:
            # if val is a class, it must be a dict with a "class" key
            class_name = val.pop("class")
            mapped_val = mapping_dict.get(key, {})
            is_mapped = "class" in mapped_val
            mapped_params = mapped_val.get("params", {})
            if not is_mapped:
                # check for nested callable/classes save instantiate based on the string
                new_params, updated_keys = _apply_custom_map(
                    val.pop("params", {}), mapped_params, updated_keys=updated_keys
                )
                updated_params[key] = _safe_instantiate(key, class_name, **new_params)
            else:
                mapped_class = mapped_val["class"]
                updated_keys.append(key)
                # Should not be hit ever, assertion for developers
                assert inspect.isclass(mapped_class), (
                    f"The parameter '{key}' passed the type check in "
                    f"``nmo.load_model`` but is not callable or class, "
                    "check why this is the case."
                )
                # map callables and nested classes
                new_params, updated_keys = _apply_custom_map(
                    val.pop("params", {}), mapped_params, updated_keys=updated_keys
                )
                # try instantiating it with the params
                # this executes code, but we are assuming that the mapped_class is safe
                updated_params[key] = mapped_class(**new_params)

    return updated_params, updated_keys


def _split_model_params(params: dict, model_class) -> tuple:
    """Split parameters into config and fit parameters."""
    model_param_names = model_class._get_param_names()
    config_params = {k: v for k, v in params.items() if k in model_param_names}
    fit_params = {k: v for k, v in params.items() if k not in model_param_names}
    return config_params, fit_params


def _set_fit_params(model: "BaseRegressor", fit_params: dict, filename: Path):
    """Set fit model attributes, warn if unrecognized."""
    check_str = (
        f"\nIf this is confusing, try calling "
        f"`{_get_name(inspect_npz)}('{filename}')` to inspect the saved object."
    )
    for key, value in fit_params.items():
        if hasattr(model, key):
            setattr(model, key, value)
        else:
            raise ValueError(
                f"Unrecognized attribute '{key}' during model loading.{check_str}"
            )


def inspect_npz(file_path: Union[str, Path]):
    """
    Examine a saved model parameter file (.npz).

    Prints out all keys and associated values.

    Parameters
    ----------
    file_path :
        Path to the `.npz` file containing the saved model parameters.
    """
    file_path = Path(file_path)
    data = np.load(file_path, allow_pickle=True)
    data = _unflatten_dict(data)

    pad_len = max(len(k) for k in data.keys()) + 2

    metadata: dict | None = data.pop("save_metadata", None)
    installed_env = get_env_metadata()
    if metadata is not None:
        print("Metadata\n--------")
        for k, v in metadata.items():
            label = f"{k} version"
            print(f"{label:<{pad_len}}: {v}" f" (installed: {installed_env[k]})")

    print("\nModel class\n-----------")
    model_class = data.pop("model_class", None)
    if model_class:
        print(f"{'Saved model class':<{pad_len}}: {model_class}")

    print("\nModel parameters\n----------------")
    config_params = {k: data.pop(k) for k in list(data) if not k.endswith("_")}
    for key in config_params:
        val = config_params[key]
        # If the value is a callable, print its name, otherwise print the value
        if hasattr(val, "__name__"):
            print(f"{key:<{pad_len}}: {_get_name(val)}")
        else:
            print(f"{key:<{pad_len}}: {val}")

    print("\nModel fit parameters\n--------------------")
    for param in data:
        print(f"{param}: {data[param]}")


def _unflattened_user_map(
    mapping_dict: dict | None, nested_dict: dict
) -> Tuple[dict, bool]:
    """Expand user key mapping path to match saved keys."""
    if mapping_dict is None:
        return {}, False
    nested_mapping = {}
    for user_key, mapped_value in mapping_dict.items():
        current = nested_mapping
        subdict = nested_dict
        parts = [k[::-1] for k in user_key[::-1].split("__")][::-1]
        for part in parts[:-1]:
            if part not in subdict:
                return {}, True
            subdict = nested_dict[part]
            if part not in current:
                current[part] = {}
            current = current[part]
            # in we hit this case, this is a parameter (this is not the last
            # part of the key, but the class is always the last element)
            if isinstance(subdict, dict) and "class" in subdict:
                subdict = subdict["params"]
                if "params" not in current:
                    current["params"] = {}
                current = current["params"]

        if parts[-1] not in subdict:
            return None, True
        elif isinstance(subdict[parts[-1]], dict) and "class" in subdict[parts[-1]]:
            current[parts[-1]] = {"class": mapped_value}
        else:
            current[parts[-1]] = mapped_value
    return nested_mapping, False


def get_user_keys_from_nested_dict(nested_dict: dict, filter_keys: bool = True) -> list:
    """
    Get the user-formatted keys from a nested dictionary.

    Retrieve user-formatted keys from a nested parameter dictionary. The formatting matches the
    sklearn-style parameter naming convention (e.g., 'regularizer__solver_name'). This format
    should be used when providing a ``mapping_dict`` to ``load_model`` to override saved parameters.

    Parameters
    ----------
    nested_dict :
        A nested parameter dictionary, typically from a saved model.
    filter_keys :
        If True, remove internal keys ('__class' and '__params') from the output and return
        only user-facing parameter names. Default is True.

    Returns
    -------
    list of str
        A sorted list of user-formatted keys in sklearn parameter style, where nested attributes
        are joined with double underscores (e.g., 'observation_model__class', 'solver_kwargs__tol').
        The 'save_metadata' key is excluded from the output.

    Notes
    -----
    - Keys are formatted using double underscores ('__') as separators to match sklearn conventions
    - The 'save_metadata' key is automatically excluded from results
    - Internal structure keys ('__params') are filtered out when filter_keys=True

    Examples
    --------
    >>> params = {
    ...     'regularizer': {'class': 'GroupLasso', 'params': {'mask': None}},
    ...     'solver_kwargs': {'tol': 1e-7, 'maxiter': 100}
    ... }
    >>> get_user_keys_from_nested_dict(params)
    ['regularizer', 'regularizer__mask', 'solver_kwargs', 'solver_kwargs__maxiter', 'solver_kwargs__tol']
    """
    valid_keys = list(nested_dict.keys())
    sep = "__"
    for key in nested_dict.keys():
        if isinstance(nested_dict[key], dict):
            new_keys = get_user_keys_from_nested_dict(nested_dict[key], False)
            valid_keys.extend([key + sep + new for new in new_keys if new != "class"])
    if filter_keys:
        valid_keys = sorted(list({re.sub("(__params)", "", key) for key in valid_keys}))
    return valid_keys


def _get_invalid_mappings(mapping_dict: dict | None) -> List:
    """
    Recursively identify invalid entries in a model mapping dictionary.

    Validate a nested mapping dictionary by collecting keys whose values are not classes, callables,
    or valid nested mappings. This function is used during model deserialization (e.g., in
    ``nmo.load_model``) to validate a user-provided ``mapping_dict`` that maps
    symbolic model components to actual Python objects such as classes,
    callables, or parameter specifications.

    A mapping entry is considered **valid** if:
        * Its value is a class (``inspect.isclass(v)``), or
        * Its value is callable (e.g., a function or lambda), or
        * It is a dictionary containing a ``"class"`` key — in which case all
          of its subparameters (e.g., under ``"params"``) are accepted without
          validation, since they are assumed to be constructor arguments for
          the specified class.
        * It is a dictionary containing only a ``"params"`` key, in which case
          the function recursively validates each entry in ``v["params"]``.

    Any entry that does not meet these criteria is considered **invalid** and
    its key (or nested key path) is returned.

    Parameters
    ----------
    mapping_dict :
        A (possibly nested) dictionary defining how symbolic model components
        should be mapped to Python classes or callables. May contain nested
        entries with the special keys ``"class"`` and/or ``"params"``.

    Returns
    -------
    list of str
        A list of invalid key paths. For nested invalid entries, keys are joined
        with double underscores (``"__"``) to indicate hierarchy, e.g.,
        ``"regularizer__mask"``.

    Notes
    -----
    This function allows a ``"class"`` entry to bypass validation of its
    parameters because those parameters are intended to be passed to the class
    constructor during model instantiation.

    Examples
    --------
    >>> import nemos as nmo
    >>> def square(x):
    ...     return x**2
    ...
    >>> class MyRegularizer(nmo.regularizer.GroupLasso):
    ...     def __init__(self, mask=None, new_param=1):
    ...         super().__init__(mask=mask)
    ...         self.new_param = new_param
    ...
    >>> invalid = _get_invalid_mappings({
    ...     "inverse_link_function": square,
    ...     "regularizer": {"class": MyRegularizer, "params": {"regularizer__new_param": 10.}}
    ... })
    >>> invalid
    []
    >>> invalid = _get_invalid_mappings({
    ...     "inverse_link_function": square,
    ...     "regularizer": {"params": {"new_param": 10.}}
    ... })
    >>> invalid
    ['regularizer__new_param']
    """
    if mapping_dict is None:
        return []

    invalid = []

    for key, v in mapping_dict.items():
        if isinstance(v, dict) and "class" in v:
            if inspect.isclass(v["class"]):
                # Valid class - skip all validation including params
                continue
            else:
                # Invalid class - mark it
                invalid.append(key)
                # But still validate params if present, for completeness in the error message
                if "params" in v:
                    invalid_sub = _get_invalid_mappings(v["params"])
                    invalid.extend(f"{key}__{k}" for k in invalid_sub)
        # Handle dict with "params" key (but no "class" or already processed)
        elif isinstance(v, dict) and "params" in v:
            invalid_sub = _get_invalid_mappings(v["params"])
            invalid.extend(f"{key}__{k}" for k in invalid_sub)
        # Handle non-dict values
        elif not inspect.isclass(v) and not callable(v):
            invalid.append(key)

    return invalid
