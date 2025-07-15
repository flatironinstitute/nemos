"""Provides functionality to load a previously saved nemos model from a `.npz` file."""

import inspect
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

MODEL_REGISTRY = {"nemos.glm.GLM": GLM, "nemos.glm.PopulationGLM": PopulationGLM}

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
    observation_model__inverse_link_function: <function one_over_x at ...>
    observation_model: GammaObservations(inverse_link_function=one_over_x)
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
    observation_model__inverse_link_function: <function one_over_x at ...>
    observation_model: GammaObservations(inverse_link_function=one_over_x)
    regularizer: Ridge()
    regularizer_strength: 0.1
    solver_kwargs: {'stepsize': 0.1, 'maxiter': 1000, 'tol': 1e-06}
    solver_name: BFGS
    >>> # If you want to load the model with a custom mapping for callables and custom classes,
    >>> # you can use the mapping_dict parameter.
    >>> mapping_dict = {
    ...     "observation_model__inverse_link_function": lambda x: x**2,
    ...     "regularizer": nmo.regularizer.UnRegularized,
    ... }
    >>> loaded_model = nmo.load_model("model_params.npz", mapping_dict=mapping_dict)
    >>> # Now the loaded model will have the updated solver_name and solver_kwargs
    >>> for key, value in loaded_model.get_params().items():
    ...     print(f"{key}: {value}")
    observation_model__inverse_link_function: <function <lambda> at ...>
    observation_model: GammaObservations(inverse_link_function=<lambda>)
    regularizer: UnRegularized()
    regularizer_strength: 0.1
    solver_kwargs: {'stepsize': 0.1, 'maxiter': 1000, 'tol': 1e-06}
    solver_name: BFGS
    """
    # load the model from a .npz file
    filename = Path(filename)
    data = np.load(filename, allow_pickle=False)

    invalid_keys = _get_invalid_mappings(mapping_dict)
    if len(invalid_keys) > 0:
        raise ValueError(
            "Invalid map parameter types detected. "
            f"The following parameters are non mappable:\n\t{invalid_keys}\n"
            "Only callables and classes can be mapped."
        )

    flat_map_dict = (
        {}
        if mapping_dict is None
        else {_expand_user_keys(k, data): v for k, v in mapping_dict.items()}
    )

    # check for keys that are not in the parameters
    if mapping_dict is not None:
        not_available = [
            key_user
            for key_expanded, key_user in zip(flat_map_dict.keys(), mapping_dict.keys())
            if key_expanded not in data.keys()
        ]
        if len(not_available) > 0:
            raise ValueError(
                "The following mapped parameters are not available in the loaded model:\n"
                f"\t{not_available}"
                "Use `nmo.inspect_npz('{filename}')` to inspect the saved parameter names."
            )

    # Unflatten the dictionary to restore the original structure
    saved_params = _unflatten_dict(data, flat_map_dict)

    # "save_metadata" is used to store versions of Nemos and Jax, not needed for loading
    saved_params.pop("save_metadata")

    # if any value from saved_params is a key in mapping_dict,
    # replace it with the corresponding value from mapping_dict
    saved_params, updated_keys = _apply_custom_map(saved_params)

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
        return instantiate_regularizer(class_name)
    elif any(class_basename.startswith(obs) for obs in AVAILABLE_OBSERVATION_MODELS):
        return instantiate_observation_model(class_name, **kwargs)
    else:
        # Should be hit only if the params had been modified
        raise ValueError(
            f"Invalid class name {class_name}. \n"
            f"Initialization is only allowed for NeMoS regularizers or "
            f"observation models."
        )


def _apply_custom_map(
    params: dict, updated_keys: List | None = None
) -> Tuple[dict, List]:
    """Apply mapping dictionary to replace values if keys match."""
    updated_params = {}

    if updated_keys is None:
        updated_keys = []

    for key, val in params.items():
        # handle classes and params separately
        if _is_param(val):
            # unpack mapping info and val
            orig_param, (is_mapped, mapped_param) = val

            if not is_mapped:
                updated_params[key] = orig_param
            else:
                updated_params[key] = mapped_param
                updated_keys.append(key)
        else:
            # if val is a class, it must be a dict with a "class" key
            class_name, (is_mapped, mapped_class) = val.pop("class")
            if not is_mapped:
                # check for nested callable/classes save instantiate based on the string
                new_params, updated_keys = _apply_custom_map(
                    val.pop("params", {}), updated_keys=updated_keys
                )
                updated_params[key] = _safe_instantiate(key, class_name, **new_params)
                updated_params[key] = _safe_instantiate(key, class_name, **new_params)
            else:
                updated_keys.append(key)
                # Should not be hit ever, assertion for developers
                assert inspect.isclass(mapped_class), (
                    f"The parameter '{key}' passed the type check in "
                    f"``nmo.load_model`` but is not callable or class, "
                    "check why this is the case."
                )
                # map callables and nested classes
                new_params, updated_keys = _apply_custom_map(
                    val.pop("params", {}), updated_keys=updated_keys
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


def _expand_user_keys(user_key, flat_keys):
    """Expand user key mapping path to match saved keys."""
    parts = user_key.split("__")

    # flat key (one level only)
    if len(parts) == 1:
        # either it is a class or a param
        if f"{parts[0]}__class" in flat_keys:
            return "__".join([parts[0], "class"])
        return parts[0]

    # interleave params, this assumes that the only nesting allowed
    # is: class__params__class__params... but not dictionaries.
    path = []
    for part in parts[:-1]:
        path.extend([part, "params"])

    flat_key = "__".join(path) + f"__{parts[-1]}__class"
    if flat_key in flat_keys:
        return flat_key
    else:
        path.append(parts[-1])
        flat_key = "__".join(path)
    return flat_key


def _get_invalid_mappings(mapping_dict: dict | None) -> List:
    if mapping_dict is None:
        return []
    return [
        k
        for k, v in mapping_dict.items()
        if (not inspect.isclass(v)) and not callable(v)
    ]
