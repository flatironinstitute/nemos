"""Provides functionality to load a previously saved nemos model from a `.npz` file."""

import importlib
import warnings
from pathlib import Path
from typing import Union

import numpy as np

from .._observation_model_builder import instantiate_observation_model
from ..glm import GLM, PopulationGLM
from ..utils import _get_name, _unflatten_dict

MODEL_REGISTRY = {"nemos.glm.GLM": GLM, "nemos.glm.PopulationGLM": PopulationGLM}


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
    >>> # If you want to load the model with a custom mapping, you can use the mapping_dict parameter.
    >>> mapping_dict = {"solver_name": "GradientDescent",
    ...                 "solver_kwargs": {"stepsize": 0.01, "acceleration": False}}
    >>> loaded_model = nmo.load_model("model_params.npz", mapping_dict=mapping_dict)
    >>> # Now the loaded model will have the updated solver_name and solver_kwargs
    >>> for key, value in loaded_model.get_params().items():
    ...     print(f"{key}: {value}")
    observation_model__inverse_link_function: <function one_over_x at ...>
    observation_model: GammaObservations(inverse_link_function=one_over_x)
    regularizer: Ridge()
    regularizer_strength: 0.1
    solver_kwargs: {'stepsize': 0.01, 'acceleration': False}
    solver_name: GradientDescent
    """

    # load the model from a .npz file
    filename = Path(filename)
    data = np.load(filename, allow_pickle=False)

    # Unflatten the dictionary to restore the original structure
    saved_params = _unflatten_dict(data)

    # "save_metadata" is used to store versions of Nemos and Jax, not needed for loading
    saved_params.pop("save_metadata")

    # if any value from saved_params is a key in mapping_dict,
    # replace it with the corresponding value from mapping_dict
    saved_params = _apply_custom_map(saved_params, mapping_dict)

    # If the observation model is a string or a dictionary, instantiate it
    # By default it is saved as a dictionary with "class" and "params"
    if "observation_model" in saved_params:
        obs_model_data = saved_params["observation_model"]
        if isinstance(obs_model_data, str):
            saved_params["observation_model"] = instantiate_observation_model(
                obs_model_data
            )
        elif isinstance(obs_model_data, dict):
            saved_params["observation_model"] = instantiate_observation_model(
                obs_model_data["class"], **obs_model_data["params"]
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


def _apply_custom_map(params: dict, mapping_dict: dict) -> dict:
    """Apply mapping dictionary to replace values if keys match."""
    if not mapping_dict:
        return params

    missing_keys = [key for key in mapping_dict if key not in params]
    if missing_keys:
        raise ValueError(
            f"Keys {missing_keys} in mapping_dict are not found in the model parameters. "
            f"Available keys: {list(params.keys())}"
        )

    updated_params = params.copy()
    updated_params.update(mapping_dict)

    warnings.warn(
        f"The following keys have been replaced in the model parameters: {list(mapping_dict.keys())}. "
    )

    return updated_params


def _split_model_params(params: dict, model_class) -> tuple:
    """Split parameters into config and fit parameters."""
    model_param_names = model_class._get_param_names()
    config_params = {k: v for k, v in params.items() if k in model_param_names}
    fit_params = {k: v for k, v in params.items() if k not in model_param_names}
    return config_params, fit_params


def _set_fit_params(model, fit_params: dict, filename: Path):
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

    metadata = data.pop("save_metadata", None)
    print("Metadata\n--------")
    print(
        f"{'nemos version':<{pad_len}}: {metadata['nemos_version']} "
        f"(installed: {importlib.metadata.version('nemos')})"
    )
    print(
        f"{'jax version':<{pad_len}}: {metadata['jax_version']} "
        f"(installed: {importlib.metadata.version('jax')})"
    )

    print("\nModel class\n-----------")
    model_class = data.pop("model_class", None)
    if model_class:
        print(f"{'Saved model class':<{pad_len}}: {model_class}")

    print("\nModel parameters \n----------------")
    config_params = {k: data.pop(k) for k in list(data) if not k.endswith("_")}
    for key in config_params:
        val = config_params[key]
        # If the value is a callable, print its name, otherwise print the value
        if hasattr(val, "__name__"):
            print(f"{key:<{pad_len}}: {_get_name(val)}")
        else:
            print(f"{key:<{pad_len}}: {val}")

    print("\nModel fit parameters \n--------------------")
    for param in data:
        print(f"{param}: {data[param]}")
