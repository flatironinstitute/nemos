"""Provides functionality to load a previously saved nemos model from a `.npz` file."""

import warnings
from pathlib import Path
from typing import Union

import numpy as np

from .._observation_model_builder import instantiate_observation_model
from ..glm import GLM, PopulationGLM
from ..utils import get_name, unflatten_dict

MODEL_REGISTRY = {"nemos.glm.GLM": GLM, "nemos.glm.PopulationGLM": PopulationGLM}


def load_model(filename: Union[str, Path], mapping_dict: dict = None):
    """
    Load a previously saved nemos model from a .npz file.

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

    >>> # Print the model parameters
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

    >>> # Initialize a default GLM
    >>> model = nmo.glm.GLM()

    >>> # Load the model from the saved file
    >>> model = nmo.load_model("model_params.npz")

    >>> # Print the parameters of the loaded model
    >>> for key, value in model.get_params().items():
    ...     print(f"{key}: {value}")
    observation_model__inverse_link_function: <function one_over_x at ...>
    observation_model: GammaObservations(inverse_link_function=one_over_x)
    regularizer: Ridge()
    regularizer_strength: 0.1
    solver_kwargs: {'stepsize': 0.1, 'maxiter': 1000, 'tol': 1e-06}
    solver_name: BFGS
    """

    # load the model from a .npz file
    filename = Path(filename)
    data = np.load(filename, allow_pickle=False)

    # Unflatten the dictionary to restore the original structure
    saved_params = unflatten_dict(data)

    # "save_metadata" is used to store versions of Nemos and Jax, not needed for loading
    saved_params.pop("save_metadata")

    # if any value from saved_params is a key in mapping_dict,
    # replace it with the corresponding value from mapping_dict
    saved_params = _apply_custom_map(saved_params, mapping_dict)

    if "observation_model" in saved_params:
        obs_model_data = saved_params.pop("observation_model")
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
            f"Failed to instantiate model class '{model_name}' with parameters: {config_params}"
            f"`{get_name(examine_saved_model)}('{filename}')` to inspect the saved object."
        )

    # Set the rest of the parameters as attributes if recognized
    _set_fit_params(model, fit_params, filename)

    return model


def _apply_custom_map(params: dict, mapping_dict: dict) -> dict:
    """Apply mapping dictionary to replace values if keys match."""
    if not mapping_dict:
        return params

    for k, v in params.items():
        try:
            if v in mapping_dict:
                try:
                    params[k] = mapping_dict[v]
                except Exception as e:
                    warnings.warn(
                        f"Failed to replace '{v}' for key '{k}': {e}. "
                        "Ensure the mapping dictionary is correctly formatted."
                    )
        except TypeError:
            pass

    return params


def _split_model_params(params: dict, model_class) -> tuple:
    """Split parameters into config and fit parameters."""
    model_param_names = model_class._get_param_names()
    config_params = {k: v for k, v in params.items() if k in model_param_names}
    fit_params = {k: v for k, v in params.items() if k not in model_param_names}
    return config_params, fit_params


def _set_fit_params(model, fit_params: dict, filename: Path):
    """Set remaining model attributes, warn if unrecognized."""
    check_str = (
        f"\nIf this is confusing, try calling "
        f"`{get_name(examine_saved_model)}('{filename}')` to inspect the saved object."
    )
    for key, value in fit_params.items():
        if hasattr(model, key):
            setattr(model, key, value)
        else:
            warnings.warn(
                f"Ignoring unrecognized attribute '{key}' during model loading.{check_str}"
            )


def examine_saved_model(file_path: Union[str, Path]):
    """
    Examine a saved model parameter file (.npz).

    Prints out all keys and associated values.

    Parameters
    ----------
    file_path :
        Path to the `.npz` file containing the saved model parameters.

    Examples
    --------
    >>> import nemos as nmo

    >>> # Define solver arguments
    >>> solver_args = {"stepsize": 0.1, "maxiter": 1000, "tol": 1e-6}

    >>> # Create a GLM model with specified regularizer and observation model
    >>> model = nmo.glm.GLM(
    ...     regularizer="Ridge",
    ...     regularizer_strength=0.1,
    ...     observation_model="Gamma",
    ...     solver_name="BFGS",
    ...     solver_kwargs=solver_args,
    ... )

    >>> # Save the model parameters to a file
    >>> model.save_params("model_params.npz")

    >>> # Examine the contents of the saved file
    >>> nmo.io.io.examine_saved_model("model_params.npz")
    observation_model     : <class 'dict'> with length 2 : {'class': 'nemos.observation_models.GammaObservations',\
'params': {'inverse_link_function': 'nemos.utils.one_over_x'}}
    regularizer           : <class 'str'> : nemos.regularizer.Ridge
    regularizer_strength  : <class 'float'> : 0.1
    solver_kwargs         : <class 'dict'> with length 3 : {'stepsize': 0.1, 'maxiter': 1000, 'tol': 1e-06}
    solver_name           : <class 'str'> : BFGS
    coef_                 : <class 'NoneType'> : None
    intercept_            : <class 'NoneType'> : None
    save_metadata         : <class 'dict'> with length 2 : {'jax_version': '0.4.38', 'nemos_version': '0.2.4.dev59'}
    model_class           : <class 'str'> : nemos.glm.GLM
    """
    file_path = Path(file_path)
    data = np.load(file_path, allow_pickle=True)
    data = unflatten_dict(data)

    pad_len = max(len(k) for k in data.keys()) + 2

    for key in data:
        val = data[key]
        if isinstance(val, np.ndarray):
            print(
                f"{key:<{pad_len}}: {type(val)}, dtype {val.dtype}, shape {val.shape} : {val}"
            )
        elif hasattr(val, "__len__") and not isinstance(val, str):
            print(f"{key:<{pad_len}}: {type(val)} with length {len(val)} : {val}")
        else:
            print(f"{key:<{pad_len}}: {type(val)} : {val}")
