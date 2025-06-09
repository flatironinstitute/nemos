"""Provides functionality to load a previously saved nemos model from a `.npz` file."""

from pathlib import Path
from typing import Union

import numpy as np

from ..glm import GLM
from ..utils import unflatten_dict

__all__ = ["load_model"]


def __dir__() -> list[str]:
    return __all__


MODEL_REGISTRY = {"nemos.glm.GLM": GLM}


def load_model(filename: Union[str, Path], mapping_dict: dict = None):
    """
    Load a previously saved nemos model from a .npz file.

    Parameters
    ----------
    filename :
        Path to the saved .npz file.

    mapping_dict :
        Optional dictionary to map costume attribute names to their actual objects.

    Returns
    -------
    model :
        An instance of the model class with the loaded parameters.

    Examples
    --------
    >>> import nemos as nmo
    >>> from nemos.io.load_model import load_model

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
    >>> print(model.get_params())
    {'observation_model__inverse_link_function': <function one_over_x at ...>, 'observation_model': GammaObservations(inverse_link_function=one_over_x), 'regularizer': Ridge(), 'regularizer_strength': 0.1, 'solver_kwargs': {'stepsize': 0.1, 'maxiter': 1000, 'tol': 1e-06}, 'solver_name': 'BFGS'}

    >>> # Save the model parameters to a file
    >>> model.save_params("model_params.npz")

    >>> # Initialize a default GLM
    >>> model = nmo.glm.GLM()

    >>> # Print the model parameters
    >>> print(model.get_params())
    {'observation_model__inverse_link_function': <PjitFunction of <function exp at ...>>, 'observation_model': PoissonObservations(inverse_link_function=exp), 'regularizer': UnRegularized(), 'regularizer_strength': None, 'solver_kwargs': {}, 'solver_name': 'GradientDescent'}

    >>> # Load the model from the saved file
    >>> model = load_model("model_params.npz")

    >>> # Print the parameters of the loaded model
    >>> print(model.get_params())
    {'observation_model__inverse_link_function': <function one_over_x at ...>, 'observation_model': GammaObservations(inverse_link_function=one_over_x), 'regularizer': Ridge(), 'regularizer_strength': 0.1, 'solver_kwargs': {'stepsize': array(0.1), 'maxiter': array(1000), 'tol': array(1.e-06)}, 'solver_name': 'BFGS'}
    """

    # load the model from a .npz file
    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"File not found: {filename}")
    data = np.load(filename, allow_pickle=False)

    # Unflatten the dictionary to restore the original structure
    saved_attrs = unflatten_dict(data)

    # Extract the model class from the saved attributes
    model_name = str(saved_attrs["model_class"])
    model_class = MODEL_REGISTRY[model_name]

    return model_class._load_from_dict(saved_attrs, mapping_dict=mapping_dict)
