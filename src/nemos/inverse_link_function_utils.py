"""Validation and construction of inverse link functions."""

from typing import Any, Callable

import jax
import jax.numpy as jnp

from .observation_models import Observations
from .utils import one_over_x

LINK_NAME_TO_FUNC = {
    "nemos.utils.one_over_x": one_over_x,
    "jax.numpy.exp": jnp.exp,
    "jax._src.numpy.ufuncs.exp": jnp.exp,
    "jax.nn.softplus": jax.nn.softplus,
    "jax._src.nn.functions.softplus": jax.nn.softplus,
    "jax.scipy.special.expit": jax.scipy.special.expit,
    "jax._src.scipy.special.expit": jax.scipy.special.expit,
    "jax.lax.logistic": jax.lax.logistic,
    "jax._src.lax.lax.logistic": jax.lax.logistic,
    "jax.scipy.stats.norm.cdf": jax.scipy.stats.norm.cdf,
    "jax._src.scipy.stats.norm.cdf": jax.scipy.stats.norm.cdf,
    "softplus": jax.nn.softplus,
    "exp": jax.numpy.exp,
    "one_over_x": one_over_x,
    "logistic": jax.lax.logistic,
    "norm.cdf": jax.scipy.stats.norm.cdf,
    "expit": jax.scipy.special.expit,
}


def link_function_from_string(link_name: str):
    """
    Get a link function from a given name.

    Parameters
    ----------
    link_name:
        A string representation of the link function, e.g. "jax.numpy.exp".

    Returns
    -------
    :
        The link function corresponding to the provided name.

    Raises
    ------
    ValueError:
        If the provided string does not match any known link function.
    """
    if link_name in LINK_NAME_TO_FUNC:
        return LINK_NAME_TO_FUNC[link_name]
    else:
        raise ValueError(
            f"Unknown link function: {link_name}. "
            f"Link function must be one of {list(LINK_NAME_TO_FUNC.keys())}"
            f"if you want to use a custom link function, please provide it as a Callable."
            f"if you think this is a bug, please open an issue at 'https://github.com/flatironinstitute/nemos/issues'."
        )


def check_inverse_link_function(inverse_link_function: Callable):
    """
    Check if the provided inverse_link_function is usable.

    This function verifies if the inverse link function:

    1. Is callable
    2. Returns a jax.numpy.ndarray
    3. Is differentiable (via jax)

    Parameters
    ----------
    inverse_link_function :
        The function to be checked.

    Raises
    ------
    TypeError
        If the function is not callable, does not return a jax.numpy.ndarray,
        or is not differentiable.
    """
    if inverse_link_function in LINK_NAME_TO_FUNC.values():
        return

    # check that it's callable
    if not callable(inverse_link_function):
        raise TypeError("The `inverse_link_function` function must be a Callable!")

    # check if the function returns a jax array for a 1D array
    array_out = inverse_link_function(jnp.array([1.0, 2.0, 3.0]))
    if not isinstance(array_out, jnp.ndarray):
        raise ValueError("The `inverse_link_function` must return a jax.numpy.ndarray!")

    # Optionally: Check for scalar input
    scalar_out = inverse_link_function(1.0)
    if not isinstance(scalar_out, (jnp.ndarray, float, int)):
        raise ValueError(
            "The `inverse_link_function` must handle scalar inputs correctly and return a scalar or a "
            "jax.numpy.ndarray!"
        )

    # check for autodiff
    try:
        gradient_fn = jax.grad(inverse_link_function)
        gradient_fn(1.0)
    except Exception as e:
        raise ValueError(
            f"The `inverse_link_function` function cannot be differentiated. Error: {e}"
        ) from e


def resolve_inverse_link_function(
    inverse_link_function: Any, observation_model: Observations
) -> Callable:
    """
    Validate and resolve an inverse link function specification.

    This helper function implements shared logic for setting inverse link functions
    in GLM-based models. It handles three cases:
    1. None: returns the observation model's default inverse link function
    2. String: parses and returns the corresponding link function
    3. Callable: validates and returns the function as-is

    Parameters
    ----------
    inverse_link_function : Callable, str, or None
        The inverse link function specification. Can be:
        - None: use the default from the observation model
        - str: name of a standard link function (e.g., "identity", "log", "logit")
        - Callable: a custom inverse link function
    observation_model : Observations
        The observation model instance that provides the default inverse link function.

    Returns
    -------
    Callable
        A validated inverse link function that maps linear predictions to the
        observation space.

    Raises
    ------
    TypeError
        If the provided function is not callable or a string.
    ValueError
        - If a callable is provided, but it does not return a jax.numpy.ndarray or scalar, or if it
        is not differentiable with respect to its inputs.
        - If a string is provided, but it cannot be parsed to a callable.

    Examples
    --------
    >>> import nemos as nmo
    >>> from nemos.inverse_link_function_utils import resolve_inverse_link_function
    >>> obs_model = nmo.observation_models.PoissonObservations()
    >>> # Use default link function
    >>> link_fn = resolve_inverse_link_function(None, obs_model)
    >>>
    >>> # Use named link function
    >>> link_fn = resolve_inverse_link_function("softplus", obs_model)
    >>>
    >>> # Use custom function
    >>> custom_fn = lambda x: x**2
    >>> link_fn = resolve_inverse_link_function(custom_fn, obs_model)
    """
    if inverse_link_function is None:
        return observation_model.default_inverse_link_function

    elif isinstance(inverse_link_function, str):
        return link_function_from_string(inverse_link_function)
    check_inverse_link_function(inverse_link_function)
    return inverse_link_function
