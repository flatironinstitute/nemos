"""Validation and construction of inverse link functions."""

from typing import Callable

import jax
import jax.numpy as jnp

from ..utils import one_over_x

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
        If the function is not callable.
    ValueError
        If the function does not return a jax.numpy.ndarray,
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
