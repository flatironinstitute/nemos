from typing import Callable

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike
from scipy.optimize import root_scalar

# dictionary of known inverse link functions.
INVERSE_FUNCS = {
    jnp.exp: jnp.log,
    jax.nn.softplus: lambda x: jnp.log(jnp.exp(x) - 1.0),
}


def scalar_root_find_elementwise(
    func: Callable, args: ArrayLike, x0: ArrayLike
) -> jnp.ndarray:
    """
    Find roots of a scalar function.

    This can be used as an attempt to find a numerical inverse of an unknown link function of a GLM; typically,
    this numerical inverse, is used to set the initial intercept to match the mean firing rate of the model.

    Parameters
    ----------
    func:
        A callable, which typically will be `inv_link_func(x) - jnp.mean(spikes)`.
    args:
        List of additional arguments passed to the function.
    x0:
        Initial values for the root-finding algorithm.

    Returns
    -------
    :
        An array containing the roots of each f(x) = func(x, args[k]), for k in 1,..., len(args).

    Raises
    ------
    ValueError:
        If any of the optimization is not successful.
    """
    opts = [root_scalar(func, arg, x0=x, method="secant") for arg, x in zip(args, x0)]

    if not all(jnp.abs(func(opt.root, args[i])) < 10**-4 for i, opt in enumerate(opts)):
        raise ValueError(
            "Could not set the initial intercept as the inverse of the firing rate for "
            "the provided link function. "
            "Please, provide initial parameters instead!"
        )

    return jnp.array([opt.root for opt in opts])


def initialize_intercept_matching_mean_rate(
    inverse_link_function: Callable, y: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute the initial intercept term for a regression models.

    This method compute an initial intercept term for a regression models such that the baseline activity
    matches the mean activity of each neuron, assuming that the model coefficients are initialized to zero.


    Parameters
    ----------
    inverse_link_function:
        The inverse link function of the model, linking the mean to the linear combination of the covariates in
        a GLM.
    y:
        The neural activity, shape either (num_sample,) for single variable regressors as `GLM`
         or (n_sample, n_neurons) for multi-variable regressors, such as `PopulaitonGLM`.

    Returns
    -------
    :
        The initial intercept term, shape (n_neurons,).

    """

    # return inverse if analytical solution is available
    analytical_inv = INVERSE_FUNCS.get(inverse_link_function, None)

    means = jnp.atleast_1d(jnp.nanmean(y, axis=0))
    if analytical_inv:
        out = analytical_inv(means)
        if jnp.any(jnp.isnan(out)):
            raise ValueError(
                "Failed to initialize the model intercept as the inverse of the firing rate for "
                "the provided link function. The mean firing rate has some non-positive values."
            )
        return out

    def func(x, mean_x):
        return inverse_link_function(x) - mean_x

    try:
        out = scalar_root_find_elementwise(func, means, means)
    except ValueError:
        raise ValueError(
            "Failed to initialize the model intercept as the inverse of the firing rate for the"
            " provided link function. Please, provide initial parameters instead!"
        )
    return out
