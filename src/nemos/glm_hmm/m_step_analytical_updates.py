"""Implementations of analyzing M-step updates."""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from .utils import Array


@partial(jax.jit, static_argnames=["negative_log_likelihood_func"])
def _m_step_scale_gaussian_observations(
    log_scale: Array, y, rate, posteriors, negative_log_likelihood_func
) -> Tuple[Array, None, None]:
    r"""
    Analytical M-step update for Gaussian observation model scale (variance).

    Computes the closed-form maximum likelihood estimate of the variance parameter
    for a Gaussian observation model given posterior distributions over latent states.
    The update is derived by setting the derivative of the expected complete-data
    log-likelihood with respect to the variance to zero.

    Parameters
    ----------
    log_scale :
        Current log-scale parameter values, shape ``(n_states,)`` for single observations
        or ``(n_neurons, n_states)`` for population data. Not used in computation
        but required to match the signature of numerical optimization methods.
    y :
        Target observations, shape ``(n_time_bins,)`` for single observations
        or ``(n_time_bins, n_neurons)`` for population data.
    rate :
        Predicted mean values for each state, shape ``(n_time_bins, n_states)``
        for single observations or ``(n_time_bins, n_neurons, n_states)`` for
        population data.
    posteriors :
        Posterior probabilities over states, shape ``(n_time_bins, n_states)``.
    negative_log_likelihood_func :
        Unnormalized negative log-likelihood function that computes squared
        residuals ``(y - rate)^2 / 2`` for Gaussian observations.

    Returns
    -------
    optimized_log_scale :
        Updated log-scale estimates, shape ``(n_states,)`` for single observations
        or ``(n_neurons, n_states)`` for population data.

    Notes
    -----
    The analytical solution is:

    .. math::

        \\sigma_k^2 = \\frac{\\sum_t \\gamma_{tk} (y_t - \\mu_{tk})^2}{\\sum_t \\gamma_{tk}}

    where :math:`\\gamma_{tk}` are the posterior probabilities and :math:`\\mu_{tk}`
    are the predicted rates for state k.

    The scale parameter is unused in the computation but present to match the
    signature of ``solver.run`` for numerical scale optimization methods.
    """
    nll = negative_log_likelihood_func(y, rate)
    sum_posteriors = jnp.sum(posteriors, axis=0, keepdims=True)  # (1, n_states)

    # population update
    if nll.ndim > 2:
        expected_nll = jnp.einsum(
            "ts, tns -> ns", posteriors, nll
        )  # (n_neurons, n_states)
        optimized_log_scale = jnp.log(expected_nll) - jnp.log(sum_posteriors)
    else:
        expected_nll = jnp.sum(posteriors * nll, axis=0, keepdims=True)  # (1, n_states)
        optimized_log_scale = jnp.squeeze(
            jnp.log(expected_nll) - jnp.log(sum_posteriors), axis=0
        )

    return optimized_log_scale, None, None
