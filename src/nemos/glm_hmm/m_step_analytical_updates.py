"""Implementations of analyzing M-step updates."""

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from .params import GLMScale


def _analytical_m_step_initial_prob(
    posteriors: jnp.ndarray,
    is_new_session: jnp.ndarray,
    dirichlet_prior_alphas: Optional[jnp.ndarray] = None,
):
    """
    Compute the M-step update for initial state probabilities.

    Analytically computes the maximum-likelihood (or MAP with Dirichlet prior)
    estimate of the initial state distribution. Computation is performed in
    probability space for efficiency.

    Parameters
    ----------
    posteriors :
        Posterior probabilities over latent states, shape ``(n_time_bins, n_states)``.
    is_new_session :
        Boolean array indicating session start time bins, shape ``(n_time_bins,)``.
        Only these positions contribute to the initial state estimate.
    dirichlet_prior_alphas :
        Dirichlet prior parameters for the initial distribution,
        shape ``(n_states,)``. If None, uses uniform prior.
        **Note**: All alpha values must be >= 1.

    Returns
    -------
    new_initial_prob :
        Initial state probabilities, shape ``(n_states,)``.
        Normalized to sum to 1.

    Notes
    -----
    The current implementation requires Dirichlet prior parameters alpha >= 1.
    Support for sparse priors (0 < alpha < 1) may be added in a future version
    using alternative optimization methods.
    """
    # Mask and sum
    new_initial_prob = jnp.sum(posteriors, axis=0, where=is_new_session[:, jnp.newaxis])

    # Add prior
    if dirichlet_prior_alphas is not None:
        new_initial_prob += dirichlet_prior_alphas - 1

    # Normalize
    new_initial_prob /= jnp.sum(new_initial_prob)

    return new_initial_prob


def _analytical_m_step_transition_prob(
    joint_posterior: jnp.ndarray,
    dirichlet_prior_alphas: Optional[jnp.ndarray] = None,
):
    """
    Compute the M-step update for the transition probability matrix.

    Analytically computes the maximum-likelihood (or MAP with Dirichlet prior)
    estimate of the transition matrix using expected transition counts.
    Computation is performed in probability space for efficiency.

    Parameters
    ----------
    joint_posterior :
        Expected transition counts from state i to j (in probability space),
        shape ``(n_states, n_states)``.
    dirichlet_prior_alphas :
        Dirichlet prior parameters for each row of the transition matrix,
        shape ``(n_states, n_states)``. If None, uses uniform prior.
        **Note**: All alpha values must be >= 1.

    Returns
    -------
    transition_prob :
        Transition probability matrix, shape ``(n_states, n_states)``.
        Each row is normalized to sum to 1.

    Notes
    -----
    The current implementation requires Dirichlet prior parameters alpha >= 1.
    Support for sparse priors (0 < alpha < 1) may be added in a future version
    using alternative optimization methods.
    """

    if dirichlet_prior_alphas is not None:
        new_transition_prob = joint_posterior + dirichlet_prior_alphas - 1
    else:
        new_transition_prob = joint_posterior

    # Normalize rows
    new_transition_prob /= jnp.sum(new_transition_prob, axis=1, keepdims=True)

    return new_transition_prob


@partial(jax.jit, static_argnames=["negative_log_likelihood_func"])
def _m_step_scale_gaussian_observations(
    log_scale: GLMScale, y, rate, posteriors, negative_log_likelihood_func
) -> Tuple[GLMScale, None]:
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

    return GLMScale(optimized_log_scale), None
