"""Implementations of analyzing M-step updates."""

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from .params import GLMScale


def _add_prior(log_val: jnp.ndarray, offset: jnp.ndarray):
    """Add prior offset in log-space.

    Computes log(exp(log_val) + offset) in a numerically stable way.

    This function assumes offset >= 0, which corresponds to Dirichlet
    prior parameters alpha >= 1.

    Parameters
    ----------
    log_val :
        Log of the value to which the offset is added.
    offset :
        The offset to add (e.g., alpha - 1 for Dirichlet prior).
        Must be >= 0 (i.e., alpha >= 1).

    Returns
    -------
    :
        log(exp(log_val) + offset)
    """
    # For offset > 0: use logaddexp for numerical stability
    # For offset = 0: return log_val unchanged
    result = jnp.where(
        offset > 0,
        jnp.logaddexp(log_val, jnp.log(jnp.maximum(offset, 1e-10))),
        log_val,
    )
    return result


_vmap_add_prior = jax.vmap(_add_prior)


def _analytical_m_step_log_initial_prob(
    log_posteriors: jnp.ndarray,
    is_new_session: jnp.ndarray,
    dirichlet_prior_alphas: Optional[jnp.ndarray] = None,
):
    """
    Calculate the M-step for initial state probabilities in log-space.

    Computes the maximum likelihood estimate (or MAP estimate with prior) of the
    initial state distribution in log-space for numerical stability.

    Parameters
    ----------
    log_posteriors :
        The log posterior distribution over latent states, shape ``(n_time_bins, n_states)``.
    is_new_session :
        Boolean array indicating session start points, shape ``(n_time_bins,)``.
    dirichlet_prior_alphas :
        The parameters of the Dirichlet prior for the initial distribution,
        shape ``(n_states,)``. If None, uses a flat (uniform) prior.
        **Note**: All alpha values must be >= 1 for the current implementation.

    Returns
    -------
    log_initial_prob :
        Updated initial state log-probabilities, shape ``(n_states,)``.
        Normalized in log-space.

    Notes
    -----
    The current implementation requires Dirichlet prior parameters alpha >= 1.
    Support for sparse priors (0 < alpha < 1) may be added in a future version
    using alternative optimization methods.
    """
    # Mask out non-session-start time points by setting to -inf
    masked_log_posteriors = jnp.where(
        is_new_session[:, jnp.newaxis], log_posteriors, -jnp.inf
    )

    # Sum over time in log-space (logsumexp ignores -inf values)
    log_tmp_initial_prob = jax.scipy.special.logsumexp(masked_log_posteriors, axis=0)

    if dirichlet_prior_alphas is not None:
        prior_offset = dirichlet_prior_alphas - 1
        log_numerator = _vmap_add_prior(log_tmp_initial_prob, prior_offset)
    else:
        log_numerator = log_tmp_initial_prob

    # Normalize in log-space
    log_sum = jax.scipy.special.logsumexp(log_numerator)
    log_initial_prob = log_numerator - log_sum

    return log_initial_prob


def _analytical_m_step_log_transition_prob(
    log_joint_posterior: jnp.ndarray,
    dirichlet_prior_alphas: Optional[jnp.ndarray] = None,
):
    """
    Calculate the M-step for state transition probabilities in log-space.

    Computes the maximum likelihood estimate (or MAP estimate with prior) of the
    transition matrix in log-space for numerical stability.

    Parameters
    ----------
    log_joint_posterior:
        Log of expected counts of transitions from state i to state j,
        shape ``(n_states, n_states)``. Computed as logsumexp(log_xis, axis=0).
    dirichlet_prior_alphas:
        The parameters of the Dirichlet prior for each row of the transition matrix,
        shape ``(n_states, n_states)``. If None, uses a flat (uniform) prior.
        **Note**: All alpha values must be >= 1 for the current implementation.

    Returns
    -------
    log_transition_prob:
        Updated log transition probability matrix, shape ``(n_states, n_states)``.
        Each row is normalized in log-space.

    Notes
    -----
    The current implementation requires Dirichlet prior parameters alpha >= 1.
    Support for sparse priors (0 < alpha < 1) may be added in a future version
    using alternative optimization methods.
    """
    if dirichlet_prior_alphas is not None:
        prior_offset = dirichlet_prior_alphas - 1
        log_numerator = _vmap_add_prior(log_joint_posterior, prior_offset)
    else:
        log_numerator = log_joint_posterior

    # Normalize each row in log-space
    log_row_sums = jax.scipy.special.logsumexp(log_numerator, axis=1, keepdims=True)
    log_transition_prob = log_numerator - log_row_sums

    return log_transition_prob


@partial(jax.jit, static_argnames=["negative_log_likelihood_func"])
def _m_step_scale_gaussian_observations(
    log_scale: GLMScale, y, rate, posteriors, negative_log_likelihood_func
) -> Tuple[GLMScale, None, None]:
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

    return GLMScale(optimized_log_scale), None, None
