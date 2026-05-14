"""Implementations of analyzing M-step updates."""

from typing import Optional

import jax
import jax.numpy as jnp


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
