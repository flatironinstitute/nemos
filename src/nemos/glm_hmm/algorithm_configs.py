"""Object facilitating EM configuration."""

from functools import partial
from typing import Callable, Type

import equinox as eqx
import jax
import jax.numpy as jnp

from ..glm.params import GLMParams
from ..observation_models import (
    BernoulliObservations,
    GaussianObservations,
    Observations,
    PoissonObservations,
)
from .m_step_analytical_updates import _m_step_scale_gaussian_observations
from .utils import Array, compute_rate_per_state

_NO_SCALE = (PoissonObservations, BernoulliObservations)
_ANALYTICAL_SCALE_UPDATE: dict[Type[Observations], Callable] = {
    GaussianObservations: _m_step_scale_gaussian_observations
}


def _posterior_weighted_objective_impl(
    y: Array,
    predicted_rate: Array,
    posteriors: Array,
    negative_log_likelihood_func: Callable,
    scale: Array | None = None,
):
    """
    Core implementation of posterior-weighted negative log-likelihood computation.

    Computes the expected negative log-likelihood by weighting sample-wise NLL
    values with posterior probabilities over states. This shared implementation
    is used by both coefficient/intercept and scale optimization objectives.

    Parameters
    ----------
    y :
        Target observations, shape ``(n_time_bins,)`` for single observations
        or ``(n_time_bins, n_neurons)`` for population data.
    predicted_rate :
        Predicted mean responses for each state, shape ``(n_time_bins, n_states)``
        for single observations or ``(n_time_bins, n_neurons, n_states)`` for
        population data.
    posteriors :
        Posterior probabilities over states, shape ``(n_time_bins, n_states)``.
    negative_log_likelihood_func :
        Function computing negative log-likelihood. Must accept ``(y, predicted_rate)``
        or ``(y, predicted_rate, scale)`` depending on whether scale is provided.
    scale :
        Optional scale parameters for distributions that require them,
        shape ``(n_states,)`` or ``(n_neurons, n_states)``.

    Returns
    -------
    weighted_nll :
        Scalar posterior-weighted negative log-likelihood summed over all
        time bins, states, and (if applicable) neurons.

    Notes
    -----
    For population GLMs with multiple neurons, the function automatically
    sums negative log-likelihoods across neurons before weighting by posteriors,
    assuming conditional independence of neurons given the latent state.
    """
    if scale is None:
        nll = negative_log_likelihood_func(y, predicted_rate)
    else:
        nll = negative_log_likelihood_func(y, predicted_rate, scale)

    if nll.ndim > 2:
        nll = nll.sum(axis=1)  # sum over neurons

    return jnp.sum(nll * posteriors)


@partial(
    jax.jit, static_argnames=["inverse_link_function", "negative_log_likelihood_func"]
)
def posterior_weighted_glm_negative_log_likelihood(
    glm_params: GLMParams,
    X: Array,
    y: Array,
    posteriors: Array,
    inverse_link_function: Callable,
    negative_log_likelihood_func: Callable,
):
    """
    Compute the posterior-weighted negative log-likelihood for GLM coefficients.

    Computes the expected negative log-likelihood as a function of GLM
    coefficients and intercept, where the expectation is taken over the
    posterior distribution over states. Rates are recomputed for each
    evaluation since they depend on the coefficients being optimized.

    This is the objective function minimized during the M-step to update
    GLM coefficients and intercept using unnormalized negative log-likelihood.

    Parameters
    ----------
    glm_params:
        Projection coefficients and intercept for the GLM.
    X:
        Design matrix of observations.
    y:
        Target responses.
    posteriors:
        Posterior probabilities over states, shape (n_time_bins, n_states).
    inverse_link_function:
        Function mapping linear predictors to rates.
    negative_log_likelihood_func:
        Unnormalized negative log-likelihood function.

    Returns
    -------
    :
        Scalar negative log-likelihood weighted by posteriors:
        sum_t sum_k posterior[t,k] * nll[t,k]
    """
    predicted_rate = compute_rate_per_state(X, glm_params, inverse_link_function)
    return _posterior_weighted_objective_impl(
        predicted_rate, y, posteriors, negative_log_likelihood_func
    )


@partial(jax.jit, static_argnames=["negative_log_likelihood_func"])
def posterior_weighted_glm_negative_log_likelihood_scale(
    glm_scale: Array,
    y: Array,
    predicted_rate: Array,
    posteriors: Array,
    negative_log_likelihood_func: Callable,
):
    """
    Compute the posterior-weighted negative log-likelihood for GLM scale.

    Computes the expected negative log-likelihood as a function of GLM
    scale parameters, where the expectation is taken over the posterior
    distribution over states. This function expects pre-computed rates
    since coefficients and intercept are held fixed during scale optimization.

    This is the objective function minimized during the M-step to update
    scale parameters using normalized negative log-likelihood.

    Parameters
    ----------
    glm_scale:
        Scale parameters for the GLM.
    y:
        Target responses.
    predicted_rate:
        Pre-computed rates from fixed coefficients, shape (n_time_bins, n_states)
        or (n_time_bins, n_states, n_neurons).
    posteriors:
        Posterior probabilities over states, shape (n_time_bins, n_states).
    negative_log_likelihood_func:
        Normalized negative log-likelihood function that accepts scale.

    Returns
    -------
    :
        Scalar negative log-likelihood weighted by posteriors:
        sum_t sum_k posterior[t,k] * nll[t,k]
    """
    return _posterior_weighted_objective_impl(
        y, predicted_rate, posteriors, negative_log_likelihood_func, scale=glm_scale
    )


def prepare_ll_estep_likelihood(
    is_population_glm: bool,
    observation_model: Observations,
) -> Callable:
    """
    Prepare log-likelihood function for the E-step (forward-backward algorithm).

    This function is always needed for computing posterior probabilities over states.

    Parameters
    ----------
    is_population_glm:
        True if it is a population GLM likelihood.
    observation_model:
        The observation model.

    Returns
    -------
    log_likelihood:
        Log-likelihood function vmapped over states for E-step.
        Signature: (y, rate, scale) -> log_likelihood per state
    """

    def log_likelihood_per_sample(x, z, s):
        return observation_model.log_likelihood(
            x, z, scale=s, aggregate_sample_scores=lambda v: v
        )

    # Vectorize over the states axis
    state_axes = 2 if is_population_glm else 1

    log_likelihood_per_sample = jax.vmap(
        log_likelihood_per_sample,
        in_axes=(None, state_axes, state_axes - 1),
        out_axes=state_axes,
    )

    def log_likelihood(y, rate, scale):
        log_like = log_likelihood_per_sample(y, rate, scale)
        if is_population_glm:
            # Multi-neuron case: sum log-likelihoods across neurons
            log_like = log_like.sum(axis=1)
        return log_like

    return log_likelihood


def prepare_nll_mstep_analytical_scale(
    is_population_glm: bool,
    observation_model: Observations,
) -> Callable:
    """
    Prepare negative log-likelihood for analytical M-step scale updates.

    Use this for distributions with closed-form scale updates (e.g., Gaussian, Gamma).

    Parameters
    ----------
    is_population_glm:
        True if it is a population GLM likelihood.
    observation_model:
        The observation model.

    Returns
    -------
    negative_log_likelihood:
        Unnormalized negative log-likelihood function vmapped over states.
        Signature: (y, rate) -> negative_log_likelihood per state
    """

    def negative_log_likelihood_per_sample(x, z):
        return observation_model._negative_log_likelihood(
            x, z, aggregate_sample_scores=lambda v: v
        )

    state_axes = 2 if is_population_glm else 1

    negative_log_likelihood = jax.vmap(
        negative_log_likelihood_per_sample,
        in_axes=(None, state_axes),
        out_axes=state_axes,
    )

    return negative_log_likelihood


def prepare_nll_mstep_numerical_params(
    is_population_glm: bool,
    observation_model: Observations,
    inverse_link_function: Callable,
) -> Callable:
    """
    Prepare objective function for numerically optimizing GLM coefficients and intercept.

    Use this when GLM parameters don't have analytical M-step updates (which is typical).

    Parameters
    ----------
    is_population_glm:
        True if it is a population GLM likelihood.
    observation_model:
        The observation model.
    inverse_link_function:
        Function mapping linear predictors to rates.

    Returns
    -------
    objective:
        Objective function for optimizing GLM coefficients and intercept.
        Signature: (glm_params, design_matrix, observations, posteriors) -> scalar
    """
    state_axes = 2 if is_population_glm else 1

    def negative_log_likelihood_per_sample(x, z):
        return observation_model._negative_log_likelihood(
            x, z, aggregate_sample_scores=lambda v: v
        )

    negative_log_likelihood = jax.vmap(
        negative_log_likelihood_per_sample,
        in_axes=(None, state_axes),
        out_axes=state_axes,
    )

    def objective(glm_params, design_matrix, observations, posteriors):
        return posterior_weighted_glm_negative_log_likelihood(
            glm_params,
            X=design_matrix,
            y=observations,
            posteriors=posteriors,
            inverse_link_function=inverse_link_function,
            negative_log_likelihood_func=negative_log_likelihood,
        )

    return objective


def prepare_objective_mstep_numerical_scale(
    is_population_glm: bool,
    observation_model: Observations,
) -> Callable:
    """
    Prepare objective function for numerically optimizing GLM scale parameters.

    Use this for distributions without closed-form scale updates (e.g., NegativeBinomial,
    Gamma if analytical derivation is unavailable).

    Parameters
    ----------
    is_population_glm:
        True if it is a population GLM likelihood.
    observation_model:
        The observation model.

    Returns
    -------
    objective:
        Objective function for optimizing scale parameters.
        Signature: (scale, observations, predicted_rate, posteriors) -> scalar
    """
    state_axes = 2 if is_population_glm else 1

    def norm_negative_log_likelihood_per_sample(x, z, s):
        return -1 * observation_model.log_likelihood(
            x, z, scale=s, aggregate_sample_scores=lambda v: v
        )

    norm_negative_log_likelihood = jax.vmap(
        norm_negative_log_likelihood_per_sample,
        in_axes=(None, state_axes, state_axes - 1),
        out_axes=state_axes,
    )

    def objective(scale, observations, predicted_rate, posteriors):
        return posterior_weighted_glm_negative_log_likelihood_scale(
            scale,
            observations,
            predicted_rate,
            posteriors,
            negative_log_likelihood_func=norm_negative_log_likelihood,
        )

    return objective


def get_analytical_scale_update(
    observation_model: Observations, is_population_glm: bool
) -> None | Callable:
    """
    Retrieve analytical M-step update function for scale parameters if available.

    Checks if the observation model has a registered closed-form solution for
    scale parameter updates. If available, returns a configured update function
    with the appropriate negative log-likelihood already bound.

    Parameters
    ----------
    observation_model :
        The observation model instance (e.g., GaussianObservations, PoissonObservations).
    is_population_glm :
        Whether this is a population GLM with multiple neurons.

    Returns
    -------
    update_func :
        Configured analytical update function with signature
        ``(scale, y, rate, posteriors) -> optimized_scale``,
        or None if no analytical update exists for this observation model.

    Notes
    -----
    Currently supported analytical updates:
    - GaussianObservations: Closed-form variance update

    Observation models without analytical updates (e.g., PoissonObservations,
    BernoulliObservations) should use numerical optimization instead.
    """
    if type(observation_model) in _ANALYTICAL_SCALE_UPDATE:
        update = _ANALYTICAL_SCALE_UPDATE[type(observation_model)]
        nll_fnc = prepare_nll_mstep_analytical_scale(
            is_population_glm, observation_model=observation_model
        )
        return eqx.Partial(update, negative_log_likelihood_func=nll_fnc)
    return None
