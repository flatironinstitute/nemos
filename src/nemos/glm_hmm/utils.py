"""Utilities for GLM HMM."""

from typing import Any, Callable

import jax
import jax.numpy as jnp

from ..hmm.utils import Array
from ..glm.params import GLMParams
from ..tree_utils import pytree_map_and_reduce


def compute_rate_per_state(
    X: Any, glm_params: GLMParams, inverse_link_function: Callable
) -> Array:
    """
    Compute GLM predicted rates for all latent states.

    Evaluates the GLM's predicted mean response (rate) for each latent state
    by computing the linear combination of features and applying the inverse
    link function. Handles both single-observation and population (multi-neuron)
    GLMs through pytree broadcasting.

    Parameters
    ----------
    X :
        Design matrix as a pytree with leaves of shape ``(n_time_bins, n_features)``.
    glm_params :
        GLM coefficients and intercepts. For single observation fits, coefficients
        have shape ``(n_features, n_states)`` and intercepts have shape ``(n_states,)``.
        For population fits, coefficients have shape ``(n_features, n_neurons, n_states)``
        and intercepts have shape ``(n_states, n_neurons)``.
    inverse_link_function :
        Function mapping linear predictors to the mean of the observation distribution
        (e.g., ``jnp.exp`` for Poisson, ``jax.nn.sigmoid`` for Bernoulli).

    Returns
    -------
    predicted_rate_given_state :
        Predicted rates for each state. For single observation fits, returns
        shape ``(n_time_bins, n_states)``. For population fits, returns
        shape ``(n_time_bins, n_neurons, n_states)``.

    Notes
    -----
    The function automatically detects whether a population GLM is being used
    by checking the dimensionality of the coefficient arrays and applies the
    appropriate einsum operation for efficient computation.
    """
    coef, intercept = glm_params.coef, glm_params.intercept

    # Predicted y
    if jax.tree_util.tree_leaves(coef)[0].ndim > 2:
        lin_comb = pytree_map_and_reduce(
            lambda x, w: jnp.einsum("ik, kjw->ijw", x, w), sum, X, coef
        )
    else:
        lin_comb = pytree_map_and_reduce(lambda x, w: jnp.matmul(x, w), sum, X, coef)
    predicted_rate_given_state = inverse_link_function(lin_comb + intercept)
    return predicted_rate_given_state
