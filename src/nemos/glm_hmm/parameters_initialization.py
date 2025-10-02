"""Initialization functions and related utility functions."""

import jax
import jax.numpy as jnp
from numpy.typing import NDArray

from ..typing import DESIGN_INPUT_TYPE


def random_projection_init(
    n_states: int, X: DESIGN_INPUT_TYPE, y: NDArray, random_key=jax.random.PRNGKey(123)
):
    """Initialize projections."""
    n_features = X.shape[1]
    return 0.1 * jax.random.normal(random_key, (n_features, n_states))


def sticky_transition_proba_init(n_states: int, prob_stay=0.95):
    """Initialize transition probabilities."""
    # assume n_state is > 1
    prob_leave = (1 - prob_stay) / (n_states - 1)
    return jnp.full((n_states, n_states), prob_leave) + jnp.diag(
        (prob_stay - prob_leave) * jnp.ones(n_states)
    )


def uniform_initial_proba_init(n_states: int, random_key=jax.random.PRNGKey(124)):
    """Initialize initial state probabilities."""
    prob = jax.random.uniform(random_key, (n_states,), minval=0, maxval=1)
    return prob / jnp.sum(prob)


def resolve_projection_init_function():
    """Resolve the projection initialization function."""
    pass
