"""Utilities for GLM HMM."""

from typing import Any, Callable

import jax
import jax.numpy as jnp
from numpy.typing import NDArray

from ..glm.params import GLMParams
from ..tree_utils import pytree_map_and_reduce

Array = NDArray | jax.numpy.ndarray


def initialize_new_session(n_samples, is_new_session):
    """Initialize new session indicator."""
    # Revise if the data is one single session or multiple sessions.
    # If new_sess is not provided, assume one session
    if is_new_session is None:
        # default: all False, but first time bin must be True
        is_new_session = jax.lax.dynamic_update_index_in_dim(
            jnp.zeros(n_samples, dtype=bool), True, 0, axis=0
        )
    else:
        # use the user-provided tree, but force the first time bin to be True
        is_new_session = jax.lax.dynamic_update_index_in_dim(
            jnp.asarray(is_new_session, dtype=bool), True, 0, axis=0
        )

    return is_new_session


def compute_rate_per_state(
    X: Any, glm_params: GLMParams, inverse_link_function: Callable
) -> Array:
    """Compute the GLM mean per state."""
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
