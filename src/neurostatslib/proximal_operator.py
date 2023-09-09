from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp


def _norm2_masked(weight_neuron: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Group-by-group norm 2.

    Parameters
    ----------
    weight_neuron:
        The feature vector for a neuron. Shape (n_features, ).
    mask:
        The mask vector for group. mask[i] = 1, if the i-th element of weight_neuron
        belongs to the group, 0 otherwise.

    Returns
    -------
    :
        The norm of the weight vector corresponding to the feature in mask.
    Notes
    -----
        The proximal gradient operator is described in article [1], Proposition 1.

        .. [1] Yuan, Ming, and Yi Lin. "Model selection and estimation in regression with grouped variables."
            Journal of the Royal Statistical Society Series B: Statistical Methodology 68.1 (2006): 49-67.
    """
    return jnp.linalg.norm(weight_neuron * mask, 2) / jnp.sqrt(mask.sum())


# vectorize the norm function above
# [(n_neurons, n_features), (n_groups, n_features)] -> (n_neurons, n_groups)
_vmap_norm2_masked_1 = jax.vmap(_norm2_masked, in_axes=(0, None), out_axes=0)
_vmap_norm2_masked_2 = jax.vmap(_vmap_norm2_masked_1, in_axes=(None, 0), out_axes=1)


def prox_group_lasso(
    params: Tuple[jnp.ndarray, jnp.ndarray],
    alpha: float,
    mask: jnp.ndarray,
    scaling: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Proximal gradient for group lasso.

    Parameters
    ----------
    params:
        Weights, shape (n_neurons, n_features)
    alpha:
        The regularization hyperparameter.
    mask:
        ND array of 0,1 as float32, feature mask. size (n_groups x n_features)
    scaling:
        The scaling factor for the group-lasso (it will be set
        depending on the step-size).
    Returns
    -------
        The rescaled weights.
    """
    weights, intercepts = params
    # returns a (n_neurons, n_groups) matrix of norm 2s.
    l2_norm = _vmap_norm2_masked_2(weights, mask)
    factor = 1 - alpha * scaling / l2_norm
    factor = jax.nn.relu(factor)
    # Avoid shrinkage of features that do not belong to any group
    # by setting the shrinkage factor to 1.
    not_regularized = jnp.outer(jnp.ones(factor.shape[0]), 1 - mask.sum(axis=0))
    return weights * (factor @ mask + not_regularized), intercepts
