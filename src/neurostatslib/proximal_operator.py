r"""Collection of proximal operators.

See the theory note on "Proximal Methods" in the package documentation for an introduction to proximal
operators and the Proximal Gradient algorithm.
"""
from typing import Tuple

import jax
import jax.numpy as jnp


def _norm2_masked(weight_neuron: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Euclidean norm of the group.

    Calculate the Euclidean norm of the weights for a specified group within a
    neuron's feature vector.

    This function computes the norm of elements that are indicated by the mask array.
    If 'mask' were of boolean type, this operation would be equivalent to performing
    `jnp.linalg.norm(weight_neuron[mask], 2)` followed by dividing the result by the
    square root of the sum of the mask (assuming each group has at least 1 feature).

    Parameters
    ----------
    weight_neuron:
        The feature vector for a neuron. Shape (n_features, ).
    mask:
        The mask vector for group. mask[i] = 1, if the i-th element of weight_neuron
        belongs to the group, 0 otherwise. Shape (n_features, )

    Returns
    -------
    :
        The norm of the weight vector corresponding to the feature in mask.

    Notes
    -----
        The proximal gradient operator is described in Ming at al.[^1], Proposition 1.

    [^1]:
        Yuan, Ming, and Yi Lin. "Model selection and estimation in regression with grouped variables."
        Journal of the Royal Statistical Society Series B: Statistical Methodology 68.1 (2006): 49-67.
    """
    return jnp.linalg.norm(weight_neuron * mask, 2) / jnp.sqrt(mask.sum())


# vectorize the norm function above
# [(n_neurons, n_features), (n_features)] -> (n_neurons, )
_vmap_norm2_masked_1 = jax.vmap(_norm2_masked, in_axes=(0, None), out_axes=0)
# [(n_neurons, n_features), (n_groups, n_features)] -> (n_neurons, n_groups)
_vmap_norm2_masked_2 = jax.vmap(_vmap_norm2_masked_1, in_axes=(None, 0), out_axes=1)


def prox_group_lasso(
    params: Tuple[jnp.ndarray, jnp.ndarray],
    regularizer_strength: float,
    mask: jnp.ndarray,
    scaling: float = 1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    r"""Proximal gradient operator for group Lasso.

    Parameters
    ----------
    params:
        Weights, shape (n_neurons, n_features)
    regularizer_strength:
        The regularization hyperparameter.
    mask:
        ND array of 0,1 as float32, feature mask. size (n_groups, n_features)
    scaling:
        The scaling factor for the group-lasso (it will be set
        depending on the step-size).

    Returns
    -------
    :
        The rescaled weights.

    Notes
    -----
    This function implements the proximal operator for a group-Lasso penalization which
    can be derived in analytical form.
    The proximal operator equation are,

    $$
    \text{prox}(\beta_g) = \text{min}_{\beta} \left[ \lambda  \sum\_{g=1}^G \Vert \beta_g \Vert_2 +
     \frac{1}{2} \Vert \hat{\beta} - \beta \Vert_2^2
    \right],
    $$
    where $G$ is the number of groups, and $\beta_g$ is the parameter vector
    associated with the $g$-th group.
    The analytical solution[^1] for the beta is,

    $$
    \text{prox}(\beta\_g) = \max \left(1 - \frac{\lambda \sqrt{p\_g}}{\Vert \hat{\beta}\_g \Vert_2},
     0\right) \cdot \hat{\beta}\_g,
    $$
    where $p_g$ is the dimensionality of $\beta\_g$ and $\hat{\beta}$ is typically the gradient step
    of the un-regularized optimization objective function. It's easy to see how the group-Lasso
    proximal operator acts as a shrinkage factor for the un-penalize update, and the half-rectification
    non-linearity that effectively sets to zero group of coefficients satisfying,
    $$
    \Vert \hat{\beta}\_g \Vert_2 \le \frac{1}{\lambda \sqrt{p\_g}}.
    $$

    [^1]:
        Yuan, Ming, and Yi Lin. "Model selection and estimation in regression with grouped variables."
        Journal of the Royal Statistical Society Series B: Statistical Methodology 68.1 (2006): 49-67.
    """
    weights, intercepts = params
    # [(n_neurons, n_features), (n_groups, n_features)] -> (n_neurons, n_groups)
    l2_norm = _vmap_norm2_masked_2(weights, mask)
    factor = 1 - regularizer_strength * scaling / l2_norm
    factor = jax.nn.relu(factor)
    # Avoid shrinkage of features that do not belong to any group
    # by setting the shrinkage factor to 1.
    not_regularized = jnp.outer(jnp.ones(factor.shape[0]), 1 - mask.sum(axis=0))
    return weights * (factor @ mask + not_regularized), intercepts
