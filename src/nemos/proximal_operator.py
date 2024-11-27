r"""Collection of proximal operators.

Proximal operators are a mathematical tools used to solve non-differentiable optimization
problems or to simplify complex ones.

A classical use-case for proximal operator is that of minimizing a penalized loss function where the
penalization is non-differentiable (Lasso, group Lasso etc.). In proximal gradient algorithms, proximal
operators are used to find the parameters that balance the minimization of the penalty term with
 the proximity to the gradient descent update of the un-penalized loss.

More formally, proximal operators solve the minimization problem,

$$
\\text{prox}_f(\bm{v}) = \arg\min_{\bm{x}} \left( f(\bm{x}) + \frac{1}{2}\Vert \bm{x} - \bm{v}\Vert_2 ^2 \right)
$$


Where $f$ is usually the non-differentiable penalization term, and $\bm{v}$  is the parameter update of the
un-penalized loss function. The first term controls the penalization magnitude, the second the proximity
with the gradient based update.

References
----------
[1]  Parikh, Neal, and Stephen Boyd. *"Proximal Algorithms, ser. Foundations and Trends (r) in Optimization."* (2013).
"""

from typing import Any, Optional, Tuple

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
        belongs to the group, 0 otherwise. Shape (n_features, ).

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
        Weights, shape (n_neurons, n_features) or pytree of same; intercept,
        shape (n_neurons, )
    regularizer_strength:
        The regularization hyperparameter.
    mask:
        ND array of 0,1 as float32, feature mask. size (n_groups, n_features)
        or pytree of same.
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
    \text{prox}(\beta_g) = \text{min}_{\beta} \left[ \lambda  \sum_{g=1}^G \Vert \beta_g \Vert_2 +
     \frac{1}{2} \Vert \hat{\beta} - \beta \Vert_2^2
    \right],
    $$
    where $G$ is the number of groups, and $\beta_g$ is the parameter vector
    associated with the $g$-th group.
    The analytical solution[$^{[1]}$](#references). for the beta is,

    $$
    \text{prox}(\beta_g) = \max \left(1 - \frac{\lambda \sqrt{p_g}}{\Vert \hat{\beta}_g \Vert_2},
     0\right) \cdot \hat{\beta}_g,
    $$
    where $p_g$ is the dimensionality of $\beta_g$ and $\hat{\beta}$ is typically the gradient step
    of the un-regularized optimization objective function. It's easy to see how the group-Lasso
    proximal operator acts as a shrinkage factor for the un-penalize update, and the half-rectification
    non-linearity that effectively sets to zero group of coefficients satisfying,
    $$
    \Vert \hat{\beta}_g \Vert_2 \le \frac{1}{\lambda \sqrt{p_g}}.
    $$

    # References
    ------------
    [1] Yuan, Ming, and Yi Lin. "Model selection and estimation in regression with grouped variables."
    Journal of the Royal Statistical Society Series B: Statistical Methodology 68.1 (2006): 49-67.

    """
    weights, intercepts = params
    shape = weights.shape
    # divide the reg strength by the number of neurons
    regularizer_strength /= intercepts.shape[0]
    # add an extra dim if not 2D, do nothing otherwise.
    weights = jnp.atleast_2d(weights.T)
    # [(n_neurons, n_features), (n_groups, n_features)] -> (n_neurons, n_groups)
    l2_norm = _vmap_norm2_masked_2(weights, mask)
    factor = 1 - regularizer_strength * scaling / l2_norm
    factor = jax.nn.relu(factor)
    # Avoid shrinkage of features that do not belong to any group
    # by setting the shrinkage factor to 1.
    not_regularized = jnp.outer(jnp.ones(factor.shape[0]), 1 - mask.sum(axis=0))
    return (weights * (factor @ mask + not_regularized)).T.reshape(shape), intercepts


def prox_lasso(x: Any, l1reg: Optional[Any] = None, scaling: float = 1.0) -> Any:
    r"""Proximal operator for the l1 norm, i.e., soft-thresholding operator.

    Minimizes the following function:

    $$
      \underset{y}{\text{argmin}} ~ \frac{1}{2} ||x - y||_2^2
      + \text{scaling} \cdot \text{l1reg} \cdot ||y||_1
    $$

    When `l1reg` is a pytree, the weights are applied coordinate-wise.

    Parameters
    ----------
    x :
        Input pytree.
    l1reg :
        Regularization strength, float or pytree with the same structure as `x`. Default is None.
    scaling : float, optional
        A scaling factor. Default is 1.0.

    Returns
    -------
    :
        Output pytree with the same structure as `x`.
    """
    if l1reg is None:
        l1reg = 1.0

    if jnp.isscalar(l1reg):
        l1reg = jax.tree_util.tree_map(lambda y: l1reg * jnp.ones_like(y), x)

    def fun(u, v):
        return jnp.sign(u) * jax.nn.relu(jnp.abs(u) - v * scaling)

    return jax.tree_util.tree_map(fun, x, l1reg)
