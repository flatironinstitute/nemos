r"""Collection of proximal operators.

## Definition

In optimization theory, the proximal operator is a mathematical tool used to solve non-differentiable optimization
problems or to simplify complex ones.

The proximal operator of a function $ f: \mathbb{R}^n \rightarrow \mathbb{R} \cup \{+\infty\} $ is defined as follows:

$$
\text{prox}_f(v) = \arg\min_x \left( f(x) + \frac{1}{2}\Vert x - v\Vert_2 ^2 \right)
$$

Here $ \text{prox}_f(v) $ is the value of $ x $ that minimizes the sum of the function $ f(x) $ and the
squared Euclidean distance between $ x $ and some point $ v $. The parameter $ f $ typically represents
a regularization term or a penalty in the optimization problem, and $ v $ is typically a vector
in the domain of $ f $.

The proximal operator can be thought of as a generalization of the projection operator. When $ f $ is the
indicator function of a convex set $ C $, then $ \text{prox}_f $ is the projection onto $ C $, since
it finds the point in $ C $ closest to $ v $.

Proximal operators are central to the implementation of proximal gradient[^1] methods and algorithms like where they
help to break down complex optimization problems into simpler sub-problems that can be solved iteratively.

## Proximal Operators in Proximal Gradient Algorithms

Proximal gradient algorithms are designed to solve optimization problems of the form:

$$
\min_{x \in \mathbb{R}^n} g(x) + f(x)
$$

where $ g $ is a differentiable (and typically convex) function, and $ f $ is a (possibly non-differentiable) convex
function that imposes certain structure or sparsity in the solution. The proximal gradient method updates the
solution iteratively through a two-step process:

1. **Gradient Step on $ g $**: Take a step towards the direction of the negative gradient of $ g $ at the current
estimate $ x_k $, with a step size $ \alpha_k $, leading to an intermediate estimate $ y_k $:
   $$
   y_k = x_k - \alpha_k \nabla g(x_k)
   $$
2. **Proximal Step on $ f $**: Apply the proximal operator of $ f $ to the intermediate
estimate $ y_k $ to obtain the new estimate $ x_{k+1} $:

   $$
   x_{k+1} = \text{prox}_{ f}(y_k) = \arg\min_x \left( f(x) + \frac{1}{2\alpha_k}\Vert x - y_k \Vert_2 ^2 \right)
   $$

The gradient step aims to reduce the value of the smooth part of the objective $ g $, and the proximal step
takes care of the non-smooth part $ f $, often enforcing properties like sparsity due to regularization terms
such as the $ \ell_1 $ norm.

By iteratively performing these two steps, the proximal gradient algorithm converges to a solution that
balances minimizing the differentiable part $ g $ while respecting the structure imposed by the non-differentiable
part $ f $. The proximal operator effectively "proximates" the solution at each iteration,
taking into account the influence of the non-smooth term $ f $, which would be otherwise challenging to
handle due to its potential non-differentiability.

[^1]: Parikh, Neal, and Stephen Boyd. "Proximal Algorithms, ser. Foundations and Trends (r) in Optimization." (2013).
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
