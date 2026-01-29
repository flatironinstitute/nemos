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

from functools import partial
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as tree_util

from nemos.tree_utils import pytree_map_and_reduce


def prox_none(x: Any, hyperparams: Optional[Any] = None, scaling: float = 1.0) -> Any:
    """Identity proximal operator."""
    del hyperparams, scaling
    return x


def prox_ridge(x: Any, l2reg: Optional[float] = None, scaling: float = 1.0) -> Any:
    r"""Proximal operator for the squared l2 norm. From JAXopt.

    .. math::

      \underset{y}{\text{argmin}} ~ \frac{1}{2} ||x - y||_2^2
      + \text{scaling} \cdot \text{l2reg} \cdot ||y||_2^2

    Parameters
    ----------
    x :
        Input pytree.
    l2reg :
        Regularization strength. Default is None (interpreted as 1.0).
    scaling :
        A scaling factor.

    Returns
    -------
    :
        Output pytree with the same structure as ``x``.
    """
    if l2reg is None:
        l2reg = 1.0

    factor = 1.0 / (1.0 + scaling * l2reg)
    return tree_util.tree_map(lambda y: factor * y, x)


def compute_normalization(mask):
    """Compute normalization constant over group size."""
    return jnp.sqrt(
        pytree_map_and_reduce(
            lambda mi: jnp.sum(mi.reshape(mi.shape[0], -1), axis=1), sum, mask
        )
    )


@partial(jax.jit, static_argnames=("normalize",))
def masked_norm_2(x: Any, mask: Any, normalize: bool = True) -> Any:
    """Euclidean norm of the group.

    Calculate the Euclidean norm of the weights for a specified group within a
    neuron's feature vector.

    This function computes the norm of elements that are indicated by the mask array.
    If 'mask' were of boolean type, this operation would be equivalent to performing
    `jnp.linalg.norm(weight_neuron[mask], 2)` followed by dividing the result by the
    square root of the sum of the mask (assuming each group has at least 1 feature).

    Parameters
    ----------
    x:
        A PyTree with array leaves.
    mask:
        PyTree of ND array of 0,1 as floats with the same struct as x. The shape of the i-th leaf is
        ``(n_groups, *x_leaf[i].shape)``, where x_leaf is the ``jax.tree_util.tree_leaves(x)[i]``.
    normalize:
        True if normalization over the sqrt of the group size is needed.

    Returns
    -------
    :
        The norm of the weight vector corresponding to the feature in mask, scaled by the
        squared root of the size of the vector.
    """
    # [(n_groups, )]
    norms = jnp.sqrt(
        pytree_map_and_reduce(
            lambda xi, mi: jnp.sum(
                (xi.reshape(1, -1) * mi.reshape(mi.shape[0], -1)) ** 2, axis=1
            ),
            sum,
            x,
            mask,
        )
    )
    if normalize:
        # [(n_groups, )]
        sqrt_group_size = compute_normalization(mask)
        norms /= sqrt_group_size
    return norms


def prox_group_lasso(
    x: Any,
    regularizer_strength: float,
    mask: Any,
    scaling: float = 1.0,
) -> Any:
    r"""Proximal gradient operator for group Lasso.

    Parameters
    ----------
    x:
        PyTree of arrays;
    regularizer_strength:
        The regularization hyperparameter.
    mask:
        PyTree of ND array of 0,1 as floats with the same struct as x. The shape of the i-th leaf is
        ``(n_groups, *x_leaf[i].shape)``, where x_leaf is the ``jax.tree_util.tree_leaves(x)[i]``.
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
    # shape: (n_groups, )
    l2_norm = masked_norm_2(x, mask)
    # compute shrinkage
    factor = 1 - regularizer_strength * scaling / l2_norm
    factor = jax.nn.relu(factor)

    # the leaf dim of regularized match that of x
    regularized = jax.tree_util.tree_map(lambda mi: mi.sum(axis=0).astype(bool), mask)
    return jax.tree_util.tree_map(
        lambda r, xi, mi: jnp.where(r, xi * jnp.einsum("i, i...->...", factor, mi), xi),
        regularized,
        x,
        mask,
    )


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
    scaling :
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


def prox_elastic_net(
    x: Any, hyperparams: Optional[Tuple[Any, Any]] = None, scaling: float = 1.0
) -> Any:
    r"""Proximal operator for the elastic net.

    .. math::

      \underset{y}{\text{argmin}} ~ \frac{1}{2} ||x - y||_2^2
      + \text{scaling} \cdot \text{hyperparams[0]} \cdot g(y)

    where :math:`g(y) = ||y||_1 + \text{hyperparams[1]} \cdot 0.5 \cdot ||y||_2^2`.

    Parameters
    ----------
    x :
        Input pytree.
    hyperparams :
        A tuple, where both ``hyperparams[0]`` and ``hyperparams[1]`` can be either floats or pytrees with the same
        structure as ``x``.
    scaling :
        A scaling factor.

    Returns
    -------
    :
        Output pytree, with the same structure as ``x``.
    """
    if hyperparams is None:
        hyperparams = (0.5, 1.0)

    # hyperparams[0] = regularizer_strength*regularizer_ratio
    lam = (
        tree_util.tree_map(lambda y: hyperparams[0] * jnp.ones_like(y), x)
        if isinstance(hyperparams[0], float)
        else hyperparams[0]
    )
    # hyperparams[1] = (1 - regularizer_ratio)/regularizer_ratio
    gam = (
        tree_util.tree_map(lambda y: hyperparams[1] * jnp.ones_like(y), x)
        if isinstance(hyperparams[1], float)
        else hyperparams[1]
    )

    def prox_l1(u, lambd):
        return jnp.sign(u) * jax.nn.relu(jnp.abs(u) - lambd)

    def fun(u, lambd, gamma):
        return prox_l1(u, scaling * lambd) / (1.0 + scaling * lambd * gamma)

    return tree_util.tree_map(fun, x, lam, gam)
