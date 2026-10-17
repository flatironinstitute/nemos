---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  name: nemos_313
  display_name: Python 3 (ipykernel)
  language: python
---

```{code-cell} ipython3
:tags: [hide-input]

%matplotlib inline
import warnings
import jax

warnings.filterwarnings(
    "ignore",
    message="plotting functions contained within `_documentation_utils` are intended for nemos's documentation.",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message="Ignoring cached namespace 'core'",
    category=UserWarning,
)

jax.config.update("jax_enable_x64", True)
```

(regularizers-overview)=
# Regularizers

A `Regularizer` object implements the machinery for penalizing a loss function and for supporting non-smooth optimization.

Models such as [`GLM`](nemos.glm.GLM) and [`GLMHMM`](nemos.glm_hmm.GLMHMM) store one such object as an attribute and use it to turn their loss into the objective they minimize. This page describes the object itself, which you need if you want to penalize a loss of your own, or to know which solvers a penalty allows.

```{contents}
:local:
:depth: 2
```

## Available Regularizers

| Regularizer | Penalty | Effect on the coefficients | Default solver |
| --- | --- | --- | --- |
| [`UnRegularized`](nemos.regularizer.UnRegularized) | none | none | `LBFGS` |
| [`Ridge`](nemos.regularizer.Ridge) | squared $\ell_2$ norm | shrinkage | `LBFGS` |
| [`Lasso`](nemos.regularizer.Lasso) | $\ell_1$ norm | sparsity | `ProximalGradient` |
| [`ElasticNet`](nemos.regularizer.ElasticNet) | weighted sum of the two | shrinkage and sparsity | `ProximalGradient` |
| [`GroupLasso`](nemos.regularizer.GroupLasso) | $\ell_2$ norm of each group of coefficients | group sparsity | `ProximalGradient` |

## Background

### The penalized objective

Fitting a model minimizes the sum of a loss and a penalty,

$$
\mathcal{L}(\beta) = f(\beta) + P(\beta).
$$

$f$ is the model's loss, which for a GLM is the negative log-likelihood of the observations. $P$ is the penalty, a function of the parameters alone, defined by the regularizer.

### The penalty of each class

Every panel below gives $P$ for a strength $\alpha$ shared by all the coefficients. The strength can also be set per coefficient or per group, which the [per-parameter strengths guide](finegrained_regularization) covers.

::::{tab-set}

:::{tab-item} UnRegularized

$$
P(\beta) = 0
$$

The objective is the loss itself, and the proximal operator is the identity.
:::

:::{tab-item} Ridge

$$
P(\beta) = \frac{\alpha}{2} \sum_i \beta_i^2
$$

Shrinks every coefficient towards zero without ever setting it to zero.
:::

:::{tab-item} Lasso

$$
P(\beta) = \alpha \sum_i |\beta_i|
$$

Can set coefficients exactly to zero, which selects a subset of the features.
:::

:::{tab-item} ElasticNet

$$
P(\beta) = \alpha \sum_i \left[ \frac{1 - \lambda}{2} \beta_i^2 + \lambda |\beta_i| \right]
$$

The ratio $\lambda \in (0, 1]$ splits the strength between the two norms: $\lambda = 1$ is the Lasso, and as $\lambda$ decreases the penalty approaches the Ridge. Strength and ratio are passed together as a tuple, `(strength, ratio)`.
:::

:::{tab-item} GroupLasso

With the coefficients partitioned into groups $\beta_{(1)}, \dots, \beta_{(G)}$, where group $g$ holds $p_g$ of them,

$$
P(\beta) = \alpha \sum_{g=1}^{G} \sqrt{p_g} \, \lVert \beta_{(g)} \rVert_2
$$

The norm is not squared, so the penalty is non-differentiable where a whole group is zero: a group can be set to zero as a block, all of its coefficients at once. The $\sqrt{p_g}$ factor makes the strength comparable across groups of different sizes. The partition is set by the `mask` attribute, and the strength is one value per group.
:::

::::

### Smooth and non-smooth penalties

The Ridge penalty is differentiable everywhere, so the penalized objective can be minimized by any gradient-based solver.

The other penalties are not differentiable at zero, so a solver cannot use the gradient of $P$. One way to minimize such an objective is to take a gradient-based step on the smooth part $f$, and then apply to the updated parameters the proximal operator of the penalty,

$$
\operatorname{prox}_{s P}(x) = \underset{u}{\operatorname{argmin}} \; \frac{1}{2} \lVert u - x \rVert_2^2 + s \, P(u),
$$

with $s$ the step size. Solvers like proximal gradient and proximal Newton iterate these two steps, with clever choices of step size, to converge to the optimal solution.

For this reason a `Regularizer` implements two methods: one that penalizes a loss, used in smooth optimization, and one that returns the proximal operator of the penalty, used in non-smooth optimization.

## Initialization

Most regularizers take no argument at initialization.

```{code-cell} ipython3
import nemos as nmo

ridge = nmo.regularizer.Ridge()
lasso = nmo.regularizer.Lasso()
elastic_net = nmo.regularizer.ElasticNet()

print(ridge)
print(lasso)
print(elastic_net)
```

The strength is not set at initialization. It is an argument of the methods below, and a model passes down its own `regularizer_strength` when it calls them, so the same regularizer object can be used at any strength.

[`GroupLasso`](nemos.regularizer.GroupLasso) takes the group structure, as a mask with one row per group.

```{code-cell} ipython3
import numpy as np

# group 0 holds features 0 and 3, group 1 holds features 1, 2 and 4
mask = np.zeros((2, 5))
mask[0, [0, 3]] = 1
mask[1, [1, 2, 4]] = 1

group_lasso = nmo.regularizer.GroupLasso(mask=mask)
group_lasso.mask
```

Each feature may belong to at most one group, and a feature left out of every row is unpenalized. Leaving the mask out is allowed too: it is then built from the parameters at fit time, one group per parameter leaf.

```{code-cell} ipython3
import jax.numpy as jnp

nmo.regularizer.GroupLasso().initialize_mask(
    {"f1": jnp.zeros(3), "f2": jnp.zeros(2)}
)
```

## Attributes

### `allowed_solvers` and `default_solver`

`allowed_solvers` lists the solvers a penalty can be minimized with.

```{code-cell} ipython3
print("Allowed solvers")
print("Ridge:", ridge.allowed_solvers)
print("Lasso:", lasso.allowed_solvers)
```

`default_solver` is the one the regularizer asks for when you do not name one. A model may override it: a `GLM` with a Ridge penalty uses `Newton`.

```{code-cell} ipython3
print("Default solver")
print("Ridge:", ridge.default_solver)
print("Lasso:", lasso.default_solver)
```

### `mask`

[`GroupLasso`](nemos.regularizer.GroupLasso) stores the group structure as an attribute, which can be replaced after initialization. The new mask is validated on assignment: entries other than 0 and 1, and features assigned to more than one group, are rejected.

```{code-cell} ipython3
:tags: [raises-exception]

group_lasso.mask = np.ones((2, 5))  # every feature in both groups
```

The number of features the mask covers must match the parameters, which is checked when the penalty is built. The same mask cannot be reused across models with different numbers of features.

## Methods

The examples below penalize the mean squared difference between the observations and a linear prediction,

$$
f(\beta) = \frac{1}{N} \sum_{n=1}^{N} \left( y_n - x_n^\top \beta \right)^2,
$$

with data generated from five coefficients, two of which are zero. The loss concatenates the leaves of the parameters, so the same function works whether they are one array or a pytree of arrays.

```{code-cell} ipython3
import jax

np.random.seed(123)

n_samples, n_features = 100, 5
X = np.random.normal(size=(n_samples, n_features))
beta_true = jnp.array([2.0, -1.0, 0.0, 0.0, 0.5])
y = X @ beta_true + 0.1 * np.random.normal(size=n_samples)


def loss(beta):
    """Mean squared difference between observations and prediction."""
    return jnp.mean((y - X @ jnp.concatenate(jax.tree.leaves(beta))) ** 2)


beta = jnp.zeros(n_features)
```

### Penalty

[`penalty_fn`](nemos.regularizer.Regularizer.penalty_fn) returns a function that evaluates $P$ alone.

```{code-cell} ipython3
penalty = ridge.penalty_fn(beta, strength=0.1)

penalty(beta_true)
```

The parameters appear twice. The first call uses their structure to match the strength to the coefficients; the second is the point where $P$ is evaluated.

### Penalized loss

[`penalized_loss`](nemos.regularizer.Regularizer.penalized_loss) takes $f$ and returns $\mathcal{L} = f + P$, the objective written at the top of this page.

```{code-cell} ipython3
objective = ridge.penalized_loss(loss, beta, strength=0.1)

print("loss:      ", loss(beta_true))
print("objective: ", objective(beta_true))
```

The loss may return a scalar, or a tuple of a scalar and an auxiliary value, for example a loss that also returns diagnostics. The penalty is added to the scalar, and the auxiliary value is returned unchanged.


### Proximal operator

[`get_proximal_operator`](nemos.regularizer.Regularizer.get_proximal_operator) returns the proximal operator of the penalty, at the strength given. It is called as `prox(params, hyperparams, scaling)`: `scaling` multiplies the strength, and is the step size in a proximal gradient step; `hyperparams` is ignored and exists to match the signature proximal solvers expect.

```{code-cell} ipython3
prox = lasso.get_proximal_operator(beta, strength=0.1)

step = jnp.array([2.0, -1.0, 0.05, -0.08, 0.5])
prox(step, None, 1.0)
```

Entries smaller than the strength are set to zero, and the others are shrunk by `0.1`. This is the soft-thresholding operator, the proximal operator of the $\ell_1$ penalty.

### Solver compatibility

Models call [`check_solver`](nemos.regularizer.Regularizer.check_solver) before accepting a solver name, which raises if the name is not in `allowed_solvers`.

```{code-cell} ipython3
:tags: [raises-exception]

lasso.check_solver("LBFGS")
```

A solver of your own is declared compatible with [`allow_solver`](nemos.regularizer.Regularizer.allow_solver), which adds its name to the class-level list. See the [custom solver guide](../how_to_guide/custom_solvers.md) for how to write one.


## Pytree parameters

The `Regularizer` methods are fully compatible with PyTree-structured parameters.

To demonstrate that, we can split the five coefficients into two feature groups and store them in a dictionary.

```{code-cell} ipython3
beta_tree = {"f1": jnp.zeros(3), "f2": jnp.zeros(2)}
beta_tree_true = {"f1": beta_true[:3], "f2": beta_true[3:]}

loss(beta_tree_true)
```

The penalty is summed over the leaves.

```{code-cell} ipython3
print("PyTree compatible penalization machinery:")
print("Penalty: ",
    ridge.penalty_fn(beta_tree_true, strength=0.1)(beta_tree_true)
)

print("Penalized loss: ",
    ridge.penalized_loss(loss, beta_tree_true, strength=0.1)(beta_tree_true)
)
```

The proximal operator returns a pytree of the same structure, holding the shrunk parameters.

```{code-cell} ipython3
lasso_prox = lasso.get_proximal_operator(beta_tree_true, strength=0.1)
print(
    "Proximal operator:",
    lasso_prox(beta_tree_true, None, 1.0)
)
```

## Choosing what to regularize

If you define your parameters as an object of type [`equinox.Module`](https://docs.kidger.site/equinox/api/module/module/), a type of PyTree, you can specify which subset of the parameter tree is regularized.

All you need to do is implement a `regularizable_subtrees` method for your parameter object. This method receives no inputs and returns a list of selector functions. Each function takes in the parameter object and returns the sub-tree of parameters to be regularized. Below is a concrete example.

### Example: ridge regularized linear regression

Fitting a ridge regression and penalizing the coefficients only is a common case: it is what scikit-learn and NeMoS do by default, since the intercept sets the baseline of the prediction, and shrinking it towards zero would bias it.

Let's use this as a concrete use-case to show how `regularizable_subtrees` works: we want to regularize the least-square loss of a linear regression with a ridge penalty, applied to the coefficients only, and not to the intercept.

To do so, we define a custom parameter object of type `equinox.Module`, and equip it with the `regularizable_subtrees` method.

```{code-cell} ipython3
import equinox as eqx
from jax import Array

class LinRegParams(eqx.Module):
    coef: Array
    intercept: Array

    # returns a list of functions selecting the parameters to be regularized
    @staticmethod
    def regularizable_subtrees():
        return [lambda p: p.coef]

# least-squares loss compatible with PyTree coef
def loss_eqx_params(params):
    coef = jnp.concatenate(jax.tree.leaves(params.coef))
    return jnp.mean((y - X @ coef - params.intercept) ** 2)
```

:::{admonition} Static method
:class: note

A static method is a method that doesn't depend on the class instance or the class itself.
:::

#### Penalty function

The penalty is computed on the coefficients alone, with no contribution from the intercept.

```{code-cell} ipython3
eqx_params = LinRegParams(coef=beta_true, intercept=jnp.array(1.0))

# hard-coded ridge on the coefficients only
hardcoded_ridge = 0.5 * 0.1 * jnp.sum(eqx_params.coef**2)

# evaluate the penalty using the Ridge regularizer applied
# to the custom parameter object
penalty_val = ridge.penalty_fn(eqx_params, strength=0.1)(eqx_params)

# check that only the coefficients contribute to the penalty
penalty_val == hardcoded_ridge
```

Setting the coefficients to zero therefore results in a penalty of zero.

```{code-cell} ipython3
zero_coef = LinRegParams(coef=jnp.zeros_like(beta_true), intercept=jnp.array(1.0))
ridge.penalty_fn(zero_coef, strength=0.1)(zero_coef)
```

#### Penalized loss

The `penalized_loss` method adds the ridge contribution of the coefficients only.

```{code-cell} ipython3
penalized = ridge.penalized_loss(loss_eqx_params, eqx_params, strength=0.1)

print("penalized loss:        ", penalized(eqx_params))
print(
    "adding ridge manually: ",
    loss_eqx_params(eqx_params) + 0.5 * 0.1 * jnp.sum(eqx_params.coef**2),
)
```

#### Proximal operator

The proximal operator returns the same container, with the coefficients shrunk and the intercept unchanged.

```{code-cell} ipython3
shrunk = lasso.get_proximal_operator(eqx_params, strength=0.5)(eqx_params, None, 1.0)

print(f"original coef: {eqx_params.coef} - original intercept: {eqx_params.intercept}")
print(f"shrunk coef:   {shrunk.coef} - shrunk intercept:   {shrunk.intercept}")
```

Nothing stops us from being even more specific: we can split the coefficients into two groups and change the selector function to penalize one group only, leaving the other unregularized.

Let's reuse `beta_tree_true`, the dictionary of arrays with keys `"f1"` and `"f2"`.

```{code-cell} ipython3
class GroupedLinRegParams(eqx.Module):
    coef: dict[str, Array]
    intercept: Array

    # returns a list of functions selecting the parameters to be regularized
    @staticmethod
    def regularizable_subtrees():
        return [lambda p: p.coef["f1"]]

eqx_params = GroupedLinRegParams(coef=beta_tree_true, intercept=jnp.array(1.0))
```

Now, if we apply the penalty function to this new parameter object, we can confirm that only the `"f1"` subset of the coefficients is penalized.

```{code-cell} ipython3
hard_coded_f1_penalty = 0.5 * 0.1 * jnp.sum(eqx_params.coef["f1"]**2)
penalty_val = ridge.penalty_fn(eqx_params, strength=0.1)(eqx_params)

hard_coded_f1_penalty == penalty_val
```

The same is true for the penalized loss.

```{code-cell} ipython3
penalized = ridge.penalized_loss(loss_eqx_params, eqx_params, strength=0.1)

print("penalized loss:        ", penalized(eqx_params))
print(
    "adding ridge manually: ",
    loss_eqx_params(eqx_params) + 0.5 * 0.1 * jnp.sum(eqx_params.coef["f1"] ** 2),
)
```

The proximal operator also applies to the `"f1"` group only: `"f2"` is not shrunk.

```{code-cell} ipython3
shrunk = lasso.get_proximal_operator(eqx_params, strength=0.5)(eqx_params, None, 1.0)

print(f"original coef: {eqx_params.coef}")
print(f"original intercept: {eqx_params.intercept}")
print(f"shrunk coef:   {shrunk.coef} - shrunk intercept:   {shrunk.intercept}")
```

## Setting the strength

The `strength` argument of the methods above, and the `regularizer_strength` of a model, accept the same values.

### One strength for all the parameters

Every regularizer accepts a scalar, which is applied to all the penalized parameters.

```{code-cell} ipython3
params = jnp.ones(5, dtype=float)

print("Ridge:      ", ridge.penalty_fn(params, strength=0.1)(params))
print("Lasso:      ", lasso.penalty_fn(params, strength=0.1)(params))
print("GroupLasso: ", group_lasso.penalty_fn(params, strength=0.1)(params))
```

[`ElasticNet`](nemos.regularizer.ElasticNet) has two hyperparameters, the strength $\alpha$ and the ratio $\lambda$ that splits it between the $\ell_1$ and the $\ell_2$ norm, so it takes the pair `(strength, ratio)`. The ratio must be in $(0, 1]$: at $\lambda = 1$ the penalty is the Lasso, and smaller values move it towards the Ridge.

```{code-cell} ipython3
print("ratio 0.5: ", elastic_net.penalty_fn(params, strength=(0.1, 0.5))(params))
print("ratio 1.0: ", elastic_net.penalty_fn(params, strength=(0.1, 1.0))(params))
print("Lasso:     ", lasso.penalty_fn(params, strength=0.1)(params))
```

### Fine-grained strengths

Beyond a scalar, each class accepts a strength with more structure, and what that structure is depends on what the penalty acts on.

#### Ridge and Lasso: matching the parameters

The strength can be an array with one entry per coefficient, or a pytree matching the parameters, with a scalar or an array on each leaf. A strength of zero leaves a coefficient unpenalized. The [per-parameter strengths guide](finegrained_regularization) covers the accepted forms, including the dictionaries that match a pytree of predictors.

```{code-cell} ipython3
print(ridge.penalty_fn(params, strength=jnp.array([0.1, 0.1, 1.0, 1.0, 0.0]))(params))
print(ridge.penalty_fn(beta_tree, strength={"f1": 0.1, "f2": 1.0})(beta_tree_true))
```

#### Elastic Net: matching the parameters, twice

Both hyperparameters take the forms above, each one on its own.

```{code-cell} ipython3
# one strength per coefficient, one ratio for all of them
print(
    elastic_net.penalty_fn(
        params, strength=(jnp.array([0.1, 0.1, 1.0, 1.0, 0.0]), 0.5)
    )(params)
)

# one strength and one ratio per leaf of the parameter tree
print(
    elastic_net.penalty_fn(
        beta_tree, strength=({"f1": 0.1, "f2": 1.0}, {"f1": 0.5, "f2": 1.0})
    )(beta_tree_true)
)
```

#### Group Lasso: one entry per group

[`GroupLasso`](nemos.regularizer.GroupLasso) penalizes groups of coefficients, so the structure it accepts is an array with one entry per group, rather than one matching the parameters.

```{code-cell} ipython3
group_lasso.penalty_fn(params, strength=jnp.array([0.1, 0.4]))(params)
```

The proximal operator shrinks each group as a block, by a factor set by the norm of the group and by its own strength. With every coefficient equal to one, that factor is `max(1 - strength, 0)`: the first group, at `0.25`, keeps three quarters of its value, and the second, at `1.5`, is set to zero.

```{code-cell} ipython3
print("mask, one row per group:")
print(group_lasso.mask)

strength = jnp.array([0.25, 1.5])
group_prox = group_lasso.get_proximal_operator(params, strength=strength)

print("strength per group: ", strength)
print("shrunk parameters:  ", group_prox(params, None, 1.0))
```

The first row of the mask holds the coefficients `0` and `3`, and those keep three quarters of their value. The second row holds `1`, `2` and `4`, and those are the zeros. The two groups are shrunk by different amounts, which is the point of a strength per group.

The groups themselves need not follow the structure of the parameters: the mask is a pytree matching them, with a leaf of shape `(n_groups, *leaf.shape)` for each parameter leaf, and a group may collect coefficients from several leaves. Here the first group holds `f1[0]`, `f1[1]` and `f2[0]`, and the second holds the two coefficients left.

```{code-cell} ipython3
tree_mask = {"f1": np.zeros((2, 3)), "f2": np.zeros((2, 2))}

tree_mask["f1"][0, :2] = 1
tree_mask["f2"][0, 0] = 1

tree_mask["f1"][1, 2] = 1
tree_mask["f2"][1, 1] = 1

grouped = nmo.regularizer.GroupLasso(mask=tree_mask)
grouped.penalty_fn(beta_tree, strength=jnp.array([0.1, 0.4]))(beta_tree_true)
```

A coefficient is penalized by the strength of its own group, whatever leaf it sits in.

## Regularizers as model attributes

A model takes its regularizer by name, together with the strength.

```{code-cell} ipython3
model = nmo.glm.GLM(regularizer="Ridge", regularizer_strength=0.1)

model.regularizer
```

The string is used to instantiate the class, so the attribute is the object itself, and every method above is reachable from the model.

```{code-cell} ipython3
model.regularizer.penalty_fn(params, model.regularizer_strength)(params)
```

Pass an instance when the class takes an argument, which here is the `GroupLasso` mask.

```{code-cell} ipython3
model = nmo.glm.GLM(
    regularizer=nmo.regularizer.GroupLasso(mask=mask),
    regularizer_strength=0.1,
)

model.regularizer
```

`fit` then minimizes the penalized loss built from the model's negative log-likelihood.

```{code-cell} ipython3
X_counts, y_counts = nmo.datasets.tuned_poisson(n_samples=500)

model = nmo.glm.GLM(regularizer="Lasso", regularizer_strength=0.05).fit(
    X_counts, y_counts
)

model.coef_
```
