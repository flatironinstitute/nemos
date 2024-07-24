# The `regularizer` Module

## Introduction

The `regularizer` module introduces an archetype class `Regularizer` which provides the structural components for each concrete sub-class.

Objects of type `Regularizer` provide methods to define a regularized optimization objective. These objects serve as attribute of the [`nemos.glm.GLM`](../05-glm/#the-concrete-class-glm), equipping the glm with an appropriate regularization scheme.

Each `Regularizer` object defines a default solver, set of allowed solvers, which depends on the loss function characteristics (smooth vs non-smooth).

```
Abstract Class Regularizer
|
├─ Concrete Class UnRegularized
|
├─ Concrete Class Ridge
|
└─ Abstract Class ProximalGradientRegularizer
    |
    ├─ Concrete Class Lasso
    |
    └─ Concrete Class GroupLasso
```

!!! note
    If we need advanced adaptive solvers (e.g., Adam, LAMB etc.) in the future, we should consider adding [`Optax`](https://optax.readthedocs.io/en/latest/) as a dependency, which is compatible with `jaxopt`, see [here](https://jaxopt.github.io/stable/_autosummary/jaxopt.OptaxSolver.html#jaxopt.OptaxSolver).

## The Abstract Class `Regularizer`

The abstract class `Regularizer` enforces the implementation of the `penalized_loss` and `get_proximal_operator` methods.

### Attributes

The attributes of `Regularizer` consist of the `default_solver` and `allowed_solvers`, which are stored as read-only properties as a string and tuple of strings respectively.

### Abstract Methods

- **`penalized_loss`**: Returns a penalized version of the input loss function which is uniquely defined by the regularization scheme and the regularizer strength parameter.
- **`get_proximal_operator`**: Returns the proximal projection operator which is uniquely defined by the regularization scheme.

## The `UnRegularized` Class

The `UnRegularized` class extends the base `Regularizer` class and is designed specifically for optimizing unregularized models. This means that the solver instantiated by this class does not add any regularization penalty to the loss function during the optimization process.


### Concrete methods specifics
- **`penalized_loss`**: Returns the original loss without any changes
- **`get_proximal_operator`**: Returns the identity operator.


## The `Ridge` Class

The `Ridge` class extends the `Regularizer` class to handle optimization problems with Ridge regularization. Ridge regularization adds a penalty to the loss function, proportional to the sum of squares of the model parameters, to prevent overfitting and stabilize the optimization.

### Concrete methods specifics
- **`penalized_loss`**: Returns the original loss penalized by a term proportional to the sum of squares of the coefficients (excluding the intercept).
- **`get_proximal_operator`**: Returns the ridge proximal operator, solving $\underset{y \ge 0}{\text{argmin}} ~ \frac{1}{2} ||x - y||_2^2
  +\text{l2reg} \cdot ||y||_2^2$, where "l2reg" is the regularizer strength.

### Example Usage

```python
import nemos as nmo

ridge = nmo.regularizer.Ridge()
model = nmo.glm.GLM(regularizer=ridge)
```

## `Lasso` Class

The `Lasso` class enables optimization using the Lasso (L1 regularization) method with Proximal Gradient.

### Concrete methods specifics
- **`penalized_loss`**: Returns the original loss penalized by a term proportional to the sum of the absolute values of the coefficients (excluding the intercept).
- **`get_proximal_operator`**: Returns the ridge proximal operator, solving $\underset{y \ge 0}{\text{argmin}} ~ \frac{1}{2} ||x - y||_2^2
  + \text{l1reg} \cdot ||y||_1$, where "l1reg" is the regularizer strength.

## `GroupLasso` Class

The `GroupLasso` class enables optimization using the Group Lasso regularization method with Proximal Gradient. It induces sparsity on groups of features rather than individual features.

### Attributes:
- **`mask`**: A mask array indicating groups of features for regularization.

### Concrete methods specifics
- **`penalized_loss`**: Returns the original loss penalized by a term proportional to the sum of the L2 norms of the coefficient vectors for each group defined by the `mask`.
- **`get_proximal_operator`**: Returns the ridge proximal operator, solving $\underset{y \ge 0}{\text{argmin}} ~ \frac{1}{2} ||x - y||_2^2
  + \text{lgreg} \cdot \sum_j ||y^{(j)}||_2$, where "lgreg" is the regularizer strength, and $j$ runs over the coefficient groups.

### Example Usage
```python
import nemos as nmo
import numpy as np

group_mask = np.zeros((2, 10))  # assume 2 groups and 10 features
group_mask[0, :4] = 1  # assume the first group consist of the first 4 coefficients
group_mask[1, 4:] = 1  # assume the second group consist of the last 6 coefficients
group_lasso = nmo.regularizer.GroupLasso(mask=group_mask)
model = nmo.glm.GLM(regularizer=group_lasso)
```



## Contributor Guidelines

### Implementing `Regularizer` Subclasses

When developing a functional (i.e., concrete) `Regularizer` class:

- **Must** inherit from `Regularizer` or one of its derivatives.
- **Must** implement the `penalized_loss` and `proximal_operator` methods.
- **Must** define a default solver and a tuple of allowed solvers.
- 

