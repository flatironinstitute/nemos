# The `regularizer` Module

## Introduction

The [`regularizer`](regularizers) module introduces an archetype class [`Regularizer`](nemos.regularizer.Regularizer) which provides the structural components for each concrete sub-class.

Objects of type [`Regularizer`](nemos.regularizer.Regularizer) provide methods to define a regularized optimization objective. These objects serve as attribute of the [`nemos.glm.GLM`](the-concrete-class-glm), equipping the glm with an appropriate regularization scheme.

Each [`Regularizer`](nemos.regularizer.Regularizer) object defines a default solver, and a set of allowed solvers, which depends on the loss function characteristics (smooth vs non-smooth).

```
Abstract Class Regularizer
|
├─ Concrete Class UnRegularized
|
├─ Concrete Class Ridge
|
├─ Concrete Class Lasso
|
└─ Concrete Class GroupLasso
```

:::{note}
If we need advanced adaptive solvers (e.g., Adam, LAMB etc.) in the future, we should consider adding [`Optax`](https://optax.readthedocs.io/en/latest/) as a dependency, which is compatible with `jaxopt`, see [here](https://jaxopt.github.io/stable/_autosummary/jaxopt.OptaxSolver.html#jaxopt.OptaxSolver).
:::

(the-abstract-class-regularizer)=
## The Abstract Class `Regularizer`

The abstract class [`Regularizer`](nemos.regularizer.Regularizer) enforces the implementation of the [`penalized_loss`](nemos.regularizer.Regularizer.penalized_loss) and [`get_proximal_operator`](nemos.regularizer.Regularizer.get_proximal_operator) methods.

### Attributes

The attributes of [`Regularizer`](nemos.regularizer.Regularizer) consist of the `default_solver` and `allowed_solvers`, which are stored as read-only properties of type string and tuple of strings respectively.

### Abstract Methods

- [`penalized_loss`](nemos.regularizer.Regularizer.penalized_loss): Returns a penalized version of the input loss function which is uniquely defined by the regularization scheme and the regularizer strength parameter.
- [`get_proximal_operator`](nemos.regularizer.Regularizer.get_proximal_operator): Returns the proximal projection operator which is uniquely defined by the regularization scheme.

## The `UnRegularized` Class

The [`UnRegularized`](nemos.regularizer.UnRegularized) class extends the base [`Regularizer`](nemos.regularizer.Regularizer) class and is designed specifically for optimizing unregularized models. This means that the solver instantiated by this class does not add any regularization penalty to the loss function during the optimization process.


### Concrete Methods Specifics
- [`penalized_loss`](nemos.regularizer.UnRegularized.penalized_loss): Returns the original loss without any changes.
- [`get_proximal_operator`](nemos.regularizer.UnRegularized.get_proximal_operator): Returns the identity operator.


## Contributor Guidelines

### Implementing `Regularizer` Subclasses

When developing a functional (i.e., concrete) `Regularizer` class:

- **Must** inherit from [`Regularizer`](nemos.regularizer.Regularizer) or one of its derivatives.
- **Must** implement the [`penalized_loss`](nemos.regularizer.Regularizer.penalized_loss) and [`get_proximal_operator`](nemos.regularizer.Regularizer.get_proximal_operator) methods.
- **Must** define a default solver and a tuple of allowed solvers.
- **May** require extra initialization parameters, like the `mask` argument of [`GroupLasso`](nemos.regularizer.GroupLasso).

:::{dropdown} Convergence Test
:icon: light-bulb
:color: success

When adding a new regularizer, you must include a convergence test, which verifies that
the model parameters the regularizer finds for a convex problem such as the GLM are identical
whether one minimizes the penalized loss directly and uses the proximal operator (i.e., when
using `ProximalGradient`). In practice, this means you should test the result of the `ProximalGradient`
optimization against that of either `GradientDescent` (if your regularization is differentiable) or
`Nelder-Mead` from [`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html) 
(or another non-gradient based method, if your regularization is non-differentiable). You can refer to NeMoS `test_lasso_convergence`
from `tests/test_convergence.py` for a concrete example.
:::