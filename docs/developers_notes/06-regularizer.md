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

(the-abstract-class-regularizer)=
## The Abstract Class `Regularizer`

The abstract class [`Regularizer`](nemos.regularizer.Regularizer) enforces the implementation of the [`penalized_loss`](nemos.regularizer.Regularizer.penalized_loss) and [`get_proximal_operator`](nemos.regularizer.Regularizer.get_proximal_operator) methods.

### Attributes

The attributes of [`Regularizer`](nemos.regularizer.Regularizer) consist of the `default_solver` and `allowed_solvers`, which are stored as read-only properties of type string and tuple of strings respectively.

### Abstract Methods

- [`penalized_loss`](nemos.regularizer.Regularizer.penalized_loss): Returns a penalized version of the input loss function which is uniquely defined by the regularization scheme and the regularizer strength parameter.
- [`get_proximal_operator`](nemos.regularizer.Regularizer.get_proximal_operator): Returns the proximal projection operator which is uniquely defined by the regularization scheme.

### Core Functions

#### `apply_operator`

The `apply_operator` function applies a transformation to all regularizable components of a parameter pytree:

```python
def apply_operator(func, params, *args, **kwargs):
    """
    Apply an operator to all regularizable subtrees of a parameter pytree.

    Uses params.regularizable_subtrees() to identify which parameters
    should be transformed, applies func to each, and returns updated params.
    """
```

This function enables **selective regularization**: models can specify which parameter components should be regularized via the `regularizable_subtrees()` method on their parameter containers. For example, GLMs regularize coefficients but not intercepts.

**Benefits**:
- No hardcoded assumptions about parameter structure
- Model-specific control over what gets regularized
- Works with any pytree structure

#### `_penalize`

Base method that computes regularization penalties using the `regularizable_subtrees()` hook. The current implementation assumes penalties are additive across parameter groups (e.g., separate penalty for each neuron's coefficients), which covers most use cases but can be extended if needed.

### Proximal Operators

Proximal operators have been updated to work with arbitrary pytree structures rather than assuming specific parameter layouts. Each regularizer's `get_proximal_operator()` method returns a function that:

1. Accepts any pytree of parameters
2. Applies the proximal operation element-wise
3. Returns a pytree with the same structure

The `apply_operator` function then uses the model's `regularizable_subtrees()` specification to apply the proximal operator only to the appropriate parameter components.

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
- **Should** implement proximal operators to work on arbitrary pytrees (element-wise operations).
- **Should** use the `regularizable_subtrees()` hook on parameter containers to determine which components to regularize.
- **May** require extra initialization parameters, like the `mask` argument of [`GroupLasso`](nemos.regularizer.GroupLasso).
- **May** override `_penalize` if penalty computation requires non-additive aggregation across parameter groups.

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

## Interaction with Parameter Containers

Regularizers interact with model parameters through the `regularizable_subtrees()` hook defined on parameter containers (e.g., `GLMParams`). This method returns a list of selector functions that identify which parameter components should be regularized.

**Example workflow**:

1. Model defines parameter container with regularization hook:
   ```python
   class GLMParams(eqx.Module):
       coef: jnp.ndarray
       intercept: jnp.ndarray

       @staticmethod
       def regularizable_subtrees():
           return [lambda p: p.coef]  # Only regularize coefficients
   ```

2. Regularizer applies operations using `apply_operator`:
   ```python
   # Apply proximal operator only to coefficients, leave intercept unchanged
   updated_params = apply_operator(proximal_op, params, strength=0.1)
   ```

3. Penalty computation respects the same hook:
   ```python
   # Compute penalty only on coefficients
   penalty = regularizer._penalize(params, strength)
   ```

This design allows:
- **Model flexibility**: Each model controls what gets regularized
- **Code reuse**: Same regularizer works with different model types
- **Extensibility**: Easy to add new models with custom regularization needs
