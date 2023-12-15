# The `regularizer` Module

## Introduction

The `regularizer` module introduces an archetype class `Regularizer` which provides the structural components for each concrete sub-class.

Objects of type `Regularizer` provide methods to define a regularized optimization objective, and instantiate a solver for it. These objects serve as attribute of the [`nemos.glm.GLM`](../05-glm/#the-concrete-class-glm), equipping the glm with a solver for learning model parameters. 

Solvers are typically optimizers from the `jaxopt` package, but in principle they could be custom optimization routines as long as they respect the `jaxopt` api (i.e., have a `run` and `update` method with the appropriate input/output types).
We choose to rely on `jaxopt` because it provides a comprehensive set of robust, GPU accelerated, batchable and differentiable optimizers in JAX, that are highly customizable.

Each `Regularizer` object defines a set of allowed optimizers, which in turn depends on the loss function characteristics (smooth vs non-smooth) and/or the optimization type (constrained, un-constrained, batched, etc.).

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
    If we need advanced adaptive optimizers (e.g., Adam, LAMB etc.) in the future, we should consider adding [`Optax`](https://optax.readthedocs.io/en/latest/) as a dependency, which is compatible with `jaxopt`, see [here](https://jaxopt.github.io/stable/_autosummary/jaxopt.OptaxSolver.html#jaxopt.OptaxSolver).

## The Abstract Class `Regularizer`

The abstract class `Regularizer` enforces the implementation of the `instantiate_solver` method on any concrete realization of a `Regularizer` object. `Regularizer` objects are equipped with a method for instantiating a solver runner with the appropriately regularized loss function, i.e., a function that receives as input the initial parameters, the endogenous and the exogenous variables, and outputs the optimization results.

Additionally, the class provides auxiliary methods for checking that the solver and loss function specifications are valid.

### Public Methods

- **`instantiate_solver`**: Instantiate a solver runner for a provided loss function, configure and return a `solver_run` callable. The loss function must be of type `Callable`.

### Auxiliary Methods

- **`_check_solver`**: This method ensures that the provided solver name is in the list of allowed solvers for the specific `Regularizer` object. This is crucial for maintaining consistency and correctness in the solver's operation.

- **`_check_solver_kwargs`**: This method checks if the provided keyword arguments are valid for the specified solver. This helps in catching and preventing potential errors in solver configuration.

## The `UnRegularized` Class

The `UnRegularized` class extends the base `Regularizer` class and is designed specifically for optimizing unregularized models. This means that the solver instantiated by this class does not add any regularization penalty to the loss function during the optimization process.

### Attributes

- **`allowed_solvers`**: A list of string identifiers for the optimization solvers that can be used with this regularizer class. The optimization methods listed here are specifically suitable for unregularized optimization problems.

### Methods

- **`__init__`**: The constructor method for this class which initializes a new `UnRegularized` object. It accepts the name of the solver algorithm to use (`solver_name`) and an optional dictionary of additional keyword arguments (`solver_kwargs`) for the solver.

- **`instantiate_solver`**: A method which prepares and returns a runner function for the specified loss function. This method ensures that the loss function is callable and prepares the necessary keyword arguments for calling the `get_runner` method from the base `Regularizer` class.

### Example Usage

```python
unregularized = UnRegularized(solver_name="GradientDescent")
runner = unregularized.instantiate_solver(loss_function)
optim_results = runner(init_params, exog_vars, endog_vars)
```

## The `Ridge` Class

The `Ridge` class extends the `Regularizer` class to handle optimization problems with Ridge regularization. Ridge regularization adds a penalty to the loss function, proportional to the sum of squares of the model parameters, to prevent overfitting and stabilize the optimization.

### Attributes

- **`allowed_solvers`**: A list containing string identifiers of optimization solvers compatible with Ridge regularization.
  
- **`regularizer_strength`**: A floating-point value determining the strength of the Ridge regularization. Higher values correspond to stronger regularization which tends to drive the model parameters towards zero.

### Methods

- **`__init__`**: The constructor method for the `Ridge` class. It accepts the name of the solver algorithm (`solver_name`), an optional dictionary of additional keyword arguments (`solver_kwargs`) for the solver, and the regularization strength (`regularizer_strength`).

- **`penalization`**: A method to compute the Ridge regularization penalty for a given set of model parameters.

- **`instantiate_solver`**: A method that prepares and returns a runner function with a penalized loss function for Ridge regularization. This method modifies the original loss function to include the Ridge penalty, ensures the loss function is callable, and prepares the necessary keyword arguments for calling the `get_runner` method from the base `Regularizer` class.

### Example Usage

```python
ridge = Ridge(solver_name="LBFGS", regularizer_strength=1.0)
runner = ridge.instantiate_solver(loss_function)
optim_results = runner(init_params, exog_vars, endog_vars)
```

## `ProxGradientRegularizer` Class

`ProxGradientRegularizer` class extends the `Regularizer` class to utilize the Proximal Gradient method for optimization. It leverages the `jaxopt` library's Proximal Gradient optimizer, introducing the functionality of a proximal operator.

### Attributes:
- **`allowed_solvers`**: A list containing string identifiers of optimization solvers compatible with this solver, specifically the "ProximalGradient".

### Methods:
- **`__init__`**: The constructor method for the `ProxGradientRegularizer` class. It accepts the name of the solver algorithm (`solver_name`), an optional dictionary of additional keyword arguments (`solver_kwargs`) for the solver, the regularization strength (`regularizer_strength`), and an optional mask array (`mask`).
    
- **`get_prox_operator`**: Abstract method to retrieve the proximal operator for this solver.
  
- **`instantiate_solver`**: Method to prepare and return a runner function for optimization with a provided loss function and proximal operator.

## `Lasso` Class

`Lasso` class extends `ProxGradientRegularizer` to specialize in optimization using the Lasso (L1 regularization) method with Proximal Gradient.

### Methods:
- **`__init__`**: Constructor method similar to `ProxGradientRegularizer` but defaults `solver_name` to "ProximalGradient".
  
- **`get_prox_operator`**: Method to retrieve the proximal operator for Lasso regularization (L1 penalty).

## `GroupLasso` Class

`GroupLasso` class extends `ProxGradientRegularizer` to specialize in optimization using the Group Lasso regularization method with Proximal Gradient. It induces sparsity on groups of features rather than individual features.

### Attributes:
- **`mask`**: A mask array indicating groups of features for regularization.

### Methods:
- **`__init__`**: Constructor method similar to `ProxGradientRegularizer`, but additionally requires a `mask` array to identify groups of features.
   
- **`get_prox_operator`**: Method to retrieve the proximal operator for Group Lasso regularization.

- **`_check_mask`**: Static method to check that the provided mask is a float `jax.numpy.ndarray` of 0s and 1s. The mask must be in floats to be applied correctly through the linear algebra operations of the `nemos.proimal_operator.prox_group_lasso` function.

### Example Usage
```python
lasso = Lasso(regularizer_strength=1.0)
runner = lasso.instantiate_solver(loss_function)
optim_results = runner(init_params, exog_vars, endog_vars)

group_lasso = GroupLasso(solver_name="ProximalGradient", mask=group_mask, regularizer_strength=1.0)
runner = group_lasso.instantiate_solver(loss_function)
optim_results = runner(init_params, exog_vars, endog_vars)
```

## Contributor Guidelines

### Implementing `Regularizer` Subclasses

When developing a functional (i.e., concrete) `Regularizer` class:

- **Must** inherit from `Regularizer` or one of its derivatives.
- **Must** implement the `instantiate_solver` method to tailor the solver instantiation based on the provided loss function.
- For any Proximal Gradient method, **must** include a `get_prox_operator` method to define the proximal operator.
- **Must** possess an `allowed_solvers` attribute to list the solver names that are permissible to be used with this regularizer.
- **May** embed additional attributes and methods such as `mask` and `_check_mask` if required by the specific Solver subclass for handling special optimization scenarios.
- **May** include a `regularizer_strength` attribute to control the strength of the regularization in scenarios where regularization is applicable.
- **May** rely on a custom solver implementation for specific optimization problems, but the implementation **must** adhere to the `jaxopt` API.

These guidelines ensure that each Solver subclass adheres to a consistent structure and behavior, facilitating ease of extension and maintenance.

## Glossary

|  Term   | Description |
|--------------------| ----------- |
| **Regularization** | Regularization is a technique used to prevent overfitting by adding a penalty to the loss function, which discourages complex models. Common regularization techniques include L1 (Lasso) and L2 (Ridge) regularization. |
| **Optimization**   | Optimization refers to the process of minimizing (or maximizing) a function by systematically choosing the values of the variables within an allowable set. In machine learning, optimization aims to minimize the loss function to train models. |
| **Solver**         | A solver is an algorithm or a set of algorithms used for solving optimization problems. In the given module, solvers are used to find the parameters that minimize the loss function, potentially subject to some constraints. |
| **Runner**         | A runner in this context refers to a callable function configured to execute the solver with the specified parameters and data. It acts as an interface to the solver, simplifying the process of running optimization tasks. |
