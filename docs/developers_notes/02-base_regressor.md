# The Abstract Class `BaseRegressor`

`BaseRegressor` is an abstract class that inherits from `Base`, stipulating the implementation of number of abstract methods such as [`fit`](nemos.glm.GLM.fit), [`predict`](nemos.glm.GLM.predict), [`score`](nemos.glm.GLM.score). This ensures seamless assimilation with `scikit-learn` pipelines and cross-validation procedures.


:::{admonition} Example
:class: note

The current package version includes a concrete class named [`nemos.glm.GLM`](nemos.glm.GLM). This class inherits from `BaseRegressor`, which in turn inherits `Base`, since it falls under the "GLM regression" category.
As a `BaseRegressor`, it **must** implement the [`fit`](nemos.glm.GLM.fit), [`score`](nemos.glm.GLM.score), [`predict`](nemos.glm.GLM.predict) and the other abstract methods of this class, see below.
:::

## Abstract Methods

For subclasses derived from `BaseRegressor` to function correctly, they must implement the following:

### Public Methods
1. [`fit`](nemos.glm.GLM.fit): Adapt the model using input data `X` and corresponding observations `y`.
2. [`predict`](nemos.glm.GLM.predict): Provide predictions based on the trained model and input data `X`.
3. [`score`](nemos.glm.GLM.score): Score the accuracy of model predictions using input data `X` against the actual observations `y`.
4. [`simulate`](nemos.glm.GLM.simulate): Simulate data based on the trained regression model.
5. [`update`](nemos.glm.GLM.update): Run a single optimization step, and stores the updated parameter and solver state. Used by stochastic optimization schemes.
6. [`compute_loss`](nemos.glm.GLM.compute_loss): Computes the loss function for given user-provided parameters, `X`, and `y`. This method validates inputs and parameters, converts user parameters to the internal representation, and delegates to `_compute_loss`.

### Private Methods
6. `_compute_loss`: Compute predictions and evaluate the loss function given the parameters, `X`, and `y`. This method operates on the internal parameter representation (e.g., `GLMParams`). The public-facing `compute_loss` wrapper handles conversion from user-provided parameters.
7. `_get_model_params`: Pack model parameters from sklearn-style attributes (`coef_`, `intercept_`) into the internal parameter representation.
8. `_set_model_params`: Unpack the internal parameter representation and store as sklearn-style attributes (`coef_`, `intercept_`).
9. `save_params`: Serialize and save model parameters to disk.
10. `_get_optimal_solver_params_config`: Return functions for computing default step size and batch size for the solver.

## Parameter Representation and Validation

`BaseRegressor` maintains two representations of model parameters:

- **User-facing parameters**: Simple structures (tuples, arrays) that users provide to methods like `fit()`, `predict()`, etc.
- **Internal parameters**: Model-specific parameter containers (e.g., `GLMParams`) with named fields for clarity and type safety.

Each `BaseRegressor` subclass has an associated `RegressorValidator` that handles:
- Validation of user-provided inputs and parameters
- Conversion between user-facing and internal parameter representations
- Consistency checks between parameters and data

The validator is accessed via the `_validator` attribute and is used throughout the class to ensure data integrity.

## Attributes

Public attributes are stored as properties:

- `regularizer`: An instance of the [`nemos.regularizer.Regularizer`](nemos.regularizer.Regularizer) class. The setter for this property accepts either the instance directly or a string that is used to instantiate the appropriate regularizer.
- `regularizer_strength`: A float quantifying the amount of regularization.
- `solver_name`: One of the supported solvers in the solver registry, currently "GradientDescent", "BFGS", "LBFGS", "ProximalGradient", "SVRG", and "NonlinearCG".
- `solver_kwargs`: Extra keyword arguments to be passed at solver initialization.
- `solver_init_state`, `solver_update`, `solver_run`: Read-only property with a partially evaluated `solver.init_state`, `solver.update` and, `solver.run` methods. The partial evaluation guarantees a consistent API for all solvers.

When implementing a new subclass of `BaseRegressor`, the only attributes you must interact directly with are those that operate on the solver, i.e. `solver_init_state`, `solver_update`, `solver_run`.

Typically, in `YourRegressor` you will call `self.solver_init_state` at the parameter initialization step, `self.sovler_run` in [`fit`](nemos.glm.GLM.fit), and `self.solver_update` in [`update`](nemos.glm.GLM.update).

:::{admonition} Solvers
:class: note

We implement a set of standard solvers in NeMoS, relying on various backends. In the future we are planning to add support for user-defined solvers, because in principle any object that adheres to our `AbstractSolver` interface should be compatible with NeMoS. For more information about the solver interface and solvers, see the [developer notes about solvers](07-solvers.md).
:::

## Parameter Containers

Model parameters are stored internally using `equinox.Module` containers, which are frozen dataclasses that are also valid JAX pytrees. These containers provide:

- **Named fields**: Self-documenting code (e.g., `params.coef`, `params.intercept`)
- **Type safety**: Full type annotation support
- **JAX compatibility**: Work seamlessly with `jax.jit`, `jax.grad`, etc.
- **Extensibility**: Easy to add model-specific hooks

For example, `GLMParams` includes a `regularizable_subtrees()` method that specifies which parameter components should be regularized. This allows regularizers and other components to interact with parameters in a controlled, model-aware manner.

## Validation Framework

Validation is handled by model-specific `RegressorValidator` subclasses (e.g., `GLMValidator` for GLM models). The validator:

1. Validates user-provided parameters and inputs
2. Converts user parameters to internal representation
3. Validates internal parameters
4. Checks consistency between parameters and data

This separation of concerns keeps the model classes focused on their core functionality while ensuring robust input validation.

## Contributor Guidelines

### Implementing Model Subclasses

When devising a new model subclass based on the `BaseRegressor` abstract class, adhere to the subsequent guidelines:

- **Must** inherit the `BaseRegressor` abstract superclass.
- **Must** realize the abstract methods, see above.
- **Should not** overwrite the `get_params` and `set_params` methods, inherited from `Base`.
- **May** introduce auxiliary methods for added utility.
- **May** re-implement the `__sklearn_tags__` method to add metadata that is relevant to the specific estimator implemented. See the [`scikit-learn` documentation](https://scikit-learn.org/stable/modules/generated/sklearn.utils.Tags.html#sklearn.utils.Tags) for the available tagging options.

:::{admonition} Tags
:class: note

Tags are not needed for estimators to function correctly, as long as the required methods (`fit`, `predict`,...) are implemented. Citing the `scikit-learn` [documentation for the estimator API](https://scikit-learn.org/stable/developers/develop.html#estimator-tags),

> Scikit-learn introduced estimator tags in version 0.21 as a private API and mostly used in tests. However, these tags expanded over time and many third party developers also need to use them. Therefore in version 1.6 the API for the tags was revamped and exposed as public API.

:::

## Glossary

|  Term   | Description |
|--------------------| ----------- |
| **Regularization** | Regularization is a technique used to prevent overfitting by adding a penalty to the loss function, which discourages complex models. Common regularization techniques include L1 (Lasso) and L2 (Ridge) regularization. |
| **Optimization**   | Optimization refers to the process of minimizing (or maximizing) a function by systematically choosing the values of the variables within an allowable set. In machine learning, optimization aims to minimize the loss function to train models. |
| **Solver**         | A solver is an algorithm or a set of algorithms used for solving optimization problems. In the given module, solvers are used to find the parameters that minimize the loss function, potentially subject to some constraints. |
| **Runner**         | A runner in this context refers to a callable function configured to execute the solver with the specified parameters and data. It acts as an interface to the solver, simplifying the process of running optimization tasks. |
