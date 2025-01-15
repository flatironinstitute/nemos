# The Abstract Class `BaseRegressor`

`BaseRegressor` is an abstract class that inherits from `Base`, stipulating the implementation of number of abstract methods such as [`fit`](nemos.glm.GLM.fit), [`predict`](nemos.glm.GLM.predict), [`score`](nemos.glm.GLM.score). This ensures seamless assimilation with `scikit-learn` pipelines and cross-validation procedures.


:::{admonition} Example
:class: note

The current package version includes a concrete class named [`nemos.glm.GLM`](nemos.glm.GLM). This class inherits from `BaseRegressor`, which in turn inherits `Base`, since it falls under the "GLM regression" category. 
As a `BaseRegressor`, it **must** implement the [`fit`](nemos.glm.GLM.fit), [`score`](nemos.glm.GLM.score), [`predict`](nemos.glm.GLM.predict) and the other abstract methods of this class, see below.
:::

### Abstract Methods

For subclasses derived from `BaseRegressor` to function correctly, they must implement the following:

1. [`fit`](nemos.glm.GLM.fit): Adapt the model using input data `X` and corresponding observations `y`.
2. [`predict`](nemos.glm.GLM.predict): Provide predictions based on the trained model and input data `X`.
3. [`score`](nemos.glm.GLM.score): Score the accuracy of model predictions using input data `X` against the actual observations `y`.
4. [`simulate`](nemos.glm.GLM.simulate): Simulate data based on the trained regression model.
5. [`update`](nemos.glm.GLM.update): Run a single optimization step, and stores the updated parameter and solver state. Used by stochastic optimization schemes.
6. `_predict_and_compute_loss`: Compute prediction and evaluates the loss function prvided the parameters and `X` and `y`. This is used by the `instantiate_solver` method which sets up the solver.
7. `_check_params`: Check the parameter structure.
8. `_check_input_dimensionality`: Check the input dimensionality matches model expectation.
9. `_check_input_and_params_consistency`: Checks that the input and the parameters are consistent.
10. `_get_coef_and_intercept` and `_set_coef_and_intercept`: set and get model coefficient and intercept term.

All the `_check_<method-name>` methods are called by the `_validate` method which checks that the provided 
input and parameters conform with the model requirements.

### Attributes

Public attributes are stored as properties:

- `regularizer`: An instance of the [`nemos.regularizer.Regularizer`](nemos.regularizer.Regularizer) class. The setter for this property accepts either the instance directly or a string that is used to instantiate the appropriate regularizer.
- `regularizer_strength`: A float quantifying the amount of regularization.
- `solver_name`: One of the `jaxopt` solver supported solvers, currently "GradientDescent", "BFGS", "LBFGS", "ProximalGradient" and, "NonlinearCG".
- `solver_kwargs`: Extra keyword arguments to be passed at solver initialization.
- `solver_init_state`, `solver_update`, `solver_run`: Read-only property with a partially evaluated `solver.init_state`, `solver.update` and, `solver.run` methods. The partial evaluation guarantees a consistent API for all solvers.

When implementing a new subclass of `BaseRegressor`, the only attributes you must interact directly with are those that operate on the solver, i.e. `solver_init_state`, `solver_update`, `solver_run`.

Typically, in `YourRegressor` you will call `self.solver_init_state` at the parameter initialization step, `self.sovler_run` in [`fit`](nemos.glm.GLM.fit), and `self.solver_update` in [`update`](nemos.glm.GLM.update). 

:::{admonition} Solvers
:class: note 

Solvers are typically optimizers from the `jaxopt` package, but in principle they could be custom optimization routines as long as they respect the `jaxopt` api (i.e., have a `run`, `init_state`, and [`update`](nemos.glm.GLM.update) method with the appropriate input/output types).
We rely on `jaxopt` because it provides a comprehensive set of robust, GPU accelerated, batchable and differentiable optimizers in JAX, that are highly customizable. In the future we may provide a number of custom solvers optimized for convex stochastic optimization.
:::

## Contributor Guidelines

### Implementing Model Subclasses

When devising a new model subclass based on the `BaseRegressor` abstract class, adhere to the subsequent guidelines:

- **Must** inherit the `BaseRegressor` abstract superclass.
- **Must** realize the abstract methods, see above.
- **Should not** overwrite the `get_params` and `set_params` methods, inherited from `Base`.
- **May** introduce auxiliary methods for added utility.


## Glossary

|  Term   | Description |
|--------------------| ----------- |
| **Regularization** | Regularization is a technique used to prevent overfitting by adding a penalty to the loss function, which discourages complex models. Common regularization techniques include L1 (Lasso) and L2 (Ridge) regularization. |
| **Optimization**   | Optimization refers to the process of minimizing (or maximizing) a function by systematically choosing the values of the variables within an allowable set. In machine learning, optimization aims to minimize the loss function to train models. |
| **Solver**         | A solver is an algorithm or a set of algorithms used for solving optimization problems. In the given module, solvers are used to find the parameters that minimize the loss function, potentially subject to some constraints. |
| **Runner**         | A runner in this context refers to a callable function configured to execute the solver with the specified parameters and data. It acts as an interface to the solver, simplifying the process of running optimization tasks. |
