## The Abstract Class `base_regressor.BaseRegressor`

`BaseRegressor` is an abstract class that inherits from `Base`, stipulating the implementation of number of abstract methods such as `fit`, `predict`, `score`. This ensures seamless assimilation with `scikit-learn` pipelines and cross-validation procedures.


!!! Example
    The current package version includes a concrete class named `nemos.glm.GLM`. This class inherits from `BaseRegressor`, which in turn inherits `Base`, since it falls under the " GLM regression" category. 
    As any `BaseRegressor`, it **must** implement the `fit`, `score`, `predict` methods.

### Abstract Methods

For subclasses derived from `BaseRegressor` to function correctly, they must implement the following:

1. `fit`: Adapt the model using input data `X` and corresponding observations `y`.
2. `predict`: Provide predictions based on the trained model and input data `X`.
3. `score`: Score the accuracy of model predictions using input data `X` against the actual observations `y`.
4. `simulate`: Simulate data based on the trained regression model.
5. `update`: Run a single optimization step, and stores the updated parameter and solver state. Used by stochastic optimization schemes.
6. `_predict_and_compute_loss`: Compute prediction and evaluates the loss function prvided the parameters and `X` and `y`.
7. `_check_params`: Check the parameter structure.
8. `_check_input_dimensionality`: Check the input dimensionality matches model expectation.
9. `_check_input_and_params_consistency`: Checks that the input and the parameters are consistent.
10. `_get_coef_and_intercept` and `_set_coef_and_intercept`: set and get model coefficient and intercept term.


### Attributes

Public attributes are stored as properties:

- `regularizer`: An instance of the `nemos.regularizer.Regularizer` class. The setter for this property accepts either the instance directly or a string that is used to instantiate the appropriate regularizer.
- `regularizer_strength`: A float quantifying the amount of regularization.
- `solver_name`: One of the `jaxopt` solver supported solvers, currently "GradientDescent", "BFGS", "LBFGS", "ProximalGradient" and, "NonlinearCG".
- `solver_kwargs`: Extra keyword arguments to be passed at solver initialization.
- `solver_init_state`, `solver_update`, `solver_run`: Read-only property with a partially evaluated `solver.init_state`, `solver.update` and, `solver.run` methods. The partial evaluation guarantees for a consistent API for all solver.

## Contributor Guidelines

### Implementing Model Subclasses

When devising a new model subclass based on the `BaseRegressor` abstract class, adhere to the subsequent guidelines:

- **Must** inherit the `BaseRegressor` abstract superclass.
- **Must** realize the abstract methods, see above.
- **Should not** overwrite the `get_params` and `set_params` methods, inherited from `Base`.
- **May** introduce auxiliary methods for added utility.
