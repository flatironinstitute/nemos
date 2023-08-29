# The `glm` Module

## Introduction

The `neurostatslib.glm` module implements variations of Generalized Linear Models (GLMs) classes. 

At this stage, the module consists of two primary classes:

1. **`_BaseGLM`:** An abstract class serving as the backbone for building GLMs.
2. **`PoissonGLM`:** A concrete implementation of the GLM for Poisson-distributed data.

Our design aligns with the `scikit-learn` API. This ensures that our GLM classes integrate seamlessly with the robust `scikit-learn` pipeline and its cross-validation capabilities.

## The class `_BaseGLM`

Designed with `scikit-learn` compatibility in mind, `_BaseGLM` provides the common computations and functionalities needed by the diverse `GLM` subclasses.
### Inheritance

The `_BaseGLM` inherits attributes and methods from the `_BaseRegressor`, as detailed in the [`base_class` module](02-base_class.md). This grants `_BaseGLM` a toolkit for managing and verifying model inputs. Leveraging the inherited abstraction, all GLM subclasses must explicitly define the `fit`, `predict`, `score`, and `simulate` methods, ensuring alignment with the `scikit-learn` framework.

### Attributes

- **`solver`**: The optimization solver from jaxopt.
- **`solver_state`**: Represents the current state of the solver.
- **`basis_coeff_`**: Holds the solution for spike basis coefficients after the model has been fitted. Initialized to `None` at class instantiation.
- **`baseline_link_fr`**: Contains the bias terms' solutions after fitting. Initialized to `None` at class instantiation.
- **`kwargs`**: Other keyword arguments, like regularization hyperparameters.


### Public Methods

1. **`predict`**: Validates the model's fit status and input consistency before calculating mean rates using the `_predict` method.

!!! note
     `_BaseGLM` lacks concrete implementations for methods like `score`, `fit`, and `simulate` since the specific behaviors of these methods are contingent upon their emission probability. For instance, the scoring method for a Poisson GLM will differ from a Gamma GLM, given their distinct likelihood functions.

### Private Methods

1. **`_check_is_fit`**: Ensures the instance has been fitted. This check is implemented here and not in `_BaseRegressor` because the model parameters are likely to be GLM specific.
2. **`_predict`**: Forecasts firing rates based on predictors and parameters.
3. **`_pseudo_r2`**: Computes the Pseudo-$R^2$ for a GLM, giving insight into the model's fit relative to a null model.
4. **`_safe_score`**: Scores the predicted firing rates against target spike counts. Can compute either the GLM mean log-likelihood or the pseudo-$R^2$.
5. **`_safe_fit`**: Fit the GLM to the neural activity. Verifies input conformity, then leverages the `jaxopt` optimizer on the designated loss function (provided by the concrete GLM subclass).
6. **`_safe_simulate`**: Simulates spike trains using the GLM as a recurrent network. It projects neural activity into the future using the fitted parameters of the GLM. The function can simulate activity based on both historical spike activity and external feedforward inputs, such as convolved currents, light intensities, etc.


!!! note
    The introduction of `_safe_score` and `_safe_simulate` offers notable benefits:

    1. It eliminates the need for subclasses to redo checks in their `score` and `simulate` methods, leading to concise code.
    2. The methods `score` and `simulate` must be defined by subclasses due to their abstract nature in `_BaseRegressor`. This ensures subclass-specific docstrings for public methods.

### Abstract Methods
Besides the methods acquired from `_BaseRegressor`, `_BaseGLM` introduces:

1. **`residual_deviance`**: Computes a GLM's residual deviance. The deviance, on par with the likelihood, is model specific.

!!! note
    The residual deviance can be formulated as a function of log-likelihood. Although a concrete `_BaseGLM` implementation is feasible, subclass-specific implementations might offer increased robustness or efficiency.

## The Concrete Class `PoissonGLM`

The class `PoissonGLM` is a concrete implementation of the un-regularized Poisson GLM model. 

### Inheritance

`PoissonGLM` inherits from `_BaseGLM`, which provides methods for predicting firing rates and "safe" methods to score and simulate spike trains. Inheritance enforces the concrete implementation of `fit`, `score`, `simulate`, and `residual_deviance`.

### Attributes

- **`solver`**: The optimization solver from jaxopt.
- **`solver_state`**: Represents the current state of the solver.
- **`basis_coeff_`**: Holds the solution for spike basis coefficients after the model has been fitted. Initialized to `None` at class instantiation.
- **`baseline_link_fr`**: Contains the bias terms' solutions after fitting. Initialized to `None` at class instantiation.


### Public Methods

- **`score`**: Scores the Poisson GLM using either log-likelihood or pseudo-$R^2$. It invokes the parent `_safe_score` method to validate input and parameters.
- **`fit`**: Fits the Poisson GLM to align with spike train data by invoking `_safe_fit` and setting Poisson negative log-likelihood as the loss function.
- **`residual_deviance`**: Computes the residual deviance for each Poisson model observation, given predicted rates and spike counts.
- **`simulate`**: Simulates spike trains using the GLM as a recurrent network, invoking `_safe_simulate` and setting `jax.random.poisson` as the emission probability mechanism.

### Private Methods

- **`_score`**: Computes the Poisson negative log-likelihood up to a normalization constant. This method is used to define the optimization loss function for the model.

## Contributor Guidelines

### Implementing Model Subclasses

To write a usable (i.e. concrete) GLM class you

- **Must** inherit `_BaseGLM` or any of its subclasses.
- **Must** implement the `fit`, `score`, `simulate`, and `residual_deviance` methods, either directly or through inheritance.
- **Should** invoke `_safe_fit`, `_safe_score`, and `_safe_simulate` within the `fit`, `score`, and `simulate` methods, respectively.
- **Should not** override `_safe_fit`, `_safe_score`, or `_safe_simulate`.
- **May** integrate supplementary parameter and input checks if mandated by the GLM subclass.