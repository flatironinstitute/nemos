# The `glm` Module

## Introduction

The `neurostatslib.glm` basis module implements variations of Generalized Linear Models (GLMs) classes. 

At stage, the module consists of two classes:

1. **`_BaseGLM`:** An abstract class serving as the backbone for building GLM variations.
2. **`PoissonGLM`:** A concrete implementation of the GLM for Poisson-distributed data.

We followed the `scikit-learn` api, making the concrete GLM model classes compatible with the powerful `scikit-learn` pipeline and cross-validation modules.

## The class `_BaseGLM`

The class `_BaseGLM` is designed to follow the `scikit-learn` api in order to guarantee compatibility with the `scikit-learn` pipelines, as well as to implement all the computation that is shared by the different `GLM` subclasses. 

### Inheritance

`_BaseGLM` inherits from the `_BaseRegressor` (detailed in the [`base_class` module](base_class.md)).  This inheritance provides `_BaseGLM` with a suite of auxiliary methods for handling and validating model inputs. Through abstraction mechanism inherited from `_BaseRegressor`, any GLM subclass is compelled to reimplement the `fit`, `predict`, `score`, and `simulate` methods facilitating compatibility with `scikit-learn`.

### Attributes

- **solver_name**: Name of the solver to use when fitting the GLM. It should be an attribute of `jaxopt`.
- **solver_kwargs**: Dictionary containing keyword arguments for initializing the solver.
- **inverse_link_function**: The link function of the GLM. It must be callable and return non-negative values.
- **kwargs**: Other keyword arguments, like regularization hyperparameters.


### Public Methods

1. **`predict`**: This method checks that the model is fit and validates input consistency and dimensions, and computes mean rates based on the current parameter estimates through the `_predict` method.

!!! note
     `_BaseGLM` lacks concrete implementations for methods like `score`, `fit`, and `simulate`. This is because the specifics of these methods depend on the chosen emission probability. For instance, the scoring method for a Poisson GLM will differ from a Gamma GLM, given their distinct likelihood functions.

### Private Methods

1. **`_check_is_fit`**: Ensures the instance has been fitted. This check is implemented here and not in `_BaseRegressor` because the model parameter names are likely to be GLM specific.
2. **`_predict`**: Predicts firing rates given predictors and parameters.
3. **`_pseudo_r2`**: Computes the Pseudo-$R^2$ for a GLM, giving insight into the model's fit relative to a null model.
4. **`_safe_score`**: Scores the predicted firing rates against target spike counts. Can compute either the GLM mean log-likelihood or the pseudo-$R^2$.
5. **`_safe_simulate`**: Simulates spike trains using the GLM as a recurrent network. It projects neural activity into the future using the fitted parameters of the GLM. The function can simulate activity based on both historical spike activity and external feedforward inputs, such as convolved currents, light intensities, etc.


!!! note
    The introduction of `_safe_score` and `_safe_simulate` offers notable benefits:

    1. It eliminates the need for subclasses to redo checks in their `score` and `simulate` methods, leading to concise code.
    2. The methods `score` and `simulate` must be defined by subclasses due to their abstract nature in `_BaseRegressor`. This mandates alignment with the `scikit-learn` API and ensures subclass-specific docstrings.

### Abstract Methods
On top of the abstract methods inherited from `_BaseRegressor`, `_BaseGLM` implements,

1. **`residual_deviance`**: Computes the residual deviance for a GLM model. The deviance, on par with the likelihood, is model specific.

!!! note
    The residual deviance can be written as a function of the log-likelihood. This allows for a concrete implementation of it in the `_BaseGLM`, however the subclass specific implementation can be more robust and/or efficient.

## The Concrete Class `PoissonGLM`

The class `PoissonGLM` implements an un-regularized Poisson GLM model. 

