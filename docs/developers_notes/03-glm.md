# The `glm` Module

## Introduction



Generalized Linear Models (GLM) provide a flexible framework for modeling a variety of data types while establishing a relationship between multiple predictors and a response variable. A GLM extends the traditional linear regression by allowing for response variables that have error distribution models other than a normal distribution, such as binomial or Poisson distributions.

The `neurostatslib.glm` module currently  offers implementations of two GLM classes:

1. **`GLM`:** A direct implementation of a feedforward GLM.
2. **`RecurrentGLM`:** An implementation of a recurrent GLM. This class inherits from `GLM` and redefines the `simulate` method to generate spikes akin to a recurrent neural network.

Our design is harmonized with the `scikit-learn` API, facilitating seamless integration of our GLM classes with the well-established `scikit-learn` pipeline and its cross-validation tools.

The classes provided here are modular by design offering a standard foundation for any GLM variant. 

Instantiating a specific GLM simply requires providing an observation noise model (Gamma, Poisson, etc.) and a regularization strategies (Ridge, Lasso, etc.) during initialization. This is done using the `neurostatslib.noise_model.NoiseModel` and `neurostatslib.solver.Solver` objects, respectively.


<figure markdown>
    <img src="../GLM_scheme.jpg"/>
    <figcaption>Schematic of the module interactions.</figcaption>
</figure>



## The Concrete Class `GLM`

The `GLM` class provides a direct implementation of the GLM model and is designed with `scikit-learn` compatibility in mind.

### Inheritance

`GLM` inherits from `BaseRegressor`. This inheritance mandates the direct implementation of methods like `predict`, `fit`, `score`, and `simulate`.

### Attributes

- **`solver`**: Refers to the optimization solver - an object of the `neurostatslib.solver.Solver` type. It uses the `jaxopt` solver to minimize the (penalized) negative log-likelihood of the GLM.
- **`noise_model`**: Represents the GLM noise model, which is an object of the `neurostatlib.noise_model.NoiseModel` type. This model determines the log-likelihood and the emission probability mechanism for the `GLM`.
- **`basis_coeff_`**: Stores the solution for spike basis coefficients as `jax.ndarray` after the fitting process. It is initialized as `None` during class instantiation.
- **`baseline_link_fr_`**: Stores the bias terms' solutions as `jax.ndarray` after the fitting process. It is initialized as `None` during class instantiation.
- **`solver_state`**: Indicates the solver's state. For specific solver states, refer to the [`jaxopt` documentation](https://jaxopt.github.io/stable/index.html#).

### Public Methods

- **`predict`**: Validates input and computes the mean rates of the `GLM` by invoking the inverse-link function of the `noise_model` attribute.
- **`score`**: Validates input and assesses the Poisson GLM using either log-likelihood or pseudo-$R^2$. This method uses the `noise_model` to determine log-likelihood or pseudo-$R^2$.
- **`fit`**: Validates input and aligns the Poisson GLM with spike train data. It leverages the `noise_model` and `solver` to define the model's loss function and instantiate the solver.
- **`simulate`**: Simulates spike trains using the GLM as a feedforward network, invoking the `noise_model.emission_probability` method for emission probability.

### Private Methods

- **`_predict`**: Forecasts rates based on current model parameters and the inverse-link function of the `noise_model`.
- **`_score`**: Determines the Poisson negative log-likelihood, excluding normalization constants.
- **`_check_is_fit`**: Validates whether the model has been appropriately fit by ensuring model parameters are set. If not, a `NotFittedError` is raised.


## The Concrete Class `RecurrentGLM`

The `RecurrentGLM` class is an extension of the `GLM`, designed to simulate models with recurrent connections. It inherits the `predict`, `fit`, and `score` methods from `GLM`, but provides its own implementation for the `simulate` method.

### Overridden Methods

- **`simulate`**: This method simulates spike trains, treating the GLM as a recurrent neural network. It utilizes the `noise_model.emission_probability` method to determine the emission probability.

## Contributor Guidelines

### Implementing Model Subclasses

When crafting a functional (i.e., concrete) GLM class:

- **Must** inherit from `BaseRegressor` or one of its derivatives.
- **Must** realize the `predict`, `fit`, `score`, and `simulate` methods, either directly or through inheritance.
- **Should** incorporate a `noise_model` attribute of type `neurostatslib.noise_model.NoiseModel` to specify the link-function, emission probability, and likelihood.
- **Should** include a `solver` attribute of type `neurostatslib.solver.Solver` to establish the solver based on penalization type.
- **May** embed additional parameter and input checks if required by the specific GLM subclass.
