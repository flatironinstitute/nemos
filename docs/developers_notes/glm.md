# The PoissonGLMBase Class

## Introduction

The `PoissonGLMBase` class serves as the foundation for implementing Poisson Generalized Linear Models (GLMs). These models are essential for analyzing neural data and other related time-series data. The class encapsulates various functionalities required for model definition, fitting, prediction, and scoring, all of which are crucial aspects of modeling neural activity.

The core features of this class are centered around abstract methods that must be implemented by any concrete subclasses, ensuring a standardized interface for all types of Poisson GLM models.

## The Class `PoissonGLMBase`

### Initialization and Configuration

The `PoissonGLMBase` class's constructor initializes various parameters and settings essential for the model's behavior. These include:

- `solver_name`: The name of the optimization solver to be used.
- `solver_kwargs`: Additional keyword arguments for the chosen solver.
- `inverse_link_function`: The callable function for the inverse link transformation.
- `score_type`: The type of scoring method, either "log-likelihood" or "pseudo-r2".

### Method `fit`

The `fit` method is an abstract method that needs to be implemented by subclasses. It is used to train the Poisson GLM model using input data `X` and spike data. The method performs the model fitting process by optimizing the provided loss function.

### Method `predict`

The `predict` method takes input data `X` and predicts firing rates using the trained Poisson GLM model. It leverages the inverse link function to transform the model's internal parameters into meaningful predictions.

### Method `score`

The `score` method evaluates the performance of the Poisson GLM model. It computes a score based on the model's predictions and the true spike data. The score can be either the negative log-likelihood or a pseudo-R2 score, depending on the specified `score_type`.

### Internal Methods

The class defines several internal methods that aid in the implementation of its functionalities:

- `_predict`: A specialized prediction method that calculates firing rates using model parameters and input data.
- `_score`: A specialized scoring method that computes a score based on predicted firing rates, true spike data, and model parameters.
- `_residual_deviance`: Calculates the residual deviance of the model's predictions.
- `_pseudo_r2`: Computes the pseudo-R2 score based on the model's predictions, true spike data, and model parameters.
- `_check_is_fit`: Ensures that the instance has been fitted before making predictions or scoring.
- `_check_and_convert_params`: Validates and converts initial parameters to the appropriate format.
- `_check_input_dimensionality`: Checks the dimensionality of input data and spike data to ensure consistency.
- `_check_input_and_params_consistency`: Validates the consistency between input data, spike data, and model parameters.
- `_check_input_n_timepoints`: Verifies that the number of time points in input data and spike data match.
- `_preprocess_fit`: Prepares input data, spike data, and initial parameters for the fitting process.

### Method `simulate`

The `simulate` method generates simulated spike data using the trained Poisson GLM model. It takes into account various parameters, including random keys, coupling basis matrix, and feedforward input. The simulated data can be generated for different devices, such as CPU, GPU, or TPU.

## The Class `PoissonGLM`

### Initialization

The `PoissonGLM` class extends the `PoissonGLMBase` class and provides a concrete implementation. It inherits the constructor from its parent class and allows additional customization through the specified parameters.

### Method `fit`

The `fit` method is implemented in the `PoissonGLM` class to perform the model fitting process using the provided input data and spike data. It leverages optimization solvers and loss functions to update the model's internal parameters.

This script defines a powerful framework for creating and training Poisson Generalized Linear Models, essential for analyzing and understanding neural activity patterns.
