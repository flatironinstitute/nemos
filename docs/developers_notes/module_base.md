# The ModuleBase Module

## Introduction

The `module_base` module primarily defines the abstract class `Model`, laying the groundwork for model definitions that ensure compatibility with sci-kit learn pipelines. The functionalities include the retrieval and setting of parameters, prediction, fitting, and scoring, all essential to any modeling process.


The core functionality is centered around abstract methods that need to be implemented by any concrete subclasses, ensuring a consistent interface for all types of models.

## The Class `module_base.Model`

### The Public Method `get_params`

`get_params` retrieves the parameters set during the initialization of the model instance. If deep inspection is chosen, the method also inspects nested objects' parameters. The return is a dictionary of parameters.

### The Public Method `set_params`

This method allows users to set or modify the parameters of an estimator. The method is designed to work with both simple estimators and nested objects, like pipelines. If an invalid parameter is passed, a `ValueError` will be raised.

### Abstract Methods

Any concrete subclass inheriting from `Model` must implement these abstract methods to be functional:

1. `fit`: To fit the model using data `X` and labels `y`.
2. `predict`: Make predictions based on the model and data `X`.
3. `score`: Score the model's predictions on data `X` with true labels `y`.
4. `_predict`: A more specialized prediction method that takes model parameters and data `X`.
5. `_score`: A specialized scoring method that takes data `X`, true labels `y`, and model parameters.

Additionally, the class has helper methods like `_get_param_names` to fetch parameter names and `_convert_to_jnp_ndarray` to convert data to JAX NumPy arrays.

## Contributor Guidelines

### Implementing Model Subclasses
When aiming to introduce a new model by inheriting the abstract `Model` class, ensure the following:

- **Must** inherit the abstract superclass `Model`.
- **Must** define the abstract methods `fit`, `predict`, `score`, `_predict`, and `_score`.
- **Should not** overwrite the `get_params` and `set_params` methods inherited from `Model`.
- **May** utilize helper methods like `_get_param_names` for convenience.

