# The `base_class` Module

## Introduction

The `base_class` module introduces the `Base` class and abstract classes defining broad model categories. These abstract classes **must** inherit from `Base`.

The `Base` class is envisioned as the foundational component for any object type (e.g., regression, dimensionality reduction, clustering, observation models, regularizers etc.). In contrast, abstract classes derived from `Base` define overarching object categories (e.g., `base_class.BaseRegressor` is building block for GLMs, GAMS, etc. while `observation_models.Observations` is the building block for the Poisson observations, Gamma observations, ... etc.).

Designed to be compatible with the `scikit-learn` API, the class structure aims to facilitate access to `scikit-learn`'s robust pipeline and cross-validation modules. This is achieved while leveraging the accelerated computational capabilities of `jax` and `jaxopt` in the backend, which is essential for analyzing extensive neural recordings and fitting large models.

Below a scheme of how we envision the architecture of the `nemos` models.

```
Abstract Class Base
│
├─ Abstract Subclass BaseRegressor
│   │
│   └─ Concrete Subclass GLM
│       │
│       └─ Concrete Subclass RecurrentGLM
│
├─ Abstract Subclass BaseManifold *(not implemented yet)
│   │
│   ...
│
├─ Abstract Subclass Regularizer
│   │
│   ├─ Concrete Subclass UnRegularized
│   │
│   ├─ Concrete Subclass Ridge
│   ... 
│
├─ Abstract Subclass Observations
│   │
│   ├─ Concrete Subclass PoissonObservations
│   │
│   ├─ Concrete Subclass GammaObservations *(not implemented yet)
│   ... 
│
...
```

!!! Example
    The current package version includes a concrete class named `nemos.glm.GLM`. This class inherits from `BaseRegressor`, which in turn inherits `Base`, since it falls under the " GLM regression" category. 
    As any `BaseRegressor`, it **must** implement the `fit`, `score`, `predict`, and `simulate` methods.


## The Class `model_base.Base`

The `Base` class aligns with the `scikit-learn` API for `base.BaseEstimator`. This alignment is achieved by implementing the `get_params` and `set_params` methods, essential for `scikit-learn` compatibility and foundational for all model implementations. Additionally, the class provides auxiliary helper methods to identify available computational devices (such as GPUs and TPUs) and to facilitate data transfer to these devices.

For a detailed understanding, consult the [`scikit-learn` API Reference](https://scikit-learn.org/stable/modules/classes.html) and [`BaseEstimator`](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html).

!!! Note
    We've intentionally omitted the `get_metadata_routing` method. Given its current experimental status and its lack of relevance to the `GLM` class, this method was excluded. Should future needs arise around parameter routing, consider directly inheriting from `sklearn.BaseEstimator`. More information can be found [here](https://scikit-learn.org/stable/metadata_routing.html#metadata-routing).

### Public methods

- **`get_params`**: The `get_params` method retrieves parameters set during model instance initialization. Opting for a deep inspection allows the method to assess nested object parameters, resulting in a comprehensive parameter dictionary.
- **`set_params`**: The `set_params` method offers a mechanism to adjust or set an estimator's parameters. It's versatile, accommodating both individual estimators and more complex nested structures like pipelines. Feeding an unrecognized parameter will raise a `ValueError`.

## The Abstract Class `model_base.BaseRegressor`

`BaseRegressor` is an abstract class that inherits from `Base`, stipulating the implementation of abstract methods: `fit`, `predict`, `score`, and `simulate`. This ensures seamless assimilation with `scikit-learn` pipelines and cross-validation procedures.

### Abstract Methods

For subclasses derived from `BaseRegressor` to function correctly, they must implement the following:

1. `fit`: Adapt the model using input data `X` and corresponding observations `y`.
2. `predict`: Provide predictions based on the trained model and input data `X`.
3. `score`: Score the accuracy of model predictions using input data `X` against the actual observations `y`.
4. `simulate`: Simulate data based on the trained regression model.

### Public Methods

To ensure the consistency and conformity of input data, the `BaseRegressor` introduces two public preprocessing methods:

1. `preprocess_fit`: Assesses and converts the input for the `fit` method into the desired `jax.ndarray` format. If necessary, this method can initialize model parameters using default values.
2. `preprocess_simulate`: Validates and converts inputs for the `simulate` method. This method confirms the integrity of the feedforward input and, when provided, the initial values for feedback.

### Auxiliary Methods

Moreover, `BaseRegressor` incorporates auxiliary methods such as `_convert_to_jnp_ndarray`, `_has_invalid_entry` 
and a number of other methods for checking input consistency.

!!! Tip
    Deciding between concrete and abstract methods in a superclass can be nuanced. As a general guideline: any method that's expected in all subclasses and isn't subclass-specific should be concretely implemented in the superclass. Conversely, methods essential for a subclass's expected behavior, but vary based on the subclass, should be abstract in the superclass. For instance, compatibility with the `sklearn.cross_validation` module demands `score`, `fit`, `get_params`, and `set_params` methods. Given their specificity to individual models, `score` and `fit` are abstract in `BaseRegressor`. Conversely, as `get_params` and `set_params` are consistent across model classes, they're inherited from `Base`. This approach typifies our general implementation strategy. However, it's important to note that while these are sound guidelines, exceptions exist based on various factors like future extensibility, clarity, and maintainability.


## Contributor Guidelines

### Implementing Model Subclasses

When devising a new model subclass based on the `BaseRegressor` abstract class, adhere to the subsequent guidelines:

- **Must** inherit the `BaseRegressor` abstract superclass.
- **Must** realize the abstract methods: `fit`, `predict`, `score`, and `simulate`.
- **Should not** overwrite the `get_params` and `set_params` methods, inherited from `Base`.
- **May** introduce auxiliary methods such as `_convert_to_jnp_ndarray` for added utility.
