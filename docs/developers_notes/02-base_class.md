# The `base_class` Module

## Introduction

The `base_class` module introduces the `_Base` class and abstract classes defining broad model categories. These abstract classes **must** inherit from `_Base`. Currently, the sole abstract class available is `_BaseRegressor`.

The `_Base` class is envisioned as the foundational component for any model type (e.g., regression, dimensionality reduction, clustering, etc.). In contrast, abstract classes derived from `_Base` define overarching model categories (e.g., `_BaseRegressor` is building block for GLMs, GAMS, etc.).

Designed to be compatible with the `scikit-learn` API, the class structure aims to facilitate access to `scikit-learn`'s robust pipeline and cross-validation modules. This is achieved while leveraging the accelerated computational capabilities of `jax` and `jaxopt` in the backend, which is essential for analyzing extensive neural recordings and fitting large models.

Below a scheme of how we envision the architecture of the `neurostatslib` models.

```
Class _Base
|
└─ Abstract Subclass _BaseRegressor
│   │
│   └─ Abstract Subclass _BaseGLM
│       │
│       ├─ Concrete Subclass PoissonGLM
│       │   │
│       │   └─ Concrete Subclass RidgePoissonGLM *(not implemented yet)
│       │   │
│       │   └─ Concrete Subclass LassoPoissonGLM *(not implemented yet)
│       │   │
│       │   ...
│       │
│       ├─ Concrete Subclass GammaGLM *(not implemented yet)
│       │   │
│       │   ...
│       │
│       ...
│
├─ Abstract Subclass _BaseManifold *(not implemented yet)
...
```

!!! Example
    The current package version includes a concrete class named `neurostatslib.glm.PoissonGLM`. This class inherits from `_BaseGLM` <- `_BaseRegressor` <- `_Base`, since it falls under the " GLM regression" category. 
    As any `_BaseRegressor`, it **must** implement the `fit`, `score`, `predict`, and `simulate` methods.


## The Class `model_base._Base`

The `_Base` class aligns with the `scikit-learn` API for `base.BaseEstimator`. This alignment is achieved by implementing the `get_params` and `set_params` methods, essential for `scikit-learn` compatibility and foundational for all model implementations.

For a detailed understanding, consult the [`scikit-learn` API Reference](https://scikit-learn.org/stable/modules/classes.html) and [`BaseEstimator`](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html).

!!! Note
    We've intentionally omitted the `get_metadata_routing` method. Given its current experimental status and its lack of relevance to the `GLM` class, this method was excluded. Should future needs arise around parameter routing, consider directly inheriting from `sklearn.BaseEstimator`. More information can be found [here](https://scikit-learn.org/stable/metadata_routing.html#metadata-routing).

### Public methods

- **`get_params`**: The `get_params` method retrieves parameters set during model instance initialization. Opting for a deep inspection allows the method to assess nested object parameters, resulting in a comprehensive parameter dictionary.
- **`set_params`**: The `set_params` method offers a mechanism to adjust or set an estimator's parameters. It's versatile, accommodating both individual estimators and more complex nested structures like pipelines. Feeding an unrecognized parameter will raise a `ValueError`.

## The Abstract Class `model_base._BaseRegressor`

`_BaseRegressor` is an abstract class that inherits from `_Base`, stipulating the implementation of abstract methods: `fit`, `predict`, `score`, and `simulate`. This ensures seamless assimilation with `scikit-learn` pipelines and cross-validation procedures.

### Abstract Methods

For subclasses derived from `_BaseRegressor` to function correctly, they must implement the following:

1. `fit`: Adapt the model using input data `X` and corresponding observations `y`.
2. `predict`: Provide predictions based on the trained model and input data `X`.
3. `score`: Gauge the accuracy of model predictions using input data `X` against the actual observations `y`.
4. `simulate`: Simulate data based on the trained regression model.

Moreover, `_BaseRegressor` incorporates auxiliary methods such as `_convert_to_jnp_ndarray`, `_has_invalid_entry` 
and a number of other methods for checking input consistency.

!!! Tip
    Deciding between concrete and abstract methods in a superclass can be nuanced. As a general guideline: any method that's expected in all subclasses and isn't subclass-specific should be concretely implemented in the superclass. Conversely, methods essential for a subclass's expected behavior, but vary based on the subclass, should be abstract in the superclass. For instance, compatibility with the `sklearn.cross_validation` module demands `score`, `fit`, `get_params`, and `set_params` methods. Given their specificity to individual models, `score` and `fit` are abstract in `BaseRegressor`. Conversely, as `get_params` and `set_params` are consistent across model classes, they're inherited from `_Base`. This approach typifies our general implementation strategy. However, it's important to note that while these are sound guidelines, exceptions exist based on various factors like future extensibility, clarity, and maintainability.


## Contributor Guidelines

### Implementing Model Subclasses

When devising a new model subclass based on the `_BaseRegressor` abstract class, adhere to the subsequent guidelines:

- **Must** inherit the `_BaseRegressor` abstract superclass.
- **Must** realize the abstract methods: `fit`, `predict`, `score`, and `simulate`.
- **Should not** overwrite the `get_params` and `set_params` methods, inherited from `_Base`.
- **May** introduce auxiliary methods such as `_convert_to_jnp_ndarray` for added utility.
