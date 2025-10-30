# The `base_class` Module

## Introduction

The `base_class` module introduces the `Base` class and abstract classes defining broad model categories and feature constructors. These abstract classes **must** inherit from `Base`.

The `Base` class is envisioned as the foundational component for any object type (e.g., basis, regression, dimensionality reduction, clustering, observation models, regularizers etc.). In contrast, abstract classes derived from `Base` define overarching object categories (e.g., `base_regressor.BaseRegressor` is building block for GLMs, GAMS, etc. while [`observation_models.Observations`](nemos.observation_models.Observations) is the building block for the Poisson observations, Gamma observations, ... etc.).

Designed to be compatible with the `scikit-learn` API, the class inherits directly from [`sklearn.BaseEstimator`](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator). The class facilitate access to `scikit-learn`'s robust pipeline and cross-validation modules, while customizing the `set_param` method for working with NeMoS basis objects. This is achieved while leveraging the accelerated computational capabilities of `jax` in the backend, which is essential for analyzing extensive neural recordings and fitting large models.

Below a scheme of how we envision the architecture of the NeMoS models.

```
Abstract Class Base
│
├─ Abstract Subclass BaseRegressor
│   │
│   └─ Concrete Subclass GLM
│       │
│       └─ Concrete Subclass PopulationGLM
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
│   ├─ Concrete Subclass GammaObservations
│   ...
│
...
```


## The Class `model_base.Base`

The `Base` class inherits from `scikit-learn` `base.BaseEstimator`, but reimplements the`set_params` method, essential for compatibility with both `scikit-learn` and NeMoS composite basis. This class is foundational for all model implementations as well as for our basis objects.

For a detailed understanding, consult the [`scikit-learn` API Reference](https://scikit-learn.org/stable/modules/classes.html) and [`BaseEstimator`](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html).


### Public methods

- **`get_params`**: The `get_params` method retrieves parameters set during model instance initialization. Opting for a deep inspection allows the method to assess nested object parameters, resulting in a comprehensive parameter dictionary.
- **`set_params`**: The `set_params` method offers a mechanism to adjust or set an estimator's parameters. It's versatile, accommodating both individual estimators and more complex nested structures like pipelines. Feeding an unrecognized parameter will raise a `ValueError`.
- **`get_metadata_routing`**: Returns a routing object that describes how keyword arguments (metadata) are handled by the estimator’s public methods (e.g., `fit`, `predict`). This is used internally by scikit-learn to automatically pass relevant metadata through routers (pipelines, meta-estimators, and cross-validation), ensuring that additional information like score type ($R^2$ vs log-likelihood) are routed correctly.
