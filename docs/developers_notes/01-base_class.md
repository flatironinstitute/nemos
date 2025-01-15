# The `base_class` Module

## Introduction

The `base_class` module introduces the `Base` class and abstract classes defining broad model categories and feature constructors. These abstract classes **must** inherit from `Base`.

The `Base` class is envisioned as the foundational component for any object type (e.g., basis, regression, dimensionality reduction, clustering, observation models, regularizers etc.). In contrast, abstract classes derived from `Base` define overarching object categories (e.g., `base_regressor.BaseRegressor` is building block for GLMs, GAMS, etc. while [`observation_models.Observations`](nemos.observation_models.Observations) is the building block for the Poisson observations, Gamma observations, ... etc.).

Designed to be compatible with the `scikit-learn` API, the class structure aims to facilitate access to `scikit-learn`'s robust pipeline and cross-validation modules. This is achieved while leveraging the accelerated computational capabilities of `jax` and `jaxopt` in the backend, which is essential for analyzing extensive neural recordings and fitting large models.

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

The `Base` class aligns with the `scikit-learn` API for `base.BaseEstimator`. This alignment is achieved by implementing the `get_params` and `set_params` methods, essential for `scikit-learn` compatibility and foundational for all model implementations.

For a detailed understanding, consult the [`scikit-learn` API Reference](https://scikit-learn.org/stable/modules/classes.html) and [`BaseEstimator`](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html).

:::{note}
We've intentionally omitted the `get_metadata_routing` method. Given its current experimental status and its lack of relevance to the [`GLM`](nemos.glm.GLM) class, this method was excluded. Should future needs arise around parameter routing, consider directly inheriting from `sklearn.BaseEstimator`. More information can be found [here](https://scikit-learn.org/stable/metadata_routing.html#metadata-routing).
:::

### Public methods

- **`get_params`**: The `get_params` method retrieves parameters set during model instance initialization. Opting for a deep inspection allows the method to assess nested object parameters, resulting in a comprehensive parameter dictionary.
- **`set_params`**: The `set_params` method offers a mechanism to adjust or set an estimator's parameters. It's versatile, accommodating both individual estimators and more complex nested structures like pipelines. Feeding an unrecognized parameter will raise a `ValueError`.

