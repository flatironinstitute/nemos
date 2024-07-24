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
