# Construct Design Matrices for Categorical Features

Capturing the effect of stimulus identity, behavioral choices, etc., are all common examples of model designs requiring the encoding of categorical predictors. In this note we will explore how to construct such designs with NeMoS `Categorical` basis, or using specialized packages ([`patsy`](https://patsy.readthedocs.io) or [`formulaic`](https://matthewwardrop.github.io/formulaic/)) that handles gracefully complex schemes that includes multiple categorical predictors and a variety of encoding schemes.

Such packages takes care transparently of a well-know identifiability issue arising with multiple categorical predictors: that such designs may result in non-identifiable models, e.g. models that have multiple equivalent solutions. We will expand on that in a dedicated technical note for people that wants to dig deeper into the problem.

# Index

```{toctree}
:maxdepth: 3

categorical_predictors.md
categorical_identifiability.md
```
