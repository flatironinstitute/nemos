# Categorical Predictors

Capturing the effect of stimulus identity, behavioral choices, etc., are all common examples of model designs requiring the encoding of categorical predictors. In this note we will explore how to construct such designs with NeMoS `Categorical` basis. It also covers specialized packages ([`patsy`](https://patsy.readthedocs.io) or [`formulaic`](https://matthewwardrop.github.io/formulaic/))  which elegantly handle complex schemes involving multiple categorical predictors and diverse encoding methods.

These packages transparently resolve a well-known identifiability issue: designs with multiple categorical predictors can result in non-identifiable models that yield multiple equivalent solutions. We expand on this topic in a dedicated technical note for readers who want to explore the problem in greater depth.

[//]: # (Format "Contents" as a header level 2 without including it in the toctree, so that it doesn't render in the card)
++++

<h2 style="font-size: 2em; font-weight: bold; margin-top: 20px; margin-bottom: 10px;">Contents</h2>


```{toctree}
:maxdepth: 3

categorical_predictors.md
categorical_identifiability.md
```
