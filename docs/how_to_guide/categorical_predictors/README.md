# Categorical Predictors

Capturing the effect of stimulus identity, behavioral choices, etc., are all common examples of model designs requiring the encoding of categorical predictors. In this section we will explore how to construct such designs with NeMoS `Category` basis.

We [also cover specialized NeMoS-compatible packages](complex-designs) which elegantly handle complex schemes involving multiple categorical predictors and diverse encoding methods. These packages transparently resolve a well-known identifiability issue: designs with multiple categorical predictors can result in non-identifiable models that yield multiple equivalent solutions.

We expand on the issue of identifiability in a dedicated [technical note](categorical_identifiability) for readers who want to explore the problem in greater depth.

[//]: # (Format "Contents" as a header level 2 without including it in the toctree, so that it doesn't render in the card inro docs/how_to_guide/README.md)
++++

<h2 style="font-size: 2em; font-weight: bold; margin-top: 20px; margin-bottom: 10px;">Contents</h2>


```{toctree}
:maxdepth: 3

categorical_predictors.md
categorical_identifiability.md
```
