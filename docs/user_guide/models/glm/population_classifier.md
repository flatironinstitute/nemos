---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(glm-population-classifier)=
# Population Classifier GLM

[`ClassifierPopulationGLM`](nemos.glm.ClassifierPopulationGLM) predicts discrete labels for several outputs at once: one categorical variable for each neuron. It uses a softmax inverse link. It is to the [`ClassifierGLM`](nemos.glm.ClassifierGLM) what the [`PopulationGLM`](nemos.glm.PopulationGLM) is to the [`GLM`](nemos.glm.GLM). It is the same model, fit jointly over the neuron axis instead of in a loop. This page covers only the neuron axis. `n_classes`, the [class labels](classifier-labels) and the [identifiability of the parameters](classifier-identifiability) work as in the [classifier GLM](glm-classifier).

```{image} ../../../assets/population_classifier_glm.svg
:alt: ClassifierPopulationGLM. X times a weight tensor W with one group of class columns per neuron, then a softmax gives the probabilities P of shape (n_samples, n_neurons, n_classes), and the argmax gives one label per sample and neuron.
:class: only-light
```

```{image} ../../../assets/population_classifier_glm_dark.svg
:alt: ClassifierPopulationGLM. X times a weight tensor W with one group of class columns per neuron, then a softmax gives the probabilities P of shape (n_samples, n_neurons, n_classes), and the argmax gives one label per sample and neuron.
:class: only-dark
```

```{contents}
:local:
:depth: 3
```

## Initializing a ClassifierPopulationGLM object

All the neurons share one set of classes. Set the number of classes with `n_classes` at initialization. As for the [`PopulationGLM`](nemos.glm.PopulationGLM), a [`feature_mask`](neuron-specific-features-classifier) selects the predictors of each neuron.

```{code-cell} ipython3
import jax.numpy as jnp
import numpy as np

import nemos as nmo

n_neurons = 3
n_classes = 4
n_samples = 1000

nmo.glm.ClassifierPopulationGLM(n_classes=n_classes)
```

## Inputs

`X` is the same as for the [single neuron GLM](glm-input-types). `y` contains the labels, with one column for each neuron.

### Arrays

`X` has shape `(n_samples, n_features)` and `y` has shape `(n_samples, n_neurons)`. Each of them can be a NumPy array, a JAX array, or a [pynapple](https://www.pynapple.org) time series.

#### NumPy

```{code-cell} ipython3
# X of shape (n_samples, n_features), labels of shape (n_samples, n_neurons)
X, labels = nmo.datasets.tuned_categorical(
    n_samples=n_samples, n_neurons=n_neurons, n_classes=n_classes
)  # JAX arrays
X_np, labels_np = np.asarray(X), np.asarray(labels, dtype=int)

print(f"X shape:      {X.shape}")
print(f"labels shape: {labels.shape}")

model_np = nmo.glm.ClassifierPopulationGLM(n_classes=n_classes).fit(X_np, labels_np)
```

#### JAX

`fit` gives the same parameters for JAX arrays and for NumPy arrays.

```{code-cell} ipython3
model_jax = nmo.glm.ClassifierPopulationGLM(n_classes=n_classes).fit(X, labels)

np.allclose(model_jax.coef_, model_np.coef_)
```

#### Pynapple

With [pynapple](https://www.pynapple.org), `y` is a `TsdFrame` with one column for each neuron. With [pynapple](https://www.pynapple.org) predictors, `predict` returns a `TsdFrame` of labels with the column names of `y`. `predict_proba` returns a `TsdTensor` with shape `(n_samples, n_neurons, n_classes)`.

```{code-cell} ipython3
import pynapple as nap

# suppress jax to numpy conversion warning
nap.nap_config.suppress_conversion_warnings = True

# 10 ms bins
time = np.arange(n_samples) / 100

predictors = nap.TsdFrame(t=time, d=X_np)
labels_nap = nap.TsdFrame(t=time, d=labels_np, columns=["unit0", "unit1", "unit2"])

model_nap = nmo.glm.ClassifierPopulationGLM(n_classes=n_classes).fit(
    predictors, labels_nap
)
model_nap.predict(predictors)
```

### PyTrees

If `X` is a PyTree, each leaf of `coef_` has the shape of its predictors, plus an axis for the neurons and an axis for the classes.

```{code-cell} ipython3
X_dict = {"stimulus": X[:, :12], "coupling": X[:, 12:]}

model_dict = nmo.glm.ClassifierPopulationGLM(n_classes=n_classes).fit(X_dict, labels)

{key: coef.shape for key, coef in model_dict.coef_.items()}
```

## Methods

### Fit

```{code-cell} ipython3
model = nmo.glm.ClassifierPopulationGLM(n_classes=n_classes).fit(X, labels)
model
```

#### Fitted attributes

The model estimates one set of coefficients for each class. Thus, the parameters have an additional last axis for the classes.

```{code-cell} ipython3
print(f"coef_ shape:      {model.coef_.shape}")
print(f"intercept_ shape: {model.intercept_.shape}")
```

- `coef_` has shape `(n_features, n_neurons, n_classes)`. `coef_[:, j, k]` holds the weights of neuron `j` for class `k`.
- `intercept_` has shape `(n_neurons, n_classes)`.

Without the class axis, these are the shapes of the [`PopulationGLM`](nemos.glm.PopulationGLM). Without the neuron axis too, they are the shapes of the single neuron [`ClassifierGLM`](nemos.glm.ClassifierGLM): `(n_features, n_classes)` and `(n_classes,)`.

The class axis has the same over-parameterization as in the single neuron model: one linear predictor for each class, for probabilities that sum to one. Thus, `Ridge` is also the default here. See [identifiability](classifier-identifiability) for what the penalty pins down and what it does not.

### Predict

[`predict`](nemos.glm.ClassifierPopulationGLM.predict) takes the arg-max over the class axis. The output has one label for each sample and neuron.

```{code-cell} ipython3
predicted = model.predict(X)
print(f"predicted shape: {predicted.shape}")
predicted[:5]
```

[`predict_proba`](nemos.glm.ClassifierPopulationGLM.predict_proba) keeps the class axis. The output has the distribution of every neuron along this axis.

```{code-cell} ipython3
proba = model.predict_proba(X, return_type="proba")
print(f"proba shape: {proba.shape}")

# the distribution of each neuron at the first sample
proba[0]
```

### Score

[`score`](nemos.glm.ClassifierPopulationGLM.score) returns the mean log-likelihood over samples and neurons, as a single number. scikit-learn cross-validation expects this from an estimator.

```{code-cell} ipython3
model.score(X, labels)
```

As for the [`PopulationGLM`](nemos.glm.PopulationGLM), an average over the sample axis only gives one score per neuron.

```{code-cell} ipython3
model.score(X, labels, aggregate_sample_scores=lambda x: jnp.mean(x, axis=0))
```

## Parameter specification

(neuron-specific-features-classifier)=
### Neuron-specific features

`feature_mask` selects the predictors of each neuron, as in the [`PopulationGLM`](nemos.glm.PopulationGLM). There is one difference: the mask has no class axis. Its shape is `(n_features, n_neurons)`. A feature reaches a neuron for all its classes, or not at all. You cannot use the mask for one class of a neuron only.

```{code-cell} ipython3
# drop the first 8 features from the predictors of the first neuron
feature_mask = np.ones((X.shape[1], n_neurons))
feature_mask[:8, 0] = 0

masked = nmo.glm.ClassifierPopulationGLM(
    n_classes=n_classes, feature_mask=feature_mask
)
masked.fit(X, labels)

# zero for every class of the first neuron
masked.coef_[:8, 0]
```

## Examples

```{nemos-examples}
:model: ClassifierPopulationGLM
```
