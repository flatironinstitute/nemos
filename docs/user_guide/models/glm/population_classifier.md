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

# Population Classifier GLM


[`ClassifierPopulationGLM`](nemos.glm.ClassifierPopulationGLM) predicts discrete labels for several outputs at once, one categorical variable per neuron, through a softmax inverse link. It stands to the [`ClassifierGLM`](nemos.glm.ClassifierGLM) as the [`PopulationGLM`](nemos.glm.PopulationGLM) stands to the [`GLM`](nemos.glm.GLM): the same model, fit jointly over the neuron axis instead of in a loop. This page covers the neuron axis; `n_classes`, the [class labels](classifier-labels) and the [identifiability of the parameters](classifier-identifiability) work as in the [classifier GLM](glm-classifier).

## Fitting a Population Classifier GLM

The `y` passed to [`fit`](nemos.glm.ClassifierPopulationGLM.fit) is a two-dimensional array of shape `(n_samples, n_neurons)`, each column holding the labels of one neuron. All the columns share one set of classes, declared at initialization with `n_classes`.

```{code-cell} ipython3
import jax.numpy as jnp
import numpy as np

import nemos as nmo

n_neurons = 3
n_classes = 4
n_samples = 1000

# generate data:
# - X array of shape (n_samples, n_features)
# - labels array of shape (n_samples, n_neurons)
X, labels = nmo.datasets.tuned_categorical(
    n_samples=n_samples, n_neurons=n_neurons, n_classes=n_classes
)
print(f"X shape:      {X.shape}")
print(f"labels shape: {labels.shape}")

# define and fit the model
model = nmo.glm.ClassifierPopulationGLM(n_classes=n_classes)
model.fit(X, labels)
```

## Shape of the parameters

The model estimates one set of coefficients per class, so the class axis is added last to the parameters.

```{code-cell} ipython3
print(f"coef_ shape:      {model.coef_.shape}")
print(f"intercept_ shape: {model.intercept_.shape}")
```

`coef_` is `(n_features, n_neurons, n_classes)` and `intercept_` is `(n_neurons, n_classes)`, so `coef_[:, j, k]` are the weights that neuron `j` assigns to class `k`. Dropping the class axis from both recovers the shapes of the [`PopulationGLM`](nemos.glm.PopulationGLM), and dropping the neuron axis as well recovers those of the single neuron [`ClassifierGLM`](nemos.glm.ClassifierGLM), `(n_features, n_classes)` and `(n_classes,)`.

The class axis carries the same over-parameterization as in the single neuron model, one linear predictor per class for probabilities that sum to one, so `Ridge` is the default here too. See [identifiability](classifier-identifiability), including what the penalty does and does not pin down.

## Predicting labels and probabilities

[`predict`](nemos.glm.ClassifierPopulationGLM.predict) consumes the class axis with its arg-max, leaving one label per sample and neuron.

```{code-cell} ipython3
predicted = model.predict(X)
print(f"predicted shape: {predicted.shape}")
predicted[:5]
```

[`predict_proba`](nemos.glm.ClassifierPopulationGLM.predict_proba) keeps the class axis, so the distribution of every neuron comes back on its own axis.

```{code-cell} ipython3
proba = model.predict_proba(X, return_type="proba")
print(f"proba shape: {proba.shape}")

# the distribution of each neuron at the first sample
proba[0]
```

(neuron-specific-features-classifier)=
## Neuron-specific features

The `feature_mask` selects the predictors of each neuron, as in the [`PopulationGLM`](nemos.glm.PopulationGLM), with one difference: it carries no class axis. Its shape is `(n_features, n_neurons)`, because a feature either reaches a neuron or it does not, and it cannot be used for one class of a neuron alone.

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

## Scoring

[`score`](nemos.glm.ClassifierPopulationGLM.score) returns the mean log-likelihood over samples and neurons, a single number, which is what scikit-learn cross-validation expects from an estimator.

```{code-cell} ipython3
model.score(X, labels)
```

As for the [`PopulationGLM`](nemos.glm.PopulationGLM), averaging over the sample axis alone leaves one score per neuron.

```{code-cell} ipython3
model.score(X, labels, aggregate_sample_scores=lambda x: jnp.mean(x, axis=0))
```

## Examples

```{nemos-examples}
:model: ClassifierPopulationGLM
```
