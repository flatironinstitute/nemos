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

(glm-classifier)=
# Classifier GLM

[`ClassifierGLM`](nemos.glm.ClassifierGLM) predicts a discrete label for each sample. It uses a softmax inverse link. It takes the arguments of the [single neuron GLM](glm-single-neuron), and you configure it the same way. The observation model is [`CategoricalObservations`](nemos.observation_models.CategoricalObservations) and the inverse link is `log_softmax`. NeMoS sets both for you. This page covers only the differences: the number of classes, the class labels, and the identifiability of the parameters.

```{image} ../../../assets/classifier_glm.svg
:alt: ClassifierGLM. It maps X through a weight matrix W, one column per class, then a softmax gives the class probabilities P, and the argmax gives one label per sample.
:class: only-light
```

```{image} ../../../assets/classifier_glm_dark.svg
:alt: ClassifierGLM. It maps X through a weight matrix W, one column per class, then a softmax gives the class probabilities P, and the argmax gives one label per sample.
:class: only-dark
```

```{contents}
:local:
:depth: 3
```

## Initializing a ClassifierGLM object

Set the number of classes at initialization. The model does not infer it from the data. `n_classes` must be an integer of at least 2.

```{code-cell} ipython3
import numpy as np

import nemos as nmo

n_classes = 3

nmo.glm.ClassifierGLM(n_classes=n_classes)
```

(glm-classifier-input-types)=
## Inputs

`X` is the same as for the [single neuron GLM](glm-input-types). `y` contains one label for each sample.

### Arrays

`X` has shape `(n_samples, n_features)` and `y` has shape `(n_samples,)`. Each of them can be a NumPy array, a JAX array, or a [pynapple](https://www.pynapple.org) time series.

#### NumPy

```{code-cell} ipython3
# X of shape (n_samples, n_features), labels of shape (n_samples,)
X, labels = nmo.datasets.tuned_categorical(n_samples=1000, n_classes=n_classes)
X_np = np.asarray(X)
labels = np.asarray(labels, dtype=int)

model_np = nmo.glm.ClassifierGLM(n_classes=n_classes).fit(X_np, labels)
```

#### JAX

`fit` gives the same parameters for JAX arrays and for NumPy arrays.

```{code-cell} ipython3
import jax.numpy as jnp

model_jax = nmo.glm.ClassifierGLM(n_classes=n_classes).fit(X, jnp.asarray(labels))

np.allclose(model_jax.coef_, model_np.coef_)
```

#### Pynapple

With [pynapple](https://www.pynapple.org), `y` is a `Tsd` of labels. With [pynapple](https://www.pynapple.org) predictors, `predict` returns the labels as a `Tsd`. `predict_proba` returns the probabilities or the log-probabilities as a `TsdFrame` with one column per class.

```{code-cell} ipython3
import pynapple as nap

# suppress jax to numpy conversion warning
nap.nap_config.suppress_conversion_warnings = True

# 10 ms bins
time = np.arange(X.shape[0]) / 100

predictors = nap.TsdFrame(t=time, d=np.asarray(X))

model_nap = nmo.glm.ClassifierGLM(n_classes=n_classes).fit(
    predictors, nap.Tsd(t=time, d=labels)
)
model_nap.predict_proba(predictors, return_type="proba")
```

### PyTrees

`X` can be a PyTree, as for the [single neuron GLM](glm-input-types). Each leaf of `coef_` has the shape of its predictors, plus a last axis for the classes.

```{code-cell} ipython3
X_dict = {"stimulus": X[:, :12], "history": X[:, 12:]}

model_dict = nmo.glm.ClassifierGLM(n_classes=n_classes).fit(X_dict, labels)

{key: coef.shape for key, coef in model_dict.coef_.items()}
```

(classifier-labels)=
### Class labels

`y` can contain any labels that fit in a NumPy array: integers, strings, or another hashable type. `classes_` lists the labels in sorted order. For the fit, the model maps the labels to the indices `0, ..., n_classes - 1`.

```{code-cell} ipython3
y_str = np.array(["left", "center", "right"])[labels]

model_str = nmo.glm.ClassifierGLM(n_classes=n_classes).fit(X, y_str)
model_str.classes_
```

The order is the sorted order. Thus, `"center"` is class 0, whatever the order of the labels in `y`. `predict` returns the labels that you gave to `fit`.

```{code-cell} ipython3
model_str.predict(X)[:5]
```

If the labels are already the integers `0, ..., n_classes - 1`, they are the internal indices, and the model does not translate them. Any other set of labels needs a translation at the input and at the output.

`n_classes` must agree with the labels. `fit` raises an error if `y` does not contain exactly `n_classes` distinct values.

```{code-cell} ipython3
:tags: [raises-exception]

nmo.glm.ClassifierGLM(n_classes=5).fit(X, labels)
```

:::{admonition} Setting the classes when fitting in batches
:class: dropdown

[`fit`](nemos.glm.ClassifierGLM.fit) and [`initialize_optimizer_and_state`](nemos.glm.ClassifierGLM.initialize_optimizer_and_state) infer the classes from the `y` that they receive. A single batch does not always contain every class.

```{code-cell} ipython3
np.unique(labels[:10])
```

Declare the labels with [`set_classes`](nemos.glm.ClassifierGLM.set_classes) before you step through the batches with [`update`](nemos.glm.ClassifierGLM.update). The model then includes the classes that are missing from a batch.

```{code-cell} ipython3
batched = nmo.glm.ClassifierGLM(n_classes=n_classes).set_classes(labels)

params = batched.initialize_params(X[:10], labels[:10])
state = batched.initialize_optimizer_and_state(params, X[:10], labels[:10])

batched.classes_
```
:::

## Methods

### Fit

```{code-cell} ipython3
model = nmo.glm.ClassifierGLM(n_classes=n_classes).fit(X, labels)
model
```

#### Fitted attributes

The model estimates one set of coefficients for each class. Thus, the parameters have an additional last axis for the classes.

```{code-cell} ipython3
print(f"coef_:      {model.coef_.shape}")
print(f"intercept_: {model.intercept_.shape}")
```

`coef_[:, k]` holds the weights of class `classes_[k]`. `intercept_[k]` holds its baseline.

### Predict

[`predict`](nemos.glm.ClassifierGLM.predict) takes the arg-max over the class axis and returns the corresponding label. The output has no class axis.

```{code-cell} ipython3
model.predict(X)[:5]
```

[`predict_proba`](nemos.glm.ClassifierGLM.predict_proba) keeps the class axis and returns the full distribution. By default, it returns log-probabilities, which matches the `log_softmax` inverse link. For probabilities, set `return_type="proba"`.

```{code-cell} ipython3
proba = model.predict_proba(X, return_type="proba")

print(f"proba shape: {proba.shape}")
proba[:3]
```

### Score

[`score`](nemos.glm.ClassifierGLM.score) returns the mean log-likelihood per sample. As for the [single neuron GLM](glm-single-neuron), `score_type` selects one of the two pseudo-$R^2$ scores instead.

```{code-cell} ipython3
model.score(X, labels)
```

None of these three scores is the accuracy. They score the probabilities that the model gives to the observed labels. They do not count the fraction of labels that the arg-max predicts correctly.

(classifier-identifiability)=
## Identifiability

The model fits one linear predictor for each class: $n$ predictors for $n$ probabilities that must sum to one. This is one predictor more than necessary. When the first $n - 1$ probabilities are known, the last one is

$$
p_n = 1 - \sum_{k=1}^{n-1} p_k .
$$

This redundancy comes from an invariance of the softmax. The softmax does not change when you add the same constant to the linear predictor of every class. If `(coef, intercept)` is a solution, `(coef + c, intercept + c)` is also a solution. Without regularization, the model has infinitely many solutions, and all of them give the same probabilities. The solution that the solver returns depends on its starting point. For this reason, `Ridge` is the default.

Ridge does not remove the invariance entirely. The penalty applies to the coefficients but not to the intercept. Thus, `coef_` is pinned and centered: it sums to zero across classes. But `intercept_` can still shift. If the starting intercept is shifted by a constant, the fit returns the same shift in `intercept_`. The coefficients and the probabilities do not change.

```{code-cell} ipython3
model.coef_.sum(axis=-1)[:4]
```

To penalize the intercept too, add it as a constant column of the design matrix and set `fit_intercept=False`. Then no parameter is free to shift.

```{code-cell} ipython3
X_with_intercept = np.concatenate([np.ones((X.shape[0], 1)), np.asarray(X)], axis=1)

folded = nmo.glm.ClassifierGLM(n_classes=n_classes, fit_intercept=False).fit(
    X_with_intercept, labels
)

# the weights of the constant column play the role of the intercept
folded.coef_[0]
```

The model is the same as before. Only the objective changes, because the penalty now shrinks the intercept with the other weights. In exchange, the problem is strictly convex and its solution is unique.

## Examples

```{nemos-examples}
:model: ClassifierGLM
```
