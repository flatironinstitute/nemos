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

[`ClassifierGLM`](nemos.glm.ClassifierGLM) predicts a discrete label per sample through a softmax inverse link. It takes the arguments of the [single neuron GLM](glm-single-neuron) and is configured the same way, with [`CategoricalObservations`](nemos.observation_models.CategoricalObservations) and `log_softmax` set for you. What is specific to it is the number of classes, the handling of the labels, and the identifiability of the parameters.

## Fitting a Classifier GLM

The number of classes is declared at initialization. It is not inferred from the data.

```{code-cell} ipython3
import numpy as np

import nemos as nmo

n_classes = 3

# X of shape (n_samples, n_features), labels of shape (n_samples,)
X, labels = nmo.datasets.tuned_categorical(n_samples=1000, n_classes=n_classes)
labels = np.asarray(labels, dtype=int)

model = nmo.glm.ClassifierGLM(n_classes=n_classes).fit(X, labels)
model
```

`n_classes` must be an integer of at least 2, and it must agree with the labels: `fit` raises unless `y` holds exactly that many distinct values.

```{code-cell} ipython3
:tags: [raises-exception]

nmo.glm.ClassifierGLM(n_classes=5).fit(X, labels)
```

(classifier-labels)=
## Class labels

`y` can hold any labels that fit in a numpy array: integers, strings, or another hashable type. `classes_` lists them sorted, and the model maps them to the indices `0, ..., n_classes - 1` for the fit.

```{code-cell} ipython3
y_str = np.array(["left", "center", "right"])[labels]

model_str = nmo.glm.ClassifierGLM(n_classes=n_classes).fit(X, y_str)
model_str.classes_
```

The order is the sorted one, so `"center"` is class 0 whatever order the labels came in. Predictions are decoded back into the labels you passed.

```{code-cell} ipython3
model_str.predict(X)[:5]
```

Labels that are already the integers `0, ..., n_classes - 1` are the internal indices, and the model skips the translation for them. Any other set of labels pays for a translation on the way in and on the way out.

:::{admonition} Setting the classes when fitting in batches
:class: dropdown

[`fit`](nemos.glm.ClassifierGLM.fit) and [`initialize_optimizer_and_state`](nemos.glm.ClassifierGLM.initialize_optimizer_and_state) infer the classes from the `y` they are given. A single batch need not contain every class.

```{code-cell} ipython3
np.unique(labels[:10])
```

Declare the labels with [`set_classes`](nemos.glm.ClassifierGLM.set_classes) before stepping through batches with [`update`](nemos.glm.ClassifierGLM.update), and the missing ones are still accounted for.

```{code-cell} ipython3
batched = nmo.glm.ClassifierGLM(n_classes=n_classes).set_classes(labels)

params = batched.initialize_params(X[:10], labels[:10])
state = batched.initialize_optimizer_and_state(params, X[:10], labels[:10])

batched.classes_
```
:::

## Shape of the parameters

The model estimates one set of coefficients per class, so the class axis is added last.

```{code-cell} ipython3
print(f"coef_:      {model.coef_.shape}")
print(f"intercept_: {model.intercept_.shape}")
```

`coef_[:, k]` are the weights of class `classes_[k]`, and `intercept_[k]` its baseline.

(classifier-identifiability)=
## Identifiability

The model fits one linear predictor per class, $n$ of them, for $n$ probabilities constrained to sum to one. That is one more than strictly needed: once the first $n - 1$ probabilities are known, the last one is

$$
p_n = 1 - \sum_{k=1}^{n-1} p_k .
$$

The redundancy is an invariance of the softmax, which is unchanged when the same constant is added to the linear predictor of every class: if `(coef, intercept)` is a solution, so is `(coef + c, intercept + c)`. Without regularization the model has infinitely many solutions, all giving the same probabilities, and the one the solver returns depends on where it started. `Ridge` is the default for that reason.

Ridge does not remove the invariance entirely. The penalty reaches the coefficients and leaves the intercept alone, so `coef_` comes out pinned and centered, summing to zero across classes, while `intercept_` is still free to slide: fitting from a starting point whose intercept is shifted by a constant returns that same shift in `intercept_`, with the coefficients and the probabilities unchanged.

```{code-cell} ipython3
model.coef_.sum(axis=-1)[:4]
```

Carrying the intercept as a constant column of the design, with `fit_intercept=False`, brings it under the penalty and leaves nothing unpinned.

```{code-cell} ipython3
X_with_intercept = np.concatenate([np.ones((X.shape[0], 1)), np.asarray(X)], axis=1)

folded = nmo.glm.ClassifierGLM(n_classes=n_classes, fit_intercept=False).fit(
    X_with_intercept, labels
)

# the weights of the constant column play the role of the intercept
folded.coef_[0]
```

The model expresses exactly what it did before. The objective is the one that changes, since the intercept is now shrunk along with the rest, and in exchange the problem is strictly convex and its solution unique.

## Predicting labels and probabilities

[`predict`](nemos.glm.ClassifierGLM.predict) takes the arg-max over the class axis and decodes it into a label, consuming the axis.

```{code-cell} ipython3
model.predict(X)[:5]
```

[`predict_proba`](nemos.glm.ClassifierGLM.predict_proba) keeps the class axis and returns the whole distribution. It returns log-probabilities by default, matching the `log_softmax` inverse link; pass `return_type="proba"` for probabilities.

```{code-cell} ipython3
proba = model.predict_proba(X, return_type="proba")

print(f"proba shape: {proba.shape}")
proba[:3]
```

With pynapple predictors, the labels come back as a `Tsd`, and both probabilities and log-probabilities as a `TsdFrame` with one column per class.

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

## Scoring

[`score`](nemos.glm.ClassifierGLM.score) returns the mean log-likelihood per sample, and `score_type` switches to one of the two pseudo-$R^2$, as for the [single neuron GLM](glm-single-neuron).

```{code-cell} ipython3
model.score(X, labels)
```

None of the three is the accuracy. They score the probabilities the model assigns to the observed labels, not the fraction of labels its arg-max gets right.

## Examples

```{nemos-examples}
:model: ClassifierGLM
```
