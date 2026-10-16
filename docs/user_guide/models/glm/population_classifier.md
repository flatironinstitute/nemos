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

[`ClassifierPopulationGLM`](nemos.glm.ClassifierPopulationGLM) predicts discrete labels for several outputs at once: one categorical variable for each neuron. It uses a softmax inverse link. It is to the [`ClassifierGLM`](nemos.glm.ClassifierGLM) what the [`PopulationGLM`](nemos.glm.PopulationGLM) is to the [`GLM`](nemos.glm.GLM). It is the same model, fit jointly over the neuron axis instead of in a loop. The difference is that axis: `classes`, the [class labels](classifier-labels) and the [identifiability of the parameters](classifier-identifiability) work as in the [classifier GLM](glm-classifier).

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

All the neurons share one set of classes. Declare them with `classes` at initialization, as an integer or as the array of labels. As for the [`PopulationGLM`](nemos.glm.PopulationGLM), a [`feature_mask`](neuron-specific-features-classifier) selects the predictors of each neuron.

```{code-cell} ipython3
import jax.numpy as jnp
import numpy as np

import nemos as nmo

n_neurons = 3
n_classes = 4
n_samples = 1000

nmo.glm.ClassifierPopulationGLM(classes=n_classes)
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

model_np = nmo.glm.ClassifierPopulationGLM(classes=n_classes).fit(X_np, labels_np)
```

#### JAX

`fit` gives the same parameters for JAX arrays and for NumPy arrays.

```{code-cell} ipython3
model_jax = nmo.glm.ClassifierPopulationGLM(classes=n_classes).fit(X, labels)

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

model_nap = nmo.glm.ClassifierPopulationGLM(classes=n_classes).fit(
    predictors, labels_nap
)
model_nap.predict(predictors)
```

### PyTrees

If `X` is a PyTree, each leaf of `coef_` has the shape of its predictors, plus an axis for the neurons and an axis for the classes.

```{code-cell} ipython3
X_dict = {"stimulus": X[:, :12], "coupling": X[:, 12:]}

model_dict = nmo.glm.ClassifierPopulationGLM(classes=n_classes).fit(X_dict, labels)

{key: coef.shape for key, coef in model_dict.coef_.items()}
```

## Methods

### Fit

```{code-cell} ipython3
model = nmo.glm.ClassifierPopulationGLM(classes=n_classes).fit(X, labels)
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

The class axis has the same over-parameterization as in the single neuron model: one linear predictor for each class, for probabilities that sum to one. Thus, `Ridge` is also the default here. See [identifiability](classifier-identifiability) for which parameters the penalty determines and which it leaves free.

#### NaN treatment

The fit ignores samples with a NaN in `X` or `y`. Examples of such samples are bins without a measurement, and the edges that a convolution leaves undefined. You do not need to remove these samples by hand.

```{code-cell} ipython3
X_nan = X.at[:10].set(np.nan)

model_nan = nmo.glm.ClassifierPopulationGLM(classes=n_classes).fit(X_nan, labels)
model_dropped = nmo.glm.ClassifierPopulationGLM(classes=n_classes).fit(
    X[10:], labels[10:]
)

np.allclose(model_nan.coef_, model_dropped.coef_)
```

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

### Simulate

[`simulate`](nemos.glm.ClassifierPopulationGLM.simulate) draws a label for each sample and each neuron from the predicted class distribution. It returns the labels and the log-probabilities they were drawn from.

```{code-cell} ipython3
import jax

labels_sim, log_proba_sim = model.simulate(jax.random.key(0), X)

print(f"labels:     {labels_sim.shape}")
print(f"log-probas: {log_proba_sim.shape}")
```

The random key makes the draw reproducible: the same key and the same probabilities give the same labels.

(population-classifier-out-of-memory)=
### Fitting out of memory

If the design matrix does not fit in memory, use [`stochastic_fit`](nemos.glm.ClassifierPopulationGLM.stochastic_fit). It takes a data loader instead of arrays, and updates the parameters one batch at a time. Each batch of `y` contains all the neurons. Stochastic solvers require acceleration and line searches to be off.

A single batch does not always contain every class, so declare the labels with [`set_classes`](nemos.glm.ClassifierPopulationGLM.set_classes) before you start.

```{code-cell} ipython3
# the loader serves the data in batches of 100 samples
loader = nmo.batching.ArrayDataLoader(X, labels, batch_size=100)

clf = nmo.glm.ClassifierPopulationGLM(
    classes=n_classes,
    solver_name="GradientDescent",
    solver_kwargs={"stepsize": 0.05, "acceleration": False},
).set_classes(labels)

clf.stochastic_fit(loader, n_passes=20)
```

See [stochastic optimization](../../scalability/README.md).

#### Update

For full control of the optimization loop, call [`update`](nemos.glm.ClassifierPopulationGLM.update) to do each step yourself. [`initialize_params`](nemos.glm.ClassifierPopulationGLM.initialize_params) returns the initial parameters. [`initialize_optimizer_and_state`](nemos.glm.ClassifierPopulationGLM.initialize_optimizer_and_state) sets up the solver.

```{code-cell} ipython3
params = clf.initialize_params(X, labels)
opt_state = clf.initialize_optimizer_and_state(params, X, labels)

# a full pass over all the batches
for X_batch, labels_batch in loader:
    params, opt_state = clf.update(params, opt_state, X_batch, labels_batch)
```

The loop body can contain any code. Examples are a log of the score for each batch, a change of the learning rate between passes, or your own stopping rule. See [custom update](../../scalability/custom_update.md).

(population-classifier-model-configuration)=
## Model configurations

### Observation models

The observation model is fixed to [`CategoricalObservations`](nemos.observation_models.CategoricalObservations), and it is not an argument of the constructor. Every neuron shares it.

### Inverse link function

The inverse link function maps the linear term to the log-probability of each label, and the default is `log_softmax`, which produces it directly. It is the only choice that keeps the log-likelihood convex.

### Regularizers

Set the regularizer and its strength when you create the model.

```{code-cell} ipython3
nmo.glm.ClassifierPopulationGLM(
    classes=n_classes, regularizer="Lasso", regularizer_strength=0.05
).fit(X, labels).coef_.shape
```

As for the [classifier GLM](glm-classifier), the default is `Ridge` with a strength of 1 rather than no regularization, because the class axis is over-parameterized. A parameter-wise `regularizer_strength` has the shape of the parameters, including the neuron and class axes.

[Regularizers](../../regularizers.md) describes the group-wise and parameter-wise forms, and the special cases of ElasticNet and GroupLasso.

### Solvers

Select the solver by name, and configure it with `solver_kwargs`. The whole population is fit in a single optimization.

```{code-cell} ipython3
print(nmo.glm.ClassifierPopulationGLM(classes=n_classes).solver_name)
print(
    nmo.glm.ClassifierPopulationGLM(
        classes=n_classes, regularizer="Lasso", regularizer_strength=0.01
    ).solver_name
)
```

[Solvers](../../solvers.md) describes which solver accepts which regularizer, which solver to use, and how to register your own solver.

## Parameter specification

### Drop the intercept

`fit_intercept=False` removes the intercept from the model. It stays at zero for every neuron and class, and uses no degree of freedom.

```{code-cell} ipython3
no_intercept = nmo.glm.ClassifierPopulationGLM(
    classes=n_classes, fit_intercept=False
).fit(X, labels)

print(f"intercept_: {no_intercept.intercept_.shape}")
print(f"dof_resid_: {no_intercept.dof_resid_}")
```

### Fix parameter values

`fix_params` keeps the intercept at a value that you choose. It has the shape of `intercept_`, one value for each neuron and class. Give `None` for the leaf to learn.

```{code-cell} ipython3
# hold every intercept at zero, learn the weights
model_fixed = nmo.glm.ClassifierPopulationGLM(
    classes=n_classes, fix_params=(None, np.zeros((n_neurons, n_classes)))
).fit(X, labels)

model_fixed.intercept_
```

To hold the weights of a group of predictors instead, use a feature mask.

(neuron-specific-features-classifier)=
### Neuron-specific features

`feature_mask` selects the predictors of each neuron, as in the [`PopulationGLM`](nemos.glm.PopulationGLM). There is one difference: the mask has no class axis. Its shape is `(n_features, n_neurons)`. A feature reaches a neuron for all its classes, or not at all. You cannot use the mask for one class of a neuron only.

```{code-cell} ipython3
# drop the first 8 features from the predictors of the first neuron
feature_mask = np.ones((X.shape[1], n_neurons))
feature_mask[:8, 0] = 0

masked = nmo.glm.ClassifierPopulationGLM(
    classes=n_classes, feature_mask=feature_mask
)
masked.fit(X, labels)

# zero for every class of the first neuron
masked.coef_[:8, 0]
```

## Examples

```{nemos-examples}
:model: ClassifierPopulationGLM
```
