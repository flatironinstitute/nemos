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

Declare the classes at initialization with `classes`. It takes either an integer of at least 2, which is the number of classes, or the array of the labels themselves.

```{code-cell} ipython3
import numpy as np

import nemos as nmo

n_classes = 3

nmo.glm.ClassifierGLM(classes=n_classes)
```

The two forms differ in what `fit` then checks. With an integer, the model infers the labels from `y`, and only their number is fixed. With an array, the labels are fixed as well, and `fit` accepts no label outside that set. [Class labels](#class-labels) covers both cases.

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

model_np = nmo.glm.ClassifierGLM(classes=n_classes).fit(X_np, labels)
```

#### JAX

`fit` gives the same parameters for JAX arrays and for NumPy arrays.

```{code-cell} ipython3
import jax.numpy as jnp

model_jax = nmo.glm.ClassifierGLM(classes=n_classes).fit(X, jnp.asarray(labels))

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

model_nap = nmo.glm.ClassifierGLM(classes=n_classes).fit(
    predictors, nap.Tsd(t=time, d=labels)
)
model_nap.predict_proba(predictors, return_type="proba")
```

### PyTrees

`X` can be a PyTree of arrays of any of the supported types, as for the [single neuron GLM](glm-input-types). Each leaf of `coef_` has the shape of its predictors, plus a last axis for the classes.

```{code-cell} ipython3
X_dict = {"stimulus": X[:, :12], "history": X[:, 12:]}

model_dict = nmo.glm.ClassifierGLM(classes=n_classes).fit(X_dict, labels)

{key: coef.shape for key, coef in model_dict.coef_.items()}
```

(classifier-labels)=
### Class labels

`y` can contain any labels that fit in a NumPy array: integers, strings, or another hashable type. `classes_` lists the labels in sorted order. For the fit, the model maps the labels to the indices `0, ..., n_classes - 1`.

```{code-cell} ipython3
y_str = np.array(["left", "center", "right"])[labels]

model_str = nmo.glm.ClassifierGLM(classes=n_classes).fit(X, y_str)
model_str.classes_
```

The order is the sorted order. Thus, `"center"` is class 0, whatever the order of the labels in `y`. `predict` returns the labels that you gave to `fit`.

```{code-cell} ipython3
model_str.predict(X)[:5]
```

If the labels are already the integers `0, ..., n_classes - 1`, they are the internal indices, and the model does not translate them. Any other set of labels needs a translation at the input and at the output.

#### Classes given as a number

With `classes` set to an integer, the model infers the labels from `y`, and only their number is fixed. `fit` raises if `y` does not hold exactly that many distinct values.

```{code-cell} ipython3
:tags: [raises-exception]

nmo.glm.ClassifierGLM(classes=5).fit(X, labels)
```

Since only the count is fixed, a later `fit` can use different labels, as long as there are still `classes` of them. The model takes the new labels as its classes.

```{code-cell} ipython3
model_inferred = nmo.glm.ClassifierGLM(classes=n_classes).fit(X, labels)
print(f"first fit:  {model_inferred.classes_}")

model_inferred.fit(X, labels + 1)
print(f"second fit: {model_inferred.classes_}")
```

#### Classes given as labels

Pass the labels themselves, at initialization or with [`set_classes`](nemos.glm.ClassifierGLM.set_classes), and they become the reference. `y` may then leave some of them out: a class that no sample shows still gets its own set of coefficients.

```{code-cell} ipython3
declared = nmo.glm.ClassifierGLM(classes=["left", "center", "right"])

# no "right" in this recording
y_subset = np.array(["left", "center"])[labels % 2]

declared.fit(X, y_subset)
declared.classes_
```

A label outside the declared set raises instead.

```{code-cell} ipython3
:tags: [raises-exception]

declared.fit(X, np.array(["left", "up"])[labels % 2])
```

This is what you want when a session does not contain every condition, and when several models have to share one set of classes to be comparable.

:::{admonition} Setting the classes when fitting in batches
:class: dropdown

[`fit`](nemos.glm.ClassifierGLM.fit) and [`initialize_optimizer_and_state`](nemos.glm.ClassifierGLM.initialize_optimizer_and_state) infer the classes from the `y` that they receive. A single batch does not always contain every class.

```{code-cell} ipython3
np.unique(labels[:10])
```

Declare the labels with [`set_classes`](nemos.glm.ClassifierGLM.set_classes) before you step through the batches with [`update`](nemos.glm.ClassifierGLM.update). The model then includes the classes that are missing from a batch.

```{code-cell} ipython3
batched = nmo.glm.ClassifierGLM(classes=n_classes).set_classes(labels)

params = batched.initialize_params(X[:10], labels[:10])
state = batched.initialize_optimizer_and_state(params, X[:10], labels[:10])

batched.classes_
```
:::

## Methods

### Fit

```{code-cell} ipython3
model = nmo.glm.ClassifierGLM(classes=n_classes).fit(X, labels)
model
```

#### Fitted attributes

The model estimates one set of coefficients for each class. Thus, the parameters have an additional last axis for the classes.

```{code-cell} ipython3
print(f"coef_:      {model.coef_.shape}")
print(f"intercept_: {model.intercept_.shape}")
```

`coef_[:, k]` holds the weights of class `classes_[k]`. `intercept_[k]` holds its baseline.

#### NaN treatment

The fit ignores samples with a NaN in `X` or `y`. Examples of such samples are bins without a measurement, and the edges that a convolution leaves undefined. You do not need to remove these samples by hand.

```{code-cell} ipython3
X_nan = X.at[:10].set(np.nan)

model_nan = nmo.glm.ClassifierGLM(classes=n_classes).fit(X_nan, labels)
model_dropped = nmo.glm.ClassifierGLM(classes=n_classes).fit(X[10:], labels[10:])

np.allclose(model_nan.coef_, model_dropped.coef_)
```

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

### Simulate

[`simulate`](nemos.glm.ClassifierGLM.simulate) draws a label for each sample from the predicted class distribution. It returns the labels and the log-probabilities they were drawn from.

```{code-cell} ipython3
import jax

labels_sim, log_proba_sim = model.simulate(jax.random.key(0), X)

print(f"labels:     {labels_sim[:5]}")
print(f"log-probas: {np.round(log_proba_sim[:2], 2)}")
```

The random key makes the draw reproducible: the same key and the same probabilities give the same labels. [`predict`](nemos.glm.ClassifierGLM.predict) returns the most probable class, while `simulate` samples from the distribution, so a less probable class can come out.

(classifier-out-of-memory)=
### Fitting out of memory

If the design matrix does not fit in memory, use [`stochastic_fit`](nemos.glm.ClassifierGLM.stochastic_fit). It takes a data loader instead of arrays, and updates the parameters one batch at a time. Stochastic solvers require acceleration and line searches to be off.

:::{admonition} In-memory example
:class: note

The arrays here fit in memory, so the loader is an [`ArrayDataLoader`](nemos.batching.ArrayDataLoader). For data that does not fit in memory, [`LazyArrayDataLoader`](nemos.batching.LazyArrayDataLoader) loads one batch at a time from a dask, zarr or HDF5 array.
:::

A single batch does not always contain every class, so declare the labels with [`set_classes`](nemos.glm.ClassifierGLM.set_classes) before you start.

```{code-cell} ipython3
# the loader serves the data in batches of 100 samples
loader = nmo.batching.ArrayDataLoader(X, labels, batch_size=100)

clf = nmo.glm.ClassifierGLM(
    classes=n_classes,
    solver_name="GradientDescent",
    solver_kwargs={"stepsize": 0.05, "acceleration": False},
).set_classes(labels)

clf.stochastic_fit(loader, n_passes=20)
```

See [stochastic optimization](../../scalability/README.md).

#### Update

For full control of the optimization loop, call [`update`](nemos.glm.ClassifierGLM.update) to do each step yourself. [`initialize_params`](nemos.glm.ClassifierGLM.initialize_params) returns the initial parameters. [`initialize_optimizer_and_state`](nemos.glm.ClassifierGLM.initialize_optimizer_and_state) sets up the solver.

```{code-cell} ipython3
params = clf.initialize_params(X, labels)
opt_state = clf.initialize_optimizer_and_state(params, X, labels)

# a full pass over all the batches
for X_batch, labels_batch in loader:
    params, opt_state = clf.update(params, opt_state, X_batch, labels_batch)
```

The loop body can contain any code. Examples are a log of the score for each batch, a change of the learning rate between passes, or your own stopping rule. See [custom update](../../scalability/custom_update.md).

(classifier-model-configuration)=
## Model configurations

### Observation models

The observation model is fixed to [`CategoricalObservations`](nemos.observation_models.CategoricalObservations), and it is not an argument of the constructor. A classifier models the probability of each label, so no other family applies.

[Observation models](../../observation_models.md) describes the family and what it computes.

### Inverse link function

The inverse link function maps the linear term to the parameter of the observation model. For the Categorical this parameter is the log-probability of each label, and the default is `log_softmax`, which produces it directly.

```{code-cell} ipython3
nmo.glm.ClassifierGLM(classes=n_classes).inverse_link_function
```

`log_softmax` is the only choice that keeps the log-likelihood convex. Another function still fits, but it converges to a local optimum that depends on the starting point.

### Regularizers

Set the regularizer and its strength when you create the model.

```{code-cell} ipython3
nmo.glm.ClassifierGLM(
    classes=n_classes, regularizer="Lasso", regularizer_strength=0.05
).fit(X, labels).coef_.shape
```

Unlike the [single neuron GLM](glm-regularizers), the default here is `Ridge` with a strength of 1, rather than no regularization. The model carries one set of coefficients per class, which is one set more than the probabilities need, and the penalty is what makes those coefficients unique. [Identifiability](classifier-identifiability) explains the redundancy.

```{code-cell} ipython3
print(nmo.glm.ClassifierGLM(classes=n_classes).regularizer)
print(nmo.glm.ClassifierGLM(classes=n_classes).regularizer_strength)
```

[Regularizers](../../regularizers.md) describes group-wise and parameter-wise strengths, and the special cases of ElasticNet and GroupLasso.

### Solvers

Select the solver by name, and configure it with `solver_kwargs`.

```{code-cell} ipython3
nmo.glm.ClassifierGLM(
    classes=n_classes, solver_name="LBFGS", solver_kwargs={"tol": 1e-10}
).fit(X, labels)
```

The keys of `solver_kwargs` must be arguments of the solver. Other keys raise an error. To see the accepted arguments, print the solver docstring with [`get_solver_documentation`](nemos.solvers.get_solver_documentation).

If you do not set the solver, NeMoS selects one that is compatible with the regularizer. The default `Ridge` gives `Newton`.

```{code-cell} ipython3
print(nmo.glm.ClassifierGLM(classes=n_classes).solver_name)
print(nmo.glm.ClassifierGLM(classes=n_classes, regularizer="Lasso", regularizer_strength=0.01).solver_name)
```

[Solvers](../../solvers.md) describes which solver accepts which regularizer, which solver to use, and how to register your own solver.

(classifier-parameter-specification)=
## Parameter specification

### Drop the intercept

`fit_intercept=False` removes the intercept from the model. It stays at zero for every class and uses no degree of freedom.

```{code-cell} ipython3
no_intercept = nmo.glm.ClassifierGLM(classes=n_classes, fit_intercept=False).fit(X, labels)

print(f"intercept_: {no_intercept.intercept_}")
print(f"dof_resid_: {no_intercept.dof_resid_}")
```

### Fix parameter values

`fix_params` keeps any leaf of the parameters at a value that you choose. The parameters are a `(coef, intercept)` tuple. If `X` is a PyTree, `coef` is also a PyTree. Thus, you can fix these leaves:

- the intercept,
- each leaf of `coef`.

For a leaf to fix, give its array of values, with the class axis last. For a leaf to learn, give `None` as a placeholder.

#### Fix a value for the intercept

```{code-cell} ipython3
# hold one intercept per class, learn the weights
model_fixed = nmo.glm.ClassifierGLM(
    classes=n_classes, fix_params=(None, np.zeros(n_classes))
).fit(X, labels)

print(f"intercept_: {model_fixed.intercept_}")
```

#### Fix a value for the coefficients

```{code-cell} ipython3
# hold all the weights at zero, learn the intercepts
model_fixed = nmo.glm.ClassifierGLM(
    classes=n_classes, fix_params=(np.zeros((X.shape[1], n_classes)), None)
).fit(X, labels)

print(f"coef_:      {model_fixed.coef_[:1]}")
print(f"intercept_: {model_fixed.intercept_}")
```

#### Partially specify the coefficients

If `X` is a PyTree, you can fix or learn each leaf of `coef` separately. The choice applies to all the classes together.

```{code-cell} ipython3
# hold the history weights of every class at zero, learn the stimulus ones
fixed = ({"stimulus": None, "history": np.zeros((12, n_classes))}, None)

model_fixed = nmo.glm.ClassifierGLM(classes=n_classes, fix_params=fixed).fit(X_dict, labels)

model_fixed.coef_["history"][:1]
```

#### Interaction between `fit_intercept` and `fix_params`

An intercept set with `fix_params` has priority over `fit_intercept`. With both arguments, the intercepts stay at the values in `fix_params`, not at zero.

```{code-cell} ipython3
both = nmo.glm.ClassifierGLM(
    classes=n_classes, fit_intercept=False, fix_params=(None, np.array([0.5, 1.0, -0.5]))
).fit(X, labels)
both.intercept_
```

(classifier-identifiability)=
## Identifiability

The model fits one linear predictor for each class: $n$ predictors for $n$ probabilities that must sum to one. This is one predictor more than necessary. When the first $n - 1$ probabilities are known, the last one is

$$
p_n = 1 - \sum_{k=1}^{n-1} p_k .
$$

This redundancy comes from an invariance of the softmax. The softmax does not change when you add the same constant to the linear predictor of every class.

```{code-cell} ipython3
import jax

# the linear predictor of one sample, one entry per class
z = np.array([0.3, -1.2, 2.0])

print(f"softmax(z):      {jax.nn.softmax(z)}")
print(f"softmax(z + 10): {jax.nn.softmax(z + 10.0)}")
```

If `(coef, intercept)` is a solution, `(coef + c, intercept + c)` is also a solution. Without regularization, the model has infinitely many solutions, and all of them give the same probabilities. The solution that the solver returns depends on its starting point. For this reason, `Ridge` is the default.

Ridge does not remove the invariance entirely. The penalty applies to the coefficients but not to the intercept. Thus, `coef_` is pinned and centered: it sums to zero across classes.

```{code-cell} ipython3
model.coef_.sum(axis=-1)[:4]
```

But `intercept_` can still shift. If the starting intercept is shifted by a constant, the fit returns the same shift in `intercept_`. The coefficients and the probabilities do not change.

```{code-cell} ipython3
coef, intercept = model.initialize_params(X, labels)

shifted = nmo.glm.ClassifierGLM(classes=n_classes).fit(
    X, labels, init_params=(coef, intercept + 5.0)
)

print(f"intercept_:         {model.intercept_}")
print(f"shifted intercept_: {shifted.intercept_}")
print(
    "same probabilities: "
    f"{np.allclose(model.predict_proba(X), shifted.predict_proba(X), atol=1e-4)}"
)
```

To penalize the intercept too, add it as a constant column of the design matrix and set `fit_intercept=False`. Then no parameter is free to shift.

```{code-cell} ipython3
X_with_intercept = np.concatenate([np.ones((X.shape[0], 1)), np.asarray(X)], axis=1)

folded = nmo.glm.ClassifierGLM(classes=n_classes, fit_intercept=False).fit(
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
