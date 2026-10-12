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

```{code-cell} ipython3
:tags: [hide-input]

%matplotlib inline
import warnings

# Ignore the first specific warning
warnings.filterwarnings(
    "ignore",
    message="plotting functions contained within `_documentation_utils` are intended for nemos's documentation.",
    category=UserWarning,
)

# Ignore the second specific warning
warnings.filterwarnings(
    "ignore",
    message="Ignoring cached namespace 'core'",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=(
        "invalid value encountered in div "
    ),
    category=RuntimeWarning,
)
```

# Population GLM

Fitting the activity of a neural population with NeMoS can be much more efficient than fitting each individual
neuron in a loop, because `JAX` vectorizes the fit across neurons and runs it on the GPU.

:::{note}
For an unregularized, Lasso, Ridge, or group-Lasso GLM, fitting a GLM one neuron at the time, or fitting jointly
the neural population is equivalent. The main difference between the approaches is that the former is more
memory efficient, the latter is computationally more efficient (it takes less time to fit).
:::

## Fitting a Population GLM

NeMoS has a dedicated [`nemos.GLM.PopulationGLM`](nemos.glm.PopulationGLM) class for fitting jointly a neural population. The API
 is very similar to that the regular [`GLM`](nemos.glm.GLM), but with a few differences:

 1. The `y` input to the methods [`fit`](nemos.glm.PopulationGLM.fit) and [`score`](nemos.glm.PopulationGLM.score) must be a two-dimensional array of shape `(n_samples, n_neurons)`.
 2. You can optionally pass a `feature_mask` in the form of an array of 0s and 1s with shape `(n_features, n_neurons)`
 that specifies which features are used as predictors for each neuron. More on this [later](#neuron-specific-features).

The rest of the configurations (observation models, regularizers etc.) work exactly as in the single neuron GLM. Refer to the [single neuron GLM](./single_neuron.md) user guide for those.

Let's generate some synthetic data and fit a population model.

```{code-cell} ipython3
import jax.numpy as jnp
import matplotlib.pyplot as plt

import numpy as np

import nemos as nmo

n_neurons = 2
n_samples = 1000

# generate data:
# - X array of shape (n_samples, n_features)
# - spikes array of shape (n_samples, n_features)
X, spikes = nmo.datasets.tuned_poisson(
    n_samples=n_samples, n_neurons=n_neurons
)
print(f"X shape:      {X.shape}")
print(f"spikes shape: {spikes.shape}")

# define and fit the model
model = nmo.glm.PopulationGLM(solver_name="Newton")
model.fit(X, spikes)
```

The neuron axis comes last in the fitted parameters.

```{code-cell} ipython3
print(f"coef_ shape:      {model.coef_.shape}")
print(f"intercept_ shape: {model.intercept_.shape}")
```

`coef_` is `(n_features, n_neurons)` and `intercept_` is `(n_neurons,)`, so column `j` of `coef_` holds the entire weight vector of neuron `j`. The single neuron [`GLM`](nemos.glm.GLM) stores the same quantities without that trailing axis: `coef_` of shape `(n_features,)` and `intercept_` of shape `(1,)`.

Predicting the rate works in the same way,

```{code-cell}
import jax

Xnew, _ = nmo.datasets.tuned_poisson(
    n_samples=n_samples, n_neurons=n_neurons, random_key=jax.random.key(2),
)
# rate in spikes/bin
rate = model.predict(Xnew)
rate.shape
```

## Scoring neuron by neuron

[`score`](nemos.glm.PopulationGLM.score) returns a single number, the average over samples and neurons, which is what scikit-learn cross-validation expects from an estimator.

```{code-cell} ipython3
model.score(X, spikes, score_type="pseudo-r2-McFadden")
```

Samples are aggregated by `aggregate_sample_scores`, which is applied to an array of shape `(n_samples, n_neurons)`. Averaging over the sample axis alone leaves one score per neuron.

```{code-cell} ipython3
model.score(
    X,
    spikes,
    score_type="pseudo-r2-McFadden",
    aggregate_sample_scores=lambda x: jnp.mean(x, axis=0),
)

```

(neuron-specific-features)=
## Neuron-specific features
With [`PopulationGLM`](nemos.glm.PopulationGLM), one can set up a model that fits a different set of predictors for each neuron. The way to do so is to specify a `feature_mask`.

Let's assume that we have two neurons, share one shared input, and have an extra private one, for a total of
3 inputs.

```{code-cell} ipython3
# let's take the first three input
n_features = 3
input_features = X[:, :3]
```

Let's assume that:

  - `input_features[:, 0]` is shared.
  - `input_features[:, 1]` is an input only for the first neuron.
  - `input_features[:, 2]` is an input only for the second neuron.


We can impose the same constraint to the [`PopulationGLM`](nemos.glm.PopulationGLM) by masking the weights.

```{code-cell} ipython3
# initialize the mask to a matrix of 1s.
feature_mask = np.ones((n_features, n_neurons))

# remove the 3rd feature from the predictors of the first neuron
feature_mask[2, 0] = 0

# remove the 2nd feature from the predictors of the second neuron
feature_mask[1, 1] = 0

# visualize the mask
print(feature_mask)
```

The mask can be passed at initialization or set after the model is initialized, but cannot be changed after the model is fit.

```{code-cell} ipython3
# set a quasi-newton solver and low tolerance for better numerical precision
model = nmo.glm.PopulationGLM(solver_name="Newton")

# set the mask
model.feature_mask = feature_mask

# fit the model
model.fit(input_features, spikes)
```

If we print the model coefficients, we can see the effect of the mask.

```{code-cell} ipython3
model.coef_
```

The coefficient for the first neuron corresponding to the last feature is zero, as well as the coefficient of the second neuron corresponding to the second feature.

The mask has the same shape as `coef_`, and `coef_[i, j]` is zero wherever `feature_mask[i, j]` is zero. The mask is stored as the `feature_mask` attribute. It can be accessed after fitting, but not reassigned.

A masked coefficient is not estimated, so it does not count towards the residual degrees of freedom.

```{code-cell} ipython3
# same design, no mask
unmasked = nmo.glm.PopulationGLM(solver_name="Newton").fit(input_features, spikes)

print(f"masked:   {model.dof_resid_}")
print(f"unmasked: {unmasked.dof_resid_}")
```

Each neuron estimates one parameter fewer under the mask. `dof_resid_` has one entry per neuron, since every neuron has its own set of predictors.

### Masking groups of features

When the predictors are a dictionary, one entry per basis or per group of features, the mask mirrors that same structure, each leaf shaped like its own coefficients.

```{code-cell} ipython3
# split the design in two groups of features
X_dict = {"stimulus": X[:, :4], "coupling": X[:, 4:]}

# the stimulus drives both neurons, the coupling only the second one
feature_mask = {
    "stimulus": np.ones((4, n_neurons)),
    "coupling": np.array([[0, 1]] * 4),
}

model_dict = nmo.glm.PopulationGLM(
    solver_name="Newton",
    feature_mask=feature_mask
)
model_dict.fit(X_dict, spikes)

model_dict.coef_["coupling"]
```

A single column of zeros removes the entire `"coupling"` group from the predictors of the first neuron, without indexing the columns of the design one by one. This is the form to use when the mask encodes a hypothesis about which inputs reach which neuron, for example dropping self-coupling or restricting coupling to a sub-population.

## Keeping track of the units with pynapple

NeMoS does not require [pynapple](https://pynapple-org.github.io/pynapple/), but it accepts its objects wherever an array is accepted. The column names and the metadata of `y` are stored at fit time, and are attached again to the output of [`predict`](nemos.glm.PopulationGLM.predict).

```{code-cell} ipython3
import pynapple as nap

# suppress jax to numpy conversion warning
nap.nap_config.suppress_conversion_warnings = True

# 10 ms bins
time = np.arange(n_samples) / 100

counts = nap.TsdFrame(
    t=time,
    d=np.asarray(spikes),
    columns=["unit0", "unit1"],
    metadata={"region": ["M1", "PFC"]},
)
predictors = nap.TsdFrame(t=time, d=np.asarray(X))

model_nap = nmo.glm.PopulationGLM(solver_name="Newton").fit(predictors, counts)
model_nap.predict(predictors)
```

The predicted rate is a `TsdFrame` carrying the unit names and the region of each unit, so it can be restricted to an epoch or split by region without matching the columns back to the units.

Only the column names and the metadata are taken from the fit. The timestamps and the time support of the output are those of the predictors passed to `predict`, which can cover any interval.

```{code-cell} ipython3
model_nap.predict(predictors[:10])
```

The same rule decides the type of the output: predictors passed as a plain array return a plain array, even for a model that was fit on pynapple objects.

## Examples

```{nemos-examples}
:model: PopulationGLM
```
