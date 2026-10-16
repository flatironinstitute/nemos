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

(glm-population)=
# Population GLM

[`PopulationGLM`](nemos.glm.PopulationGLM) fits a GLM to every neuron of a population at once. It takes the same arguments and has the same methods as the [single neuron GLM](glm-single-neuron). This page covers only the differences:

- `y` has one column per neuron.
- Every parameter has an additional last axis for the neurons.
- Each neuron can have its own set of predictors.

Fitting the population jointly is faster than fitting one neuron at a time in a loop. JAX vectorizes the fit across neurons and can run it on a GPU. The loop uses less memory.

```{image} ../../../assets/population_glm.svg
:alt: PopulationGLM. X of shape (n_samples, n_features) times a weight matrix W with one column per neuron, through the inverse link f, gives an output Y of shape (n_samples, n_neurons).
:class: only-light
```

```{image} ../../../assets/population_glm_dark.svg
:alt: PopulationGLM. X of shape (n_samples, n_features) times a weight matrix W with one column per neuron, through the inverse link f, gives an output Y of shape (n_samples, n_neurons).
:class: only-dark
```

:::{note}

For an unregularized, Lasso, or Ridge GLM, the loss is a sum of one term per neuron. Fitting the population jointly and fitting each neuron on its own give the same parameters. The same is true for a GroupLasso whose groups do not mix neurons. In this case, each group contains the parameters of one neuron only. The default mask gives this grouping.

:::

```{contents}
:local:
:depth: 3
```

## Initializing a PopulationGLM object

The arguments and their defaults are the same as for [`GLM`](nemos.glm.GLM). There is one additional argument, [`feature_mask`](neuron-specific-features), which selects the predictors of each neuron.

```{code-cell} ipython3
import nemos as nmo

nmo.glm.PopulationGLM()
```

(glm-population-input-types)=
## Inputs

The inputs are the same as for the [single neuron GLM](glm-input-types). `y` is now two-dimensional, with shape `(n_samples, n_neurons)`.

### Arrays

`X` has shape `(n_samples, n_features)` and `y` has shape `(n_samples, n_neurons)`. Each of them can be a NumPy array, a JAX array, or a [pynapple](https://www.pynapple.org) time series.

#### NumPy

```{code-cell} ipython3
import numpy as np

# X of shape (n_samples, n_features), y of shape (n_samples, n_neurons)
X, y = nmo.datasets.tuned_poisson(n_samples=1000, n_neurons=2)  # JAX arrays
X_np, y_np = np.asarray(X), np.asarray(y)

model_np = nmo.glm.PopulationGLM().fit(X_np, y_np)
```

#### JAX

`fit` gives the same parameters for JAX arrays and for NumPy arrays.

```{code-cell} ipython3
model_jax = nmo.glm.PopulationGLM().fit(X, y)

np.allclose(model_jax.coef_, model_np.coef_)
```

#### Pynapple

With [pynapple](https://www.pynapple.org), `y` is a `TsdFrame` with one column per neuron. `fit` stores its column names and metadata. [`predict`](nemos.glm.PopulationGLM.predict) and [`simulate`](nemos.glm.PopulationGLM.simulate) attach them to their outputs.

```{code-cell} ipython3
import pynapple as nap

# suppress jax to numpy conversion warning
nap.nap_config.suppress_conversion_warnings = True

# 10 ms bins
time = np.arange(X.shape[0]) / 100

predictors = nap.TsdFrame(t=time, d=np.asarray(X))
counts = nap.TsdFrame(
    t=time,
    d=np.asarray(y),
    columns=["unit0", "unit1"],
    metadata={"region": ["M1", "PFC"]},
)

model_nap = nmo.glm.PopulationGLM().fit(predictors, counts)
```

### PyTrees

If `X` is a PyTree, each leaf of `coef_` has the shape of its predictors, plus a last axis for the neurons.

```{code-cell} ipython3
# the coupling filters play the role of the single neuron history
X_dict = {"stimulus": X[:, :4], "coupling": X[:, 4:]}

model_dict = nmo.glm.PopulationGLM().fit(X_dict, y)

{key: coef.shape for key, coef in model_dict.coef_.items()}
```

(glm-population-methods)=
## Methods

### Fit

```{code-cell} ipython3
model = nmo.glm.PopulationGLM().fit(X, y)
model
```

#### Fitted attributes

Every fitted attribute has one entry per neuron, along the last axis.

```{code-cell} ipython3
print(f"coef_:       {model.coef_.shape}")
print(f"intercept_:  {model.intercept_.shape}")
print(f"scale_:      {model.scale_}")
print(f"dof_resid_:  {model.dof_resid_}")
```

- `coef_`: column `j` holds the weights of neuron `j`.
- `intercept_`: entry `j` holds the baseline of neuron `j`.
- `scale_`: entry `j` holds the dispersion of neuron `j`.
- `dof_resid_`: entry `j` holds the residual degrees of freedom of neuron `j`.

The single neuron [`GLM`](nemos.glm.GLM) stores the same quantities without the neuron axis.

#### NaN treatment

`fit` ignores a sample if `X` or any column of `y` contains a NaN. Thus, a NaN for one neuron removes that sample for the whole population.

```{code-cell} ipython3
y_nan = np.asarray(y, dtype=float).copy()
# a gap in the recording of the first neuron only
y_nan[:10, 0] = np.nan

model_nan = nmo.glm.PopulationGLM().fit(X, y_nan)
model_dropped = nmo.glm.PopulationGLM().fit(np.asarray(X)[10:], np.asarray(y)[10:])

np.allclose(model_nan.coef_, model_dropped.coef_)
```

To keep the samples of the other neurons, fit the neuron with the gap separately, with a [`GLM`](nemos.glm.GLM).

### Predict

[`predict`](nemos.glm.PopulationGLM.predict) returns one rate per sample and neuron, in counts per bin.

```{code-cell} ipython3
model.predict(X).shape
```

With [pynapple](https://www.pynapple.org) predictors, `predict` returns a `TsdFrame` with one column per unit. The single neuron GLM returns a `Tsd`.

```{code-cell} ipython3
rate = model_nap.predict(predictors)
rate
```

A `TsdFrame` can store metadata that labels its columns. Here, the columns are the units of the population. `predict` copies the metadata of the counts used for the fit.

```{code-cell} ipython3
rate.metadata
```

You can then select the rate by any of these labels. You do not need to match the columns to the units by hand.

```{code-cell} ipython3
rate[:, rate.region == "M1"]
```

Only the column names and the metadata come from the fit. The type and the timestamps come from the predictors given to `predict`.

### Score

[`score`](nemos.glm.PopulationGLM.score) returns a single number: the average over samples and neurons. scikit-learn cross-validation expects this from an estimator.

```{code-cell} ipython3
model.score(X, y, score_type="pseudo-r2-McFadden")
```

`score` applies `aggregate_sample_scores` to an array of shape `(n_samples, n_neurons)`. An average over the sample axis only gives one score per neuron.

```{code-cell} ipython3
import jax.numpy as jnp

model.score(
    X,
    y,
    score_type="pseudo-r2-McFadden",
    aggregate_sample_scores=lambda x: jnp.mean(x, axis=0),
)
```

### Simulate

[`simulate`](nemos.glm.PopulationGLM.simulate) draws counts for every neuron. It returns the counts and the rate, both with shape `(n_samples, n_neurons)`.

```{code-cell} ipython3
import jax

counts_sim, rate_sim = model.simulate(jax.random.key(0), X)

print(f"counts: {counts_sim.shape}")
print(f"rate:   {rate_sim.shape}")
```

The simulation is feed-forward: the simulated counts do not change the rate through the coupling filters. For a recurrently coupled population, use [`simulate_recurrent`](nemos.simulation.simulate_recurrent). See [simulate a coupled population](../../../how_to_guide/simulate_coupled_population.md).

### Fitting out of memory

[`stochastic_fit`](nemos.glm.PopulationGLM.stochastic_fit) and [`update`](nemos.glm.PopulationGLM.update) work as for the [single neuron GLM](glm-out-of-memory), with the same arguments. Each batch of `y` from the loader contains all the neurons.

(glm-population-model-configuration)=
## Model configurations

You set observation models, inverse link functions, regularizers and solvers exactly as in the [single neuron GLM](glm-model-configuration).

Only one argument changes shape: a parameter-wise `regularizer_strength`. It has the shape of the parameters, including the neuron axis. `PopulationGLM` rejects an array with one strength per feature, as used by the single neuron GLM.

```{code-cell} ipython3
n_features, n_neurons = X.shape[1], y.shape[1]

# penalize the second neuron heavily, the first one barely
strength = np.full((n_features, n_neurons), 0.01)
strength[:, 1] = 100.0

nmo.glm.PopulationGLM(regularizer="Ridge", regularizer_strength=strength).fit(X, y).coef_
```

[Regularizers](../../regularizers.md) describes the group-wise and parameter-wise forms, and the special cases of ElasticNet and GroupLasso.

### GroupLasso

[`GroupLasso`](nemos.regularizer.GroupLasso) defines its groups per neuron. With the default mask, each leaf of `coef` is a separate group for each neuron. A group of predictors can then shrink to zero for one neuron and stay nonzero for another.

```{code-cell} ipython3
model_group = nmo.glm.PopulationGLM(
    regularizer="GroupLasso",
    regularizer_strength=0.3,
).fit(X_dict, y)

# norm of each group, one entry per neuron
{key: np.linalg.norm(coef, axis=0) for key, coef in model_group.coef_.items()}
```

With 2 leaves and 2 neurons there are 4 groups. Here, the first neuron keeps its stimulus weights and drops its coupling. The second neuron does the opposite.

A custom mask has shape `(n_groups, n_features, n_neurons)` for each leaf. The neuron axis is last, as in the parameters.

```{code-cell} ipython3
# group 0: the stimulus of every neuron; group 1: the coupling of every neuron
mask = np.zeros((2, n_features, n_neurons))
mask[0, :4] = 1
mask[1, 4:] = 1

nmo.glm.PopulationGLM(
    regularizer=nmo.regularizer.GroupLasso(mask=mask),
    regularizer_strength=0.3,
).fit(X, y)
```

:::{warning}

A group that contains several neurons, like the two groups above, couples the fits of these neurons. The joint fit is then different from separate fits for each neuron.

:::

(glm-population-parameter-specification)=
## Parameter specification

`fit_intercept` and `fix_params` work as in the [single neuron GLM](glm-parameter-specification). The fixed values have the shape of the parameters, with the neuron axis last.

### Fix parameter values

```{code-cell} ipython3
# hold one intercept per neuron, learn the weights
model_fixed = nmo.glm.PopulationGLM(
    fix_params=(None, np.array([0.5, 1.0]))
).fit(X, y)

print(f"intercept_: {model_fixed.intercept_}")
```

If `X` is a PyTree, you can fix or learn each leaf of `coef` separately. The choice applies to all the neurons together.

```{code-cell} ipython3
# hold the coupling weights of every neuron at zero, learn the stimulus ones
fixed = ({"stimulus": None, "coupling": np.zeros((4, n_neurons))}, None)

model_fixed = nmo.glm.PopulationGLM(fix_params=fixed).fit(X_dict, y)

model_fixed.coef_["coupling"]
```

To remove a group of predictors from some neurons only, use a feature mask.

(neuron-specific-features)=
### Neuron-specific features

`feature_mask` selects the predictors of each neuron. It has the shape of `coef_`. A 1 means that the model learns the weight. A 0 means that the model keeps the weight at zero.

For example, take three predictors and two neurons:

- Predictor 0 drives both neurons.
- Predictor 1 drives only the first neuron.
- Predictor 2 drives only the second neuron.

```{code-cell} ipython3
input_features = X[:, :3]

feature_mask = np.ones((3, n_neurons))
# drop predictor 2 from the first neuron
feature_mask[2, 0] = 0
# drop predictor 1 from the second neuron
feature_mask[1, 1] = 0

model_masked = nmo.glm.PopulationGLM(feature_mask=feature_mask).fit(input_features, y)

model_masked.coef_
```

`coef_[i, j]` is zero where `feature_mask[i, j]` is zero. You can give the mask at initialization, or set it as an attribute before the fit. After the fit, you can still read the attribute, but a new assignment raises an error.

The fit does not estimate a masked weight. Thus, a masked weight does not count in the residual degrees of freedom of its neuron.

```{code-cell} ipython3
unmasked = nmo.glm.PopulationGLM().fit(input_features, y)

print(f"masked:   {model_masked.dof_resid_}")
print(f"unmasked: {unmasked.dof_resid_}")
```

#### Masking groups of features

If `X` is a PyTree, the mask has the same structure. Each leaf of the mask has the shape of its coefficients.

```{code-cell} ipython3
# the stimulus drives both neurons, the coupling only the second one
feature_mask = {
    "stimulus": np.ones((4, n_neurons)),
    "coupling": np.array([[0, 1]] * 4),
}

model_dict = nmo.glm.PopulationGLM(feature_mask=feature_mask).fit(X_dict, y)

model_dict.coef_["coupling"]
```

A single column of zeros removes the entire `"coupling"` group from the predictors of the first neuron. You do not have to index the columns of the design one by one. Use this form when the mask encodes a hypothesis about which inputs reach which neuron. For example, remove self-coupling, or limit coupling to a sub-population.

## Examples

```{nemos-examples}
:model: PopulationGLM
```
