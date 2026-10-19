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

(glm-single-neuron)=
# Generalized Linear Model (GLM)

This page gives an overview of the GLM features in NeMoS.

```{image} ../../../assets/glm.svg
:alt: GLM. X of shape (n_samples, n_features) times a weight vector w, through the inverse link f, gives one output y of shape (n_samples,).
:class: only-light
```

```{image} ../../../assets/glm_dark.svg
:alt: GLM. X of shape (n_samples, n_features) times a weight vector w, through the inverse link f, gives one output y of shape (n_samples,).
:class: only-dark
```

```{contents}
:local:
:depth: 3
```

## Initializing a GLM object

Every argument of [`GLM`](nemos.glm.GLM) has a default value. Without arguments, the model is a Poisson GLM with no regularization.

```{code-cell} ipython3
import nemos as nmo

nmo.glm.GLM()
```

The arguments describe two things:

- The data: the observation model and the inverse link function.
- The estimation of the parameters: the regularizer and the solver.

[Model configurations](glm-model-configuration) explains each argument.

```{code-cell} ipython3
model = nmo.glm.GLM(
    observation_model="Gamma",
    regularizer="Ridge",
    regularizer_strength=0.1,
    solver_name="Newton",
)
model
```

(glm-input-types)=
## Inputs

`y` is always an array. `X` can be an array, or a container of arrays with one group of predictors per leaf. JAX calls this container a [PyTree](https://docs.jax.dev/en/latest/pytrees.html) of arrays. The supported array formats are [numpy](https://numpy.org/), [jax](https://docs.jax.dev/), and [pynapple](https://www.pynapple.org) time series.

### Arrays

`X` has shape `(n_samples, n_features)` and `y` has shape `(n_samples,)`. Each of them can be a NumPy array, a JAX array, or a [pynapple](https://www.pynapple.org) time series.

#### NumPy

```{code-cell} ipython3
import numpy as np

# X of shape (n_samples, n_features), y of shape (n_samples,)
X, y = nmo.datasets.tuned_poisson(n_samples=500)  # JAX arrays
X_np, y_np = np.asarray(X), np.asarray(y)

model_np = nmo.glm.GLM().fit(X_np, y_np)
```

The fitted parameters and the output of `predict` are JAX arrays, also for a NumPy input.

```{code-cell} ipython3
type(model_np.predict(X_np))
```

#### JAX

`fit` gives the same parameters for JAX arrays and for NumPy arrays.

```{code-cell} ipython3
model_jax = nmo.glm.GLM().fit(X, y)

np.allclose(model_jax.coef_, model_np.coef_)
```

#### Pynapple

`fit` also accepts [pynapple](https://www.pynapple.org) time series. A [pynapple](https://www.pynapple.org) object stores the timestamps with the values. It also stores a time support, which records the epochs of the session that the samples come from.

```{code-cell} ipython3
import pynapple as nap

# suppress jax to numpy conversion warning
nap.nap_config.suppress_conversion_warnings = True

# 10 ms bins
time = np.arange(X.shape[0]) / 100

# two epochs of different duration, labeled by trial type
epochs = nap.IntervalSet(
    start=[0.0, 3.0],
    end=[1.5, 5.],
    metadata={"contrast": ["high", "low"]},
)

predictors = nap.TsdFrame(t=time, d=np.asarray(X), time_support=epochs)
counts = nap.Tsd(t=time, d=np.asarray(y), time_support=epochs)

model_nap = nmo.glm.GLM().fit(predictors, counts)
model_nap
```

If the input is a [pynapple](https://www.pynapple.org) object, [`predict`](nemos.glm.GLM.predict) and [`simulate`](nemos.glm.GLM.simulate) return a [pynapple](https://www.pynapple.org) object.

### PyTrees

`X` can also be a PyTree of arrays, with one leaf per group of predictors. All leaves must have the same number of samples. You do not need to concatenate the groups into a single design matrix. The coefficients have the same structure as `X`. Examples of PyTrees are dictionaries, lists, and your own classes registered as PyTree nodes.

With a dictionary:

```{code-cell} ipython3
X_dict = {"stimulus": X[:, :4], "history": X[:, 4:]}

model_dict = nmo.glm.GLM().fit(X_dict, y)

model_dict.coef_
```

JAX flattens a dictionary in sorted key order. Thus, `history` comes before `stimulus`, whatever the order of the keys in the dictionary.

With a list:

```{code-cell} ipython3
# swap array order to match dict print order
model_list = nmo.glm.GLM().fit([X[:, 4:], X[:, :4]], y)

model_list.coef_
```

With your own class, registered as a PyTree node:

```{code-cell} ipython3
import equinox as eqx
from jax import Array


class Input(eqx.Module):
    stimulus: Array
    history: Array


model_eqx = nmo.glm.GLM().fit(Input(X[:, :4], X[:, 4:]), y)

model_eqx.coef_
```

The leaves can be [pynapple](https://www.pynapple.org) objects. In this case, the predicted rate keeps the time axis.

```{code-cell} ipython3
X_nap_tree = [predictors[:, :4], predictors[:, 4:]]

model_nap_tree = nmo.glm.GLM().fit(X_nap_tree, counts)

model_nap_tree.predict(X_nap_tree)
```

(glm-methods)=
## Methods

### Fit

[`fit`](nemos.glm.GLM.fit) estimates the parameters and returns the model. You can call it directly on the constructor.

```{code-cell} ipython3
model = nmo.glm.GLM().fit(X, y)
model
```

#### Fitted attributes

The fit stores the estimated quantities as attributes with a trailing underscore.

```{code-cell} ipython3
print(f"coef_:       {model.coef_.shape}")
print(f"intercept_:  {model.intercept_.shape}")
print(f"scale_:      {model.scale_}")
print(f"dof_resid_:  {model.dof_resid_}")
```

- `coef_`: one weight per column of the design matrix, on the scale of the linear term.
- `intercept_`: the baseline, on the scale of the linear term.
- `scale_`: the estimated dispersion.
- `dof_resid_`: the residual degrees of freedom.
- `solver_state_`: the state of the solver at the end of the fit.

#### NaN treatment

The fit ignores samples with a NaN in `X` or `y`. Examples of such samples are bins without a measurement, and the edges that a convolution leaves undefined. You do not need to remove these samples by hand.

```{code-cell} ipython3
X_nan = np.asarray(X).copy()
X_nan[:10] = np.nan

model_nan = nmo.glm.GLM().fit(X_nan, y)
model_dropped = nmo.glm.GLM().fit(np.asarray(X)[10:], np.asarray(y)[10:])

np.allclose(model_nan.coef_, model_dropped.coef_)
```

### Predict

[`predict`](nemos.glm.GLM.predict) returns the firing rate in counts per bin, one value per sample.

```{code-cell} ipython3
rate = model.predict(X)
rate.shape
```

To convert the rate to spikes per second, divide it by the bin size.

```{code-cell} ipython3
bin_size = 0.01  # 10 ms bins, in seconds

rate[:10] / bin_size  # rate in Hz
```

The output type matches the type of the predictors given to `predict`. The type of the data used for the fit does not matter.

```{code-cell} ipython3
print(type(model_nap.predict(X)))
print(type(model_nap.predict(predictors)))
```

A [pynapple](https://www.pynapple.org) output keeps the time axis of the predictors. You can restrict it to an epoch, or cross-correlate it with other time series, without adding the timestamps again. The timestamps come from the predictors given to `predict`. These predictors can cover any interval.

### Score

[`score`](nemos.glm.GLM.score) evaluates the model on data. By default, it returns the mean log-likelihood per sample.

```{code-cell} ipython3
model.score(X, y)
```

Set `score_type` to use one of the two pseudo-$R^2$ scores instead. Unlike the log-likelihood, a pseudo-$R^2$ has a maximum of 1, and you can compare it across datasets.

```{code-cell} ipython3
model.score(X, y, score_type="pseudo-r2-McFadden")
```

`aggregate_sample_scores` combines the per-sample scores into one value. The default is the mean. The function receives the array of per-sample scores. Thus, the identity function returns this array instead of a number.

```{code-cell} ipython3
model.score(X, y, aggregate_sample_scores=lambda x: x).shape
```

The sum gives the total log-likelihood over the samples.

```{code-cell} ipython3
import jax.numpy as jnp

model.score(X, y, aggregate_sample_scores=jnp.sum)
```

### Simulate

[`simulate`](nemos.glm.GLM.simulate) draws counts from the observation model at the predicted rate. It returns the counts and the rate.

```{code-cell} ipython3
import jax

counts_sim, rate_sim = model.simulate(jax.random.key(0), X)

print(f"counts: {counts_sim[:5]}")
print(f"rate:   {np.round(np.asarray(rate_sim[:5]), 2)}")
```

The random key makes the draw reproducible: the same key and the same rate give the same counts.

:::{admonition} Simulating a coupled population
:class: dropdown

[`simulate`](nemos.glm.GLM.simulate) is feed-forward: the simulated counts do not change the rate. To simulate a population in which spikes drive the next time step through coupling filters, use [`simulate_recurrent`](nemos.simulation.simulate_recurrent). See [simulate a coupled population](../../../how_to_guide/simulate_coupled_population.md).
:::

(glm-out-of-memory)=
### Fitting out of memory

If the design matrix does not fit in memory, use [`stochastic_fit`](nemos.glm.GLM.stochastic_fit). It takes a data loader instead of arrays, and updates the parameters one batch at a time. Stochastic solvers require acceleration and line searches to be off.

:::{admonition} In-memory example
:class: note

The arrays here fit in memory, so the loader is an [`ArrayDataLoader`](nemos.batching.ArrayDataLoader). For data that does not fit in memory, [`LazyArrayDataLoader`](nemos.batching.LazyArrayDataLoader) loads one batch at a time from a dask, zarr or HDF5 array.
:::

```{code-cell} ipython3
# the loader serves the data in batches of 100 samples
loader = nmo.batching.ArrayDataLoader(X, y, batch_size=100)

glm = nmo.glm.GLM(
    solver_name="GradientDescent",
    solver_kwargs={"stepsize": 0.05, "acceleration": False},
)
glm.stochastic_fit(loader, n_passes=20)
```

See [stochastic optimization](../../scalability/README.md).

#### Update

For full control of the optimization loop, call [`update`](nemos.glm.GLM.update) to do each step yourself. [`initialize_params`](nemos.glm.GLM.initialize_params) returns the initial parameters. [`initialize_optimizer_and_state`](nemos.glm.GLM.initialize_optimizer_and_state) sets up the solver.

```{code-cell} ipython3
params = glm.initialize_params(X, y)
opt_state = glm.initialize_optimizer_and_state(params, X, y)

# a full pass over all the batches
for X_batch, y_batch in loader:
    params, opt_state = glm.update(params, opt_state, X_batch, y_batch)
```

The loop body can contain any code. Examples are a log of the score for each batch, a change of the learning rate between passes, or your own stopping rule. See [custom update](../../scalability/custom_update.md).

(glm-model-configuration)=
## Model configurations

(glm-observation-model)=
### Observation models

The observation model is the distribution of the counts around the predicted rate. You can set it by name:

```{code-cell} ipython3
nmo.glm.GLM("Gamma")
```

You can also give the observation model object. The object gives access to the parameters of the family. The Negative Binomial needs the object, because its `dispersion` is a fixed value, not an estimated one.

```{code-cell} ipython3
from nemos.observation_models import NegativeBinomialObservations

nmo.glm.GLM(NegativeBinomialObservations(dispersion=2.0))
```

[Observation models](../../observation_models.md) describes each family, the signals it suits, and what it estimates.

(glm-inverse-link)=
### Inverse link function

The inverse link function maps the linear term to the mean of the observation model. Each family has a default. Only a few other choices keep the log-likelihood convex.

| Observation model | Default | Convexity preserving |
| --- | --- | --- |
| `Poisson` | `jnp.exp` | `jnp.exp`, `jax.nn.softplus` |
| `Gamma` | `one_over_x` | `jnp.exp`, `one_over_x` |
| `Gaussian` | `identity` | `identity` |
| `Bernoulli` | `logistic` | `logistic`, `expit` |
| `NegativeBinomial` | `jnp.exp` | `jnp.exp` |

:::{admonition} Convexity depends on the pair
:class: warning

`jax.nn.softplus` is safe for a Poisson GLM, but it makes the Negative Binomial log-likelihood non-convex. NeMoS does not reject such a model. The fit runs, but it converges to a local optimum that depends on the starting point. Also, the solver can no longer rely on a positive semi-definite curvature. NeMoS treats a link that is not in the table the same way.
:::

```{code-cell} ipython3
# convex
nmo.glm.GLM("Poisson", inverse_link_function=jax.nn.softplus)

# not convex
nmo.glm.GLM("NegativeBinomial", inverse_link_function=jax.nn.softplus)
```

(glm-regularizers)=
### Regularizers

Set the regularizer and its strength when you create the model. If the regularizer needs a strength and `regularizer_strength` is not set, NeMoS gives a warning and uses 1.

```{code-cell} ipython3
nmo.glm.GLM(regularizer="Lasso", regularizer_strength=0.05).fit(X, y).coef_
```

[Regularizers](../../regularizers.md) describes group-wise and parameter-wise strengths, and the special cases of ElasticNet and GroupLasso.

(glm-solvers)=
### Solvers

Select the solver by name, and configure it with `solver_kwargs`.

```{code-cell} ipython3
model = nmo.glm.GLM(solver_name="Newton", solver_kwargs={"tol": 1e-10}).fit(X, y)
```

The keys of `solver_kwargs` must be arguments of the solver. Other keys raise an error. To see the accepted arguments, print the solver docstring with [`get_solver_documentation`](nemos.solvers.get_solver_documentation). The next cell prints the start of the `LBFGS` docstring, up to the list of accepted arguments.

```{code-cell} ipython3
from nemos.solvers import get_solver_documentation

doc = get_solver_documentation("LBFGS")
print("\n".join(doc.splitlines()[:27]))
```

If you do not set the solver, NeMoS selects one that is compatible with the regularizer:

- `Newton` for Ridge.
- `LBFGS` without regularization.
- `ProximalGradient` for the non-smooth penalties.

```{code-cell} ipython3
print(nmo.glm.GLM().solver_name)
print(nmo.glm.GLM(regularizer="Ridge", regularizer_strength=0.01).solver_name)
print(nmo.glm.GLM(regularizer="Lasso", regularizer_strength=0.01).solver_name)
```

The registry lists the installed solvers as `algo_name` and `backend` pairs.

```{code-cell} ipython3
from nemos.solvers import list_available_solvers

sorted(f"{spec.algo_name}[{spec.backend}]" for spec in list_available_solvers())
```

[Solvers](../../solvers.md) describes which solver accepts which regularizer, which solver to use, and how to register your own solver.

(glm-parameter-specification)=
## Parameter specification

### Drop the intercept

`fit_intercept=False` removes the intercept from the model. The intercept stays at zero and uses no degree of freedom.

```{code-cell} ipython3
no_intercept = nmo.glm.GLM(fit_intercept=False).fit(X, y)

print(f"intercept_: {no_intercept.intercept_}")
print(f"dof_resid_: {no_intercept.dof_resid_}")
```

### Fix parameter values

`fix_params` keeps any leaf of the parameters at a value that you choose. The parameters are a `(coef, intercept)` tuple. If `X` is a PyTree, `coef` is also a PyTree. Thus, you can fix these leaves:

- the intercept,
- each leaf of `coef`.

For a leaf to fix, give its array of values. For a leaf to learn, give `None` as a placeholder.

If `X` is an array, `coef` is a single leaf. The fit fixes or learns all of it.

#### Fix a value for the intercept

```{code-cell} ipython3
# hold the intercept at 0.5, learn the weights
model_fixed = nmo.glm.GLM(fix_params=(None, np.array([0.5]))).fit(X, y)

print(f"coef_:      {model_fixed.coef_}")
print(f"intercept_: {model_fixed.intercept_}")
```

#### Fix a value for the coefficients

```{code-cell} ipython3
# hold all the weights at zero, learn the intercept
model_fixed = nmo.glm.GLM(fix_params=(np.zeros(8), None)).fit(X, y)

print(f"coef_:      {model_fixed.coef_}")
print(f"intercept_: {model_fixed.intercept_}")
```

#### Partially specify the coefficients

If `X` is a PyTree, you can fix or learn each leaf of `coef` separately.

```{code-cell} ipython3
# hold the history weights at zero, learn the stimulus ones
fixed = ({"stimulus": None, "history": np.zeros(4)}, None)

model_fixed = nmo.glm.GLM(fix_params=fixed).fit(X_dict, y)

print(f"coef_:      {model_fixed.coef_}")
print(f"intercept_: {model_fixed.intercept_}")
```

:::{admonition} Example use case
:class: dropdown

Fit the same design two times: one time with a group of predictors fixed at zero, and one time with this group free. The result is two nested models. Compare them to test whether that group explains the activity.
:::

#### Interaction between `fit_intercept` and `fix_params`

An intercept set with `fix_params` has priority over `fit_intercept`. With both arguments, the intercept stays at the value in `fix_params`, not at zero.

```{code-cell} ipython3
both = nmo.glm.GLM(fit_intercept=False, fix_params=(None, np.array([0.5]))).fit(X, y)
both.intercept_
```

## Examples

```{nemos-examples}
:model: GLM
```
