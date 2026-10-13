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
# Single Neuron GLM

This documentation page is a quick overview of the GLM capabilities in NeMoS.

## Initializing a GLM object

Every argument of [`GLM`](nemos.glm.GLM) has a default, so an unconfigured model is already a Poisson GLM with no regularization.

```{code-cell} ipython3
import nemos as nmo

nmo.glm.GLM()
```

The arguments describe either the data — the observation model and the inverse link function — or the way its parameters are estimated — the regularizer and the solver. Each of them is covered in [model configurations](glm-model-configuration).

```{code-cell} ipython3
model = nmo.glm.GLM(
    observation_model="Gamma",
    regularizer="Ridge",
    regularizer_strength=0.1,
    solver_name="Newton",
)
model
```

(glm-methods)=
## Methods

(glm-input-types)=
:::{admonition} Supported inputs
:class: note

`y` is always an array. `X` can be an array, or a container of arrays holding one group of predictors per leaf, which in JAX is called a [PyTree](https://docs.jax.dev/en/latest/pytrees.html) of arrays.

Supported array formats are: [numpy](https://numpy.org/), [jax](https://docs.jax.dev/), and [pynapple](https://pynapple-org.github.io/pynapple/) time series. [`predict`](nemos.glm.GLM.predict) and [`simulate`](nemos.glm.GLM.simulate) return a pynapple object whenever they are given one.

Examples of PyTrees are dictionaries, lists, and classes of your own registered as PyTree nodes.
:::

### Fit

[`fit`](nemos.glm.GLM.fit) estimates the parameters and returns the model itself, so it can be chained to the constructor.

```{code-cell} ipython3
# X of shape (n_samples, n_features), y of shape (n_samples,)
X, y = nmo.datasets.tuned_poisson(n_samples=500)

model = nmo.glm.GLM().fit(X, y)
model
```

#### Pynapple Inputs

The same call takes pynapple objects, which carry the timestamps along with the values, and a time support recording which epochs of the session the samples come from.

```{code-cell} ipython3
import numpy as np
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

#### PyTree Inputs

`X` can also be a PyTree of arrays, one leaf per group of predictors, each leaf holding the same number of samples. This saves concatenating the groups into a single design matrix, and the coefficients come back with the structure of `X`.


With a dictionary,

```{code-cell} ipython3
X_dict = {"stimulus": X[:, :4], "history": X[:, 4:]}

model_dict = nmo.glm.GLM().fit(X_dict, y)

model_dict.coef_
```

A dictionary is flattened by sorted key, so `history` comes out before `stimulus` whatever the order it was built in.

With a list,

```{code-cell} ipython3
# swap array order to match dict print order
model_list = nmo.glm.GLM().fit([X[:, 4:], X[:, :4]], y)

model_list.coef_
```

Or with a class of your own, registered as a PyTree node,

```{code-cell} ipython3
import equinox as eqx
from jax import Array


class Input(eqx.Module):
    stimulus: Array
    history: Array


model_eqx = nmo.glm.GLM().fit(Input(X[:, :4], X[:, 4:]), y)

model_eqx.coef_
```

The leaves can themselves be pynapple objects, and the rate comes back with its time axis.

```{code-cell} ipython3
X_nap_tree = [predictors[:, :4], predictors[:, 4:]]

model_nap_tree = nmo.glm.GLM().fit(X_nap_tree, counts)

model_nap_tree.predict(X_nap_tree)
```

#### Fitted Attributes

The estimated quantities are stored as attributes with a trailing underscore.

```{code-cell} ipython3
print(f"coef_:       {model.coef_.shape}")
print(f"intercept_:  {model.intercept_.shape}")
print(f"scale_:      {model.scale_}")
print(f"dof_resid_:  {model.dof_resid_}")
```

`coef_` holds one weight per column of the design and `intercept_` the baseline, both on the scale of the linear term. `scale_` is the estimated dispersion, `dof_resid_` the residual degrees of freedom, and `solver_state_` the state the solver stopped at.


#### NaN Treatment

Samples with a NaN in either `X` or `y` are left out of the fit, so bins without a measurement, or the edges left undefined by a convolution, do not have to be removed by hand.

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

Dividing by the bin size converts it to spikes per second.

```{code-cell} ipython3
bin_size = 0.01  # 10 ms bins, in seconds

rate[:10] / bin_size  # rate in Hz
```

The type of the output follows the predictors passed to `predict`, whatever the model was fit on.

```{code-cell} ipython3
print(type(model_nap.predict(X)))
print(type(model_nap.predict(predictors)))
```

A pynapple output carries the time axis of the predictors, so it can be restricted to an epoch or cross-correlated with other time series without re-attaching the timestamps. The timestamps are those of the predictors passed to `predict`, which can cover any interval.

### Score

[`score`](nemos.glm.GLM.score) evaluates the model on data, by default the mean log-likelihood per sample.

```{code-cell} ipython3
model.score(X, y)
```

`score_type` switches to one of the two pseudo-$R^2$, which unlike the log-likelihood are bounded above by 1 and comparable across datasets.

```{code-cell} ipython3
model.score(X, y, score_type="pseudo-r2-McFadden")
```

The samples are aggregated by `aggregate_sample_scores`, which defaults to the mean. It is applied to the array of per-sample scores, so passing the identity returns that array instead of a number.

```{code-cell} ipython3
model.score(X, y, aggregate_sample_scores=lambda x: x).shape
```

Summing instead gives the total log-likelihood over the samples.

```{code-cell} ipython3
import jax.numpy as jnp

model.score(X, y, aggregate_sample_scores=jnp.sum)
```

### Simulate

[`simulate`](nemos.glm.GLM.simulate) draws counts from the observation model at the predicted rate, returning the counts and the rate they were drawn from.

```{code-cell} ipython3
import jax

counts_sim, rate_sim = model.simulate(jax.random.key(0), X)

print(f"counts: {counts_sim[:5]}")
print(f"rate:   {np.round(np.asarray(rate_sim[:5]), 2)}")
```

The random key makes the draw reproducible: the same key and the same rate give the same counts.

:::{admonition} Simulating a coupled population
:class: dropdown

[`simulate`](nemos.glm.GLM.simulate) is feed-forward: the counts it draws do not feed back into the rate. To simulate a population whose spikes drive the next time step through coupling filters, use [`simulate_recurrent`](nemos.simulation.simulate_recurrent), shown in [simulate a coupled population](../../../how_to_guide/simulate_coupled_population.md).
:::

### Fitting out of memory

When the design does not fit in memory, [`stochastic_fit`](nemos.glm.GLM.stochastic_fit) takes a data loader instead of arrays and updates the parameters one batch at a time. Stochastic solvers require acceleration and line searches to be turned off.

:::{admonition} In-memory example
:class: note

The arrays here fit in memory, so the loader is an [`ArrayDataLoader`](nemos.batching.ArrayDataLoader). For data that does not fit, [`LazyArrayDataLoader`](nemos.batching.LazyArrayDataLoader) loads one batch at a time from a dask, zarr or HDF5 array.
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

For full control over the optimization loop, step the parameters yourself with [`update`](nemos.glm.GLM.update). [`initialize_params`](nemos.glm.GLM.initialize_params) returns a starting point and [`initialize_optimizer_and_state`](nemos.glm.GLM.initialize_optimizer_and_state) sets up the solver.

```{code-cell} ipython3
params = glm.initialize_params(X, y)
opt_state = glm.initialize_optimizer_and_state(params, X, y)

# a full pass over all the batches
for X_batch, y_batch in loader:
    params, opt_state = glm.update(params, opt_state, X_batch, y_batch)
```

Anything can go in the loop body: a score logged per batch, a learning rate changed between passes, a stopping rule of your own. See [custom update](../../scalability/custom_update.md).

(glm-model-configuration)=
## Model Configurations

(glm-observation-model)=
### Observation Models

The observation model is the distribution of the counts around the predicted rate. It can be set by name.

```{code-cell} ipython3
nmo.glm.GLM("Gamma")
```

Passing the object instead gives access to the parameters of the family. The Negative Binomial is the case that needs it, since its `scale` is fixed rather than estimated.

```{code-cell} ipython3
from nemos.observation_models import NegativeBinomialObservations

nmo.glm.GLM(NegativeBinomialObservations(scale=2.0))
```

The families, the signals they suit, and what each one estimates are covered in [observation models](../../observation_models.md).

(glm-inverse-link)=
### Inverse Link Function

The inverse link maps the linear term to the mean of the observation model. Each family comes with a default, and only a few other choices keep the log-likelihood convex.

| Observation model | Default | Convexity preserving |
| --- | --- | --- |
| `Poisson` | `jnp.exp` | `jnp.exp`, `jax.nn.softplus` |
| `Gamma` | `one_over_x` | `jnp.exp`, `one_over_x` |
| `Gaussian` | `identity` | `identity` |
| `Bernoulli` | `logistic` | `logistic`, `expit` |
| `NegativeBinomial` | `jnp.exp` | `jnp.exp` |

:::{admonition} Convexity depends on the pair
:class: warning

The same `jax.nn.softplus` that is safe for a Poisson GLM makes the Negative Binomial log-likelihood non-convex. NeMoS does not reject such a model: it fits, but the solution it converges to is a local optimum that depends on the starting point, and the solver can no longer rely on the curvature being positive semi-definite. A link that is not in the table is treated the same way.
:::

```{code-cell} ipython3
# convex
nmo.glm.GLM("Poisson", inverse_link_function=jax.nn.softplus)

# not convex
nmo.glm.GLM("NegativeBinomial", inverse_link_function=jax.nn.softplus)
```

(glm-regularizers)=
### Regularizers

The regularizer and its strength are set at initialization. Leaving `regularizer_strength` unset for a model that needs it warns and falls back to 1.

```{code-cell} ipython3
nmo.glm.GLM(regularizer="Lasso", regularizer_strength=0.05).fit(X, y).coef_
```

Group-wise and parameter-wise strengths, and the special cases of ElasticNet and GroupLasso, are covered in [regularizers](../../regularizers.md).

(glm-solvers)=
### Solvers

The solver is chosen by name and configured through `solver_kwargs`.

```{code-cell} ipython3
model = nmo.glm.GLM(solver_name="Newton", solver_kwargs={"tol": 1e-10}).fit(X, y)
```

The keys `solver_kwargs` takes are the arguments of the solver itself, which it can be asked for. Anything else raises.

```{code-cell} ipython3
from nemos.solvers import get_solver

sorted(get_solver("Newton").implementation.get_accepted_arguments())
```

[`get_solver_documentation`](nemos.solvers.get_solver_documentation) returns the solver's docstring, which says what each argument does. Expand the cell below for the `Newton` one.

```{code-cell} ipython3
:tags: [hide-cell]

from nemos.solvers import get_solver_documentation

print(get_solver_documentation("Newton"))
```

Left unset, NeMoS picks a solver that the regularizer admits: `Newton` for Ridge, `LBFGS` without regularization, and `ProximalGradient` for the non-smooth penalties.

```{code-cell} ipython3
print(nmo.glm.GLM().solver_name)
print(nmo.glm.GLM(regularizer="Ridge", regularizer_strength=0.01).solver_name)
print(nmo.glm.GLM(regularizer="Lasso", regularizer_strength=0.01).solver_name)
```

The registry lists what is installed, as `algo_name` and `backend` pairs.

```{code-cell} ipython3
from nemos.solvers import list_available_solvers

sorted(f"{spec.algo_name}[{spec.backend}]" for spec in list_available_solvers())
```

Which solver accepts which regularizer, which one to reach for, and how to register one of your own are covered in [solvers](../../solvers.md).

(glm-parameter-specification)=
## Parameter specification

### Drop the intercept

`fit_intercept=False` drops the intercept from the model, which is then held at zero and costs no degree of freedom.

```{code-cell} ipython3
no_intercept = nmo.glm.GLM(fit_intercept=False).fit(X, y)

print(f"intercept_: {no_intercept.intercept_}")
print(f"dof_resid_: {no_intercept.dof_resid_}")
```

### Fix parameter values

`fix_params` holds any leaf of the parameters fixed at a value of your choosing. The parameters are a `(coef, intercept)` tuple, and `coef` is a PyTree whenever `X` is one, so the leaves that can be fixed are:

- the intercept,
- each leaf of `coef`.

A leaf to fix carries its array of values, a leaf to learn carries `None` as a placeholder.

With `X` an array, `coef` is a single leaf, held or learned as a whole.


### Fix a value for the intercept

```{code-cell} ipython3
# hold the intercept at 0.5, learn the weights
model_fixed = nmo.glm.GLM(fix_params=(None, np.array([0.5]))).fit(X, y)

print(f"coef_:      {model_fixed.coef_}")
print(f"intercept_: {model_fixed.intercept_}")
```

### Fix a value for the coefficients

```{code-cell} ipython3
# hold all the weights at zero, learn the intercept
model_fixed = nmo.glm.GLM(fix_params=(np.zeros(8), None)).fit(X, y)

print(f"coef_:      {model_fixed.coef_}")
print(f"intercept_: {model_fixed.intercept_}")
```

### Partially specify the coefficients

With `X` a PyTree, each leaf of `coef` is held or learned on its own.

```{code-cell} ipython3
# hold the history weights at zero, learn the stimulus ones
fixed = ({"stimulus": None, "history": np.zeros(4)}, None)

model_fixed = nmo.glm.GLM(fix_params=fixed).fit(X_dict, y)

print(f"coef_:      {model_fixed.coef_}")
print(f"intercept_: {model_fixed.intercept_}")
```

:::{admonition} Example use case
:class: dropdown

Fitting the same design once with a group of predictors held at zero and once with it free gives two nested models, which can be compared to test whether that group explains the activity.
:::

### Interaction between `fit_intercept` and `fix_params`

An intercept set through `fix_params` takes precedence over `fit_intercept`: the two together hold the intercept at the value given, rather than at zero.

```{code-cell} ipython3
both = nmo.glm.GLM(fit_intercept=False, fix_params=(None, np.array([0.5]))).fit(X, y)
both.intercept_
```

## Examples

```{nemos-examples}
:model: GLM
```
