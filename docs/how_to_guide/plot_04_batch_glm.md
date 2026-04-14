---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
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
    message=("invalid value encountered in div "),
    category=RuntimeWarning,
)
```

# Batching example

Here we demonstrate how to set up and run a stochastic gradient descent in `nemos`.
\
Data will be batched using `pynapple`, fed to the GLM's [`stochastic_fit`](nemos.glm.GLM.stochastic_fit) method with a [`DataLoader`](nemos.batching.DataLoader), and the loss will be logged by a custom callback.

Then we show how callbacks can be used to stop iteration.

I think custom learning rate schedules could also be done with a callback. It has the state and can modify the stepsize.
But the states are immutable and are not read back, so it doesn't have an effect currently.

:::{admonition} Simpler alternative: `stochastic_fit`
:class: tip

The manual approach shown in this guide is useful when you need more control over the optimization loop, such as custom learning rate schedules, logging between batches, custom stopping criteria, etc.
:::

```{code-cell} ipython3
import jax
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import seaborn as sns

jax.enable_x64()

import nemos as nmo

nap.nap_config.suppress_conversion_warnings = True

# set random seed
np.random.seed(123)
```

## Simulate data

Let's generate some data artificially

```{code-cell} ipython3
n_neurons = 10
T = 50

times = np.linspace(0, T, 5000).reshape(-1, 1)
rate = np.exp(np.sin(times + np.linspace(0, np.pi * 2, n_neurons).reshape(1, n_neurons)))
```

Get the spike times from the rate and generate a `TsGroup` object

```{code-cell} ipython3
spike_t, spike_id = np.where(np.random.poisson(rate))
units = nap.Tsd(spike_t / T, spike_id).to_tsgroup()
```

## Basis instantiation

Here we instantiate the basis. `ws` is 40 time bins. It corresponds to a 200 ms windows

```{code-cell} ipython3
ws = 40
basis = nmo.basis.RaisedCosineLogConv(5, window_size=ws)
```

## Data loader definition

We define a data loader that samples a random 5s interval, bins spikes into spike counts, and generates a design matrix.

(The batch size needs to be larger than the window size of the convolution kernel defined above.)

```{code-cell} ipython3
batch_size = 5  # second
bin_size = 0.005
```

Data loaders included in NeMoS already cover the most common use-cases:

- `ArrayDataLoader`: use with in-memory arrays. Input and output are converted to jax arrays before use. Useful if data fits into memory, but calculations run out of memory, as well as for quick prototyping.
- `LazyArrayDataLoader`: use with lazy/out-of-memory arrays, such as dask, zarr, HDF5.

+++

Here, for illustration, we will define a data loader that uses `pynapple` to create batches of data of a given length.

To create a data loader, we have to define 3 things:
- `__iter__`: Iterate over tuples containing input and output data, e.g. (X_batch, y_batch). Must return a fresh iterator each call (re-iterable). Please note the use of `yield` in the code.
- `n_samples` property: Total number of samples in the dataset.
- `sample_batch`: Single batch for initialization purposes. Should be cheap to evaluate and deterministic.

Note that for best performane (i.e. avoid unnecesssary function recompilations) it is advised to generate batches that all have the same or at least a limited number of distinct sizes.

```{code-cell} ipython3
class PynappleDataLoader(nmo.batching.DataLoader):
    def __init__(self, batch_size: float, random_seed: int = 123):
        self.batch_size = batch_size
        self.n_batches_per_epoch = int(units.time_support.tot_length() // self.batch_size)
        self.rng = np.random.default_rng(seed=random_seed)

        # initialize the cached sample batch
        self._sample_batch = None

    def __iter__(self):
        for i in range(self.n_batches_per_epoch):
            yield self._random_batch()

    @property
    def n_samples(self):
        """Number of samples in the full dataset"""
        return int(np.round(units.time_support.tot_length() / bin_size))

    def sample_batch(self):
        """Generate a sample batch at the start of the time support."""
        if self._sample_batch is None:
            self._sample_batch = self._batch_at_t(units.time_support[0, 0])

        return self._sample_batch

    def _batch_at_t(self, t: float):
        """Generate a batch starting at time t."""
        # create epoch
        ep = nap.IntervalSet(t, t + self.batch_size)
        # bin the spike train
        counts = units.restrict(ep).count(bin_size)
        # bonvolve
        X = basis.compute_features(counts)

        return X, counts

    def _random_batch(self):
        """Generate a batch at a random time within the time support."""
        t = self.rng.uniform(
            units.time_support[0, 0],
            units.time_support[0, 1] - self.batch_size,
        )
        return self._batch_at_t(t)
```

Create the data loader:

```{code-cell} ipython3
loader = PynappleDataLoader(batch_size)
```

## Callback for logging

+++

To monitor training progress and the optimization's state during the fitting run, NeMoS has a callback system.

Callbacks can run defined functionality on the following events:
- on training beginning and end
- before and after each epoch
- before and after each batch

Information available to callbacks given through a ``TrainingContext`` object, which carries information about the state of the training such as the state, the current parameters, the current epoch's and batch's index.
\
For convenience, the model being fit is also added to the context, so if we want to log the score through training, we have access to it.

Custom callbacks should inherit from `nmo.callbacks.Callback` and overwrite the appropriate methods. In the current example we will implement `on_batch_end` to log the test score after parameters were updated on each batch of data.

For illustration, as it is small, we will just use the whole data and log the score at every batch, but generally this would be expensive and you would only evaluate every N-th batch or at the end of every epoch using a held-out smaller test set.

```{code-cell} ipython3
class TestScoreLoggingCallback(nmo.callbacks.Callback):
    """Callback logging the loss function evaluated on a constant test set."""

    def __init__(self, X_test, y_test):
        self.logl = []
        self.X_test = X_test
        self.y_test = y_test

    def _log_test_score(self, ctx):
        # TODO: this is a bit awkward that compute_loss can't just take ctx.params
        test_score = ctx.model.compute_loss(
            (ctx.params.coef, ctx.params.intercept),
            self.X_test,
            self.y_test,
        )
        self.logl.append(test_score)

    def on_train_begin(self, ctx):
        # log the loss right after initialization, before any training
        self._log_test_score(ctx)

    def on_batch_end(self, ctx):
        # log the loss at the end of each batch
        self._log_test_score(ctx)

    def plot_loss(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        batch_indices = np.arange(len(self.logl))
        ax.plot(
            batch_indices,
            self.logl,
            label="Loss per batch",
        )
        ax.scatter(
            batch_indices[:: loader.n_batches_per_epoch],
            self.logl[:: loader.n_batches_per_epoch],
            color="tab:orange",
            label="Loss at epoch boundaries",
        )

        ax.set_xlabel("Batch index")
        ax.set_ylabel("Loss value")

        ax.legend()
        sns.despine(ax=ax)

        return ax
```

NOTE for users:
some solvers evaluate and save the loss on each batch's data. This can be accessed by `ctx.state.stats.function_val`

+++

NOTE for users:
using model.score would require setting the model parameters and for some observation models estimating the scale. Without these you would get inaccurate scores, but for large data this is a lot of computation, so it is not recommended, just use compute_loss instead.

```{code-cell} ipython3
# as in this example it fits into memory, use full data for loss logging
full_counts = units.count(bin_size)
X = basis.compute_features(full_counts)

batch_logger = TestScoreLoggingCallback(X, full_counts)
```

## Model configuration

Let's imagine this dataset does not fit into memory. In such cases, we can use a batching approach to train the GLM, meaning that we divide our data into smaller epochs that do fit into memory and use one epoch at a time to update our model parameters.

Let's instantiate the [`PopulationGLM`](nemos.glm.PopulationGLM) . The default algorithm for [`PopulationGLM`](nemos.glm.PopulationGLM) is gradient descent.
Currently, we suggest using gradient descent for batching.
\
Alternatively, you can try using SVRG as it converges to an optimumum with a fixed stepsize. Setting the optimal stepsize and batch size for SVRG is not yet automated for large data.

tip for users:
Other solvers that can be used for stochastic optimization can be listed using ``nmo.solvers.list_stochastic_solvers()``.

:::{note}
You must shut down the dynamic update of the step for fitting a batched (also called stochastic) gradient descent.
For the `GradientDescent` solver, this can be done by setting the parameters `acceleration` to False and setting the `stepsize`.
`stochastic_fit` will throw an error if acceleration and linesearches are not turned off.
:::

```{code-cell} ipython3
glm = nmo.glm.PopulationGLM(
    solver_name="GradientDescent",
    solver_kwargs={"stepsize": 0.1, "acceleration": False},
)
```

## Running the optimization

```{code-cell} ipython3
glm.stochastic_fit(loader, num_epochs=10, callbacks=batch_logger)
```

NOTE for users:
stochastic_fit ignores the max_steps solver kwarg, num_epochs and the callbacks control when optimization stops.

```{code-cell} ipython3
batch_logger.plot_loss()
```

### Continuing fitting

We see that the model has not converged yet. If we want to continue a fit that was stopped, we can do that by providing the current model parameters as the starting point using the `init_params` argument.

```{code-cell} ipython3
# TODO: Wouldn't it be good to have a public get_model_params? Or a params property?
# users shouldn't know what parameters are optimized. Or there should be a way to check.
# get_params is a bit confusing, but I guess is needed for sklearn
# TODO: stochastic_fit could take a resume argument instead of only init_params?

glm.stochastic_fit(
    loader,
    num_epochs=10,
    init_params=(glm.coef_, glm.intercept_),
    callbacks=batch_logger,
)
```

```{code-cell} ipython3
batch_logger.plot_loss()
```

# Multiple callbacks and stopping the optimization

+++

We're getting closer, but still need some training.
Instead of guessing how many epochs we need, undershooting and restarting, or overshooting and waiting too long, we can use callbacks to stop optimization on convergence.

Callbacks can trigger stopping the optimization by calling ``ctx.request_stop()``.
\
Here, we will demonstrate this by defining a callback that stops based on the loss evaluated on a test set if it hasn't improved much for a given number of epochs.

Also note that multiple callbacks serving different functions can be used simulataneosly by passing them as a list to `stochastic_fit`.
\
Alternatively, a `CallbackList` can be constructed beforehand and passed a single callback.

tip for users:
NeMoS provides a `SolverConvergenceCallback` which evaluates the solver's own convergence criterion defined as its `stochastic_convergence_criterion`. For built-in solvers this means examining the change in parameter values at the end of each epoch.

```{code-cell} ipython3
class EarlyStoppingCallback(nmo.callbacks.Callback):
    """
    Stop training when test loss stops improving.

    Evaluates ``model.compute_loss(params, X_test, y_test)`` at the end of each epoch. If the loss does not improve by at least ``min_delta`` for
    ``patience`` consecutive epochs, requests an early stop.

    Parameters
    ----------
    X_test :
        Test input data.
    y_test :
        Test target data.
    patience :
        Number of epochs with no improvement before stopping.
    min_delta :
        Minimum decrease in loss to count as an improvement.
    reset :
        Reset loss tracking when starting a fit. Default True.
        Setting to False can be useful when continuing a fit.
    """

    def __init__(self, X_test, y_test, patience=5, min_delta=0.0, reset: bool = True):
        self.X_test = X_test
        self.y_test = y_test
        self.patience = patience
        self.min_delta = min_delta
        self.reset = reset

        self._best_loss = np.inf
        self._wait = 0

    def on_train_begin(self, ctx: nmo.callbacks.TrainingContext) -> None:
        """(Optionally) reset state at the start of training."""
        if self.reset:
            self._best_loss = np.inf
            self._ref_loss = np.inf
            self._wait = 0

    def on_epoch_end(self, ctx: nmo.callbacks.TrainingContext) -> None:
        """Check whether test loss has improved; request stop if patience exceeded."""
        current_loss = ctx.model.compute_loss(
            (ctx.params.coef, ctx.params.intercept),
            self.X_test,
            self.y_test,
        )
        if current_loss < self._best_loss:
            self._best_loss = current_loss

        if current_loss < self._ref_loss - self.min_delta:
            self._ref_loss = current_loss
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                ctx.request_stop(
                    f"Less than {self.min_delta} improvement for {self.patience} epochs.\n"
                    f"last loss: {current_loss:.6f}\n"
                    f"loss{self.patience} epochs ago: {self._ref_loss:.6f}\n"
                    f"best loss: {self._best_loss:.6f}\n"
                )
```

Let's set the number of epochs to a very high number, so we don't stop too soon and stopping is triggered by the early stopping callback.

To avoid running for too long in this demo, we set `min_delta` relatively high. In practice you want to set this to a level that represents no true improvement.

```{code-cell} ipython3
early_stopping = EarlyStoppingCallback(X, full_counts, patience=5, min_delta=0.2)

glm.stochastic_fit(
    loader,
    num_epochs=10_000,
    init_params=(glm.coef_, glm.intercept_),
    callbacks=[batch_logger, early_stopping],
)
```

```{code-cell} ipython3
ax = batch_logger.plot_loss()
ax.axhline(
    early_stopping._ref_loss,
    color="black",
    label=f"Loss level after which less than {early_stopping.min_delta} improvement was made",
)
ax.legend()
```

```{code-cell} ipython3
:tags: [hide-input]

## save image for thumbnail
# import os
# from pathlib import Path
#
# root = os.environ.get("READTHEDOCS_OUTPUT")
# if root:
#    path = Path(root) / "html/_static/thumbnails/how_to_guide"
## if local store in ../_build/html/...
# else:
#    path = Path("../_build/html/_static/thumbnails/how_to_guide")
#
## make sure the folder exists if run from build
# if root or Path("../assets/stylesheets").exists():
#    path.mkdir(parents=True, exist_ok=True)
#
# if path.exists():
#    fig.savefig(path / "plot_04_batch_glm.svg")
```

We can see that the log-likelihood is increasing but did not reach plateau yet.
The number of iterations can be increased to continue learning.

We can take a look at the coefficients.
Here we extract the weight matrix of shape `(n_neurons*n_basis, n_neurons)`
and reshape it to `(n_neurons, n_basis, n_neurons)`.
We then average along basis to get a weight matrix of shape `(n_neurons, n_neurons)`.

```{code-cell} ipython3
W = glm.coef_.reshape(len(units), basis.n_basis_funcs, len(units))
Wm = np.mean(np.abs(W), 1)

# Let's plot it.

plt.figure()
plt.imshow(Wm)
plt.xlabel("Neurons")
plt.ylabel("Neurons")
plt.show()
```

## Model comparison

Since this example is small enough, we can fit the full model until convergence and compare the scores.
Here we generate the design matrix and spike counts for the whole dataset.

```{code-cell} ipython3
full_model = nmo.glm.PopulationGLM(solver_name="LBFGS", solver_kwargs={"maxiter": 10_000}).fit(X, full_counts)
```

Now that the full model is fitted, we are scoring the full model and the batch model against the full datasets to compare the scores.
The score is pseudo-R2

```{code-cell} ipython3
full_scores = full_model.score(
    X, full_counts, aggregate_sample_scores=lambda x: np.mean(x, axis=0), score_type="pseudo-r2-McFadden"
)
batch_scores = glm.score(
    X, full_counts, aggregate_sample_scores=lambda x: np.mean(x, axis=0), score_type="pseudo-r2-McFadden"
)
```

Let's compare scores for each neurons as well as the coefficients.

```{code-cell} ipython3
plt.figure(figsize=(10, 8))
gs = plt.GridSpec(3, 2)
plt.subplot(gs[0, :])
plt.bar(np.arange(0, n_neurons), full_scores, 0.4, label="Full model")
plt.bar(np.arange(0, n_neurons) + 0.5, batch_scores, 0.4, label="Batch model")
plt.ylabel("Pseudo R2")
plt.xlabel("Neurons")
plt.ylim(0, 1)
plt.legend()
plt.subplot(gs[1:, 0])
plt.imshow(Wm)
plt.title("Batch model")
plt.subplot(gs[1:, 1])
Wm2 = np.mean(np.abs(full_model.coef_.reshape(len(units), basis.n_basis_funcs, len(units))), 1)
plt.imshow(Wm2)
plt.title("Full model")
plt.tight_layout()
plt.show()
```

As we can see, with a few iterations, the batch model manage to recover a similar coefficient matrix.

+++

### Compare solvers

```{code-cell} ipython3
import time

solver_logs = {}
solver_times = []
for solver_name in ("GradientDescent", "SVRG"):
    solver_spec = nmo.solvers.get_solver(solver_name)

    solver_kwargs = {"stepsize": 0.1}
    if "acceleration" in solver_spec.implementation.get_accepted_arguments():
        solver_kwargs["acceleration"] = False

    glm = nmo.glm.PopulationGLM(
        solver_name=solver_spec.full_name,
        solver_kwargs=solver_kwargs,
    )
    callback = TestScoreLoggingCallback(X, full_counts)

    loader = PynappleDataLoader(batch_size)

    tic = time.perf_counter()
    glm.stochastic_fit(loader, num_epochs=500, callbacks=callback)
    fit_time = time.perf_counter() - tic

    solver_logs[solver_spec.full_name] = callback.logl
    solver_times.append((solver_spec.full_name, fit_time))
    print(f"{solver_spec.full_name} done.")
```

```{code-cell} ipython3
fig, ax = plt.subplots()
for solver_name, ll in solver_logs.items():
    ax.plot(ll, label=solver_name, alpha=0.3)
ax.legend()
```

```{code-cell} ipython3
fig, ax = plt.subplots()
for solver_name, ll in solver_logs.items():
    ax.plot(ll[1000:1500], label=solver_name, alpha=0.3)
ax.legend()
```

```{code-cell} ipython3
fig, ax = plt.subplots()
for solver_name, ll in solver_logs.items():
    ax.plot(ll[-100:], label=solver_name, alpha=0.3)
ax.legend()
```

```{code-cell} ipython3
fig, ax = plt.subplots()
for solver_name, ll in solver_logs.items():
    # ax.plot(ll[1 :: loader.n_batches_per_epoch][-100:], label=solver_name, alpha=0.3)
    ax.plot(ll[:: loader.n_batches_per_epoch][-100:], label=solver_name, alpha=0.3)
ax.legend()
```

```{code-cell} ipython3
solver_times
```

# Hand-written (old way)

+++

## Solver initialization

First we need to initialize the gradient descent solver within the [`PopulationGLM`](nemos.glm.PopulationGLM) .
This gets you the initial parameters and the first state of the solver.

```{code-cell} ipython3
glm = nmo.glm.PopulationGLM(
    solver_name="GradientDescent",
    solver_kwargs={"stepsize": 0.1, "acceleration": False},
)
params = glm.initialize_params(*loader.sample_batch())
state = glm.initialize_optimizer_and_state(params, *loader.sample_batch())
```

Let's do a few iterations of gradient descent.

```{code-cell} ipython3
n_step = 500

for i in range(n_step):

    # Get a batch of data
    x, y = loader._random_batch()

    # Do one step of gradient descent.
    params, state = glm.update(params, state, x, y)
```

**This is not correct for SVRG.**

+++

:::{admonition} Input validation
:class: warning

The `update` method does not perform input validation each time it is called.
This design choice speeds up computation by avoiding repetitive checks. However,
it requires that all inputs to the `update` method strictly conform to the expected
dimensionality and structure as established during the initialization of the solver.
Failure to comply with these expectations will likely result in runtime errors or
incorrect computations.
:::

**TODO**: Is this true? Not just recompilation?
