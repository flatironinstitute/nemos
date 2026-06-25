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

# Ignore the specific warning
warnings.filterwarnings(
    "ignore",
    message="plotting functions contained within `_documentation_utils` are intended for nemos's documentation.",
    category=UserWarning,
)
```

# Creating a custom `DataLoader`

In the previous section we loaded the model's input and output data from disk. These were already preprocessed and stored as arrays with matching lengths (`X` and `spike_trains`), making them easy to plug into [`LazyArrayDataLoader`](nemos.batching.LazyArrayDataLoader).

Now, for illustration purposes, let's imagine we only have the raw spike times saved on disk.
Here, we show how to create a custom data loader that creates arrays the GLM can work with from these spike times on the fly.

```{code-cell} ipython3
import jax
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import seaborn as sns

from nemos._documentation_utils._stochastic_optim_toy_data import (
    DEFAULT_NWB_PATH,
    _simulate_and_write_to_disk,
)
from nemos._documentation_utils.plotting import plot_loss_history

jax.enable_x64()

import nemos as nmo

nap.nap_config.suppress_conversion_warnings = True

np.random.seed(123)
```

Load the spike times:

```{code-cell} ipython3
_simulate_and_write_to_disk()
data = nap.load_file(DEFAULT_NWB_PATH)
units = data["units"]  # contains spike times per unit
```

## Define `PynappleDataLoader`

+++

Batches of inputs and outputs will be created the following way:
1. Use `pynapple` to sample random 1 s intervals and bin spikes into counts.
2. Pass the counts through a convolutional basis to build a design matrix.

To create a [`DataLoader`](nemos.batching.DataLoader) for use with [`stochastic_fit`](nemos.glm.GLM.stochastic_fit), we have to define 3 things:
- `__iter__`: called every epoch; yields `(X_batch, y_batch)` tuples. Must return a fresh iterator each call (re-iterable). Note the use of `yield` in the code.
- `n_samples` property: total number of samples in the dataset.
- `sample_batch`: called at initialization. Should be cheap to evaluate and deterministic.

:::{note}
For best performance (i.e. to avoid unnecessary function recompilations), generate batches that all have the same or at least a limited number of distinct sizes.
:::

```{code-cell} ipython3
class PynappleDataLoader(nmo.batching.DataLoader):
    def __init__(self, spike_times, basis, batch_size, bin_size):
        self.spike_times = spike_times
        self.basis = basis
        self.batch_size = batch_size
        self.bin_size = bin_size
        self.n_batches_per_epoch = int(
            spike_times.time_support.tot_length() // self.batch_size
        )
        self.rng = np.random.default_rng(seed=123)

        self._sample_batch = None

    def __iter__(self):
        """
        Yield one batch per step for a full epoch.

        Defining this is what lets `stochastic_fit` loop over the loader
        (`for X_batch, y_batch in loader`) one batch at a time.
        It is called once at the start of every epoch, so it has to begin a
        fresh pass over the data each time. Using `yield` here (which turns
        the method into a generator) takes care of that automatically.
        """
        for i in range(self.n_batches_per_epoch):
            yield self._random_batch()

    @property
    def n_samples(self):
        """Number of samples in the full dataset"""
        return int(np.round(self.spike_times.time_support.tot_length() / self.bin_size))

    def sample_batch(self):
        """Generate a sample batch at the start of the time support."""
        if self._sample_batch is None:
            self._sample_batch = self._batch_at_t(self.spike_times.time_support[0, 0])

        return self._sample_batch

    def _batch_at_t(self, t: float):
        """Generate a batch starting at time t."""
        ep = nap.IntervalSet(t, t + self.batch_size)
        counts = self.spike_times.restrict(ep).count(self.bin_size)
        X = self.basis.compute_features(counts)

        return X, counts

    def _random_batch(self):
        """Generate a batch at a random time within the time support."""
        t = self.rng.uniform(
            self.spike_times.time_support[0, 0],
            self.spike_times.time_support[0, 1] - self.batch_size,
        )
        return self._batch_at_t(t)
```

## Create the loader

+++

Spike times will be binned in 5 ms bins and each neuron's firing rate will be predicted from the recent spike history of the whole population.
We build those history features using a NeMoS convolutional basis with a window size of 200 ms.

```{code-cell} ipython3
batch_size = 1  # seconds
bin_size = 5 / 1000  # seconds

basis = nmo.basis.RaisedCosineLogConv(5, window_size=int(0.2 / bin_size))
```

Then use these to create the loader:

```{code-cell} ipython3
loader = PynappleDataLoader(units, basis, batch_size, bin_size)
```

:::{note}
The convolution sets the first 200 ms of data at the start of a processed interval to NaNs. Applying this separately to each 1 second batch essentially "throws away" a fifth of the data.
By not storing the results, it also repeats processing the full data each epoch.

It is generally faster to do this processing by iterating through the whole data once and saving the results to disk, then use [`LazyArrayDataLoader`](nemos.batching.LazyArrayDataLoader) as shown in the [basics section](./stochastic_fit.md).
:::

+++

## Set up logging and run the optimization

+++

We again use the preprocessed full dataset as the test set, but in practice this would be a smaller held-out dataset.

```{code-cell} ipython3
batch_logger = nmo.callbacks.TestLossLogger(
    data["X"],
    data["spike_trains"],
    events={"train_begin", "batch_end"},
)
```

Fit for 30 epochs:

```{code-cell} ipython3
glm = nmo.glm.PopulationGLM(
    solver_name="GradientDescent",
    regularizer="Ridge",
    regularizer_strength=0.01,
    solver_kwargs={"stepsize": 0.01, "acceleration": False},
)
```

```{code-cell} ipython3
glm.stochastic_fit(loader, num_epochs=30, callbacks=batch_logger)
```

```{code-cell} ipython3
plot_loss_history(batch_logger.loss_history)
```
