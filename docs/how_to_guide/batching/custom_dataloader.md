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

In the previous section we loaded preprocessed arrays (`X` and `spike_trains`) straight into a [`LazyArrayDataLoader`](nemos.batching.LazyArrayDataLoader). Here we start one step earlier, from raw spike times on disk, and build the counts and design matrix on the fly inside the loader.

The spikes are loaded and preprocessed with `pynapple`, which supports lazy, out-of-core loading. The example uses a small simulated dataset, but the same loader scales to large recordings without holding them in memory.

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

jax.config.update("jax_enable_x64", True)

import nemos as nmo

nap.nap_config.suppress_conversion_warnings = True

np.random.seed(123)
```

## The `DataLoader` interface

A [`DataLoader`](nemos.batching.DataLoader) is any object that streams `(X_batch, y_batch)` pairs to [`stochastic_fit`](nemos.glm.PopulationGLM.stochastic_fit). It has to provide three methods:
- `__iter__`: yields `(X_batch, y_batch)` tuples. It is called once at the start of every epoch and must return a fresh iterator each time (using `yield` here makes the method a generator, which does this automatically).
- `n_samples`: the total number of samples in the dataset.
- `sample_batch`: a cheap, deterministic batch used once to initialize the solver state.

Anything that provides these three works. The rest of this page builds one for a specific problem: batching correctly a recording that is split into disjoint temporal intervals.

## Custom `DataLoader` for discontinuous recordings

### Simulating the dataset

We simulate one continuous recording, write it to disk as an NWB file, and load the spike times back with `pynapple`.

```{code-cell} ipython3
_simulate_and_write_to_disk()
data = nap.load_file(DEFAULT_NWB_PATH)
units = data["units"]  # contains spike times per unit
```

The simulation is a single uninterrupted stretch. To reproduce a realistic recording, we define three disjoint intervals -- standing in for the valid recording periods, or the experimental condition of interest -- and restrict the spike times to them. Restricting sets the `TsGroup`'s `time_support` to these intervals.

```{code-cell} ipython3
recording = nap.IntervalSet(start=[0.0, 25.0, 42.0], end=[20.0, 40.0, 50.0])
units = units.restrict(recording)
units.time_support
```

### Batching a fully coupled GLM

The model we want to fit is a fully coupled GLM: each neuron's firing rate is predicted from the recent spike history of the entire population (the [head-direction tutorial](../../tutorials/plot_02_head_direction.md) works through such a model in full). Forming that predictor means convolving the spike counts with a basis to build the design matrix, and it is this convolution that makes batching a discontinuous recording non-trivial: a batch must not convolve across the gap between two intervals, and the batches should be uniform in size.

The simplest way to keep batches from crossing a gap is to build them as contiguous chunks. We split each recording interval into equal-duration chunks -- so a chunk never spans a gap -- and visit them in a random order on every pass over the dataset. Reshuffling the chunk order each pass decorrelates successive gradient steps and covers every chunk exactly once.

Each feature is a convolution that looks back over the previous `window_size` bins, so the first `window_size` bins of a chunk have no history to convolve when the chunk is processed on its own. To give them that history, before convolving we extend the chunk backward by `window_size` bins -- a *context* window of `window_size * bin_size` seconds -- clipped at the recording interval start so it never reaches back across a gap. These context bins exist only to supply history to the chunk's first real bins; they are not training samples themselves. Only the true interval starts, where no earlier data exists, keep an unavoidable gap of `window_size` bins.

:::{note}
[`compute_features`](nemos.basis.RaisedCosineLogConv.compute_features) marks bins without a full history window as NaN: the prepended context bins, plus the first `window_size` bins of each recording interval. We leave those rows in each batch and let [`stochastic_fit`](nemos.glm.PopulationGLM.stochastic_fit) drop them internally.
:::

Putting the pieces together:

```{code-cell} ipython3
class PynappleDataLoader(nmo.batching.DataLoader):
    def __init__(self, spike_times, basis, interval_size, bin_size, shuffle=True, seed=123):
        self.spike_times = spike_times
        self.basis = basis
        self.bin_size = bin_size
        # window length is in bins; the left context to prepend is that many bins, in seconds
        self.context_duration = basis.window_size * bin_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        # Equal-duration chunks built within each recording interval, so none crosses a gap; the
        # sub-interval_size tail of an interval is kept as a final short chunk. Each row is
        # (chunk_start, chunk_end, interval_start): the interval start lets us clip the left
        # context without reaching back into the previous interval.
        self._chunks = np.array([
            (start_chunk, min(start_chunk + interval_size, end), start)
            for start, end in spike_times.time_support.values
            for start_chunk in np.arange(start, end, interval_size)
        ])

    @property
    def n_samples(self):
        """Number of samples in the full dataset."""
        return int(round(self.spike_times.time_support.tot_length() / self.bin_size))

    def _batch(self, start, end, interval_start):
        """Count and convolve one chunk, prepending a window of history for the convolution."""
        start_ext = max(start - self.context_duration, interval_start)
        counts = self.spike_times.count(self.bin_size, ep=nap.IntervalSet(start_ext, end))
        return self.basis.compute_features(counts), counts

    def sample_batch(self):
        """Deterministic first-chunk batch used to initialize the solver state."""
        return self._batch(*self._chunks[0])

    def __iter__(self):
        """Yield one batch per chunk, reshuffling the chunk order each epoch."""
        chunks = self._chunks
        if self.shuffle:
            chunks = chunks[self.rng.permutation(len(chunks))]
        for start, end, interval_start in chunks:
            yield self._batch(start, end, interval_start)
```

## Create the loader

Spikes are binned in 5 ms bins, and each neuron's firing rate is predicted from the recent spike history of the whole population, built with a NeMoS convolutional basis over a 200 ms window. We use 5 s chunks as batches.

```{code-cell} ipython3
bin_size = 5 / 1000  # seconds
interval_size = 5.0  # seconds

basis = nmo.basis.RaisedCosineLogConv(5, window_size=int(0.2 / bin_size))

loader = PynappleDataLoader(units, basis, interval_size, bin_size)
```

:::{note}
Prepending the history window means the loader loses samples only at the start of each recording interval (`window_size` bins each), not at every batch boundary. It does re-bin and re-convolve the data on every pass, though. If the design matrix is larger than RAM but fits on disk, it is faster to compute it once and store it in a memory-mappable format that `pynapple` [loads lazily](https://pynapple.org/user_guide/02_input_output.html) -- Zarr, HDF5, NWB, or similar -- then stream it with [`LazyArrayDataLoader`](nemos.batching.LazyArrayDataLoader), as in the [basics section](./stochastic_fit.md).
:::

## Set up logging and run the optimization

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
    solver_kwargs={"stepsize": 0.05, "acceleration": False},
)
```

```{code-cell} ipython3
glm.stochastic_fit(loader, num_epochs=30, callbacks=batch_logger)
```

```{code-cell} ipython3
plot_loss_history(batch_logger.loss_history)
```
