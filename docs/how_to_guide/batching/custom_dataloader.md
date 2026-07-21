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

Two features of the data shape the loader. The recording is split into several disjoint intervals -- experiments rarely produce one uninterrupted stretch -- and the design matrix is built by convolution. A batch must therefore be a contiguous span of time that stays within a single recording interval: convolving across the gap between two intervals would mix spike history that is seconds or minutes apart.

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

Load the spike times:

```{code-cell} ipython3
_simulate_and_write_to_disk()
data = nap.load_file(DEFAULT_NWB_PATH)
units = data["units"]  # contains spike times per unit
```

The toy recording is one continuous stretch, so we carve it into three valid intervals with gaps between them to mimic pauses in the recording. Restricting the `TsGroup` sets its `time_support` to these intervals, and the loader will keep every batch inside one of them.

```{code-cell} ipython3
recording = nap.IntervalSet(start=[0.0, 25.0, 42.0], end=[20.0, 40.0, 50.0])
units = units.restrict(recording)
units.time_support
```

## Define `PynappleDataLoader`

Each batch is one contiguous chunk of a single recording interval. We build the batches by splitting each interval into equal-duration chunks -- a chunk never spans a gap -- and visit them in a fresh random order every epoch. Reshuffling the chunk order each epoch decorrelates successive gradient steps and covers every chunk exactly once per pass, while keeping each chunk's samples in their original temporal order, which the convolution depends on.

Convolving a chunk in isolation would leave its first `window_size` bins without preceding history. To avoid discarding that fraction of every batch, we prepend one convolution window of preceding data before convolving -- `window_size` bins, converted to seconds as `window_size * bin_size` and clipped at the interval start so the context never reaches back across a gap. Only the true interval starts, where no prior history exists, keep an unavoidable gap of `window_size` bins.

:::{note}
[`compute_features`](nemos.basis.RaisedCosineLogConv.compute_features) marks bins without a full history window as NaN: the prepended context bins, plus the first `window_size` bins of each recording interval. We leave those rows in each batch and let [`stochastic_fit`](nemos.glm.PopulationGLM.stochastic_fit) drop them in `_preprocess_inputs`, so a batch spanning `[start, end]` reaches the solver as exactly its valid rows, each with correct history.
:::

A [`DataLoader`](nemos.batching.DataLoader) needs three methods:
- `__iter__`: yields `(X_batch, y_batch)` tuples. It is called once at the start of every epoch and must return a fresh iterator each time (using `yield` here makes the method a generator, which does this automatically).
- `n_samples`: the total number of samples in the dataset.
- `sample_batch`: a cheap, deterministic batch used once to initialize the solver state.

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
Prepending the history window means this loses samples only at the start of each recording interval (`window_size` bins each), not at every batch boundary. The loader still re-bins and re-convolves the data on every epoch, so when it fits on disk it is generally faster to preprocess the design matrix once and use [`LazyArrayDataLoader`](nemos.batching.LazyArrayDataLoader), as in the [basics section](./stochastic_fit.md).
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
    solver_kwargs={"stepsize": 0.1, "acceleration": False},
)
```

```{code-cell} ipython3
glm.stochastic_fit(loader, num_epochs=100, callbacks=batch_logger)
```

```{code-cell} ipython3
plot_loss_history(batch_logger.loss_history)
```
