---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
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

# Fit V1 cell

The data presented in this notebook was collected by [Sonica Saraf](https://www.cns.nyu.edu/~saraf/) from the [Movshon lab](https://www.cns.nyu.edu/labs/movshonlab/) at NYU.

The notebook focuses on fitting a V1 cell model.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

import nemos as nmo

# configure plots some
plt.style.use(nmo.styles.plot_style)


# utility for filling a time series
def fill_forward(time_series, data, ep=None, out_of_range=np.nan):
    """
    Fill a time series forward in time with data.

    Parameters
    ----------
    time_series:
        The time series to match.
    data: Tsd, TsdFrame, or TsdTensor
        The time series with data to be extend.

    Returns
    -------
    : Tsd, TsdFrame, or TsdTensor
        The data time series filled forward.

    """
    assert isinstance(data, (nap.Tsd, nap.TsdFrame, nap.TsdTensor))

    if ep is None:
        ep = time_series.time_support
    else:
        assert isinstance(ep, nap.IntervalSet)
        time_series.restrict(ep)

    data = data.restrict(ep)
    starts = ep.start
    ends = ep.end

    filled_d = np.full((time_series.t.shape[0], *data.shape[1:]), out_of_range, dtype=data.dtype)
    fill_idx = 0
    for start, end in zip(starts, ends):
        data_ep = data.get(start, end)
        ts_ep = time_series.get(start, end)
        idxs = np.searchsorted(data_ep.t, ts_ep.t, side="right") - 1
        filled_d[fill_idx:fill_idx + ts_ep.t.shape[0]][idxs >= 0] = data_ep.d[idxs[idxs>=0]]
        fill_idx += ts_ep.t.shape[0]
    return type(data)(t=time_series.t, d=filled_d, time_support=ep)
```

## Data Streaming



```{code-cell} ipython3
path = nmo.fetch.fetch_data("m691l1.nwb")
```

## Pynapple
The data have been copied to your local station.
We are gonna open the NWB file with pynapple


```{code-cell} ipython3
dataset = nap.load_file(path)
```

What does it look like?


```{code-cell} ipython3
print(dataset)
```

Let's extract the data.


```{code-cell} ipython3
epochs = dataset["epochs"]
units = dataset["units"]
stimulus = dataset["whitenoise"]
```

Stimulus is white noise shown at 40 Hz


```{code-cell} ipython3
fig, ax = plt.subplots(1, 1, figsize=(12,4))
ax.imshow(stimulus[0], cmap='Greys_r')
stimulus.shape
```

There are 73 neurons recorded together in V1. To fit the GLM faster, we will focus on one neuron.


```{code-cell} ipython3
print(units)
# this returns TsGroup with one neuron only
spikes = units[[34]]
```

How could we predict neuron's response to white noise stimulus?

- we could fit the instantaneous spatial response. that is, just predict
  neuron's response to a given frame of white noise. this will give an x by y
  filter. implicitly assumes that there's no temporal info: only matters what
  we've just seen

- could fit spatiotemporal filter. instead of an x by y that we use
  independently on each frame, fit (x, y, t) over, say 100 msecs. and then
  fit each of these independently (like in head direction example)

- that's a lot of parameters! can simplify by assumping that the response is
  separable: fit a single (x, y) filter and then modulate it over time. this
  wouldn't catch e.g., direction-selectivity because it assumes that phase
  preference is constant over time

- could make use of our knowledge of V1 and try to fit a more complex
  functional form, e.g., a Gabor.

That last one is very non-linear and thus non-convex. we'll do the third one.

in this example, we'll fit the spatial filter outside of the GLM framework,
using spike-triggered average, and then we'll use the GLM to fit the temporal
timecourse.

## Spike-triggered average

Spike-triggered average says: every time our neuron spikes, we store the
stimulus that was on the screen. for the whole recording, we'll have many of
these, which we then average to get this STA, which is the "optimal stimulus"
/ spatial filter.

In practice, we do not just the stimulus on screen, but in some window of
time around it. (it takes some time for info to travel through the eye/LGN to
V1). Pynapple makes this easy:


```{code-cell} ipython3
sta = nap.compute_event_trigger_average(spikes, stimulus, binsize=0.025,
                                        windowsize=(-0.15, 0.0))
```

sta is a [`TsdTensor`](https://pynapple.org/generated/pynapple.TsdTensor.html), which gives us the 2d receptive field at each of the
time points.


```{code-cell} ipython3
sta
```

We index into this in a 2d manner: row, column (here we only have 1 column).


```{code-cell} ipython3
sta[1, 0]
```

we can easily plot this


```{code-cell} ipython3
fig, axes = plt.subplots(1, len(sta), figsize=(3*len(sta),3))
for i, t in enumerate(sta.t):
    axes[i].imshow(sta[i,0], vmin = np.min(sta), vmax = np.max(sta),
                   cmap='Greys_r')
    axes[i].set_title(str(t)+" s")
```

that looks pretty reasonable for a V1 simple cell: localized in space,
orientation, and spatial frequency. that is, looks Gabor-ish

To convert this to the spatial filter we'll use for the GLM, let's take the
average across the bins that look informative: -.125 to -.05


```{code-cell} ipython3
# mkdocs_gallery_thumbnail_number = 3
receptive_field = np.mean(sta.get(-0.125, -0.05), axis=0)[0]

fig, ax = plt.subplots(1, 1, figsize=(4,4))
ax.imshow(receptive_field, cmap='Greys_r')
```

```{code-cell} ipython3
:tags: [hide-input]

# save image for thumbnail
from pathlib import Path
import os

root = os.environ.get("READTHEDOCS_OUTPUT")
if root:
   path = Path(root) / "html/_static/thumbnails/tutorials"
# if local store in assets
else:
   path = Path("../_build/html/_static/thumbnails/tutorials")
 
# make sure the folder exists if run from build
if root or Path("../assets/stylesheets").exists():
   path.mkdir(parents=True, exist_ok=True)

if path.exists():
  fig.savefig(path / "plot_04_v1_cells.svg")
```

This receptive field gives us the spatial part of the linear response: it
gives a map of weights that we use for a weighted sum on an image. There are
multiple ways of performing this operation:


```{code-cell} ipython3
# element-wise multiplication and sum
print((receptive_field * stimulus[0]).sum())
# dot product of flattened versions
print(np.dot(receptive_field.flatten(), stimulus[0].flatten()))
```

When performing this operation on multiple stimuli, things become slightly
more complicated. For loops on the above methods would work, but would be
slow. Reshaping and using the dot product is one common method, as are
methods like `np.tensordot`.

We'll use einsum to do this, which is a convenient way of representing many
different matrix operations:


```{code-cell} ipython3
filtered_stimulus = np.einsum('t h w, h w -> t', stimulus, receptive_field)
```

This notation says: take these arrays with dimensions `(t,h,w)` and `(h,w)`
and multiply and sum to get an array of shape `(t,)`. This performs the same
operations as above.

And this remains a pynapple object, so we can easily visualize it!


```{code-cell} ipython3
fig, ax = plt.subplots(1, 1, figsize=(12,4))
ax.plot(filtered_stimulus)
```

But what is this? It's how much each frame in the video should drive our
neuron, based on the receptive field we fit using the spike-triggered
average.

This, then, is the spatial component of our input, as described above.

## Preparing data for NeMoS

We'll now use the GLM to fit the temporal component. To do that, let's get
this and our spike counts into the proper format for NeMoS:


```{code-cell} ipython3
# grab spikes from when we were showing our stimulus, and bin at 1 msec
# resolution
bin_size = .001
counts = spikes[34].restrict(filtered_stimulus.time_support).count(bin_size)
print(counts.rate)
print(filtered_stimulus.rate)
```

Hold on, our stimulus is at a much lower rate than what we want for our rates
-- in previous tutorials, our input has been at a higher rate than our spikes,
and so we used `bin_average` to down-sample to the appropriate rate. When the
input is at a lower rate, we need to think a little more carefully about how
to up-sample.


```{code-cell} ipython3
print(counts[:5])
print(filtered_stimulus[:5])
```

What was the visual input to the neuron at time 0.005? It was the same input
as time 0. At time 0.0015? Same thing, up until we pass time 0.025017. Thus,
we want to "fill forward" the values of our input, and we have pynapple
convenience function to do so:


```{code-cell} ipython3
filtered_stimulus = fill_forward(counts, filtered_stimulus)
filtered_stimulus
```

We can see that the time points are now aligned, and we've filled forward the
values the way we'd like.

Now, similar to the [head direction tutorial](plot_02_head_direction), we'll
use the log-stretched raised cosine basis to create the predictor for our
GLM:


```{code-cell} ipython3
window_size = 100
basis = nmo.basis.RaisedCosineLogConv(8, window_size=window_size)

convolved_input = basis.compute_features(filtered_stimulus)
```

convolved_input has shape (n_time_pts, n_features * n_basis_funcs), because
n_features is the singleton dimension from filtered_stimulus.

## Fitting the GLM

Now we're ready to fit the model! Let's do it, same as before:


```{code-cell} ipython3
model = nmo.glm.GLM()
model.fit(convolved_input, counts)
```

We have our coefficients for each of our 8 basis functions, let's combine
them to get the temporal time course of our input:


```{code-cell} ipython3
time, basis_kernels = basis.evaluate_on_grid(window_size)
time *= bin_size * window_size
temp_weights = np.einsum('b, t b -> t', model.coef_, basis_kernels)
plt.plot(time, temp_weights)
plt.xlabel("time[sec]")
plt.ylabel("amplitude")
```

When taken together, the results of the GLM and the spike-triggered average
give us the linear component of our LNP model: the separable spatio-temporal
filter.
