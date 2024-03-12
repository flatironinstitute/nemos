# # -*- coding: utf-8 -*-
#
"""# Fit V1 cell

## Learning objectives {.keep-text}

 - Learn how to combine GLM with other modeling approach.
 - Review previous background.

"""

import jax
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from examples_utils import data, plotting

import nemos as nmo

# configure plots some
plt.style.use("examples_utils/nemos.mplstyle")

# %%
# ## Data Streaming
#
# Here we load the data from OSF. This data comes from Sonica Saraf, in Tony
# Movshon's lab.

path = data.download_data("m691l1.nwb", "https://osf.io/xesdm/download",
                                         '../data')


# %%
# ## Pynapple
# The data have been copied to your local station.
# We are gonna open the NWB file with pynapple

dataset = nap.load_file(path)

# %%
# What does it look like?
print(dataset)

# %%
# Let's extract the data.
epochs = dataset["epochs"]
spikes = dataset["units"]
stimulus = dataset["whitenoise"]

# %%
# stimulus is white noise shown at 40 Hz
# <div class="notes">
#   - stimulus is white noise shown at 40 Hz
#   - white noise is a good stimulus for mapping basic stimulus properties of
#     V1 simple cells
# </div>
fig, ax = plt.subplots(1, 1, figsize=(12,4))
ax.imshow(stimulus[0], cmap='Greys_r')
stimulus.shape

# %%
# There are 73 neurons recorded together in V1. To fit the GLM faster, we will focus on one neuron.
print(spikes)
spikes = spikes[[34]]

# %%
#
# <div class="notes">
#   - goal is to predict the neuron's response to this white noise stimuli
#   - several ways we could do this, what do you think?
# </div>
#
# How could we predict neuron's response to white noise stimulus?
# 
# - we could fit the instantaneous spatial response. that is, just predict
#   neuron's response to a given frame of white noise. this will give an x by y
#   filter. implicitly assumes that there's no temporal info: only matters what
#   we've just seen
#
# - could fit spatiotemporal filter. instead of an x by y that we use
#   independently on each frame, fit (x, y, t) over, say 100 msecs. and then
#   fit each of these independently (like in head direction example)
#
# - that's a lot of parameters! can simplify by assumping that the response is
#   separable: fit a single (x, y) filter and then modulate it over time. this
#   wouldn't catch e.g., direction-selectivity because it assumes that phase
#   preference is constant over time
#
# - could make use of our knowledge of V1 and try to fit a more complex
#   functional form, e.g., a Gabor.
#
# That last one is very non-linear and thus non-convex. we'll do the third one.
#
# in this example, we'll fit the spatial filter outside of the GLM framework,
# using spike-triggered average, and then we'll use the GLM to fit the temporal
# timecourse.
#
# ## Spike-triggered average {.strip-code}
#
# Spike-triggered average says: every time our neuron spikes, we store the
# stimulus that was on the screen. for the whole recording, we'll have many of
# these, which we then average to get this STA, which is the "optimal stimulus"
# / spatial filter.
#
# In practice, we do not just the stimulus on screen, but in some window of
# time around it. (it takes some time for info to travel through the eye/LGN to
# V1). Pynapple makes this easy:
#
# <div class="notes">
#   - compute spike-triggered average to visualize receptive field.
# </div>

sta = nap.compute_event_trigger_average(spikes, stimulus, binsize=0.025,
                                        windowsize=(-0.15, 0.0))
# %%
#
# sta is a `TsdTensor`, which gives us the 2d receptive field at each of the
# time points.
sta

# %%
#
# We index into this in a 2d manner: row, column (here we only have 1 column).
sta[1, 0]

# %%
# we can easily plot this
#
# <div class="notes">
#   - visualize spike-triggered average and decide on our spatial filter.
# </div>

fig, axes = plt.subplots(1, len(sta), figsize=(3*len(sta),3))
for i, t in enumerate(sta.t):
    axes[i].imshow(sta[i,0], vmin = np.min(sta), vmax = np.max(sta),
                   cmap='Greys_r')
    axes[i].set_title(str(t)+" s")


# %%
#
# that looks pretty reasonable for a V1 simple cell: localized in space,
# orientation, and spatial frequency. that is, looks Gabor-ish
#
# To convert this to the spatial filter we'll use for the GLM, let's take the
# average across the bins that look informative: -.125 to -.05

# mkdocs_gallery_thumbnail_number = 3
receptive_field = np.mean(sta.get(-0.125, -0.05), axis=0)[0]

fig, ax = plt.subplots(1, 1, figsize=(4,4))
ax.imshow(receptive_field, cmap='Greys_r')

# %%
#
# This receptive field gives us the spatial part of the linear response: it
# gives a map of weights that we use for a weighted sum on an image. There are
# multiple ways of performing this operation:
#
# <div class="notes">
#   - use the spike-triggered average to preprocess our visual input.
# </div>

# element-wise multiplication and sum
print((receptive_field * stimulus[0]).sum())
# dot product of flattened versions
print(np.dot(receptive_field.flatten(), stimulus[0].flatten()))

# %%
#
# When performing this operation on multiple stimuli, things become slightly
# more complicated. For loops on the above methods would work, but would be
# slow. Reshaping and using the dot product is one common method, as are
# methods like `np.tensordot`.
#
# We'll use einsum to do this, which is a convenient way of representing many
# different matrix operations:

filtered_stimulus = np.einsum('t h w, h w -> t', stimulus, receptive_field)
# add the extra dimension for feature
filtered_stimulus = np.expand_dims(filtered_stimulus, 1)

# %%
#
# This notation says: take these arrays with dimensions `(t,h,w)` and `(h,w)`
# and multiply and sum to get an array of shape `(t,)`. This performs the same
# operations as above.
#
# And this remains a pynapple object, so we can easily visualize it!

fig, ax = plt.subplots(1, 1, figsize=(12,4))
ax.plot(filtered_stimulus)

# %%
#
# But what is this? It's how much each frame in the video should drive our
# neuron, based on the receptive field we fit using the spike-triggered
# average.
#
# This, then, is the spatial component of our input, as described above.
#
# ## Preparing data for nemos {.strip-code}
#
# We'll now use the GLM to fit the temporal component. To do that, let's get
# this and our spike counts into the proper format for nemos:
#
# <div class="notes">
#   - get `counts` and `filtered_stimulus` into proper shape for nemos
# </div>

# grab spikes from when we were showing our stimulus, and bin at 1 msec
# resolution
bin_size = .001
counts = spikes.restrict(filtered_stimulus.time_support).count(bin_size)
print(counts.rate)
print(filtered_stimulus.rate)

# %%
#
# Hold on, our stimulus is at a much lower rate than what we want for our rates
# -- in previous neural_modeling, our input has been at a higher rate than our spikes,
# and so we used `bin_average` to down-sample to the appropriate rate. When the
# input is at a lower rate, we need to think a little more carefully about how
# to up-sample.
print(counts[:5])
print(filtered_stimulus[:5])

# %%
#
# What was the visual input to the neuron at time 0.005? It was the same input
# as time 0. At time 0.0015? Same thing, up until we pass time 0.025017. Thus,
# we want to "fill forward" the values of our input, and we have pynapple
# convenience function to do so:
filtered_stimulus = data.fill_forward(counts, filtered_stimulus)
filtered_stimulus

# %%
#
# We can see that the time points are now aligned, and we've filled forward the
# values the way we'd like.
#
# Now, similar to the [head direction tutorial](../02_head_direction), we'll
# use the log-stretched raised cosine basis to create the predictor for our
# GLM:
#
# <div class="notes">
#   - Set up the basis and prepare the temporal predictor for the GLM.
# </div>
basis = nmo.basis.RaisedCosineBasisLog(8)
window_size = 100
time, basis_kernels = basis.evaluate_on_grid(window_size)
time *= bin_size * window_size
convolved_input = nmo.utils.convolve_1d_trials(basis_kernels, np.expand_dims(filtered_stimulus,0))[0]
# convolved_input has shape (n_time_pts, n_features, n_basis_funcs), and
# n_features is the singleton dimension from filtered_stimulus, so let's
# squeeze it out:
convolved_input = np.squeeze(convolved_input)
# and, as also described in the head direction tutorial, when doing this we
# need to remove the first window_size time points from the neuron counts and
# the last time point from the convolved input:
counts = counts[window_size:]
convolved_input = convolved_input[:-1]
# and grab the counts for our single neuron
counts = counts[:, :1]

# %%
#
# ## Fitting the GLM {.strip-code}
#
# Now we're ready to fit the model! Let's do it, same as before:
#
# <div class="notes">
#   - Fit the GLM
# </div>
model = nmo.glm.GLM(regularizer=nmo.regularizer.UnRegularized(solver_name="LBFGS"))
model.fit(np.expand_dims(convolved_input, 1), counts)

# %%
#
# We have our coefficients for each of our 8 basis functions, let's combine
# them to get the temporal time course of our input:
#
# <div class="notes">
#   - Examine the resulting temporal filter
# </div>

temp_weights = np.einsum('i b, t b -> t', model.coef_, basis_kernels)
plt.plot(time, temp_weights)

# %%
#
# When taken together, the results of the GLM and the spike-triggered average
# give us the linear component of our LNP model: the separable spatio-temporal
# filter.
#
# ## Further exercises
#
# There's more that could (and should) be done here. First, we should probably
# split our data into separate test and train sets, to see how consistent our
# estimates of the spatial and temporal filters are. Then, using the test and
# train sets, we can:
#
# - try different choices for the spatial receptive field: modify the
#   parameters of the STA, pick one of the time bins directly (instead of
#   averaging), lowpass filter the receptive field (to remove the high
#   frequency noise), manually create or fit a Gabor to match the STA results.
#
# - try different choices for the temporal filter: change basis functions,
#   change the parameters of the basis object.
#
# - try adding regularization to the GLM for fitting the temporal filter.
