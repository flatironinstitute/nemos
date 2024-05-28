# # -*- coding: utf-8 -*-
#
"""# Fit V1 cell

The data were collected by Sonica Saraf from the Movshon lab.

!!! warning
    To run this notebook locally, please download the [utility functions](https://github.com/flatironinstitute/nemos/tree/main/docs/neural_modeling/examples_utils) in the same folder as the example notebook.

"""
import jax
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from examples_utils import data

import nemos as nmo
import jax.numpy as jnp

# configure plots some
plt.style.use("examples_utils/nemos.mplstyle")

# %%
# ## Data Streaming
#

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
units = dataset["units"]
stimulus = dataset["whitenoise"]

# %%
# Stimulus is white noise shown at 40 Hz

fig, ax = plt.subplots(1, 1, figsize=(12,4))
ax.imshow(stimulus[0], cmap='Greys_r')
stimulus.shape

# %%
# There are 73 neurons recorded together in V1. To fit the GLM faster, we will focus on one neuron.
print(units)
# this returns TsGroup with one neuron only
spikes = units[units.rate >= 5.0]

# %%
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
# ## Spike-triggered average
#
# Spike-triggered average says: every time our neuron spikes, we store the
# stimulus that was on the screen. for the whole recording, we'll have many of
# these, which we then average to get this STA, which is the "optimal stimulus"
# / spatial filter.
#
# In practice, we do not just the stimulus on screen, but in some window of
# time around it. (it takes some time for info to travel through the eye/LGN to
# V1). Pynapple makes this easy:


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
# # we can easily plot this
#
# fig, axes = plt.subplots(1, len(sta), figsize=(3*len(sta),3))
# for i, t in enumerate(sta.t):
#     axes[i].imshow(sta[i,0], vmin = np.min(sta), vmax = np.max(sta),
#                    cmap='Greys_r')
#     axes[i].set_title(str(t)+" s")


# %%
#
# that looks pretty reasonable for a V1 simple cell: localized in space,
# orientation, and spatial frequency. that is, looks Gabor-ish
#
# To convert this to the spatial filter we'll use for the GLM, let's take the
# average across the bins that look informative: -.125 to -.05

# mkdocs_gallery_thumbnail_number = 3
receptive_field = np.mean(sta.get(-0.125, -0.05), axis=0)[0]
#
# fig, ax = plt.subplots(1, 1, figsize=(4,4))
# ax.imshow(receptive_field, cmap='Greys_r')

# %%
#
# This receptive field gives us the spatial part of the linear response: it
# gives a map of weights that we use for a weighted sum on an image. There are
# multiple ways of performing this operation:

# element-wise multiplication and sum
print((receptive_field * stimulus[0]).sum())
# dot product of flattened versions
print(np.dot(receptive_field.flatten(), stimulus[0].flatten()))

# %%
# ## Firing rate model
# What we want is to model the log-firing rate as a linear combination of the past
# stimuli $\bm{x}_t$ over a fixed window, here $x_t$ is an array representing the
# flattened image of shape `(nm, )`, where n and m are the pixel of the x and y axes
# of the noise stimuli.
# Mathematically, this can be expressed as,
# $$
# \log \mu_t = \sum \beta_{i} \bm{x}_{t-i}
# $$
# Where beta is a vector of coefficients, also of shape `(nm, )`. This is quite a lot of coefficients.
# For example, if you want to use a window of 150ms at 10 ms resolution on a 51x51 image,
# you'll end up with 51^2 x 10 = 26010 coefficients.
# We can use a basis set to reduce the dimensionality: first we create a bank of basis with 51x51
# elements.
n_bas = 14
basis = nmo.basis.RaisedCosineBasisLinear(n_basis_funcs=n_bas)**2

basis_coupling = nmo.basis.RaisedCosineBasisLog(3, mode="conv", window_size=20)
X, Y, basis_eval = basis.evaluate_on_grid(51, 51)

# plot the basis set
fig, axs = plt.subplots(n_bas,n_bas, figsize=(10, 8))
for i in range(n_bas):
    for j in range(n_bas):
        axs[i, j].contourf(X, Y, basis_eval[..., i*n_bas + j])
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])

# %%
# Let's create step-by-step the predictor starting from a single sample.

bin_size = 0.01  # 10 ms

# fix the window size used for prediction (125 ms of video)
prediction_window = 0.13  # duration of the window in sec

# number of lags
lags = int(np.ceil(prediction_window / bin_size))

# batch size in sec
batch_size = 2000

def batcher(time: float):

    # get the stimulus in a 10 sec interval plus a window
    ep = nap.IntervalSet(start=time, end=time + batch_size + prediction_window)

    y = spikes[34].count(0.01, ep)
    # up-sample the stimulus
    x = data.fill_forward(y, stimulus.restrict(ep))

    # vectorize roll
    def roll_and_crop(x, lag):
        return jnp.roll(x, lag, axis=0)[:-lags]

    roll = jax.vmap(roll_and_crop, in_axes=(None, 0), out_axes=0)
    rolled_stim = roll(x.d, -jnp.arange(lags))
    features = jnp.einsum("ltnm, nmk -> tlk", rolled_stim, basis_eval).reshape(rolled_stim.shape[1], -1)
    coupling = basis_coupling.compute_features(spikes.count(0.01, ep))[lags:]
    return rolled_stim, np.hstack((coupling, features)), y[lags:]

model = nmo.glm.GLM(
    regularizer=nmo.regularizer.Ridge(
        regularizer_strength=0.01,
        solver_kwargs={"stepsize": 0.001, "acceleration": False}
    )
)
rstim, X, Y = batcher(0)
n_coupling_coef = len(spikes) * basis_coupling.n_basis_funcs

model = nmo.glm.GLM(
    observation_model=nmo.observation_models.PoissonObservations(inverse_link_function=jax.nn.softplus),
    regularizer=nmo.regularizer.UnRegularized()
)
model.fit(X, Y)

model_unc = nmo.glm.GLM(
    observation_model=nmo.observation_models.PoissonObservations(inverse_link_function=jax.nn.softplus),
    regularizer=nmo.regularizer.UnRegularized()
)
model_unc.fit(X[:, n_coupling_coef:], Y)
# params, state = model.initialize_solver(*batcher(0))
# np.random.seed(123)
# for k in range(500):
#     print(k)
#     time = np.random.uniform(0, 2400)
#     X, Y = batcher(time)
#     params, state = model.update(params, state, X, Y)
#     print(k, np.any(np.isnan(params[0])))
#     if np.any(np.isnan(params[0])):
#         break
#
#
# resp = np.einsum("tj, nmj->tnm", params[0].reshape(13,-1), basis_eval)
#
# # plot the basis set
aa = np.einsum("ltij, t", rstim, Y)
bb = np.einsum("lk, ijk->ijl", np.einsum("tlk,t->lk", X[:, n_coupling_coef:].reshape((-1, lags, n_bas**2)), Y), basis_eval)
cc = np.einsum("lk,ijk->ijl", model.coef_[n_coupling_coef:].reshape(lags, -1), basis_eval)
fig, axs = plt.subplots(3,lags, figsize=(10, 3.5))
mn0,mx0 = aa.min(), aa.max()
mn1,mx1 = bb.min(), bb.max()
mn2,mx2 = cc.min(), cc.max()
for i in range(13):
    axs[0, i].imshow(aa[..., i], vmin=mn0, vmax=mx0)
    axs[0, i].set_xticks([])
    axs[0, i].set_yticks([])
    axs[1, i].imshow(bb[..., i], vmin=mn1, vmax=mx1)
    axs[1, i].set_xticks([])
    axs[1, i].set_yticks([])

    axs[2, i].imshow(cc[..., i], vmin=mn2, vmax=mx2)
    axs[2, i].set_xticks([])
    axs[2, i].set_yticks([])

rate = nap.Tsd(t=Y.t, d=model.predict(X), time_support=Y.time_support)
rate_unc = nap.Tsd(t=Y.t, d=model_unc.predict(X[:, n_coupling_coef:]), time_support=Y.time_support)
cc_model = nap.compute_event_trigger_average(spikes, rate/0.01, binsize=0.01, windowsize=(-8, 0))
cc_model_unc = nap.compute_event_trigger_average(spikes, rate_unc/0.01, binsize=0.01, windowsize=(-8, 0))
cc_spks = nap.compute_event_trigger_average(spikes, Y/0.01, binsize=0.01, windowsize=(-8, 0))

fig, axs = plt.subplots(2,10, figsize=(10, 3.5))

for k in range(len(spikes)):
    axs[k // 10, k % 10].plot(cc_model_unc[:, k], "b")
    axs[k // 10, k % 10].plot(cc_model[:, k], "r")
    axs[k // 10, k % 10].plot(cc_spks[:, k], "k")
plt.tight_layout()


cross_corr = nap.compute_crosscorrelogram(group=spikes, binsize=0.01, windowsize=8)
for k in range(19):
    plt.plot(cross_corr.iloc[:, k])

# The resulting predictor will be of shape `(13, 64)`, i.e. number of frames in the past
# by number of basis.

# # %%
# #
# # When performing this operation on multiple stimuli, things become slightly
# # more complicated. For loops on the above methods would work, but would be
# # slow. Reshaping and using the dot product is one common method, as are
# # methods like `np.tensordot`.
# #
# # We'll use einsum to do this, which is a convenient way of representing many
# # different matrix operations:
#
# filtered_stimulus = np.einsum('t h w, h w -> t', stimulus, receptive_field)
#
# # %%
# #
# # This notation says: take these arrays with dimensions `(t,h,w)` and `(h,w)`
# # and multiply and sum to get an array of shape `(t,)`. This performs the same
# # operations as above.
# #
# # And this remains a pynapple object, so we can easily visualize it!
#
# fig, ax = plt.subplots(1, 1, figsize=(12,4))
# ax.plot(filtered_stimulus)
#
# # %%
# #
# # But what is this? It's how much each frame in the video should drive our
# # neuron, based on the receptive field we fit using the spike-triggered
# # average.
# #
# # This, then, is the spatial component of our input, as described above.
# #
# # ## Preparing data for nemos
# #
# # We'll now use the GLM to fit the temporal component. To do that, let's get
# # this and our spike counts into the proper format for nemos:
#
# # grab spikes from when we were showing our stimulus, and bin at 1 msec
# # resolution
# bin_size = .001
# counts = spikes[34].restrict(filtered_stimulus.time_support).count(bin_size)
# print(counts.rate)
# print(filtered_stimulus.rate)
#
# # %%
# #
# # Hold on, our stimulus is at a much lower rate than what we want for our rates
# # -- in previous neural_modeling, our input has been at a higher rate than our spikes,
# # and so we used `bin_average` to down-sample to the appropriate rate. When the
# # input is at a lower rate, we need to think a little more carefully about how
# # to up-sample.
#
# print(counts[:5])
# print(filtered_stimulus[:5])
#
# # %%
# #
# # What was the visual input to the neuron at time 0.005? It was the same input
# # as time 0. At time 0.0015? Same thing, up until we pass time 0.025017. Thus,
# # we want to "fill forward" the values of our input, and we have pynapple
# # convenience function to do so:
# filtered_stimulus = data.fill_forward(counts, filtered_stimulus)
# filtered_stimulus
#
# # %%
# #
# # We can see that the time points are now aligned, and we've filled forward the
# # values the way we'd like.
# #
# # Now, similar to the [head direction tutorial](../02_head_direction), we'll
# # use the log-stretched raised cosine basis to create the predictor for our
# # GLM:
#
# window_size = 100
# basis = nmo.basis.RaisedCosineBasisLog(8, mode="conv", window_size=window_size)
#
# convolved_input = basis.compute_features(filtered_stimulus)
#
# # %%
# #
# # convolved_input has shape (n_time_pts, n_features * n_basis_funcs), because
# # n_features is the singleton dimension from filtered_stimulus.
# #
# # ## Fitting the GLM
# #
# # Now we're ready to fit the model! Let's do it, same as before:
#
#
# model = nmo.glm.GLM(regularizer=nmo.regularizer.UnRegularized(solver_name="LBFGS"))
# model.fit(convolved_input, counts)
#
# # %%
# #
# # We have our coefficients for each of our 8 basis functions, let's combine
# # them to get the temporal time course of our input:
#
# time, basis_kernels = basis.evaluate_on_grid(window_size)
# time *= bin_size * window_size
# temp_weights = np.einsum('b, t b -> t', model.coef_, basis_kernels)
# plt.plot(time, temp_weights)
# plt.xlabel("time[sec]")
# plt.ylabel("amplitude")
#
# # %%
# #
# # When taken together, the results of the GLM and the spike-triggered average
# # give us the linear component of our LNP model: the separable spatio-temporal
# # filter.
#
#
