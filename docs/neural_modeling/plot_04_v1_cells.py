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

# suppress jax to numpy conversion warning in pynapple
nap.nap_config.suppress_conversion_warnings = True

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
#   neuron's response to a given frame of white noise. this will give an x-pixel by y-pixel
#   filter. implicitly assumes that there's no temporal info: only matters what
#   we've just seen
#
# - could fit spatiotemporal filter. instead of an x by y that we use
#   independently on each frame, fit (x, y, t) over, say 130 msecs. and then
#   fit each of these independently (like in head direction example)
#
# - could reduce the dimensionality by using a bank of k two-dimensional basis
#   functions of shape (x-pixel, y-pixel, k). We can "project" each (x-pixel, y-pixel) stimulus
#   image over the basis by computing the dot product of the two, leaving us with a k-dimensional
#   vector, where k can be much smaller than the original pixel size.
#
# - that's still a lot of parameters! can simplify by assuming that the response is
#   separable: fit a single (x, y) filter and then modulate it over time. this
#   wouldn't catch e.g., direction-selectivity because it assumes that phase
#   preference is constant over time
#
# - could make use of our knowledge of V1 and try to fit a more complex
#   functional form, e.g., a Gabor.
#
# That last one is very non-linear and thus non-convex. We'll do the third one.
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
# We can easily plot this.

fig, axes = plt.subplots(1, len(sta), figsize=(3*len(sta),3))
for i, t in enumerate(sta.t):
    axes[i].imshow(sta[i,0], vmin=np.min(sta), vmax=np.max(sta),
                   cmap='Greys_r')
    axes[i].set_title(str(t)+" s")


# %%
#
# That looks pretty reasonable for a V1 simple cell: localized in space,
# orientation, and spatial frequency. That is, looks Gabor-ish.
# This receptive field gives us the spatial part of the linear response: it
# gives a map of weights that we use for a weighted sum on an image. There are
# multiple ways of performing this operation:
#
# ## Firing rate model
# What we want is to model the log-firing rate as a linear combination of the past
# stimuli $\bm{x}\_t$ over a fixed window, here $x\_t$ is an array representing the
# flattened image of shape `(nm, )`, where n and m are the pixel of the x and y axes
# of the noise stimuli.
# Mathematically, this can be expressed as,
# $$
# \log \mu\_t = \sum \beta\_{i} \bm{x}\_{t-i}
# $$
# Where beta is a vector of coefficients, also of shape `(nm, )`. This is quite a lot of coefficients.
# For example, if you want to use a window of 130ms at 130 ms resolution on a 51x51 image,
# you'll end up with 51^2 x 13 = 33813 coefficients.
# We can use a basis set to reduce the dimensionality: first we create a bank of basis with 51x51 of
# elements 15 elements, reducing the problem to 15^2 x 13 = 2925 parameters.

# define a two-dimensional basis as a product of two "RaisedCosineBasisLinear" basis.
n_bas = 15
basis = nmo.basis.RaisedCosineBasisLinear(n_basis_funcs=n_bas) ** 2

# evaluate the basis on a (51, 51) grid of points
X, Y, basis_eval = basis.evaluate_on_grid(51, 51)

print(basis_eval.shape)



# plot the basis set
fig, axs = plt.subplots(n_bas, n_bas, figsize=(10, 8))
for i in range(n_bas):
    for j in range(n_bas):
        axs[i, j].contourf(X, Y, basis_eval[..., i*n_bas + j])
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])

# %
# Now we can project the stimulus onto the bases.

# project stimulus into the basis
projected_stim = nap.TsdFrame(
    t=stimulus.t,
    d=jnp.einsum("tnm, nmk -> tk", stimulus.d[:], basis_eval),  # read the HDF5 (needed for jax to work)
    time_support=stimulus.time_support
)

# %%
# And additionally, we could jointly model the all-to-all functional connectivity of the V1 population.
#
# !!! note
#     See the [head direction](#plot_02_head_direction.py) tutorial for a detailed
#     overview on how to infer the functional connectivity with a GLM.

# Define a basis for the coupling filters
basis_coupling = nmo.basis.RaisedCosineBasisLog(3, mode="conv", window_size=20)

# %%
# Since the number of parameters and samples is still quite large, we could try a stochastic
# optimization approach, in which we update the parameters using a few random samples from
# the time series of predictors and counts at each iteration. Let's define the parameters
# that we are going to use to sample the features and spike counts.

# the sampling rate for counts and features
bin_size = 0.01  # 10 ms

# the window size used for prediction (130 ms of video)
prediction_window = 0.13  # duration of the window in sec

# number of past frames used for predicting the current firing rate
lags = int(np.ceil(prediction_window / bin_size))

# the duration of the data chunk that we will use at ech iteration
batch_size = 30  # seconds


# define a function that returns the chunk of data from "time" to "time + batch_size"
def batcher(time: float):
    # get the stimulus in a 10 sec interval plus a window
    ep = nap.IntervalSet(start=time, end=time + batch_size + prediction_window)

    # count the spikes of the neuron that we are fitting
    y = spikes[34].count(bin_size, ep)

    # up-sample the projected stimulus to 0.1sec
    x = data.fill_forward(y, projected_stim.restrict(ep))

    # function that shifts tha stimulus of a lag and crops
    def roll_and_crop(x, lag):
        return jnp.roll(x, lag, axis=0)[:-lags]

    # vectorize the function over the lags
    roll = jax.vmap(roll_and_crop, in_axes=(None, 0), out_axes=1)

    # roll and reshape to get the predictors
    features = roll(x.d, -jnp.arange(lags)).reshape(x.shape[0] - lags, -1)

    # convolve the counts with the basis to get the coupling features
    coupling = basis_coupling.compute_features(spikes.count(bin_size, ep))[lags:]

    # concatenate the features and return features and counts
    return np.hstack((coupling, features)), y[lags:]

# %%
# We are now ready to run learn the model parameters.

# instantiate two models: one that will estimate the functional connectivity and one that will not.
model_coupled = nmo.glm.GLM(
    regularizer=nmo.regularizer.Ridge(
        regularizer_strength=0.01,
        solver_kwargs={"stepsize": 0.001, "acceleration": False}
    )
)

model_uncoupled = nmo.glm.GLM(
    regularizer=nmo.regularizer.Ridge(
        regularizer_strength=0.01,
        solver_kwargs={"stepsize": 0.001, "acceleration": False}
    )
)

# initialize the solver
X, Y = batcher(0)

# initialize params coupled
params_cp, state_cp = model_coupled.initialize_solver(X, Y)

# initialize uncoupled (remove the column corresponding to coupling parameters)
n_coupling_coef = len(spikes) * basis_coupling.n_basis_funcs
params_uncp, state_uncp = model_uncoupled.initialize_solver(X[:, n_coupling_coef:], Y)

# %%
# Finally, we can run a loop that grabs a chunk of data and updates the model parameters.

# run the stochastic gradient descent for a few iterations
np.random.seed(123)

for k in range(500):
    if k % 50 == 0:
        print(f"iter {k}")

    #  select a random time point in the recording
    time = np.random.uniform(0, 2400)

    # grab a 30sec batch starting from time.
    X, Y = batcher(time)

    # update the parameters of the coupled model
    params_cp, state_cp = model_coupled.update(params_cp, state_cp, X, Y)

    # update the uncoupled model dropping the column of the features that corresponds to the coupling
    # filters
    params_uncp, state_uncp = model_uncoupled.update(params_uncp, state_uncp, X[:, n_coupling_coef:], Y)


# %%
# We can now plot the receptive fields estimated by the models.

# get the coefficient for the spatiotemporal filters
coeff_coupled = model_coupled.coef_[n_coupling_coef:]
coeff_uncoupled = model_uncoupled.coef_

# weight the basis nby the coefficients to get the estimated receptive fields.
rf_coupled = np.einsum("lk,ijk->lij", coeff_coupled.reshape(lags, -1), basis_eval)
rf_uncoupled = np.einsum("lk,ijk->lij", coeff_uncoupled.reshape(lags, -1), basis_eval)

# compare the receptive fields
mn1, mx1 = rf_uncoupled.min(), rf_uncoupled.max()
mn2, mx2 = rf_coupled.min(), rf_coupled.max()

fig1, axs1 = plt.subplots(1, lags, figsize=(10, 3.5))
fig2, axs2 = plt.subplots(1, lags, figsize=(10, 3.5))
fig1.suptitle("uncoupled model RF")
fig2.suptitle("coupled model RF")
for i in range(lags):
    axs1[i].set_title(f"{prediction_window - i * bin_size:.2} s")
    axs1[i].imshow(rf_uncoupled[i], vmin=mn1, vmax=mx1, cmap="Greys")
    axs1[i].set_xticks([])
    axs1[i].set_yticks([])

    axs2[i].set_title(f"{prediction_window - i * bin_size:.2} s")
    axs2[i].imshow(rf_coupled[i], vmin=mn2, vmax=mx2, cmap="Greys")
    axs2[i].set_xticks([])
    axs2[i].set_yticks([])
fig1.tight_layout()
fig2.tight_layout()