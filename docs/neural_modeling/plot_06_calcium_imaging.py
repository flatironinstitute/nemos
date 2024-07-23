# -*- coding: utf-8 -*-
"""
Fit Calcium Imaging
============


For the example dataset, we will be working with a recording of a freely-moving mouse imaged with a Miniscope (1-photon imaging at 30Hz using the genetically encoded calcium indicator GCaMP6f). The area recorded for this experiment is the postsubiculum - a region that is known to contain head-direction cells, or cells that fire when the animal's head is pointing in a specific direction.

The data were collected by Sofia Skromne Carrasco from the [Peyrache Lab](https://www.peyrachelab.com/).

!!! warning
    To run this notebook locally, please download the [utility functions](https://github.com/flatironinstitute/nemos/tree/main/docs/neural_modeling/examples_utils) in the same folder as the example notebook.


"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pynapple as nap
from examples_utils import data, plotting
from sklearn.linear_model import LinearRegression

import nemos as nmo

# %%
# configure plots
plt.style.use("examples_utils/nemos.mplstyle")


# %%
# ## Data Streaming
#
# Here we load the data from OSF. The data is a NWB file.

path = data.download_data(
    "A0670-221213.nwb", "https://osf.io/sbnaw/download", "../data"
)


# %%
# ***
# ## pynapple preprocessing
# 
# Now that we have the file, let's load the data. The NWB file contains multiple entries.
data = nap.load_file(path)
print(data)

# %%
# In the NWB file, the calcium traces are saved the RoiResponseSeries field. Let's save them in a variable called 'transients' and print it.

transients = data['RoiResponseSeries']
print(transients)

# %%
# `transients` is a `TsdFrame`. Each column contains the activity of one neuron.
# 
# The mouse was recorded for a 20 minute recording epoch as we can see from the `time_support` property of the `transients` object.

ep = transients.time_support
print(ep)


# %%
# There are a few different ways we can explore the data. First, let's inspect the raw calcium traces for neurons 4 and 35 for the first 250 seconds of the experiment.

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(transients[:, 4].get(0,250))
ax[0].set_ylabel("Firing rate (Hz)")
ax[0].set_title("Trace 4")
ax[0].set_xlabel("Time(s)")
ax[1].plot(transients[:, 35].get(0,250))
ax[1].set_title("Trace 35")
ax[1].set_xlabel("Time(s)")
plt.tight_layout()

# %%
# You can see that the calcium signals are both nonnegative, and noisy. One (neuron 4) has much higher SNR than the other. We cannot typically resolve individual action potentials, but instead see slow calcium fluctuations that result from an unknown underlying electrical signal (estimating the spikes from calcium traces is known as _deconvolution_ and is beyond the scope of this demo). 

# %%
# We can also plot tuning curves, plotting mean calcium activity as a function of head direction, using the function `compute_1d_tuning_curves_continuous`.
# Here `data['ry']` is a `Tsd` that contains the angular head-direction of the animal between 0 and 2$\pi$.

tcurves = nap.compute_1d_tuning_curves_continuous(transients, data['ry'], 120)



# %%
# The function returns a pandas DataFrame. Let's plot the tuning curves for neurons 4 and 35.

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(tcurves.iloc[:, 4])
ax[0].set_xlabel("Angle (rad)")
ax[0].set_ylabel("Firing rate (Hz)")
ax[0].set_title("Trace 4")
ax[1].plot(tcurves.iloc[:, 35])
ax[1].set_xlabel("Angle (rad)")
ax[1].set_title("Trace 35")
plt.tight_layout()

# %%
# As a first processing step, let's bin the calcium traces to a 100ms resolution.

Y = transients.bin_average(0.1, ep)

# %% 
# We can visualize the downsampled transients for the first 50 seconds of data.
plt.figure()
plt.plot(transients[:,0].get(0, 50), linewidth=5, label="30 Hz")
plt.plot(Y[:,0].get(0, 50), '--', linewidth=2, label="10 Hz")
plt.xlabel("Time (s)")
plt.ylabel("Fluorescence")
plt.legend()
plt.show()

# %%
# The downsampling did not destroy the fast transient dynamics, so seems fine to use. We can now move on to using NeMoS to fit a model.

# %%
# ## Basis instantiation
# 
# We can define a cyclic-BSpline for capturing the encoding of the heading angle, and a log-spaced raised cosine basis for the coupling filters between neurons. Note that we are not including a self-coupling (spike history) filter, because in practice we have found it results in overfitting.
#
# We can combine the two bases.

heading_basis = nmo.basis.CyclicBSplineBasis(n_basis_funcs=12)
coupling_basis = nmo.basis.RaisedCosineBasisLog(3, mode="conv", window_size=10)

# %%
# We need to make sure the design matrix will be full-rank by applying identifiability constraints to the Cyclic Bspline, and then combine the bases (the resturned object will be an `AdditiveBasis` object).

heading_basis.identifiability_constraints = True
basis = heading_basis + coupling_basis


# %%
# ## Gamma GLM
#
# Until now, we have been modeling spike trains, and have used a Poisson distribution for the observation model. With calcium traces, things are quite different: we no longer have counts but continuous signals, so the Poisson assumption is no longer appropriate. A Gaussian model is also not ideal since the calcium traces are non-negative. To satisfy these constraints, we will use a Gamma distribution from NeMoS with a soft-plus non linearity.
# !!! note "Non-linearity"
#     Different option are possible. With a soft-plus we are assuming an "additive" effect of the predictors, while an exponential non-linearity assumes multiplicative effects. Deciding which firing rate model works best is an empirical question. You can fit different configurations to see which one capture best the neural activity.

model = nmo.glm.GLM(
    solver_name="LBFGS",
    solver_kwargs=dict(tol=10**-13),
    regularizer="Ridge",
    regularizer_strength=0.02,
    observation_model=nmo.observation_models.GammaObservations(inverse_link_function=jax.nn.softplus)
)


# %%
# We select one neuron to fit later, so remove it from the list of predictors

neu = 4
selected_neurons = jnp.hstack(
    (jnp.arange(0, neu), jnp.arange(neu+1, Y.shape[1]))
)

print(selected_neurons)

# %%
# We need to bring the head-direction of the animal to the same size as the transients matrix.
# We can use the function `bin_average` of pynapple. Notice how we pass the parameter `ep`
# that is the `time_support` of the transients.

head_direction = data['ry'].bin_average(0.1, ep)

# %%
# Let's check that `head_direction` and `Y` are of the same size.
print(head_direction.shape)
print(Y.shape)

# %%
# ## Design matrix
#
# We can now create the design matrix by combining the head-direction of the animal and the activity of all other neurons.
# 
X = basis.compute_features(head_direction, Y[:, selected_neurons])

# %%
# ## Train & test set
#
# Let's create a train epoch and a test epoch to fit and test the models. Since `X` is a pynapple time series, we can create `IntervalSet` objects to restrict them into a train set and test set.

train_ep = nap.IntervalSet(start=X.time_support.start, end=X.time_support.get_intervals_center().t)
test_ep = X.time_support.set_diff(train_ep) # Removing the train_ep from time_support

print(train_ep)
print(test_ep)

# %%
# We can now restrict the `X` and `Y` to create our train set and test set.
Xtrain = X.restrict(train_ep)
Ytrain = Y.restrict(train_ep)

Xtest = X.restrict(test_ep)
Ytest = Y.restrict(test_ep)

# %%
# ## Model fitting
# 
# It's time to fit the model on the data from the neuron we left out.

model.fit(Xtrain, Ytrain[:, neu])


# %%
# ## Model comparison

# %%
# We can compare this to scikit-learn [linear regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).

mdl = LinearRegression()
valid = ~jnp.isnan(Xtrain.d.sum(axis=1)) # Scikit learn does not like nans.
mdl.fit(Xtrain[valid], Ytrain[valid, neu])


# %% 
# We now have 2 models we can compare. Let's predict the activity of the neuron during the test epoch.

yp = model.predict(Xtest)
ylreg = mdl.predict(Xtest) # Notice that this is not a pynapple object.

# %%
# Unfortunately scikit-learn has not adopted pynapple yet. Let's convert `ylreg` to a pynapple object to make our life easier.

ylreg = nap.Tsd(t=yp.t, d=ylreg, time_support = yp.time_support)


# %%
# Let's plot the predicted activity for the first 60 seconds of data.

# mkdocs_gallery_thumbnail_number = 3

ep_to_plot = nap.IntervalSet(test_ep.start+20, test_ep.start+80)

plt.figure()
plt.plot(Ytest[:,neu].restrict(ep_to_plot), "r", label="true", linewidth=2)
plt.plot(yp.restrict(ep_to_plot), "k", label="gamma-nemos", alpha=1)
plt.plot(ylreg.restrict(ep_to_plot), "g", label="linreg-sklearn", alpha=0.5)
plt.legend(loc='best')
plt.xlabel("Time (s)")
plt.ylabel("Fluorescence")
plt.show()

# %%
# While there is some variability in the fit for both models, one advantage of the gamma distribution is clear: the nonnegativity constraint is followed with the data. 
#  This is required for using GLMs to predict the firing rate, which must be positive, in response to simulated inputs. See Peyrache et al. 2018[^1] for an example of simulating activity with a GLM.
# [^1]: Peyrache, A., Schieferstein, N. & Buzsáki, G. Transformation of the head-direction signal into a spatial code. Nat Commun 8, 1752 (2017). https://doi.org/10.1038/s41467-017-01908-3
#
# Another way to compare models is to compute tuning curves. Here we use the function `compute_1d_tuning_curves_continuous` from pynapple.

real_tcurves = nap.compute_1d_tuning_curves_continuous(transients, data['ry'], 120, ep=test_ep)
gamma_tcurves = nap.compute_1d_tuning_curves_continuous(yp, data['ry'], 120, ep=test_ep)
linreg_tcurves = nap.compute_1d_tuning_curves_continuous(ylreg, data['ry'], 120, ep=test_ep)

# %%
# Let's plot them.

plt.figure()
plt.plot(real_tcurves[neu], "r", label="true", linewidth=2)
plt.plot(gamma_tcurves, "k", label="gamma-nemos", alpha=1)
plt.plot(linreg_tcurves, "g", label="linreg-sklearn", alpha=0.5)
plt.legend(loc='best')
plt.ylabel("Fluorescence")
plt.xlabel("Head-direction (rad)")
plt.show()

#%%
# !!! note "Gamma-GLM for Calcium Imaging Analysis"
#     Using Gamma-GLMs for fitting calcium imaging data is still in early stages, and hasn't been through
#     the levels of review and validation that they have for fitting spike data. Users should consider
#     this a relatively unexplored territory, and we hope that we hope that NeMoS will help researchers
#     explore this new space of models.
