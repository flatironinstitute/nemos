# -*- coding: utf-8 -*-
"""
Fit Calcium Imaging
============


For the example dataset, we will be working with a recording of a freely-moving mouse imaged with a Miniscope (1-photon imaging). The area recorded for this experiment is the postsubiculum - a region that is known to contain head-direction cells, or cells that fire when the animal's head is pointing in a specific direction.

The data were collected by Sofia Skromne Carrasco from the Peyrache Lab.

!!! warning
    To run this notebook locally, please download the [utility functions](https://github.com/flatironinstitute/nemos/tree/main/docs/neural_modeling/examples_utils) in the same folder as the example notebook.


"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pynapple as nap
from sklearn.linear_model import LinearRegression
from examples_utils import data, plotting

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
# Let's save the RoiResponseSeries as a variable called 'transients' and print it.

transients = data['RoiResponseSeries']
print(transients)

# %%
# `transients` is a `TsdFrame`. Each column contains the activity of one neuron.
# 
# The mouse was recorded for 20 minutes as we can see from the `time_support` property of the `transients` object.

ep = transients.time_support
print(ep)

# %%
# We can compute the tuning curves and plot them by calling the function `compute_1d_tuning_curves_continuous`.
# Here `data['ry']` is a `Tsd` that contains the angular head-direction of the animal between 0 and 2$\pi$.

tcurves = nap.compute_1d_tuning_curves_continuous(transients, data['ry'], 120)


# %%
# The function returns a pandas DataFrame. Let's plot the tuning curve of neuron 4.

plt.figure()
plt.plot(tcurves[4])
plt.xlabel("Angle")
plt.ylabel("Fluorescence")
plt.show()

# %%
# As a first processing step, let's bin the transients to a 100ms resolution.
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
# The downsampling did not destroy the transient dynamic. We can now move on to using nemos to fit a model.

# %%
# ## Basis instantiation
# 
# We can define a cyclic-BSpline for capturing the encoding of the heading angle, and a
# log-spaced raised cosine basis for the coupling filters. We can combine the two basis.

heading_basis = nmo.basis.CyclicBSplineBasis(n_basis_funcs=12)
coupling_basis = nmo.basis.RaisedCosineBasisLog(3, mode="conv", window_size=10)

# %%
# We need to make sure the design matrix will be full-rank by applying identifiability constraints
# to the Cyclic Bspline
heading_basis.identifiability_constraints = True

# Here we combine the basis. The returned object is an `AdditiveBasis` object.
basis = heading_basis + coupling_basis

# %%
# ## Gamma GLM
#
# Since the transients are non-negative, we will use a Gamma distribution from `nemos` with a soft-plus non linearity.
#

model = nmo.glm.GLM(
    regularizer=nmo.regularizer.UnRegularized(solver_name="LBFGS", solver_kwargs=dict(tol=10**-13)),
    observation_model=nmo.observation_models.GammaObservations(inverse_link_function=jax.nn.softplus)
)

# %%
# We select a neuron to fit and remove it from the list of predictors
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
# It's time to fit the model

model.fit(Xtrain, Ytrain[:, neu])


# %%
# ## Model comparison

# %%
# We can compare this to scikit-learn [linear regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).

mdl = LinearRegression()
valid = ~jnp.isnan(Xtrain.d.sum(axis=1)) # Scikit learn does not like nans.
mdl.fit(Xtrain[valid], Ytrain[valid, neu])


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




