# -*- coding: utf-8 -*-

"""
# Fit Grid Cells population


"""

import jax
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import nemos as nmo
from scipy.ndimage import gaussian_filter

from examples_utils import data

# %%
# ## DATA STREAMING
# 
# Here we load the data from OSF. The data is a NWB file.
#
# <div class="notes">
#   - Stream the data
# </div>

io = data.download_dandi_data("000582", "sub-11265/sub-11265_ses-07020602_behavior+ecephys.nwb",
                                 )

# %%
# ## PYNAPPLE
# 
#
# <div class="notes">
#   - Load the data with pynapple
# </div>

dataset = nap.NWBFile(io.read())

# %%
# Let's see what is in our data
#
# <div class="notes">
#   - Print the data 
# </div>


print(dataset)


# %%
# In this case, the data were used in this [publication](https://www.science.org/doi/full/10.1126/science.1125572).
# We thus expect to find neurons tuned to position and head-direction of the animal. 
# Let's verify that with pynapple first.
# Let's extract the spike times and the position of the animal.
#
# <div class="notes">
#   - extract the spike times and the position of the animal frome the `data` object
# </div>

spikes = dataset["units"]  # Get spike timings
position = dataset["SpatialSeriesLED1"] # Get the tracked orientation of the animal


# %%
# Here we compute quickly the head-direction of the animal from the position of the LEDs
#
# <div class="notes">
#   - compute the head-direction of the animal from `SpatialSeriesLED1` and `SpatialSeriesLED1`
# </div>
diff = dataset['SpatialSeriesLED1'].values - dataset['SpatialSeriesLED2'].values
head_dir = (np.arctan2(*diff.T) + (2*np.pi)) % (2*np.pi)
head_dir = nap.Tsd(dataset['SpatialSeriesLED1'].index, head_dir).dropna()


# %%
# Let's quickly compute some tuning curves for head-direction and spatial position
#
# <div class="notes">
#   - compute the head-direction and position tuning curves
# </div>

hd_tuning = nap.compute_1d_tuning_curves(
    group=spikes, 
    feature=head_dir,
    nb_bins=61, 
    minmax=(0, 2 * np.pi)
    )

pos_tuning, binsxy = nap.compute_2d_tuning_curves(
    group=spikes, 
    features=position, 
    nb_bins=12)



# %%
# Let's plot the tuning curves for each neurons.
# <div class="notes">
#   - plot the tuning curves for each neurons
# </div>
fig = plt.figure(figsize = (12, 4))
gs = plt.GridSpec(2, len(spikes))
for i in range(len(spikes)):
    ax = plt.subplot(gs[0,i], projection='polar')
    ax.plot(hd_tuning.loc[:,i])
    
    ax = plt.subplot(gs[1,i])
    ax.imshow(gaussian_filter(pos_tuning[i], sigma=1))
plt.tight_layout()


# %%
# ## NEMOS {.strip-code}
# It's time to use nemos. 
# Let's try to predict the spikes as a function of position and see if we can generate better tuning curves
# First we start by binning the spike trains in 10 ms bins.
#
# <div class="notes">
#   - bin spike trains in 10 ms bin size
# </div>

bin_size = 0.01 # second
counts = spikes.count(bin_size, ep=position.time_support)

# %%
# We need to interpolate the position to the same time resolution.
# We can still use pynapple for this.
#
# <div class="notes">
#   - interpolate the position to the timestamps of counts using `interpolate` function of pynapple
# </div>

position = position.interpolate(counts)

# %%
# It's time to use nemos
# Let's define a multiplicative basis for position in 2 dimensions.
#
# <div class="notes">
#   - define a basis in 2D using nemos `RaisedCosineBasisLinear`
# </div>

basis_2d = nmo.basis.RaisedCosineBasisLinear(n_basis_funcs=10) * \
            nmo.basis.RaisedCosineBasisLinear(n_basis_funcs=10)

# %%
# Let's see what a few basis look like 
# Here we evaluate the basis on a 100x100 grid
#
# <div class="notes">
#   - evaluate the basis on a 100x100 grid using `evaluate_on_grid`
# </div>

X, Y, Z = basis_2d.evaluate_on_grid(100, 100)

# %%
# We can plot the evaluated basis.
#
# <div class="notes">
#   - plot the evaluated basis
# </div>

fig, axs = plt.subplots(2,5, figsize=(10, 4))
for k in range(2):
  for h in range(5):
    axs[k][h].contourf(X, Y, Z[:, :, 50+2*(k+h)], cmap='Blues')

plt.tight_layout()

# %%
# Each basis represent a possible position of the animal in an arena whose borders are between 0 and 1.
# To make sure that we evaluate the true position of the animal, we need to rescale the position between 0 and 1.
#
# <div class="notes">
#   - rescale the position between 0 and 1 to match the basis functions
# </div>

position = (position - np.min(position, 0)) / (np.max(position, 0) - np.min(position, 0))

# %%
# Now we can "evaluate" the basis for each position of the animal
#
# <div class="notes">
#   - evaluate the basis for each position of the animal
# </div>

position_basis = basis_2d.evaluate(position['x'], position['y'])

# %%
# Now try to make sense of what it is
print(position_basis.shape)

# %%
# THe shape is (T, N_basis). It means for each time point, we evaluated the value of basis at the particular position 
# Let's plot 5 time steps.


fig = plt.figure(figsize = (12, 4))
gs = plt.GridSpec(2, 5)
xt = np.arange(0, 1000, 200)
cmap = plt.get_cmap("rainbow")
colors = np.linspace(0,1, len(xt))
for cnt, i in enumerate(xt):
    ax = plt.subplot(gs[0, i // 200])
    ax.imshow(position_basis[i].reshape(10, 10).T, origin='lower')
    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_color(cmap(colors[cnt]))
        ax.spines[spine].set_linewidth(3)
    plt.title("T "+str(i))

ax = plt.subplot(gs[1, 2])

ax.plot(position['x'][0:1000], position['y'][0:1000])
for i in range(len(xt)):
    ax.plot(position['x'][xt[i]], position['y'][xt[i]], 'o', color = cmap(colors[i]))

plt.tight_layout()


# %%
# Now we can fit the GLM and see what we get. In this case, we use Ridge for regularization.
# Here we will focus on the last neuron (neuron 7) who has a nice grid pattern
#
# <div class="notes">
#   - instantiate a GLM model with Ridge regularization.
#   - set `regularizer_strength=1.0`
# </div>

model = nmo.glm.GLM(
        regularizer=nmo.regularizer.Ridge(regularizer_strength=1.0, solver_name="LBFGS")
    )

# %%
# Let's fit the model
#
# <div class="notes">
#   - fit the model only to neuron 7 for faster computation
# </div>

neuron = 7

model.fit(np.expand_dims(position_basis,1), counts[:, neuron:neuron+1])


# %%
# We can look at the tuning curves
#
# <div class="notes">
#   - predict the rate and compute a tuning curves using `compute_2d_tuning_curves_continuous` from pynapple
# </div>

rate_pos = model.predict(np.expand_dims(position_basis,1))


# %%
# And compute a tuning curves again

model_tuning, binsxy = nap.compute_2d_tuning_curves_continuous(
    tsdframe=rate_pos * rate_pos.rate,
    features=position,
    nb_bins=12)


# %%
# Let's compare tuning curves
#
# <div class="notes">
#   - compare the tuning curves
# </div>

smooth_pos_tuning = gaussian_filter(pos_tuning[neuron], sigma=1)
smooth_model = gaussian_filter(model_tuning[0], sigma=1)

vmin = min(smooth_pos_tuning.min(), smooth_model.min())
vmax = max(smooth_pos_tuning.max(), smooth_model.max())

fig = plt.figure(figsize = (12, 4))
gs = plt.GridSpec(1, 2)
ax = plt.subplot(gs[0, 0])
ax.imshow(gaussian_filter(pos_tuning[neuron], sigma=1), vmin=vmin,vmax=vmax)
ax = plt.subplot(gs[0, 1])
ax.imshow(gaussian_filter(model_tuning[0], sigma=1), vmin=vmin,vmax=vmax)
plt.tight_layout()


# %%
# The grid does not show at all, we over regularized. We can fix this by tuining the regularization
# strength by means of cross-validation. This can be done through scikit-learn.
#
#
# <div class="notes">
#   - find the better `regularizer_strength` using `sklearn.model_selection.GriSearchCV`
# </div>
from sklearn.model_selection import GridSearchCV
param_grid = dict(regularizer__regularizer_strength=[1e-6, 1e-3, 1])

cls = GridSearchCV(model, param_grid=param_grid)

cls.fit(np.expand_dims(position_basis, 1), counts[:, neuron: neuron+1])

# %%
# Let's get the best estimator and see what we get
# <div class="notes">
#   - instantiate the best model from scikit-learn
# </div>

best_model = cls.best_estimator_

# %%
# Let's predict and compute new tuning curves
#
# <div class="notes">
#   - predict the rate of the best model and compute a 2d tuning curves
# </div>

best_rate_pos = best_model.predict(np.expand_dims(position_basis, 1))

best_model_tuning, binsxy = nap.compute_2d_tuning_curves_continuous(
    tsdframe=best_rate_pos * best_rate_pos.rate,
    features=position,
    nb_bins=12)


# %%
#
# <div class="notes">
#   - compare the 2d tuning curves
# </div>


smooth_best_model = gaussian_filter(best_model_tuning[0], sigma=1)

vmin = min(smooth_pos_tuning.min(), smooth_model.min(), smooth_best_model.min())
vmax = max(smooth_pos_tuning.max(), smooth_model.max(), smooth_best_model.max())

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
plt.suptitle("Rate predictions\n")
axs[0].set_title("Raw Counts")
axs[0].imshow(smooth_pos_tuning, vmin=vmin, vmax=vmax)
axs[1].set_title(f"Ridge - strength: {model.regularizer.regularizer_strength}")
axs[1].imshow(smooth_model, vmin=vmin, vmax=vmax)
axs[2].set_title(f"Ridge - strength: {best_model.regularizer.regularizer_strength}")
axs[2].imshow(smooth_best_model, vmin=vmin, vmax=vmax)
plt.tight_layout()








