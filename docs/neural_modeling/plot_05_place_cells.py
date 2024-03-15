# -*- coding: utf-8 -*-

"""
# Fit Place cell



"""
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
from examples_utils import data, plotting
import pandas as pd
import nemos as nmo

# configure plots some
plt.style.use("examples_utils/nemos.mplstyle")

# %%
# ## Data Streaming
#
# Here we load the data from OSF. The data is a NWB file.

# path = data.download_data("Mouse32-140822.nwb", "https://osf.io/jb2gd/download",
                                         # '../data')

path = "Achilles_10252013.nwb"

# %%
# ## Pynapple
# We are going to open the NWB file with pynapple

data = nap.load_file(path)

data

# %%
# Let's extract the spike times, the position and the theta phase.

spikes = data['units']
position = data['position']
theta = data['theta_phase']

# %%
# The NWB file also contains the time at which the animal was traversing the linear track. We can use it to restrict the position and assign it as the `time_support` of position.

position = position.restrict(data['trials'])

# %%
# The recording contains both inhibitory and excitatory neurons. Here we will focus of the excitatory cells. Neurons have already been labelled before.
spikes = spikes.getby_category("cell_type")["pE"]

# %%
# We can discard the low firing neurons as well.
spikes = spikes.getby_threshold("rate", 0.3)

# %%
# ## Place fields
# Let's plot some data. We start by making place fields i.e firing rate as a function of position.

pf = nap.compute_1d_tuning_curves(spikes, position, 50, position.time_support)

# %%
# Let's do a quick sort of the place fields for display
order = pf.idxmax().sort_values().index.values

# %%
# Here each row is one neuron

plt.figure(figsize=(12, 10))
gs = plt.GridSpec(len(spikes), 1)
for i, n in enumerate(order):
    plt.subplot(gs[i,0])
    plt.fill_between(
        pf.index.values,
        np.zeros(len(pf)),
        pf[n].values
        )
    if i < len(spikes)-1:
        plt.xticks([])
    else:
        plt.xlabel("Position (cm)")
    plt.yticks([])


# %%
# ## Phase precession
#
# In addition to place modulation, place cells are also modulated by the theta oscillation. The phase at which neurons fire is dependant of the position. This phenomen is called "phase precession" (see "J. O’Keefe, M. L. Recce, Phase relationship between hippocampal place units and the EEG theta rhythm. Hippocampus 3, 317–330 (1993)."
#
# Let's compute the response of the neuron as a function of both theta and position. The phase of theta has already been computed but we have to bring it to the same dimension as the position feature. While the position has been sampled at 40Hz, the theta phase has been computed at 1250Hz.
# Later on during the analysis, we will use a bin size of 5 ms for counting the spikes. Since this corresponds to an intermediate frequency between 40 and 1250 Hz, we will bring all the features to 200Hz already.

bin_size = 0.005

theta = theta.bin_average(bin_size, position.time_support)
theta = (theta+2*np.pi)%(2*np.pi)

data = nap.TsdFrame(t = theta.t, 
    d = np.vstack((
        position.interpolate(theta, ep=position.time_support).values,
        theta.values
    )).T,
    time_support = position.time_support,
    columns = ['position', 'theta'])

tc_pos_theta, xybins = nap.compute_2d_tuning_curves(spikes, data, 30, data.time_support)

# %%
# There are a lot of neurons but for this analysis, we will focus on one neuron only.
# print(spikes.keys())

neuron = 175

# %%
# To show the theta phase precession, we can also display the spike as a functin of both position and theta. In this case, we use the function `value_from` from pynapple.

theta_pos_spikes = spikes[neuron].value_from(data)

plt.figure()
gs = plt.GridSpec(2,2)
plt.subplot(gs[0,0])
plt.fill_between(
    pf[neuron].index.values,
    np.zeros(len(pf)),
    pf[neuron].values
    )
plt.xlabel("Position (cm)")
plt.ylabel("Firing rate (Hz)")

plt.subplot(gs[1,0])
extent = (xybins[0][0], xybins[0][-1], xybins[1][0], xybins[1][-1])
plt.imshow(
    tc_pos_theta[neuron].T, 
    aspect='auto',
    origin='lower', 
    extent=extent)
plt.xlabel("Position (cm)")
plt.ylabel("Theta Phase (rad)")

plt.subplot(gs[1,1])
plt.plot(
    theta_pos_spikes['position'], 
    theta_pos_spikes['theta'], 
    'o',
    markersize=0.5
    )
plt.xlabel("Position (cm)")
plt.ylabel("Theta Phase (rad)")

plt.tight_layout()

plt.show()

# %%
# ## Speed modulation
# The speed at which the animal traverse the field is not homogeneous. Does it influence the firing rate of hippocampal neurons. We can compute tuning curves for speed as well as average speed across the maze.

speed = [np.pad(np.abs(np.diff(data['position'].get(s, e))), [0, 1], mode='edge')*data.rate for s, e in data.time_support.values]
speed = nap.Tsd(t=data.t, d=np.hstack(speed), time_support = data.time_support)

tc_speed = nap.compute_1d_tuning_curves(spikes, speed, 20)

bins = np.linspace(np.min(data['position']), np.max(data['position']), 20)
idx = np.digitize(data['position'].values, bins)

speed_mod = pd.DataFrame(
    index = bins,
    data = np.array([[np.mean(speed[idx==i]),np.std(speed[idx==i])] for i in np.unique(idx)]),
    columns = ['mean', 'std']
    )

# %%
# Here we plot the tuning curve of one neuron for speed as well as the average speed as a function of the animal position

plt.figure(figsize=(8,3))
plt.subplot(121)
plt.plot(speed_mod['mean'])
plt.fill_between(
    speed_mod.index.values,
    speed_mod['mean']-speed_mod['std'],
    speed_mod['mean']+speed_mod['std'],
    alpha=0.1)
plt.xlabel("Position (cm)")
plt.ylabel("Speed (cm/s)")
plt.title("Animal speed")
plt.subplot(122)
plt.fill_between(
    tc_speed.index.values,
    np.zeros(len(tc_speed)),
    tc_speed[neuron].values
    )
plt.xlabel("Speed (cm/s)")
plt.ylabel("Firing rate (Hz)")
plt.title("Neuron {}".format(neuron))
plt.tight_layout()

# %%
# This neurons show a strong modulation of firing rate as a function of speed but we can also notice that the animal, on average, accelerates when travering the field. Is the speed tuning we observe a true modulation or spurious correlation caused by traversing the place field at different speed and for different theta phase? We can use `nemos` to model the activity and give the position, the phase and the speed as input variable.
#
# We will use speed, phase and position to model the activity of the neuron.
# All the feature have already been brought to the same dimension thanks to `pynapple`.

position = data['position']
theta = data['theta']
count = spikes[neuron].count(bin_size, data.time_support)

print(position.shape)
print(theta.shape)
print(speed.shape)
print(count.shape)

# %%
# ## Basis evaluation
#
# For each feature, we will use a different set of basis :
#
#   -   position : `nmo.basis.MSplineBasis`
#   -   theta phase : `nmo.basis.CyclicBSplineBasis`
#   -   speed : `nmo.basis.MSplineBasis`

position_basis = nmo.basis.MSplineBasis(n_basis_funcs=10)
phase_basis = nmo.basis.CyclicBSplineBasis(n_basis_funcs=12)
speed_basis = nmo.basis.MSplineBasis(n_basis_funcs=15)

# %%
# In addition, we will consider position and phase to be a joint variable. In `nemos`, we can combine basis by multiplying them and adding them. In this case the final basis object for our model can be made in one line :

basis = position_basis*phase_basis + speed_basis

# %%
# The object basis only tell us how each basis covers the feature space. For each timestep, we need to _evaluate_ what are the features value. We can use the `evaluate` function of `nemos`:

X = basis.evaluate(position, theta, speed)

# %%
# `X` is our design matrix. For each timestamps, it contains the information about the current position, speed and theta phase of the experiment. Notice how passing a pynapple object to `evaluate` also returns a `pynapple` object.

print(X)

# %%
# ## Model learning
#
# We can now use the Poisson GLM from nemos to learn the model.

glm = nmo.glm.GLM(
    regularizer=nmo.regularizer.UnRegularized(
        "LBFGS", 
        solver_kwargs=dict(tol=10**-10)
        )
    )

glm.fit(X[:,np.newaxis,:], count[:,np.newaxis])

# %%
# ## Prediction
# 
# Let's check first if our model can accurately predict the different tuning curves we displayed above. We can use the `predict` function of nemos and then compute new tuning curves

predicted_rate = glm.predict(X[:,np.newaxis,:]) / bin_size

glm_pf = nap.compute_1d_tuning_curves_continuous(predicted_rate, position, 50)
glm_pos_theta, xybins = nap.compute_2d_tuning_curves_continuous(predicted_rate, data, 30)
glm_speed = nap.compute_1d_tuning_curves_continuous(predicted_rate, speed, 30)

# %%
# Let's display both tuning curves together.

plt.figure()
gs = plt.GridSpec(2,2)
plt.subplot(gs[0,0])
plt.plot(pf[neuron])
plt.plot(glm_pf[0], label = 'GLM')
plt.xlabel("Position (cm)")
plt.ylabel("Firing rate (Hz)")
plt.legend()

plt.subplot(gs[0,1])
plt.plot(tc_speed[neuron])
plt.plot(glm_speed[0], label = 'GLM')
plt.xlabel("Speed (cm/s)")
plt.ylabel("Firing rate (Hz)")
plt.legend()

plt.subplot(gs[1,0])
extent = (xybins[0][0], xybins[0][-1], xybins[1][0], xybins[1][-1])
plt.imshow(
    tc_pos_theta[neuron].T, 
    aspect='auto',
    origin='lower', 
    extent=extent)
plt.xlabel("Position (cm)")
plt.ylabel("Theta Phase (rad)")

plt.subplot(gs[1,1])
plt.imshow(
    glm_pos_theta[0].T, 
    aspect='auto',
    origin='lower', 
    extent=extent)
plt.xlabel("Position (cm)")
plt.ylabel("Theta Phase (rad)")
plt.title("GLM")

plt.tight_layout()

# %%
# The GLM captures the features-rate relationship. Yet we can examine the model's coefficients in order to determine the contribution from each different features. We can use the `evaluate_on_grid` method of `nemos`.
# We can look at the speed basis and the position-phase basis separetely.

# Position-phase
n = position_basis.n_basis_funcs*phase_basis.n_basis_funcs

XX, YY, Z = (position_basis*phase_basis).evaluate_on_grid(50, 50)
weight_pos_theta = np.einsum('ijk,k->ij', Z, glm.coef_[0,0:n])

# Speed
samples, eval_basis_speed = speed_basis.evaluate_on_grid(100)
weight_speed = np.dot(eval_basis_speed, glm.coef_[0,-speed_basis.n_basis_funcs:])

# %%
# Let's plot the coefficients of the model

plt.figure(figsize = (6,2))
gs = plt.GridSpec(1,2)
plt.subplot(gs[0,0])
plt.plot(weight_speed)
plt.xlabel("Speed (a.u.)")

plt.subplot(gs[0,1])
plt.imshow(
    weight_pos_theta.T,
    aspect='auto',
    origin='lower')
plt.xlabel("Position (a.u.)")
plt.ylabel("Theta Phase (a.u.)")
plt.tight_layout()

# %%
# While the coefficients for the position-phase basis are very similar to the observed tuning curves, we do not observe the same relationship for the speed coefficients. 
# 
# To compare the contribution of the speed modulation to the prediciton of the model, we can set the average prediction of the position-phase basis and recompute a tuning curves.

predicted_rate_from_speed = np.exp(
        glm.intercept_ +
        np.dot(np.mean((position_basis*phase_basis).evaluate(position, theta), axis=0), 
            glm.coef_[0, 0:n]) +
        np.dot(speed_basis.evaluate(speed), glm.coef_[0, -speed_basis.n_basis_funcs:])
        )/bin_size

glm_only_speed = nap.compute_1d_tuning_curves_continuous(predicted_rate_from_speed[:,None], speed, 30)
glm_pf_only_speed = nap.compute_1d_tuning_curves_continuous(predicted_rate_from_speed[:,None], position, 30)

# %%
# Let's display both prediction and the observed tuning curves

plt.figure(figsize = (10, 4))
gs = plt.GridSpec(1,2)
plt.subplot(gs[0,0])
plt.plot(pf[neuron])
plt.plot(glm_pf[0], label = 'GLM(position*phase + speed)')
plt.plot(glm_pf_only_speed[0], label = 'GLM(speed)')
plt.xlabel("Position (cm)")
plt.ylabel("Firing rate (Hz)")
plt.legend()
plt.subplot(gs[0,1])
plt.plot(tc_speed[neuron])
plt.plot(glm_speed[0], label = 'GLM(position*phase + speed)')
plt.plot(glm_only_speed[0], label = 'GLM(speed)')
plt.xlabel("Speed (cm/s)")
plt.ylabel("Firing rate (Hz)")
# plt.legend()
plt.tight_layout()

# %%
# Using only the speed feature, we observe a strong discrepancy between the observed rate and the predicted rate for both speed modulation and position modulation.

