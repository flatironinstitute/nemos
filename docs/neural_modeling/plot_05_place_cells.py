# -*- coding: utf-8 -*-

"""
# Fit place cell

!!! warning
    To run this notebook locally, please download the [utility functions](https://github.com/flatironinstitute/nemos/tree/main/docs/neural_modeling/examples_utils) in the same folder as the example notebook.

The data for this example are from [Grosmark, Andres D., and György Buzsáki. "Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences." Science 351.6280 (2016): 1440-1443](https://www.science.org/doi/full/10.1126/science.aad1935).

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
from examples_utils import data, plotting
from scipy.ndimage import gaussian_filter

import nemos as nmo

# configure plots some
plt.style.use("examples_utils/nemos.mplstyle")

# %%
# ## Data Streaming
#
# Here we load the data from OSF. The data is a NWB file.

path = data.download_data(
    "Achilles_10252013.nwb", "https://osf.io/hu5ma/download", "../data"
)

# %%
# ## Pynapple
# We are going to open the NWB file with pynapple

data = nap.load_file(path)

data

# %%
# Let's extract the spike times, the position and the theta phase.

spikes = data["units"]
position = data["position"]
theta = data["theta_phase"]

# %%
# The NWB file also contains the time at which the animal was traversing the linear track. We can use it to restrict the position and assign it as the `time_support` of position.

position = position.restrict(data["trials"])

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
    plt.subplot(gs[i, 0])
    plt.fill_between(pf.index.values, np.zeros(len(pf)), pf[n].values)
    if i < len(spikes) - 1:
        plt.xticks([])
    else:
        plt.xlabel("Position (cm)")
    plt.yticks([])


# %%
# ## Phase precession
#
# In addition to place modulation, place cells are also modulated by the theta oscillation. The phase at which neurons fire is dependant of the position. This phenomen is called "phase precession" (see [J. O’Keefe, M. L. Recce, "Phase relationship between hippocampal place units and the EEG theta rhythm." Hippocampus 3, 317–330 (1993).](https://doi.org/10.1002/hipo.450030307)
#
# Let's compute the response of the neuron as a function of both theta and position. The phase of theta has already been computed but we have to bring it to the same dimension as the position feature. While the position has been sampled at 40Hz, the theta phase has been computed at 1250Hz.
# Later on during the analysis, we will use a bin size of 5 ms for counting the spikes. Since this corresponds to an intermediate frequency between 40 and 1250 Hz, we will bring all the features to 200Hz already.

bin_size = 0.005

theta = theta.bin_average(bin_size, position.time_support)
theta = (theta + 2 * np.pi) % (2 * np.pi)

data = nap.TsdFrame(
    t=theta.t,
    d=np.vstack(
        (position.interpolate(theta, ep=position.time_support).values, theta.values)
    ).T,
    time_support=position.time_support,
    columns=["position", "theta"],
)

print(data)





# %%
# `data` is a `TsdFrame` that contains the position and phase. Before calling `compute_2d_tuning_curves` from pynapple to observe the theta phase precession, we will restrict the analysis to the place field of one neuron. 
#
# There are a lot of neurons but for this analysis, we will focus on one neuron only.
neuron = 175

plt.figure(figsize=(5,3))
plt.fill_between(pf[neuron].index.values, np.zeros(len(pf)), pf[neuron].values)
plt.xlabel("Position (cm)")
plt.ylabel("Firing rate (Hz)")

# %%
# This neurons place field is between 0 and 60 cm within the linear track. Here we will use the `threshold` function of pynapple to quickly compute the epochs for which the animal is within the place field :

within_ep = position.threshold(60.0, method="below").time_support

# %%
# `within_ep` is an `IntervalSet`. We can now give it to `compute_2d_tuning_curves` along with the spiking activity and the position-phase features.

tc_pos_theta, xybins = nap.compute_2d_tuning_curves(spikes, data, 20, within_ep)

# %%
# To show the theta phase precession, we can also display the spike as a function of both position and theta. In this case, we use the function `value_from` from pynapple.

theta_pos_spikes = spikes[neuron].value_from(data, ep = within_ep)

# %%
# Now we can plot everything together :

plt.figure()
gs = plt.GridSpec(2, 2)
plt.subplot(gs[0, 0])
plt.fill_between(pf[neuron].index.values, np.zeros(len(pf)), pf[neuron].values)
plt.xlabel("Position (cm)")
plt.ylabel("Firing rate (Hz)")

plt.subplot(gs[1, 0])
extent = (xybins[0][0], xybins[0][-1], xybins[1][0], xybins[1][-1])
plt.imshow(gaussian_filter(tc_pos_theta[neuron].T, 1), aspect="auto", origin="lower", extent=extent)
plt.xlabel("Position (cm)")
plt.ylabel("Theta Phase (rad)")

plt.subplot(gs[1, 1])
plt.plot(theta_pos_spikes["position"], theta_pos_spikes["theta"], "o", markersize=0.5)
plt.xlabel("Position (cm)")
plt.ylabel("Theta Phase (rad)")

plt.tight_layout()

# %%
# ## Speed modulation
# The speed at which the animal traverse the field is not homogeneous. Does it influence the firing rate of hippocampal neurons? We can compute tuning curves for speed as well as average speed across the maze.
# In the next block, we compute the speed of the animal for each epoch (i.e. crossing of the linear track) by doing the difference of two consecutive position multiplied by the sampling rate of the position.

speed = []
for s, e in data.time_support.values: # Time support contains the epochs
    pos_ep = data["position"].get(s, e)
    speed_ep = np.abs(np.diff(pos_ep)) # Absolute difference of two consecutive points
    speed_ep = np.pad(speed_ep, [0, 1], mode="edge") # Adding one point at the end to match the size of the position array
    speed_ep = speed_ep * data.rate # Converting to cm/s
    speed.append(speed_ep)

speed = nap.Tsd(t=data.t, d=np.hstack(speed), time_support=data.time_support)

# %%
# Now that we have the speed of the animal, we can compute the tuning curves for speed modulation. Here we call pynapple `compute_1d_tuning_curves`:

tc_speed = nap.compute_1d_tuning_curves(spikes, speed, 20)

# %%
# To assess the variabilty in speed when the animal is travering the linear track, we can compute the average speed and estimate the standard deviation. Here we use numpy only and put the results in a pandas `DataFrame`:

bins = np.linspace(np.min(data["position"]), np.max(data["position"]), 20)

idx = np.digitize(data["position"].values, bins)

mean_speed = np.array([np.mean(speed[idx==i]) for i in np.unique(idx)])
std_speed = np.array([np.std(speed[idx==i]) for i in np.unique(idx)])

# %%
# Here we plot the tuning curve of one neuron for speed as well as the average speed as a function of the animal position

plt.figure(figsize=(8, 3))
plt.subplot(121)
plt.plot(bins, mean_speed)
plt.fill_between(
    bins,
    mean_speed - std_speed,
    mean_speed + std_speed,
    alpha=0.1,
)
plt.xlabel("Position (cm)")
plt.ylabel("Speed (cm/s)")
plt.title("Animal speed")
plt.subplot(122)
plt.fill_between(
    tc_speed.index.values, np.zeros(len(tc_speed)), tc_speed[neuron].values
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

position = data["position"]
theta = data["theta"]
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

basis = position_basis * phase_basis + speed_basis

# %%
# The object basis only tell us how each basis covers the feature space. For each timestep, we need to _evaluate_ what are the features value. For that we can call `nemos` basis:

X = basis(position, theta, speed)

# %%
# `X` is our design matrix. For each timestamps, it contains the information about the current position,
# speed and theta phase of the experiment. Notice how passing a pynapple object to the basis
# also returns a `pynapple` object.

print(X)

# %%
# ## Model learning
#
# We can now use the Poisson GLM from nemos to learn the model.

glm = nmo.glm.GLM(
    regularizer=nmo.regularizer.UnRegularized("LBFGS", solver_kwargs=dict(tol=10**-12))
)

glm.fit(X, count)

# %%
# ## Prediction
#
# Let's check first if our model can accurately predict the different tuning curves we displayed above. We can use the `predict` function of nemos and then compute new tuning curves

predicted_rate = glm.predict(X) / bin_size

glm_pf = nap.compute_1d_tuning_curves_continuous(predicted_rate[:, np.newaxis], position, 50)
glm_pos_theta, xybins = nap.compute_2d_tuning_curves_continuous(
    predicted_rate[:, np.newaxis], data, 30, ep=within_ep
)
glm_speed = nap.compute_1d_tuning_curves_continuous(predicted_rate[:, np.newaxis], speed, 30)

# %%
# Let's display both tuning curves together.
fig = plotting.plot_position_phase_speed_tuning(
    pf[neuron], 
    glm_pf[0], 
    tc_speed[neuron], 
    glm_speed[0], 
    tc_pos_theta[neuron],
    glm_pos_theta[0], 
    xybins
    )

# %%
# ## Model selection
#
# While this model captures nicely the features-rate relationship, it is not necessarily the simplest model. Let's construct several models and evaluate their score to determine the best model.
#
# !!! note
#     To shorten this notebook, only a few combinations are tested. Feel free to expand this list.
#

models = {
    "position": position_basis,
    "position + speed": position_basis + speed_basis,
    "position + phase": position_basis + phase_basis,
    "position * phase + speed": position_basis * phase_basis + speed_basis,
}
features = {
    "position": (position,),
    "position + speed": (position, speed),
    "position + phase": (position, theta),
    "position * phase + speed": (position, theta, speed),
}

# %%
# In a loop, we can (1) evaluate the basis, (2), fit the model, (3) compute the score and (4) predict the firing rate. For evaluating the score, we can define a train set of intervals and a test set of intervals.

train_iset = position.time_support[::2] # Taking every other epoch
test_iset = position.time_support[1::2]

# %%
# Let's train all the models.
scores = {}
predicted_rates = {}

for m in models:
    print("1. Evaluating basis : ", m)
    X = models[m](*features[m])

    print("2. Fitting model : ", m)
    # glm = nmo.glm.GLM()
    glm.fit(
        X.restrict(train_iset),
        count.restrict(train_iset),
    )

    print("3. Scoring model : ", m)
    scores[m] = glm.score(
        X.restrict(test_iset),
        count.restrict(test_iset),
        score_type="pseudo-r2-McFadden",
    )

    print("4. Predicting rate")
    predicted_rates[m] = glm.predict(X.restrict(test_iset)) / bin_size


scores = pd.Series(scores)
scores = scores.sort_values()

# %%
# Let's compute scores for each models.

plt.figure(figsize=(5, 3))
plt.barh(np.arange(len(scores)), scores)
plt.yticks(np.arange(len(scores)), scores.index)
plt.xlabel("Pseudo r2")
plt.tight_layout()


# %%
# Some models are doing better than others.
#
# !!! warning
#     A proper model comparison should be done by scoring models repetitively on various train and test set. Here we are only doing partial models comparison for the sake of conciseness.
#
# Alternatively, we can plot some tuning curves to compare each models visually.

tuning_curves = {}

for m in models:
    tuning_curves[m] = {
        "position": nap.compute_1d_tuning_curves_continuous(
            predicted_rates[m][:, np.newaxis], position, 50, ep=test_iset
        ),
        "speed": nap.compute_1d_tuning_curves_continuous(
            predicted_rates[m][:, np.newaxis], speed, 20, ep=test_iset
        ),
    }

# recompute tuning from spikes restricting to the test-set
pf = nap.compute_1d_tuning_curves(spikes, position, 50, ep=test_iset)
tc_speed = nap.compute_1d_tuning_curves(spikes, speed, 20, ep=test_iset)


fig = plt.figure(figsize=(8, 4))
outer_grid = fig.add_gridspec(2, 2)
for i, m in enumerate(models):
    plotting.plot_position_speed_tuning(
        outer_grid[i // 2, i % 2], 
        tuning_curves[m],
        pf[neuron],
        tc_speed[neuron],
        m)

plt.tight_layout()
plt.show()

# %%
# ## Conclusion
#
# Various combinations of features can lead to different results. Feel free to explore more. To go beyond this notebook, you can check the following references :
#
#   - [Hardcastle, Kiah, et al. "A multiplexed, heterogeneous, and adaptive code for navigation in medial entorhinal cortex." Neuron 94.2 (2017): 375-387](https://www.cell.com/neuron/pdf/S0896-6273(17)30237-4.pdf)
#
#   - [McClain, Kathryn, et al. "Position–theta-phase model of hippocampal place cell activity applied to quantification of running speed modulation of firing rate." Proceedings of the National Academy of Sciences 116.52 (2019): 27035-27042](https://www.pnas.org/doi/abs/10.1073/pnas.1912792116)
#
#   - [Peyrache, Adrien, Natalie Schieferstein, and Gyorgy Buzsáki. "Transformation of the head-direction signal into a spatial code." Nature communications 8.1 (2017): 1752.](https://www.nature.com/articles/s41467-017-01908-3)
#
