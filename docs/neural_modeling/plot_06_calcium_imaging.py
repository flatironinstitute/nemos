# -*- coding: utf-8 -*-
"""
Calcium Imaging
============

Working with calcium data.

For the example dataset, we will be working with a recording of a freely-moving mouse imaged with a Miniscope (1-photon imaging). The area recorded for this experiment is the postsubiculum - a region that is known to contain head-direction cells, or cells that fire when the animal's head is pointing in a specific direction.

The NWB file for the example is hosted on [OSF](https://osf.io/sbnaw). We show below how to stream it.

See the [documentation](https://pynapple-org.github.io/pynapple/) of Pynapple for instructions on installing the package.

This tutorial was made by Sofia Skromne Carrasco and Guillaume Viejo.

"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pynapple as nap
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

import nemos as nmo

# %%
# !!! warning
#     This tutorial uses seaborn and matplotlib for displaying the figure and `scikit-learn`
#     for linear regression.
#
#     You can install all with `pip install matplotlib seaborn tqdm scikit-learn`
#
# mkdocs_gallery_thumbnail_number = 1
#
# Now, import the necessary libraries:


jax.config.update("jax_enable_x64", True)

custom_params = {"axes.spines.right": False, "axes.spines.top": False}

# %%
# ***
# Downloading the data
# ------------------
# First things first: Let's find our file
path = "../data/A0670-221213.nwb"
# if path not in os.listdir("."):
#   r = requests.get(f"https://osf.io/sbnaw/download", stream=True)
#   block_size = 1024*1024
#   with open(path, 'wb') as f:
#     for data in tqdm.tqdm(r.iter_content(block_size), unit='MB', unit_scale=True,
#       total=math.ceil(int(r.headers.get('content-length', 0))//block_size)):
#       f.write(data)

# %%
# ***
# Parsing the data
# ------------------
# Now that we have the file, let's load the data
data = nap.load_file(path)
print(data)

# %%
# Let's save the RoiResponseSeries as a variable called 'transients' and print it
transients = data['RoiResponseSeries']
print(transients)

ep = transients.time_support

# %%
# As usual, we can check the tuning curves and plot it

tc = nap.compute_1d_tuning_curves_continuous(transients, data['ry'], 120)

# %%
# As a first processing step, let's bin the transients to a 100ms resolution.

Y = transients.bin_average(0.1, ep)


# %%
# We can define a cyclic-BSpline for capturing the encoding of the heading angle, and a
# log-spaced raised cosine basis for the coupling filters. We can combine the two basis.

heading_basis = nmo.basis.CyclicBSplineBasis(n_basis_funcs=12)
coupling_basis = nmo.basis.RaisedCosineBasisLog(3, mode="conv", window_size=10)

# make sure the design matrix will be full-rank by applying identifiability constraints
# to the Cyclic Bspline
heading_basis.identifiability_constraints = True

# combine the basis
basis = heading_basis + coupling_basis

# %%
# Let's fit a Gamma GLM with a soft-plus non-linearity.

# initialize a Gamma GLM with Ridge regularization
model = nmo.glm.GLM(
    regularizer=nmo.regularizer.UnRegularized(solver_name="LBFGS", solver_kwargs=dict(tol=10**-13)),
    observation_model=nmo.observation_models.GammaObservations(inverse_link_function=jax.nn.softplus)
)

# select a neuron to fit and remove it from the predictor
neu = 4
selected_neurons = jnp.hstack(
    (jnp.arange(0, neu), jnp.arange(neu+1, Y.shape[1]))
)

# bin the heading angle and compute the features
X = basis.compute_features(data['ry'].bin_average(0.1, ep), Y[:, selected_neurons])

model.fit(X, Y[:, neu])

mdl = LinearRegression()
valid = ~jnp.isnan(X.d.sum(axis=1))
mdl.fit(X[valid], Y[valid, neu])

model_sm = sm.GLM(
    endog=Y.d[valid, neu],
    exog=sm.add_constant(X[valid].d),
    family=sm.families.Gamma(link=sm.families.links.Log()), scale=1.)
res_sm = model_sm.fit()

yp = model.predict(X)
ylreg = mdl.predict(X[valid])
ysm = res_sm.predict(sm.add_constant(X[valid].d))

plt.figure()
plt.plot(Y[valid, neu], "r", label="true")
plt.plot(yp[valid], "k", label="gamma-nemos")
plt.plot(yp[valid].t, ylreg, "g", label="linreg")
plt.plot(yp[valid].t, ysm, "b", label="gamma-sms")
plt.plot()
plt.show()
plt.xlim(800, 1850)
plt.legend()


