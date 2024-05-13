"""
# Batch Population GLM

Example batch glm with nemos
"""
# %%

import pynapple as nap
import nemos as nmo
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# jax.config.update("jax_enable_x64", True)
nap.config.nap_config.suppress_conversion_warnings = True

# set random seed
np.random.seed(123)

# %%
# Load the data
data = nap.load_file("../data/KSresult.nwb")

units = data['units']
units = units.getby_threshold("rate", 10.0)

unit_num = 10

# %%
# Instantiate the GLM
step_size_decay = lambda iter_num: jnp.clip(jnp.exp(-0.0005 * iter_num), 0.005, jnp.inf)
glm = nmo.glm.GLM(
	regularizer=nmo.regularizer.Lasso(
		solver_kwargs={"stepsize": 0.2, "acceleration": False},
		regularizer_strength=0.0005
		)
	)

# %%
# Instantiate the basis
ws = 200
basis = nmo.basis.RaisedCosineBasisLog(8, mode="conv", window_size=ws)

# %%
# Define a batch size
batch_size = 30  # second


# %%
# Define the batch function
def batcher():
	# Grab a random time within the time support
	t = np.random.uniform(units.time_support[0, 0], units.time_support[0, 1]-batch_size)

	# Bin the spike train in a 1s batch
	ep = nap.IntervalSet(t, t+batch_size)
	counts = units.restrict(ep).count(0.001)  # count in 1 ms bins

	# Convolve
	X = basis.compute_features(counts)

	# Return X and counts
	return X, counts[:, unit_num]


# %%
# Initialize the GLM
params, state = glm.initialize_solver(*batcher())

# %%
# Run the batch
loglike = np.zeros(1500)
err = np.zeros(1500)
for i in range(1500):

	X_batch, y_batch = batcher()
	params, state = glm.update(params, state, X_batch, y_batch)
	err[i] = state.error
	loglike[i] = glm.score(X_batch, y_batch, score_type="log-likelihood")
	print(i, loglike[i])

# %%
# Plot the weights
# W = glm.coef_.reshape(len(units), basis.n_basis_funcs, len(units))
# responses = np.einsum("ti,nim -> nmt", basis.evaluate_on_grid(50)[1], W)
# Wm = np.mean(W, 1)

# import matplotlib.pyplot as plt
# plt.figure(figsize=(15, 5))
# plt.subplot(121)
# plt.plot(units.to_tsd().get(0, 2), '.')
# plt.subplot(122)
# plt.imshow(Wm, cmap='jet')
# plt.show()

model = nmo.glm.GLM()
model.coef_ = glm.coef_
model.intercept_ = glm.intercept_
model.scale=1.
rate = model.predict(basis.compute_features(units.count(0.001)))
count = units[units.index[unit_num]].count(0.001)
rate_ta = nap.compute_event_trigger_average(units[[units.index[unit_num]]], rate, 0.001, (0, 0.1))
count_ta = nap.compute_event_trigger_average(units[[units.index[unit_num]]], count, 0.001, (0, 0.1))

plt.figure()
sel = rate_ta.t > 0
plt.plot(count_ta[sel, 0], label="counts")
plt.plot(rate_ta[sel, 0], label="batched")
# # fit full model
# srt_resp = np.argsort(np.diag(np.linalg.norm(responses, axis=2)))[::-1]
# model = nmo.glm.PopulationGLM(
# 	regularizer=nmo.regularizer.UnRegularized()
# 	)
# counts = units.count(0.001)
# X = basis.compute_features(counts)
# model.fit(X, counts)
# print(f"score full fit: {model.score(X, counts)}")
# print(f"score batched fit: {glm.score(X, counts, aggregate_sample_scores=lambda: np.mean(axis=0))}")

# rate_full = model.predict(X)
#rate_batched = glm.predict(X)


# rate_ta = nap.compute_event_trigger_average(units, rate_full[:, :1], 0.001, (0, 0.1))
# rate_ta_batch = nap.compute_event_trigger_average(units, rate_batched[:, :1], 0.001, (0, 0.1))
# count_ta = nap.compute_event_trigger_average(units, counts[:, :1], 0.001, (0, 0.1))

# plt.figure()
# sel = rate_ta.t > 0
# plt.plot(count_ta[sel, 0, 0], label="counts")
# plt.plot(rate_ta[sel, 0, 0], label="full")
# plt.plot(rate_ta_batch[sel, 0, 0], label="batched")


# for i in range(5000):
# 	print(i)
# 	if i % 1000 == 0:
# 		rate_ta_batch = nap.compute_event_trigger_average(units, glm.predict(X), 0.001, (0, 0.1))
# 		plt.plot(rate_ta_batch[sel, 0, 0], label=f"batched iter {state.iter_num}")
# 	params, state = glm.update(params, state, *batcher())
#
# plt.legend()

# plt.figure()
# plt.plot(basis.evaluate_on_grid(50)[1])

# # Plot the weights
# Wfull = model.coef_.reshape(len(units), basis.n_basis_funcs, len(units))
# Wmfull = np.mean(Wfull, 1)
# plt.figure(figsize=(15, 5))
# plt.subplot(121)
# plt.plot(units.to_tsd().get(0, 2), '.')
# plt.subplot(122)
# plt.imshow(Wmfull, cmap='jet')
# plt.show()


