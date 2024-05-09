"""
# Batch Population GLM

Example batch glm with nemos
"""
# %%

import pynapple as nap
import nemos as nmo
import numpy as np
from scipy.linalg import block_diag

nap.config.nap_config.suppress_conversion_warnings = True

# %%
# Load the data
data = nap.load_file("KSresult.nwb")

units = data['units']
units = units.getby_threshold("rate", 1.0)

# %%
# Instantiate the GLM
glm = nmo.glm.PopulationGLM(
	regularizer=nmo.regularizer.Lasso(
		solver_kwargs={"stepsize":0.001},
		regularizer_strength=0.0001
		)
	)

# %%
# Define a connectivity matrix
glm.feature_mask = np.ones_like(glm.coef_) - block_diag(*([np.ones((5, 1))]*len(units)))

# %%
# Instantiate the basis
basis = nmo.basis.RaisedCosineBasisLog(5, mode="conv", window_size=100)

# %%
# Define a batch size 
batch_size = 2 # second

# %%
# Define the batch function
def batcher():
	# Grab a random time within the time support
	t = np.random.uniform(units.time_support[0,0], units.time_support[0,1]-batch_size)

	# Bin the spike train in a 1s batch
	ep = nap.IntervalSet(t, t+batch_size)
	counts = units.restrict(ep).count(0.001) # count in 1 ms bins

	# Convolve
	X = basis.compute_features(counts)

	# Return X and counts
	return X, counts

# %%
# Initialize the GLM
params, state = glm.initialize_solver(*batcher())

# %%
# Run the batch
for i in range(200):
	print(i)
	params, state = glm.update(params, state, *batcher())

# %%
# Plot the weights
W = glm.coef_.reshape(len(units), basis.n_basis_funcs, len(units))
Wm = np.mean(W, 1)

import matplotlib.pyplot as plt
plt.figure(figsize = (15, 5))
plt.subplot(121)
plt.plot(units.to_tsd().get(0, 2), '.')
plt.subplot(122)
plt.imshow(Wm, cmap = 'jet')
plt.show()

