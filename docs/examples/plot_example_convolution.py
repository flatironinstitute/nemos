"""
One-dimensional convolutions.
"""

# %%
# ## Generate synthetic data
# Generate some simulated spike counts.

import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as patches
from neurostatslib.utils import convolve_1d_trials, nan_pad_conv

np.random.seed(10)
ws = 7
# timepoints
T = 100

# number of neurons
n_neurons = 1
spk = np.random.poisson(lam=0.1, size=(n_neurons, T))

# add borders (extreme case, general border effect are represented)
spk[0, 0] = 1
spk[0, 3] = 1
spk[0, -1] = 1
spk[0, -4] = 1


# %%
# Create & plot a filter and perform a convolution in "valid" mode for all trials and neurons.

# create two filters
w = np.vstack([
    np.ones((1, ws)),
    np.hstack([np.arange(ws//2 + ws % 2),
               np.arange(ws//2 - 1, -1, -1)]).reshape(1, -1)
                ])
plt.plot(w.T)

# convolve the spikes:
# the function requires an iterable (one element per trial)
# and returns a list of convolutions
spk_conv = convolve_1d_trials(w, [spk, np.zeros((1,20))])

# %%
# NaN pad for causal, anti-causal and acausal filters.
# A causal filters captures how an event influences the future firing-rate. As an example, we may think of
# a spike event which is followed by an immediate drop in firing rate, known as refractory period.
# On the other hand,
# anti-causal filters can be used to describe how spike activity may lead to an event, for example spiking
# activity in pre-motor cortex may be predictive of the event "movement onset".


# pad according to the causal direction of the filter
spk_causal_utils = np.squeeze(nan_pad_conv(spk_conv, ws, convolution_type="causal")[0])
spk_anticausal_utils = np.squeeze(nan_pad_conv(spk_conv, ws, convolution_type="anti-causal")[0])
spk_acausal_utils = np.squeeze(nan_pad_conv(spk_conv, ws, convolution_type="acausal")[0])


# %%
# Plot the results

# NaN area
rect_causal = patches.Rectangle((0,0), ws, 3, alpha=0.3, color='grey')
rect_anticausal = patches.Rectangle((len(spk[0])-ws, 0), ws, 3, alpha=0.3, color='grey')
rect_acausal_left = patches.Rectangle((0, 0), (ws-1)//2, 3, alpha=0.3, color='grey')
rect_acausal_right = patches.Rectangle((len(spk[0]) - (ws-1)//2, 0), (ws-1)//2, 3, alpha=0.3, color='grey')


plt.figure(figsize=(6,4))

ax = plt.subplot(311)

plt.title('valid + nan-pad')
ax.add_patch(rect_causal)
plt.vlines(np.arange(spk.shape[1]), 0, spk[0], color='k')
plt.plot(np.arange(spk.shape[1]), spk_causal_utils.T)
plt.ylabel('causal')

ax=plt.subplot(312)
ax.add_patch(rect_anticausal)
plt.vlines(np.arange(spk.shape[1]), 0, spk[0], color='k')
plt.plot(np.arange(spk.shape[1]), spk_anticausal_utils.T)
plt.ylabel('anti-causal')

ax=plt.subplot(313)
ax.add_patch(rect_acausal_left)
ax.add_patch(rect_acausal_right)
plt.vlines(np.arange(spk.shape[1]), 0, spk[0], color='k')
plt.plot(np.arange(spk.shape[1]), spk_acausal_utils.T)
plt.ylabel('acausal')
plt.tight_layout()
