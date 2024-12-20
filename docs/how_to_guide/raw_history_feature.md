---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Fit GLMs For Neural Coupling

GLMs can be used to capture pairwise interactions (couplings) between simultaneously recorded neurons. Let's see how to do this in NeMoS.

:::{admonition} Learn More

You can learn more about GLMs for pairwise interactions by following our [head direction](head-direction-tutorial) tutorial.
:::

(fully_coupled_glm_how_to)=
## Setting up a Fully Coupled GLM

### Raw Spike History as a Feature

We will start by learning how to model the pairwise interaction using the raw spike history. Let's first generate some 
population counts.

```{code-cell} ipython3
import numpy as np
import nemos as nmo
import matplotlib.pyplot as plt

np.random.seed(42)

n_samples = 100
n_neurons = 4

# generate population counts
counts = np.random.poisson(size=(n_samples, n_neurons))

```

Let's use the `HistoryConv` basis to construct a feature matrix capturing the effect of 10 samples of spiking history.

```{code-cell} ipython3

basis = nmo.basis.HistoryConv(window_size=10, label="spike-history")

X = basis.compute_features(counts)
print("Shape of the feature matrix: ", X.shape)

# visualize the output (skip the first window_size of nans)
plt.title("Features", fontsize=16)
plt.imshow(X[10:], cmap="Greys")
plt.xticks([5, 15, 25, 35], [0, 1, 2, 3])
plt.xlabel("Neurons", fontsize=12)
plt.ylabel("Samples", fontsize=12)
plt.show()
```

This is all you need to fit a fully coupled GLM.

```{code-cell} ipython3

model = nmo.glm.PopulationGLM().fit(X, counts)

model.coef_
```

### Reducing Dimensionality With Basis

Alternatively, one can use basis to reduce the number of parameters (useful when you have a history filter the spans hundreds of samples).
Reducing the dimensionality of the features helps to prevent overfitting and speeds up computation, particularly for long history windows.

```{code-cell} ipython3

raised_cos = nmo.basis.RaisedCosineLogConv(3, window_size=3)

X2 = raised_cos.compute_features(counts)

# 12 Features: 3 basis funcs x 4 neurons
print("Shape of the feature matrix: ", X2.shape)

plt.title("Basis Feature Matrix", fontsize=16)
plt.imshow(X2[10:], cmap="Greys")
plt.xticks([1, 4, 7, 10], [0, 1, 2, 3])
plt.xlabel("Neurons", fontsize=12)
plt.ylabel("Samples", fontsize=12)
plt.show()
```

This model can be fit with the usual `model_reduced = nmo.glm.PopulationGLM().fit(X2, counts)`.

## Interpreting the coefficients

The learned model coefficients are stored in a 2D array of shape `(n_features, n_neurons)`. This array concatenates the coefficients representing pairwise couplings along the first dimension. Using the `split_by_feature` method of the basis object, the coefficients can be reshaped into a 3D array of shape `(n_neurons, n_basis_funcs, n_neurons)`, where the first dimension corresponds to sender neurons, the second dimension contains the basis function coefficients, and the third dimension corresponds to receiver neurons.

```{code-cell} ipython3

# get the dictionary (1 key per basis component, here there is single basis component)
split_coef = basis.split_by_feature(model.coef_, axis=0)

print(split_coef["spike-history"].shape)
```

This makes it easy to retrieve the coefficients capturing how the spike history of the neuron `i` affects the firing of neuron `j`, 

```{code-cell} ipython3

sender_neuron_i = 1
receiver_neuron_j = 2

coeff_ij = split_coef["spike-history"][sender_neuron_i, :, receiver_neuron_j]

coeff_ij
```


## Selecting The Connectivity Map

It is also possible select which couplings we want to learn by specifying a boolean mask (a mask of 0s and 1s) of shape `(n_features, n_neurons)`. The mask will be elementwise multiplied to the coefficients, selecting which will be included in the GLM.
For instance, if `mask[i, j] == 0`, the i-th feature won't be included in the GLM fit of the `j-th` neuron.

For example, we can exclude the bi-directional coupling between neuron 0 and 1, and the directional coupling between neuron 2 (sender) and neuron 3 (receiver).

```{code-cell} ipython3

# initialize a neuron x neuron mask
mask = np.ones((n_neurons, n_neurons))

# remove the bi-directional coupling between 0 and 1
mask[0, 1] = 0
mask[1, 0] = 0

# remove the directional coupling from 2 to 3
mask[2, 3] = 0

# repeat over the rows by the number of basis functions
mask = np.repeat(mask, basis.n_basis_funcs, axis=0)
```

We are now ready to fit the GLM masking the coefficients.

```{code-cell} ipython3

model_with_mask = nmo.glm.PopulationGLM(feature_mask=mask).fit(X, counts)

print(model_with_mask.coef_)

```

Check that the coefficient for the connection we removed, are actually null.

```{code-cell} ipython3

split_coef = basis.split_by_feature(model_with_mask.coef_, axis=0)["spike-history"]

# check sender=2 receiver=3
print("Coupling 2 -> 3: ", split_coef[2, :, 3], "\n")

# check that we kept the sender=3, receiver=2
print(f"Coupling 3 -> 2: ", split_coef[3, :, 2])
```




