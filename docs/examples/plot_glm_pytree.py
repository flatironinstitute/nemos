"""# FeaturePytree example

This small example notebook shows how to use our custom FeaturePytree objects
instead of arrays to represent the design matrix. It will show that these two
representations are equivalent.

This demo will fit the Poisson-GLM to some synthetic data. We will first show
the simple case, with a single neuron receiving some input. We will then show a
two-neuron system, to demonstrate how FeaturePytree can make it easier to
separate examine separate types of inputs.

First, let's generate our synthetic single-neuron data.

"""

import jax
import jax.numpy as jnp
import numpy as np
import nemos as nmo

# enable float64 precision (optional)
jax.config.update("jax_enable_x64", True)
np.random.seed(111)

# random design tensor. Shape (n_time_points, n_neurons, n_features).
X = 0.5*np.random.normal(size=(100, 1, 5))

# log-rates & weights, shape (n_neurons, ) and (n_neurons, n_features) respectively.
b_true = np.zeros((1, ))
w_true = np.random.normal(size=(1, 5))
# sparsify weights
w_true[0, 1:4] = 0.

# generate counts
rate = jax.numpy.exp(jax.numpy.einsum("ik,tik->ti", w_true, X) + b_true[None, :])
spikes = np.random.poisson(rate)

# %%
# ## FeaturePytrees
#
# A FeaturePytree is a custom nemos object used to represent design matrices,
# GLM coefficients, and other similar variables. It is a simple
# [pytree](https://jax.readthedocs.io/en/latest/pytrees.html), a dictionary
# with strings as keys and arrays as values. These arrays must all have the
# same number of elements along the first dimension, which represents the time
# points, but can have different numbers of elements along the other dimensions
# (and even different numbers of dimensions).

example_pytree = nmo.pytrees.FeaturePytree(feature_0=np.random.normal(size=(100, 1, 2)),
                                           feature_1=np.random.normal(size=(100, 2)),
                                           feature_2=np.random.normal(size=(100, 5)))
example_pytree

# %%
#
# FeaturePytrees can be indexed into like dictionary, so we can grab a
# single one of their features:

example_pytree['feature_0'].shape

# %%
#
# We can grab the number of time points by getting the length or using the
# `shape` attribute

print(len(example_pytree))
print(example_pytree.shape)

# %%
#
# We can new features after initialization, as long as they have the same
# number of time points.

example_pytree['feature_3'] = np.zeros((100, 2, 4))

# %%
#
# However, if we try to add a new feature with the wrong number of time points,
# we'll get an exception:

try:
    example_pytree['feature_4'] = np.zeros((99, 2, 4))
except ValueError as e:
    print(e)

# %%
#
# Similarly, if we try to add a feature that's not an array:

try:
    example_pytree['feature_4'] = "Strings are very predictive"
except ValueError as e:
    print(e)

# %%
#
# FeaturePytrees are intended to be used with
# [jax.tree_map](https://jax.readthedocs.io/en/latest/_autosummary/jax.tree_util.tree_map.html),
# a useful function for performing computations on arbitrary pytrees,
# preserving their structure.

# %%
# We can map lambda functions:
mapped = jax.tree_map(lambda x: x**2, example_pytree)
print(mapped)
mapped['feature_1']
# %%
# Or functions from jax or numpy that operate on arrays:
mapped = jax.tree_map(jnp.exp, example_pytree)
print(mapped)
mapped['feature_1']
# %%
# We can change the dimensionality of our pytree:
mapped = jax.tree_map(lambda x: jnp.mean(x, axis=-1), example_pytree)
print(mapped)
mapped['feature_1']
# %%
# Or the number of time points:
mapped = jax.tree_map(lambda x: jnp.mean(x, axis=-1), example_pytree)
print(mapped)
mapped['feature_1']
# %%
#
# If we map something whose output cannot be a FeaturePytree (because its
# values are scalars or non-arrays), we return a dictionary of arrays instead:
print(jax.tree_map(jnp.mean, example_pytree))
print(jax.tree_map(lambda x: x.shape, example_pytree))
# %%
#
# These properties make FeaturePytrees useful for representing design matrices
# and similar objects for the GLM, which we'll see in the next section.
#
# ## FeaturePytrees and GLM
#
# Let's take the design matrix that we used to generate our synthetic data and
# turn it into a FeaturePytree. To start, we'll make the simplest possible
# FeaturePytree: a single feature.

design_matrix = nmo.pytrees.FeaturePytree(stimulus=X)

# %%
#
# We can pass this variable to the GLM object just as we would the array, and
# nemos will fit the model without any problems:

model = nmo.glm.GLM()
model.fit(design_matrix, spikes)

# %%
#
# We can see that the coefficient parameters are represented as a FeaturePytree
# with the same structure as the input `design_matrix` variable. Where each
# element of the FeaturePytree was an array of shape `(n_time_points,
# n_neurons, n_features_i)`, each element of the coefficients is an array of
# shape `(n_neurons, n_features_i)`.

print(model.coef_)
model.coef_['stimulus']

# %%
#
# To compare, let's fit the GLM using the array as our design matrix. You'll
# see that the coefficient parameters are now a regular array, just like `X`.
# Their shapes also have the same relationship: `X` had shape`(n_time_points,
# n_neurons, n_features)`, whereas `coef_` has shape `(n_neurons,
# n_features_i)`.
#
# You'll also notice that the way of interacting the the GLM object is
# identical! As are the coefficients we found!

model = nmo.glm.GLM()
model.fit(X, spikes)
model.coef_

# %%
#
# ## Multiple feature types
#
# Reading the previous section, you might wonder -- why bother? The power of
# using FeaturePytrees to represent your design matrix becomes more apparent
# when we have more features and thus more complicated design matrices.
#
# For this example, let's generate a population with two connected neurons, one
# of which is receiving a square-wave stimulus as input.
