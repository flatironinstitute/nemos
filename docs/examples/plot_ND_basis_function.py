# -*- coding: utf-8 -*-

"""
# Multidimensional Basis

## Background

In many cases, it's necessary to model the response of a neuron to multiple different inputs
(such as velocity, spatial position, LFP phase, etc.). Such response functions can often be expressed as a linear
combination of some multidimensional basis elements.

In this document, we introduce two strategies for defining a high-dimensional basis function by combining
two lower-dimensional bases. We refer to these strategies as "addition" and "multiplication" of bases,
and the resulting basis objects will be referred to as additive or multiplicative basis respectively.


Consider we have two inputs $\mathbf{x} \in \mathbb{R}^N,\; \mathbf{y}\in \mathbb{R}^M$.
Let's say we've defined two basis functions for these inputs:

- $ [ a_1 (\mathbf{x}), ..., a_k (\mathbf{x}) ] $ for $\mathbf{x}$
- $[b_1 (\mathbf{y}), ..., b_h (\mathbf{y}) ]$ for $\mathbf{y}$.

These basis functions can be combined in the following ways:

1. **Addition:** If we assume that the response function can be adequately described as the sum of the two components, the function is defined as:
   $$
   f(\mathbf{x}, \mathbf{y}) \\approx \sum_{i} \\alpha_{i} \, a_i (\mathbf{x})  + \sum_{j} \\beta_j b_j(\mathbf{y}).
   $$
   The resulting additive basis simply consists of the concatenation of the two basis sets: $[a_1 (\mathbf{x}), ..., a_k (\mathbf{x}),b_1 (\mathbf{y}), ..., b_h (\mathbf{y}) ]$, for a total of $k+h$ basis elements.

2. **Multiplication:** If we expect the response function to capture arbitrary interactions between the inputs, we can approximate an arbitrary response function as the external product of the two bases:
   $$
   f(\mathbf{x}, \mathbf{y}) \\approx \sum_{ij} \\alpha_{ij} \, a_i (\mathbf{x}) b_j(\mathbf{y}).
   $$
   In this case, the resulting basis consists of the $h \cdot k$ products of the individual bases: $[a_1(\mathbf{x})b_1(\mathbf{y}),..., a_k(\mathbf{x})b_h(\mathbf{y})]$.


In the subsequent sections, we will:

1. Demonstrate the definition, evaluation, and visualization of 2D additive and multiplicative bases.
2. Illustrate how to iteratively apply addition and multiplication operations to extend to dimensions beyond two.

## 2D Basis functions

Consider an instance where we want to capture a neuron's response to an animal's position within a given arena.
In this scenario, the stimuli are the 2D coordinates (x, y) that represent the animal's position at each time point.

"""

# %%
# ### 2D Additive Basis
# One way to model the response to our 2D stimuli is to hypothesize that it decomposes into two factors:
# one due to the x-coordinate and another due to the y-coordinate. We can express this relationship as:
# $$
# f(x,y) \\approx \sum_i \alpha_i \cdot a_i(x) + \sum_j \beta_j \cdot b_j(y).
# $$
# Here, we simply add two basis objects, 'a_basis' and 'b_basis', together to define the additive basis.


import numpy as np
import matplotlib.pyplot as plt
import neurostatslib as nsl

# Define 1D basis objects
a_basis = nsl.basis.MSplineBasis(n_basis_funcs=20, order=3)
b_basis = nsl.basis.MSplineBasis(n_basis_funcs=10, order=2)

# Define the 2D additive basis object
additive_basis = a_basis + b_basis

# %%
# Evaluating the additive basis will require two inputs, one for each coordinate.
# The total number of elements of the additive basis will be the sum of the elements of the 1D basis.

# Define a trajectory with 1000 time-points representing the recorded trajectory of the animal
T = 1000

x_coord = np.linspace(0, 10, 1000)
y_coord = np.linspace(0, 50, 1000)

# Evaluate the basis functions for the given trajectory.
eval_basis = additive_basis.evaluate(x_coord, y_coord)

print(f"Sum of two 1D splines with {eval_basis.shape[0]} "
      f"basis element and {eval_basis.shape[1]} samples:\n"
      f"\t- a_basis had {a_basis._n_basis_funcs} elements\n\t- b_basis had {b_basis._n_basis_funcs} elements.")

# %%
# To plot a 2D basis set, we evaluate the basis on a grid of points over the basis function domain.
# We use the `evaluate_on_grid` method for this.

X, Y, Z = additive_basis.evaluate_on_grid(200, 200)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Only plot the support (not required but facilitates visualizing the different basis element)
Z[Z == 0] = np.nan
ax.plot_surface(X, Y, Z[5], cmap="viridis", alpha=0.8)
ax.plot_surface(X, Y, Z[25], cmap="inferno", alpha=0.8)
plt.title(f"Additive basis with {eval_basis.shape[0]} elements")
plt.show()

# %%
# ### Multiplicative Basis Object
#
# If the aim is to capture interactions between the coordinates, the response function can be modeled as the external
# product of two 1D basis functions. The approximation of the response function in this scenario would be:
#
# $$
# f(x, y) \\approx \sum_{ij} \\alpha_{ij} \, a_i (x) b_j(y).
# $$
#
# In this model, we define the 2D basis function as the product of two 1D basis objects.
# This allows the response to capture non-linear and interaction effects between the x and y coordinates.

# 2D basis function as the product of the two 1D basis objects
prod_basis = a_basis * b_basis

# %%
# Again evaluating an input will require 2 inputs.
# The number of elements of the product basis will be the product of the elements of the two 1D bases.

# Evaluate the product basis at the x and y coordinates
eval_basis = prod_basis.evaluate(x_coord, y_coord)

# Output the number of elements and samples of the evaluated basis, 
# as well as the number of elements in the original 1D basis objects
print(f"Product of two 1D splines with {eval_basis.shape[0]} "
      f"basis element and {eval_basis.shape[1]} samples:\n"
      f"\t- a_basis had {a_basis._n_basis_funcs} elements\n\t- b_basis had {b_basis._n_basis_funcs} elements.")

# %%
# Plotting works in the same way as before

X, Y, Z = prod_basis.evaluate_on_grid(200, 200)

# Setup a 3D plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plot only the support
Z[Z == 0] = np.nan
ax.plot_surface(X, Y, Z[50], cmap="viridis", alpha=0.8)
ax.plot_surface(X, Y, Z[83], cmap="rainbow", alpha=0.8)
ax.plot_surface(X, Y, Z[125], cmap="inferno", alpha=0.8)
plt.title(f"Product basis with {eval_basis.shape[0]} elements")
plt.show()

# %%
# !!! info
#     Basis objects of different types can be combined through multiplication or addition.
#     This feature is particularly useful when one of the axes represents a periodic variable and another is non-periodic.
#     A practical example would be characterizing the responses to position
#     in a linear maze and the LFP phase angle.



# %%
# N-Dimensional Basis
# -------------------
# Sometimes it may be useful to model even higher dimensional interactions, for example between the heding direction of
# an animal and its spatial position. In order to model an N-dimensional response function, you can combine
# N 1D basis objects using additions and multiplications.
#
# !!! warning
#     If you multiply basis together, the dimension of the evaluated basis function
#     will increase exponentially with the number of dimensions potentially causing memory errors.
#     For example, evaluating a product of $N$ 1D bases with $T$ samples and $K$ basis element,
#     will output a $K^N \times T$ matrix.


T = 10
n_basis = 8

a_basis = nsl.basis.RaisedCosineBasisLinear(n_basis_funcs=n_basis)
b_basis = nsl.basis.RaisedCosineBasisLinear(n_basis_funcs=n_basis)
c_basis = nsl.basis.RaisedCosineBasisLinear(n_basis_funcs=n_basis)

prod_basis_3 = a_basis * b_basis * c_basis
samples = np.linspace(0, 1, T)
eval_basis = prod_basis_3.evaluate(samples, samples, samples)

print(f"Product of three 1D splines results in {prod_basis_3._n_basis_funcs} "
      f"basis elements.\nEvaluation output of shape {eval_basis.shape}")

# %%
# The evaluation of the product of 3 basis is a 4 dimensional tensor; we can visualize slices of it.

X, Y, W, Z = prod_basis_3.evaluate_on_grid(30, 30, 30)
slices = [1, 27]
basis_elem = {1:224, 27:407}
cmaps = {1:'viridis', 27:'inferno'}
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
for slice in slices:
      X_slice = X[:, :, slice]
      Y_slice = X[:, :, slice]
      Z_slice = Z[:, :, :, slice]
      ax.plot_surface(X_slice, Y_slice, Z_slice[basis_elem[slice]],
                      alpha=0.5,cmap=cmaps[slice])
plt.show()
