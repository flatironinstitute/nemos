# -*- coding: utf-8 -*-

"""
# 2D Basis Functions Plotting

## Defining a 2D Basis Function

To model the response of a neuron to the position of an animal in an arena, we need a 2D model for stimuli.
In this case, the stimuli are the 2D coordinates (x, y) representing the animal's position at each time step.

We can model a response function for multi-dimensional stimuli as the external product of two 1D basis functions.
The response function approximation for the 2D case would be:

$$
f(x, y) \\approx \sum_{ij} \\alpha_{ij} \, a_i (x) b_j(y).
$$

Here, we define this 2D basis function as the multiplication of two 1D basis objects.
"""

import numpy as np
import matplotlib.pyplot as plt
import neurostatslib as nsl

# Define 1D basis objects
a_basis = nsl.basis.MSplineBasis(n_basis_funcs=20, order=3)
b_basis = nsl.basis.MSplineBasis(n_basis_funcs=10, order=2)

# 2D basis function as the product of the two 1D basis objects
prod_basis = a_basis * b_basis

# %%
# Basis Evaluation and Plotting
# ------------------------------------
# We evaluate the product of two 1D basis objects which requires two inputs: 
# one for the x-coordinate and one for the y-coordinate.
# The number of elements of the product basis is the product of the elements of the 1D basis.

# Define a trajectory with 1000 time-points representing the recorded trajectory of the animal
T = 1000
x_coord = np.linspace(0, 10, 1000)
y_coord = np.linspace(0, 50, 1000)

# Evaluate the product basis at the x and y coordinates
eval_basis = prod_basis.evaluate(x_coord, y_coord)

# Output the number of elements and samples of the evaluated basis, 
# as well as the number of elements in the original 1D basis objects
print(f"Product of two 1D splines with {eval_basis.shape[0]} "
      f"basis element and {eval_basis.shape[1]} samples:\n"
      f"\t- a_basis had {a_basis._n_basis_funcs} elements\n\t- b_basis had {b_basis._n_basis_funcs} elements.")

# %%
# 2D Basis Elements Plotting
# -----------------------
# To plot a 2D basis set, we evaluate the basis on a grid of points over the basis function domain.
# We use the `evaluate_on_grid` method of neurostatslib.basis for this.

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
# Define and Evaluate a 2D Additive Basis
# ---------------------------------------
# Just like in the multiplicative case, we can create an additive basis function as the sum of two 1D basis functions.
# We approximate the response function for multi-dimensional stimuli as the sum of two 1D basis functions.
# For the 2D case, we can approximate our response function as follows:
# $$
# f(x,y) \\approx \alpha_i \cdot a_i(x) + \beta_j \cdot b_j(y)
# $$
# Here, we simply add two basis objects, 'a_basis' and 'b_basis', together to get the additive basis.


additive_basis = a_basis + b_basis

# Evaluate the basis functions for the given trajectory.
eval_basis = additive_basis.evaluate(x_coord, y_coord)

print(f"Sum of two 1D splines with {eval_basis.shape[0]} "
      f"basis element and {eval_basis.shape[1]} samples:\n"
      f"\t- a_basis had {a_basis._n_basis_funcs} elements\n\t- b_basis had {b_basis._n_basis_funcs} elements.")

# %%
# Plot the Additive Basis Elements
# --------------------------
# To plot a 2D additive basis set, we need to evaluate the basis on a grid of points over the basis function domain.
# This can be done using the `neurostatslib.basis.evaluate_on_grid` method.

X, Y, Z = additive_basis.evaluate_on_grid(200, 200)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Only plot the support
Z[Z == 0] = np.nan
ax.plot_surface(X, Y, Z[5], cmap="viridis", alpha=0.8)
ax.plot_surface(X, Y, Z[25], cmap="inferno", alpha=0.8)
plt.title(f"Additive basis with {eval_basis.shape[0]} elements")
plt.show()

# %%
# Combining Basis Types
# ---------------------
# Basis objects of different types can be combined through multiplication or addition.
# This feature is particularly beneficial when one of the axes represents a periodic variable
# such as an angle. A practical example would be characterizing responses to position
# in a linear maze and the LFP phase angle.

c_basis = nsl.basis.RaisedCosineBasisLinear(n_basis_funcs=8)

# Multiply or add basis of different types
prod_basis = b_basis * c_basis
add_basis = b_basis + c_basis

print(f"Product of two 1D splines of different type with {eval_basis.shape[0]} "
      f"basis element and {eval_basis.shape[1]} samples:\n"
      f"\t- b_basis is of type {b_basis.__class__}\n\t- c_basis is of type {c_basis.__class__}")

# %%
# N-Dimensional Basis
# -------------------
# If you need to model an N-dimensional response function, you can combine
# N 1D basis objects. However, be aware that the dimension of the evaluated basis function
# will increase exponentially with the number of dimensions.
# For example, evaluating a product of $N$ 1D bases with $T$ samples and $K$ basis element,
# will output a $K^N \times T$ matrix.

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





