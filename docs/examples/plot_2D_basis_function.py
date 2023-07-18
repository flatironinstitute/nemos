# -*- coding: utf-8 -*-

"""
# 2D Basis Functions Plotting

## Defining a 2D Basis Function

To model the response of a neuron to the position of an animal in an arena, we need a 2D model for stimuli.
In this case, the stimuli are the 2D coordinates (x, y) representing the animal's position at each time step.

We can model a response function for multi-dimensional stimuli as the external product of two 1D basis functions.
The response function approximation for the 2D case would be:

\\[
f(x, y) \\approx \sum_{ij} \\alpha_{ij} \, a_i (x) b_j(y).
\\]

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

# Define a trajectory with 1000 time-points representing 
# the recorded trajectory of the animal
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

X, Y, Z = prod_basis.evaluate_on_grid(100,100)

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
# Mix & match different basis types
# ---------------------------------
# Different types of basis element can be combined together with multiplication and addition,
# this will be particularly useful one of the coordinates will be periodic (e.g. an angle).
# In principle, one can build N-dimensional basis by multiplying N-basis objects or add them 
# or any combination of the two operations. Be mindful of the exponential growth of the 
# number of basis elements for multiplicative compositions.


# Define 1D basis objects of different kinds
a_basis = nsl.basis.RaisedCosineBasisLinear(n_basis_funcs=10)
b_basis = nsl.basis.MSplineBasis(n_basis_funcs=10, order=2)

# add and multiply basis
add_basis = a_basis + b_basis
mult_basis = a_basis * b_basis

print(
      f"Additive basis with {add_basis._n_basis_funcs} elements.\n"
      f"Multiplicative basis with {mult_basis._n_basis_funcs} elements."
)

# %%
# Plotting the Log-spaced Raised Cosine Basis
# -----------------------------------
# Now, we plot the log-spaced Raised Cosine basis elements
