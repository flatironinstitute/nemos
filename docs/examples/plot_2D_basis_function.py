# -*- coding: utf-8 -*-
""" 
# Plot 2D basis functions

## Define a 2D basis function

What if I want to model the response of a neuron to the position of the animal in an arena? 
In this case, the stimuli in this case would be 2D, the (x,y) position of the animal at each time step. 
We can model a response function for multi dimensional stimuli as the external product of two 1D basis functions.
For the 2D case we can approximate our resonse function as,

\\[
f(x,y) \\approx \sum_{ij} \\alpha_{ij}  \, a_i (x) b_j(y).
\\]

Defining such a basis is as simple as multiply to basis 1D basis objects.
"""

import numpy as np
import matplotlib.pyplot as plt

import neurostatslib as nsl

a_basis = nsl.basis.MSplineBasis(n_basis_funcs=20, order=3)
b_basis = nsl.basis.MSplineBasis(n_basis_funcs=10, order=2)
prod_basis = a_basis * b_basis

# %%
# Evaluate a basis and plot the results
# ------------------------------------
# Evaluating the product of two 1D basis objects will requires two input, one for the x-coordinate and one for the y-coordinate.
# The number of basis elements of the product basis will be the product of the elements of the 1D basis.


# %%

# define a trajectory with 1000 time-points, you can think of this as the recorded trajectory of the animal
T = 1000 
x_coord = np.linspace(0, 10, 1000)
y_coord = np.linspace(0, 50, 1000)
eval_basis = prod_basis.evaluate(x_coord, y_coord)

print(f"Product of two 1D splines with {eval_basis.shape[0]} "
      f"basis element and {eval_basis.shape[1]} samples:\n"
      f"\t- a_basis had {a_basis._n_basis_funcs} elements\n\t- b_basis had {b_basis._n_basis_funcs} elements.")


# %%
# Plot the basis elements
# -----------------------
# In order to plot a 2D basis set one would need to evaluate the basis on a grid of points over the basis function domain.
# This can be done through the method `neurostatslib.basis.evaluate_on_grid`.

# %%

X, Y, Z = prod_basis.evaluate_on_grid(200,200)

#plt.figure()
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# plot only the support
Z[Z == 0] = np.nan
ax.plot_surface(X, Y, Z[50], cmap="viridis", alpha=0.8)
ax.plot_surface(X, Y, Z[83], cmap="rainbow", alpha=0.8)
ax.plot_surface(X, Y, Z[125], cmap="inferno", alpha=0.8)
plt.title(f"Product basis with {eval_basis.shape[0]} elements")
plt.show()



# %%
# Evaluate a basis and plot a log-spaced Cosine Raised basis
# ----------------------------------------------------------
# This basis initialization requires only the number of basis as an input. 
# The basis has support on the interval [0,1].

# %%
raised_cosine_log = nsl.basis.RaisedCosineBasisLog(n_basis_funcs=10)
samples = np.linspace(0, 1, 1000)
eval_basis = raised_cosine_log.evaluate(samples)


# %%
# Plot log-spaced Raised Cosine basis
# -----------------------------------
plt.figure()
plt.title(f"Log-spaced Raised Cosine basis with {eval_basis.shape[0]} elements")
plt.plot(eval_basis.T)
plt.show()





