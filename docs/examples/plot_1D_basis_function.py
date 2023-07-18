# -*- coding: utf-8 -*-

"""
# Plotting 1D Basis Functions

## Defining a 1D Basis Function

We'll start by defining a 1D basis function object of the type `MSplineBasis`.
The hyperparameters required to initialize this class are:

- The number of basis functions, which should be a positive integer.
- The order of the spline, which should be an integer greater than 1.
"""

import numpy as np
import matplotlib.pylab as plt
import neurostatslib as nsl

# Initialize hyperparameters
order = 4
n_basis = 10

# Define the 1D basis function object
mspline_basis = nsl.basis.MSplineBasis(n_basis_funcs=n_basis, order=order)

# %%
# Evaluating and Plotting Basis Functions
# ------------------------------------
# We can evaluate a basis function using the `Basis.evaluate` method.
# For `SplineBasis`, the domain is determined by the samples provided to the `evaluate` method.
# An equi-spaced set of knots covering the range between the minimum and maximum of `samples` will be generated.

# Generate an array of sample points
samples = np.linspace(0,10, 1000)

# Evaluate the basis at the sample points
eval_basis = mspline_basis.evaluate(samples)

# Output information about the evaluated basis
print(f"Evaluated M-spline of order {order} with {eval_basis.shape[0]} "
      f"basis element and {eval_basis.shape[1]} samples.")

# %%
# Plotting the Basis Elements
# -----------------------

# Plot the evaluated basis elements
plt.figure()
plt.title(f"M-spline basis with {eval_basis.shape[0]} elements")
plt.plot(eval_basis.T)
plt.show()

# %%
# Evaluating and Plotting a Log-spaced Cosine Raised Basis
# ----------------------------------------------------------
# We initialize this basis with just the number of basis elements as an input.
# This basis is defined over the interval [0,1].

raised_cosine_log = nsl.basis.RaisedCosineBasisLog(n_basis_funcs=10)
samples = np.linspace(0, 1, 1000)

# Evaluate the raised cosine basis at the sample points
eval_basis = raised_cosine_log.evaluate(samples)

# %%
# Plotting the Log-spaced Raised Cosine Basis
# -----------------------------------

# Plot the evaluated log-spaced raised cosine basis
plt.figure()
plt.title(f"Log-spaced Raised Cosine basis with {eval_basis.shape[0]} elements")
plt.plot(eval_basis.T)
plt.show()
