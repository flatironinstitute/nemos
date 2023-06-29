# -*- coding: utf-8 -*-
""" 
# Plot 1D basis functions

## Define a basis function

Let's start by defining a basis function object of the type `MSplineBasis`. The hyperparameters required to initialize the class are:

- the number of basis functions, which should be a positive integer.
- the order of the spline, which should be an integer greater than 1.
"""

import numpy as np
import matplotlib.pylab as plt

import neurostatslib as nsl

order = 4
n_basis = 10
mspline_basis = nsl.basis.MSplineBasis(n_basis_funcs=n_basis, order=order)

# %%
# Evaluate a basis and plot the results
# ------------------------------------
# A basis can be evaluated with the public method "basis.Basis.evaluate".
# In the case of SplineBasis, the domain over which the basis is defined is determined by the `samples` that are provided
#  as inputs to the "evaluate" method;
#
# An equi-spaced set of knots will cover the range between min( `samples` )  and max(`samples`) will be generated.

# %%

samples = np.linspace(0,10, 1000)
eval_basis = mspline_basis.evaluate(samples)

print(f"Evaluated M-spline of order {order} with {eval_basis.shape[0]} "
      f"basis element and {eval_basis.shape[1]} samples.")
# %%
# Plot the basis elements
# -----------------------

# %%

plt.figure()
plt.title(f"M-spline basis with {eval_basis.shape[0]} elements")
plt.plot(eval_basis.T)
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





