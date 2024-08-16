# -*- coding: utf-8 -*-

"""
# One-Dimensional Basis

## Defining a 1D Basis Object

We'll start by defining a 1D basis function object of the type `MSplineBasis`.
The hyperparameters required to initialize this class are:

- The number of basis functions, which should be a positive integer.
- The order of the spline, which should be an integer greater than 1.
"""

import matplotlib.pylab as plt
import numpy as np
import pynapple as nap

import nemos as nmo

# Initialize hyperparameters
order = 4
n_basis = 10

# Define the 1D basis function object
bspline = nmo.basis.BSplineBasis(n_basis_funcs=n_basis, order=order)

# %%
# ## Evaluating a Basis
#
# The `Basis` object is callable, and can be evaluated as a function. By default, the support of the basis
# is defined by the samples that we input to the `__call__` method, and covers from the smallest to the largest value.

# Generate a time series of sample points
samples = nap.Tsd(t=np.arange(1001), d=np.linspace(0, 1,1001))

# Evaluate the basis at the sample points
eval_basis = bspline(samples)

# Output information about the evaluated basis
print(f"Evaluated B-spline of order {order} with {eval_basis.shape[1]} "
      f"basis element and {eval_basis.shape[0]} samples.")

plt.figure()
plt.title("B-spline basis")
plt.plot(eval_basis)

# %%
# ## Setting the basis support
# Sometimes, it is useful to restrict the basis to a fixed range. This can help manage outliers or ensure that
# your basis covers the same range across multiple experimental sessions.
# You can specify a range for the support of your basis by setting the `bounds`
# parameter at initialization. Evaluating the basis at any sample outside the bounds will result in a NaN.

bspline_range = nmo.basis.BSplineBasis(n_basis_funcs=n_basis, order=order, bounds=(0.2, 0.8))

print("Evaluated basis:")
# 0.5  is within the support, 0.1 is outside the support
print(np.round(bspline_range([0.5, 0.1]), 3))


# %%
# Let's compare the default behavior of basis (estimating the range from the samples) with
# the fixed range basis.

fig, axs = plt.subplots(2,1, sharex=True)
plt.suptitle("B-spline basis ")
axs[0].plot(bspline(samples), color="k")
axs[0].set_title("default")
axs[1].plot(bspline_range(samples), color="tomato")
axs[1].set_title("bounds=[0.2, 0.8]")
plt.tight_layout()

# %%
# ## Basis `mode`
# In constructing features, `Basis` objects can be used in two modalities: `"eval"` for evaluate or `"conv"`
# for convolve. These two modalities change the behavior of the `construct_features` method of `Basis`, in particular,
#
# - If a basis is in mode `"eval"`, then `construct_features` simply returns the evaluated basis.
# - If a basis is in mode `"conv"`, then `construct_features` will convolve the input with a kernel of basis
#   with `window_size` specified by the user.
#
# Let's see how this two modalities operate.

eval_mode = nmo.basis.MSplineBasis(n_basis_funcs=n_basis, mode="eval")
conv_mode = nmo.basis.MSplineBasis(n_basis_funcs=n_basis, mode="conv", window_size=100)

# define an input
angles = np.linspace(0, np.pi*4, 201)
y = np.cos(angles)

# compute features in the two modalities
eval_feature = eval_mode.compute_features(y)
conv_feature = conv_mode.compute_features(y)

# plot results
fig, axs = plt.subplots( 3, 1, sharex="all", figsize=(6, 4))

# plot signal
axs[0].set_title("Input")
axs[0].plot(y)
axs[0].set_xticks([])
axs[0].set_ylabel("signal", fontsize=12)

# plot eval results
axs[1].set_title("eval features")
axs[1].imshow(eval_feature.T, aspect="auto")
axs[1].set_xticks([])
axs[1].set_ylabel("basis", fontsize=12)

# plot conv results
axs[2].set_title("convolutional features")
axs[2].imshow(conv_feature.T, aspect="auto")
axs[2].set_xlabel("time", fontsize=12)
axs[2].set_ylabel("basis", fontsize=12)
plt.tight_layout()

# %%
#
# !!! note "NaN-Padding"
#     Convolution is performed in "valid" mode, and then NaN-padded. The default behavior
#     is padding left, which makes the output feature causal.
#     This is why the first half of the `conv_feature` is full of NaNs and appears as white.
#     If you want to learn more about convolutions, as well as how and when to change defaults
#     check out the tutorial on [1D convolutions](../plot_03_1D_convolution).

# %%
# Plotting the Basis Function Elements:
# --------------------------------------
# We suggest visualizing the basis post-instantiation by evaluating each element on a set of equi-spaced sample points
# and then plotting the result. The method `Basis.evaluate_on_grid` is designed for this, as it generates and returns
# the equi-spaced samples along with the evaluated basis functions. The benefits of using Basis.evaluate_on_grid become
# particularly evident when working with multidimensional basis functions. You can find more details and visual
# background in the
# [2D basis elements plotting section](../plot_02_ND_basis_function/#plotting-2d-additive-basis-elements).

# Call evaluate on grid on 100 sample points to generate samples and evaluate the basis at those samples
n_samples = 100
equispaced_samples, eval_basis = bspline.evaluate_on_grid(n_samples)

# Plot each basis element
plt.figure()
plt.title(f"B-spline basis with {eval_basis.shape[1]} elements\nevaluated at {eval_basis.shape[0]} sample points")
plt.plot(equispaced_samples, eval_basis)
plt.show()

# %%
# Other Basis Types
# -----------------
# Each basis type may necessitate specific hyperparameters for instantiation. For a comprehensive description,
# please refer to the  [Code References](../../../reference/nemos/basis). After instantiation, all classes
# share the same syntax for basis evaluation. The following is an example of how to instantiate and
# evaluate a log-spaced cosine raised function basis.

# Instantiate the basis noting that the `RaisedCosineBasisLog` does not require an `order` parameter
raised_cosine_log = nmo.basis.RaisedCosineBasisLog(n_basis_funcs=10, width=1.5, time_scaling=50)

# Evaluate the raised cosine basis at the equi-spaced sample points
# (same method in all Basis elements)
samples, eval_basis = raised_cosine_log.evaluate_on_grid(100)

# Plot the evaluated log-spaced raised cosine basis
plt.figure()
plt.title(f"Log-spaced Raised Cosine basis with {eval_basis.shape[1]} elements")
plt.plot(samples, eval_basis)
plt.show()

