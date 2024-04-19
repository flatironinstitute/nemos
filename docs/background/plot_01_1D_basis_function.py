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

import nemos as nmo

# Initialize hyperparameters
order = 4
n_basis = 10

# Define the 1D basis function object
mspline_basis = nmo.basis.MSplineBasis(n_basis_funcs=n_basis, order=order)

# %%
# Evaluating a Basis
# ------------------------------------
# The `Basis` object is callable, and can be evaluated as a function. For `SplineBasis`, the domain is defined by
# the samples that we input to the `__call__` method. This results in an equi-spaced set of knots, which spans
# the range from the smallest to the largest sample. These knots are then used to construct a uniformly spaced basis.

# Generate an array of sample points
samples = np.random.uniform(0, 10, size=1000)

# Evaluate the basis at the sample points
eval_basis = mspline_basis(samples)

# Output information about the evaluated basis
print(f"Evaluated M-spline of order {order} with {eval_basis.shape[1]} "
      f"basis element and {eval_basis.shape[0]} samples.")

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

eval_mode = nmo.basis.BSplineBasis(n_basis_funcs=n_basis, mode="eval")
conv_mode = nmo.basis.BSplineBasis(n_basis_funcs=n_basis, mode="conv", window_size=100)

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
equispaced_samples, eval_basis = mspline_basis.evaluate_on_grid(n_samples)

# Plot each basis element
plt.figure()
plt.title(f"M-spline basis with {eval_basis.shape[1]} elements\nevaluated at {eval_basis.shape[0]} sample points")
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

