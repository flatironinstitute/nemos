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

```{code-cell} ipython3
:tags: [hide-input]

%matplotlib inline
import warnings

# Ignore the first specific warning
warnings.filterwarnings(
    "ignore",
    message="plotting functions contained within `_documentation_utils` are intended for nemos's documentation.",
    category=UserWarning,
)

# Ignore the second specific warning
warnings.filterwarnings(
    "ignore",
    message="Ignoring cached namespace 'core'",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=(
        "invalid value encountered in div "
    ),
    category=RuntimeWarning,
)
```

(simple_basis_function)=
# Simple Basis Function

## Defining a 1D Basis Object

We'll start by defining a 1D basis function object of the type [`EvalMSpline`](nemos.basis.basis.EvalMSpline).
The hyperparameters required to initialize this class are:

- The number of basis functions, which should be a positive integer.
- The order of the spline, which should be an integer greater than 1.

```{code-cell} ipython3
import matplotlib.pylab as plt
import numpy as np
import pynapple as nap

import nemos as nmo

# Initialize hyperparameters
order = 4
n_basis = 10

# Define the 1D basis function object
bspline = nmo.basis.EvalBSpline(n_basis_funcs=n_basis, order=order)
```

## Evaluating a Basis

The [`Basis`](nemos.basis._basis.Basis) object is callable, and can be evaluated as a function. By default, the support of the basis
is defined by the samples that we input to the [`__call__`](nemos.basis._basis.Basis.__call__) method, and covers from the smallest to the largest value.


```{code-cell} ipython3

# Generate a time series of sample points
samples = nap.Tsd(t=np.arange(1001), d=np.linspace(0, 1,1001))

# Evaluate the basis at the sample points
eval_basis = bspline(samples)

# Output information about the evaluated basis
print(f"Evaluated B-spline of order {order} with {eval_basis.shape[1]} "
      f"basis element and {eval_basis.shape[0]} samples.")

fig = plt.figure()
plt.title("B-spline basis")
plt.plot(samples, eval_basis);
```

```{code-cell} ipython3
:tags: [hide-input]

# save image for thumbnail
from pathlib import Path
import os

root = os.environ.get("READTHEDOCS_OUTPUT")
if root:
   path = Path(root) / "html/_static/thumbnails/background"
# if local store in ../_build/html/...
else:
   path = Path("../_build/html/_static/thumbnails/background")
 
# make sure the folder exists if run from build
if root or Path("../_build/html/_static").exists():
   path.mkdir(parents=True, exist_ok=True)

if path.exists():
  fig.savefig(path / "plot_01_1D_basis_function.svg")
```

## Setting the basis support
Sometimes, it is useful to restrict the basis to a fixed range. This can help manage outliers or ensure that
your basis covers the same range across multiple experimental sessions.
You can specify a range for the support of your basis by setting the `bounds`
parameter at initialization. Evaluating the basis at any sample outside the bounds will result in a NaN.


```{code-cell} ipython3
bspline_range = nmo.basis.EvalBSpline(n_basis_funcs=n_basis, order=order, bounds=(0.2, 0.8))

print("Evaluated basis:")
# 0.5  is within the support, 0.1 is outside the support
print(np.round(bspline_range([0.5, 0.1]), 3))
```

Let's compare the default behavior of basis (estimating the range from the samples) with
the fixed range basis.


```{code-cell} ipython3
fig, axs = plt.subplots(2,1, sharex=True)
plt.suptitle("B-spline basis ")
axs[0].plot(samples, bspline(samples), color="k")
axs[0].set_title("default")
axs[1].plot(samples, bspline_range(samples), color="tomato")
axs[1].set_title("bounds=[0.2, 0.8]")
plt.tight_layout()
```

## Feature Computation
The bases in the module `nemos.basis` can be classified in two categories:

- **Evaluation Bases**: Objects for which [`compute_features`](nemos.basis._basis.Basis.compute_features) that returns the evaluated basis. This means that the basis are applying a non-linear transformation of the input. The class name for this kind of bases starts with "Eval", e.g. "EvalBSpline".
- **Convolution Bases**: Objects for which [`compute_features`](nemos.basis._basis.Basis.compute_features) will convolve the input with a kernel of basis elements with `window_size` specified by the user. The class name for this kind of bases starts with "Conv", e.g. "ConvBSpline".

Let's see how this two modalities operate.


```{code-cell} ipython3
eval_mode = nmo.basis.EvalMSpline(n_basis_funcs=n_basis)
conv_mode = nmo.basis.ConvMSpline(n_basis_funcs=n_basis, window_size=100)

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
```

:::{admonition} NaN-Padding
:class: note
Convolution is performed in "valid" mode, and then NaN-padded. The default behavior
is padding left, which makes the output feature causal.
This is why the first half of the `conv_feature` is full of NaNs and appears as white.
If you want to learn more about convolutions, as well as how and when to change defaults
check out the tutorial on [1D convolutions](plot_03_1D_convolution).
:::



Plotting the Basis Function Elements:
--------------------------------------
We suggest visualizing the basis post-instantiation by evaluating each element on a set of equi-spaced sample points
and then plotting the result. The method [`Basis.evaluate_on_grid`](nemos.basis._basis.Basis.evaluate_on_grid) is designed for this, as it generates and returns
the equi-spaced samples along with the evaluated basis functions. The benefits of using Basis.evaluate_on_grid become
particularly evident when working with multidimensional basis functions. You can find more details and visual
background in the
[2D basis elements plotting section](plotting-2d-additive-basis-elements).


```{code-cell} ipython3
# Call evaluate on grid on 100 sample points to generate samples and evaluate the basis at those samples
n_samples = 100
equispaced_samples, eval_basis = bspline.evaluate_on_grid(n_samples)

# Plot each basis element
plt.figure()
plt.title(f"B-spline basis with {eval_basis.shape[1]} elements\nevaluated at {eval_basis.shape[0]} sample points")
plt.plot(equispaced_samples, eval_basis)
plt.show()
```

Other Basis Types
-----------------
Each basis type may necessitate specific hyperparameters for instantiation. For a comprehensive description,
please refer to the  [API Guide](nemos_basis). After instantiation, all classes
share the same syntax for basis evaluation. The following is an example of how to instantiate and
evaluate a log-spaced cosine raised function basis.


```{code-cell} ipython3
# Instantiate the basis noting that the `RaisedCosineLog` basis does not require an `order` parameter
raised_cosine_log = nmo.basis.EvalRaisedCosineLog(n_basis_funcs=10, width=1.5, time_scaling=50)

# Evaluate the raised cosine basis at the equi-spaced sample points
# (same method in all Basis elements)
samples, eval_basis = raised_cosine_log.evaluate_on_grid(100)

# Plot the evaluated log-spaced raised cosine basis
plt.figure()
plt.title(f"Log-spaced Raised Cosine basis with {eval_basis.shape[1]} elements")
plt.plot(samples, eval_basis)
plt.show()
```
