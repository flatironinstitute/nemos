# nemos 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/flatironinstitute/nemos/blob/main/LICENSE)
![Python version](https://img.shields.io/badge/python-3.10-blue.svg)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![codecov](https://codecov.io/gh/flatironinstitute/nemos/graph/badge.svg?token=vvtrcTFNeu)](https://codecov.io/gh/flatironinstitute/nemos)
[![Documentation Status](https://readthedocs.org/projects/nemos/badge/?version=latest)](https://nemos.readthedocs.io/en/latest/?badge=latest)
[![nemos CI](https://github.com/flatironinstitute/nemos/actions/workflows/ci.yml/badge.svg)](https://github.com/flatironinstitute/nemos/actions/workflows/ci.yml)


`nemos` (NEural MOdelS) is a statistical modeling framework optimized for systems neuroscience and powered by [JAX](https://jax.readthedocs.io/en/latest/). 
It streamlines the process of defining and selecting models, through a collection of easy-to-use methods for feature design.

The core of `nemos` includes GPU-accelerated, well-tested implementations of standard statistical models, currently 
focusing on the Generalized Linear Model (GLM). 

The package is under active development and more methods will be added in the future.

For those looking to get a better grasp of the Generalized Linear Model, we recommend checking out the 
Neuromatch Academy's lesson [here](https://www.youtube.com/watch?v=NFeGW5ljUoI&t=424s) and Jonathan Pillow's tutorial 
from Cosyne 2018 [here](https://www.youtube.com/watch?v=NFeGW5ljUoI&t=424s).



## Overview

At his core, `nemos` consists of two primary modules: the `basis` and the `glm` module.

The `basis` module focuses on designing model features (inputs) for the GLM, while the `glm` module is responsible 
for learning GLM parameters, predicting neuronal firing rates, and evaluating model performance.

### `basis` Module
The basis module includes a suite of composable feature constructors that accept time-series data as inputs. These inputs can be any observed variables, such as presented stimuli, head direction, position, or spike counts. The basis objects can perform two types of transformations on these inputs:

1. **Non-linear Mapping:** Transforms the input data through a non-linear function.
2. **Convolution:** Applies a convolution of the input data with a bank of filters.

Both transformations produce a transformed time-series of features X, with a shape of (n_samples, n_features).

### `glm` Module

The `glm` objects implements three key methods:

- **`fit`:** This method takes the feature matrix `X` and the spike counts time-series `y`. It learns the GLM coefficients that best map `X` to the firing rate, maximizing the likelihood of observing the spike counts `y`.
- **`predict`:** Receives the feature matrix `X` and uses the learned coefficients to predict the firing rate.
- **`score`:** Takes both the feature matrix X and the spike counts y, returning the log-likelihood and other metrics to assess model fit.

### Examples

Here's a brief demonstration of how the basis and glm modules work together within nemos.
#### Poisson GLM for features analysis

<img src="assets/glm_features_scheme.svg" width="100%">

In this example, we'll construct a time-series of features using the basis objects, applying a non-linear mapping by default:

```python
import nemos as nmo

# Instantiate the basis
basis_1 = nmo.basis.MSplineBasis(n_basis_funcs=5)
basis_2 = nmo.basis.CyclicBSplineBasis(n_basis_funcs=6)
basis_3 = nmo.basis.MSplineBasis(n_basis_funcs=7)

basis = basis_1 * basis_2 + basis_3

# Generate the design matrix starting from some raw 
# input time series, i.e. LFP phase, position, etc.
X = basis.compute_features(input_1, input_2, input_3)

# Fit the model mapping X to the spike count
# time-series y
glm = nmo.glm.GLM().fit(X, y)

# Inspect the learned coefficients
print(glm.coef_, glm.intercept_)

# compute the rate
firing_rate = glm.predict(X)

# compute log-likelihood
ll = glm.score(X, y)
```

#### Poisson GLM for neural population

<img src="assets/glm_population_scheme.svg" width="100%">

This second example demonstrates feature construction by convolving the simultaneously recorded population spike counts with a bank of filters, utilizing the basis in `conv` mode:

```python
import nemos as nmo

# assume that the population spike counts time-series is stored 
# in a 2D array spike_counts, shape (n_samples, n_neurons).

# generate 5 basis functions of 100 time-bin, 
# and convolve the counts with the basis.
X = nmo.basis.RaisedCosineBasisLog(5, mode="conv", window_size=100
    ).compute_features(spike_counts)

# fit a GLM to the first neuron counts time-series
glm = nmo.glm.GLM().fit(X, spike_counts[:, 0])

# compute the rate
firing_rate = glm.predict(X)

# compute log-likelihood
ll = glm.score(X, spike_counts[:, 0])
```
For a deeper dive, see our [Quickstart](https://nemos.readthedocs.io/en/latest/quickstart/)  guide and consider using [pynapple](https://github.com/pynapple-org/pynapple) for data exploration and preprocessing. When initializing the GLM object, you may optionally specify an [observation
model](https://nemos.readthedocs.io/en/latest/reference/nemos/observation_models/) and a [regularizer](https://nemos.readthedocs.io/en/latest/reference/nemos/regularizer/).

!!! note "Multi-epoch convolution"
    If your data is formatted as a `pynapple` time-series, the convolution performed by the basis objects will be 
    executed epoch-by-epoch, avoiding the risk of introducing artifacts from gaps in your time-series.

## Installation
Run the following `pip` command in your virtual environment.

**For macOS/Linux users:**
 ```bash
 pip install git+https://github.com/flatironinstitute/nemos.git
 ```

**For Windows users:**
 ```
 python -m pip install git+https://github.com/flatironinstitute/nemos.git
 ```

For more details, including specifics for GPU users and developers, refer to `nemos` [docs](https://nemos.readthedocs.io/en/latest/installation/).


## Disclaimer

Please note that this package is currently under development. While you can
download and test the functionalities that are already present, please be aware
that syntax and functionality may change before our preliminary release.

## Getting help and getting in touch

We communicate via several channels on Github:

- To report a bug, open an
  [issue](https://github.com/flatironinstitute/nemos/issues).
- To ask usage questions, discuss broad issues, or show off what you’ve made
  with nemos, go to
  [Discussions](https://github.com/flatironinstitute/nemos/discussions).
- To send suggestions for extensions or enhancements, please post in the
  [ideas](https://github.com/flatironinstitute/nemos/discussions/categories/ideas)
  section of discussions first. We’ll discuss it there and, if we decide to
  pursue it, open an issue to track progress.
- To contribute to the project, see the [contributing
  guide](CONTRIBUTING.md).

In all cases, we request that you respect our [code of
conduct](CODE_OF_CONDUCT.md).

