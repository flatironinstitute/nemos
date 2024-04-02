# nemos 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/flatironinstitute/nemos/blob/main/LICENSE)
![Python version](https://img.shields.io/badge/python-3.10-blue.svg)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![codecov](https://codecov.io/gh/flatironinstitute/nemos/graph/badge.svg?token=vvtrcTFNeu)](https://codecov.io/gh/flatironinstitute/nemos)
[![Documentation Status](https://readthedocs.org/projects/nemos/badge/?version=latest)](https://nemos.readthedocs.io/en/latest/?badge=latest)
[![nemos CI](https://github.com/flatironinstitute/nemos/actions/workflows/ci.yml/badge.svg)](https://github.com/flatironinstitute/nemos/actions/workflows/ci.yml)


`nemos` (NEural MOdelS) is a statistical modeling framework optimized for systems neuroscience and powered by [JAX](https://jax.readthedocs.io/en/latest/). 
It streamlines the process of creating and selecting models, through a collection of easy-to-use methods for feature design.

The core of `nemos` includes GPU-accelerated, well-tested implementations of standard statistical models, currently 
focusing on the Generalized Linear Model (GLM). 

The package is under active development and more methods will be added in the future.

For those looking to get a better grasp of the Generalized Linear Model, we recommend checking out the 
Neuromatch Academy's lesson [here](https://www.youtube.com/watch?v=NFeGW5ljUoI&t=424s) and Jonathan Pillow's tutorial 
from Cosyne 2018 [here](https://www.youtube.com/watch?v=NFeGW5ljUoI&t=424s).



## Usage

 The `glm.GLM` object from `nemos` predict spiking from a single neuron in response to user-specified predictors. 
 The predictors `X` must be a 2d array with shape `(n_timebins, n_features)`, and `y` (or spike count) must be 
 a 1d array with shape `(n_timebins, )`.

### Poisson GLM for features analysis

<img src="docs/assets/glm_features_scheme.svg" width="100%">


`nemos` streamlines the design of a GLM with multiples features. Implementing the model in the figure above can be done in a few lines of code :

```python
import nemos as nmo

# Instantiate the basis
basis_1 = nmo.basis.MSplineBasis(n_basis_funcs=5)
basis_2 = nmo.basis.CyclicBSplineBasis(n_basis_funcs=6)
basis_3 = nmo.basis.MSplineBasis(n_basis_funcs=7)

basis = basis_1 * basis_2 + basis_3

# Generate the design matrix
X = basis.compute_features(feature_1, feature_2, feature_3)

# Fit the model
nmo.glm.GLM().fit(X, y)
```

### Poisson GLM for neural population

<img src="docs/assets/glm_population_scheme.svg" width="100%">

Building the model above is more complicated. In this case, input spikes are convolved with a bank of basis functions that stretch in time. With nemos, you can convolve the basis functions with a few lines of code:

```python
import nemos as nmo

# generate 5 basis functions of 100 time-bin, 
# and convolve the counts with the basis.
X = nmo.basis.RaisedCosineBasisLog(5, mode="conv", window_size=100
    ).compute_features(spike_counts)

# fit a GLM to the first neuron spike counts
glm = nmo.glm.GLM().fit(X, spike_counts[:, 0])

# compute the rate
firing_rate = glm.predict(X)

# compute log-likelihood
ll = glm.score(X, spike_counts[:, 0])
```


See [Quickstart](https://nemos.readthedocs.io/en/latest/quickstart/) for an overview of basic nemos functionality.

We recommend using [pynapple](https://github.com/pynapple-org/pynapple) for initial exploration and reshaping of your data!

When initializing the `GLM` object, users can optionally specify the
[observation
model](https://nemos.readthedocs.io/en/latest/reference/nemos/observation_models/)
 and the
[regularizer](https://nemos.readthedocs.io/en/latest/reference/nemos/regularizer/).



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

