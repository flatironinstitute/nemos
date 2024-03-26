# nemos 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/flatironinstitute/nemos/blob/main/LICENSE)
![Python version](https://img.shields.io/badge/python-3.10-blue.svg)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![codecov](https://codecov.io/gh/flatironinstitute/nemos/graph/badge.svg?token=vvtrcTFNeu)](https://codecov.io/gh/flatironinstitute/nemos)
[![Documentation Status](https://readthedocs.org/projects/nemos/badge/?version=latest)](https://nemos.readthedocs.io/en/latest/?badge=latest)
[![nemos CI](https://github.com/flatironinstitute/nemos/actions/workflows/ci.yml/badge.svg)](https://github.com/flatironinstitute/nemos/actions/workflows/ci.yml)

`nemos` (NEural MOdelS) is a statistical modeling framework optimized for systems neuroscience and powered by [JAX](jax.readthedocs.io/). 
It streamlines the process of creating and selecting models, through a collection of easy-to-use methods for feature design.

The core of `nemos` includes GPU-accelerated, well-tested implementations of standard statistical models, currently 
focusing on the Generalized Linear Model (GLM). 

The package is under active development and more methods will be added in the future.

For those looking to get a better grasp of the Generalized Linear Model, we recommend checking out the 
Neuromatch Academy's lesson [here](https://www.youtube.com/watch?v=NFeGW5ljUoI&t=424s) and Jonathan Pillow's tutorial 
from Cosyne 2018 [here](https://www.youtube.com/watch?v=NFeGW5ljUoI&t=424s).

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


## Basic usage

### Feature Design

Using basis functions, `nemos` facilitates building a set of predictors. Given a 1-dimensional `feature`, you can generate a set of predictor `X` with predefined basis:
```python
basis = nmo.basis.MSplineBasis(n_basis_funcs=10)
X = basis.evaluate(feature)
```
Other basis are available depending on the type of feature :

  * [`BSplineBasis`](https://nemos.readthedocs.io/en/latest/reference/nemos/basis/#nemos.basis.BSplineBasis)

  * [`CyclicBSplineBasis`](https://nemos.readthedocs.io/en/latest/reference/nemos/basis/#nemos.basis.CyclicBSplineBasis) for angular features (i.e. head-direction)

  * [`RaisedCosineBasisLinear`](https://nemos.readthedocs.io/en/latest/reference/nemos/basis/#nemos.basis.RaisedCosineBasisLinear) or [`RaisedCosineBasisLog`](https://nemos.readthedocs.io/en/latest/reference/nemos/basis/#nemos.basis.RaisedCosineBasisLog) for temporal feature (i.e. spiking activity)

  * [`OrthExponentialBasis`](https://nemos.readthedocs.io/en/latest/reference/nemos/basis/#nemos.basis.OrthExponentialBasis)
  
`nemos` makes it easy to combine features. Basis can be added or multiplied together and the returned object will still be a basis.

```python
basis_1 = nmo.basis.MSplineBasis(n_basis_funcs=10)
basis_2 = nmo.basis.CyclicBSplineBasis(n_basis_funcs=12)
basis_3 = nmo.basis.MSplineBasis(n_basis_funcs=15)

basis = basis_1 * basis_2 + basis_3

X = basis.evaluate(feature_1, feature_2, feature_3)
```

### Model Fitting

For now, the core model of nemos is the
[`Poisson GLM`](https://nemos.readthedocs.io/en/latest/reference/nemos/glm/) object that predict firing rate of a single neuron in response to
user-specified predictors. 

```python
glm = nmo.glm.GLM()
glm.fit(X, y)
```
We recommend using [pynapple](https://github.com/pynapple-org/pynapple) for initial exploration and reshaping of your data!


When initializing the `GLM` object, users can optionally specify the
[observation
model](https://nemos.readthedocs.io/en/latest/reference/nemos/observation_models/)
(also known as the noise model) and the
[regularizer](https://nemos.readthedocs.io/en/latest/reference/nemos/regularizer/).

See [Quickstart](https://nemos.readthedocs.io/en/latest/quickstart/) for a
slightly longer overview of basic nemos functionality.

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

