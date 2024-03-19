# nemos 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/flatironinstitute/nemos/blob/main/LICENSE)
![Python version](https://img.shields.io/badge/python-3.10-blue.svg)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![codecov](https://codecov.io/gh/flatironinstitute/nemos/graph/badge.svg?token=vvtrcTFNeu)](https://codecov.io/gh/flatironinstitute/nemos)
[![Documentation Status](https://readthedocs.org/projects/nemos/badge/?version=latest)](https://nemos.readthedocs.io/en/latest/?badge=latest)
[![nemos CI](https://github.com/flatironinstitute/nemos/actions/workflows/ci.yml/badge.svg)](https://github.com/flatironinstitute/nemos/actions/workflows/ci.yml)

`nemos` ("NEural MOdelS") is a statistical modeling framework for systems
neuroscience, built on top of [jax](jax.readthedocs.io/). nemos aims to provide
well-tested, GPU-accelerated implementations of standard statistical modeling
methods. nemos is not attempting to provide all the latest and greatest methods,
but instead to provide a stable place to start, which you can build off of or
compare against. For now, we are focusing on the Generalized Linear Model (GLM);
the package is under active development and more methods will be added in the
future.

To learn more about the Generalized Linear Model, we recommend [Neuromatch
Academy's
lesson](https://compneuro.neuromatch.io/tutorials/W1D3_GeneralizedLinearModels/student/W1D3_Intro.html)
and [Jonathan Pillow's Cosyne 2018
tutorial](https://www.youtube.com/watch?v=NFeGW5ljUoI&t=424s).

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

For more comprehensive instructions, including specifics for GPU users and developers, refer to [the installation page](installation.md).

## Basic usage

The core object of nemos is the [`GLM`][nemos.glm.GLM] object, which is built as an
extension of scikit-learn's
[estimator](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html#sklearn.base.BaseEstimator)
object: 

```python
import nemos as nmo
# Create predictors X and targets y
X = ...
y = ...

glm = nmo.glm.GLM()
glm.fit(X, y)

# Investigate GLM model parameters
glm.coef_
glm.intercept_
```

Nemos `GLM` objects predict spiking from a single neuron in response to
user-specified predictors. The predictors `X` must be a 3d array with shape
`(n_timebins, n_neurons, n_features)`, and `y` must be a 2d array with shape
`(n_timebins, n_neurons)`. We recommend using
[pynapple](https://github.com/pynapple-org/pynapple) for initial exploration and
reshaping of your data!

When initializing the `GLM` object, users can optionally specify the
[observation model][nemos.observation_models] (also known as the noise model)
and the [regularizer][nemos.regularizer].

Nemos also provides a variety of [basis functions][nemos.basis] for estimating
more complicated, non-linear relationships between experimental variables and
neuronal spiking.

## Navigating Our Tutorials and Guides

Our "API Guide", "Background" and "Neural Modeling" sections are continuously evolving and expanding. 
While they are under development, we ensure they are regularly tested and ready for you to explore. 

If your goal is a quick-start that would bring you up to speed with the main functionalities of the package, 
you should focus on the [API Guide](generated/api_guide) section.

If you seek a concise overview of the GLM framework and a detailed, step-by-step guide on building 
your model with `nemos`, refer to the [Background](generated/background) section.

If you want to learn more about the GLM framework and its applications in the field of Neuroscience, you should 
check out the [Neural Modeling](generated/neural_modeling). Here you will find a number of real-data examples covering a 
variety of Neuroscientific domains and brain regions. We sorted our examples from the simplest to the more complex,
with the intent of gradually introducing the model components.


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
  guide](https://github.com/flatironinstitute/nemos/blob/main/CONTRIBUTING.md).

In all cases, we request that you respect our [code of
conduct](https://github.com/flatironinstitute/nemos/blob/main/CODE_OF_CONDUCT.md).
