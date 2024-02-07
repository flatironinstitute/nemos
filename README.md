[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/flatironinstitute/nemos/blob/main/LICENSE)
[![codecov](https://codecov.io/gh/flatironinstitute/nemos/graph/badge.svg?token=vvtrcTFNeu)](https://codecov.io/gh/flatironinstitute/nemos)
[![Documentation Status](https://readthedocs.org/projects/nemos/badge/?version=latest)](https://nemos.readthedocs.io/en/latest/?badge=latest)
[![nemos CI](https://github.com/flatironinstitute/nemos/actions/workflows/ci.yml/badge.svg)](https://github.com/flatironinstitute/nemos/actions/workflows/ci.yml)

# nemos
NEural MOdelS, a statistical modeling framework for neuroscience.

## Disclaimer
This is an alpha version, the code is in active development and the API is subject to change.

## Installation

To install `nemos` we recommend to creating and activating a virtual environment. You can do so through
`venv`.

For  Mac and Linux,
```shell
python -m venv <DIR>
source <DIR>/bin/activate
```

For Windows,
```shell
python -m venv <DIR>
<DIR>\Scripts\activate
```

### CPU Install

To install `nemos` in your environment run,

``` shell
pip install --upgrade pip
pip install nemos@git+https://github.com/flatironinstitute/nemos.git
```

### GPU install
If you have a GPU, you must need to install jax separately first to get the proper
build. 
In order to install `jax` and `jaxlib` for GPU, follow the instruction provided in the 
[jax docs](https://jax.readthedocs.io/en/latest/installation.html).

To check that `jax` for GPU is installed correctly, run python and type,

```python
import jax
print(jax.devices())
```
This should print a list of devices that `jax` can access. If your `GPU` is listed, your installation 
was successful.

Once `jax` is installed correctly, continue with the usual steps for installing `nemos`,

``` shell
pip install --upgrade pip
pip install nemos@git+https://github.com/flatironinstitute/nemos.git
```

