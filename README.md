[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/flatironinstitute/nemos/blob/main/LICENSE)
[![codecov](https://codecov.io/gh/flatironinstitute/nemos/graph/badge.svg?token=vvtrcTFNeu)](https://codecov.io/gh/flatironinstitute/nemos)
[![Documentation Status](https://readthedocs.org/projects/nemos/badge/?version=latest)](https://nemos.readthedocs.io/en/latest/?badge=latest)
[![nemos CI](https://github.com/flatironinstitute/nemos/actions/workflows/ci.yml/badge.svg)](https://github.com/flatironinstitute/nemos/actions/workflows/ci.yml)

# nemos
NEural MOdelS, a statistical modeling framework for neuroscience.

## Disclaimer
This is an alpha version, the code is in active development and the API is subject to change.

## Installation

### Prerequisites

Before installing `nemos`, we recommend creating and activating a Python virtual environment using `venv`. This helps to manage dependencies and avoid conflicts with other Python packages.

#### Creating a Virtual Environment

- **For macOS and Linux:**
  Open a terminal and run the following commands:

  ```
  python -m venv <DIR>
  source <DIR>/bin/activate
  ```

- **For Windows:**
  Open a command prompt and execute:

  ```
  python -m venv <DIR>
  <DIR>\Scripts\activate
  ```

Replace `<DIR>` with the directory where you want to create the virtual environment.

### Installation Steps

#### Standard CPU Installation

To install `nemos` on a system without a GPU, follow these steps:

1. Ensure your pip is up to date:

   ```
   pip install --upgrade pip
   ```

2. Install `nemos` directly from the GitHub repository:

   ```
   pip install git+https://github.com/flatironinstitute/nemos.git
   ```

#### GPU Installation

For systems equipped with a GPU, a specific installation process is required to utilize the GPU with `nemos`.

1. **Install `jax` and `jaxlib` for GPU:**

   - Follow the instructions provided in the [JAX documentation](https://jax.readthedocs.io/en/latest/installation.html) to install `jax` and `jaxlib` for GPU support.

2. **Verify GPU Installation:**

   - To ensure `jax` recognizes your GPU, execute the following in Python:

     ```
     import jax
     print(jax.devices())
     ```

     If your GPU is listed among the devices, the installation was successful.

3. **Install `nemos`:**

   - After successfully installing and configuring `jax` for GPU, install `nemos` using the same steps as the CPU installation:

     ```
     pip install --upgrade pip
     pip install git+https://github.com/flatironinstitute/nemos.git
     ```
