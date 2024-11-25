# Install

## Prerequisites

1. **Ensure you have Python version `3.10` or above**: Check that Python `3.10` or above is installed in your system by opening the terminal/command prompt and executing the following command,
    ```bash
    python --version
    ```
    If you are unable to run the command, or if your Python version is below `3.10`, install/update Python.

    We suggest downloading an installer from [Anaconda](https://docs.anaconda.com/free/anaconda/install/) or [Miniconda](https://docs.anaconda.com/free/miniconda/). The former includes a comprehensive selection of Python packages for data-science and machine learning. The latter is a minimal installer which includes only few packages. 
    If you are not sure which one works best for you, take a look at the [Anaconda guidelines](https://docs.anaconda.com/free/distro-or-miniconda/).

2. **Create and activate a virtual environment:** Once your updated python is up and running, we recommend creating and activating a Python virtual environment using [`venv`](https://docs.python.org/3/library/venv.html) before installing NeMoS. This practice helps manage dependencies and avoid conflicts with other Python packages. 

    For `venv`, create your virtual environment in a specific directory. This example uses `~/python_venvs/nemos` for Linux/Mac and `C:%HOMEPATH%\python_venvs\nemos` for Windows; in general, free to use whatever directory works best for you.

### Creating and Activating a Virtual Environment

**For macOS and Linux:**

1. Open a terminal.

2. Run the following commands to create and activate the virtual environment:

    ```bash
    python -m venv ~/python_venvs/nemos
    source ~/python_venvs/nemos/bin/activate
    ```

**For Windows:**

1. Open a command prompt.

2. Execute the commands below to create and activate the virtual environment:
    ```
    python -m venv C:%HOMEPATH%\python_venvs\nemos
    cd C:%HOMEPATH%\python_venvs\nemos\
    .\Scripts\activate
    ```

## Installation Steps
After creating you virtual environment, follow one of the following sections below, depending on whether you need GPU support or not:
### CPU Installation

To install NeMoS on a system without a GPU, run this command from within your activated environment, 

**For macOS/Linux users:**
 ```bash
  pip install nemos
 ```

**For Windows users:**
 ```
 python -m pip install nemos
 ```

### GPU Installation

:::{warning}

JAX does not guarantee GPU support for Windows, see [here](https://jax.readthedocs.io/en/latest/installation.html#supported-platforms) for updates.

:::

For systems equipped with a GPU, you need to specifically install the GPU-enabled versions of `jax` and `jaxlib` before installing NeMoS.

1. **Install `jax` and `jaxlib` for GPU:** Follow the [JAX documentation](https://jax.readthedocs.io/en/latest/installation.html) instructions to install `jax` and `jaxlib` with GPU support.

2. **Verify GPU Installation:** To ensure `jax` correctly recognizes your GPU, execute the following in Python:
    ```python
    import jax
    print(jax.devices())
    ```

    If your GPU is listed among the devices, the installation was successful.

3. **Install NeMoS:** After successfully installing and configuring `jax` for GPU support, install NeMoS using the same command as in the [CPU installation](#cpu-installation).

### Installation For Developers

Developers should clone the repository and install NeMoS in editable mode, including developer dependencies. Follow these steps:

1. **Clone the repo:** From your environment, execute the following commands to clone the repository and navigate to its directory:
    ```bash
    git clone https://github.com/flatironinstitute/nemos.git
    cd nemos
    ```

2. **Install in editable mode:** Install the package in editable mode (using the `-e` option) and include the developer dependencies (using `[dev]`):

    ```bash
    pip install -e .[dev]
    ```
