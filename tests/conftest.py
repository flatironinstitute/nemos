"""
Testing configurations for the `neurostatslib` library.

This module contains test fixtures required to set up and verify the functionality
of the modules of the `neurostatslib` library.

Dependencies:
    - jax: Used for efficient numerical computing.
    - jax.numpy: JAX's version of NumPy, used for matrix operations.
    - numpy: Standard Python numerical computing library.
    - pytest: Testing framework.
    - yaml: For parsing and loading YAML configuration files.

Functions:
    - poissonGLM_model_instantiation: Sets up a Poisson GLM, instantiating its parameters
      with random values and returning a set of test data and expected output.

    - poissonGLM_coupled_model_config_simulate: Reads from a YAML configuration file,
      sets up a Poisson GLM with predefined parameters, and returns the initialized model
      along with other related parameters.

Note:
    This module primarily serves as a utility for test configurations, setting up initial conditions,
    and loading predefined parameters for testing various functionalities of the `neurostatslib` library.
"""
import inspect
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

import neurostatslib as nsl


@pytest.fixture
def poissonGLM_model_instantiation():
    """Set up a Poisson GLM for testing purposes.

    This fixture initializes a Poisson GLM with random parameters, simulates its response, and
    returns the test data, expected output, the model instance, true parameters, and the rate
    of response.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): Simulated input data.
            - np.random.poisson(rate) (numpy.ndarray): Simulated spike responses.
            - model (nsl.glm.PoissonGLM): Initialized model instance.
            - (w_true, b_true) (tuple): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of response.
    """
    np.random.seed(123)
    X = np.random.normal(size=(100, 1, 5))
    b_true = np.zeros((1, ))
    w_true = np.random.normal(size=(1, 5))
    noise_model = nsl.observation_noise.PoissonNoiseModel(jnp.exp)
    solver = nsl.solver.UnRegularizedSolver('GradientDescent', {})
    model = nsl.glm.GLM(noise_model, solver, score_type="log-likelihood")
    rate = jax.numpy.exp(jax.numpy.einsum("ik,tik->ti", w_true, X) + b_true[None, :])
    return X, np.random.poisson(rate), model, (w_true, b_true), rate


@pytest.fixture
def poissonGLM_coupled_model_config_simulate():
    """Set up a Poisson GLM from a predefined configuration in a YAML file.

    This fixture reads parameters for a Poisson GLM from a YAML configuration file, initializes
    the model accordingly, and returns the model instance with other related parameters.

    Returns:
        tuple: A tuple containing:
            - model (nsl.glm.PoissonGLM): Initialized model instance.
            - coupling_basis (jax.numpy.ndarray): Coupling basis values from the config.
            - feedforward_input (jax.numpy.ndarray): Feedforward input values from the config.
            - init_spikes (jax.numpy.ndarray): Initial spike values from the config.
            - jax.random.PRNGKey(123) (jax.random.PRNGKey): A pseudo-random number generator key.
    """
    current_file = inspect.getfile(inspect.currentframe())
    test_dir = os.path.dirname(os.path.abspath(current_file))
    with open(os.path.join(test_dir,
                           "simulate_coupled_neurons_params.yml"), "r") as fh:
        config_dict = yaml.safe_load(fh)

    model = nsl.glm.PoissonGLM(inverse_link_function=jax.numpy.exp)
    model.basis_coeff_ = jnp.asarray(config_dict["basis_coeff_"])
    model.baseline_link_fr_ = jnp.asarray(config_dict["baseline_link_fr_"])
    coupling_basis = jnp.asarray(config_dict["coupling_basis"])
    feedforward_input = jnp.asarray(config_dict["feedforward_input"])
    init_spikes = jnp.asarray(config_dict["init_spikes"])

    return model, coupling_basis, feedforward_input, init_spikes, jax.random.PRNGKey(123)

@pytest.fixture
def jaxopt_solvers():
    return [
        "GradientDescent",
        "BFGS",
        "LBFGS",
        "ScipyMinimize",
        "NonlinearCG",
        "ScipyBoundedMinimize",
        "LBFGSB",
        "ProximalGradient"
    ]
