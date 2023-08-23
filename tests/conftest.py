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
    np.random.seed(123)
    X = np.random.normal(size=(100, 1, 5))
    b_true = np.zeros((1, ))
    w_true = np.random.normal(size=(1, 5))
    model = nsl.glm.PoissonGLM(inverse_link_function=jax.numpy.exp)
    rate = jax.numpy.exp(jax.numpy.einsum("ik,tik->ti", w_true, X) + b_true[None, :])
    return X, np.random.poisson(rate), model, (w_true, b_true), rate


@pytest.fixture
def poissonGLM_coupled_model_config_simulate():
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
