"""
Testing configurations for the `nemos` library.

This module contains test fixtures required to set up and verify the functionality
of the modules of the `nemos` library.

Note:
    This module primarily serves as a utility for test configurations, setting up initial conditions,
    and loading predefined parameters for testing various functionalities of the `nemos` library.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nemos as nmo


# Sample subclass to test instantiation and methods
class MockRegressor(nmo.base_class.BaseRegressor):
    """
    Mock implementation of the BaseRegressor abstract class for testing purposes.
    Implements all required abstract methods as empty methods.
    """

    def __init__(self, std_param: int = 0):
        """Initialize a MockBaseRegressor instance with optional standard parameters."""
        self.std_param = std_param
        super().__init__()

    def fit(self, X, y):
        pass

    def predict(self, X) -> jnp.ndarray:
        pass

    def score(
        self,
        X,
        y,
        **kwargs,
    ) -> jnp.ndarray:
        pass

    def simulate(
        self,
        random_key: jax.Array,
        feed_forward_input,
        **kwargs,
    ):
        pass

    def _check_params(self, *args, **kwargs):
        pass

    def _check_input_and_params_consistency(self, *args, **kwargs):
        pass

    def _check_input_dimensionality(self, *args, **kwargs):
        pass

    def _get_coef_and_intercept(self):
        pass

    def _set_coef_and_intercept(self, params):
        pass


class MockRegressorNested(MockRegressor):
    def __init__(self, other_param: int, std_param: int = 0):
        super().__init__(std_param=std_param)
        self.other_param = MockGLM(std_param=other_param)


class MockGLM(nmo.glm.GLM):
    """
    Mock implementation of the BaseRegressor abstract class for testing purposes.
    Implements all required abstract methods as empty methods.
    """

    def __init__(self, std_param: int = 0):
        """Initialize a MockBaseRegressor instance with optional standard parameters."""
        self.std_param = std_param
        super().__init__()

    def fit(self, X, y):
        pass

    def predict(self, X) -> jnp.ndarray:
        pass

    def score(
        self,
        X,
        y,
        **kwargs,
    ) -> jnp.ndarray:
        pass

    def simulate(
        self,
        random_key: jax.Array,
        feed_forward_input,
        **kwargs,
    ):
        pass

    def _get_coef_and_intercept(self):
        pass

    def _set_coef_and_intercept(self, params):
        pass


@pytest.fixture
def mock_regressor():
    return MockRegressor(std_param=2)


@pytest.fixture
def mock_regressor_nested():
    return MockRegressorNested(other_param=1, std_param=2)


@pytest.fixture
def mock_glm():
    return MockGLM(std_param=2)


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
            - model (nmo.glm.PoissonGLM): Initialized model instance.
            - (w_true, b_true) (tuple): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of response.
    """
    np.random.seed(123)
    X = np.random.normal(size=(100, 5))
    b_true = np.zeros((1,))
    w_true = np.random.normal(size=(5,))
    observation_model = nmo.observation_models.PoissonObservations(jnp.exp)
    regularizer = nmo.regularizer.UnRegularized("GradientDescent", {})
    model = nmo.glm.GLM(observation_model, regularizer)
    rate = jax.numpy.exp(jax.numpy.einsum("k,tk->t", w_true, X) + b_true)
    return X, np.random.poisson(rate), model, (w_true, b_true), rate


@pytest.fixture
def poisson_population_GLM_model():
    """Set up a population Poisson GLM for testing purposes.

    This fixture initializes a Poisson GLM with random parameters, simulates its response, and
    returns the test data, expected output, the model instance, true parameters, and the rate
    of response.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): Simulated input data.
            - np.random.poisson(rate) (numpy.ndarray): Simulated spike responses.
            - model (nmo.glm.PoissonGLM): Initialized model instance.
            - (w_true, b_true) (tuple): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of response.
    """
    np.random.seed(123)
    X = np.random.normal(size=(500, 5))
    b_true = -2 * np.ones((3,))
    w_true = np.random.normal(size=(5, 3))
    observation_model = nmo.observation_models.PoissonObservations(jnp.exp)
    regularizer = nmo.regularizer.UnRegularized("GradientDescent", {})
    model = nmo.glm.PopulationGLM(observation_model, regularizer)
    rate = jnp.exp(jnp.einsum("ki,tk->ti", w_true, X) + b_true)
    return X, np.random.poisson(rate), model, (w_true, b_true), rate


@pytest.fixture
def poisson_population_GLM_model_pytree(poisson_population_GLM_model):
    """Set up a population Poisson GLM for testing purposes.

    This fixture initializes a Poisson GLM with random parameters, simulates its response, and
    returns the test data, expected output, the model instance, true parameters, and the rate
    of response.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): Simulated input data.
            - np.random.poisson(rate) (numpy.ndarray): Simulated spike responses.
            - model (nmo.glm.PoissonGLM): Initialized model instance.
            - (w_true, b_true) (tuple): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of response.
    """
    X, spikes, model, true_params, rate = poisson_population_GLM_model
    X_tree = nmo.pytrees.FeaturePytree(input_1=X[..., :3], input_2=X[..., 3:])
    true_params_tree = (
        dict(input_1=true_params[0][:3], input_2=true_params[0][3:]),
        true_params[1],
    )
    model_tree = nmo.glm.PopulationGLM(model.observation_model, model.regularizer)
    return X_tree, np.random.poisson(rate), model_tree, true_params_tree, rate


@pytest.fixture
def poissonGLM_model_instantiation_pytree(poissonGLM_model_instantiation):
    """Set up a Poisson GLM for testing purposes.

    This fixture initializes a Poisson GLM with random parameters, simulates its response, and
    returns the test data, expected output, the model instance, true parameters, and the rate
    of response.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): Simulated input data.
            - np.random.poisson(rate) (numpy.ndarray): Simulated spike responses.
            - model (nmo.glm.PoissonGLM): Initialized model instance.
            - (w_true, b_true) (tuple): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of response.
    """
    X, spikes, model, true_params, rate = poissonGLM_model_instantiation
    X_tree = nmo.pytrees.FeaturePytree(input_1=X[..., :3], input_2=X[..., 3:])
    true_params_tree = (
        dict(input_1=true_params[0][:3], input_2=true_params[0][3:]),
        true_params[1],
    )
    model_tree = nmo.glm.GLM(model.observation_model, model.regularizer)
    return X_tree, np.random.poisson(rate), model_tree, true_params_tree, rate


@pytest.fixture
def coupled_model_simulate():
    """Set up a Poisson GLM from a predefined configuration in a json file.

    This fixture reads parameters for a Poisson GLM from a json configuration file, initializes
    the model accordingly, and returns the model instance with other related parameters.

    Returns:
        tuple: A tuple containing:
            - the coupling coeffs
            - the feedforward coeffs
            - the intercepts
            - jax.random.key(123) (jax.Array): A pseudo-random number generator key.
            - feedforward_input (jax.numpy.ndarray): Feedforward input values from the config.
            - coupling_basis (jax.numpy.ndarray): Coupling basis values from the config.
            - init_spikes (jax.numpy.ndarray): Initial spike values from the config.
            - a link function.

    """

    n_neurons, coupling_duration, sim_duration = 2, 100, 1000
    coupling_filter_bank = np.zeros((coupling_duration, n_neurons, n_neurons))
    for unit_i in range(n_neurons):
        for unit_j in range(n_neurons):
            coupling_filter_bank[:, unit_i, unit_j] = (
                nmo.simulation.difference_of_gammas(coupling_duration)
            )
    # shrink the filters for simulation stability
    coupling_filter_bank *= 0.8
    basis = nmo.basis.RaisedCosineBasisLog(20)

    # approximate the coupling filters in terms of the basis function
    _, coupling_basis = basis.evaluate_on_grid(coupling_filter_bank.shape[0])
    coupling_coeff = nmo.simulation.regress_filter(coupling_filter_bank, coupling_basis)
    feedforward_coeff = np.ones((n_neurons, 2))
    intercepts = -3 * jnp.ones(n_neurons)

    feedforward_input = jnp.c_[
        jnp.cos(jnp.linspace(0, np.pi * 4, sim_duration)),
        jnp.sin(jnp.linspace(0, np.pi * 4, sim_duration)),
    ]

    feedforward_input = jnp.tile(feedforward_input[:, None], (1, n_neurons, 1))
    init_spikes = jnp.zeros((coupling_duration, n_neurons))

    return (
        coupling_coeff,
        feedforward_coeff,
        intercepts,
        jax.random.key(123),
        feedforward_input,
        coupling_basis,
        init_spikes,
        jnp.exp,
    )


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
        "ProximalGradient",
    ]


@pytest.fixture
def group_sparse_poisson_glm_model_instantiation():
    """Set up a Poisson GLM for testing purposes with group sparse weights.

    This fixture initializes a Poisson GLM with random, group sparse, parameters, simulates its response, and
    returns the test data, expected output, the model instance, true parameters, and the rate
    of response

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): Simulated input data.
            - np.random.poisson(rate) (numpy.ndarray): Simulated spike responses.
            - model (nmo.glm.PoissonGLM): Initialized model instance.
            - (w_true, b_true) (tuple): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of response.
    """
    np.random.seed(123)
    X = np.random.normal(size=(100, 5))
    b_true = np.zeros((1,))
    w_true = np.random.normal(size=(5,))
    w_true[1:4] = 0.0
    mask = np.zeros((2, 5))
    mask[0, 1:4] = 1
    mask[1, [0, 4]] = 1
    observation_model = nmo.observation_models.PoissonObservations(jnp.exp)
    regularizer = nmo.regularizer.UnRegularized("GradientDescent", {})
    model = nmo.glm.GLM(observation_model, regularizer)
    rate = jax.numpy.exp(jax.numpy.einsum("k,tk->t", w_true, X) + b_true)
    return X, np.random.poisson(rate), model, (w_true, b_true), rate, mask


@pytest.fixture
def group_sparse_poisson_glm_population():
    """Set up a Poisson GLM for testing purposes with group sparse weights.

    This fixture initializes a Poisson GLM with random, group sparse, parameters, simulates its response, and
    returns the test data, expected output, the model instance, true parameters, and the rate
    of response

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): Simulated input data.
            - np.random.poisson(rate) (numpy.ndarray): Simulated spike responses.
            - model (nmo.glm.PoissonGLM): Initialized model instance.
            - (w_true, b_true) (tuple): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of response.
    """
    np.random.seed(123)
    X = np.random.normal(size=(100, 5))
    b_true = np.zeros((3,))
    w_true = np.random.normal(size=(5, 3))
    w_true[1:4, 0] = 0.0
    mask = np.zeros((2, 5))
    mask[0, 1:4] = 1
    mask[1, [0, 4]] = 1
    observation_model = nmo.observation_models.PoissonObservations(jnp.exp)
    regularizer = nmo.regularizer.UnRegularized("GradientDescent", {})
    model = nmo.glm.PopulationGLM(observation_model, regularizer)
    rate = jax.numpy.exp(jax.numpy.einsum("k,tk->t", w_true, X) + b_true)
    return X, np.random.poisson(rate), model, (w_true, b_true), rate, mask


@pytest.fixture
def example_data_prox_operator():
    n_features = 4

    params = (
        jnp.ones((n_features)),
        jnp.zeros(
            1,
        ),
    )
    regularizer_strength = 0.1
    mask = jnp.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=jnp.float32)
    scaling = 0.5

    return params, regularizer_strength, mask, scaling


@pytest.fixture
def example_data_prox_operator_multineuron():
    n_features = 4
    n_neurons = 3

    params = (
        jnp.ones((n_features, n_neurons)),
        jnp.zeros(
            n_neurons,
        ),
    )
    regularizer_strength = 0.1
    mask = jnp.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=jnp.float32)
    scaling = 0.5

    return params, regularizer_strength, mask, scaling


@pytest.fixture
def poisson_observation_model():
    return nmo.observation_models.PoissonObservations(jnp.exp)


@pytest.fixture
def ridge_regularizer():
    return nmo.regularizer.Ridge(solver_name="LBFGS", regularizer_strength=0.1)


@pytest.fixture
def lasso_regularizer():
    return nmo.regularizer.Lasso(
        solver_name="ProximalGradient", regularizer_strength=0.1
    )


@pytest.fixture
def group_lasso_2groups_5features_regularizer():
    mask = np.zeros((2, 5))
    mask[0, :2] = 1
    mask[1, 2:] = 1
    return nmo.regularizer.GroupLasso(
        solver_name="ProximalGradient", mask=mask, regularizer_strength=0.1
    )


@pytest.fixture
def mock_data():
    return jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), jnp.array([[1, 2], [3, 4]])


@pytest.fixture()
def glm_class():
    return nmo.glm.GLM


@pytest.fixture()
def population_glm_class():
    return nmo.glm.PopulationGLM


@pytest.fixture
def gammaGLM_model_instantiation():
    """Set up a Gamma GLM for testing purposes.

    This fixture initializes a Gamma GLM with random parameters, simulates its response, and
    returns the test data, expected output, the model instance, true parameters, and the rate
    of response.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): Simulated input data.
            - np.random.poisson(rate) (numpy.ndarray): Simulated spike responses.
            - model (nmo.glm.PoissonGLM): Initialized model instance.
            - (w_true, b_true) (tuple): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of response.
    """
    np.random.seed(123)
    X = np.random.uniform(size=(100, 5))
    b_true = np.zeros((1,))
    w_true = np.random.uniform(size=(5,))
    observation_model = nmo.observation_models.GammaObservations()
    regularizer = nmo.regularizer.UnRegularized("GradientDescent", {})
    model = nmo.glm.GLM(observation_model, regularizer)
    rate = (jax.numpy.einsum("k,tk->t", w_true, X) + b_true) ** -1
    theta = 3
    k = rate / theta
    model.scale = theta
    return X, np.random.gamma(k, scale=theta), model, (w_true, b_true), rate


@pytest.fixture()
def gamma_population_GLM_model():
    np.random.seed(123)
    X = np.random.uniform(size=(500, 5))
    b_true = 0.5 * np.ones((3,))
    w_true = np.random.uniform(size=(5, 3))
    observation_model = nmo.observation_models.GammaObservations()
    regularizer = nmo.regularizer.UnRegularized("GradientDescent", {})
    model = nmo.glm.PopulationGLM(observation_model, regularizer)
    rate = 1 / (jnp.einsum("ki,tk->ti", w_true, X) + b_true)
    theta = 3
    model.scale = theta
    y = jax.random.gamma(jax.random.PRNGKey(123), rate / theta) * theta
    return X, y, model, (w_true, b_true), rate
