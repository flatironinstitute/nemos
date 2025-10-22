"""
Testing configurations for the NeMoS library.

This module contains test fixtures required to set up and verify the functionality
of the modules of the NeMoS library.

Note:
    This module primarily serves as a utility for test configurations, setting up initial conditions,
    and loading predefined parameters for testing various functionalities of the NeMoS library.
"""

import abc
import os
from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pynapple as nap
import pytest

import nemos as nmo
import nemos._inspect_utils as inspect_utils
import nemos.basis.basis as basis
from nemos.basis import AdditiveBasis, CustomBasis, MultiplicativeBasis
from nemos.basis._basis import Basis
from nemos.basis._basis_mixin import BasisMixin
from nemos.basis._transformer_basis import TransformerBasis

DEFAULT_KWARGS = {
    "n_basis_funcs": 5,
    "frequencies": 4,
    "window_size": 11,
    "decay_rates": np.arange(1, 1 + 5),
}

# shut-off conversion warnings
nap.nap_config.suppress_conversion_warnings = True


@pytest.fixture()
def basis_class_specific_params():
    """Returns all the params for each class."""
    all_cls = (
        list_all_basis_classes("Conv") + list_all_basis_classes("Eval") + [CustomBasis]
    )
    return {cls.__name__: cls._get_param_names() for cls in all_cls}


class BasisFuncsTesting(abc.ABC):
    """
    An abstract base class that sets the foundation for individual basis function testing.
    This class requires an implementation of a 'cls' method, which is utilized by the meta-test
    that verifies if all basis functions are properly tested.
    """

    @abc.abstractmethod
    def cls(self):
        pass


def power_func(n, x):
    return jnp.power(x, n)


def custom_basis(n_basis_funcs=5, label=None, **kwargs):
    funcs = [partial(power_func, n) for n in range(1, n_basis_funcs + 1)]
    ndim_input = kwargs.get("ndim_input", 1)
    out_shape = kwargs.get("output_shape", None)
    pynapple_support = kwargs.get("pynapple_support", True)
    return CustomBasis(
        funcs,
        label=label,
        ndim_input=ndim_input,
        output_shape=out_shape,
        pynapple_support=pynapple_support,
    )


def power_add(n, x, y):
    return x**n + y**n


def custom_basis_2d(n_basis_funcs=5, label=None):
    funcs = [partial(power_add, n) for n in range(1, n_basis_funcs + 1)]
    return CustomBasis(funcs, label=label)


def basis_collapse_all_non_vec_axis(n_basis_funcs=5, label=None, **kwargs):
    funcs = [lambda x: x.reshape(x.shape[0], -1)[:, 0] for _ in range(n_basis_funcs)]
    ndim_input = kwargs.get("ndim_input", 1)
    return CustomBasis(funcs, label=label, ndim_input=ndim_input)


def basis_with_add_kwargs(label=None, basis_kwargs=None):
    def func(x, add=0):
        return x + add

    return CustomBasis([func], label=label, basis_kwargs=basis_kwargs)


class CombinedBasis(BasisFuncsTesting):
    """
    This class is used to run tests on combination operations (e.g., addition, multiplication) among Basis functions.

    Properties:
    - cls: Class (default = None)
    """

    cls = None

    @staticmethod
    def instantiate_basis(
        n_basis, basis_class, class_specific_params, window_size=10, **kwargs
    ):
        """Instantiate and return two basis of the type specified."""

        # Set non-optional args
        new_kwargs = {
            "n_basis_funcs": n_basis,
            "window_size": window_size,
            "decay_rates": np.arange(1, 1 + n_basis),
            "frequencies": np.arange(
                (n_basis + 1) % 2, 1 + (n_basis - n_basis % 2) // 2
            ),
            "frequency_mask": None,
        }
        default_kwargs = DEFAULT_KWARGS.copy()
        default_kwargs.update(new_kwargs)
        repeated_keys = set(new_kwargs.keys()).intersection(kwargs.keys())
        if repeated_keys:
            raise ValueError(
                "Cannot set `n_basis_funcs, window_size, decay_rates` with kwargs"
            )

        # Merge with provided  extra kwargs
        kwargs = {**default_kwargs, **kwargs}

        if basis_class == AdditiveBasis:
            kwargs_mspline = inspect_utils.trim_kwargs(
                basis.MSplineEval, kwargs, class_specific_params
            )
            kwargs_raised_cosine = inspect_utils.trim_kwargs(
                basis.RaisedCosineLinearConv, kwargs, class_specific_params
            )
            b1 = basis.MSplineEval(**kwargs_mspline)
            b2 = basis.RaisedCosineLinearConv(**kwargs_raised_cosine)
            basis_obj = b1 + b2
        elif basis_class == MultiplicativeBasis:
            kwargs_mspline = inspect_utils.trim_kwargs(
                basis.MSplineEval, kwargs, class_specific_params
            )
            kwargs_raised_cosine = inspect_utils.trim_kwargs(
                basis.RaisedCosineLinearConv, kwargs, class_specific_params
            )
            b1 = basis.MSplineEval(**kwargs_mspline)
            b2 = basis.RaisedCosineLinearConv(**kwargs_raised_cosine)
            basis_obj = b1 * b2
        elif basis_class == CustomBasis:
            basis_obj = custom_basis(
                n_basis,
                **inspect_utils.trim_kwargs(basis_class, kwargs, class_specific_params),
            )
        else:
            basis_obj = basis_class(
                **inspect_utils.trim_kwargs(basis_class, kwargs, class_specific_params)
            )
        return basis_obj


# automatic define user accessible basis and check the methods
def list_all_basis_classes(filter_basis="all") -> list[BasisMixin]:
    """
    Return all the classes in nemos.basis which are a subclass of Basis,
    which should be all concrete classes except TransformerBasis.
    """
    all_basis = (
        [
            class_obj
            for _, class_obj in inspect_utils.get_non_abstract_classes(basis)
            if issubclass(class_obj, Basis)
        ]
        + [
            bas
            for _, bas in inspect_utils.get_non_abstract_classes(nmo.basis._basis)
            if bas != TransformerBasis
        ]
        + [CustomBasis]
    )
    if filter_basis != "all":
        all_basis = [a for a in all_basis if filter_basis in a.__name__]
    return all_basis


def list_all_real_basis_classes(filter_basis="all"):
    list_all_basis = list_all_basis_classes(filter_basis)
    return [cls for cls in list_all_basis if not getattr(cls, "_is_complex", False)]


# Sample subclass to test instantiation and methods
class MockRegressor(nmo.base_regressor.BaseRegressor):
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

    def update(self, *args, **kwargs):
        pass

    def initialize_state(self, *args, **kwargs):
        pass

    def initialize_params(self, *args, **kwargs):
        pass

    def _predict_and_compute_loss(self, params, X, y):
        pass

    def _get_optimal_solver_params_config(self):
        return None, None, None

    def save_params(self, *args):
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
        feedforward_input,
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
    observation_model = nmo.observation_models.PoissonObservations()
    regularizer = nmo.regularizer.UnRegularized()
    model = nmo.glm.GLM(observation_model, regularizer=regularizer)
    rate = jax.numpy.exp(jax.numpy.einsum("k,tk->t", w_true, X) + b_true)
    return X, np.random.poisson(rate), model, (w_true, b_true), rate


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
    model_tree = nmo.glm.GLM(model.observation_model, regularizer=model.regularizer)
    return X_tree, np.random.poisson(rate), model_tree, true_params_tree, rate


@pytest.fixture
def poissonGLM_fitted_model_instantiation(poissonGLM_model_instantiation):
    X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
    model.fit(X, y)

    return X, y, model, true_params, firing_rate


@pytest.fixture
def population_poissonGLM_model_instantiation():
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
    observation_model = nmo.observation_models.PoissonObservations()
    regularizer = nmo.regularizer.UnRegularized()
    model = nmo.glm.PopulationGLM(
        observation_model=observation_model, regularizer=regularizer
    )
    rate = jnp.exp(jnp.einsum("ki,tk->ti", w_true, X) + b_true)
    return X, np.random.poisson(rate), model, (w_true, b_true), rate


@pytest.fixture
def population_poissonGLM_model_instantiation_pytree(
    population_poissonGLM_model_instantiation,
):
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
    X, spikes, model, true_params, rate = population_poissonGLM_model_instantiation
    X_tree = nmo.pytrees.FeaturePytree(input_1=X[..., :3], input_2=X[..., 3:])
    true_params_tree = (
        dict(input_1=true_params[0][:3], input_2=true_params[0][3:]),
        true_params[1],
    )
    model_tree = nmo.glm.PopulationGLM(
        observation_model=model.observation_model, regularizer=model.regularizer
    )
    return X_tree, np.random.poisson(rate), model_tree, true_params_tree, rate


@pytest.fixture
def population_poissonGLM_fitted_model_instantiation(
    population_poissonGLM_model_instantiation,
):
    X, y, model, true_params, firing_rate = population_poissonGLM_model_instantiation
    model.fit(X, y)

    return X, y, model, true_params, firing_rate


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
    basis = nmo.basis.RaisedCosineLogEval(20)

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
def poissonGLM_model_instantiation_group_sparse():
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
    observation_model = nmo.observation_models.PoissonObservations()
    regularizer = nmo.regularizer.UnRegularized()
    model = nmo.glm.GLM(observation_model, regularizer=regularizer)
    rate = jax.numpy.exp(jax.numpy.einsum("k,tk->t", w_true, X) + b_true)
    return X, np.random.poisson(rate), model, (w_true, b_true), rate, mask


@pytest.fixture
def population_poissonGLM_model_instantiation_group_sparse():
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
    observation_model = nmo.observation_models.PoissonObservations()
    regularizer = nmo.regularizer.UnRegularized()
    model = nmo.glm.PopulationGLM(observation_model, regularizer=regularizer)
    rate = jax.numpy.exp(jax.numpy.einsum("kn,tk->tn", w_true, X) + b_true)
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
    mask = jnp.array([[1, 0, 1, 0], [0, 1, 0, 1]]).astype(float)
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
    return nmo.regularizer.Ridge()


@pytest.fixture
def lasso_regularizer():
    return nmo.regularizer.Lasso(solver_name="ProximalGradient")


@pytest.fixture
def group_lasso_2groups_5features_regularizer():
    mask = np.zeros((2, 5))
    mask[0, :2] = 1
    mask[1, 2:] = 1
    return nmo.regularizer.GroupLasso(solver_name="ProximalGradient", mask=mask)


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
    regularizer = nmo.regularizer.UnRegularized()
    model = nmo.glm.GLM(observation_model, regularizer=regularizer)
    rate = (jax.numpy.einsum("k,tk->t", w_true, X) + b_true) ** -1
    theta = 3
    k = rate / theta
    model.scale_ = theta
    return X, np.random.gamma(k, scale=theta), model, (w_true, b_true), rate


@pytest.fixture
def gammaGLM_model_instantiation_pytree(gammaGLM_model_instantiation):
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
    X, spikes, model, true_params, rate = gammaGLM_model_instantiation
    X_tree = nmo.pytrees.FeaturePytree(input_1=X[..., :3], input_2=X[..., 3:])
    true_params_tree = (
        dict(input_1=true_params[0][:3], input_2=true_params[0][3:]),
        true_params[1],
    )
    model_tree = nmo.glm.GLM(model.observation_model, regularizer=model.regularizer)
    return X_tree, spikes, model_tree, true_params_tree, rate


@pytest.fixture()
def population_gammaGLM_model_instantiation():
    np.random.seed(123)
    X = np.random.uniform(size=(500, 5))
    b_true = 0.5 * np.ones((3,))
    w_true = np.random.uniform(size=(5, 3))
    observation_model = nmo.observation_models.GammaObservations()
    regularizer = nmo.regularizer.UnRegularized()
    model = nmo.glm.PopulationGLM(
        observation_model=observation_model, regularizer=regularizer
    )
    rate = 1 / (jnp.einsum("ki,tk->ti", w_true, X) + b_true)
    theta = 3
    model.scale_ = theta
    y = jax.random.gamma(jax.random.PRNGKey(123), rate / theta) * theta
    return X, y, model, (w_true, b_true), rate


@pytest.fixture()
def population_gammaGLM_model_instantiation_pytree(
    population_gammaGLM_model_instantiation,
):
    X, spikes, model, true_params, rate = population_gammaGLM_model_instantiation
    X_tree = nmo.pytrees.FeaturePytree(input_1=X[..., :3], input_2=X[..., 3:])
    true_params_tree = (
        dict(input_1=true_params[0][:3], input_2=true_params[0][3:]),
        true_params[1],
    )
    model_tree = nmo.glm.PopulationGLM(
        observation_model=model.observation_model, regularizer=model.regularizer
    )
    return X_tree, spikes, model_tree, true_params_tree, rate


@pytest.fixture
def regr_data():
    np.random.seed(123)
    # define inputs and coeff
    n_samples, n_features = 50, 3
    X = np.random.normal(size=(n_samples, n_features))
    coef = np.random.normal(size=(n_features))
    # set y according to lin reg eqn
    y = X.dot(coef) + 0.1 * np.random.normal(size=(n_samples,))
    return X, y, coef


@pytest.fixture
def linear_regression(regr_data):
    X, y, coef = regr_data
    # solve least-squares
    ols, _, _, _ = np.linalg.lstsq(X, y, rcond=-1)

    # set the loss
    def loss(params, X, y):
        return jnp.power(y - jnp.dot(X, params), 2).mean()

    return X, y, coef, ols, loss


@pytest.fixture
def ridge_regression(regr_data):
    X, y, coef = regr_data

    # solve least-squares
    yagu = np.hstack((y, np.zeros_like(coef)))
    Xagu = np.vstack((X, np.sqrt(0.5) * np.eye(coef.shape[0])))
    ridge, _, _, _ = np.linalg.lstsq(Xagu, yagu, rcond=-1)

    # set the loss
    def loss(params, XX, yy):
        return (
            jnp.power(yy - jnp.dot(XX, params), 2).sum()
            + 0.5 * jnp.power(params, 2).sum()
        )

    return X, y, coef, ridge, loss


@pytest.fixture
def linear_regression_tree(linear_regression):
    X, y, coef, ols, loss = linear_regression
    X_tree = dict(input_1=X[..., :2], input_2=X[..., 2:])
    coef_tree = dict(input_1=coef[:2], input_2=coef[2:])
    ols_tree = dict(input_1=ols[:2], input_2=ols[2:])

    nmo.tree_utils.pytree_map_and_reduce(jnp.dot, sum, X_tree, coef_tree)

    def loss_tree(params, XX, yy):
        pred = nmo.tree_utils.pytree_map_and_reduce(jnp.dot, sum, XX, params)
        return jnp.power(yy - pred, 2).sum()

    return X_tree, y, coef_tree, ols_tree, loss_tree


@pytest.fixture()
def ridge_regression_tree(ridge_regression):
    X, y, coef, ridge, loss = ridge_regression
    X_tree = dict(input_1=X[..., :2], input_2=X[..., 2:])
    coef_tree = dict(input_1=coef[:2], input_2=coef[2:])
    ridge_tree = dict(input_1=ridge[:2], input_2=ridge[2:])

    def loss_tree(params, XX, yy):
        pred = nmo.tree_utils.pytree_map_and_reduce(jnp.dot, sum, XX, params)
        norm = (
            0.5
            * nmo.tree_utils.pytree_map_and_reduce(
                lambda x: jnp.power(x, 2).sum(), sum, params
            ).sum()
        )
        return jnp.power(yy - pred, 2).sum() + norm

    return X_tree, y, coef_tree, ridge_tree, loss_tree


@pytest.fixture()
def example_X_y_high_firing_rates():
    """Example that used failed with NeMoS original initialization."""
    np.random.seed(123)

    n_features = 18
    n_neurons = 60
    n_samples = 500

    # random design array. Shape (n_time_points, n_features).
    X = 0.5 * np.random.normal(size=(n_samples, n_features))

    # log-rates & weights
    b_true = np.random.uniform(size=(n_neurons,)) * 3  # baseline rates
    w_true = np.random.uniform(size=(n_features, n_neurons)) * 0.1  # real weights:

    # generate counts (spikes will be (n_samples, n_features)
    rate = jnp.exp(jnp.dot(X, w_true) + b_true)
    spikes = np.random.poisson(rate)
    return X, spikes


@pytest.fixture
def bernoulliGLM_model_instantiation():
    """Set up a Bernoulli GLM for testing purposes.

    This fixture initializes a Bernoulli GLM with random parameters, simulates its response, and
    returns the test data, expected output, the model instance, true parameters, and the rate
    of response.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): Simulated input data.
            - np.random.binomial(1,rate) (numpy.ndarray): Simulated spike responses.
            - model (nmo.glm.GLM): Initialized model instance.
            - (w_true, b_true) (tuple): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of response.
    """
    np.random.seed(123)
    X = np.random.normal(size=(100, 5))
    b_true = np.zeros((1,))
    w_true = np.random.normal(size=(5,))
    observation_model = nmo.observation_models.BernoulliObservations()
    regularizer = nmo.regularizer.UnRegularized()
    model = nmo.glm.GLM(observation_model, regularizer=regularizer)
    rate = jax.lax.logistic(jnp.einsum("k,tk->t", w_true, X) + b_true)
    return X, np.random.binomial(1, rate), model, (w_true, b_true), rate


@pytest.fixture
def bernoulliGLM_model_instantiation_pytree(bernoulliGLM_model_instantiation):
    """Set up a Bernoulli GLM for testing purposes.

    This fixture initializes a Bernoulli GLM with random parameters, simulates its response, and
    returns the test data, expected output, the model instance, true parameters, and the rate
    of response.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): Simulated input data.
            - np.random.binomial(1,rate) (numpy.ndarray): Simulated spike responses.
            - model (nmo.glm.GLM): Initialized model instance.
            - (w_true, b_true) (tuple): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of response.
    """
    X, spikes, model, true_params, rate = bernoulliGLM_model_instantiation
    X_tree = nmo.pytrees.FeaturePytree(input_1=X[..., :3], input_2=X[..., 3:])
    true_params_tree = (
        dict(input_1=true_params[0][:3], input_2=true_params[0][3:]),
        true_params[1],
    )
    model_tree = nmo.glm.GLM(model.observation_model, regularizer=model.regularizer)
    return X_tree, np.random.binomial(1, rate), model_tree, true_params_tree, rate


@pytest.fixture
def population_bernoulliGLM_model_instantiation():
    """Set up a population Bernoulli GLM for testing purposes.

    This fixture initializes a Bernoulli GLM with random parameters, simulates its response, and
    returns the test data, expected output, the model instance, true parameters, and the rate
    of response.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): Simulated input data.
            - np.random.binomial(1,rate) (numpy.ndarray): Simulated spike responses.
            - model (nmo.glm.PopulationGLM): Initialized model instance.
            - (w_true, b_true) (tuple): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of response.
    """
    np.random.seed(123)
    X = np.random.normal(size=(500, 5))
    b_true = np.zeros((3,))
    w_true = np.random.normal(size=(5, 3))
    observation_model = nmo.observation_models.BernoulliObservations()
    regularizer = nmo.regularizer.UnRegularized()
    model = nmo.glm.PopulationGLM(
        observation_model=observation_model, regularizer=regularizer
    )
    rate = jax.lax.logistic(jnp.einsum("ki,tk->ti", w_true, X) + b_true)
    return X, np.random.binomial(1, rate), model, (w_true, b_true), rate


@pytest.fixture
def population_bernoulliGLM_model_instantiation_pytree(
    population_bernoulliGLM_model_instantiation,
):
    """Set up a population Bernoulli GLM for testing purposes.

    This fixture initializes a Bernoulli GLM with random parameters, simulates its response, and
    returns the test data, expected output, the model instance, true parameters, and the rate
    of response.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): Simulated input data.
            - np.random.binomial(1,rate) (numpy.ndarray): Simulated spike responses.
            - model (nmo.glm.PopulationGLM): Initialized model instance.
            - (w_true, b_true) (tuple): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of response.
    """
    X, spikes, model, true_params, rate = population_bernoulliGLM_model_instantiation
    X_tree = nmo.pytrees.FeaturePytree(input_1=X[..., :3], input_2=X[..., 3:])
    true_params_tree = (
        dict(input_1=true_params[0][:3], input_2=true_params[0][3:]),
        true_params[1],
    )
    model_tree = nmo.glm.PopulationGLM(
        observation_model=model.observation_model, regularizer=model.regularizer
    )
    return X_tree, np.random.binomial(1, rate), model_tree, true_params_tree, rate


SizeTerminal = namedtuple("SizeTerminal", ["columns", "lines"])


class NestedRegularizer(nmo.regularizer.Ridge):
    def __init__(self, sub_regularizer, func=np.exp):
        self.sub_regularizer = sub_regularizer
        self.func = func
        super().__init__()


@pytest.fixture
def nested_regularizer():
    """
    Nested retularizer for testing save/load.
    """
    return NestedRegularizer(nmo.regularizer.Lasso())


@pytest.fixture
def negativeBinomialGLM_model_instantiation():
    """Set up a Negative Binomial GLM with array inputs for testing purposes.

    This fixture initializes a Negative Binomial GLM with random parameters, simulates its response, and
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
    observation_model = nmo.observation_models.NegativeBinomialObservations()
    regularizer = nmo.regularizer.UnRegularized()
    model = nmo.glm.GLM(observation_model, regularizer=regularizer, solver_name="LBFGS")
    rate = jax.numpy.exp(jax.numpy.einsum("k,tk->t", w_true, X) + b_true)
    r = 1 / model.observation_model.scale
    spikes = np.random.poisson(np.random.gamma(shape=r, size=rate.shape) * (r / rate))
    return X, spikes, model, (w_true, b_true), rate


@pytest.fixture
def negativeBinomialGLM_model_instantiation_pytree(
    negativeBinomialGLM_model_instantiation,
):
    """Set up a Negative Binomial GLM with pytree inputs for testing purposes .

    This fixture initializes a Negative Binomial GLM with random parameters, simulates its response, and
    returns the test data, expected output, the model instance, true parameters, and the rate
    of response.

    Returns:
        tuple: A tuple containing:
            - X (FeaturePytree): Simulated input data.
            - np.random.poisson(rate) (numpy.ndarray): Simulated spike responses.
            - model (nmo.glm.PoissonGLM): Initialized model instance.
            - (w_true, b_true) (tuple): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of response.
    """
    X, spikes, model, true_params, rate = negativeBinomialGLM_model_instantiation
    X_tree = nmo.pytrees.FeaturePytree(input_1=X[..., :3], input_2=X[..., 3:])
    true_params_tree = (
        dict(input_1=true_params[0][:3], input_2=true_params[0][3:]),
        true_params[1],
    )
    model_tree = nmo.glm.GLM(
        model.observation_model, regularizer=model.regularizer, solver_name="LBFGS"
    )
    return X_tree, np.random.poisson(rate), model_tree, true_params_tree, rate


@pytest.fixture
def population_negativeBinomialGLM_model_instantiation():
    """Set up a population Negative Binomial GLM for testing purposes.

    This fixture initializes a Negative Binomial GLM with random parameters, simulates its response, and
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
    w_true = 0.1 * np.random.normal(size=(5, 3))
    observation_model = nmo.observation_models.NegativeBinomialObservations()
    regularizer = nmo.regularizer.UnRegularized()
    model = nmo.glm.PopulationGLM(
        observation_model=observation_model,
        regularizer=regularizer,
        solver_name="LBFGS",
    )
    rate = jnp.exp(jnp.einsum("ki,tk->ti", w_true, X) + b_true)
    spikes = model.observation_model.sample_generator(jax.random.PRNGKey(123), rate)
    # make sure that at least one entry is non-zero
    spikes = spikes.at[-1].set(1)
    return X, spikes, model, (w_true, b_true), rate


@pytest.fixture
def population_negativeBinomialGLM_model_instantiation_pytree(
    population_negativeBinomialGLM_model_instantiation,
):
    """Set up a population Negative Binomial GLM for testing purposes.

    This fixture initializes a Negative Binomial GLM with random parameters, simulates its response, and
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
    X, spikes, model, true_params, rate = (
        population_negativeBinomialGLM_model_instantiation
    )
    X_tree = nmo.pytrees.FeaturePytree(input_1=X[..., :3], input_2=X[..., 3:])
    true_params_tree = (
        dict(input_1=true_params[0][:3], input_2=true_params[0][3:]),
        true_params[1],
    )
    model_tree = nmo.glm.PopulationGLM(
        observation_model=model.observation_model,
        regularizer=model.regularizer,
        solver_name="LBFGS",
    )
    return X_tree, np.random.poisson(rate), model_tree, true_params_tree, rate


# Select solver backend for tests if requested via environment variable
_common_solvers = {
    "SVRG": nmo.solvers.WrappedSVRG,
    "ProxSVRG": nmo.solvers.WrappedProxSVRG,
}
_solver_registry_per_backend = {
    "jaxopt": {
        **_common_solvers,
        "GradientDescent": nmo.solvers.JaxoptGradientDescent,
        "ProximalGradient": nmo.solvers.JaxoptProximalGradient,
        "LBFGS": nmo.solvers.JaxoptLBFGS,
        "BFGS": nmo.solvers.JaxoptBFGS,
        "NonlinearCG": nmo.solvers.JaxoptNonlinearCG,
    },
    "optimistix": {
        **_common_solvers,
        "GradientDescent": nmo.solvers.OptimistixOptaxGradientDescent,
        "ProximalGradient": nmo.solvers.OptimistixOptaxProximalGradient,
        "LBFGS": nmo.solvers.OptimistixOptaxLBFGS,
        "BFGS": nmo.solvers.OptimistixBFGS,
        "NonlinearCG": nmo.solvers.OptimistixNonlinearCG,
    },
}


@pytest.fixture(autouse=True, scope="session")
def configure_solver_backend():
    """
    Patch the solver registry depending on ``NEMOS_SOLVER_BACKEND``.

    Used for running solver-dependent tests in separate tox environments
    for the JAXopt and the Optimistix backends.
    """
    backend = os.getenv("NEMOS_SOLVER_BACKEND")
    if not backend:
        yield  # run with default solver registry
        return  # don't execute the remainder on teardown

    try:
        _backend_solver_registry = _solver_registry_per_backend[backend]
    except KeyError:
        available = ", ".join(_solver_registry_per_backend.keys())
        pytest.fail(f"Unknown solver backend: {backend}. Available: {available}")

    # save the original registry so that we can restore it after
    original = nmo.solvers.solver_registry.copy()
    nmo.solvers.solver_registry.clear()
    nmo.solvers.solver_registry.update(_backend_solver_registry)

    try:
        yield
    finally:
        nmo.solvers.solver_registry.clear()
        nmo.solvers.solver_registry.update(original)
