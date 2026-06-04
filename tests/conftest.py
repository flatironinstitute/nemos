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
import re
from collections import defaultdict, namedtuple
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Any, Callable, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import pynapple as nap
import pytest

import nemos as nmo
import nemos._inspect_utils as inspect_utils
import nemos.basis.basis as basis
from nemos.base_regressor import BaseRegressor
from nemos.base_validator import RegressorValidator
from nemos.basis import AdditiveBasis, Category, CustomBasis, MultiplicativeBasis, Zero
from nemos.basis._basis import Basis
from nemos.basis._basis_mixin import BasisMixin
from nemos.basis._transformer_basis import TransformerBasis
from nemos.glm.params import GLMParams
from nemos.glm.validation import GLMValidator
from nemos.glm_hmm.initialize_parameters import random_glm_params_init
from nemos.glm_hmm.params import GLMHMMModelParams, GLMHMMParams
from nemos.hmm.hmm import BaseHMM
from nemos.hmm.initialize_parameters import (
    HMM_INITIALIZATION_FN_DICT,
    sticky_transition_proba_init,
    uniform_initial_proba_init,
)
from nemos.hmm.params import HMMParams
from nemos.hmm.utils import initialize_session_starts
from nemos.hmm.validation import HMMValidator, from_hmm_params, to_hmm_params
from nemos.params import ModelParams
from nemos.tree_utils import tree_full_like

_totals = defaultdict(float)
_counts = defaultdict(int)

_TIMEIT_ENABLED = False


def pytest_configure(config):
    global _TIMEIT_ENABLED
    _TIMEIT_ENABLED = config.getoption("timeit")


def pytest_runtest_logreport(report):
    if not _TIMEIT_ENABLED:
        return
    if report.when == "call":
        # strip [param] suffix to group by base test name
        base = re.sub(r"\[.*\]$", "", report.nodeid)
        _totals[base] += report.duration
        _counts[base] += 1


def pytest_terminal_summary(terminalreporter):
    if not _TIMEIT_ENABLED:
        return
    terminalreporter.write_sep("=", "aggregated param durations")
    rows = sorted(_totals.items(), key=lambda x: -x[1])
    for name, total in rows:
        n = _counts[name]
        avg = total / n
        terminalreporter.write_line(
            f"{total:6.2f}s total  {avg:.4f}s/param  {n:>5} params  {name}"
        )


@pytest.fixture
def mock_glm_fit(monkeypatch):
    """Replace GLM.fit with a pure-Python no-op that sets coef_/intercept_ from X/y shapes.

    Use in tests that only care about sklearn routing (cloning, parameter
    setting, pipeline plumbing) and do not need any fit validation logic.
    For tests that need validation but not solver iterations, use mock_optimizer_run.
    """

    def _fit(self, X, y, init_params=None, **kwargs):
        n_features = X.shape[1]
        y_arr = y.d if hasattr(y, "d") else np.asarray(y)
        if y_arr.ndim == 1:
            self.coef_ = jnp.zeros(n_features)
            self.intercept_ = jnp.zeros(1)
        else:
            self.coef_ = jnp.zeros((n_features, y_arr.shape[1]))
            self.intercept_ = jnp.zeros(y_arr.shape[1])
        self.scale_ = jnp.ones_like(self.intercept_)
        return self

    monkeypatch.setattr(nmo.glm.GLM, "fit", _fit)
    monkeypatch.setattr(nmo.glm.PopulationGLM, "fit", _fit)


# No-op _optimizer_run per model class. Only models whose fit() unpacks _optimizer_run
# differently from the default 3-tuple (params, state, aux) need an entry here.
# The sole model-specific detail is return arity: GLM expects (params, state, aux),
# GLMHMM expects (params, state) and checks state.iterations to detect non-convergence.
_NOOP_OPTIMIZER_RUN = {
    nmo.glm_hmm.GLMHMM: lambda p, *a, **kw: (
        p,
        SimpleNamespace(iterations=1, converged=True),
    ),
}
_DEFAULT_NOOP_OPTIMIZER_RUN = lambda p, *a, **kw: (  # noqa: E731
    p,
    SimpleNamespace(converged=True),
    None,
)


def _make_optimizer_run_patch(monkeypatch, model_cls):
    """Patch _initialize_optimizer_and_state on any BaseRegressor subclass.

    After the real initializer runs, replaces _optimizer_run with a no-op that
    returns init_params unchanged and a converged state. All validation logic
    executes normally; only the solver iterations are skipped.
    """
    real_init = model_cls._initialize_optimizer_and_state
    noop = _NOOP_OPTIMIZER_RUN.get(model_cls, _DEFAULT_NOOP_OPTIMIZER_RUN)

    def _patched(self, init_params, data, y):
        result = real_init(self, init_params, data, y)
        self._optimizer_run = noop
        return result

    monkeypatch.setattr(model_cls, "_initialize_optimizer_and_state", _patched)


@pytest.fixture
def patch_optimizer_run(monkeypatch):
    """Fixture factory: call with a model class to bypass its JAX solver.

    Returns a callable ``patch(model_cls)`` that patches
    ``_initialize_optimizer_and_state`` on *model_cls* so that ``_optimizer_run``
    becomes a no-op returning init_params unchanged with a converged state.
    Can be called multiple times in one test to patch several classes.
    """
    return lambda model_cls: _make_optimizer_run_patch(monkeypatch, model_cls)


@pytest.fixture
def mock_glm_optimizer_run(monkeypatch):
    """Bypass the JAX solver in GLM.fit() while keeping all validation logic."""
    _make_optimizer_run_patch(monkeypatch, nmo.glm.GLM)


@pytest.fixture
def mock_glm_hmm_optimizer_run(monkeypatch):
    """Bypass the JAX solver in GLMHMM.fit() while keeping all validation logic."""
    _make_optimizer_run_patch(monkeypatch, nmo.glm_hmm.GLMHMM)


# No-op _optimizer_update per model class. Default covers GLM-family models (3-tuple);
# GLM-HMM's update() unpacks a 2-tuple (params, state).
# Add an entry only when a model's update() unpacks _optimizer_update differently.
_NOOP_OPTIMIZER_UPDATE = {
    nmo.glm_hmm.GLMHMM: lambda p, s, *a, **kw: (p, s),  # noqa: E731
}
_DEFAULT_NOOP_OPTIMIZER_UPDATE = lambda p, s, *a, **kw: (p, s, None)  # noqa: E731


def _make_optimizer_update_patch(monkeypatch, model_cls):
    """Patch _initialize_optimizer_and_state on any BaseRegressor subclass.

    After the real initializer runs, replaces _optimizer_update with a no-op
    that returns params and state unchanged. All validation logic executes
    normally; only the solver step is skipped.
    """
    real_init = model_cls._initialize_optimizer_and_state
    noop = _NOOP_OPTIMIZER_UPDATE.get(model_cls, _DEFAULT_NOOP_OPTIMIZER_UPDATE)

    def _patched(self, init_params, data, y):
        result = real_init(self, init_params, data, y)
        self._optimizer_update = noop
        return result

    monkeypatch.setattr(model_cls, "_initialize_optimizer_and_state", _patched)


@pytest.fixture
def patch_optimizer_update(monkeypatch):
    """Fixture factory: call with a model class to bypass its JAX solver step.

    Returns a callable ``patch(model_cls)`` that patches
    ``_initialize_optimizer_and_state`` on *model_cls* so that
    ``_optimizer_update`` becomes a no-op returning params and state unchanged.
    Can be called multiple times in one test to patch several classes.
    """
    return lambda model_cls: _make_optimizer_update_patch(monkeypatch, model_cls)


@pytest.fixture
def mock_optimizer_update(monkeypatch):
    """Bypass the JAX solver step in GLM.update() while keeping all validation logic."""
    _make_optimizer_update_patch(monkeypatch, nmo.glm.GLM)


# Named tuple for model fixture returns (clearer than tuple indexing)
ModelFixture = namedtuple(
    "ModelFixture",
    ["X", "y", "model", "params", "rates", "extra"],
    defaults=[None, None],  # rates and extra default to None
)


def initialize_feature_mask_for_population_glm(
    X, n_neurons: int, n_classes: int = 0, coef=None
):
    """
    Create a feature mask of ones for PopulationGLM testing.

    This is a test utility function that creates a feature mask with all ones,
    matching the structure of X.

    Parameters
    ----------
    X :
        The design matrix. Can be a dict or array.
    n_neurons :
        Number of neurons (determines the second dimension of the mask).
        Ignored if coef is provided.
    coef :
        Optional coefficient array/pytree. If provided, the mask shape will match
        coef shape exactly (required for ClassifierPopulationGLM).
    n_classes:
        Number of classes (determines the second dimension of the mask).

    Returns
    -------
    :
        A feature mask with all ones. If coef is provided, returns ones_like(coef).
        Otherwise, if X is a dict, returns a dict with arrays
        of shape (n_neurons,) for each key.
        If X is an array, returns an array of shape (n_features, n_neurons).
    """
    extra_shape = (n_classes,) if n_classes else ()
    if coef is not None:
        return jax.tree_util.tree_map(lambda c: jnp.ones(c.shape), coef)
    if isinstance(X, dict):
        return jax.tree_util.tree_map(lambda x: jnp.ones((n_neurons, *extra_shape)), X)
    else:
        return jnp.ones((X.shape[1], n_neurons, *extra_shape))


DEFAULT_KWARGS = {
    "n_basis_funcs": 5,
    "frequencies": 4,
    "window_size": 11,
    "decay_rates": np.arange(1, 1 + 5),
    "categories": 4,
}

# shut-off conversion warnings
nap.nap_config.suppress_conversion_warnings = True


@pytest.fixture(autouse=True, scope="function")
def set_jax_precision_per_test(request):
    """
    Automatically set JAX precision based on test marker.

    Tests marked with @pytest.mark.requires_x64 get float64.
    All other tests get float32 (JAX default).

    This fixture runs automatically for every test to ensure consistent
    precision behavior, especially important for parallel test execution
    with pytest-xdist.
    """
    if request.node.get_closest_marker("requires_x64"):
        # This test needs x64
        original = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", True)
        yield
        jax.config.update("jax_enable_x64", original)
    else:
        # Default: float32
        original = jax.config.jax_enable_x64
        jax.config.update("jax_enable_x64", False)
        yield
        jax.config.update("jax_enable_x64", original)


@pytest.fixture()
def basis_class_specific_params():
    """Returns all the params for each class."""
    all_cls = (
        list_all_basis_classes("Conv")
        + list_all_basis_classes("Eval")
        + [CustomBasis, Zero, Category]
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
    bounds = kwargs.get("bounds", None)
    fill_value = kwargs.get("fill_value", float("nan"))
    return CustomBasis(
        funcs,
        label=label,
        ndim_input=ndim_input,
        output_shape=out_shape,
        pynapple_support=pynapple_support,
        bounds=bounds,
        fill_value=fill_value,
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


def is_eval_basis(basis_cls) -> bool:
    is_eval = "Eval" in basis_cls.__name__ or issubclass(
        basis_cls, (basis.Zero, Category)
    )
    return is_eval


def is_conv_basis(basis_cls) -> bool:
    is_eval = "Conv" in basis_cls.__name__
    return is_eval


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
        + [CustomBasis, Category]
    )
    if filter_basis != "all":
        cond_fn = is_eval_basis if filter_basis == "Eval" else is_conv_basis
        all_basis = [a for a in all_basis if cond_fn(a)]
    return all_basis


def list_all_real_basis_classes(filter_basis="all"):
    list_all_basis = list_all_basis_classes(filter_basis)
    return [cls for cls in list_all_basis if not getattr(cls, "_is_complex", False)]


# Sample subclass to test instantiation and methods
class MockRegressor(BaseRegressor):
    """
    Mock implementation of the BaseRegressor abstract class for testing purposes.
    Implements all required abstract methods as empty methods.
    """

    _validator = GLMValidator()

    def __init__(self, std_param: int = 0):
        """Initialize a MockBaseRegressor instance with optional standard parameters."""
        self.std_param = std_param
        self._solver_spec = None
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

    def _get_model_params(self):
        pass

    def _set_model_params(self, params):
        pass

    def update(self, *args, **kwargs):
        pass

    def _initialize_optimizer_and_state(self, *args, **kwargs):
        pass

    def initialize_params(self, *args, **kwargs):
        pass

    def _initialize_parameters(self, *args, **kwargs):
        pass

    def _compute_loss(self, params, X, y):
        pass

    def _get_optimal_solver_params_config(self):
        return None, None, None

    def save_params(self, *args):
        pass

    def _model_specific_initialization(self, *args, **kwargs):
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

    def _get_model_params(self):
        pass

    def _set_model_params(self, params):
        pass


MockHMMUserParams = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]


class MockHMMModelParams(ModelParams):
    param: jnp.ndarray


class MockHMMParams(ModelParams):
    model_params: MockHMMModelParams
    hmm_params: HMMParams


def to_mock_params(user_params: MockHMMUserParams) -> MockHMMParams:
    return MockHMMParams(
        model_params=MockHMMModelParams(user_params[0]),
        hmm_params=to_hmm_params(user_params[1:]),
    )


def from_mock_params(params: MockHMMParams) -> MockHMMUserParams:
    initial_prob, transition_prob = from_hmm_params(params.hmm_params)
    return (
        params.model_params.param,
        initial_prob,
        transition_prob,
    )


@dataclass(frozen=True, repr=False)
class MockHMMValidator(HMMValidator[MockHMMUserParams, MockHMMParams]):
    model_param_names: Tuple[str] = (
        "param",
        *HMMValidator.model_param_names,
    )
    to_model_params: Callable[[MockHMMUserParams], MockHMMParams] = to_mock_params
    from_model_params: Callable[[MockHMMParams], MockHMMUserParams] = from_mock_params
    model_class: str = "MockHMM"
    X_dimensionality: int = 2
    y_dimensionality: int = 1
    params_validation_sequence: Tuple[Tuple[str, None] | Tuple[str, dict[str, Any]]] = (
        *RegressorValidator.params_validation_sequence[:2],
        *HMMValidator.params_validation_sequence,
        *RegressorValidator.params_validation_sequence[3:],
    )

    def validate_consistency(self, *args, **kwargs) -> None:
        return True


class MockHMM(
    BaseHMM[
        MockHMMParams, MockHMMUserParams, HMM_INITIALIZATION_FN_DICT, MockHMMValidator
    ]
):
    _validator_class = MockHMMValidator
    _model_default_init_dict = {
        "param_init": None,
        "param_init_kwargs": {},
        "param_init_custom": False,
    }

    def __init__(
        self,
        n_states: int,
        dirichlet_initial_proba: Union[jnp.ndarray, None] = None,  # (n_state, )
        dirichlet_transition_proba: Union[
            jnp.ndarray | None
        ] = None,  # (n_state, n_state):
        maxiter: int = 1000,
        tol: float = 1e-8,
        seed=jax.random.PRNGKey(123),
        hmm_initialization_funcs: HMM_INITIALIZATION_FN_DICT = None,
        model_initialization_funcs: HMM_INITIALIZATION_FN_DICT = None,
    ):
        BaseHMM.__init__(
            self,
            n_states=n_states,
            dirichlet_initial_proba=dirichlet_initial_proba,
            dirichlet_transition_proba=dirichlet_transition_proba,
            maxiter=maxiter,
            tol=tol,
            seed=seed,
            hmm_initialization_funcs=hmm_initialization_funcs,
        )
        self.param_: jnp.ndarray | None = None
        self.model_initialization_funcs = model_initialization_funcs

    def _model_setup(
        self,
        param_init: Optional[str | Callable] = None,
        param_init_kwargs: Optional[dict] = None,
    ):
        self._model_use_kmeans = {"param_init": param_init == "kmeans"}

    def _check_model_is_fit(self):
        if self.param_ is None:
            raise ValueError("Model is not fitted yet.")

    def _get_model_params(self) -> MockHMMParams:
        return self._validator.to_model_params(
            (
                self.param_,
                self.initial_prob_,
                self.transition_prob_,
            )
        )

    def _set_model_params(self, params):
        param, initial_prob, transition_prob = self._validator.from_model_params(params)
        self.param_ = param
        self.initial_prob_ = initial_prob
        self.transition_prob_ = transition_prob

    def _log_likelihood(self, params, X, y):
        return jnp.zeros((y.shape[0], self.n_states))

    def _model_params_initialization(self, X, y, session_starts, random_key=None):
        return (
            jnp.arange(self._n_states),
            False,
        )

    def fit(self, X, y, session_starts=None, init_params=None):
        session_starts = initialize_session_starts(X, y, session_starts)
        fit_params = self._model_specific_initialization(X, y, session_starts)
        self._set_model_params(fit_params)

    def _initialize_optimizer_and_state(self, *args, **kwargs):
        pass

    def _compute_loss(self, *args, **kwargs):
        pass

    def _get_optimal_solver_params_config(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass

    def simulate(self, *args, **kwargs):
        pass

    def save_params(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def score(self, *args, **kwargs):
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
            - GLMParams(w_true, b_true) (tuple): True weight and bias parameters.
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
    return X, np.random.poisson(rate), model, GLMParams(w_true, b_true), rate


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
            - GLMParams(w_true, b_true) (tuple): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of response.
    """
    X, spikes, model, true_params, rate = poissonGLM_model_instantiation
    X_tree = {"input_1": X[..., :3], "input_2": X[..., 3:]}
    true_params_tree = GLMParams(
        dict(input_1=true_params.coef[:3], input_2=true_params.coef[3:]),
        true_params.intercept,
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
            - GLMParams(w_true, b_true) (tuple): True weight and bias parameters.
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
    return X, np.random.poisson(rate), model, GLMParams(w_true, b_true), rate


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
            - GLMParams(w_true, b_true) (tuple): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of response.
    """
    X, spikes, model, true_params, rate = population_poissonGLM_model_instantiation
    X_tree = {"input_1": X[..., :3], "input_2": X[..., 3:]}
    true_params_tree = GLMParams(
        dict(input_1=true_params.coef[:3], input_2=true_params.coef[3:]),
        true_params.intercept,
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
            - GLMParams(w_true, b_true) (tuple): True weight and bias parameters.
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
    return X, np.random.poisson(rate), model, GLMParams(w_true, b_true), rate, mask


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
            - GLMParams(w_true, b_true) (tuple): True weight and bias parameters.
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
    return X, np.random.poisson(rate), model, GLMParams(w_true, b_true), rate, mask


@pytest.fixture
def example_data_prox_operator():
    n_features = 4

    params = GLMParams(
        jnp.ones((n_features)),
        jnp.zeros(1),
    )
    regularizer_strength = tree_full_like(params, 0.1)
    # Mask as PyTree with same structure as params, shape (n_groups, *param_shape)
    # Intercept mask is zeros (not regularized)
    mask = GLMParams(
        jnp.array([[1, 0, 1, 0], [0, 1, 0, 1]]).astype(float),
        jnp.zeros((2, 1), dtype=float),
    )
    scaling = 0.5

    return params, regularizer_strength, mask, scaling


@pytest.fixture
def example_data_prox_operator_multineuron():
    n_features = 4
    n_neurons = 3

    params = GLMParams(
        jnp.ones((n_features, n_neurons)),
        jnp.zeros(n_neurons),
    )
    regularizer_strength = tree_full_like(params, 0.1)
    # Mask as PyTree with same structure as params
    # For multi-neuron: mask shape is (n_groups, n_features, n_neurons)
    # Intercept mask is zeros (not regularized)
    mask_coef = jnp.array(
        [
            [[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]],
            [[0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1]],
        ],
        dtype=jnp.float32,
    )
    mask = GLMParams(
        mask_coef,
        jnp.zeros((2, n_neurons), dtype=jnp.float32),
    )
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
            - GLMParams(w_true, b_true) (tuple): True weight and bias parameters.
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
    return X, np.random.gamma(k, scale=theta), model, GLMParams(w_true, b_true), rate


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
            - GLMParams(w_true, b_true) (tuple): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of response.
    """
    X, spikes, model, true_params, rate = gammaGLM_model_instantiation
    X_tree = {"input_1": X[..., :3], "input_2": X[..., 3:]}
    true_params_tree = GLMParams(
        dict(input_1=true_params.coef[:3], input_2=true_params.coef[3:]),
        true_params.intercept,
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
    return X, y, model, GLMParams(w_true, b_true), rate


@pytest.fixture()
def population_gammaGLM_model_instantiation_pytree(
    population_gammaGLM_model_instantiation,
):
    X, spikes, model, true_params, rate = population_gammaGLM_model_instantiation
    X_tree = {"input_1": X[..., :3], "input_2": X[..., 3:]}
    true_params_tree = GLMParams(
        dict(input_1=true_params.coef[:3], input_2=true_params.coef[3:]),
        true_params.intercept,
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
            - GLMParams(w_true, b_true) (tuple): True weight and bias parameters.
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
    return X, np.random.binomial(1, rate), model, GLMParams(w_true, b_true), rate


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
            - GLMParams(w_true, b_true) (tuple): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of response.
    """
    X, spikes, model, true_params, rate = bernoulliGLM_model_instantiation
    X_tree = {"input_1": X[..., :3], "input_2": X[..., 3:]}
    true_params_tree = GLMParams(
        dict(input_1=true_params.coef[:3], input_2=true_params.coef[3:]),
        true_params.intercept,
    )
    model_tree = nmo.glm.GLM(model.observation_model, regularizer=model.regularizer)
    return X_tree, spikes, model_tree, true_params_tree, rate


@pytest.fixture
def classifierGLM_model_instantiation():
    """Set up a categorical GLM for testing purposes.

    This fixture initializes a categorical GLM with random parameters, simulates its response, and
    returns the test data, expected output, the model instance, true parameters, and the rate
    of response.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): Simulated input data.
            - jax.random.categorical(key, rate) (numpy.ndarray): Simulated spike responses.
            - model (nmo.glm.GLM): Initialized model instance.
            - GLMParams(w_true, b_true) (tuple): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of log-proba.
    """
    np.random.seed(123)
    num_classes = 3
    X = np.random.normal(size=(100, 5))
    # Over-parameterized model: one set of params per class
    b_true = np.zeros((num_classes,))
    w_true = np.random.normal(size=(5, num_classes))
    regularizer = nmo.regularizer.UnRegularized()
    model = nmo.glm.ClassifierGLM(num_classes, regularizer=regularizer)
    # Use jax.nn.log_softmax directly (no padding)
    rate = jax.nn.log_softmax(jnp.einsum("ki,tk->ti", w_true, X) + b_true)
    key = jax.random.PRNGKey(123)
    y = jax.random.categorical(key, rate)
    model.set_classes(np.arange(num_classes))
    return X, y, model, GLMParams(w_true, b_true), rate


@pytest.fixture
def classifierGLM_model_instantiation_pytree(classifierGLM_model_instantiation):
    """Set up a categorical GLM for testing purposes.

    This fixture initializes a categorical GLM with random parameters, simulates its response, and
    returns the test data, expected output, the model instance, true parameters, and the rate
    of response.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): Simulated input data.
            - jax.random.categorical(key, rate) (numpy.ndarray): Simulated spike responses.
            - model (nmo.glm.GLM): Initialized model instance.
            - GLMParams(w_true, b_true) (tuple): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of log-proba.
    """
    X, spikes, model, true_params, rate = classifierGLM_model_instantiation
    X_tree = {"input_1": X[..., :3], "input_2": X[..., 3:]}
    true_params_tree = GLMParams(
        dict(input_1=true_params.coef[:3], input_2=true_params.coef[3:]),
        true_params.intercept,
    )
    model_tree = nmo.glm.ClassifierGLM(
        n_classes=model.n_classes, regularizer=model.regularizer
    )
    model_tree.set_classes(np.arange(model.n_classes))
    return X_tree, spikes, model_tree, true_params_tree, rate


@pytest.fixture
def population_classifierGLM_model_instantiation():
    """Set up a categorical GLM for testing purposes.

    This fixture initializes a categorical GLM with random parameters, simulates its response, and
    returns the test data, expected output, the model instance, true parameters, and the rate
    of response.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): Simulated input data.
            - jax.random.categorical(key, rate) (numpy.ndarray): Simulated spike responses.
            - model (nmo.glm.GLM): Initialized model instance.
            - GLMParams(w_true, b_true) (tuple): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of log-proba.
    """
    np.random.seed(123)
    num_classes = 3
    n_neurons = 3
    X = np.random.normal(size=(100, 5))
    # Over-parameterized model: one set of params per class
    b_true = np.zeros((n_neurons, num_classes))
    w_true = np.random.normal(size=(5, n_neurons, num_classes))
    regularizer = nmo.regularizer.UnRegularized()
    model = nmo.glm.ClassifierPopulationGLM(num_classes, regularizer=regularizer)
    # Use jax.nn.log_softmax directly (no padding)
    rate = jax.nn.log_softmax(jnp.einsum("kni,tk->tni", w_true, X) + b_true)
    key = jax.random.PRNGKey(123)
    y = jax.random.categorical(key, rate)
    model.set_classes(np.arange(num_classes))
    return X, y, model, GLMParams(w_true, b_true), rate


@pytest.fixture
def population_classifierGLM_model_instantiation_pytree(
    population_classifierGLM_model_instantiation,
):
    """Set up a categorical GLM for testing purposes.

    This fixture initializes a categorical GLM with random parameters, simulates its response, and
    returns the test data, expected output, the model instance, true parameters, and the rate
    of response.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): Simulated input data.
            - jax.random.categorical(key, rate) (numpy.ndarray): Simulated spike responses.
            - model (nmo.glm.GLM): Initialized model instance.
            - GLMParams(w_true, b_true) (tuple): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of log-proba.
    """
    X, spikes, model, true_params, rate = population_classifierGLM_model_instantiation
    X_tree = {"input_1": X[..., :3], "input_2": X[..., 3:]}
    true_params_tree = GLMParams(
        dict(input_1=true_params.coef[:3], input_2=true_params.coef[3:]),
        true_params.intercept,
    )
    model_tree = nmo.glm.ClassifierPopulationGLM(
        n_classes=model.n_classes, regularizer=model.regularizer
    )
    model_tree.set_classes(np.arange(model.n_classes))
    return X_tree, spikes, model_tree, true_params_tree, rate


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
            - GLMParams(w_true, b_true) (tuple): True weight and bias parameters.
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
    return X, np.random.binomial(1, rate), model, GLMParams(w_true, b_true), rate


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
            - GLMParams(w_true, b_true) (tuple): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of response.
    """
    X, spikes, model, true_params, rate = population_bernoulliGLM_model_instantiation
    X_tree = {"input_1": X[..., :3], "input_2": X[..., 3:]}
    true_params_tree = GLMParams(
        dict(input_1=true_params.coef[:3], input_2=true_params.coef[3:]),
        true_params.intercept,
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
            - GLMParams(w_true, b_true) (tuple): True weight and bias parameters.
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
    return X, spikes, model, GLMParams(w_true, b_true), rate


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
            - X (dict): Simulated input data.
            - np.random.poisson(rate) (numpy.ndarray): Simulated spike responses.
            - model (nmo.glm.PoissonGLM): Initialized model instance.
            - GLMParams(w_true, b_true) (tuple): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of response.
    """
    X, spikes, model, true_params, rate = negativeBinomialGLM_model_instantiation
    X_tree = {"input_1": X[..., :3], "input_2": X[..., 3:]}
    true_params_tree = GLMParams(
        dict(input_1=true_params.coef[:3], input_2=true_params.coef[3:]),
        true_params.intercept,
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
            - GLMParams(w_true, b_true) (tuple): True weight and bias parameters.
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
    return X, spikes, model, GLMParams(w_true, b_true), rate


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
            - GLMParams(w_true, b_true) (tuple): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of response.
    """
    X, spikes, model, true_params, rate = (
        population_negativeBinomialGLM_model_instantiation
    )
    X_tree = {"input_1": X[..., :3], "input_2": X[..., 3:]}
    true_params_tree = GLMParams(
        dict(input_1=true_params.coef[:3], input_2=true_params.coef[3:]),
        true_params.intercept,
    )
    model_tree = nmo.glm.PopulationGLM(
        observation_model=model.observation_model,
        regularizer=model.regularizer,
        solver_name="LBFGS",
    )
    return X_tree, np.random.poisson(rate), model_tree, true_params_tree, rate


def instantiate_glm_func(
    obs_model: (
        Literal["Poisson", "Gamma", "Bernoulli", "NegativeBinomial"]
        | nmo.observation_models.Observations
    ) = "Bernoulli",
    regularizer: str = "UnRegularized",
    solver_name: str = None,
    simulate=False,
):
    np.random.seed(123)
    n_features = 2
    X = np.ones((500, n_features))
    X[:250, 0] = 0
    X[np.arange(500) % 2 == 1, 1] = 0
    if obs_model == "Gamma":
        inv_link = jax.nn.softplus
    else:
        inv_link = None
    model = nmo.glm.GLM(
        observation_model=obs_model,
        regularizer=regularizer,
        solver_name=solver_name,
        inverse_link_function=inv_link,
    )
    model.coef_ = np.random.randn(n_features)
    model.intercept_ = np.random.randn(1)
    model.scale_ = 1.0
    if simulate:
        counts, rates = model.simulate(jax.random.PRNGKey(1234), X)
    else:
        counts, rates = None, None
    return ModelFixture(
        X=X,
        y=counts,
        model=model,
        params=GLMParams(model.coef_, model.intercept_),
        rates=rates,
        extra=None,
    )


def instantiate_population_glm_func(
    n_neurons=3,
    obs_model: (
        Literal["Poisson", "Gamma", "Bernoulli", "NegativeBinomial"]
        | nmo.observation_models.Observations
    ) = "Bernoulli",
    regularizer: str = "UnRegularized",
    solver_name: str = None,
    simulate=False,
):
    np.random.seed(123)
    n_features = 2
    X = np.ones((500, n_features))
    X[:250, 0] = 0
    X[np.arange(500) % 2 == 1, 1] = 0
    if obs_model == "Gamma":
        inv_link = jax.nn.softplus
    else:
        inv_link = None
    model = nmo.glm.PopulationGLM(
        observation_model=obs_model,
        regularizer=regularizer,
        solver_name=solver_name,
        inverse_link_function=inv_link,
    )
    model.coef_ = np.random.randn(n_features, n_neurons)
    model.intercept_ = np.random.randn(n_neurons)
    model.scale_ = 1.0
    if simulate:
        counts, rates = model.simulate(jax.random.PRNGKey(1234), X)
    else:
        counts, rates = None, None
    return ModelFixture(
        X=X,
        y=counts,
        model=model,
        params=GLMParams(model.coef_, model.intercept_),
        rates=rates,
        extra=None,
    )


def instantiate_classifier_glm_func(
    n_neurons=3,
    regularizer: str = "UnRegularized",
    solver_name: str = None,
    simulate=False,
):
    np.random.seed(124)
    n_features = 2
    n_classes = 4
    X = np.ones((500, n_features))
    X[:250, 0] = 0
    X[np.arange(500) % 2 == 1, 1] = 0
    model = nmo.glm.ClassifierGLM(
        n_classes=n_classes,
        regularizer=regularizer,
        solver_name=solver_name,
    )
    model.coef_ = np.random.randn(n_features, n_classes)
    model.intercept_ = np.random.randn(n_classes)
    model.set_classes(np.arange(n_classes))
    if simulate:
        counts, rates = model.simulate(jax.random.PRNGKey(123), X)
    else:
        counts, rates = None, None
    return ModelFixture(
        X=X,
        y=counts,
        model=model,
        params=GLMParams(model.coef_, model.intercept_),
        rates=rates,
        extra=None,
    )


def instantiate_population_classifier_glm_func(
    n_neurons=3,
    regularizer: str = "UnRegularized",
    solver_name: str = None,
    simulate=False,
):
    np.random.seed(124)
    n_features = 2
    n_classes = 4
    X = np.ones((500, n_features))
    X[:250, 0] = 0
    X[np.arange(500) % 2 == 1, 1] = 0
    model = nmo.glm.ClassifierPopulationGLM(
        n_classes=n_classes,
        regularizer=regularizer,
        solver_name=solver_name,
    )
    model.set_classes(np.arange(n_classes))
    model.coef_ = np.random.randn(n_features, n_neurons, n_classes)
    model.intercept_ = np.random.randn(n_neurons, n_classes)
    if simulate:
        counts, rates = model.simulate(jax.random.PRNGKey(123), X)
    else:
        counts, rates = None, None
    return ModelFixture(
        X=X,
        y=counts,
        model=model,
        params=GLMParams(model.coef_, model.intercept_),
        rates=rates,
        extra=None,
    )


def run_simulation_glm_hmm(
    design_matrix: jnp.ndarray, model: nmo.glm_hmm.GLMHMM, seed: int
):
    n_timepoints = design_matrix.shape[0]
    coef, intercept = model.coef_, model.intercept_
    n_neurons = coef.shape[1] if coef.ndim > 2 else 1
    n_states = intercept.shape[-1]
    initial_prob = model.initial_prob_
    transition_prob = model.transition_prob_

    glm = nmo.glm.PopulationGLM(
        observation_model=model.observation_model,
        inverse_link_function=model.inverse_link_function,
    )

    latent_states = np.zeros((n_timepoints, n_states), dtype=int)
    rates = np.zeros((n_timepoints, n_neurons))
    counts = np.zeros((n_timepoints, n_neurons))

    np.random.seed(seed)
    init_prob_arr = np.asarray(initial_prob, dtype=float)
    initial_state = np.random.choice(n_states, p=init_prob_arr / init_prob_arr.sum())
    latent_states[0, initial_state] = 1

    glm.coef_ = coef[..., initial_state].reshape(coef.shape[0], n_neurons)
    glm.intercept_ = intercept[..., initial_state].reshape((n_neurons,))
    glm.scale_ = 1.0

    key = jax.random.PRNGKey(seed)
    counts[0], rates[0] = glm.simulate(key, design_matrix[:1])

    for t in range(1, n_timepoints):
        key, subkey = jax.random.split(key)
        prev_state_vec = latent_states[t - 1]
        transition_probs = transition_prob.T @ prev_state_vec
        next_state = jax.random.choice(subkey, jnp.arange(n_states), p=transition_probs)
        latent_states[t, next_state] = 1

        glm.coef_ = coef[..., next_state].reshape(coef.shape[0], n_neurons)
        glm.intercept_ = intercept[..., next_state].reshape((n_neurons,))
        key, subkey = jax.random.split(key)
        counts[t], rates[t] = glm.simulate(subkey, design_matrix[t : t + 1])

    counts = jnp.squeeze(counts)
    rates = jnp.squeeze(rates)
    return counts, rates, latent_states


def instantiate_glm_hmm_func(
    n_states: int = 3,
    obs_model: (
        Literal["Poisson", "Gamma", "Bernoulli", "NegativeBinomial"]
        | nmo.observation_models.Observations
    ) = "Bernoulli",
    regularizer: str = "UnRegularized",
    solver_name: str = None,
    simulate: bool = False,
    solver_kwargs=None,
    maxiter: int = 2,
):
    np.random.seed(123)
    if solver_kwargs is None:
        solver_kwargs = {"maxiter": 1}

    model = nmo.glm_hmm.GLMHMM(
        n_states=n_states,
        observation_model=obs_model,
        regularizer=regularizer,
        solver_name=solver_name,
        solver_kwargs=solver_kwargs,
        maxiter=maxiter,
    )
    n_features = 2
    X = np.ones((500, n_features))
    X[:250, 0] = 0
    X[np.arange(500) % 2 == 1, 1] = 0
    y = np.zeros(X.shape[0])
    y[np.random.choice(y.shape[0], size=y.shape[0] // 3, replace=False)] = 1

    coef, intercept = random_glm_params_init(
        n_states=n_states,
        X=X,
        y=y,
        inverse_link_function=model.inverse_link_function,
        session_starts=None,
        random_key=jax.random.PRNGKey(123),
    )
    coef = jnp.squeeze(coef)
    intercept = jnp.squeeze(intercept)
    transition_prob = sticky_transition_proba_init(n_states)
    init_prob = uniform_initial_proba_init(n_states, random_key=jax.random.PRNGKey(124))
    scale = jnp.ones_like(intercept)

    if simulate:
        model.coef_ = coef
        model.intercept_ = intercept
        model.scale_ = scale
        model.initial_prob_ = init_prob
        model.transition_prob_ = transition_prob
        y, rates, latent_states = run_simulation_glm_hmm(X, model, seed=1234)
        # reset fit attributes so fixture.model is unfitted
        model.coef_ = None
        model.intercept_ = None
        model.scale_ = None
        model.initial_prob_ = None
        model.transition_prob_ = None
    else:
        rates, latent_states = None, None

    return ModelFixture(
        X=X,
        y=y,
        model=model,
        params=GLMHMMParams(
            model_params=GLMHMMModelParams(
                coef=coef,
                intercept=intercept,
                log_scale=scale,
            ),
            hmm_params=HMMParams(
                log_initial_prob=jnp.log(init_prob),
                log_transition_prob=jnp.log(transition_prob),
            ),
        ),
        rates=rates,
        extra=latent_states,
    )


_MODEL_CACHE = {}


# Registry for model-specific configurations
MODEL_CONFIG = {
    "GLM": {
        "is_population": False,
        "default_y_shape": (500,),
    },
    "ClassifierGLM": {
        "is_population": False,
        "default_y_shape": (500,),
    },
    "PopulationGLM": {
        "is_population": True,
        "default_y_shape": (500, 3),
    },
    "ClassifierPopulationGLM": {
        "is_population": True,
        "default_y_shape": (500, 3),
    },
    "GLMHMM": {
        "is_population": False,
        "default_y_shape": (500,),
    },
}


def is_population_model(model) -> bool:
    """Check if a model is a population model using registry instead of string matching."""
    model_name = model.__class__.__name__
    return MODEL_CONFIG.get(model_name, {}).get("is_population", False)


@pytest.fixture
def instantiate_base_regressor_subclass(request):
    """
    Instantiate the concrete BaseRegressor sub-classes with caching.
    """
    model_name: str = request.param["model"]
    obs_model: str | nmo.observation_models.Observations = request.param["obs_model"]
    simulate: bool = request.param["simulate"]

    # Create cache key (class-scoped)
    cache_key = (
        model_name,
        str(obs_model),
        simulate,
        id(request.cls) if request.cls else id(request.module),
    )

    # Check cache
    if cache_key not in _MODEL_CACHE:
        if model_name == "GLM":
            result = instantiate_glm_func(obs_model=obs_model, simulate=simulate)
        elif model_name == "PopulationGLM":
            result = instantiate_population_glm_func(
                obs_model=obs_model, simulate=simulate
            )
        elif model_name == "ClassifierGLM":
            result = instantiate_classifier_glm_func(simulate=simulate)
        elif model_name == "ClassifierPopulationGLM":
            result = instantiate_population_classifier_glm_func(simulate=simulate)
        elif model_name == "GLMHMM":
            result = instantiate_glm_hmm_func(obs_model=obs_model, simulate=simulate)
        else:
            raise ValueError("model_name {} unknown".format(model_name))
        _MODEL_CACHE[cache_key] = result
        return deepcopy(result)

    # Get cached data and return a complete deepcopy of everything
    # this is different from a function level fixture because it
    # would not re-run any potentially heavy setup code (like model.simulate).
    cached_result = deepcopy(_MODEL_CACHE[cache_key])
    return cached_result


# Auto-clear cache after each test module run
@pytest.fixture(scope="module", autouse=True)
def _clear_model_cache():
    """Clear model cache after each test module."""
    yield
    _MODEL_CACHE.clear()


# Select solver backend for tests if requested via environment variable
_common_solvers = [
    nmo.solvers.SolverSpec("SVRG", "nemos", nmo.solvers.WrappedSVRG),
    nmo.solvers.SolverSpec("ProxSVRG", "nemos", nmo.solvers.WrappedProxSVRG),
]
_solvers_per_backend = {
    "optimistix": [
        *_common_solvers,
        nmo.solvers.SolverSpec(
            "GradientDescent", "optimistix", nmo.solvers.OptimistixNAG
        ),
        nmo.solvers.SolverSpec(
            "ProximalGradient", "optimistix", nmo.solvers.OptimistixFISTA
        ),
        nmo.solvers.SolverSpec("LBFGS", "optimistix", nmo.solvers.OptimistixOptaxLBFGS),
        nmo.solvers.SolverSpec("BFGS", "optimistix", nmo.solvers.OptimistixBFGS),
        nmo.solvers.SolverSpec(
            "NonlinearCG", "optimistix", nmo.solvers.OptimistixNonlinearCG
        ),
    ],
}

if nmo.solvers.JAXOPT_AVAILABLE:
    _solvers_per_backend["jaxopt"] = [
        *_common_solvers,
        nmo.solvers.SolverSpec(
            "GradientDescent", "jaxopt", nmo.solvers.JaxoptGradientDescent
        ),
        nmo.solvers.SolverSpec(
            "ProximalGradient", "jaxopt", nmo.solvers.JaxoptProximalGradient
        ),
        nmo.solvers.SolverSpec("LBFGS", "jaxopt", nmo.solvers.JaxoptLBFGS),
        nmo.solvers.SolverSpec("BFGS", "jaxopt", nmo.solvers.JaxoptBFGS),
        nmo.solvers.SolverSpec("NonlinearCG", "jaxopt", nmo.solvers.JaxoptNonlinearCG),
    ]


@pytest.fixture(autouse=True, scope="session")
def configure_solver_backend(request):
    """
    Patch the solver registry depending on `NEMOS_SOLVER_BACKEND` and `override_solver`.

    The `NEMOS_SOLVER_BACKEND` env variable is used for running solver-dependent tests
    in separate tox environments for the JAXopt and the Optimistix backends.

    The `override_solver` pytest option is used to set a given solver algorithm's
    implementation to a class available in nemos.solvers.
    """
    backend = os.getenv("NEMOS_SOLVER_BACKEND")

    if backend is None:
        _solvers_to_use = nmo.solvers.list_available_solvers()
    else:
        if backend == "jaxopt" and not nmo.solvers.JAXOPT_AVAILABLE:
            pytest.fail("jaxopt backend requested but jaxopt is not installed.")
        try:
            _solvers_to_use = _solvers_per_backend[backend]
        except KeyError:
            available = ", ".join(_solvers_per_backend.keys())
            pytest.fail(f"Unknown solver backend: {backend}. Available: {available}")

    override_solver = request.config.getini("override_solver")
    if override_solver:
        try:
            algo_name, impl_name = override_solver.split(":", 1)
        except ValueError:
            raise ValueError(
                f"override_solver must be in format 'algo:implementation', got: {override_solver}"
            )
        for i, solver in enumerate(_solvers_to_use):
            if solver.algo_name == algo_name:
                _solvers_to_use[i] = nmo.solvers.SolverSpec(
                    algo_name, "replaced_for_pytest", getattr(nmo.solvers, impl_name)
                )

    # save the original registry so that we can restore it after
    original_registry = nmo.solvers._solver_registry._registry.copy()
    original_defaults = nmo.solvers._solver_registry._defaults.copy()
    nmo.solvers._solver_registry._registry.clear()
    nmo.solvers._solver_registry._defaults.clear()
    for solver in _solvers_to_use:
        nmo.solvers._solver_registry.register(
            solver.algo_name, solver.implementation, solver.backend, default=True
        )

    try:
        yield
    finally:
        nmo.solvers._solver_registry._registry.clear()
        nmo.solvers._solver_registry._defaults.clear()
        nmo.solvers._solver_registry._registry.update(original_registry)
        nmo.solvers._solver_registry._defaults.update(original_defaults)


def pytest_addoption(parser):
    """Register custom ini options."""
    parser.addini("solver_backend", "Solver backend to use")
    parser.addini("override_solver", "Override solver as 'algorithm:implementation'")


@pytest.fixture
def gaussianGLM_model_instantiation():
    """Set up a Gaussian GLM for testing purposes.

    This fixture initializes a Gaussian GLM with random parameters, simulates its response, and
    returns the test data, expected output, the model instance, true parameters, and the rate
    of response.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): Simulated input data.
            - np.random.normal(rate) (numpy.ndarray): Simulated spike responses.
            - model (nmo.glm.PoissonGLM): Initialized model instance.
            - GLMParams(w_true, b_true): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of response.
    """
    np.random.seed(123)
    X = np.random.normal(size=(100, 5))
    b_true = np.zeros((1,))
    w_true = np.random.normal(size=(5,))
    observation_model = nmo.observation_models.GaussianObservations()
    regularizer = nmo.regularizer.UnRegularized()
    model = nmo.glm.GLM(
        observation_model, regularizer=regularizer, solver_name="LBFGS"
    )  # , solver_kwargs={"tol":1e-12})
    model.scale_ = 1.0
    rate = jax.numpy.einsum("k,tk->t", w_true, X) + b_true
    return X, np.random.normal(rate), model, GLMParams(w_true, b_true), rate


@pytest.fixture
def population_gaussianGLM_model_instantiation():
    """Set up a Population Gaussian GLM for testing purposes.

    This fixture initializes a Population Gaussian GLM with random parameters, simulates its response, and
    returns the test data, expected output, the model instance, true parameters, and the rate
    of response.

    Returns:
        tuple: A tuple containing:
            - X (numpy.ndarray): Simulated input data.
            - np.random.normal(rate) (numpy.ndarray): Simulated spike responses.
            - model (nmo.glm.PoissonGLM): Initialized model instance.
            - GLMParams(w_true, b_true): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of response.
    """
    np.random.seed(123)
    X = np.random.normal(size=(200, 5)) * 10
    b_true = np.zeros((3,))
    w_true = np.random.normal(size=(5, 3))
    observation_model = nmo.observation_models.GaussianObservations()
    regularizer = nmo.regularizer.UnRegularized()
    model = nmo.glm.PopulationGLM(
        observation_model=observation_model,
        regularizer=regularizer,
        solver_name="LBFGS",
    )
    model.scale_ = 1.0
    rate = jax.numpy.einsum("ki,tk->ti", w_true, X) + b_true
    return X, np.random.normal(rate), model, GLMParams(w_true, b_true), rate


@pytest.fixture
def gaussianGLM_model_instantiation_pytree(gaussianGLM_model_instantiation):
    """Set up a Gaussian GLM for testing purposes.

    This fixture initializes a Gaussian GLM with random parameters, simulates its response, and
    returns the test data, expected output, the model instance, true parameters, and the rate
    of response.

    Returns:
        tuple: A tuple containing:
            - X (dict): Simulated input data.
            - np.random.normal(rate) (numpy.ndarray): Simulated spike responses.
            - model (nmo.glm.PoissonGLM): Initialized model instance.
            - GLMParams(w_true, b_true): True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of response.
    """
    X, spikes, model, true_params, rate = gaussianGLM_model_instantiation
    X_tree = {"input_1": X[..., :3], "input_2": X[..., 3:]}
    true_params_tree = GLMParams(
        dict(input_1=true_params.coef[:3], input_2=true_params.coef[3:]),
        true_params.intercept,
    )
    model_tree = nmo.glm.GLM(
        model.observation_model, regularizer=model.regularizer, solver_name="LBFGS"
    )  # , solver_kwargs={"tol":1e-12})
    return X_tree, spikes, model_tree, true_params_tree, rate


@pytest.fixture
def population_gaussianGLM_model_instantiation_pytree(
    population_gaussianGLM_model_instantiation,
):
    """Set up a Population Gaussian GLM for testing purposes.

    This fixture initializes a Population Gaussian GLM with random parameters, simulates its response, and
    returns the test data, expected output, the model instance, true parameters, and the rate
    of response.

    Returns:
        tuple: A tuple containing:
            - X (dict): Simulated input data.
            - np.random.normal(rate) (numpy.ndarray): Simulated spike responses.
            - model (nmo.glm.PoissonGLM): Initialized model instance.
            - GLMParams(w_true, b_true) : True weight and bias parameters.
            - rate (jax.numpy.ndarray): Simulated rate of response.
    """
    X, spikes, model, true_params, rate = population_gaussianGLM_model_instantiation
    X_tree = {"input_1": X[..., :3], "input_2": X[..., 3:]}
    true_params_tree = GLMParams(
        dict(input_1=true_params.coef[:3], input_2=true_params.coef[3:]),
        true_params.intercept,
    )
    model_tree = nmo.glm.PopulationGLM(
        observation_model=model.observation_model,
        regularizer=model.regularizer,
        solver_name="LBFGS",
    )
    return X_tree, np.random.normal(rate), model_tree, true_params_tree, rate
