import inspect
import warnings
from contextlib import nullcontext as does_not_raise
from copy import deepcopy
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy as sp
import scipy.stats as sts
import sklearn
import statsmodels.api as sm
from numba import njit
from pynapple import Tsd, TsdFrame
from sklearn.linear_model import GammaRegressor, LogisticRegression, PoissonRegressor
from sklearn.model_selection import GridSearchCV

import nemos as nmo
from nemos._observation_model_builder import instantiate_observation_model
from nemos._regularizer_builder import instantiate_regularizer
from nemos.glm.inverse_link_function_utils import LINK_NAME_TO_FUNC
from nemos.observation_models import NegativeBinomialObservations
from nemos.pytrees import FeaturePytree
from nemos.tree_utils import pytree_map_and_reduce, tree_l2_norm, tree_slice, tree_sub
from nemos.utils import _get_name

GLM_COMMON_PARAMS_NAMES = {
    "inverse_link_function",
    "observation_model",
    "regularizer",
    "regularizer_strength",
    "solver_kwargs",
    "solver_name",
}
OBSERVATION_MODEL_EXTRA_PARAMS_NAMES = {
    "NegativeBinomialObservations": {"observation_model__scale"},
}
POPULATION_GLM_EXTRA_PARAMS = {"feature_mask"}


def convert_to_nap(arr, t):
    return TsdFrame(t=t, d=getattr(arr, "d", arr))


def test_validate_higher_dimensional_data_X(mock_glm):
    """Test behavior with higher-dimensional input data."""
    X = jnp.array([[[[1, 2], [3, 4]]]])
    y = jnp.array([1, 2])
    with pytest.raises(ValueError, match="X must be two-dimensional"):
        mock_glm._validate(X, y, mock_glm._initialize_parameters(X, y))


def test_preprocess_fit_higher_dimensional_data_y(mock_glm):
    """Test behavior with higher-dimensional input data."""
    X = jnp.array([[[1, 2], [3, 4]]])
    y = jnp.array([[[1, 2]]])
    with pytest.raises(ValueError, match="y must be one-dimensional"):
        p0 = jnp.zeros((X.shape[1])), jnp.atleast_1d(jnp.log(y.mean()))
        mock_glm._validate(X, y, p0)


def test_validate_lower_dimensional_data_X(mock_glm):
    """Test behavior with lower-dimensional input data."""
    X = jnp.array([1, 2])
    y = jnp.array([1, 2])
    with pytest.raises(ValueError, match="X must be two-dimensional"):
        mock_glm._validate(X, y, mock_glm._initialize_parameters(X, y))


@pytest.fixture
def model_instantiation_type(glm_class_type):
    """
    Fixure to grab the appropriate model instantiation function based on the type of GLM class.
    Used by TestGLM and TestPoissonGLM classes.
    """
    if "population" in glm_class_type:
        return "population_poissonGLM_model_instantiation"
    else:
        return "poissonGLM_model_instantiation"


@pytest.mark.parametrize("glm_class_type", ["", "population"])
@pytest.mark.solver_related
def test_get_fit_attrs(request, glm_class_type, model_instantiation_type):
    X, y, model = request.getfixturevalue(model_instantiation_type)[:3]
    expected_state = {
        "coef_": None,
        "intercept_": None,
        "scale_": None,
        "solver_state_": None,
        "dof_resid_": None,
    }
    assert model._get_fit_state() == expected_state
    model.solver_kwargs = {"maxiter": 1}
    model.fit(X, y)
    assert all(val is not None for val in model._get_fit_state().values())
    assert model._get_fit_state().keys() == expected_state.keys()


@pytest.mark.parametrize("glm_class_type", ["glm_class", "population_glm_class"])
class TestGLM:
    """
    Unit tests for the GLM class that do not depend on the observation model.
    i.e. tests that do not call observation model methods, or tests that do not check the output when
    observation model methods are called (e.g. error testing for input validation)
    """

    #######################
    # Test model.__init__
    #######################
    @pytest.mark.parametrize(
        "solver_name, expectation",
        [
            # test solver at initialization, where test_regularizers.py tests solvers with set_params
            (None, does_not_raise()),
            ("BFGS", does_not_raise()),
            ("ProximalGradient", does_not_raise()),
            ("LBFGS", does_not_raise()),
            ("NonlinearCG", does_not_raise()),
            ("SVRG", does_not_raise()),
            ("ProxSVRG", does_not_raise()),
            (
                1,
                pytest.raises(ValueError, match="The solver: 1 is not allowed "),
            ),
        ],
    )
    def test_init_solver_type(self, solver_name, expectation, request, glm_class_type):
        """
        Test that an error is raised if a non-compatible solver is passed.
        """
        glm_class = request.getfixturevalue(glm_class_type)
        with expectation:
            glm_class(solver_name=solver_name)

    def test_non_differentiable_inverse_link(self, request, glm_class_type):
        glm_class = request.getfixturevalue(glm_class_type)
        model = glm_class()

        # define a jax non-diff function
        non_diff = lambda y: jnp.asarray(njit(lambda x: x)(np.atleast_1d(y)))

        with pytest.raises(
            ValueError,
            match="The `inverse_link_function` function cannot be differentiated",
        ):
            model.inverse_link_function = non_diff
        with pytest.raises(
            ValueError,
            match="The `inverse_link_function` function cannot be differentiated",
        ):
            glm_class(inverse_link_function=non_diff)

    @pytest.mark.parametrize(
        "link_function",
        [
            jnp.exp,
            lambda x: jnp.exp(x) if isinstance(x, jnp.ndarray) else "not a number",
        ],
    )
    def test_initialization_link_returns_scalar(
        self,
        link_function,
        request,
        glm_class_type,
    ):
        """Check that the observation model initializes when a callable is passed."""
        raise_exception = not isinstance(link_function(1.0), (jnp.ndarray, float))
        model = request.getfixturevalue(glm_class_type)()

        if raise_exception:
            with pytest.raises(
                ValueError,
                match="The `inverse_link_function` must handle scalar inputs correctly",
            ):
                model.set_params(inverse_link_function=link_function)
        else:
            model.set_params(inverse_link_function=link_function)

    @pytest.mark.parametrize(
        "link_function",
        [jnp.exp, np.exp, lambda x: 1 / x, sm.families.links.Log()],
    )
    def test_initialization_link_is_jax(
        self,
        link_function,
        request,
        glm_class_type,
    ):
        """Check that the observation model initializes when a callable is passed."""
        glm_class = request.getfixturevalue(glm_class_type)

        raise_exception = isinstance(link_function, np.ufunc) | isinstance(
            link_function, sm.families.links.Link
        )
        if raise_exception:
            with pytest.raises(
                ValueError,
                match="The `inverse_link_function` must return a jax.numpy.ndarray",
            ):
                glm_class(inverse_link_function=link_function)
        else:
            glm_class(inverse_link_function=link_function)

    @pytest.mark.parametrize(
        "link_function, expectation",
        [
            (jax.scipy.special.expit, does_not_raise()),
            (
                sp.special.expit,
                pytest.raises(
                    ValueError,
                    match="The `inverse_link_function` must return a jax.numpy.ndarray!",
                ),
            ),
            (jax.scipy.stats.norm.cdf, does_not_raise()),
            (
                sts.norm.cdf,
                pytest.raises(
                    ValueError,
                    match="The `inverse_link_function` must return a jax.numpy.ndarray!",
                ),
            ),
            (
                np.exp,
                pytest.raises(
                    ValueError,
                    match="The `inverse_link_function` must return a jax.numpy.ndarray!",
                ),
            ),
            (lambda x: x, does_not_raise()),
            (
                sm.families.links.Log(),
                pytest.raises(
                    ValueError,
                    match="The `inverse_link_function` must return a jax.numpy.ndarray!",
                ),
            ),
        ],
    )
    def test_initialization_link_is_jax_set_params(
        self, link_function, request, glm_class_type, expectation
    ):
        glm_class = request.getfixturevalue(glm_class_type)

        with expectation:
            glm_class().set_params(inverse_link_function=link_function)

    @pytest.mark.parametrize("link_function", [jnp.exp, jax.nn.softplus, 1])
    def test_initialization_link_is_callable(
        self, link_function, request, glm_class_type
    ):
        """Check that the observation model initializes when a callable is passed."""
        glm_class = request.getfixturevalue(glm_class_type)
        raise_exception = not callable(link_function)
        if raise_exception:
            with pytest.raises(
                TypeError,
                match="The `inverse_link_function` function must be a Callable",
            ):
                glm_class(inverse_link_function=link_function)
        else:
            glm_class(inverse_link_function=link_function)

    @pytest.mark.parametrize("link_function", [jnp.exp, jax.nn.softplus, 1])
    def test_initialization_link_is_callable_set_params(
        self, link_function, request, glm_class_type
    ):
        """Check that the observation model initializes when a callable is passed."""
        glm_class = request.getfixturevalue(glm_class_type)
        raise_exception = not callable(link_function)
        if raise_exception:
            with pytest.raises(
                TypeError,
                match="The `inverse_link_function` function must be a Callable",
            ):
                glm_class().set_params(inverse_link_function=link_function)
        else:
            glm_class().set_params(inverse_link_function=link_function)

    @pytest.mark.parametrize(
        "regularizer, expectation",
        [
            # regularizer with class objects are tested in test_regularizers.py
            # so here we only test the string input names and None type
            (None, does_not_raise()),
            ("UnRegularized", does_not_raise()),
            ("Ridge", does_not_raise()),
            ("Lasso", does_not_raise()),
            ("ElasticNet", does_not_raise()),
            ("GroupLasso", does_not_raise()),
            (
                nmo.regularizer.Ridge,
                pytest.raises(
                    TypeError, match="The regularizer should be either a string from "
                ),
            ),
        ],
    )
    def test_init_regularizer_type(
        self, regularizer, expectation, request, glm_class_type
    ):
        """
        Test initialization with different regularizer types.
        Test that an error is raised if a non-compatible regularizer is passed.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="Unused parameter `regularizer_strength`.*",
            )
            glm_class = request.getfixturevalue(glm_class_type)
            with expectation:
                glm_class(regularizer=regularizer, regularizer_strength=1)

    @pytest.mark.parametrize(
        "observation, expectation",
        [
            (nmo.observation_models.PoissonObservations(), does_not_raise()),
            (nmo.observation_models.GammaObservations(), does_not_raise()),
            (nmo.observation_models.BernoulliObservations(), does_not_raise()),
            (nmo.observation_models.NegativeBinomialObservations(), does_not_raise()),
            (
                nmo.regularizer.Regularizer,
                pytest.raises(
                    AttributeError,
                    match="The provided object does not have the required",
                ),
            ),
            (
                1,
                pytest.raises(
                    AttributeError,
                    match="The provided object does not have the required",
                ),
            ),
        ],
    )
    def test_init_observation_type(
        self, observation, expectation, request, glm_class_type, ridge_regularizer
    ):
        """
        Test initialization with different observation models. Check if an appropriate exception is raised
        when the observation model does not have the required attributes.
        """
        glm_class = request.getfixturevalue(glm_class_type)
        with expectation:
            glm_class(
                regularizer=ridge_regularizer,
                regularizer_strength=0.1,
                observation_model=observation,
            )

    def test_get_params(self, request, glm_class_type):
        """
        Test that get_params() contains expected values.
        """
        glm_class = request.getfixturevalue(glm_class_type)

        if "population" in glm_class_type:
            expected_keys = {
                "feature_mask",
                "inverse_link_function",
                "observation_model",
                "regularizer",
                "regularizer_strength",
                "solver_kwargs",
                "solver_name",
            }
        else:
            expected_keys = {
                "inverse_link_function",
                "observation_model",
                "regularizer",
                "regularizer_strength",
                "solver_kwargs",
                "solver_name",
            }

        model = glm_class()

        def get_expected_values(model):
            if "population" in glm_class_type:
                return [
                    model.feature_mask,
                    model.inverse_link_function,
                    model.observation_model,
                    model.regularizer,
                    model.regularizer_strength,
                    model.solver_kwargs,
                    model.solver_name,
                ]

            else:
                return [
                    model.inverse_link_function,
                    model.observation_model,
                    model.regularizer,
                    model.regularizer_strength,
                    model.solver_kwargs,
                    model.solver_name,
                ]

        expected_values = get_expected_values(model)
        assert set(model.get_params().keys()) == expected_keys
        assert list(model.get_params().values()) == expected_values

        # passing params
        model = glm_class(solver_name="LBFGS", regularizer="UnRegularized")

        expected_values = get_expected_values(model)
        assert set(model.get_params().keys()) == expected_keys
        assert list(model.get_params().values()) == expected_values

        # changing regularizer
        model.set_params(regularizer="Ridge", regularizer_strength=1.0)

        expected_values = get_expected_values(model)
        assert set(model.get_params().keys()) == expected_keys
        assert list(model.get_params().values()) == expected_values

        # changing solver
        model.solver_name = "ProximalGradient"

        expected_values = get_expected_values(model)
        assert set(model.get_params().keys()) == expected_keys
        assert list(model.get_params().values()) == expected_values

    #######################
    # Test model.fit
    #######################
    @pytest.mark.parametrize(
        "n_params, expectation",
        [
            (0, pytest.raises(ValueError, match="Params must have length two.")),
            (1, pytest.raises(ValueError, match="Params must have length two.")),
            (2, does_not_raise()),
            (3, pytest.raises(ValueError, match="Params must have length two.")),
        ],
    )
    @pytest.mark.solver_related
    def test_fit_param_length(
        self, n_params, expectation, request, glm_class_type, model_instantiation_type
    ):
        """
        Test the `fit` method with different numbers of initial parameters.
        Check for correct number of parameters.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        if n_params == 0:
            init_params = tuple()
        elif n_params == 1:
            init_params = (true_params[0],)
        else:
            init_params = true_params + (true_params[0],) * (n_params - 2)
        with expectation:
            model.fit(X, y, init_params=init_params)

    @pytest.fixture
    def fit_weights_dimensionality_expectation(self, glm_class_type):
        """
        Fixture to define the expected behavior for test_fit_weights_dimensionality based on the type of GLM class.
        """
        if "population" in glm_class_type:
            return {
                0: pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
                1: pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
                2: does_not_raise(),
                3: pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
            }
        else:
            return {
                0: pytest.raises(
                    ValueError,
                    match=r"Inconsistent number of features",
                ),
                1: does_not_raise(),
                2: pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
                3: pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
            }

    @pytest.mark.parametrize("dim_weights", [0, 1, 2, 3])
    @pytest.mark.solver_related
    def test_fit_weights_dimensionality(
        self,
        dim_weights,
        request,
        glm_class_type,
        model_instantiation_type,
        fit_weights_dimensionality_expectation,
    ):
        """
        Test the `fit` method with weight matrices of different dimensionalities.
        Check for correct dimensionality.
        """
        expectation = fit_weights_dimensionality_expectation[dim_weights]
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        n_samples, n_features = X.shape
        if "population" in glm_class_type:
            n_neurons = 3
        else:
            n_neurons = 4
        if dim_weights == 0:
            init_w = jnp.array([])
        elif dim_weights == 1:
            init_w = jnp.zeros((n_features,))
        elif dim_weights == 2:
            init_w = jnp.zeros((n_features, n_neurons))
        else:
            init_w = jnp.zeros((n_features, n_neurons) + (1,) * (dim_weights - 2))
        with expectation:
            model.fit(X, y, init_params=(init_w, true_params[1]))

    @pytest.mark.parametrize(
        "dim_intercepts, expectation",
        [
            (0, pytest.raises(ValueError, match=r"params\[1\] must be of shape")),
            (1, does_not_raise()),
            (2, pytest.raises(ValueError, match=r"params\[1\] must be of shape")),
            (3, pytest.raises(ValueError, match=r"params\[1\] must be of shape")),
        ],
    )
    @pytest.mark.solver_related
    def test_fit_intercepts_dimensionality(
        self,
        dim_intercepts,
        expectation,
        request,
        glm_class_type,
        model_instantiation_type,
    ):
        """
        Test the `fit` method with intercepts of different dimensionalities. Check for correct dimensionality.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        n_samples, n_features = X.shape

        if "population" in glm_class_type:
            init_b = jnp.zeros((y.shape[1],) * dim_intercepts)
            init_w = jnp.zeros((n_features, y.shape[1]))
        else:
            init_b = jnp.zeros((1,) * dim_intercepts)
            init_w = jnp.zeros((n_features,))

        with expectation:
            model.fit(X, y, init_params=(init_w, init_b))

    """
    Parameterization used by test_fit_init_params_type and test_initialize_solver_init_params_type
    Contains the expected behavior and separate initial parameters for regular and population GLMs
    """
    fit_init_params_type_init_params = (
        "expectation, init_params_glm, init_params_population_glm",
        [
            (
                does_not_raise(),
                [jnp.zeros((5,)), jnp.zeros((1,))],
                [jnp.zeros((5, 3)), jnp.zeros((3,))],
            ),
            (
                pytest.raises(ValueError, match="Params must have length two."),
                [[jnp.zeros((1, 5)), jnp.zeros((1,))]],
                [[jnp.zeros((1, 5)), jnp.zeros((3,))]],
            ),
            (
                pytest.raises(KeyError),
                dict(p1=jnp.zeros((5,)), p2=jnp.zeros((1,))),
                dict(p1=jnp.zeros((3, 3)), p2=jnp.zeros((3, 2))),
            ),
            (
                pytest.raises(
                    TypeError, match=r"X and params\[0\] must be the same type"
                ),
                [dict(p1=jnp.zeros((5,)), p2=jnp.zeros((1,))), jnp.zeros((1,))],
                [dict(p1=jnp.zeros((3, 3)), p2=jnp.zeros((2, 3))), jnp.zeros((3,))],
            ),
            (
                pytest.raises(
                    TypeError, match=r"X and params\[0\] must be the same type"
                ),
                [
                    FeaturePytree(p1=jnp.zeros((5,)), p2=jnp.zeros((5,))),
                    jnp.zeros((1,)),
                ],
                [
                    FeaturePytree(p1=jnp.zeros((3, 3)), p2=jnp.zeros((3, 2))),
                    jnp.zeros((3,)),
                ],
            ),
            (pytest.raises(ValueError, match="Params must have length two."), 0, 0),
            (
                pytest.raises(TypeError, match="Initial parameters must be array-like"),
                {0, 1},
                {0, 1},
            ),
            (
                pytest.raises(TypeError, match="Initial parameters must be array-like"),
                [jnp.zeros((1, 5)), ""],
                [jnp.zeros((1, 5)), ""],
            ),
            (
                pytest.raises(TypeError, match="Initial parameters must be array-like"),
                ["", jnp.zeros((1,))],
                ["", jnp.zeros((1,))],
            ),
        ],
    )

    @pytest.mark.parametrize(*fit_init_params_type_init_params)
    @pytest.mark.solver_related
    def test_fit_init_params_type(
        self,
        request,
        glm_class_type,
        model_instantiation_type,
        expectation,
        init_params_glm,
        init_params_population_glm,
    ):
        """
        Test the `fit` method with various types of initial parameters. Ensure that the provided initial parameters
        are array-like.
        """
        if "population" in glm_class_type:
            init_params = init_params_population_glm
        else:
            init_params = init_params_glm
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        with expectation:
            model.fit(X, y, init_params=init_params)

    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (-1, pytest.raises(ValueError, match="X must be two-dimensional")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="X must be two-dimensional")),
        ],
    )
    @pytest.mark.solver_related
    def test_fit_x_dimensionality(
        self, delta_dim, expectation, request, glm_class_type, model_instantiation_type
    ):
        """
        Test the `fit` method with X input data of different dimensionalities. Ensure correct dimensionality for X.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        if delta_dim == -1:
            X = np.zeros((X.shape[0],))
        elif delta_dim == 1:
            X = np.zeros((X.shape[0], 1, X.shape[1]))
        with expectation:
            model.fit(X, y, init_params=true_params)

    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (-1, pytest.raises(ValueError, match=r"y must be (one|two)-dimensional")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match=r"y must be (one|two)-dimensional")),
        ],
    )
    @pytest.mark.solver_related
    def test_fit_y_dimensionality(
        self,
        delta_dim,
        expectation,
        request,
        glm_class_type,
        model_instantiation_type,
    ):
        """
        Test the `fit` method with y target data of different dimensionalities. Ensure correct dimensionality for y.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        if "population" in glm_class_type:
            if delta_dim == -1:
                y = y[:, 0]
            elif delta_dim == 1:
                y = np.zeros((*y.shape, 1))
        else:
            if delta_dim == -1:
                y = np.zeros([])
            elif delta_dim == 1:
                y = np.zeros((y.shape[0], 1))
        with expectation:
            model.fit(X, y, init_params=true_params)

    @pytest.mark.parametrize(
        "delta_n_features, expectation",
        [
            (-1, pytest.raises(ValueError, match="Inconsistent number of features")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="Inconsistent number of features")),
        ],
    )
    @pytest.mark.solver_related
    def test_fit_n_feature_consistency_weights(
        self,
        delta_n_features,
        expectation,
        request,
        glm_class_type,
        model_instantiation_type,
    ):
        """
        Test the `fit` method for inconsistencies between data features and initial weights provided.
        Ensure the number of features align.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        if "population" in glm_class_type:
            init_w = jnp.zeros((X.shape[1] + delta_n_features, y.shape[1]))
            init_b = jnp.zeros(
                y.shape[1],
            )
        else:
            init_w = jnp.zeros((X.shape[1] + delta_n_features))
            init_b = jnp.zeros(
                1,
            )
        with expectation:
            model.fit(X, y, init_params=(init_w, init_b))

    @pytest.mark.parametrize(
        "delta_n_features, expectation",
        [
            (-1, pytest.raises(ValueError, match="Inconsistent number of features")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="Inconsistent number of features")),
        ],
    )
    @pytest.mark.solver_related
    def test_fit_n_feature_consistency_x(
        self,
        delta_n_features,
        expectation,
        request,
        glm_class_type,
        model_instantiation_type,
    ):
        """
        Test the `fit` method for inconsistencies between data features and model's expectations.
        Ensure the number of features in X aligns.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        if delta_n_features == 1:
            X = jnp.concatenate((X, jnp.zeros((X.shape[0], 1))), axis=1)
        elif delta_n_features == -1:
            X = X[..., :-1]
        with expectation:
            model.fit(X, y, init_params=true_params)

    @pytest.mark.parametrize(
        "delta_tp, expectation",
        [
            (
                -1,
                pytest.raises(ValueError, match="The number of time-points in X and y"),
            ),
            (0, does_not_raise()),
            (
                1,
                pytest.raises(ValueError, match="The number of time-points in X and y"),
            ),
        ],
    )
    @pytest.mark.solver_related
    def test_fit_time_points_x(
        self, delta_tp, expectation, request, glm_class_type, model_instantiation_type
    ):
        """
        Test the `fit` method for inconsistencies in time-points in data X. Ensure the correct number of time-points.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        X = jnp.zeros((X.shape[0] + delta_tp,) + X.shape[1:])
        with expectation:
            model.fit(X, y, init_params=true_params)

    @pytest.mark.parametrize(
        "delta_tp, expectation",
        [
            (
                -1,
                pytest.raises(ValueError, match="The number of time-points in X and y"),
            ),
            (0, does_not_raise()),
            (
                1,
                pytest.raises(ValueError, match="The number of time-points in X and y"),
            ),
        ],
    )
    @pytest.mark.solver_related
    def test_fit_time_points_y(
        self, delta_tp, expectation, request, glm_class_type, model_instantiation_type
    ):
        """
        Test the `fit` method for inconsistencies in time-points in y. Ensure the correct number of time-points.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        y = jnp.zeros((y.shape[0] + delta_tp,) + y.shape[1:])
        with expectation:
            model.fit(X, y, init_params=true_params)

    @pytest.mark.parametrize(
        "fill_val, expectation",
        [
            (0, does_not_raise()),
            (
                jnp.inf,
                pytest.raises(
                    ValueError, match="At least a NaN or an Inf at all sample points"
                ),
            ),
            (
                jnp.nan,
                pytest.raises(
                    ValueError, match="At least a NaN or an Inf at all sample points"
                ),
            ),
        ],
    )
    @pytest.mark.solver_related
    def test_fit_all_invalid_X(
        self, fill_val, expectation, request, glm_class_type, model_instantiation_type
    ):
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        X.fill(fill_val)
        with expectation:
            model.fit(X, y)

    #######################
    # Test model.score
    #######################
    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (-1, pytest.raises(ValueError, match="X must be two-dimensional")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="X must be two-dimensional")),
        ],
    )
    def test_score_x_dimensionality(
        self, delta_dim, expectation, request, glm_class_type, model_instantiation_type
    ):
        """
        Test the `score` method with X input data of different dimensionalities. Ensure correct dimensionality for X.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        if delta_dim == -1:
            X = np.zeros((X.shape[0],))
        elif delta_dim == 1:
            X = np.zeros((X.shape[0], X.shape[1], 1))
        with expectation:
            model.score(X, y)

    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (
                -1,
                pytest.raises(
                    ValueError, match=r"y must be (one|two)-dimensional, with shape"
                ),
            ),
            (0, does_not_raise()),
            (
                1,
                pytest.raises(
                    ValueError, match=r"y must be (one|two)-dimensional, with shape"
                ),
            ),
        ],
    )
    def test_score_y_dimensionality(
        self, delta_dim, expectation, request, glm_class_type, model_instantiation_type
    ):
        """
        Test the `score` method with y of different dimensionalities.
        Ensure correct dimensionality for y.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        if "population" in glm_class_type:
            if delta_dim == -1:
                y = y[:, 0]
            elif delta_dim == 1:
                y = np.zeros((*y.shape, 1))
        else:
            if delta_dim == -1:
                y = np.zeros([])
            elif delta_dim == 1:
                y = np.zeros((X.shape[0], X.shape[1]))
        with expectation:
            model.score(X, y)

    @pytest.mark.parametrize(
        "delta_n_features, expectation",
        [
            (-1, pytest.raises(ValueError, match="Inconsistent number of features")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="Inconsistent number of features")),
        ],
    )
    def test_score_n_feature_consistency_x(
        self,
        delta_n_features,
        expectation,
        request,
        glm_class_type,
        model_instantiation_type,
    ):
        """
        Test the `score` method for inconsistencies in features of X.
        Ensure the number of features in X aligns with the model params.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        if delta_n_features == 1:
            X = jnp.concatenate((X, jnp.zeros((X.shape[0], 1))), axis=1)
        elif delta_n_features == -1:
            X = X[..., :-1]
        with expectation:
            model.score(X, y)

    @pytest.mark.parametrize(
        "delta_tp, expectation",
        [
            (
                -1,
                pytest.raises(ValueError, match="The number of time-points in X and y"),
            ),
            (0, does_not_raise()),
            (
                1,
                pytest.raises(ValueError, match="The number of time-points in X and y"),
            ),
        ],
    )
    def test_score_time_points_x(
        self, delta_tp, expectation, request, glm_class_type, model_instantiation_type
    ):
        """
        Test the `score` method for inconsistencies in time-points in X.
        Ensure that the number of time-points in X and y matches.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        X = jnp.zeros((X.shape[0] + delta_tp,) + X.shape[1:])
        with expectation:
            model.score(X, y)

    @pytest.mark.parametrize(
        "delta_tp, expectation",
        [
            (
                -1,
                pytest.raises(ValueError, match="The number of time-points in X and y"),
            ),
            (0, does_not_raise()),
            (
                1,
                pytest.raises(ValueError, match="The number of time-points in X and y"),
            ),
        ],
    )
    def test_score_time_points_y(
        self, delta_tp, expectation, request, glm_class_type, model_instantiation_type
    ):
        """
        Test the `score` method for inconsistencies in time-points in y.
        Ensure that the number of time-points in X and y matches.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        y = jnp.zeros((y.shape[0] + delta_tp,) + y.shape[1:])
        with expectation:
            model.score(X, y)

    #######################
    # Test model.predict
    #######################
    @pytest.mark.parametrize(
        "is_fit, expectation",
        [
            (True, does_not_raise()),
            (
                False,
                pytest.raises(ValueError, match="This GLM instance is not fitted yet"),
            ),
        ],
    )
    def test_predict_is_fit(
        self, is_fit, expectation, request, glm_class_type, model_instantiation_type
    ):
        """
        Test the `score` method on models based on their fit status.
        Ensure scoring is only possible on fitted models.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        if is_fit:
            model.fit(X, y)
        with expectation:
            model.predict(X)

    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (-1, pytest.raises(ValueError, match="X must be two-dimensional")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="X must be two-dimensional")),
        ],
    )
    def test_predict_x_dimensionality(
        self, delta_dim, expectation, request, glm_class_type, model_instantiation_type
    ):
        """
        Test the `predict` method with x input data of different dimensionalities.
        Ensure correct dimensionality for x.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        if "population" in glm_class_type:
            model._initialize_feature_mask(X, y)
        if delta_dim == -1:
            X = np.zeros((X.shape[0],))
        elif delta_dim == 1:
            X = np.zeros((X.shape[0], X.shape[1], 1))
        with expectation:
            model.predict(X)

    @pytest.mark.parametrize(
        "delta_n_features, expectation",
        [
            (-1, pytest.raises(ValueError, match="Inconsistent number of features")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="Inconsistent number of features")),
        ],
    )
    def test_predict_n_feature_consistency_x(
        self,
        delta_n_features,
        expectation,
        request,
        glm_class_type,
        model_instantiation_type,
    ):
        """
        Test the `predict` method ensuring the number of features in x input data
        is consistent with the model's `model.coef_`.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        if "population" in glm_class_type:
            model._initialize_feature_mask(X, y)
        if delta_n_features == 1:
            X = jnp.concatenate((X, jnp.zeros((X.shape[0], 1))), axis=1)
        elif delta_n_features == -1:
            X = X[..., :-1]
        with expectation:
            model.predict(X)

    #############################
    # Test model.initialize_state
    #############################
    @pytest.mark.parametrize(
        "n_params, expectation",
        [
            (0, pytest.raises(ValueError, match="Params must have length two.")),
            (1, pytest.raises(ValueError, match="Params must have length two.")),
            (2, does_not_raise()),
            (3, pytest.raises(ValueError, match="Params must have length two.")),
        ],
    )
    @pytest.mark.solver_related
    def test_initialize_solver_param_length(
        self, n_params, expectation, request, glm_class_type, model_instantiation_type
    ):
        """
        Test the `initialize_solver` method with different numbers of initial parameters.
        Check for correct number of parameters.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        if n_params == 0:
            init_params = tuple()
        elif n_params == 1:
            init_params = (true_params[0],)
        else:
            init_params = true_params + (true_params[0],) * (n_params - 2)
        with expectation:
            params = model.initialize_params(X, y, init_params=init_params)
            # check that params are set
            init_state = model.initialize_state(X, y, params)

    @pytest.fixture
    def initialize_solver_weights_dimensionality_expectation(self, glm_class_type):
        if "population" in glm_class_type:
            return {
                0: pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
                1: pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
                2: does_not_raise(),
                3: pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
            }
        else:
            return {
                0: pytest.raises(
                    ValueError,
                    match=r"Inconsistent number of features",
                ),
                1: does_not_raise(),
                2: pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
                3: pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
            }

    @pytest.mark.parametrize("dim_weights", [0, 1, 2, 3])
    @pytest.mark.solver_related
    def test_initialize_solver_weights_dimensionality(
        self,
        dim_weights,
        request,
        glm_class_type,
        model_instantiation_type,
        initialize_solver_weights_dimensionality_expectation,
    ):
        """
        Test the `initialize_solver` method with weight matrices of different dimensionalities.
        Check for correct dimensionality.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        expectation = initialize_solver_weights_dimensionality_expectation[dim_weights]
        n_samples, n_features = X.shape
        if "population" in glm_class_type:
            n_neurons = 3
        else:
            n_neurons = 4
        if dim_weights == 0:
            init_w = jnp.array([])
        elif dim_weights == 1:
            init_w = jnp.zeros((n_features,))
        elif dim_weights == 2:
            init_w = jnp.zeros((n_features, n_neurons))
        else:
            init_w = jnp.zeros((n_features, n_neurons) + (1,) * (dim_weights - 2))
        with expectation:
            params = model.initialize_params(X, y, init_params=(init_w, true_params[1]))
            # check that params are set
            init_state = model.initialize_state(X, y, params)

    @pytest.mark.parametrize(
        "dim_intercepts, expectation",
        [
            (0, pytest.raises(ValueError, match=r"params\[1\] must be of shape")),
            (1, does_not_raise()),
            (2, pytest.raises(ValueError, match=r"params\[1\] must be of shape")),
            (3, pytest.raises(ValueError, match=r"params\[1\] must be of shape")),
        ],
    )
    @pytest.mark.solver_related
    def test_initialize_solver_intercepts_dimensionality(
        self,
        dim_intercepts,
        expectation,
        request,
        glm_class_type,
        model_instantiation_type,
    ):
        """
        Test the `initialize_solver` method with intercepts of different dimensionalities.

        Check for correct dimensionality.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        n_samples, n_features = X.shape
        if "population" in glm_class_type:
            init_b = jnp.zeros((y.shape[1],) * dim_intercepts)
            init_w = jnp.zeros((n_features, y.shape[1]))
        else:
            init_b = jnp.zeros((1,) * dim_intercepts)
            init_w = jnp.zeros((n_features,))
        with expectation:
            params = model.initialize_params(X, y, init_params=(init_w, init_b))
            # check that params are set
            init_state = model.initialize_state(X, y, params)

    @pytest.mark.parametrize(*fit_init_params_type_init_params)
    @pytest.mark.solver_related
    def test_initialize_solver_init_params_type(
        self,
        request,
        glm_class_type,
        model_instantiation_type,
        expectation,
        init_params_glm,
        init_params_population_glm,
    ):
        """
        Test the `initialize_solver` method with various types of initial parameters.

        Ensure that the provided initial parameters are array-like.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        if "population" in glm_class_type:
            init_params = init_params_population_glm
        else:
            init_params = init_params_glm
        with expectation:
            params = model.initialize_params(X, y, init_params=init_params)
            # check that params are set
            init_state = model.initialize_state(X, y, params)

    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (-1, pytest.raises(ValueError, match="X must be two-dimensional")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="X must be two-dimensional")),
        ],
    )
    @pytest.mark.solver_related
    def test_initialize_solver_x_dimensionality(
        self, delta_dim, expectation, request, glm_class_type, model_instantiation_type
    ):
        """
        Test the `initialize_solver` method with X input data of different dimensionalities.

        Ensure correct dimensionality for X.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        if delta_dim == -1:
            X = np.zeros((X.shape[0],))
        elif delta_dim == 1:
            X = np.zeros((X.shape[0], 1, X.shape[1]))
        with expectation:
            params = model.initialize_params(X, y, init_params=true_params)
            # check that params are set
            init_state = model.initialize_state(X, y, params)

    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (-1, pytest.raises(ValueError, match="y must be ...-dimensional")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="y must be ...-dimensional")),
        ],
    )
    @pytest.mark.solver_related
    def test_initialize_solver_y_dimensionality(
        self, delta_dim, expectation, request, glm_class_type, model_instantiation_type
    ):
        """
        Test the `initialize_solver` method with y target data of different dimensionalities.

        Ensure correct dimensionality for y.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        if "population" in glm_class_type:
            if delta_dim == -1:
                y = y[:, 0]
            elif delta_dim == 1:
                y = np.zeros((*y.shape, 1))
        else:
            if delta_dim == -1:
                y = np.zeros([])
            elif delta_dim == 1:
                y = np.zeros((y.shape[0], 1))
        with expectation:
            params = model.initialize_params(X, y, init_params=true_params)
            # check that params are set
            init_state = model.initialize_state(X, y, params)

    @pytest.mark.parametrize(
        "delta_n_features, expectation",
        [
            (-1, pytest.raises(ValueError, match="Inconsistent number of features")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="Inconsistent number of features")),
        ],
    )
    @pytest.mark.solver_related
    def test_initialize_solver_n_feature_consistency_weights(
        self,
        delta_n_features,
        expectation,
        request,
        glm_class_type,
        model_instantiation_type,
    ):
        """
        Test the `initialize_solver` method for inconsistencies between data features and initial weights provided.
        Ensure the number of features align.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        if "population" in glm_class_type:
            init_w = jnp.zeros((X.shape[1] + delta_n_features, y.shape[1]))
            init_b = jnp.zeros(
                y.shape[1],
            )
        else:
            init_w = jnp.zeros((X.shape[1] + delta_n_features))
            init_b = jnp.zeros(
                1,
            )
        with expectation:
            params = model.initialize_params(X, y, init_params=(init_w, init_b))
            # check that params are set
            init_state = model.initialize_state(X, y, params)

    @pytest.mark.parametrize(
        "delta_n_features, expectation",
        [
            (-1, pytest.raises(ValueError, match="Inconsistent number of features")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="Inconsistent number of features")),
        ],
    )
    @pytest.mark.solver_related
    def test_initialize_solver_n_feature_consistency_x(
        self,
        delta_n_features,
        expectation,
        request,
        glm_class_type,
        model_instantiation_type,
    ):
        """
        Test the `initialize_solver` method for inconsistencies between data features and model's expectations.
        Ensure the number of features in X aligns.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        if delta_n_features == 1:
            X = jnp.concatenate((X, jnp.zeros((X.shape[0], 1))), axis=1)
        elif delta_n_features == -1:
            X = X[..., :-1]
        with expectation:
            params = model.initialize_params(X, y, init_params=true_params)
            # check that params are set
            init_state = model.initialize_state(X, y, params)

    @pytest.mark.parametrize(
        "delta_tp, expectation",
        [
            (
                -1,
                pytest.raises(ValueError, match="The number of time-points in X and y"),
            ),
            (0, does_not_raise()),
            (
                1,
                pytest.raises(ValueError, match="The number of time-points in X and y"),
            ),
        ],
    )
    @pytest.mark.solver_related
    def test_initialize_solver_time_points_x(
        self, delta_tp, expectation, request, glm_class_type, model_instantiation_type
    ):
        """
        Test the `initialize_solver` method for inconsistencies in time-points in data X.

        Ensure the correct number of time-points.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        X = jnp.zeros((X.shape[0] + delta_tp,) + X.shape[1:])
        with expectation:
            params = model.initialize_params(X, y, init_params=true_params)
            # check that params are set
            init_state = model.initialize_state(X, y, params)

    @pytest.mark.parametrize(
        "delta_tp, expectation",
        [
            (
                -1,
                pytest.raises(ValueError, match="The number of time-points in X and y"),
            ),
            (0, does_not_raise()),
            (
                1,
                pytest.raises(ValueError, match="The number of time-points in X and y"),
            ),
        ],
    )
    @pytest.mark.solver_related
    def test_initialize_solver_time_points_y(
        self, delta_tp, expectation, request, glm_class_type, model_instantiation_type
    ):
        """
        Test the `initialize_solver` method for inconsistencies in time-points in y.

        Ensure the correct number of time-points.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        y = jnp.zeros((y.shape[0] + delta_tp,) + y.shape[1:])
        with expectation:
            params = model.initialize_params(X, y, init_params=true_params)
            # check that params are set
            init_state = model.initialize_state(X, y, params)

    @pytest.mark.solver_related
    def test_initialize_solver_mask_grouplasso(
        self, request, glm_class_type, model_instantiation_type
    ):
        """Test that the group lasso initialize_solver goes through"""
        X, y, model, params, rate, mask = request.getfixturevalue(
            model_instantiation_type + "_group_sparse"
        )
        model.set_params(
            regularizer=nmo.regularizer.GroupLasso(mask=mask),
            solver_name="ProximalGradient",
            regularizer_strength=1.0,
        )
        params = model.initialize_params(X, y)
        init_state = model.initialize_state(X, y, params)

    @pytest.mark.parametrize(
        "fill_val, expectation",
        [
            (0, does_not_raise()),
            (
                jnp.inf,
                pytest.raises(
                    ValueError, match="At least a NaN or an Inf at all sample points"
                ),
            ),
            (
                jnp.nan,
                pytest.raises(
                    ValueError, match="At least a NaN or an Inf at all sample points"
                ),
            ),
        ],
    )
    @pytest.mark.solver_related
    def test_initialize_solver_all_invalid_X(
        self, fill_val, expectation, request, glm_class_type, model_instantiation_type
    ):
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        X.fill(fill_val)
        with expectation:
            params = model.initialize_params(X, y)
            init_state = model.initialize_state(X, y, params)

    #######################
    # Test model.simulate
    #######################
    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (-1, pytest.raises(ValueError, match="X must be two-dimensional")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="X must be two-dimensional")),
        ],
    )
    def test_simulate_input_dimensionality(
        self, delta_dim, expectation, request, glm_class_type, model_instantiation_type
    ):
        """
        Test the `simulate` method with input data of different dimensionalities.
        Ensure correct dimensionality for input.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        if "population" in glm_class_type:
            model._initialize_feature_mask(X, y)
        if delta_dim == -1:
            X = np.zeros(X.shape[:-1])
        elif delta_dim == 1:
            X = np.zeros(X.shape + (1,))
        with expectation:
            model.simulate(
                random_key=jax.random.key(123),
                feedforward_input=X,
            )

    @pytest.mark.parametrize(
        "is_fit, expectation",
        [
            (True, does_not_raise()),
            (
                False,
                pytest.raises(ValueError, match="This GLM instance is not fitted yet"),
            ),
        ],
    )
    def test_simulate_is_fit(
        self, is_fit, expectation, request, glm_class_type, model_instantiation_type
    ):
        """
        Test if the model raises a ValueError when trying to simulate before it's fitted.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        if is_fit:
            model.coef_ = true_params[0]
            model.intercept_ = true_params[1]
            if "population" in glm_class_type:
                model._initialize_feature_mask(X, y)
        with expectation:
            model.simulate(
                random_key=jax.random.key(123),
                feedforward_input=X,
            )

    @pytest.mark.parametrize(
        "delta_features, expectation",
        [
            (
                -1,
                pytest.raises(
                    ValueError,
                    match="Inconsistent number of features. spike basis coefficients has",
                ),
            ),
            (0, does_not_raise()),
            (
                1,
                pytest.raises(
                    ValueError,
                    match="Inconsistent number of features. spike basis coefficients has",
                ),
            ),
        ],
    )
    def test_simulate_feature_consistency_input(
        self,
        delta_features,
        expectation,
        request,
        glm_class_type,
        model_instantiation_type,
    ):
        """
        Test the `simulate` method ensuring the number of features in `feedforward_input` is
        consistent with the model's expected number of features.

        Notes
        -----
        The total feature number `model.coef_.shape[1]` must be equal to
        `feedforward_input.shape[2] + coupling_basis.shape[1]*n_neurons`
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        if "population" in glm_class_type:
            model._initialize_feature_mask(X, y)
        feedforward_input = jnp.zeros(
            (
                X.shape[0],
                X.shape[1] + delta_features,
            )
        )
        with expectation:
            model.simulate(
                random_key=jax.random.key(123),
                feedforward_input=feedforward_input,
            )

    #######################################
    # Compare with standard implementation
    #######################################

    @pytest.mark.parametrize("reg", ["Ridge", "Lasso", "GroupLasso", "ElasticNet"])
    def test_reg_strength_reset(self, reg, request, glm_class_type):
        glm_class = request.getfixturevalue(glm_class_type)
        model = glm_class(regularizer=reg, regularizer_strength=1.0)
        model.regularizer = "UnRegularized"
        assert model.regularizer_strength is None

    @pytest.mark.parametrize(
        "params, warns",
        [
            # set regularizer
            (
                {"regularizer": "Ridge"},
                does_not_raise(),
            ),
            (
                {"regularizer": "Lasso"},
                does_not_raise(),
            ),
            (
                {"regularizer": "GroupLasso"},
                does_not_raise(),
            ),
            (
                {"regularizer": "ElasticNet"},
                does_not_raise(),
            ),
            ({"regularizer": "UnRegularized"}, does_not_raise()),
            # set both None or number
            (
                {"regularizer": "Ridge", "regularizer_strength": None},
                does_not_raise(),
            ),
            ({"regularizer": "Ridge", "regularizer_strength": 1.0}, does_not_raise()),
            (
                {"regularizer": "Lasso", "regularizer_strength": None},
                does_not_raise(),
            ),
            ({"regularizer": "Lasso", "regularizer_strength": 1.0}, does_not_raise()),
            (
                {"regularizer": "GroupLasso", "regularizer_strength": None},
                does_not_raise(),
            ),
            (
                {"regularizer": "GroupLasso", "regularizer_strength": 1.0},
                does_not_raise(),
            ),
            (
                {"regularizer": "ElasticNet", "regularizer_strength": None},
                does_not_raise(),
            ),
            (
                {"regularizer": "ElasticNet", "regularizer_strength": 1.0},
                does_not_raise(),
            ),
            (
                {"regularizer": "ElasticNet", "regularizer_strength": (1.0, 0.5)},
                does_not_raise(),
            ),
            (
                {"regularizer": "UnRegularized", "regularizer_strength": None},
                does_not_raise(),
            ),
            (
                {"regularizer": "UnRegularized", "regularizer_strength": 1.0},
                does_not_raise(),
            ),
            # set regularizer str only
            (
                {"regularizer_strength": 1.0},
                does_not_raise(),
            ),
            ({"regularizer_strength": None}, does_not_raise()),
        ],
    )
    def test_reg_set_params(self, params, warns, request, glm_class_type):
        glm_class = request.getfixturevalue(glm_class_type)
        model = glm_class()
        with warns:
            model.set_params(**params)

    @pytest.mark.parametrize(
        "params, warns",
        [
            # set regularizer str only
            ({"regularizer_strength": 1.0}, does_not_raise()),
            (
                {"regularizer_strength": None},
                does_not_raise(),
            ),
        ],
    )
    @pytest.mark.parametrize("reg", ["Ridge", "Lasso", "GroupLasso"])
    def test_reg_set_params_reg_str_only(
        self, params, warns, reg, request, glm_class_type
    ):
        glm_class = request.getfixturevalue(glm_class_type)
        model = glm_class(regularizer=reg, regularizer_strength=1)
        with warns:
            model.set_params(**params)

    @pytest.mark.parametrize(
        "regularizer", ["Ridge", "UnRegularized", "Lasso", "ElasticNet"]
    )
    @pytest.mark.parametrize(
        "obs_model",
        [
            "PoissonObservations",
            "BernoulliObservations",
            "GammaObservations",
        ],
    )
    @pytest.mark.parametrize(
        "solver_name",
        [
            "GradientDescent",
            "BFGS",
            "LBFGS",
            "NonlinearCG",
            "ProximalGradient",
            "SVRG",
            "ProxSVRG",
        ],
    )
    @pytest.mark.parametrize(
        "model_class, fit_state_attrs",
        [
            (
                nmo.glm.GLM,
                {
                    "coef_": jnp.zeros(
                        3,
                    ),
                    "intercept_": jnp.array([1.0]),
                    "scale_": 2.0,
                    "dof_resid_": 3,
                },
            ),
            (
                nmo.glm.PopulationGLM,
                {
                    "coef_": jnp.zeros((3, 1)),
                    "intercept_": jnp.array([1.0]),
                    "scale_": 2.0,
                    "dof_resid_": 3,
                },
            ),
        ],
    )
    def test_save_and_load(
        self,
        regularizer,
        obs_model,
        solver_name,
        tmp_path,
        glm_class_type,
        fit_state_attrs,
        model_class,
    ):
        """
        Test saving and loading a model with various observation models and regularizers.
        Ensure all parameters are preserved.
        """
        if (
            regularizer == "Lasso"
            or regularizer == "GroupLasso"
            or regularizer == "ElasticNet"
            and solver_name not in ["ProximalGradient", "ProxSVRG"]
        ):
            pytest.skip(
                f"Skipping {solver_name} for Lasso type regularizer; not an approximate solver."
            )

        kwargs = dict(
            observation_model=obs_model,
            solver_name=solver_name,
            regularizer=regularizer,
            regularizer_strength=2.0,
            solver_kwargs={"tol": 10**-6},
        )

        if regularizer == "UnRegularized":
            kwargs.pop("regularizer_strength")

        model = model_class(**kwargs)

        initial_params = model.get_params()
        # set fit states
        for key, val in fit_state_attrs.items():
            setattr(model, key, val)
            initial_params[key] = val

        # Save
        save_path = tmp_path / "test_model.npz"
        model.save_params(save_path)

        # Load
        loaded_model = nmo.load_model(save_path)
        loaded_params = loaded_model.get_params()
        fit_state = loaded_model._get_fit_state()
        fit_state.pop("solver_state_")
        loaded_params.update(fit_state)

        # Assert matching keys and values
        assert (
            initial_params.keys() == loaded_params.keys()
        ), "Parameter keys mismatch after load."

        for key in initial_params:
            init_val = initial_params[key]
            load_val = loaded_params[key]
            if isinstance(init_val, (int, float, str, type(None))):
                assert init_val == load_val, f"{key} mismatch: {init_val} != {load_val}"
            elif isinstance(init_val, dict):
                assert (
                    init_val == load_val
                ), f"{key} dict mismatch: {init_val} != {load_val}"
            elif isinstance(init_val, (np.ndarray, jnp.ndarray)):
                assert np.allclose(
                    np.array(init_val), np.array(load_val)
                ), f"{key} array mismatch"
            elif isinstance(init_val, Callable):
                assert _get_name(init_val) == _get_name(
                    load_val
                ), f"{key} function mismatch: {_get_name(init_val)} != {_get_name(load_val)}"

    @pytest.mark.parametrize("regularizer", ["Ridge"])
    @pytest.mark.parametrize(
        "obs_model",
        [
            "PoissonObservations",
        ],
    )
    @pytest.mark.parametrize(
        "solver_name",
        [
            "ProxSVRG",
        ],
    )
    @pytest.mark.parametrize(
        "model_class, fit_state_attrs",
        [
            (
                nmo.glm.GLM,
                {
                    "coef_": jnp.zeros(
                        3,
                    ),
                    "intercept_": jnp.array([1.0]),
                    "scale_": 2.0,
                    "dof_resid_": 3,
                },
            ),
            (
                nmo.glm.PopulationGLM,
                {
                    "coef_": jnp.zeros(
                        (3, 1),
                    ),
                    "intercept_": jnp.array([1.0]),
                    "scale_": 2.0,
                    "dof_resid_": 3,
                },
            ),
        ],
    )
    @pytest.mark.parametrize(
        "mapping_dict, expectation",
        [
            ({}, does_not_raise()),
            (
                {
                    "observation_model": nmo.observation_models.GammaObservations,
                    "regularizer": nmo.regularizer.Lasso,
                    "inverse_link_function": lambda x: x**2,
                },
                pytest.warns(
                    UserWarning, match="The following keys have been replaced"
                ),
            ),
            (
                {
                    "observation_model": nmo.observation_models.GammaObservations(),  # fails, only class or callable
                    "regularizer": nmo.regularizer.Lasso,
                    "inverse_link_function": lambda x: x**2,
                },
                pytest.raises(ValueError, match="Invalid map parameter types detected"),
            ),
            (
                {
                    "observation_model": "GammaObservations",  # fails, only class or callable
                    "regularizer": nmo.regularizer.Lasso,
                },
                pytest.raises(ValueError, match="Invalid map parameter types detected"),
            ),
            (
                {
                    "regularizer": nmo.regularizer.Lasso,
                    "regularizer_strength": 3.0,  # fails, only class or callable
                },
                pytest.raises(ValueError, match="Invalid map parameter types detected"),
            ),
            (
                {
                    "solver_kwargs": {"tol": 10**-1},
                },
                pytest.raises(ValueError, match="Invalid map parameter types detected"),
            ),
            (
                {
                    "some__nested__dictionary": {"tol": 10**-1},
                },
                pytest.raises(
                    ValueError,
                    match="The following keys in your mapping do not match",
                ),
            ),
            # valid mapping dtype, invalid name
            (
                {
                    "some__nested__dictionary": nmo.regularizer.Ridge,
                },
                pytest.raises(
                    ValueError,
                    match="The following keys in your mapping do not match",
                ),
            ),
        ],
    )
    def test_save_and_load_with_custom_mapping(
        self,
        regularizer,
        obs_model,
        solver_name,
        mapping_dict,
        tmp_path,
        glm_class_type,
        fit_state_attrs,
        model_class,
        expectation,
    ):
        """
        Test saving and loading a model with various observation models and regularizers.
        Ensure all parameters are preserved.
        """

        if (
            regularizer == "Lasso"
            or regularizer == "GroupLasso"
            and solver_name not in ["ProximalGradient", "SVRG", "ProxSVRG"]
        ):
            pytest.skip(
                f"Skipping {solver_name} for Lasso type regularizer; not an approximate solver."
            )

        model = model_class(
            observation_model=obs_model,
            solver_name=solver_name,
            regularizer=regularizer,
            regularizer_strength=2.0,
        )

        initial_params = model.get_params()
        # set fit states
        for key, val in fit_state_attrs.items():
            setattr(model, key, val)
            initial_params[key] = val

        # Save
        save_path = tmp_path / "test_model.npz"
        model.save_params(save_path)

        # Load
        with expectation:
            loaded_model = nmo.load_model(save_path, mapping_dict=mapping_dict)
            loaded_params = loaded_model.get_params()
            fit_state = loaded_model._get_fit_state()
            fit_state.pop("solver_state_")
            loaded_params.update(fit_state)

            # Assert matching keys and values
            assert (
                initial_params.keys() == loaded_params.keys()
            ), "Parameter keys mismatch after load."

            unexpected_keys = set(mapping_dict) - set(initial_params)
            raise_exception = bool(unexpected_keys)
            if raise_exception:
                with pytest.raises(
                    ValueError, match="mapping_dict contains unexpected keys"
                ):
                    raise ValueError(
                        f"mapping_dict contains unexpected keys: {unexpected_keys}"
                    )

            for key in initial_params:
                init_val = initial_params[key]
                load_val = loaded_params[key]

                if key == "observation_model__inverse_link_function":
                    if "observation_model" in mapping_dict:
                        continue
                if key in mapping_dict:
                    if key == "observation_model":
                        if isinstance(mapping_dict[key], str):
                            mapping_obs = instantiate_observation_model(
                                mapping_dict[key]
                            )
                        else:
                            mapping_obs = mapping_dict[key]
                        assert _get_name(mapping_obs) == _get_name(
                            load_val
                        ), f"{key} observation model mismatch: {mapping_dict[key]} != {load_val}"
                    elif key == "regularizer":
                        if isinstance(mapping_dict[key], str):
                            mapping_reg = instantiate_regularizer(mapping_dict[key])
                        else:
                            mapping_reg = mapping_dict[key]
                        assert _get_name(mapping_reg) == _get_name(
                            load_val
                        ), f"{key} regularizer mismatch: {mapping_dict[key]} != {load_val}"
                    elif key == "solver_name":
                        assert (
                            mapping_dict[key] == load_val
                        ), f"{key} solver name mismatch: {mapping_dict[key]} != {load_val}"
                    elif key == "regularizer_strength":
                        assert (
                            mapping_dict[key] == load_val
                        ), f"{key} regularizer strength mismatch: {mapping_dict[key]} != {load_val}"
                    continue

            if isinstance(init_val, (int, float, str, type(None))):
                assert init_val == load_val, f"{key} mismatch: {init_val} != {load_val}"

            elif isinstance(init_val, dict):
                assert (
                    init_val == load_val
                ), f"{key} dict mismatch: {init_val} != {load_val}"

            elif isinstance(init_val, (np.ndarray, jnp.ndarray)):
                assert np.allclose(
                    np.array(init_val), np.array(load_val)
                ), f"{key} array mismatch"

            elif isinstance(init_val, Callable):
                assert _get_name(init_val) == _get_name(
                    load_val
                ), f"{key} function mismatch: {_get_name(init_val)} != {_get_name(load_val)}"

    def test_save_and_load_nested_class(
        self, nested_regularizer, tmp_path, glm_class_type
    ):
        """Test that save and load works with nested classes."""
        model = nmo.glm.GLM(regularizer=nested_regularizer, regularizer_strength=1.0)
        save_path = tmp_path / "test_model.npz"
        model.save_params(save_path)

        mapping_dict = {
            "regularizer": nested_regularizer.__class__,
            "regularizer__func": jnp.exp,
        }
        with pytest.warns(UserWarning, match="The following keys have been replaced"):
            loaded_model = nmo.load_model(save_path, mapping_dict=mapping_dict)

        assert isinstance(loaded_model.regularizer, nested_regularizer.__class__)
        assert isinstance(
            loaded_model.regularizer.sub_regularizer,
            nested_regularizer.sub_regularizer.__class__,
        )
        assert loaded_model.regularizer.func == mapping_dict["regularizer__func"]

        # change mapping
        mapping_dict = {
            "regularizer": nested_regularizer.__class__,
            "regularizer__sub_regularizer": nmo.regularizer.Ridge,
            "regularizer__func": lambda x: x**2,
        }
        with pytest.warns(UserWarning, match="The following keys have been replaced"):
            loaded_model = nmo.load_model(save_path, mapping_dict=mapping_dict)
        assert isinstance(loaded_model.regularizer, nested_regularizer.__class__)
        assert isinstance(
            loaded_model.regularizer.sub_regularizer, nmo.regularizer.Ridge
        )
        assert loaded_model.regularizer.func == mapping_dict["regularizer__func"]

    @pytest.mark.parametrize(
        "fitted_glm_type",
        [
            "poissonGLM_fitted_model_instantiation",
            "population_poissonGLM_fitted_model_instantiation",
        ],
    )
    def test_save_and_load_fitted_model(
        self, request, fitted_glm_type, glm_class_type, tmp_path
    ):
        """
        Test saving and loading a fitted model with various observation models and regularizers.
        Ensure all parameters are preserved.
        """
        _, _, fitted_model, _, _ = request.getfixturevalue(fitted_glm_type)

        initial_params = fitted_model.get_params()
        fit_state = fitted_model._get_fit_state()
        fit_state.pop("solver_state_")
        initial_params.update(fit_state)

        # Save
        save_path = tmp_path / "test_model.npz"
        fitted_model.save_params(save_path)

        # Load
        loaded_model = nmo.load_model(save_path)
        loaded_params = loaded_model.get_params()
        fit_state = loaded_model._get_fit_state()
        fit_state.pop("solver_state_")
        loaded_params.update(fit_state)

        # Assert states are close
        for k, v in fit_state.items():
            assert np.allclose(initial_params[k], v), f"{k} mismatch after load."

    @pytest.mark.parametrize(
        "fitted_glm_type",
        [
            "poissonGLM_fitted_model_instantiation",
            "population_poissonGLM_fitted_model_instantiation",
        ],
    )
    @pytest.mark.parametrize(
        "param_name, param_value, expectation",
        [
            # Replace observation model class name  with a string
            (
                "observation_model__class",
                "InvalidObservations",
                pytest.raises(
                    ValueError, match="The class '[A-z]+' is not a native NeMoS"
                ),
            ),
            # Full path string
            (
                "observation_model__class",
                "nemos.observation_models.InvalidObservations",
                pytest.raises(
                    ValueError, match="The class '[A-z]+' is not a native NeMoS"
                ),
            ),
            # Replace observation model class name  with an instance
            (
                "observation_model__class",
                nmo.observation_models.GammaObservations(),
                pytest.raises(
                    ValueError,
                    match="Object arrays cannot be loaded when allow_pickle=False",
                ),
            ),
            # Replace observation model class name with class
            (
                "observation_model__class",
                nmo.observation_models.GammaObservations,
                pytest.raises(
                    ValueError,
                    match="Object arrays cannot be loaded when allow_pickle=False",
                ),
            ),
            # Replace link function with another callable
            (
                "observation_model__params__inverse_link_function",
                np.exp,
                pytest.raises(
                    ValueError,
                    match="Object arrays cannot be loaded when allow_pickle=False",
                ),
            ),
            # Unexpected dtype for class name
            (
                "dict__regularizer__item__class",
                1,
                pytest.raises(
                    ValueError, match="Parameter ``regularizer`` cannot be initialized"
                ),
            ),
            # Invalid fit parameter
            (
                "scales_",  # wrong name for the params
                1,
                pytest.raises(ValueError, match="Unrecognized attribute 'scales_'"),
            ),
        ],
    )
    def test_modified_saved_file_raises(
        self,
        param_name,
        param_value,
        expectation,
        glm_class_type,
        fitted_glm_type,
        request,
        tmp_path,
    ):
        _, _, fitted_model, _, _ = request.getfixturevalue(fitted_glm_type)
        save_path = tmp_path / "test_model.npz"
        fitted_model.save_params(save_path)
        # load and edit
        data = np.load(save_path, allow_pickle=True)
        load_data = dict((k, v) for k, v in data.items())
        load_data[param_name] = param_value
        np.savez(save_path, **load_data, allow_pickle=True)

        with expectation:
            nmo.load_model(save_path)

    @pytest.mark.parametrize(
        "fitted_glm_type",
        [
            "poissonGLM_fitted_model_instantiation",
            "population_poissonGLM_fitted_model_instantiation",
        ],
    )
    def test_key_suggestions(self, fitted_glm_type, request, glm_class_type, tmp_path):
        _, _, fitted_model, _, _ = request.getfixturevalue(fitted_glm_type)
        save_path = tmp_path / "test_model.npz"
        fitted_model.save_params(save_path)

        invalid_mapping = {
            "regulsriaer": nmo.regularizer.Ridge,
            "observatino_mdels": nmo.observation_models.GammaObservations,
            "inv_link_function": jax.numpy.exp,
            "total_nonsense": jax.numpy.exp,
        }
        match = (
            r"The following keys in your mapping do not match any parameters in the loaded model:\n\n"
            r"\t- 'inv_link_function', did you mean 'inverse_link_function'\?\n"
            r"\t- 'observatino_mdels', did you mean 'observation_model'\?\n"
            r"\t- 'regulsriaer', did you mean 'regularizer'\?\n"
            r"\t- 'total_nonsense'\n\n"
            r"Please double-check your mapping dictionary\."
        )

        with pytest.raises(ValueError, match=match):
            nmo.load_model(save_path, mapping_dict=invalid_mapping)

        with pytest.raises(ValueError, match=match):
            nmo.load_model(save_path, mapping_dict=invalid_mapping)

    @pytest.mark.parametrize(
        "params, warns",
        [
            # set regularizer str only
            (
                {"regularizer_strength": 1.0},
                does_not_raise(),
            ),
            (
                {"regularizer_strength": None},
                does_not_raise(),
            ),
        ],
    )
    @pytest.mark.parametrize("reg", ["ElasticNet"])
    def test_reg_set_params_reg_str_only_elasticnet(
        self, params, warns, reg, request, glm_class_type
    ):
        glm_class = request.getfixturevalue(glm_class_type)
        model = glm_class(regularizer=reg, regularizer_strength=11)
        model.set_params(**params)
        assert model.regularizer_strength == (1.0, 0.5)


@pytest.mark.parametrize("glm_type", ["", "population_"])
@pytest.mark.parametrize(
    "model_instantiation",
    [
        "poissonGLM_model_instantiation",
        "gammaGLM_model_instantiation",
        "bernoulliGLM_model_instantiation",
        "negativeBinomialGLM_model_instantiation",
    ],
)
class TestGLMObservationModel:
    """
    Shared unit tests of the GLM class that do depend on obeservation model.
    i.e. tests that directly depend on observation model methods (e.g. model.fit, model.score, model.update),
    and tests that inspect the output when obervation model methods are called.

    For new observation models, add it in the class parameterization above, and add cases for the fixtures below.
    """

    @pytest.mark.parametrize(
        "link_func_string, expectation",
        [
            *((link_name, does_not_raise()) for link_name in LINK_NAME_TO_FUNC),
            (
                "nemos.utils.invalid_link",
                pytest.raises(ValueError, match="Unknown link function"),
            ),
            (
                "jax.numpy.invalid_link",
                pytest.raises(ValueError, match="Unknown link function"),
            ),
            ("invalid", pytest.raises(ValueError, match="Unknown link function")),
        ],
    )
    def test_glm_link_func_from_string(
        self, link_func_string, expectation, model_instantiation, glm_type, request
    ):
        _, _, model, _, _ = request.getfixturevalue(glm_type + model_instantiation)
        with expectation:
            model.__class__(inverse_link_function=link_func_string)

    ########################################################
    # Observation model specific fixtures for shared tests #
    ########################################################
    @pytest.fixture
    def ll_scipy_stats(self, model_instantiation):
        """
        Fixture for test_loglikelihood_against_scipy_stats
        """
        if "poisson" in model_instantiation:

            def ll(y, mean_firing):
                return jax.scipy.stats.poisson.logpmf(y, mean_firing).mean()

        elif "gamma" in model_instantiation:

            def ll(y, mean_firing, scale):
                if y.ndim == 1:
                    norm = y.shape[0]
                elif y.ndim == 2:
                    norm = y.shape[0] * y.shape[1]
                return sm.families.Gamma().loglike(y, mean_firing, scale=scale) / norm

        elif "bernoulli" in model_instantiation:

            def ll(y, mean_firing):
                return jax.scipy.stats.bernoulli.logpmf(y, mean_firing).mean()

        elif "negativeBinomial" in model_instantiation:

            def ll(y, mean_firing):
                if y.ndim == 1:
                    norm = y.shape[0]
                elif y.ndim == 2:
                    norm = y.shape[0] * y.shape[1]
                return (
                    sm.families.NegativeBinomial(alpha=1.0).loglike(y, mean_firing)
                    / norm
                )

        else:
            raise ValueError("Unknown model instantiation")
        return ll

    @pytest.fixture
    def sklearn_model(self, model_instantiation):
        """
        Fixture for test_glm_fit_matches_sklearn
        """
        if "poisson" in model_instantiation:
            return PoissonRegressor(fit_intercept=True, tol=10**-12, alpha=0.0)

        elif "gamma" in model_instantiation:
            return GammaRegressor(fit_intercept=True, tol=10**-12, alpha=0.0)

        elif "bernoulli" in model_instantiation:
            return LogisticRegression(
                fit_intercept=True,
                tol=10**-12,
                penalty=None,
            )

        elif "negativeBinomial" in model_instantiation:
            return None

        else:
            raise ValueError("Unknown model instantiation")

    @pytest.fixture
    def dof_lasso_strength(self, model_instantiation):
        """
        Fixture for test_estimate_dof_resid
        """
        if "poisson" in model_instantiation:
            return 1.0

        elif "gamma" in model_instantiation:
            return 0.02

        elif "bernoulli" in model_instantiation:
            return 0.1

        elif "negativeBinomial" in model_instantiation:
            return 0.01

        else:
            raise ValueError("Unknown model instantiation")

    @pytest.fixture
    def dof_lasso_dof(self, glm_type, model_instantiation):
        """
        Fixture for test_estimate_dof_resid
        """
        if "poisson" in model_instantiation:
            if "population" in glm_type:
                return np.array([3, 0, 0])
            else:
                return np.array([3])

        elif "gamma" in model_instantiation:
            if "population" in glm_type:
                return np.array([1, 4, 3])
            else:
                return np.array([3])

        elif "bernoulli" in model_instantiation:
            if "population" in glm_type:
                return np.array([3, 2, 1])
            else:
                return np.array([3])

        elif "negativeBinomial" in model_instantiation:
            if "population" in glm_type:
                return np.array([3, 2, 4])
            else:
                return np.array([5])

        else:
            raise ValueError("Unknown model instantiation")

    @pytest.fixture
    def obs_has_defaults(self, model_instantiation):
        """
        Fixture for test_optimize_solver_params
        """
        if "poisson" in model_instantiation:
            return True

        elif "gamma" in model_instantiation:
            return False

        elif "bernoulli" in model_instantiation:
            return False

        elif "negativeBinomial" in model_instantiation:
            return False

        else:
            raise ValueError("Unknown model instantiation")

    @pytest.fixture
    def model_repr(self, glm_type, model_instantiation):
        """
        Fixture for test_repr_out
        """
        if "poisson" in model_instantiation:
            if "population" in glm_type:
                return "PopulationGLM(\n    observation_model=PoissonObservations(),\n    inverse_link_function=exp,\n    regularizer=UnRegularized(),\n    solver_name='GradientDescent'\n)"
            else:
                return "GLM(\n    observation_model=PoissonObservations(),\n    inverse_link_function=exp,\n    regularizer=UnRegularized(),\n    solver_name='GradientDescent'\n)"

        elif "gamma" in model_instantiation:
            if "population" in glm_type:
                return "PopulationGLM(\n    observation_model=GammaObservations(),\n    inverse_link_function=one_over_x,\n    regularizer=UnRegularized(),\n    solver_name='GradientDescent'\n)"
            else:
                return "GLM(\n    observation_model=GammaObservations(),\n    inverse_link_function=one_over_x,\n    regularizer=UnRegularized(),\n    solver_name='GradientDescent'\n)"

        elif "bernoulli" in model_instantiation:
            if "population" in glm_type:
                return "PopulationGLM(\n    observation_model=BernoulliObservations(),\n    inverse_link_function=logistic,\n    regularizer=UnRegularized(),\n    solver_name='GradientDescent'\n)"
            else:
                return "GLM(\n    observation_model=BernoulliObservations(),\n    inverse_link_function=logistic,\n    regularizer=UnRegularized(),\n    solver_name='GradientDescent'\n)"

        elif "negative_binomial":
            if "population" in glm_type:
                return "PopulationGLM(\n    observation_model=NegativeBinomialObservations(scale=1.0),\n    inverse_link_function=exp,\n    regularizer=UnRegularized(),\n    solver_name='LBFGS'\n)"
            else:
                return "GLM(\n    observation_model=NegativeBinomialObservations(scale=1.0),\n    inverse_link_function=exp,\n    regularizer=UnRegularized(),\n    solver_name='LBFGS'\n)"

        else:
            raise ValueError("Unknown model instantiation")

    #######################
    # Test initialization #
    #######################
    @pytest.mark.parametrize(
        "X, y",
        [
            (jnp.zeros((2, 4)), jnp.zeros((2,))),
            (jnp.ones((2, 4)), jnp.ones((2,))),
            (jnp.zeros((2, 4)), jnp.ones((2,))),
            (jnp.ones((2, 4)), jnp.zeros((2,))),
        ],
    )
    def test_parameter_initialization(
        self, X, y, request, glm_type, model_instantiation
    ):
        _, _, model, _, _ = request.getfixturevalue(glm_type + model_instantiation)

        # right now default initialization is specific to poissonGLMs and will fail for the others
        # TODO: this test will need to be updated once we move parameter initialization to be observation model specific
        if "population" in glm_type:
            y = np.tile(y[:, None], (1, 3))

        if "poisson" in model_instantiation:
            coef, inter = model._initialize_parameters(X, y)

            if "population" in glm_type:
                assert coef.shape == (X.shape[1], y.shape[1])
                assert inter.shape == (y.shape[1],)
            else:
                assert coef.shape == (X.shape[1],)
                assert inter.shape == (1,)
        else:
            return

    ###################
    # Test get_params #
    ###################
    def test_get_params(self, request, glm_type, model_instantiation):
        """
        Test that get_params() contains expected values.
        """
        _, _, model, _, _ = request.getfixturevalue(glm_type + model_instantiation)
        if "population" in glm_type:
            expected_keys = GLM_COMMON_PARAMS_NAMES.union(
                OBSERVATION_MODEL_EXTRA_PARAMS_NAMES.get(
                    model.observation_model.__class__.__name__, {}
                )
            ).union(POPULATION_GLM_EXTRA_PARAMS)

            def get_expected_values(model):
                vals = [
                    model.feature_mask,
                    model.inverse_link_function,
                    model.observation_model,
                    model.regularizer,
                    model.regularizer_strength,
                    model.solver_kwargs,
                    model.solver_name,
                ]
                if isinstance(model.observation_model, NegativeBinomialObservations):
                    vals = vals[:2] + [model.observation_model.scale] + vals[2:]
                return vals

        else:
            expected_keys = GLM_COMMON_PARAMS_NAMES.union(
                OBSERVATION_MODEL_EXTRA_PARAMS_NAMES.get(
                    model.observation_model.__class__.__name__, {}
                )
            )

            def get_expected_values(model):
                vals = [
                    model.inverse_link_function,
                    model.observation_model,
                    model.regularizer,
                    model.regularizer_strength,
                    model.solver_kwargs,
                    model.solver_name,
                ]
                if isinstance(model.observation_model, NegativeBinomialObservations):
                    vals = vals[:1] + [model.observation_model.scale] + vals[1:]
                return vals

        expected_values = get_expected_values(model)
        assert set(model.get_params().keys()) == expected_keys
        assert list(model.get_params().values()) == expected_values

        # passing params
        model = type(model)(
            observation_model=model.observation_model,
            solver_name="LBFGS",
            regularizer="UnRegularized",
        )

        expected_values = get_expected_values(model)
        assert set(model.get_params().keys()) == expected_keys
        assert list(model.get_params().values()) == expected_values

        # changing regularizer
        model.set_params(regularizer="Ridge", regularizer_strength=1.0)

        expected_values = get_expected_values(model)
        assert set(model.get_params().keys()) == expected_keys
        assert list(model.get_params().values()) == expected_values

        # changing solver
        model.solver_name = "ProximalGradient"

        expected_values = get_expected_values(model)
        assert set(model.get_params().keys()) == expected_keys
        assert list(model.get_params().values()) == expected_values

    ##################
    # Test model.fit #
    ##################
    @pytest.mark.solver_related
    def test_fit_mask_grouplasso(self, request, glm_type, model_instantiation):
        """Test that the group lasso fit goes through"""

        if "poisson" in model_instantiation:
            X, y, model, params, rate, mask = request.getfixturevalue(
                glm_type + model_instantiation + "_group_sparse"
            )
            model.set_params(
                regularizer_strength=1.0,
                regularizer=nmo.regularizer.GroupLasso(mask=mask),
                solver_name="ProximalGradient",
            )
            model.fit(X, y)
        else:
            # TODO: need to define this fixture for the other models
            return

    @pytest.mark.solver_related
    def test_fit_pytree_equivalence(self, request, glm_type, model_instantiation):
        """Check that the glm fit with pytree learns the same parameters."""

        # required for numerical precision of coeffs
        jax.config.update("jax_enable_x64", True)
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        X_tree, _, model_tree, true_params_tree, _ = request.getfixturevalue(
            glm_type + model_instantiation + "_pytree"
        )
        # fit both models
        model.fit(X, y, init_params=true_params)
        model_tree.fit(X_tree, y, init_params=true_params_tree)

        # get the flat parameters
        if "population" in glm_type:
            flat_coef = np.concatenate(
                jax.tree_util.tree_flatten(model_tree.coef_)[0], axis=0
            )
        else:
            flat_coef = np.concatenate(
                jax.tree_util.tree_flatten(model_tree.coef_)[0], axis=-1
            )

        # assert equivalence of solutions
        assert np.allclose(model.coef_, flat_coef)
        assert np.allclose(model.intercept_, model_tree.intercept_)
        assert np.allclose(model.score(X, y), model_tree.score(X_tree, y))
        assert np.allclose(model.predict(X), model_tree.predict(X_tree))
        assert np.allclose(model.scale_, model_tree.scale_)

    ####################
    # Test model.score #
    ####################
    @pytest.mark.parametrize(
        "score_type, expectation",
        [
            ("pseudo-r2-McFadden", does_not_raise()),
            ("pseudo-r2-Cohen", does_not_raise()),
            ("log-likelihood", does_not_raise()),
            (
                "not-implemented",
                pytest.raises(
                    NotImplementedError,
                    match="Scoring method not-implemented not implemented",
                ),
            ),
        ],
    )
    def test_score_type_r2(
        self, score_type, expectation, request, glm_type, model_instantiation
    ):
        """
        Test the `score` method for unsupported scoring types.
        Ensure only valid score types are used.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        with expectation:
            model.score(X, y, score_type=score_type)

    def test_loglikelihood_against_scipy_stats(
        self, request, glm_type, model_instantiation, ll_scipy_stats
    ):
        """
        Compare the model's log-likelihood computation against `jax.scipy`.
        Ensure consistent and correct calculations.
        """
        jax.config.update("jax_enable_x64", True)

        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        # set model coeff
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        if "population" in glm_type:
            model._initialize_feature_mask(X, y)
        # get the rate
        mean_firing = model.predict(X)
        # compute the log-likelihood using jax.scipy
        if "gamma" in model_instantiation:
            mean_ll_jax = ll_scipy_stats(y, mean_firing, model.scale_)
        else:
            mean_ll_jax = ll_scipy_stats(y, mean_firing)

        model_ll = model.score(X, y, score_type="log-likelihood")
        if not np.allclose(mean_ll_jax, model_ll):
            raise ValueError(
                f"Log-likelihood of {glm_type + model_instantiation} does not match "
                "that of jax.scipy!"
            )

    ###############################
    # Test model.initialize_state #
    ###############################
    @pytest.mark.solver_related
    def test_initializer_solver_set_solver_callable(
        self, request, glm_type, model_instantiation
    ):
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        assert model.solver_init_state is None
        assert model.solver_update is None
        assert model.solver_run is None
        init_params = model.initialize_params(X, y)
        model.initialize_state(X, y, init_params)
        assert isinstance(model.solver_init_state, Callable)
        assert isinstance(model.solver_update, Callable)
        assert isinstance(model.solver_run, Callable)

    #####################
    # Test model.update #
    #####################
    @pytest.mark.parametrize(
        "n_samples, expectation",
        [
            (None, does_not_raise()),
            (100, does_not_raise()),
            (1.0, pytest.raises(TypeError, match="`n_samples` must either `None` or")),
            (
                "str",
                pytest.raises(TypeError, match="`n_samples` must either `None` or"),
            ),
        ],
    )
    @pytest.mark.parametrize("batch_size", [1, 10])
    @pytest.mark.solver_related
    def test_update_n_samples(
        self, n_samples, expectation, batch_size, request, glm_type, model_instantiation
    ):
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        params = model.initialize_params(X, y)
        state = model.initialize_state(X, y, params)
        with expectation:
            model.update(
                params, state, X[:batch_size], y[:batch_size], n_samples=n_samples
            )

    @pytest.mark.parametrize("batch_size", [1, 10])
    @pytest.mark.solver_related
    def test_update_params_stored(
        self, batch_size, request, glm_type, model_instantiation
    ):
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        params = model.initialize_params(X, y)
        state = model.initialize_state(X, y, params)
        assert model.coef_ is None
        assert model.intercept_ is None
        if "gamma" not in model_instantiation:
            # gamma model instantiation sets the scale
            assert model.scale_ is None
        _, _ = model.update(params, state, X[:batch_size], y[:batch_size])
        assert model.coef_ is not None
        assert model.intercept_ is not None
        assert model.scale_ is not None

    @pytest.mark.parametrize("nan_inputs", [True, False])
    @pytest.mark.parametrize(
        "solver_name", ["ProximalGradient", "GradientDescent", "LBFGS"]
    )
    @pytest.mark.solver_related
    def test_update_params_are_finite(
        self, nan_inputs, solver_name, request, glm_type, model_instantiation
    ):
        """
        Fitting a GLM to data containing NaNs with the jaxopt.LBFGS solver worked when using GLM.fit,
        but not when writing the training loop in Python and calling GLM.update repeatedly.
        The problem was that this solver uses the data to initialize its state, and the state was populated
        with NaNs by GLM.initialize_state.
        The solution was dropping NaNs from the input data in GLM.initialize_state -- as is done in GLM.update and GLM.run.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        model.solver_name = solver_name

        if nan_inputs:
            X[: X.shape[0] // 2, :] = np.nan

        params = model.initialize_params(X, y)
        state = model.initialize_state(X, y, params)

        assert model.coef_ is None
        assert model.intercept_ is None
        if "gamma" not in model_instantiation:
            # gamma model instantiation sets the scale
            assert model.scale_ is None

        # take an update step using the initialized state
        _, _ = model.update(params, state, X, y)

        assert model.coef_ is not None
        assert model.intercept_ is not None
        assert model.scale_ is not None

        # parameters should not have NaN
        assert jnp.all(jnp.isfinite(model.coef_))
        assert jnp.all(jnp.isfinite(model.intercept_))
        assert jnp.all(jnp.isfinite(model.scale_))

    @pytest.mark.parametrize("batch_size", [2, 10])
    @pytest.mark.solver_related
    def test_update_nan_drop_at_jit_comp(
        self, batch_size, request, glm_type, model_instantiation
    ):
        """Test that jit compilation does not affect the update in the presence of nans."""
        jax.config.update("jax_enable_x64", True)
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        params = model.initialize_params(X, y)
        state = model.initialize_state(X, y, params)
        # extract batch and add nans
        Xnan = X[:batch_size]
        Xnan[: batch_size // 2] = np.nan

        # run 3 iterations
        tot_iter = 3
        jit_update = deepcopy(params)
        jit_state = deepcopy(state)
        for _ in range(tot_iter):
            jit_update, jit_state = model.update(
                jit_update, jit_state, Xnan, y[:batch_size]
            )
        # make sure there is an update
        assert any(~jnp.allclose(p0, jit_update[k]) for k, p0 in enumerate(params))

        # update without jitting
        nojit_update = deepcopy(params)
        nojit_state = deepcopy(state)
        with jax.disable_jit(True):
            for _ in range(tot_iter):
                nojit_update, nojit_state = model.update(
                    nojit_update, nojit_state, Xnan, y[:batch_size]
                )
        # check for equivalence update
        assert all(jnp.allclose(p0, jit_update[k]) for k, p0 in enumerate(nojit_update))

    #######################
    # Test model.simulate #
    #######################
    @pytest.mark.parametrize(
        "input_type, expected_out_type",
        [
            (TsdFrame, Tsd),
            (np.ndarray, jnp.ndarray),
            (jnp.ndarray, jnp.ndarray),
        ],
    )
    def test_simulate_pynapple(
        self, input_type, expected_out_type, request, glm_type, model_instantiation
    ):
        """
        Test that the `simulate` method retturns the expected data type for different allowed inputs.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        if "population" in glm_type:
            model._initialize_feature_mask(X, y)
        if input_type == TsdFrame:
            X = TsdFrame(t=np.arange(X.shape[0]), d=X)
        count, rate = model.simulate(
            random_key=jax.random.key(123),
            feedforward_input=X,
        )
        if ("population" in glm_type) and (expected_out_type == Tsd):
            assert isinstance(count, TsdFrame)
            assert isinstance(rate, TsdFrame)
        else:
            assert isinstance(count, expected_out_type)
            assert isinstance(rate, expected_out_type)

    def test_simulate_feedforward_glm(self, request, glm_type, model_instantiation):
        """Test that simulate goes through"""
        X, y, model, params, rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        model.coef_ = params[0]
        model.intercept_ = params[1]
        model.scale_ = model.observation_model.scale
        if "population" in glm_type:
            model._initialize_feature_mask(X, y)
        ysim, ratesim = model.simulate(jax.random.key(123), X)
        # check that the expected dimensionality is returned
        assert ysim.ndim == 1 + (1 if "population" in glm_type else 0)
        assert ratesim.ndim == 1 + (1 if "population" in glm_type else 0)
        # check that the rates and spikes has the same shape
        assert ratesim.shape[0] == ysim.shape[0]
        # check the time point number is that expected (same as the input)
        assert ysim.shape[0] == X.shape[0]

    ########################################
    # Compare with standard implementation #
    ########################################
    @pytest.mark.solver_related
    def test_compatibility_with_sklearn_cv(
        self, request, glm_type, model_instantiation
    ):
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        param_grid = {"solver_name": ["BFGS", "GradientDescent"]}
        cls = GridSearchCV(model, param_grid).fit(X, y)
        # check that the repr works after cloning
        repr(cls)

    @pytest.mark.parametrize("regr_setup", ["", "_pytree"])
    @pytest.mark.parametrize("key", [jax.random.key(0), jax.random.key(19)])
    @pytest.mark.parametrize(
        "regularizer_class, solver_name",
        [
            (nmo.regularizer.UnRegularized, "SVRG"),
            (nmo.regularizer.Ridge, "SVRG"),
            (nmo.regularizer.Lasso, "ProxSVRG"),
            (nmo.regularizer.ElasticNet, "ProxSVRG"),
            # (nmo.regularizer.GroupLasso, "ProxSVRG"),
        ],
    )
    @pytest.mark.solver_related
    def test_glm_update_consistent_with_fit_with_svrg(
        self,
        request,
        glm_type,
        model_instantiation,
        regr_setup,
        key,
        regularizer_class,
        solver_name,
    ):
        """
        Make sure that calling GLM.update with the rest of the algorithm implemented outside in a naive loop
        is consistent with running the compiled GLM.fit on the same data with the same parameters
        """
        jax.config.update("jax_enable_x64", True)
        X, y, model, true_params, rate = request.getfixturevalue(
            glm_type + model_instantiation + regr_setup
        )

        N = y.shape[0]
        batch_size = 1
        maxiter = 3  # number of epochs
        tol = 1e-12
        stepsize = 1e-3

        # has to match how the number of iterations is calculated in SVRG
        m = int((N + batch_size - 1) // batch_size)

        regularizer_kwargs = {}
        if regularizer_class.__name__ == "GroupLasso":
            n_features = sum(x.shape[1] for x in jax.tree.leaves(X))
            regularizer_kwargs["mask"] = (
                (np.random.randn(n_features) > 0).reshape(1, -1).astype(float)
            )

        reg = regularizer_class(**regularizer_kwargs)
        strength = None if isinstance(reg, nmo.regularizer.UnRegularized) else 1.0
        glm = type(model)(
            regularizer=reg,
            regularizer_strength=strength,
            solver_name=solver_name,
            solver_kwargs={
                "batch_size": batch_size,
                "stepsize": stepsize,
                "tol": tol,
                "maxiter": maxiter,
                "key": key,
            },
        )
        glm2 = type(model)(
            regularizer=reg,
            solver_name=solver_name,
            solver_kwargs={
                "batch_size": batch_size,
                "stepsize": stepsize,
                "tol": tol,
                "maxiter": maxiter,
                "key": key,
            },
            regularizer_strength=strength,
        )
        glm2.fit(X, y)

        params = glm.initialize_params(X, y)
        state = glm.initialize_state(X, y, params)
        glm.instantiate_solver()

        # NOTE these two are not the same because for example Ridge augments the loss
        # loss_grad = jax.jit(jax.grad(glm._predict_and_compute_loss))
        loss_grad = jax.jit(jax.grad(glm._solver_loss_fun))

        # copied from GLM.fit
        # grab data if needed (tree map won't function because param is never a FeaturePytree).
        if isinstance(X, FeaturePytree):
            X = X.data

        iter_num = 0
        while iter_num < maxiter:
            state = state._replace(
                full_grad_at_reference_point=loss_grad(params, X, y),
            )

            prev_params = params
            for _ in range(m):
                key, subkey = jax.random.split(key)
                ind = jax.random.randint(subkey, (batch_size,), 0, N)
                xi, yi = tree_slice(X, ind), tree_slice(y, ind)
                params, state = glm.update(params, state, xi, yi)

            state = state._replace(
                reference_point=params,
            )

            iter_num += 1

            _error = tree_l2_norm(tree_sub(params, prev_params)) / tree_l2_norm(
                prev_params
            )
            if _error < tol:
                break

        assert iter_num == glm2.solver_state_.iter_num

        assert pytree_map_and_reduce(
            lambda a, b: np.allclose(a, b, atol=10**-5, rtol=0.0),
            all,
            (glm.coef_, glm.intercept_),
            (glm2.coef_, glm2.intercept_),
        )

    @pytest.mark.parametrize("solver_name", ["GradientDescent", "SVRG"])
    @pytest.mark.solver_related
    def test_glm_fit_matches_sklearn(
        self, solver_name, request, glm_type, model_instantiation, sklearn_model
    ):
        """Test that different solvers converge to the same solution."""
        if sklearn_model is None:
            pytest.skip(f"sklearn model is not available for {model_instantiation}")
        jax.config.update("jax_enable_x64", True)
        X, y, model_obs, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )

        model = type(model_obs)(
            regularizer=nmo.regularizer.UnRegularized(),
            observation_model=model_obs.observation_model,
            solver_name=solver_name,
            solver_kwargs={"tol": 10**-12},
        )

        # set gamma inverse link function to match sklearn
        if "gamma" in model_instantiation:
            model.inverse_link_function = jnp.exp

        # set precision to float64 for accurate matching of the results
        model.data_type = jnp.float64
        model.fit(X, y)

        if "population" in glm_type:
            # test by fitting each neuron separately in sklearn
            for n, yn in enumerate(y.T):
                sklearn_model.fit(X, yn)

                match_weights = jnp.allclose(
                    sklearn_model.coef_, model.coef_[:, n], atol=1e-5, rtol=0.0
                )
                # this will fail for poisson with GradientDescent for the third neuron
                # with tol=1e-5
                match_intercepts = jnp.allclose(
                    sklearn_model.intercept_,
                    model.intercept_[n],
                    atol=1.18e-5,
                    rtol=0.0,
                )
                if (not match_weights) or (not match_intercepts):
                    raise ValueError("GLM.fit estimate does not match sklearn!")

        else:
            sklearn_model.fit(X, y)

            match_weights = jnp.allclose(
                sklearn_model.coef_, model.coef_, atol=1e-5, rtol=0.0
            )
            match_intercepts = jnp.allclose(
                sklearn_model.intercept_, model.intercept_, atol=1e-5, rtol=0.0
            )
            if (not match_weights) or (not match_intercepts):
                raise ValueError("GLM.fit estimate does not match sklearn!")

    #####################
    # Test redidual DOF #
    #####################
    @pytest.mark.parametrize(
        "reg, dof, strength",
        [
            (nmo.regularizer.UnRegularized(), np.array([5, 5, 5]), None),
            (
                nmo.regularizer.Lasso(),
                "dof_lasso_dof",
                "dof_lasso_strength",
            ),  # this lasso fit has only 3 coeff of the first neuron
            # surviving
            (nmo.regularizer.Ridge(), np.array([5, 5, 5]), 1.0),
        ],
    )
    @pytest.mark.parametrize("n_samples", [1, 20])
    @pytest.mark.solver_related
    def test_estimate_dof_resid(
        self,
        n_samples,
        strength,
        dof,
        reg,
        request,
        glm_type,
        model_instantiation,
    ):
        """
        Test that the dof is an integer.
        """
        jax.config.update("jax_enable_x64", True)

        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        # different dof for different obs models with lasso
        if isinstance(dof, str):
            dof = request.getfixturevalue(dof)
        elif "population" in glm_type:
            # this should exclude lasso dof, where pop vs single neuron
            # is handled in the fixture
            dof = np.array([dof[0]])
        # need different strengths for different obs models with lasso reg
        # for 3 coefs to survive
        if isinstance(strength, str):
            strength = request.getfixturevalue(strength)
        model.set_params(regularizer=reg, regularizer_strength=strength)
        model.solver_name = model.regularizer.default_solver
        model.fit(X, y)
        num = model._estimate_resid_degrees_of_freedom(X, n_samples=n_samples)
        assert np.allclose(num, n_samples - dof - 1)

    ######################
    # Optimizer defaults #
    ######################
    @pytest.mark.parametrize("reg_setup", ["", "_pytree"])
    @pytest.mark.parametrize("batch_size", [None, 1, 10])
    @pytest.mark.parametrize("stepsize", [None, 0.01])
    @pytest.mark.parametrize(
        "regularizer", ["UnRegularized", "Ridge", "Lasso", "GroupLasso", "ElasticNet"]
    )
    @pytest.mark.parametrize(
        "solver_name, has_defaults",
        [
            ("GradientDescent", False),
            ("LBFGS", False),
            ("ProximalGradient", False),
            ("SVRG", True),
            ("ProxSVRG", True),
        ],
    )
    @pytest.mark.parametrize(
        "inv_link, link_has_defaults",
        [(jax.nn.softplus, True), (jax.numpy.exp, False), (jax.lax.logistic, False)],
    )
    @pytest.mark.solver_related
    def test_optimize_solver_params(
        self,
        batch_size,
        stepsize,
        regularizer,
        solver_name,
        inv_link,
        has_defaults,
        link_has_defaults,
        obs_has_defaults,
        request,
        glm_type,
        reg_setup,
        model_instantiation,
    ):
        """Test the behavior of `optimize_solver_params` for different solver, regularizer, and observation model configurations."""
        X, y, model, _, _ = request.getfixturevalue(
            glm_type + model_instantiation + reg_setup
        )

        obs = model.observation_model
        model.inverse_link_function = inv_link
        solver_kwargs = dict(stepsize=stepsize, batch_size=batch_size)
        # use glm static methods to check if the solver is batchable
        # if not pop the batch_size kwarg
        try:
            slv_class = nmo.solvers.solver_registry[solver_name]
            nmo.glm.GLM._check_solver_kwargs(slv_class, solver_kwargs)
        except NameError:
            solver_kwargs.pop("batch_size")

        # if the regularizer is not allowed for the solver type, return
        try:
            model = nmo.glm.GLM(
                regularizer=regularizer,
                solver_name=solver_name,
                inverse_link_function=inv_link,
                observation_model=obs,
                solver_kwargs=solver_kwargs,
                regularizer_strength=None if regularizer == "UnRegularized" else 1.0,
            )
        except ValueError as e:
            if not str(e).startswith(
                rf"The solver: {solver_name} is not allowed for {regularizer} regularization"
            ):
                raise e
            return

        kwargs = model._optimize_solver_params(X, y)
        if isinstance(batch_size, int) and "batch_size" in solver_kwargs:
            # if batch size was provided, then it should be returned unchanged
            assert batch_size == kwargs["batch_size"]
        elif has_defaults and link_has_defaults and obs_has_defaults:
            # if defaults are available, a batch size is computed
            assert isinstance(kwargs["batch_size"], int) and kwargs["batch_size"] > 0
        elif "batch_size" in solver_kwargs:
            # return None otherwise
            assert isinstance(kwargs["batch_size"], type(None))

        if isinstance(stepsize, float):
            # if stepsize was provided, then it should be returned unchanged
            assert stepsize == kwargs["stepsize"]
        elif has_defaults and link_has_defaults and obs_has_defaults:
            # if defaults are available, compute a value
            assert isinstance(kwargs["stepsize"], float) and kwargs["stepsize"] > 0
        else:
            # return None otherwise
            assert isinstance(kwargs["stepsize"], type(None))

    def test_repr_out(self, request, glm_type, model_instantiation, model_repr):
        model = request.getfixturevalue(glm_type + model_instantiation)[2]
        assert repr(model) == model_repr


class TestPopulationGLM:
    """
    Unit tests specific to the PopulationGLM class that are independent of the observation model.
    """

    #######################################
    # Compare with standard implementation
    #######################################

    def test_sklearn_clone(self, population_poissonGLM_model_instantiation):
        X, y, model, true_params, firing_rate = (
            population_poissonGLM_model_instantiation
        )
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        model._initialize_feature_mask(X, y)
        # hardcode metadata
        model._metadata = {"columns": 1, "metadata": 2}
        cloned = sklearn.clone(model)
        assert cloned.feature_mask is None, "cloned GLM shouldn't have feature mask!"
        assert model.feature_mask is not None, "fit GLM should have feature mask!"
        assert model._metadata == cloned._metadata

    @pytest.mark.parametrize(
        "mask, expectation",
        [
            (np.array([0, 1, 1] * 5).reshape(5, 3), does_not_raise()),
            (
                {"input_1": [0, 1, 0], "input_2": [1, 0, 1]},
                pytest.raises(
                    ValueError,
                    match="'feature_mask' of 'populationGLM' must be a 2-dimensional array",
                ),
            ),
            (
                {"input_1": np.array([0, 1, 0]), "input_2": np.array([1, 0, 1])},
                does_not_raise(),
            ),
            (
                {"input_1": np.array([0, 1, 0]), "input_2": np.array([1, 0, 1.1])},
                pytest.raises(
                    ValueError, match="'feature_mask' must contain only 0s and 1s"
                ),
            ),
            (
                np.array([0.1, 1, 1] * 5).reshape(5, 3),
                pytest.raises(
                    ValueError, match="'feature_mask' must contain only 0s and 1s"
                ),
            ),
        ],
    )
    def test_feature_mask_setter(
        self, mask, expectation, population_poissonGLM_model_instantiation
    ):
        _, _, model, _, _ = population_poissonGLM_model_instantiation
        with expectation:
            model.feature_mask = mask

    # @pytest.fixture
    # def feature_mask_compatibility_fit_expectation(self, reg_setup):
    """
    Fixture to return the expected exceptions for test_feature_mask_compatibility_fit
    based on the setup of the model inputs.
    """
    feature_mask_compatibility_fit_expectation = (
        "mask, expectation_np, expectation_pytree",
        [
            (
                np.array([0, 1, 1] * 5).reshape(5, 3),
                does_not_raise(),
                pytest.raises(
                    TypeError, match="feature_mask and X must have the same structure"
                ),
            ),
            (
                np.array([0, 1, 1] * 4).reshape(4, 3),
                pytest.raises(ValueError, match="Inconsistent number of features"),
                pytest.raises(
                    TypeError, match="feature_mask and X must have the same structure"
                ),
            ),
            (
                np.array([0, 1, 1, 1] * 5).reshape(5, 4),
                pytest.raises(ValueError, match="Inconsistent number of neurons"),
                pytest.raises(
                    TypeError, match="feature_mask and X must have the same structure"
                ),
            ),
            (
                {"input_1": np.array([0, 1, 0]), "input_2": np.array([1, 0, 1])},
                pytest.raises(
                    TypeError, match="feature_mask and X must have the same structure"
                ),
                does_not_raise(),
            ),
            (
                {"input_1": np.array([0, 1, 0, 1]), "input_2": np.array([1, 0, 1, 0])},
                pytest.raises(
                    TypeError, match="feature_mask and X must have the same structure"
                ),
                pytest.raises(ValueError, match="Inconsistent number of neurons"),
            ),
            (
                {"input_1": np.array([0, 1, 0])},
                pytest.raises(
                    TypeError, match="feature_mask and X must have the same structure"
                ),
                pytest.raises(
                    TypeError, match="feature_mask and X must have the same structure"
                ),
            ),
            (
                {"input_1": np.array([0, 1, 0, 1])},
                pytest.raises(
                    TypeError, match="feature_mask and X must have the same structure"
                ),
                pytest.raises(
                    TypeError, match="feature_mask and X must have the same structure"
                ),
            ),
        ],
    )

    @pytest.mark.parametrize(*feature_mask_compatibility_fit_expectation)
    @pytest.mark.parametrize("attr_name", ["fit", "predict", "score"])
    @pytest.mark.parametrize(
        "reg_setup",
        [
            "population_poissonGLM_model_instantiation",
            "population_poissonGLM_model_instantiation_pytree",
        ],
    )
    def test_feature_mask_compatibility_fit(
        self,
        mask,
        expectation_np,
        expectation_pytree,
        attr_name,
        request,
        reg_setup,
    ):
        X, y, model, true_params, firing_rate = request.getfixturevalue(reg_setup)
        if "pytree" in reg_setup:
            expectation = expectation_pytree
        else:
            expectation = expectation_np
        model.feature_mask = mask
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        with expectation:
            if attr_name == "predict":
                getattr(model, attr_name)(X)
            else:
                getattr(model, attr_name)(X, y)

    @pytest.mark.parametrize(
        "reg_setup",
        [
            "population_poissonGLM_model_instantiation",
            "population_poissonGLM_model_instantiation_pytree",
        ],
    )
    @pytest.mark.solver_related
    def test_metadata_pynapple_fit(self, reg_setup, request):
        X, y, model, true_params, firing_rate = request.getfixturevalue(reg_setup)
        y = TsdFrame(
            t=np.arange(y.shape[0]), d=y, metadata={"y": np.arange(y.shape[1])}
        )
        model.fit(X, y)
        assert hasattr(model, "_metadata") and (model._metadata is not None)
        assert np.all(y._metadata == model._metadata["metadata"])
        assert np.all(y.columns == model._metadata["columns"])

    @pytest.mark.parametrize(
        "reg_setup",
        [
            "population_poissonGLM_model_instantiation",
            "population_poissonGLM_model_instantiation_pytree",
        ],
    )
    @pytest.mark.solver_related
    def test_metadata_pynapple_is_deepcopied(self, reg_setup, request):
        X, y, model, true_params, firing_rate = request.getfixturevalue(reg_setup)
        y = TsdFrame(
            t=np.arange(y.shape[0]), d=y, metadata={"y": np.arange(y.shape[1])}
        )
        model.fit(X, y)
        X = jax.tree_util.tree_map(lambda x: convert_to_nap(x, y.t), X)
        rate = model.predict(X)
        # modify metadata of y
        y["newdata"] = np.arange(1, 1 + y.shape[1])
        if "newdata" in rate._metadata:
            raise RuntimeError("Metadata was shallow copied by pynapple init")

    @pytest.mark.parametrize(
        "reg_setup",
        [
            "population_poissonGLM_model_instantiation",
            "population_poissonGLM_model_instantiation_pytree",
        ],
    )
    @pytest.mark.solver_related
    def test_metadata_pynapple_predict(self, reg_setup, request):
        X, y, model, true_params, firing_rate = request.getfixturevalue(reg_setup)
        y = TsdFrame(
            t=np.arange(y.shape[0]),
            d=y,
            metadata={"y": np.arange(y.shape[1])},
            columns=range(1, 1 + y.shape[1]),
        )

        X = jax.tree_util.tree_map(lambda x: convert_to_nap(x, y.t), X)
        model.fit(X, y)
        rate = model.predict(X)
        assert hasattr(rate, "metadata") and (rate.metadata is not None)
        assert np.all(y._metadata == rate._metadata)
        assert np.all(y.columns == rate.columns)


@pytest.mark.parametrize(
    "model_instantiation",
    [
        "population_poissonGLM_model_instantiation",
        "population_gammaGLM_model_instantiation",
        "population_bernoulliGLM_model_instantiation",
        "population_negativeBinomialGLM_model_instantiation",
    ],
)
class TestPopulationGLMObservationModel:
    """
    Unit tests specific to the PopulationGLM class that are dependent on the observation model.
    """

    #######################
    # Test model.score
    #######################

    @pytest.mark.parametrize(
        "score_type", ["log-likelihood", "pseudo-r2-McFadden", "pseudo-r2-Cohen"]
    )
    def test_score_aggregation_ndim(self, score_type, request, model_instantiation):
        """
        Test that the aggregate samples returns the right dimensional object.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation
        )
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        mn = model.score(X, y, score_type=score_type, aggregate_sample_scores=jnp.mean)
        mn_n = model.score(
            X,
            y,
            score_type=score_type,
            aggregate_sample_scores=lambda x: jnp.mean(x, axis=0),
        )
        assert mn.ndim == 0
        assert mn_n.ndim == 1

    @pytest.mark.parametrize(
        "regularizer, regularizer_strength, solver_name, solver_kwargs",
        [
            (
                nmo.regularizer.UnRegularized(),
                None,
                "LBFGS",
                {"stepsize": 0.1, "tol": 10**-14},
            ),
            (
                nmo.regularizer.UnRegularized(),
                None,
                "GradientDescent",
                {"tol": 10**-14},
            ),
            (
                nmo.regularizer.Ridge(),
                1.0,
                "LBFGS",
                {"tol": 10**-14},
            ),
            (
                nmo.regularizer.Ridge(),
                1.0,
                "LBFGS",
                {"stepsize": 0.1, "tol": 10**-14},
            ),
            (
                nmo.regularizer.Lasso(),
                0.001,
                "ProximalGradient",
                {"tol": 10**-14},
            ),
            (
                nmo.regularizer.Lasso(),
                0.1,
                "ProximalGradient",
                {"tol": 10**-14},
            ),
            (
                nmo.regularizer.ElasticNet(),
                (1.0, 0.5),
                "ProximalGradient",
                {"tol": 10**-14},
            ),
        ],
    )
    @pytest.mark.parametrize(
        "mask",
        [
            np.array(
                [
                    [0, 0, 1],
                    [0, 1, 0],
                    [1, 1, 1],
                    [1, 0, 1],
                    [0, 1, 0],
                ]
            ),
            {"input_1": np.array([0, 1, 0]), "input_2": np.array([1, 0, 1])},
        ],
    )
    @pytest.mark.solver_related
    def test_masked_fit_vs_loop(
        self,
        regularizer,
        regularizer_strength,
        solver_name,
        solver_kwargs,
        mask,
        request,
        model_instantiation,
    ):
        jax.config.update("jax_enable_x64", True)
        if isinstance(mask, dict):
            X, y, _, true_params, firing_rate = request.getfixturevalue(
                model_instantiation + "_pytree"
            )

            def map_neu(k, coef_):
                key_ind = {"input_1": [0, 1, 2], "input_2": [3, 4]}
                ind_array = np.zeros((0,), dtype=int)
                coef_stack = np.zeros((0,), dtype=int)
                for key, msk in mask.items():
                    if msk[k]:
                        ind_array = np.hstack((ind_array, key_ind[key]))
                        coef_stack = np.hstack((coef_stack, coef_[key]))
                return ind_array, coef_stack

        else:
            X, y, _, true_params, firing_rate = request.getfixturevalue(
                model_instantiation
            )

            def map_neu(k, coef_):
                ind_array = np.where(mask[:, k])[0]
                coef_stack = coef_
                return ind_array, coef_stack

        mask_bool = jax.tree_util.tree_map(lambda x: np.asarray(x.T, dtype=bool), mask)
        # fit pop glm
        kwargs = dict(
            feature_mask=mask,
            regularizer=regularizer,
            regularizer_strength=regularizer_strength,
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
        )
        model = nmo.glm.PopulationGLM(**kwargs)
        model.fit(X, y)
        coef_vectorized = np.vstack(jax.tree_util.tree_leaves(model.coef_))

        coef_loop = np.zeros((5, 3))
        intercept_loop = np.zeros((3,))
        # loop over neuron
        for k in range(y.shape[1]):
            model_single_neu = nmo.glm.GLM(
                regularizer=regularizer,
                regularizer_strength=regularizer_strength,
                solver_name=solver_name,
                solver_kwargs=solver_kwargs,
            )
            if isinstance(mask_bool, dict):
                X_neu = {}
                for key, xx in X.items():
                    if mask_bool[key][k]:
                        X_neu[key] = X[key]
                X_neu = FeaturePytree(**X_neu)
            else:
                X_neu = X[:, mask_bool[k]]

            model_single_neu.fit(X_neu, y[:, k])
            idx, coef = map_neu(k, model_single_neu.coef_)
            coef_loop[idx, k] = coef
            intercept_loop[k] = np.array(model_single_neu.intercept_)[0]
        print(f"\nMAX ERR: {np.abs(coef_loop - coef_vectorized).max()}")
        assert np.allclose(coef_loop, coef_vectorized, atol=10**-5, rtol=0)


@pytest.mark.parametrize("glm_class_type", ["glm_class", "population_glm_class"])
class TestPoissonGLM:
    """
    Unit tests specific to Poisson GLM.
    """

    @pytest.mark.parametrize(
        "inv_link", [jnp.exp, lambda x: jnp.exp(x), jax.nn.softplus, jax.nn.relu]
    )
    def test_high_firing_rate_initialization(
        self, inv_link, example_X_y_high_firing_rates, request, glm_class_type
    ):
        glm_class = request.getfixturevalue(glm_class_type)
        model = glm_class(
            observation_model=nmo.observation_models.PoissonObservations(),
            inverse_link_function=inv_link,
        )
        X, y = example_X_y_high_firing_rates
        if "population" in glm_class_type:
            model.initialize_params(X, y)
        else:
            model.initialize_params(X, y[:, 0])

    @pytest.mark.parametrize("reg_setup", ["", "_pytree"])
    @pytest.mark.parametrize(
        "solver_name, reg",
        [
            ("SVRG", "Ridge"),
            ("SVRG", "UnRegularized"),
            ("ProxSVRG", "Ridge"),
            ("ProxSVRG", "UnRegularized"),
            ("ProxSVRG", "Lasso"),
            ("ProxSVRG", "ElasticNet"),
            ("ProxSVRG", "GroupLasso"),
        ],
    )
    @pytest.mark.parametrize(
        "obs",
        [nmo.observation_models.PoissonObservations()],
    )
    @pytest.mark.parametrize("batch_size", [None, 1, 10])
    @pytest.mark.parametrize("stepsize", [None, 0.01])
    @pytest.mark.solver_related
    def test_glm_optimal_config_set_initial_state(
        self,
        solver_name,
        batch_size,
        stepsize,
        reg,
        obs,
        request,
        glm_class_type,
        model_instantiation_type,
        reg_setup,
    ):
        """
        Test special initialization of Poisson GLM + softmax inverse link function for SVRG and ProxSVRG.
        """
        glm_class = request.getfixturevalue(glm_class_type)
        X, y, _, true_params, _ = request.getfixturevalue(
            model_instantiation_type + reg_setup
        )
        if reg == "GroupLasso":
            if reg_setup == "_pytree":
                # this was not tested for pytree when the test was separate
                return
            else:
                reg = nmo.regularizer.GroupLasso(mask=jnp.ones((1, X.shape[1])))
        model = glm_class(
            solver_name=solver_name,
            inverse_link_function=jax.nn.softplus,
            solver_kwargs=dict(batch_size=batch_size, stepsize=stepsize),
            observation_model=obs,
            regularizer=reg,
            regularizer_strength=None if reg == "UnRegularized" else 1.0,
        )
        opt_state = model.initialize_state(X, y, true_params)
        solver = model._solver

        if stepsize is not None:
            assert opt_state.stepsize == stepsize
            assert solver.stepsize == stepsize
        else:
            assert opt_state.stepsize > 0
            assert isinstance(opt_state.stepsize, float)

        if batch_size is not None:
            assert solver.batch_size == batch_size
        else:
            assert isinstance(solver.batch_size, int)
            assert solver.batch_size > 0

    @pytest.mark.parametrize(
        "regularizer, expected_type_convexity",
        [
            ("UnRegularized", type(None)),
            ("Lasso", type(None)),
            ("ElasticNet", type(None)),
            ("GroupLasso", type(None)),
            ("Ridge", float),
        ],
    )
    @pytest.mark.parametrize(
        "solver_name, expected_type_solver",
        [
            ("GradientDescent", type(None)),
            ("ProximalGradient", type(None)),
            ("LBFGS", type(None)),
            ("SVRG", Callable),
            ("ProxSVRG", Callable),
        ],
    )
    @pytest.mark.parametrize(
        "inv_link_func, expected_type_link",
        [(jax.nn.softplus, Callable), (jax.numpy.exp, type(None))],
    )
    def test_optimal_config_outputs(
        self,
        regularizer,
        solver_name,
        inv_link_func,
        expected_type_convexity,
        expected_type_link,
        expected_type_solver,
        request,
        glm_class_type,
    ):
        """Test that 'required_params' is a dictionary."""
        glm_class = request.getfixturevalue(glm_class_type)
        obs = nmo.observation_models.PoissonObservations()

        # if the regularizer is not allowed for the solver type, return
        try:
            model = glm_class(
                inverse_link_function=inv_link_func,
                regularizer=regularizer,
                solver_name=solver_name,
                observation_model=obs,
                regularizer_strength=None if regularizer == "UnRegularized" else 1.0,
            )
        except ValueError as e:
            if not str(e).startswith(
                rf"The solver: {solver_name} is not allowed for {regularizer} regularization"
            ):
                raise e
            return

        # if there is no callable for the model specs, then convexity should be None
        func1, func2, convexity = (
            nmo.solvers._compute_defaults.glm_compute_optimal_stepsize_configs(model)
        )
        assert isinstance(func1, expected_type_solver)
        assert isinstance(func2, expected_type_link)
        assert isinstance(
            convexity, expected_type_convexity
        ), f"convexity type: {type(convexity)}, expected type: {expected_type_convexity}"


@pytest.mark.parametrize("inv_link", [jnp.exp, lambda x: 1 / x])
@pytest.mark.parametrize("glm_type", ["", "population_"])
@pytest.mark.parametrize("model_instantiation", ["gammaGLM_model_instantiation"])
class TestGammaGLM:
    """
    Unit tests specific to Gamma GLM.
    """

    @pytest.mark.solver_related
    def test_fit_glm(self, inv_link, request, glm_type, model_instantiation):
        """
        Ensure that the model can be fit with different link functions.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        model.observation_model.inverse_link_function = inv_link
        model.fit(X, y)
        if "population" in glm_type:
            assert np.all(model.scale_ != 1)
        else:
            assert model.scale_ != 1

    def test_score_glm(self, inv_link, request, glm_type, model_instantiation):
        """
        Ensure that the model can be scored with different link functions.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        model.observation_model.inverse_link_function = inv_link
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        model.score(X, y)

    def test_simulate_glm(self, inv_link, request, glm_type, model_instantiation):
        """
        Ensure that data can be simulated with different link functions.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        model.observation_model.inverse_link_function = inv_link
        if "population" in glm_type:
            model.feature_mask = jnp.ones((X.shape[1], y.shape[1]))
            model.scale_ = jnp.ones((y.shape[1]))
        else:
            model.scale_ = 1.0
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        ysim, ratesim = model.simulate(jax.random.PRNGKey(123), X)
        assert ysim.shape == y.shape
        assert ratesim.shape == y.shape


@pytest.mark.parametrize("inv_link", [jax.lax.logistic, jax.scipy.stats.norm.cdf])
@pytest.mark.parametrize("glm_type", ["", "population_"])
@pytest.mark.parametrize("model_instantiation", ["bernoulliGLM_model_instantiation"])
class TestBernoulliGLM:
    """
    Unit tests specific to Bernoulli GLM.
    """

    @pytest.mark.solver_related
    def test_fit_glm(self, inv_link, request, glm_type, model_instantiation):
        """
        Ensure that the model can be fit with different link functions.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        model.inverse_link_function = inv_link
        model.fit(X, y)

    def test_score_glm(self, inv_link, request, glm_type, model_instantiation):
        """
        Ensure that the model can be scored with different link functions.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        model.inverse_link_function = inv_link
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        if "population" in glm_type:
            model.scale_ = np.ones((y.shape[1]))
        else:
            model.scale_ = 1.0
        model.score(X, y)

    def test_simulate_glm(self, inv_link, request, glm_type, model_instantiation):
        """
        Ensure that data can be simulated with different link functions.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        model.inverse_link_function = inv_link
        if "population" in glm_type:
            model.feature_mask = jnp.ones((X.shape[1], y.shape[1]))
            model.scale_ = jnp.ones((y.shape[1]))
        else:
            model.scale_ = 1.0
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        ysim, ratesim = model.simulate(jax.random.PRNGKey(123), X)
        assert ysim.shape == y.shape
        assert ratesim.shape == y.shape


@pytest.mark.parametrize("inv_link", [jax.nn.softplus, jax.numpy.exp])
@pytest.mark.parametrize("glm_type", ["", "population_"])
@pytest.mark.parametrize(
    "model_instantiation", ["negativeBinomialGLM_model_instantiation"]
)
class TestNegativeBinomialGLM:
    """
    Unit tests specific to Negative Binomial GLM.
    """

    @pytest.mark.solver_related
    def test_fit_glm(self, inv_link, request, glm_type, model_instantiation):
        """
        Ensure that the model can be fit with different link functions.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        # intialize to true params
        model.inverse_link_function = inv_link
        model.fit(X, y, init_params=true_params)

    def test_score_glm(self, inv_link, request, glm_type, model_instantiation):
        """
        Ensure that the model can be scored with different link functions.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        model.inverse_link_function = inv_link
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        if "population" in glm_type:
            model.scale_ = np.ones((y.shape[1]))
        else:
            model.scale_ = 1.0
        model.score(X, y)

    def test_simulate_glm(self, inv_link, request, glm_type, model_instantiation):
        """
        Ensure that data can be simulated with different link functions.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        model.inverse_link_function = inv_link
        if "population" in glm_type:
            model.feature_mask = jnp.ones((X.shape[1], y.shape[1]))
            model.scale_ = jnp.ones((y.shape[1]))
        else:
            model.scale_ = 1.0
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        ysim, ratesim = model.simulate(jax.random.PRNGKey(123), X)
        assert ysim.shape == y.shape
        assert ratesim.shape == y.shape
