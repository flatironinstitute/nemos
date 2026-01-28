import inspect
from contextlib import nullcontext as does_not_raise
from copy import deepcopy
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats as sts
import sklearn
import statsmodels.api as sm
from conftest import initialize_feature_mask_for_population_glm
from pynapple import Tsd, TsdFrame
from sklearn.linear_model import (
    GammaRegressor,
    LinearRegression,
    LogisticRegression,
    PoissonRegressor,
)
from sklearn.model_selection import GridSearchCV

import nemos as nmo
from nemos import solvers
from nemos._observation_model_builder import instantiate_observation_model
from nemos._regularizer_builder import instantiate_regularizer
from nemos.inverse_link_function_utils import identity, log_softmax
from nemos.pytrees import FeaturePytree
from nemos.tree_utils import (
    pytree_map_and_reduce,
    tree_l2_norm,
    tree_slice,
    tree_sub,
)
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

# Comprehensive model configuration - single source of truth for all model types
# Each model class has its configuration for dimensions, fixtures, and properties
MODEL_CONFIG = {
    "GLM": {
        "coef_dim": 1,
        "intercept_dim": 1,
        "is_population": False,
        "is_classifier": False,
        "class_fixture": "glm_class",
        "instantiation_fixture": "poissonGLM_model_instantiation",
        "glm_type_prefix": "",  # Used in glm_type parametrization (e.g., "" + "poissonGLM_model_instantiation")
    },
    "PopulationGLM": {
        "coef_dim": 2,
        "intercept_dim": 1,
        "is_population": True,
        "is_classifier": False,
        "class_fixture": "population_glm_class",
        "instantiation_fixture": "population_poissonGLM_model_instantiation",
        "glm_type_prefix": "population_",
    },
    "ClassifierGLM": {
        "coef_dim": 2,
        "intercept_dim": 1,
        "is_population": False,
        "is_classifier": True,
        "class_fixture": "classifier_glm_class",
        "instantiation_fixture": "classifierGLM_model_instantiation",
        "glm_type_prefix": "classifier_",
    },
    "ClassifierPopulationGLM": {
        "coef_dim": 3,
        "intercept_dim": 2,
        "is_population": True,
        "is_classifier": True,
        "class_fixture": "classifier_population_glm_class",
        "instantiation_fixture": "population_classifierGLM_model_instantiation",
        "glm_type_prefix": "classifier_population_",
    },
}

# Derived lookups from MODEL_CONFIG for backwards compatibility and convenience
DIMENSIONALITY_PARAMS = {
    name: {"coef": cfg["coef_dim"], "intercept": cfg["intercept_dim"]}
    for name, cfg in MODEL_CONFIG.items()
}

POPULATION_MODEL_NAMES = {
    name for name, cfg in MODEL_CONFIG.items() if cfg["is_population"]
}

CLASSIFIER_MODEL_NAMES = {
    name for name, cfg in MODEL_CONFIG.items() if cfg["is_classifier"]
}

# Build mappings from various parametrization styles to class names
# Supports: glm_class_type (e.g., "glm_class"), glm_type prefix (e.g., "population_")
GLM_CLASS_TYPE_TO_NAME = {}
GLM_TYPE_PREFIX_TO_NAME = {}
MODEL_INSTANTIATION_FIXTURES = {}

for name, cfg in MODEL_CONFIG.items():
    # Map class fixture name to class name
    GLM_CLASS_TYPE_TO_NAME[cfg["class_fixture"]] = name
    # Map glm_type prefix to class name
    GLM_TYPE_PREFIX_TO_NAME[cfg["glm_type_prefix"]] = name
    # Map class fixture to instantiation fixture
    MODEL_INSTANTIATION_FIXTURES[cfg["class_fixture"]] = cfg["instantiation_fixture"]


def is_population_model(model) -> bool:
    """Check if a model is a population model based on its class name."""
    return model.__class__.__name__ in POPULATION_MODEL_NAMES


def is_classifier_model(model) -> bool:
    """Check if a model is a classifier model based on its class name."""
    return model.__class__.__name__ in CLASSIFIER_MODEL_NAMES


def is_population_glm_type(glm_type: str) -> bool:
    """Check if a glm_type parameter string indicates a population model."""
    class_name = GLM_TYPE_PREFIX_TO_NAME.get(glm_type)
    if class_name:
        return MODEL_CONFIG[class_name]["is_population"]
    return False


def get_model_config(model) -> dict:
    """Get the configuration for a model instance."""
    return MODEL_CONFIG[model.__class__.__name__]


def convert_to_nap(arr, t):
    return TsdFrame(t=t, d=getattr(arr, "d", arr))


def _create_grouplasso_mask(X, y, model):
    params = model._validator.to_model_params(model.initialize_params(X, y))

    def set_mask(par):
        msk = jax.tree_util.tree_map(
            lambda x: jnp.zeros((2, *x.shape), dtype=float), par
        )
        msk = jax.tree_util.tree_map(lambda x: x.at[0, ::2].set(1), msk)
        msk = jax.tree_util.tree_map(lambda x: x.at[1, 1::2].set(1), msk)
        return msk

    struct = jax.tree_util.tree_structure(params)
    mask = jax.tree_util.tree_unflatten(struct, [None] * struct.num_leaves)
    for where in params.regularizable_subtrees():
        mask = eqx.tree_at(
            where, mask, replace=set_mask(where(params)), is_leaf=lambda x: x is None
        )
    return mask


@pytest.fixture
def model_instantiation_type(glm_class_type):
    """
    Fixture to grab the appropriate model instantiation function based on the type of GLM class.
    Used by TestGLM and TestPoissonGLM classes.
    """
    return MODEL_INSTANTIATION_FIXTURES[glm_class_type]


@pytest.mark.parametrize("glm_class_type", ["glm_class", "population_glm_class"])
@pytest.mark.solver_related
@pytest.mark.filterwarnings("ignore:The fit did not converge:RuntimeWarning")
def test_get_fit_attrs(request, glm_class_type, model_instantiation_type):
    X, y, model = request.getfixturevalue(model_instantiation_type)[:3]
    expected_state = {
        "coef_": None,
        "intercept_": None,
        "scale_": None,
        "solver_state_": None,
        "dof_resid_": None,
        "aux_": None,
        "optim_info_": None,
    }
    assert model._get_fit_state() == expected_state
    model.solver_kwargs = {"maxiter": 1}
    model.fit(X, y)
    assert not model._has_aux
    assert all(
        val is not None for key, val in model._get_fit_state().items() if key != "aux_"
    )
    assert model._get_fit_state().keys() == expected_state.keys()


def get_param_shape(model, X, y):
    empty_par = model._validator.get_empty_params(X, y)
    return jax.tree_util.tree_map(lambda x: x.shape, empty_par)


@pytest.mark.parametrize(
    "glm_class_type",
    [
        "glm_class",
        "population_glm_class",
        "classifier_glm_class",
        "classifier_population_glm_class",
    ],
)
class TestGLM:
    """
    Unit tests for the GLM class that do not depend on the observation model.
    i.e. tests that do not call observation model methods, or tests that do not check the output when
    observation model methods are called (e.g. error testing for input validation)
    """

    @staticmethod
    def fit_weights_dimensionality_expectation(
        model, expected_dim: int, param_name: str
    ):
        """
        Fixture to define the expected behavior for test_fit_weights_dimensionality based on the type of GLM class.
        """
        correct_dim = DIMENSIONALITY_PARAMS[model.__class__.__name__][param_name]

        if expected_dim == correct_dim:
            return does_not_raise()
        else:
            if param_name == "intercept":
                message = "Intercept term should be a|Invalid parameter dimensionality"
            else:
                message = r"coef must be an array or .* of shape \(n_features|Inconsistent number of features"
            return pytest.raises(ValueError, match=message)

    @pytest.mark.parametrize("dim_weights", [0, 1, 2, 3])
    @pytest.mark.solver_related
    def test_fit_weights_dimensionality(
        self,
        dim_weights,
        request,
        glm_class_type,
        model_instantiation_type,
    ):
        """
        Test the `fit` method with weight matrices of different dimensionalities.
        Check for correct dimensionality.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        expectation = self.fit_weights_dimensionality_expectation(
            model, dim_weights, "coef"
        )
        par_shape = get_param_shape(model, X, y)
        if dim_weights == 0:
            init_w = jnp.array([])
        elif dim_weights <= len(par_shape.coef):
            slc = (slice(None),) * dim_weights + (0,) * (
                len(par_shape.coef) - dim_weights
            )
            init_w = np.zeros(par_shape.coef)[slc]
        else:
            delta = dim_weights - len(par_shape.coef)
            init_w = jnp.zeros((*par_shape.coef, *(1,) * delta))

        with expectation:
            model.fit(X, y, init_params=(init_w, true_params.intercept))

    @pytest.mark.parametrize("dim_intercepts", [0, 1, 2, 3, 4])
    @pytest.mark.solver_related
    def test_fit_intercepts_dimensionality(
        self,
        dim_intercepts,
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
        expectation = self.fit_weights_dimensionality_expectation(
            model, dim_intercepts, "intercept"
        )

        par_shape = get_param_shape(model, X, y)
        init_w = jnp.zeros(par_shape.coef)
        if dim_intercepts == 0:
            init_b = jnp.array([])
        elif dim_intercepts <= len(par_shape.intercept):
            slc = (slice(None),) * dim_intercepts + (0,) * (
                len(par_shape.intercept) - dim_intercepts
            )
            init_b = np.zeros(par_shape.intercept)[slc]
        else:
            delta = dim_intercepts - len(par_shape.intercept)
            init_b = jnp.zeros((*par_shape.intercept, *(1,) * delta))

        with expectation:
            model.fit(X, y, init_params=(init_w, init_b))

    """
    Parameterization used by test_fit_init_params_type and test_initialize_solver_init_params_type.
    Uses a dict keyed by model class name for model-specific params, or a single value for
    model-independent test cases. Easy to extend by adding new model keys to the dicts.
    """
    fit_init_params_type_init_params = (
        "expectation, init_params_by_model",
        [
            (
                does_not_raise(),
                {
                    "GLM": [jnp.zeros((5,)), jnp.zeros((1,))],
                    "PopulationGLM": [jnp.zeros((5, 3)), jnp.zeros((3,))],
                    "ClassifierGLM": [jnp.zeros((5, 3)), jnp.zeros((3,))],
                    "ClassifierPopulationGLM": [
                        jnp.zeros((5, 3, 3)),
                        jnp.zeros((3, 3)),
                    ],
                },
            ),
            (
                pytest.raises(ValueError, match="Params must have length two."),
                {
                    "GLM": [[jnp.zeros((1, 5)), jnp.zeros((1,))]],
                    "PopulationGLM": [[jnp.zeros((1, 5)), jnp.zeros((3,))]],
                    "ClassifierGLM": [[jnp.zeros((1, 5)), jnp.zeros((3,))]],
                    "ClassifierPopulationGLM": [[jnp.zeros((1, 5)), jnp.zeros((3, 3))]],
                },
            ),
            (
                pytest.raises(
                    TypeError, match="GLM params must be a tuple/list of length two"
                ),
                {
                    "GLM": dict(p1=jnp.zeros((5,)), p2=jnp.zeros((1,))),
                    "PopulationGLM": dict(p1=jnp.zeros((3, 3)), p2=jnp.zeros((3, 2))),
                    "ClassifierGLM": dict(p1=jnp.zeros((5, 3)), p2=jnp.zeros((3,))),
                    "ClassifierPopulationGLM": dict(
                        p1=jnp.zeros((5, 3, 3)), p2=jnp.zeros((3, 3))
                    ),
                },
            ),
            (
                pytest.raises(TypeError, match="X and coef have mismatched structure"),
                {
                    "GLM": [
                        dict(p1=jnp.zeros((5,)), p2=jnp.zeros((1,))),
                        jnp.zeros((1,)),
                    ],
                    "PopulationGLM": [
                        dict(p1=jnp.zeros((3, 3)), p2=jnp.zeros((2, 3))),
                        jnp.zeros((3,)),
                    ],
                    "ClassifierGLM": [
                        dict(p1=jnp.zeros((5, 3)), p2=jnp.zeros((1, 3))),
                        jnp.zeros((3,)),
                    ],
                    "ClassifierPopulationGLM": [
                        dict(p1=jnp.zeros((5, 3, 3)), p2=jnp.zeros((1, 3, 3))),
                        jnp.zeros((3, 3)),
                    ],
                },
            ),
            (
                pytest.raises(TypeError, match="X and coef have mismatched structure"),
                {
                    "GLM": [
                        FeaturePytree(p1=jnp.zeros((5,)), p2=jnp.zeros((5,))),
                        jnp.zeros((1,)),
                    ],
                    "PopulationGLM": [
                        FeaturePytree(p1=jnp.zeros((3, 3)), p2=jnp.zeros((3, 2))),
                        jnp.zeros((3,)),
                    ],
                    "ClassifierGLM": [
                        FeaturePytree(p1=jnp.zeros((5, 3)), p2=jnp.zeros((5, 3))),
                        jnp.zeros((3,)),
                    ],
                    "ClassifierPopulationGLM": [
                        FeaturePytree(p1=jnp.zeros((5, 3, 3)), p2=jnp.zeros((5, 3, 3))),
                        jnp.zeros((3, 3)),
                    ],
                },
            ),
            # Model-independent invalid params - single value used for all models
            (pytest.raises(ValueError, match="Params must have length two."), 0),
            (
                pytest.raises(
                    TypeError, match="GLM params must be a tuple/list of length two"
                ),
                {0, 1},
            ),
            (
                pytest.raises(
                    TypeError, match="Failed to convert parameters to JAX arrays"
                ),
                [jnp.zeros((1, 5)), ""],
            ),
            (
                pytest.raises(
                    TypeError, match="Failed to convert parameters to JAX arrays"
                ),
                ["", jnp.zeros((1,))],
            ),
        ],
    )

    @staticmethod
    def get_init_params_for_model(init_params_by_model, model):
        """Get init_params for a specific model from dict or return as-is if model-independent.

        If init_params_by_model is a dict keyed by model class names and the model's class
        is not present, raises KeyError. This ensures test data is explicitly added for new models.
        """
        model_name = model.__class__.__name__
        if isinstance(init_params_by_model, dict):
            return init_params_by_model[model_name]
        return init_params_by_model

    @pytest.mark.parametrize(*fit_init_params_type_init_params)
    @pytest.mark.solver_related
    def test_fit_init_params_type(
        self,
        request,
        glm_class_type,
        model_instantiation_type,
        expectation,
        init_params_by_model,
    ):
        """
        Test the `fit` method with various types of initial parameters. Ensure that the provided initial parameters
        are array-like.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        init_params = self.get_init_params_for_model(init_params_by_model, model)
        with expectation:
            model.fit(X, y, init_params=init_params)

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
        par_shape = get_param_shape(model, X, y)
        # Create init_w with wrong number of features (first dim)
        wrong_coef_shape = (par_shape.coef[0] + delta_n_features,) + par_shape.coef[1:]
        init_w = jnp.zeros(wrong_coef_shape)
        init_b = jnp.zeros(par_shape.intercept)
        with expectation:
            model.fit(X, y, init_params=(init_w, init_b))

    #######################
    # Test model.score
    #######################
    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (-1, pytest.raises(ValueError, match="X must be 2-dimensional")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="X must be 2-dimensional")),
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
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
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
                pytest.raises(ValueError, match=r"y must be [12]-dimensional."),
            ),
            (0, does_not_raise()),
            (
                1,
                pytest.raises(ValueError, match=r"y must be [12]-dimensional."),
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
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
        if is_population_model(model):
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
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
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
                pytest.raises(
                    ValueError, match="X and y must have the same number of samples"
                ),
            ),
            (0, does_not_raise()),
            (
                1,
                pytest.raises(
                    ValueError, match="X and y must have the same number of samples"
                ),
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
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
        X = jnp.zeros((X.shape[0] + delta_tp,) + X.shape[1:])
        with expectation:
            model.score(X, y)

    @pytest.mark.parametrize(
        "delta_tp, expectation",
        [
            (
                -1,
                pytest.raises(
                    ValueError, match="X and y must have the same number of samples"
                ),
            ),
            (0, does_not_raise()),
            (
                1,
                pytest.raises(
                    ValueError, match="X and y must have the same number of samples"
                ),
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
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
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
            (-1, pytest.raises(ValueError, match="X must be 2-dimensional.")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="X must be 2-dimensional.")),
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
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
        if is_population_model(model):
            model._feature_mask = initialize_feature_mask_for_population_glm(
                X, y.shape[1], coef=true_params.coef
            )
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
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
        if is_population_model(model):
            model._feature_mask = initialize_feature_mask_for_population_glm(
                X, y.shape[1], coef=true_params.coef
            )
        if delta_n_features == 1:
            X = jnp.concatenate((X, jnp.zeros((X.shape[0], 1))), axis=1)
        elif delta_n_features == -1:
            X = X[..., :-1]
        with expectation:
            model.predict(X)

    @pytest.fixture
    def initialize_solver_weights_dimensionality_expectation(self, glm_class_type):
        class_name = GLM_CLASS_TYPE_TO_NAME[glm_class_type]
        expected_coef_dim = DIMENSIONALITY_PARAMS[class_name]["coef"]
        # Build expectation dict: correct dim passes, others fail
        expectations = {}
        for dim in range(4):
            if dim == expected_coef_dim:
                expectations[dim] = does_not_raise()
            else:
                # Use pattern that matches multiple possible error messages
                expectations[dim] = pytest.raises(
                    ValueError,
                    match=r"Inconsistent number of features|Invalid parameter dimensionality|coef must be an array",
                )
        return expectations

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
        par_shape = get_param_shape(model, X, y)
        n_samples, n_features = X.shape
        # Build init_w of the requested dimensionality
        if dim_weights == 0:
            init_w = jnp.array([])
        elif dim_weights <= len(par_shape.coef):
            # Use correct shape up to dim_weights, then slice off extra dims
            slc = (slice(None),) * dim_weights + (0,) * (
                len(par_shape.coef) - dim_weights
            )
            init_w = jnp.zeros(par_shape.coef)[slc]
        else:
            # Add extra dimensions beyond what's expected
            init_w = jnp.zeros(
                par_shape.coef + (1,) * (dim_weights - len(par_shape.coef))
            )
        with expectation:
            model.initialize_solver_and_state(X, y, (init_w, true_params.intercept))

    @pytest.mark.parametrize("dim_intercepts", [0, 1, 2, 3])
    @pytest.mark.solver_related
    def test_initialize_solver_intercepts_dimensionality(
        self,
        dim_intercepts,
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
        par_shape = get_param_shape(model, X, y)
        expected_intercept_dim = DIMENSIONALITY_PARAMS[model.__class__.__name__][
            "intercept"
        ]

        # Determine expectation based on model's expected intercept dimensionality
        if dim_intercepts == expected_intercept_dim:
            expectation = does_not_raise()
        else:
            expectation = pytest.raises(
                ValueError, match=r"Invalid parameter dimensionality"
            )

        # Build init_b of the requested dimensionality
        if dim_intercepts == 0:
            init_b = jnp.array([])
        elif dim_intercepts <= len(par_shape.intercept):
            slc = (slice(None),) * dim_intercepts + (0,) * (
                len(par_shape.intercept) - dim_intercepts
            )
            init_b = jnp.zeros(par_shape.intercept)[slc]
        else:
            init_b = jnp.zeros(
                par_shape.intercept + (1,) * (dim_intercepts - len(par_shape.intercept))
            )

        init_w = jnp.zeros(par_shape.coef)
        with expectation:
            model.initialize_solver_and_state(X, y, (init_w, init_b))

    @pytest.mark.parametrize(*fit_init_params_type_init_params)
    @pytest.mark.solver_related
    def test_initialize_solver_init_params_type(
        self,
        request,
        glm_class_type,
        model_instantiation_type,
        expectation,
        init_params_by_model,
    ):
        """
        Test the `initialize_solver` method with various types of initial parameters.
        Ensure that the provided initial parameters are array-like.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation_type
        )
        init_params = self.get_init_params_for_model(init_params_by_model, model)
        with expectation:
            model.initialize_solver_and_state(X, y, init_params)

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
        par_shape = get_param_shape(model, X, y)
        # Create init_w with wrong number of features (first dim)
        wrong_coef_shape = (par_shape.coef[0] + delta_n_features,) + par_shape.coef[1:]
        init_w = jnp.zeros(wrong_coef_shape)
        init_b = jnp.zeros(par_shape.intercept)
        with expectation:
            model.initialize_solver_and_state(X, y, (init_w, init_b))

    #######################
    # Test model.simulate
    #######################
    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (-1, pytest.raises(ValueError, match="X must be 2-dimensional.")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="X must be 2-dimensional.")),
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
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
        if is_population_model(model):
            model._feature_mask = initialize_feature_mask_for_population_glm(
                X, y.shape[1], coef=true_params.coef
            )
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
            model.coef_ = true_params.coef
            model.intercept_ = true_params.intercept
            if is_population_model(model):
                model._feature_mask = initialize_feature_mask_for_population_glm(
                    X, y.shape[1], model._validator.get_empty_params(X, y).coef
                )
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
                    match="Inconsistent number of features.",
                ),
            ),
            (0, does_not_raise()),
            (
                1,
                pytest.raises(
                    ValueError,
                    match="Inconsistent number of features.",
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
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
        if is_population_model(model):
            model._feature_mask = initialize_feature_mask_for_population_glm(
                X, y.shape[1], coef=true_params.coef
            )
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
                    "aux_": None,
                },
            ),
            (
                nmo.glm.PopulationGLM,
                {
                    "coef_": jnp.zeros((3, 1)),
                    "intercept_": jnp.array([1.0]),
                    "scale_": 2.0,
                    "dof_resid_": 3,
                    "aux_": None,
                },
            ),
            (
                nmo.glm.ClassifierGLM,
                {
                    "coef_": jnp.zeros((3, 2, 1)),
                    "intercept_": jnp.array([1.0]),
                    "scale_": 2.0,
                    "dof_resid_": 3,
                    "aux_": None,
                    "_classes_": np.array([2, 3, 5]),
                    "_class_to_index_": {0: 2, 1: 3, 2: 5},
                },
            ),
            (
                nmo.glm.ClassifierPopulationGLM,
                {
                    "coef_": jnp.zeros((3, 2, 1)),
                    "intercept_": jnp.ones((2, 1)),
                    "scale_": 2.0,
                    "dof_resid_": 3,
                    "aux_": None,
                    "_classes_": np.array([2, 3, 5]),
                    "_class_to_index_": {0: 2, 1: 3, 2: 5},
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
        clean_kwargs = dict(
            (k, p) for k, p in kwargs.items() if k in model_class._get_param_names()
        )

        if regularizer == "UnRegularized":
            kwargs.pop("regularizer_strength")

        model = model_class(**clean_kwargs)

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
        fit_state.pop("optim_info_")
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
                    "aux_": None,
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
                    "aux_": None,
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
            fit_state.pop("optim_info_")
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
        fit_state.pop("optim_info_")
        initial_params.update(fit_state)

        # Save
        save_path = tmp_path / "test_model.npz"
        fitted_model.save_params(save_path)

        # Load
        loaded_model = nmo.load_model(save_path)
        loaded_params = loaded_model.get_params()
        fit_state = loaded_model._get_fit_state()
        fit_state.pop("solver_state_")
        fit_state.pop("optim_info_")
        loaded_params.update(fit_state)

        # Assert states are close
        for k, v in fit_state.items():
            if v is None:
                assert initial_params[k] is None
            else:
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
                "observation_model::class",
                "InvalidObservations",
                pytest.raises(
                    ValueError, match="The class '[A-z]+' is not a native NeMoS"
                ),
            ),
            # Full path string
            (
                "observation_model::class",
                "nemos.observation_models.InvalidObservations",
                pytest.raises(
                    ValueError, match="The class '[A-z]+' is not a native NeMoS"
                ),
            ),
            # Replace observation model class name  with an instance
            (
                "observation_model::class",
                nmo.observation_models.GammaObservations(),
                pytest.raises(
                    ValueError,
                    match="Object arrays cannot be loaded when allow_pickle=False",
                ),
            ),
            # Replace observation model class name with class
            (
                "observation_model::class",
                nmo.observation_models.GammaObservations,
                pytest.raises(
                    ValueError,
                    match="Object arrays cannot be loaded when allow_pickle=False",
                ),
            ),
            # Replace link function with another callable
            (
                "observation_model::params::inverse_link_function",
                np.exp,
                pytest.raises(
                    ValueError,
                    match="Object arrays cannot be loaded when allow_pickle=False",
                ),
            ),
            # Unexpected dtype for class name
            (
                "dict::regularizer::item::class",
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


@pytest.mark.parametrize("glm_type", ["", "population_"])
@pytest.mark.parametrize(
    "model_instantiation",
    [
        "gaussianGLM_model_instantiation",
        "poissonGLM_model_instantiation",
        "gammaGLM_model_instantiation",
        "bernoulliGLM_model_instantiation",
        "negativeBinomialGLM_model_instantiation",
        "classifierGLM_model_instantiation",
    ],
)
class TestGLMObservationModel:
    """
    Shared unit tests of the GLM class that do depend on observation model.
    i.e. tests that directly depend on observation model methods (e.g. model.fit, model.score, model.update),
    and tests that inspect the output when observation model methods are called.

    For new observation models, add it in the class parameterization above, and add cases for the fixtures below.
    """

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

        elif "gaussian" in model_instantiation:

            def ll(y, mean_firing, scale):
                if y.ndim == 1:
                    norm = y.shape[0]
                elif y.ndim == 2:
                    norm = y.shape[0] * y.shape[1]
                return (
                    sm.families.Gaussian().loglike(y, mean_firing, scale=scale) / norm
                )

        elif "classifier" in model_instantiation:

            def ll(y, log_proba):
                proba = jnp.exp(log_proba)
                proba = proba.reshape(-1, proba.shape[-1])
                y = y.reshape(-1)
                res = np.array(
                    [
                        sts.multinomial(1, pi).logpmf(
                            jax.nn.one_hot(yi, proba.shape[-1])
                        )
                        for pi, yi in zip(proba, y)
                    ]
                ).sum()
                res /= y.shape[0]
                return res

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
                C=np.inf,
            )

        elif "negativeBinomial" in model_instantiation:
            return None

        elif "gaussian" in model_instantiation:
            return LinearRegression(fit_intercept=True)

        elif "classifier" in model_instantiation:
            # In sklearn 1.5+, multinomial is the default with lbfgs solver
            # Use C=1.0 (Ridge with strength=1.0) for identifiable parameters
            return LogisticRegression(
                fit_intercept=True,
                tol=10**-12,
                C=1.0,
                solver="lbfgs",
                max_iter=1000,
            )

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

        elif "gaussian" in model_instantiation:
            return 1.0

        elif "classifier" in model_instantiation:
            return 0.1

        else:
            raise ValueError("Unknown model instantiation")

    @pytest.fixture
    def dof_lasso_dof(self, glm_type, model_instantiation):
        """
        Fixture for test_estimate_dof_resid
        """
        if "poisson" in model_instantiation:
            if is_population_glm_type(glm_type):
                return np.array([3, 0, 0])
            else:
                return np.array([3])

        elif "gamma" in model_instantiation:
            if is_population_glm_type(glm_type):
                return np.array([1, 4, 3])
            else:
                return np.array([3])

        elif "bernoulli" in model_instantiation:
            if is_population_glm_type(glm_type):
                return np.array([3, 2, 1])
            else:
                return np.array([3])

        elif "negativeBinomial" in model_instantiation:
            if is_population_glm_type(glm_type):
                return np.array([3, 2, 4])
            else:
                return np.array([5])

        elif "gaussian" in model_instantiation:
            if is_population_glm_type(glm_type):
                return np.array([5, 5, 5])
            else:
                return np.array([3])

        elif "classifier" in model_instantiation:
            # Classifier models have (n_features, n_classes) coef shape
            # For lasso, count surviving coefficients across all classes
            # Note: LASSO convergence can vary slightly across environments
            if is_population_glm_type(glm_type):
                return np.array([6, 4, 5])
            else:
                return np.array([5])

        else:
            raise ValueError("Unknown model instantiation")

    @pytest.fixture
    def dof_non_lasso_dof(self, glm_type, model_instantiation):
        """
        Fixture for test_estimate_dof_resid
        """
        if "classifier" in model_instantiation:
            if "population" in glm_type:
                return np.array([10, 10, 10])
            else:
                return np.array([10])
        else:
            if "population" in glm_type:
                return np.array([5, 5, 5])
            else:
                return np.array([5])

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

        elif "gaussian" in model_instantiation:
            return False

        elif "classifier" in model_instantiation:
            return False

        else:
            raise ValueError("Unknown model instantiation")

    @pytest.fixture
    def model_repr(self, glm_type, model_instantiation):
        """
        Fixture for test_repr_out
        """
        if "poisson" in model_instantiation:
            if is_population_glm_type(glm_type):
                return "PopulationGLM(\n    observation_model=PoissonObservations(),\n    inverse_link_function=exp,\n    regularizer=UnRegularized(),\n    solver_name='GradientDescent'\n)"
            else:
                return "GLM(\n    observation_model=PoissonObservations(),\n    inverse_link_function=exp,\n    regularizer=UnRegularized(),\n    solver_name='GradientDescent'\n)"

        elif "gamma" in model_instantiation:
            if is_population_glm_type(glm_type):
                return "PopulationGLM(\n    observation_model=GammaObservations(),\n    inverse_link_function=one_over_x,\n    regularizer=UnRegularized(),\n    solver_name='GradientDescent'\n)"
            else:
                return "GLM(\n    observation_model=GammaObservations(),\n    inverse_link_function=one_over_x,\n    regularizer=UnRegularized(),\n    solver_name='GradientDescent'\n)"

        elif "bernoulli" in model_instantiation:
            if is_population_glm_type(glm_type):
                return "PopulationGLM(\n    observation_model=BernoulliObservations(),\n    inverse_link_function=logistic,\n    regularizer=UnRegularized(),\n    solver_name='GradientDescent'\n)"
            else:
                return "GLM(\n    observation_model=BernoulliObservations(),\n    inverse_link_function=logistic,\n    regularizer=UnRegularized(),\n    solver_name='GradientDescent'\n)"

        elif "negativeBinomial" in model_instantiation:
            if is_population_glm_type(glm_type):
                return "PopulationGLM(\n    observation_model=NegativeBinomialObservations(scale=1.0),\n    inverse_link_function=exp,\n    regularizer=UnRegularized(),\n    solver_name='LBFGS'\n)"
            else:
                return "GLM(\n    observation_model=NegativeBinomialObservations(scale=1.0),\n    inverse_link_function=exp,\n    regularizer=UnRegularized(),\n    solver_name='LBFGS'\n)"

        elif "gaussian" in model_instantiation:
            if is_population_glm_type(glm_type):
                return "PopulationGLM(\n    observation_model=GaussianObservations(),\n    inverse_link_function=identity,\n    regularizer=UnRegularized(),\n    solver_name='LBFGS'\n)"
            else:
                return "GLM(\n    observation_model=GaussianObservations(),\n    inverse_link_function=identity,\n    regularizer=UnRegularized(),\n    solver_name='LBFGS'\n)"

        elif "classifier" in model_instantiation:
            if is_population_glm_type(glm_type):
                return "ClassifierPopulationGLM(\n    n_classes=3,\n    inverse_link_function=log_softmax,\n    regularizer=UnRegularized(),\n    solver_name='GradientDescent'\n)"
            else:
                return "ClassifierGLM(\n    n_classes=3,\n    inverse_link_function=log_softmax,\n    regularizer=UnRegularized(),\n    solver_name='GradientDescent'\n)"

        else:
            raise ValueError("Unknown model instantiation")

    #######################
    # Test initialization #
    #######################
    @pytest.mark.parametrize(
        "X, y",
        [
            (jnp.ones((2, 4)), jnp.ones((2,))),
            (jnp.zeros((2, 4)), jnp.ones((2,))),
        ],
    )
    def test_parameter_initialization(
        self, X, y, request, glm_type, model_instantiation
    ):
        _, _, model, _, _ = request.getfixturevalue(glm_type + model_instantiation)

        # right now default initialization is specific to poissonGLMs and will fail for the others
        # TODO: this test will need to be updated once we move parameter initialization to be observation model specific
        if is_population_glm_type(glm_type):
            y = np.tile(y[:, None], (1, 3))

        if "poisson" in model_instantiation:
            params = model._model_specific_initialization(X, y)

            if is_population_glm_type(glm_type):
                assert params.coef.shape == (X.shape[1], y.shape[1])
                assert params.intercept.shape == (y.shape[1],)
            else:
                assert params.coef.shape == (X.shape[1],)
                assert params.intercept.shape == (1,)
        else:
            return

    @pytest.mark.requires_x64
    @pytest.mark.solver_related
    @pytest.mark.filterwarnings("ignore:The fit did not converge:RuntimeWarning")
    def test_fit_pytree_equivalence(self, request, glm_type, model_instantiation):
        """Check that the glm fit with pytree learns the same parameters."""
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        X_tree, _, model_tree, true_params_tree, _ = request.getfixturevalue(
            glm_type + model_instantiation + "_pytree"
        )
        # fit both models
        model.solver_kwargs.update(dict(tol=1e-12, maxiter=10**5))
        model_tree.solver_kwargs.update(dict(tol=1e-12, maxiter=10**5))
        model.fit(X, y, init_params=(true_params.coef, true_params.intercept))
        model_tree.fit(
            X_tree, y, init_params=(true_params_tree.coef, true_params_tree.intercept)
        )

        # get the flat parameters
        flat_coef = np.concatenate(jax.tree_util.tree_leaves(model_tree.coef_), axis=0)

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
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
        with expectation:
            model.score(X, y, score_type=score_type)

    @pytest.mark.requires_x64
    def test_loglikelihood_against_scipy_stats(
        self, request, glm_type, model_instantiation, ll_scipy_stats
    ):
        """
        Compare the model's log-likelihood computation against `jax.scipy`.
        Ensure consistent and correct calculations.
        """

        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        # set model coeff
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
        if is_population_model(model):
            model._feature_mask = initialize_feature_mask_for_population_glm(
                X, y.shape[1], coef=true_params.coef
            )
        # get the rate
        mean_firing = getattr(model, "predict_proba", model.predict)(X)
        # compute the log-likelihood using jax.scipy
        if "gamma" in model_instantiation or "gaussian" in model_instantiation:
            mean_ll_jax = ll_scipy_stats(y, mean_firing, model.scale_)
        else:
            mean_ll_jax = ll_scipy_stats(y, mean_firing)

        model_ll = model.score(X, y, score_type="log-likelihood")
        if not np.allclose(mean_ll_jax, model_ll):
            raise ValueError(
                f"Log-likelihood of {glm_type + model_instantiation} does not match "
                "that of jax.scipy!"
            )

    #####################
    # Test model.update #
    #####################
    @pytest.mark.parametrize(
        "n_samples, expectation",
        [
            (None, does_not_raise()),
            (100, does_not_raise()),
            (
                1.0,
                pytest.raises(
                    TypeError, match="`n_samples` must be `None` or of type `int`"
                ),
            ),
            (
                "str",
                pytest.raises(
                    TypeError, match="`n_samples` must be `None` or of type `int`"
                ),
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
        state = model.initialize_solver_and_state(X, y, params)
        with expectation:
            model.update(
                params,
                state,
                X[:batch_size],
                y[:batch_size],
                n_samples=n_samples,
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
        state = model.initialize_solver_and_state(X, y, params)
        assert model.coef_ is None
        assert model.intercept_ is None
        if "gamma" not in model_instantiation and "gaussian" not in model_instantiation:
            # gamma model instantiation sets the scale
            assert model.scale_ is None
        _, _ = model.update(params, state, X[:batch_size], y[:batch_size])
        assert model.coef_ is not None
        assert model.intercept_ is not None
        assert model.scale_ is not None

    @pytest.mark.parametrize("nan_inputs", [True, False])
    @pytest.mark.parametrize(
        "solver_name", ["ProximalGradient", "GradientDescent", "LBFGS", "BFGS"]
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
        state = model.initialize_solver_and_state(X, y, params)
        assert model.coef_ is None
        assert model.intercept_ is None
        if "gamma" not in model_instantiation and "gaussian" not in model_instantiation:
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
    @pytest.mark.requires_x64
    def test_update_nan_drop_at_jit_comp(
        self, batch_size, request, glm_type, model_instantiation
    ):
        """Test that jit compilation does not affect the update in the presence of nans."""
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        model.solver_kwargs.update({"stepsize": 0.01})
        params = model.initialize_params(X, y)
        state = model.initialize_solver_and_state(X, y, params)
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
        assert not jnp.allclose(params[0], jit_update[0]) or not jnp.allclose(
            params[1], jit_update[1]
        )

        # update without jitting
        nojit_update = deepcopy(params)
        nojit_state = deepcopy(state)
        with jax.disable_jit(True):
            for _ in range(tot_iter):
                nojit_update, nojit_state = model.update(
                    nojit_update, nojit_state, Xnan, y[:batch_size]
                )
        # check for equivalence update
        assert jnp.allclose(nojit_update[0], jit_update[0]) and jnp.allclose(
            nojit_update[1], jit_update[1]
        )

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
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
        if is_population_model(model):
            model._feature_mask = initialize_feature_mask_for_population_glm(
                X, y.shape[1], coef=true_params.coef
            )
        if input_type == TsdFrame:
            X = TsdFrame(t=np.arange(X.shape[0]), d=X)
        count, rate = model.simulate(
            random_key=jax.random.key(123),
            feedforward_input=X,
        )
        if is_population_model(model) and (expected_out_type == Tsd):
            assert isinstance(count, TsdFrame)
            # For classifier population models, rate has shape (n_samples, n_neurons, n_classes)
            if is_classifier_model(model):
                from pynapple.core.time_series import TsdTensor

                assert isinstance(rate, TsdTensor)
            else:
                assert isinstance(rate, TsdFrame)
        elif is_classifier_model(model) and (expected_out_type == Tsd):
            # For classifier single neuron, count is Tsd but rate is TsdFrame (n_samples, n_classes)
            assert isinstance(count, expected_out_type)
            assert isinstance(rate, TsdFrame)
        else:
            assert isinstance(count, expected_out_type)
            assert isinstance(rate, expected_out_type)

    def test_simulate_feedforward_glm(self, request, glm_type, model_instantiation):
        """Test that simulate goes through"""
        X, y, model, params, rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        model.coef_ = params.coef
        model.intercept_ = params.intercept
        model.scale_ = model.observation_model.scale
        if is_population_model(model):
            model._feature_mask = initialize_feature_mask_for_population_glm(
                X, y.shape[1], coef=params.coef
            )
        ysim, ratesim = model.simulate(jax.random.key(123), X)
        # check that the expected dimensionality is returned
        # Classifier models have an extra dimension for categories in ratesim (log-probabilities)
        expected_base_ndim = 1 + (1 if is_population_model(model) else 0)
        assert ysim.ndim == expected_base_ndim
        # ratesim has +1 dimension for classifier models (probabilities per category)
        assert ratesim.ndim == expected_base_ndim + (
            1 if is_classifier_model(model) else 0
        )
        # check that the rates and spikes has the same shape for the first dims
        assert ratesim.shape[0] == ysim.shape[0]
        # check the time point number is that expected (same as the input)
        assert ysim.shape[0] == X.shape[0]

    ########################################
    # Compare with standard implementation #
    ########################################
    @pytest.mark.solver_related
    @pytest.mark.filterwarnings("ignore:The fit did not converge:RuntimeWarning")
    def test_compatibility_with_sklearn_cv(
        self, request, glm_type, model_instantiation
    ):
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        param_grid = {"solver_name": ["BFGS", "GradientDescent"]}
        model.solver_kwargs.update(dict(maxiter=2))
        cls = GridSearchCV(model, param_grid).fit(X, y)
        # check that the repr works after cloning
        repr(cls)

    @staticmethod
    def _format_sklearn_params(sklearn_model, nemos_model):
        """Format sklearn params for comparison with nemos.

        sklearn LogisticRegression uses shape (n_classes, n_features).
        nemos uses shape (n_features, n_classes).

        Returns coef and intercept in nemos format.
        """
        if is_classifier_model(nemos_model):
            # sklearn: (n_classes, n_features) -> nemos: (n_features, n_classes)
            coef = sklearn_model.coef_.T
            intercept = sklearn_model.intercept_
        else:
            coef, intercept = sklearn_model.coef_, sklearn_model.intercept_
        return coef, intercept

    @staticmethod
    def _assert_params_match(
        sklearn_coef, sklearn_intercept, nemos_coef, nemos_intercept, atol=1e-6
    ):
        """Assert that sklearn and nemos parameters match within tolerance."""
        match_weights = jnp.allclose(sklearn_coef, nemos_coef, atol=atol, rtol=0.0)
        match_intercepts = jnp.allclose(
            sklearn_intercept, nemos_intercept, atol=atol, rtol=0.0
        )
        if not (match_weights and match_intercepts):
            raise ValueError("GLM.fit estimate does not match sklearn!")

    @pytest.mark.parametrize("solver_name", ["LBFGS"])
    @pytest.mark.solver_related
    @pytest.mark.requires_x64
    @pytest.mark.filterwarnings("ignore:Setting penalty=None will ignore:UserWarning")
    def test_glm_fit_matches_sklearn(
        self, solver_name, request, glm_type, model_instantiation, sklearn_model
    ):
        """Test that nemos GLM produces the same estimates as sklearn."""
        if sklearn_model is None:
            pytest.skip(f"sklearn model is not available for {model_instantiation}")

        X, y, model_obs, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )

        n_samples = (
            X.shape[0] if not isinstance(X, dict) else list(X.values())[0].shape[0]
        )

        # Classifier models need Ridge regularization for identifiable parameters
        # (over-parameterized model). sklearn uses sum(NLL) + (1/2C)*||w||^2,
        # nemos uses mean(NLL) + strength/2*||w||^2, so strength = 1/(C*n_samples).
        if "classifier" in model_instantiation.lower():
            regularizer = nmo.regularizer.Ridge()
            regularizer_strength = 1.0 / n_samples  # Match sklearn C=1.0
        else:
            regularizer = nmo.regularizer.UnRegularized()
            regularizer_strength = None

        kwargs = dict(
            n_classes=getattr(model_obs, "n_classes", None),
            regularizer=regularizer,
            regularizer_strength=regularizer_strength,
            observation_model=model_obs.observation_model,
            solver_name=solver_name,
            solver_kwargs={"tol": 10**-12},
        )
        clean_kwargs = dict(
            (k, p) for k, p in kwargs.items() if k in model_obs._get_param_names()
        )

        model = type(model_obs)(**clean_kwargs)

        # set gamma inverse link function to match sklearn
        if "gamma" in model_instantiation:
            model.inverse_link_function = jnp.exp

        model.fit(X, y)

        is_population = is_population_model(model)

        if is_population:
            # Population GLM: fit each neuron separately in sklearn and compare
            for n in range(y.shape[1]):
                sklearn_model.fit(X, y[:, n])
                sk_coef, sk_intercept = self._format_sklearn_params(
                    sklearn_model, model
                )
                self._assert_params_match(
                    sk_coef,
                    sk_intercept,
                    model.coef_[:, n],
                    model.intercept_[n],
                    atol=1e-6,
                )
        else:
            sklearn_model.fit(X, y)
            sk_coef, sk_intercept = self._format_sklearn_params(sklearn_model, model)
            self._assert_params_match(
                sk_coef,
                sk_intercept,
                model.coef_,
                model.intercept_,
                atol=1e-6,
            )

    @staticmethod
    def _get_expected_par_shape(X, y, model):

        X_flat = jax.tree_util.tree_leaves(X)
        n_features = [x.shape[1] for x in X_flat]
        is_population = is_population_model(model)
        if is_population:
            n_neurons = y.shape[1]
        if is_classifier_model(model):
            n_classes = model.n_classes
            if is_population:
                coef_shape = [(nf, n_neurons, n_classes) for nf in n_features]
                intercept_shape = (n_neurons, n_classes)
            else:
                coef_shape = [(nf, n_classes) for nf in n_features]
                intercept_shape = (n_classes,)
        else:
            if is_population:
                coef_shape = [(nf, n_neurons) for nf in n_features]
                intercept_shape = (n_neurons,)
            else:
                coef_shape = [(nf,) for nf in n_features]
                intercept_shape = (1,)
        return coef_shape, intercept_shape

    @pytest.mark.parametrize(
        "X_shape, y_shape",
        [
            ((10, 2), (10,)),
            ((11, 3), (11,)),
            # pytree X
            ([(10, 3), (10, 2)], (10,)),
        ],
    )
    def test_initialize_params(
        self, request, glm_type, model_instantiation, X_shape, y_shape
    ):
        _, y, model, _, _ = request.getfixturevalue(glm_type + model_instantiation)
        if isinstance(X_shape, tuple):
            X = np.ones(X_shape)
        else:
            X = {f"{k}": np.ones(s) for k, s in enumerate(X_shape)}
        y = y[: y_shape[0]]
        coef, intercept = model.initialize_params(X, y)
        coef_shape, intercept_shape = self._get_expected_par_shape(X, y, model)

        if any(
            c.shape != s for c, s in zip(jax.tree_util.tree_leaves(coef), coef_shape)
        ):
            raise ValueError("Shape mismatch coefficients")
        if intercept.shape != intercept_shape:
            raise ValueError("Shape mismatch intercepts")

    #####################
    # Test residual DOF #
    #####################
    @pytest.mark.parametrize(
        "reg, dof, strength",
        [
            (nmo.regularizer.UnRegularized(), "dof_non_lasso_dof", None),
            (
                nmo.regularizer.Lasso(),
                "dof_lasso_dof",
                "dof_lasso_strength",
            ),  # this lasso fit has only 3 coeff of the first neuron
            # surviving
            (nmo.regularizer.Ridge(), "dof_non_lasso_dof", 1.0),
        ],
    )
    @pytest.mark.parametrize("n_samples", [1, 20])
    @pytest.mark.solver_related
    @pytest.mark.requires_x64
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
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        # different dof for different obs models with lasso
        dof = request.getfixturevalue(dof)
        # need different strengths for different obs models with lasso reg
        # for 3 coefs to survive
        if isinstance(strength, str):
            strength = request.getfixturevalue(strength)
        n_m1_classes = getattr(model, "n_classes", 2) - 1
        model.set_params(regularizer=reg, regularizer_strength=strength)
        model.solver_name = model.regularizer.default_solver
        model.solver_kwargs.update({"maxiter": 10**5})
        model.fit(X, y)
        num = model._estimate_resid_degrees_of_freedom(X, n_samples=n_samples)
        expected_dof_resid = n_samples - dof - n_m1_classes
        assert np.allclose(num, expected_dof_resid)

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
            slv_class = solvers.solver_registry[solver_name]
            nmo.glm.GLM._check_solver_kwargs(slv_class, solver_kwargs)
        except NameError:
            solver_kwargs.pop("batch_size")

        # if the regularizer is not allowed for the solver type, return
        try:
            kwargs = dict(
                n_classes=getattr(model, "n_classes", None),
                regularizer=regularizer,
                solver_name=solver_name,
                inverse_link_function=inv_link,
                observation_model=obs,
                solver_kwargs=solver_kwargs,
                regularizer_strength=None if regularizer == "UnRegularized" else 1.0,
            )
            clean_kwargs = dict(
                (k, p) for k, p in kwargs.items() if k in model._get_param_names()
            )
            model = model.__class__(**clean_kwargs)
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

    @pytest.mark.solver_related
    @pytest.mark.requires_x64
    def test_fit_mask_grouplasso(self, glm_type, model_instantiation, request):
        """Test that the group lasso fit goes through"""
        X, y, model, _, _ = request.getfixturevalue(glm_type + model_instantiation)

        mask = _create_grouplasso_mask(X, y, model)

        model.set_params(
            regularizer=nmo.regularizer.GroupLasso(mask=mask),
            solver_name="ProximalGradient",
            regularizer_strength=1.0,
        )
        model.fit(X, y)


@pytest.mark.parametrize(
    "model_instantiation",
    [
        "population_poissonGLM_model_instantiation",
        "population_classifierGLM_model_instantiation",
    ],
)
class TestPopulationGLM:
    """
    Unit tests specific to the PopulationGLM class that are independent of the observation model.
    """

    #######################################
    # Compare with standard implementation
    #######################################

    def test_sklearn_clone(self, model_instantiation, request):
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation
        )
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
        model._feature_mask = initialize_feature_mask_for_population_glm(X, y.shape[1])
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
                does_not_raise(),
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
    def test_feature_mask_setter(self, mask, expectation, model_instantiation, request):
        _, _, model, _, _ = request.getfixturevalue(model_instantiation)
        with expectation:
            model.feature_mask = mask

    @pytest.fixture
    def feature_mask_compatibility_fit_expectation(self, model_instantiation):
        """
        Fixture to return the expected exceptions for test_feature_mask_compatibility_fit
        based on the model type (classifier vs non-classifier).

        For classifier models, the feature_mask shape is (n_features, n_neurons, n_classes)
        which means all the test masks (which lack the n_classes dimension) will fail
        shape validation.
        """
        is_classifier = "classifier" in model_instantiation

        type_error_match = "feature_mask and X must have the same structure|feature_mask and coef must have the same structure"
        shape_mismatch_match = "Inconsistent feature mask shape|The shape of the ``feature_mask`` array must match"

        if is_classifier:
            # Classifier models expect feature_mask shape (n_features, n_neurons, n_classes)
            # Masks without n_classes dimension fail shape validation
            return {
                "correct_shape_np": pytest.raises(
                    ValueError, match=shape_mismatch_match
                ),
                "correct_shape_classifier_np": does_not_raise(),
                "wrong_n_features_np": pytest.raises(
                    ValueError, match=shape_mismatch_match
                ),
                "wrong_n_neurons_np": pytest.raises(
                    ValueError, match=shape_mismatch_match
                ),
                "correct_shape_pytree": pytest.raises(
                    ValueError, match=shape_mismatch_match
                ),
                "correct_shape_classifier_pytree": does_not_raise(),
                "wrong_n_neurons_pytree": pytest.raises(
                    ValueError, match=shape_mismatch_match
                ),
                "missing_key_pytree": pytest.raises(TypeError, match=type_error_match),
                "missing_key_wrong_shape_pytree": pytest.raises(
                    TypeError, match=type_error_match
                ),
            }
        else:
            # Non-classifier models expect feature_mask shape (n_features, n_neurons)
            return {
                "correct_shape_np": does_not_raise(),
                "correct_shape_classifier_np": pytest.raises(
                    ValueError, match=shape_mismatch_match
                ),
                "wrong_n_features_np": pytest.raises(
                    ValueError,
                    match="The shape of the ``feature_mask`` array must match that of the ``coef``",
                ),
                "wrong_n_neurons_np": pytest.raises(
                    ValueError,
                    match="The shape of the ``feature_mask`` array must match that of the ``coef``",
                ),
                "correct_shape_pytree": does_not_raise(),
                "correct_shape_classifier_pytree": pytest.raises(
                    ValueError, match="Inconsistent number of neurons. feature_mask has"
                ),
                "wrong_n_neurons_pytree": pytest.raises(
                    ValueError, match="Inconsistent number of neurons. feature_mask has"
                ),
                "missing_key_pytree": pytest.raises(TypeError, match=type_error_match),
                "missing_key_wrong_shape_pytree": pytest.raises(
                    TypeError, match=type_error_match
                ),
            }

    # Parametrization for test_feature_mask_compatibility_fit masks
    feature_mask_compatibility_fit_masks = (
        "mask, mask_key_np, mask_key_pytree",
        [
            # Non-classifier correct shape: (n_features, n_neurons) = (5, 3)
            (
                np.array([0, 1, 1] * 5).reshape(5, 3),
                "correct_shape_np",
                "missing_key_pytree",  # pytree expects dict, not array
            ),
            # Classifier correct shape: (n_features, n_neurons, n_classes) = (5, 3, 3)
            (
                np.ones((5, 3, 3), dtype=int),
                "correct_shape_classifier_np",
                "missing_key_pytree",  # pytree expects dict, not array
            ),
            (
                np.array([0, 1, 1] * 4).reshape(4, 3),
                "wrong_n_features_np",
                "missing_key_pytree",
            ),
            (
                np.array([0, 1, 1, 1] * 5).reshape(5, 4),
                "wrong_n_neurons_np",
                "missing_key_pytree",
            ),
            # Non-classifier pytree correct shape: {'input_1': (3, 3), 'input_2': (2, 3)}
            (
                {"input_1": np.array([0, 1, 0]), "input_2": np.array([1, 0, 1])},
                "missing_key_pytree",  # np expects array, not dict
                "correct_shape_pytree",
            ),
            # Classifier pytree correct shape: {'input_1': (3, 3, 3), 'input_2': (2, 3, 3)}
            (
                {
                    "input_1": np.ones((3, 3, 3), dtype=int),
                    "input_2": np.ones((2, 3, 3), dtype=int),
                },
                "missing_key_pytree",  # np expects array, not dict
                "correct_shape_classifier_pytree",
            ),
            (
                {"input_1": np.array([0, 1, 0, 1]), "input_2": np.array([1, 0, 1, 0])},
                "missing_key_pytree",
                "wrong_n_neurons_pytree",
            ),
            (
                {"input_1": np.array([0, 1, 0])},
                "missing_key_pytree",
                "missing_key_pytree",
            ),
            (
                {"input_1": np.array([0, 1, 0, 1])},
                "missing_key_pytree",
                "missing_key_wrong_shape_pytree",
            ),
        ],
    )

    @pytest.mark.parametrize(*feature_mask_compatibility_fit_masks)
    @pytest.mark.parametrize("attr_name", ["fit", "predict", "score"])
    @pytest.mark.parametrize(
        "model_suffix",
        [
            "",
            "_pytree",
        ],
    )
    def test_feature_mask_compatibility_fit(
        self,
        mask,
        mask_key_np,
        mask_key_pytree,
        attr_name,
        request,
        model_suffix,
        model_instantiation,
        feature_mask_compatibility_fit_expectation,
    ):
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation + model_suffix
        )
        if "pytree" in model_suffix:
            expectation = feature_mask_compatibility_fit_expectation[mask_key_pytree]
        else:
            expectation = feature_mask_compatibility_fit_expectation[mask_key_np]
        model.feature_mask = mask
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
        with expectation:
            if attr_name == "predict":
                getattr(model, attr_name)(X)
            else:
                getattr(model, attr_name)(X, y)

    @pytest.mark.parametrize(
        "model_suffix",
        [
            "",
            "_pytree",
        ],
    )
    @pytest.mark.solver_related
    def test_metadata_pynapple_fit(self, model_suffix, request, model_instantiation):
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation + model_suffix
        )
        y = TsdFrame(
            t=np.arange(y.shape[0]), d=y, metadata={"y": np.arange(y.shape[1])}
        )
        model.fit(X, y)
        assert hasattr(model, "_metadata") and (model._metadata is not None)
        assert np.all(y._metadata == model._metadata["metadata"])
        assert np.all(y.columns == model._metadata["columns"])

    @pytest.mark.parametrize(
        "model_suffix",
        [
            "",
            "_pytree",
        ],
    )
    @pytest.mark.solver_related
    def test_metadata_pynapple_is_deepcopied(
        self, model_suffix, model_instantiation, request
    ):
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation + model_suffix
        )
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
        "model_suffix",
        [
            "",
            "_pytree",
        ],
    )
    @pytest.mark.solver_related
    def test_metadata_pynapple_predict(
        self, model_suffix, model_instantiation, request
    ):
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            model_instantiation + model_suffix
        )
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
        "population_gaussianGLM_model_instantiation",
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
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
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
                {"stepsize": 0.1, "tol": 10**-9},
            ),
            (
                nmo.regularizer.Ridge(),
                1.0,
                "LBFGS",
                {"tol": 10**-9},
            ),
            (
                nmo.regularizer.Ridge(),
                1.0,
                "LBFGS",
                {"stepsize": 0.1, "tol": 10**-9},
            ),
            (
                nmo.regularizer.Lasso(),
                0.001,
                "ProximalGradient",
                {"tol": 10**-8, "maxiter": 10**5},
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
    @pytest.mark.requires_x64
    @pytest.mark.filterwarnings("ignore:The fit did not converge:RuntimeWarning")
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
        if isinstance(mask, dict):
            X, y, model_class, true_params, firing_rate = request.getfixturevalue(
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
            X, y, model_class, true_params, firing_rate = request.getfixturevalue(
                model_instantiation
            )

            def map_neu(k, coef_):
                ind_array = np.where(mask[:, k])[0]
                coef_stack = coef_
                return ind_array, coef_stack

        mask_bool = jax.tree_util.tree_map(lambda x: np.asarray(x.T, dtype=bool), mask)
        # fit pop glm
        kwargs = dict(
            observation_model=model_class.observation_model,
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
        kwargs.pop("feature_mask")
        for k in range(y.shape[1]):
            model_single_neu = nmo.glm.GLM(**kwargs)
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
        print(model)
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
        if is_population_model(model):
            model._model_specific_initialization(X, y)
        else:
            model._model_specific_initialization(X, y[:, 0])

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
        opt_state = model._initialize_solver_and_state(X, y, true_params)
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

    @pytest.mark.parametrize("glm_type", ["", "population_"])
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
    @pytest.mark.filterwarnings("ignore:The fit did not converge:RuntimeWarning")
    @pytest.mark.requires_x64
    def test_glm_update_consistent_with_fit_with_svrg(
        self,
        request,
        glm_type,
        regr_setup,
        key,
        regularizer_class,
        solver_name,
        glm_class_type,
    ):
        """
        Make sure that calling GLM.update with the rest of the algorithm implemented outside in a naive loop
        is consistent with running the compiled GLM.fit on the same data with the same parameters
        """
        X, y, model, true_params, rate = request.getfixturevalue(
            glm_type + "poissonGLM_model_instantiation" + regr_setup
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
            observation_model=model.observation_model,
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
            observation_model=model.observation_model,
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
        state = glm.initialize_solver_and_state(X, y, params)
        # glm.instantiate_solver(glm.compute_loss)

        # NOTE these two are not the same because for example Ridge augments the loss
        # loss_grad = jax.jit(jax.grad(glm.compute_loss))
        loss_grad = jax.jit(jax.grad(glm._solver_loss_fun))

        # copied from GLM.fit
        # grab data if needed (tree map won't function because param is never a FeaturePytree).
        if isinstance(X, FeaturePytree):
            X = X.data

        iter_num = 0
        while iter_num < maxiter:
            state = state._replace(
                full_grad_at_reference_point=loss_grad(
                    nmo.glm.params.GLMParams(*params), X, y
                ),
            )

            prev_params = params
            for _ in range(m):
                key, subkey = jax.random.split(key)
                ind = jax.random.randint(subkey, (batch_size,), 0, N)
                xi, yi = tree_slice(X, ind), tree_slice(y, ind)
                params, state = glm.update(params, state, xi, yi)

            state = state._replace(
                reference_point=nmo.glm.params.GLMParams(*params),
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
        if is_population_glm_type(glm_type):
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
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
        model.score(X, y)

    def test_simulate_glm(self, inv_link, request, glm_type, model_instantiation):
        """
        Ensure that data can be simulated with different link functions.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        model.observation_model.inverse_link_function = inv_link
        if is_population_glm_type(glm_type):
            model.feature_mask = jnp.ones((X.shape[1], y.shape[1]))
            model.scale_ = jnp.ones((y.shape[1]))
        else:
            model.scale_ = 1.0
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
        ysim, ratesim = model.simulate(jax.random.PRNGKey(123), X)
        assert ysim.shape == y.shape
        assert ratesim.shape == y.shape


@pytest.mark.parametrize(
    "inv_link", [identity, jnp.exp]
)  # identity from inverse_link_function_utils
@pytest.mark.parametrize("glm_type", ["", "population_"])
@pytest.mark.parametrize("model_instantiation", ["gaussianGLM_model_instantiation"])
class TestGaussianGLM:
    """
    Unit tests specific to Gaussian GLM.
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
        if is_population_glm_type(glm_type):
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
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
        model.score(X, y)

    def test_simulate_glm(self, inv_link, request, glm_type, model_instantiation):
        """
        Ensure that data can be simulated with different link functions.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        model.inverse_link_function = inv_link
        if is_population_glm_type(glm_type):
            model.feature_mask = jnp.ones((X.shape[1], y.shape[1]))
            model.scale_ = jnp.ones((y.shape[1]))
        else:
            model.scale_ = 1.0
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
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
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
        if is_population_glm_type(glm_type):
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
        if is_population_glm_type(glm_type):
            model.feature_mask = jnp.ones((X.shape[1], y.shape[1]))
            model.scale_ = jnp.ones((y.shape[1]))
        else:
            model.scale_ = 1.0
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
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
        model.fit(X, y, init_params=(true_params.coef, true_params.intercept))

    def test_score_glm(self, inv_link, request, glm_type, model_instantiation):
        """
        Ensure that the model can be scored with different link functions.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        model.inverse_link_function = inv_link
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
        if is_population_glm_type(glm_type):
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
        if is_population_glm_type(glm_type):
            model.feature_mask = jnp.ones((X.shape[1], y.shape[1]))
            model.scale_ = jnp.ones((y.shape[1]))
        else:
            model.scale_ = 1.0
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
        ysim, ratesim = model.simulate(jax.random.PRNGKey(123), X)
        assert ysim.shape == y.shape
        assert ratesim.shape == y.shape


@pytest.mark.parametrize("inv_link", [log_softmax])
@pytest.mark.parametrize("glm_type", ["", "population_"])
@pytest.mark.parametrize("model_instantiation", ["classifierGLM_model_instantiation"])
class TestClassifierGLM:
    """
    Unit tests specific to classifier GLM.
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

    @pytest.mark.solver_related
    def test_fit_glm_too_few_classes(
        self, inv_link, request, glm_type, model_instantiation
    ):
        """
        Ensure that the model can be fit with different link functions.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        model.inverse_link_function = inv_link
        y = jnp.where(y == 2, 1, y)  # reduce to only 2 classes
        with pytest.raises(ValueError, match="Found only .* unique class labels"):
            model.fit(X, y)

    @pytest.mark.solver_related
    def test_fit_glm_too_many_classes(
        self, inv_link, request, glm_type, model_instantiation
    ):
        """
        Ensure that the model can be fit with different link functions.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        model.inverse_link_function = inv_link
        y = y.at[:10].set(3)  # add another class
        with pytest.raises(ValueError, match="Found .* unique class labels"):
            model.fit(X, y)

    def test_score_glm(self, inv_link, request, glm_type, model_instantiation):
        """
        Ensure that the model can be scored with different link functions.
        """
        X, y, model, true_params, firing_rate = request.getfixturevalue(
            glm_type + model_instantiation
        )
        model.inverse_link_function = inv_link
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
        if is_population_glm_type(glm_type):
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
        if is_population_glm_type(glm_type):
            model.feature_mask = jnp.ones((X.shape[1], y.shape[1], model.n_classes))
            model.scale_ = jnp.ones((y.shape[1]))
            shape_log_proba = (X.shape[0], y.shape[1], model.n_classes)
        else:
            model.scale_ = 1.0
            shape_log_proba = (X.shape[0], model.n_classes)
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
        ysim, log_proba = model.simulate(jax.random.PRNGKey(123), X)
        assert ysim.shape == y.shape
        assert log_proba.shape == shape_log_proba
        assert jnp.all(ysim == ysim.astype(int))

    @pytest.mark.parametrize(
        "n_classes, expectation",
        [
            (
                0,
                pytest.raises(
                    ValueError, match="The number of classes must be an integer"
                ),
            ),
            (
                1,
                pytest.raises(
                    ValueError, match="The number of classes must be an integer"
                ),
            ),
            (2, does_not_raise()),
            (3, does_not_raise()),
            (
                "2",
                pytest.raises(
                    ValueError, match="The number of classes must be an integer"
                ),
            ),
            (
                -2,
                pytest.raises(
                    ValueError, match="The number of classes must be an integer"
                ),
            ),
            (np.array(2), does_not_raise()),
        ],
    )
    def test_n_classes_kind(
        self,
        inv_link,
        n_classes,
        expectation,
        glm_type,
        model_instantiation,
        request,
    ):
        _, _, model, _, _ = request.getfixturevalue(glm_type + model_instantiation)
        with expectation:
            model.__class__(n_classes=n_classes)

        with expectation:
            model.n_classes = n_classes

    @pytest.mark.parametrize(
        "extra_x_dim, expectation",
        [
            (0, does_not_raise()),
            (-1, pytest.raises(ValueError, match="X must be 2-dimensional")),
            (1, pytest.raises(ValueError, match="X must be 2-dimensional")),
        ],
    )
    @pytest.mark.parametrize("xtype", ["", "_pytree"])
    def test_predict_proba_xshape(
        self,
        extra_x_dim,
        expectation,
        inv_link,
        glm_type,
        model_instantiation,
        request,
        xtype,
    ):
        X, _, model, true_params, _ = request.getfixturevalue(
            glm_type + model_instantiation + xtype
        )
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
        if extra_x_dim == 1:
            X = jax.tree_util.tree_map(lambda x: np.expand_dims(x, axis=-1), X)
        if extra_x_dim == -1:
            X = jax.tree_util.tree_map(lambda x: x[..., 0], X)
        with expectation:
            model.predict_proba(X)

    @pytest.mark.parametrize(
        "return_type, expectation",
        [
            ("proba", does_not_raise()),
            ("log-proba", does_not_raise()),
            ("invalid", pytest.raises(ValueError, match="Unrecognized return type")),
        ],
    )
    def test_predict_proba_return_type(
        self,
        return_type,
        expectation,
        inv_link,
        glm_type,
        model_instantiation,
        request,
    ):
        X, _, model, true_params, _ = request.getfixturevalue(
            glm_type + model_instantiation
        )
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
        with expectation:
            model.predict_proba(X, return_type=return_type)

    @pytest.mark.parametrize(
        "X, expectation",
        [
            (np.ones((3, 5)), does_not_raise()),
            (
                nmo.pytrees.FeaturePytree(
                    input_1=np.ones((3, 3)), input_2=np.ones((3, 2))
                ),
                does_not_raise(),
            ),
            # string type
            (
                "invalid",
                pytest.raises(
                    AttributeError, match="'str' object has no attribute 'ndim'"
                ),
            ),
            # wrong number of features
            (
                np.ones((3, 4)),
                pytest.raises(ValueError, match="Inconsistent number of features"),
            ),
            (
                nmo.pytrees.FeaturePytree(
                    input_1=np.ones((3, 1)), input_2=np.ones((3, 2))
                ),
                pytest.raises(ValueError, match="Inconsistent number of features"),
            ),
        ],
    )
    def test_predict_proba_x_structure(
        self,
        X,
        expectation,
        inv_link,
        glm_type,
        model_instantiation,
        request,
    ):
        if isinstance(X, nmo.pytrees.FeaturePytree):
            xtype = "_pytree"
        else:
            xtype = ""
        _, _, model, true_params, _ = request.getfixturevalue(
            glm_type + model_instantiation + xtype
        )
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
        with expectation:
            model.predict_proba(X)

    @pytest.mark.parametrize(
        "method_name",
        [
            "predict_proba",
            "predict",
            "update",
            "compute_loss",
            "simulate",
            "initialize_params",
            "initialize_solver_and_state",
        ],
    )
    def test_must_set_classes_before_calling(
        self,
        method_name,
        inv_link,
        glm_type,
        model_instantiation,
        request,
    ):
        _, _, model, true_params, _ = request.getfixturevalue(
            glm_type + model_instantiation
        )
        model = deepcopy(model)
        model._classes_ = None

        # superset of all possible required inputs
        input_dict = {
            "X": None,
            "y": None,
            "params": None,
            "random_key": None,
            "feedforward_input": None,
            "opt_state": None,
            "init_params": None,
        }
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
        method = getattr(model, method_name)
        required = [
            name
            for name, param in inspect.signature(method).parameters.items()
            if param.default is inspect.Parameter.empty
            and param.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]
        expectation = pytest.raises(
            RuntimeError, match=rf"Classes are not set\..*{method_name}"
        )
        with expectation:
            method(**{k: input_dict[k] for k in required})

    def test_predict_to_label(self, inv_link, glm_type, model_instantiation, request):
        X, _, model, true_params, _ = request.getfixturevalue(
            glm_type + model_instantiation
        )
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
        model.set_classes(np.arange(model.n_classes))
        y = model.predict(X)
        label = np.array([chr(i) for i in range(ord("a"), ord("a") + model.n_classes)])
        model.set_classes(label)
        y_label = model.predict(X)
        assert set(y_label.flatten()).intersection(label) == set(y_label.flatten())
        for i, l in enumerate(label):
            assert np.array_equal(y_label == l, y == i)

    def test_score_from_label(self, inv_link, glm_type, model_instantiation, request):
        X, y, model, true_params, _ = request.getfixturevalue(
            glm_type + model_instantiation
        )
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
        model.set_classes(np.arange(model.n_classes))
        score_regular = model.score(X, y)
        label = np.array([chr(i) for i in range(ord("a"), ord("a") + model.n_classes)])
        model.set_classes(label)
        y_label = model._decode_labels(y)
        score = model.score(X, y_label)
        assert isinstance(score, jnp.ndarray)
        assert jnp.issubdtype(score.dtype, np.floating)
        assert score == score_regular

    def test_fit_from_label(self, inv_link, glm_type, model_instantiation, request):
        X, y, model, true_params, _ = request.getfixturevalue(
            glm_type + model_instantiation
        )
        model_label = deepcopy(model)

        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
        model.set_classes(np.arange(model.n_classes))
        model.fit(X, y)

        label = np.array([chr(i) for i in range(ord("a"), ord("a") + model.n_classes)])
        model_label.set_classes(label)
        y_label = model._decode_labels(y)
        model_label.fit(X, y_label)
        assert jnp.array_equal(model.coef_, model_label.coef_)
        assert jnp.array_equal(model.intercept_, model_label.intercept_)

    def test_simulate_from_label(
        self, inv_link, glm_type, model_instantiation, request
    ):
        X, _, model, true_params, _ = request.getfixturevalue(
            glm_type + model_instantiation
        )
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept
        model.set_classes(np.arange(model.n_classes))
        y, log_prob = model.simulate(jax.random.PRNGKey(1), X)

        label = np.array([chr(i) for i in range(ord("a"), ord("a") + model.n_classes)])
        model.set_classes(label)
        y_label, log_prob_label = model.simulate(jax.random.PRNGKey(1), X)
        assert jnp.array_equal(model._encode_labels(y_label), y)
        assert jnp.array_equal(log_prob_label, log_prob)

    def test_classes_none_initially(
        self, inv_link, glm_type, model_instantiation, request
    ):
        """Test that classes_ is None before set_classes is called."""
        _, _, model, _, _ = request.getfixturevalue(glm_type + model_instantiation)
        # Create a fresh model without set_classes
        if "population" in glm_type:
            fresh_model = nmo.glm.ClassifierPopulationGLM(n_classes=model.n_classes)
        else:
            fresh_model = nmo.glm.ClassifierGLM(n_classes=model.n_classes)
        assert fresh_model.classes_ is None
        assert fresh_model._skip_encoding is False

    def test_skip_encoding_flag(self, inv_link, glm_type, model_instantiation, request):
        """Test that _skip_encoding is True for default labels, False otherwise."""
        _, _, model, _, _ = request.getfixturevalue(glm_type + model_instantiation)
        model = deepcopy(model)

        # Default labels [0, 1, ..., n-1] should skip encoding
        model.set_classes(np.arange(model.n_classes))
        assert model._skip_encoding is True
        assert model._class_to_index_ is None

        # Non-default labels should not skip encoding
        label = np.array([chr(i) for i in range(ord("a"), ord("a") + model.n_classes)])
        model.set_classes(label)
        assert model._skip_encoding is False
        assert model._class_to_index_ is not None

    def test_set_classes_too_many_classes(
        self, inv_link, glm_type, model_instantiation, request
    ):
        """Test that set_classes raises when y has more classes than n_classes."""
        _, _, model, _, _ = request.getfixturevalue(glm_type + model_instantiation)
        model = deepcopy(model)
        # Create labels with more classes than model.n_classes
        too_many = np.arange(model.n_classes + 2)
        with pytest.raises(ValueError, match="Found .* unique class labels"):
            model.set_classes(too_many)

    def test_set_classes_too_few_classes(
        self, inv_link, glm_type, model_instantiation, request
    ):
        """Test that set_classes raises when y has fewer classes than n_classes."""
        _, _, model, _, _ = request.getfixturevalue(glm_type + model_instantiation)
        model = deepcopy(model)
        # Create labels with fewer classes than model.n_classes
        too_few = np.arange(model.n_classes - 1)
        with pytest.raises(ValueError, match="Found only .* unique class labels"):
            model.set_classes(too_few)

    def test_encode_invalid_label_raises(
        self, inv_link, glm_type, model_instantiation, request
    ):
        """Test that encoding an unknown label raises ValueError."""
        X, y, model, true_params, _ = request.getfixturevalue(
            glm_type + model_instantiation
        )
        model = deepcopy(model)
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept

        # Set up string labels
        label = np.array([chr(i) for i in range(ord("a"), ord("a") + model.n_classes)])
        model.set_classes(label)

        # Create y with an invalid label
        if is_population_glm_type(glm_type):
            y_invalid = np.full(y.shape, "z")  # 'z' is not in labels
        else:
            y_invalid = np.array(["z"] * len(y))

        with pytest.raises(ValueError, match="Unrecognized label"):
            model.score(X, y_invalid)

    def test_non_contiguous_integer_labels(
        self, inv_link, glm_type, model_instantiation, request
    ):
        """Test that non-contiguous integer labels work correctly."""
        X, y, model, true_params, _ = request.getfixturevalue(
            glm_type + model_instantiation
        )
        model_nc = deepcopy(model)

        # Fit with default labels first
        model.fit(X, y)

        # Use non-contiguous integers like [5, 10, 15] instead of [0, 1, 2]
        nc_labels = np.array([5 + i * 5 for i in range(model.n_classes)])
        y_nc = nc_labels[y]  # Map 0->5, 1->10, 2->15, etc.

        model_nc.fit(X, y_nc)

        # Coefficients should be the same
        assert jnp.allclose(model.coef_, model_nc.coef_, atol=1e-5)
        assert jnp.allclose(model.intercept_, model_nc.intercept_, atol=1e-5)

        # Predictions should use the non-contiguous labels
        pred_nc = model_nc.predict(X)
        pred = model.predict(X)
        assert jnp.array_equal(pred_nc, nc_labels[pred])

    def test_compute_loss_with_labels(
        self, inv_link, glm_type, model_instantiation, request
    ):
        """Test that compute_loss works with custom labels."""
        X, y, model, true_params, _ = request.getfixturevalue(
            glm_type + model_instantiation
        )
        model = deepcopy(model)
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept

        # Compute loss with default labels
        model.set_classes(np.arange(model.n_classes))
        loss_default = model.compute_loss((model.coef_, model.intercept_), X, y)

        # Compute loss with string labels
        label = np.array([chr(i) for i in range(ord("a"), ord("a") + model.n_classes)])
        model.set_classes(label)
        y_label = model._decode_labels(y)
        loss_label = model.compute_loss((model.coef_, model.intercept_), X, y_label)

        assert jnp.allclose(loss_default, loss_label)

    @pytest.mark.parametrize("return_type", ["proba", "log-proba"])
    def test_predict_proba_with_labels(
        self, inv_link, glm_type, model_instantiation, request, return_type
    ):
        """Test that predict_proba works correctly with custom labels."""
        X, y, model, true_params, _ = request.getfixturevalue(
            glm_type + model_instantiation
        )
        model = deepcopy(model)
        model.coef_ = true_params.coef
        model.intercept_ = true_params.intercept

        # Get probabilities with default labels
        model.set_classes(np.arange(model.n_classes))
        proba_default = model.predict_proba(X, return_type=return_type)

        # Get probabilities with string labels
        label = np.array([chr(i) for i in range(ord("a"), ord("a") + model.n_classes)])
        model.set_classes(label)
        proba_label = model.predict_proba(X, return_type=return_type)

        # Probabilities should be identical (only label interpretation changes)
        assert jnp.allclose(proba_default, proba_label)

    def test_encode_decode_roundtrip(
        self, inv_link, glm_type, model_instantiation, request
    ):
        """Test that encoding then decoding returns original labels."""
        _, y, model, _, _ = request.getfixturevalue(glm_type + model_instantiation)
        model = deepcopy(model)

        # Test with string labels
        label = np.array([chr(i) for i in range(ord("a"), ord("a") + model.n_classes)])
        model.set_classes(label)
        y_label = model._decode_labels(y)

        # Roundtrip: decode -> encode should give original indices
        y_roundtrip = model._encode_labels(y_label)
        assert np.array_equal(y, y_roundtrip)

        # Roundtrip: encode -> decode should give original labels
        y_label_roundtrip = model._decode_labels(model._encode_labels(y_label))
        assert np.array_equal(y_label, y_label_roundtrip)
