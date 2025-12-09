import inspect
import itertools
import warnings
from contextlib import nullcontext as does_not_raise

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy as sp
import scipy.stats as sts
import statsmodels.api as sm

# Import helpers from conftest
from conftest import is_population_model
from numba import njit

import nemos as nmo
from nemos._observation_model_builder import AVAILABLE_OBSERVATION_MODELS
from nemos.inverse_link_function_utils import LINK_NAME_TO_FUNC

MODEL_REGISTRY = {
    "GLM": nmo.glm.GLM,
    "PopulationGLM": nmo.glm.PopulationGLM,
}

INIT_PARAM_LENGTH = {
    "GLM": 2,
    "PopulationGLM": 2,
}

DEFAULT_OBS_SHAPE = {
    "GLM": (500,),
    "PopulationGLM": (500, 3),
}

HARD_CODED_GET_PARAMS_KEYS = {
    "GLM": {
        "inverse_link_function",
        "observation_model",
        "regularizer",
        "regularizer_strength",
        "solver_kwargs",
        "solver_name",
    },
    "PopulationGLM": {
        "inverse_link_function",
        "observation_model",
        "regularizer",
        "regularizer_strength",
        "solver_kwargs",
        "solver_name",
        "feature_mask",
    },
}

# as of now, all models are glm type... in the future this may change.
MODEL_WITH_LINK_FUNCTION_REGISTRY = {
    "GLM": nmo.glm.GLM,
    "PopulationGLM": nmo.glm.PopulationGLM,
}

DEFAULTS = {"GLM": dict(), "PopulationGLM": dict()}


INSTANTIATE_MODEL_ONLY = [
    {"model": m, "obs_model": o, "simulate": False}
    for m, o in itertools.product(MODEL_REGISTRY.keys(), AVAILABLE_OBSERVATION_MODELS)
]

INSTANTIATE_MODEL_AND_SIMULATE = [
    {"model": m, "obs_model": o, "simulate": True}
    for m, o in itertools.product(MODEL_REGISTRY.keys(), AVAILABLE_OBSERVATION_MODELS)
]

INSTANTIATE_MODEL_ONLY_LINK = [
    {"model": m, "obs_model": o, "simulate": False}
    for m, o in itertools.product(
        MODEL_WITH_LINK_FUNCTION_REGISTRY.keys(), AVAILABLE_OBSERVATION_MODELS
    )
]

INSTANTIATE_MODEL_AND_SIMULATE_LINK = [
    {"model": m, "obs_model": o, "simulate": True}
    for m, o in itertools.product(
        MODEL_WITH_LINK_FUNCTION_REGISTRY.keys(), AVAILABLE_OBSERVATION_MODELS
    )
]


def test_all_defaults_assigned():
    PARS_WITHOUT_DEFAULTS = {}
    for k, cls in MODEL_REGISTRY.items():
        PARS_WITHOUT_DEFAULTS[k] = {
            k: v
            for k, v in inspect.signature(cls.__init__).parameters.items()
            if v.default == inspect.Parameter.empty
        }

        PARS_WITHOUT_DEFAULTS[k].pop("self")
        PARS_WITHOUT_DEFAULTS[k].pop("kwargs", None)

    for k, pars_without_default in PARS_WITHOUT_DEFAULTS.items():
        if DEFAULTS[k].keys() != pars_without_default.keys():
            raise ValueError(
                f"The model {k} has the following required parameters:\n"
                f"{list(pars_without_default.keys())}\n"
                f"But the provided defaults are:\n{list(DEFAULTS[k].keys())}.\n"
                "Therefore, the following parameters do not have a default:\n"
                f"{set(pars_without_default.keys()).difference(DEFAULTS[k].keys())}\n"
                f"Please update the ``DEFAULTS`` dictionary for model {k}.\n"
            )


@pytest.mark.parametrize(
    "instantiate_base_regressor_subclass",
    INSTANTIATE_MODEL_ONLY,
    indirect=True,
)
def test_validate_lower_dimensional_data_X(instantiate_base_regressor_subclass):
    """Test behavior with lower-dimensional input data."""
    fixture = instantiate_base_regressor_subclass
    model = fixture.model
    X = jnp.array([1, 2, 3])
    y = jnp.array([0, 1, 1])
    if is_population_model(model):
        y = y[None]
    err_msg = "X must be 2-dimensional"
    with pytest.raises(ValueError, match=err_msg):
        model._validate(X, y, model._model_specific_initialization(X, y))


@pytest.mark.parametrize(
    "instantiate_base_regressor_subclass",
    INSTANTIATE_MODEL_ONLY,
    indirect=True,
)
def test_preprocess_fit_higher_dimensional_data_y(instantiate_base_regressor_subclass):
    """Test behavior with higher-dimensional input data."""
    fixture = instantiate_base_regressor_subclass
    model = fixture.model
    X = jnp.array([[1, 2], [3, 4]])  # Valid 2D X
    y = jnp.array([[[1.0, 1.0, 1.0]]])  # Invalid 3D y
    if is_population_model(model):
        err_msg = "y must be 2-dimensional"
    else:
        err_msg = "y must be 1-dimensional"
    with pytest.raises(ValueError, match=err_msg):
        model._validate(X, y, model._model_specific_initialization(X, y))


@pytest.mark.parametrize(
    "instantiate_base_regressor_subclass",
    INSTANTIATE_MODEL_ONLY,
    indirect=True,
)
def test_validate_higher_dimensional_data_X(instantiate_base_regressor_subclass):
    """Test behavior with higher-dimensional input data."""
    fixture = instantiate_base_regressor_subclass
    model = fixture.model
    X = jnp.array([[[[1, 2], [3, 4]]]])
    y = jnp.array([1, 1])
    if is_population_model(model):
        y = y[None]
    with pytest.raises(ValueError, match="X must be 2-dimensional\\."):
        model._validate(X, y, model._model_specific_initialization(X, y))


@pytest.mark.parametrize(
    "instantiate_base_regressor_subclass",
    INSTANTIATE_MODEL_ONLY,
    indirect=True,
)
class TestModelCommons:
    """
    Unit tests for methods common to all BaseRegressor subclasses.
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
    def test_init_solver_type(
        self, solver_name, expectation, instantiate_base_regressor_subclass
    ):
        """
        Test that an error is raised if a non-compatible solver is passed.
        """
        fixture = instantiate_base_regressor_subclass
        model = fixture.model.__class__
        pars = DEFAULTS[model.__name__].copy()
        pars.update(dict(solver_name=solver_name))
        with expectation:
            model(**pars)

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
        self, regularizer, expectation, instantiate_base_regressor_subclass
    ):
        """
        Test initialization with different regularizer types.
        Test that an error is raised if a non-compatible regularizer is passed.
        """
        fixture = instantiate_base_regressor_subclass
        model_cls = fixture.model.__class__
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="Unused parameter `regularizer_strength`.*",
            )
            with expectation:
                model_cls(
                    **DEFAULTS[model_cls.__name__],
                    regularizer=regularizer,
                    regularizer_strength=1,
                )

    def test_get_params(self, instantiate_base_regressor_subclass):
        """
        Test that get_params() contains expected values.
        """
        fixture = instantiate_base_regressor_subclass
        model_cls = fixture.model.__class__

        expected_keys = HARD_CODED_GET_PARAMS_KEYS[model_cls.__name__]
        model = model_cls(**DEFAULTS[model_cls.__name__])
        expected_values = {
            par_name: getattr(model, par_name) for par_name in expected_keys
        }
        actual_values = model.get_params()
        assert set(model.get_params().keys()) == expected_keys
        assert all(np.all(actual_values[k] == v) for k, v in expected_values.items())

        # passing params
        model = model_cls(
            **DEFAULTS[model_cls.__name__],
            solver_name="LBFGS",
            regularizer="UnRegularized",
        )
        expected_values = {
            par_name: getattr(model, par_name) for par_name in expected_keys
        }
        actual_values = model.get_params()
        assert set(model.get_params().keys()) == expected_keys
        assert all(np.all(actual_values[k] == v) for k, v in expected_values.items())

        # changing regularizer
        model.set_params(
            **DEFAULTS[model_cls.__name__],
            regularizer="Ridge",
            regularizer_strength=1.0,
        )
        expected_values = {
            par_name: getattr(model, par_name) for par_name in expected_keys
        }
        actual_values = model.get_params()
        assert set(model.get_params().keys()) == expected_keys
        assert all(np.all(actual_values[k] == v) for k, v in expected_values.items())

        # changing solver
        model.solver_name = "ProximalGradient"
        expected_values = {
            par_name: getattr(model, par_name) for par_name in expected_keys
        }
        actual_values = model.get_params()
        assert set(model.get_params().keys()) == expected_keys
        assert all(np.all(actual_values[k] == v) for k, v in expected_values.items())

    @pytest.mark.parametrize(
        "n_params",
        [0, 1, 2, 3, 4],
    )
    @pytest.mark.solver_related
    def test_initialize_solver_param_length(
        self, n_params, instantiate_base_regressor_subclass
    ):
        """
        Test the `initialize_solver` method with different numbers of initial parameters.
        Check for correct number of parameters.
        """
        fixture = instantiate_base_regressor_subclass
        X, model, true_params = fixture.X, fixture.model, fixture.params

        model_name = model.__class__.__name__
        y = np.zeros(DEFAULT_OBS_SHAPE[model_name])
        expectation = (
            pytest.raises(
                ValueError,
                match="Params must have length.",
            )
            if n_params != INIT_PARAM_LENGTH[model_name]
            else does_not_raise()
        )

        if n_params < INIT_PARAM_LENGTH[model_name]:
            # Convert GLMParams to tuple for slicing
            params_tuple = (true_params.coef, true_params.intercept)
            init_params = params_tuple[:n_params]
        else:
            # Convert GLMParams to tuple for concatenation
            params_tuple = (true_params.coef, true_params.intercept)
            init_params = params_tuple + (true_params.coef,) * (
                n_params - INIT_PARAM_LENGTH[model_name]
            )
        with expectation:
            params = model._initialize_params(X, y, init_params=init_params)
            # check that params are set
            init_state = model._initialize_solver_and_state(X, y, params)
            # optimistix solvers do not have a velocity attr
            assert getattr(init_state, "velocity", params) == params

    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (-1, pytest.raises(ValueError, match="X must be 2-dimensional\\.")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="X must be 2-dimensional\\.")),
        ],
    )
    @pytest.mark.solver_related
    def test_initialize_solver_x_dimensionality(
        self, delta_dim, expectation, instantiate_base_regressor_subclass
    ):
        """
        Test the `initialize_solver` method with X input data of different dimensionalities.

        Ensure correct dimensionality for X.
        """
        fixture = instantiate_base_regressor_subclass
        X, model, true_params = fixture.X, fixture.model, fixture.params
        y = np.zeros(DEFAULT_OBS_SHAPE[model.__class__.__name__])
        if delta_dim == -1:
            X = np.zeros((X.shape[0],))
        elif delta_dim == 1:
            X = np.zeros((X.shape[0], 1, X.shape[1]))
        with expectation:
            model._validator.validate_inputs(X, y)
            params = model._validator.validate_and_cast(
                (true_params.coef, true_params.intercept)
            )
            # check that params are set
            init_state = model._initialize_solver_and_state(X, y, params)
            # optimistix solvers do not have a velocity attr
            assert getattr(init_state, "velocity", params) == params

    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (-1, pytest.raises(ValueError, match="y must be [12]-dimensional\\.")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="y must be [12]-dimensional\\.")),
        ],
    )
    @pytest.mark.solver_related
    def test_initialize_solver_y_dimensionality(
        self, delta_dim, expectation, instantiate_base_regressor_subclass
    ):
        """
        Test the `initialize_solver` method with y target data of different dimensionalities.

        Ensure correct dimensionality for y.
        """
        fixture = instantiate_base_regressor_subclass
        X, model, true_params = fixture.X, fixture.model, fixture.params
        y = np.zeros(DEFAULT_OBS_SHAPE[model.__class__.__name__])
        if is_population_model(model):
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
            model._validator.validate_inputs(X, y)
            params = model._validator.validate_and_cast(
                (true_params.coef, true_params.intercept)
            )
            # check that params are set
            init_state = model._initialize_solver_and_state(X, y, params)
            # optimistix solvers do not have a velocity attr
            assert getattr(init_state, "velocity", params) == params

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
        self, delta_n_features, expectation, instantiate_base_regressor_subclass
    ):
        """
        Test the `initialize_solver` method for inconsistencies between data features and model's expectations.
        Ensure the number of features in X aligns.
        """
        fixture = instantiate_base_regressor_subclass
        X, model, true_params = fixture.X, fixture.model, fixture.params
        y = np.zeros(DEFAULT_OBS_SHAPE[model.__class__.__name__])
        if delta_n_features == 1:
            X = jnp.concatenate((X, jnp.zeros((X.shape[0], 1))), axis=1)
        elif delta_n_features == -1:
            X = X[..., :-1]
        with expectation:
            params = model._initialize_params(
                X, y, init_params=(true_params.coef, true_params.intercept)
            )
            # check that params are set
            init_state = model._initialize_solver_and_state(X, y, params)
            # optimistix solvers do not have a velocity attr
            assert getattr(init_state, "velocity", params) == params

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
    @pytest.mark.solver_related
    def test_initialize_solver_time_points_x(
        self, delta_tp, expectation, instantiate_base_regressor_subclass
    ):
        """
        Test the `initialize_solver` method for inconsistencies in time-points in data X.

        Ensure the correct number of time-points.
        """
        fixture = instantiate_base_regressor_subclass
        X, model, true_params = fixture.X, fixture.model, fixture.params
        y = np.zeros(DEFAULT_OBS_SHAPE[model.__class__.__name__])
        X = jnp.zeros((X.shape[0] + delta_tp,) + X.shape[1:])
        with expectation:
            params = model._initialize_params(
                X, y, init_params=(true_params.coef, true_params.intercept)
            )
            model._validator.validate_inputs(X, y)
            # check that params are set
            init_state = model._initialize_solver_and_state(X, y, params)
            # optimistix solvers do not have a velocity attr
            assert getattr(init_state, "velocity", params) == params

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
    @pytest.mark.solver_related
    def test_initialize_solver_time_points_y(
        self, delta_tp, expectation, instantiate_base_regressor_subclass
    ):
        """
        Test the `initialize_solver` method for inconsistencies in time-points in y.

        Ensure the correct number of time-points.
        """
        fixture = instantiate_base_regressor_subclass
        X, y, model, true_params = fixture.X, fixture.y, fixture.model, fixture.params
        shape = DEFAULT_OBS_SHAPE[model.__class__.__name__]
        y = jnp.zeros((shape[0] + delta_tp,) + shape[1:])
        with expectation:
            params = model._initialize_params(
                X, y, init_params=(true_params.coef, true_params.intercept)
            )
            model._validator.validate_inputs(X, y)
            # check that params are set
            init_state = model._initialize_solver_and_state(X, y, params)
            # optimistix solvers do not have a velocity attr
            assert getattr(init_state, "velocity", params) == params

    @pytest.mark.solver_related
    def test_initialize_solver_mask_grouplasso(
        self, instantiate_base_regressor_subclass
    ):
        """Test that the group lasso initialize_solver goes through"""
        fixture = instantiate_base_regressor_subclass
        X, model, params = fixture.X, fixture.model, fixture.params
        y = np.ones(DEFAULT_OBS_SHAPE[model.__class__.__name__])
        n_groups = 2
        n_features = X.shape[1]
        mask = np.ones((n_groups, n_features), dtype=float)
        mask[0, : n_features // 2] = 0
        mask[1, n_features // 2 :] = 0
        model.set_params(
            regularizer=nmo.regularizer.GroupLasso(mask=mask),
            solver_name="ProximalGradient",
            regularizer_strength=1.0,
        )
        params = model._initialize_params(X, y)
        init_state = model._initialize_solver_and_state(X, y, params)
        # optimistix solvers do not have a velocity attr
        assert getattr(init_state, "velocity", params) == params

    @pytest.mark.solver_related
    def test_fit_mask_grouplasso(self, instantiate_base_regressor_subclass):
        """Test that the group lasso fit goes through"""

        fixture = instantiate_base_regressor_subclass
        X, model = fixture.X, fixture.model
        y = np.ones(DEFAULT_OBS_SHAPE[model.__class__.__name__])
        n_groups = 2
        n_features = X.shape[1]
        mask = np.ones((n_groups, n_features), dtype=float)
        mask[0, : n_features // 2] = 0
        mask[1, n_features // 2 :] = 0
        model.set_params(
            regularizer=nmo.regularizer.GroupLasso(mask=mask),
            solver_name="ProximalGradient",
            regularizer_strength=1.0,
        )
        model.fit(X, y)

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
        self, fill_val, expectation, instantiate_base_regressor_subclass
    ):
        fixture = instantiate_base_regressor_subclass
        X, model, true_params = fixture.X, fixture.model, fixture.params
        y = np.ones(DEFAULT_OBS_SHAPE[model.__class__.__name__])
        X.fill(fill_val)
        with expectation:
            params = model._initialize_params(X, y)
            model._validator.validate_inputs(X, y)
            init_state = model._initialize_solver_and_state(X, y, params)
            # optimistix solvers do not have a velocity attr
            assert getattr(init_state, "velocity", params) == params

    @pytest.mark.parametrize("reg", ["Ridge", "Lasso", "GroupLasso", "ElasticNet"])
    def test_reg_strength_reset(self, reg, instantiate_base_regressor_subclass):
        fixture = instantiate_base_regressor_subclass
        model_cls = fixture.model.__class__
        model = model_cls(
            **DEFAULTS[model_cls.__name__], regularizer=reg, regularizer_strength=1.0
        )
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
    def test_reg_set_params(self, params, warns, instantiate_base_regressor_subclass):
        fixture = instantiate_base_regressor_subclass
        model_cls = fixture.model.__class__
        model = model_cls(**DEFAULTS[model_cls.__name__])
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
        self, params, warns, reg, instantiate_base_regressor_subclass
    ):
        fixture = instantiate_base_regressor_subclass
        model_cls = fixture.model.__class__
        model = model_cls(
            **DEFAULTS[model_cls.__name__], regularizer=reg, regularizer_strength=1
        )
        with warns:
            model.set_params(**params)

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
        self, params, warns, reg, instantiate_base_regressor_subclass
    ):
        fixture = instantiate_base_regressor_subclass
        model_cls = fixture.model.__class__
        model = model_cls(
            **DEFAULTS[model_cls.__name__], regularizer=reg, regularizer_strength=11
        )
        model.set_params(**params)
        assert model.regularizer_strength == (1.0, 0.5)

    ################################
    # Test model.initialize_solver #
    ################################
    @pytest.mark.solver_related
    def test_initializer_solver_set_solver_callable(
        self, instantiate_base_regressor_subclass
    ):
        fixture = instantiate_base_regressor_subclass
        X, model, true_params = fixture.X, fixture.model, fixture.params
        y = np.ones(DEFAULT_OBS_SHAPE[model.__class__.__name__])
        assert model.solver_init_state is None
        assert model.solver_update is None
        assert model.solver_run is None
        init_params = model._initialize_params(X, y)
        model._initialize_solver_and_state(X, y, init_params)
        assert callable(model.solver_init_state)
        assert callable(model.solver_update)
        assert callable(model.solver_run)


@pytest.mark.parametrize(
    "instantiate_base_regressor_subclass",
    INSTANTIATE_MODEL_ONLY_LINK,
    indirect=True,
)
class TestLinkFunctionModels:
    def test_non_differentiable_inverse_link(self, instantiate_base_regressor_subclass):
        fixture = instantiate_base_regressor_subclass
        model = fixture.model

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
            model.__class__(
                **DEFAULTS[model.__class__.__name__], inverse_link_function=non_diff
            )

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
        instantiate_base_regressor_subclass,
    ):
        """Check that the observation model initializes when a callable is passed."""
        raise_exception = not isinstance(link_function(1.0), (jnp.ndarray, float))
        fixture = instantiate_base_regressor_subclass
        model = fixture.model

        if raise_exception:
            with pytest.raises(
                ValueError,
                match="The `inverse_link_function` must handle scalar inputs correctly",
            ):
                model.set_params(
                    **DEFAULTS[model.__class__.__name__],
                    inverse_link_function=link_function,
                )
        else:
            model.set_params(
                **DEFAULTS[model.__class__.__name__],
                inverse_link_function=link_function,
            )

    @pytest.mark.parametrize(
        "link_function",
        [jnp.exp, np.exp, lambda x: 1 / x, sm.families.links.Log()],
    )
    def test_initialization_link_is_jax(
        self,
        link_function,
        instantiate_base_regressor_subclass,
    ):
        """Check that the observation model initializes when a callable is passed."""
        fixture = instantiate_base_regressor_subclass
        model = fixture.model

        raise_exception = isinstance(link_function, np.ufunc) | isinstance(
            link_function, sm.families.links.Link
        )
        if raise_exception:
            with pytest.raises(
                ValueError,
                match="The `inverse_link_function` must return a jax.numpy.ndarray",
            ):
                model.__class__(
                    **DEFAULTS[model.__class__.__name__],
                    inverse_link_function=link_function,
                )
        else:
            model.__class__(
                **DEFAULTS[model.__class__.__name__],
                inverse_link_function=link_function,
            )

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
        self, link_function, instantiate_base_regressor_subclass, expectation
    ):
        fixture = instantiate_base_regressor_subclass
        model_cls = fixture.model.__class__

        with expectation:
            model_cls(**DEFAULTS[model_cls.__name__]).set_params(
                inverse_link_function=link_function
            )

    @pytest.mark.parametrize("link_function", [jnp.exp, jax.nn.softplus, 1])
    def test_initialization_link_is_callable(
        self, link_function, instantiate_base_regressor_subclass
    ):
        """Check that the observation model initializes when a callable is passed."""
        fixture = instantiate_base_regressor_subclass
        model_cls = fixture.model.__class__
        raise_exception = not callable(link_function)
        if raise_exception:
            with pytest.raises(
                TypeError,
                match="The `inverse_link_function` function must be a Callable",
            ):
                model_cls(
                    **DEFAULTS[model_cls.__name__], inverse_link_function=link_function
                )
        else:
            model_cls(
                **DEFAULTS[model_cls.__name__], inverse_link_function=link_function
            )

    @pytest.mark.parametrize("link_function", [jnp.exp, jax.nn.softplus, 1])
    def test_initialization_link_is_callable_set_params(
        self, link_function, instantiate_base_regressor_subclass
    ):
        """Check that the observation model initializes when a callable is passed."""
        fixture = instantiate_base_regressor_subclass
        model_cls = fixture.model.__class__
        raise_exception = not callable(link_function)
        if raise_exception:
            with pytest.raises(
                TypeError,
                match="The `inverse_link_function` function must be a Callable",
            ):
                model_cls(**DEFAULTS[model_cls.__name__]).set_params(
                    inverse_link_function=link_function
                )
        else:
            model_cls(**DEFAULTS[model_cls.__name__]).set_params(
                inverse_link_function=link_function
            )

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
    def test_link_func_from_string(
        self, link_func_string, expectation, instantiate_base_regressor_subclass
    ):
        fixture = instantiate_base_regressor_subclass
        model_cls = fixture.model.__class__
        with expectation:
            model_cls(
                **DEFAULTS[model_cls.__name__], inverse_link_function=link_func_string
            )


@pytest.mark.parametrize(
    "instantiate_base_regressor_subclass",
    [
        {"model": m, "obs_model": "Poisson", "simulate": False}
        for m in MODEL_REGISTRY.keys()
    ],
    indirect=True,
)
class TestObservationModel:
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
        self, observation, expectation, instantiate_base_regressor_subclass
    ):
        """
        Test initialization with different observation models. Check if an appropriate exception is raised
        when the observation model does not have the required attributes.
        """
        fixture = instantiate_base_regressor_subclass
        model_cls = fixture.model.__class__
        with expectation:
            model_cls(**DEFAULTS[model_cls.__name__], observation_model=observation)


@pytest.mark.parametrize(
    "instantiate_base_regressor_subclass",
    [
        {"model": m, "obs_model": "Poisson", "simulate": True}
        for m in MODEL_REGISTRY.keys()
    ],
    indirect=True,
)
class TestModelSimulation:

    @pytest.mark.parametrize(
        "n_params",
        [0, 1, 2, 3, 4],
    )
    @pytest.mark.solver_related
    def test_fit_param_length(self, n_params, instantiate_base_regressor_subclass):
        """
        Test the `fit` method with different numbers of initial parameters.
        Check for correct number of parameters.
        """
        fixture = instantiate_base_regressor_subclass
        X, y, model, true_params = fixture.X, fixture.y, fixture.model, fixture.params
        model_name = model.__class__.__name__
        expectation = (
            pytest.raises(
                ValueError,
                match="Params must have length.|GLM-HMM requires three parameters",
            )
            if n_params != INIT_PARAM_LENGTH[model_name]
            else does_not_raise()
        )

        if n_params < INIT_PARAM_LENGTH[model_name]:
            # Convert GLMParams to tuple for slicing
            params_tuple = (true_params.coef, true_params.intercept)
            init_params = params_tuple[:n_params]
        else:
            # Convert GLMParams to tuple for concatenation
            params_tuple = (true_params.coef, true_params.intercept)
            init_params = params_tuple + (true_params.coef,) * (
                n_params - INIT_PARAM_LENGTH[model_name]
            )
        with expectation:
            model.fit(X, y, init_params=init_params)

    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (-1, pytest.raises(ValueError, match="X must be 2-dimensional\\.")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="X must be 2-dimensional\\.")),
        ],
    )
    @pytest.mark.solver_related
    def test_fit_x_dimensionality(
        self, delta_dim, expectation, instantiate_base_regressor_subclass
    ):
        """
        Test the `fit` method with X input data of different dimensionalities. Ensure correct dimensionality for X.
        """
        fixture = instantiate_base_regressor_subclass
        X, y, model, true_params = fixture.X, fixture.y, fixture.model, fixture.params
        if delta_dim == -1:
            X = np.zeros((X.shape[0],))
        elif delta_dim == 1:
            X = np.zeros((X.shape[0], 1, X.shape[1]))
        with expectation:
            model.fit(X, y, init_params=(true_params.coef, true_params.intercept))

    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (-1, pytest.raises(ValueError, match=r"y must be [12]-dimensional\.")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match=r"y must be [12]-dimensional\.")),
        ],
    )
    @pytest.mark.solver_related
    def test_fit_y_dimensionality(
        self, delta_dim, expectation, instantiate_base_regressor_subclass
    ):
        """
        Test the `fit` method with y target data of different dimensionalities. Ensure correct dimensionality for y.
        """
        fixture = instantiate_base_regressor_subclass
        X, y, model, true_params = fixture.X, fixture.y, fixture.model, fixture.params
        if is_population_model(model):
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
            model.fit(X, y, init_params=(true_params.coef, true_params.intercept))

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
        instantiate_base_regressor_subclass,
    ):
        """
        Test the `fit` method for inconsistencies between data features and model's expectations.
        Ensure the number of features in X aligns.
        """
        fixture = instantiate_base_regressor_subclass
        X, y, model, true_params = fixture.X, fixture.y, fixture.model, fixture.params
        if delta_n_features == 1:
            X = jnp.concatenate((X, jnp.zeros((X.shape[0], 1))), axis=1)
        elif delta_n_features == -1:
            X = X[..., :-1]
        with expectation:
            model.fit(X, y, init_params=(true_params.coef, true_params.intercept))

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
    @pytest.mark.solver_related
    def test_fit_time_points_x(
        self, delta_tp, expectation, instantiate_base_regressor_subclass
    ):
        """
        Test the `fit` method for inconsistencies in time-points in data X. Ensure the correct number of time-points.
        """
        fixture = instantiate_base_regressor_subclass
        X, y, model, true_params = fixture.X, fixture.y, fixture.model, fixture.params
        X = jnp.zeros((X.shape[0] + delta_tp,) + X.shape[1:])
        with expectation:
            model.fit(X, y, init_params=(true_params.coef, true_params.intercept))

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
    @pytest.mark.solver_related
    def test_fit_time_points_y(
        self, delta_tp, expectation, instantiate_base_regressor_subclass
    ):
        """
        Test the `fit` method for inconsistencies in time-points in y. Ensure the correct number of time-points.
        """
        fixture = instantiate_base_regressor_subclass
        X, y, model, true_params = fixture.X, fixture.y, fixture.model, fixture.params
        y = jnp.zeros((y.shape[0] + delta_tp,) + y.shape[1:])
        with expectation:
            model.fit(X, y, init_params=(true_params.coef, true_params.intercept))

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
        self, fill_val, expectation, instantiate_base_regressor_subclass
    ):
        fixture = instantiate_base_regressor_subclass
        X, y, model, true_params = fixture.X, fixture.y, fixture.model, fixture.params
        X.fill(fill_val)
        with expectation:
            model.fit(X, y)
