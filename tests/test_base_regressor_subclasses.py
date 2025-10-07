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
from numba import njit

import nemos as nmo
from nemos._observation_model_builder import AVAILABLE_OBSERVATION_MODELS

MODEL_REGISTRY = {
    "GLM": nmo.glm.GLM,
    "PopulationGLM": nmo.glm.PopulationGLM,
    "GLMHMM": nmo.glm_hmm.GLMHMM,
}

INIT_PARAM_LENGTH = {
    "GLM": 2,
    "PopulationGLM": 2,
    "GLMHMM": 3,
}

DEFAULT_OBS_SHAPE = {
    "GLM": (500,),
    "PopulationGLM": (500, 3),
    "GLMHMM": (500,),
}

HARD_CODED_GET_PARAMS_KEYS = {
    "GLMHMM": {
        "dirichlet_prior_alphas_init_prob",
        "dirichlet_prior_alphas_transition",
        "initialize_glm_params",
        "initialize_init_proba",
        "initialize_transition_proba",
        "inverse_link_function",
        "maxiter",
        "n_states",
        "observation_model",
        "regularizer",
        "regularizer_strength",
        "seed",
        "solver_kwargs",
        "solver_name",
        "tol",
    },
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
    "GLMHMM": nmo.glm_hmm.GLMHMM,
}

DEFAULTS = {"GLMHMM": dict(n_states=3), "GLM": dict(), "PopulationGLM": dict()}


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
        model = instantiate_base_regressor_subclass[2].__class__
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
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="Unused parameter `regularizer_strength`.*",
            )
            model_cls = instantiate_base_regressor_subclass[2].__class__
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
        model_cls = instantiate_base_regressor_subclass[2].__class__

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
    def test_initialize_solver_param_length(
        self, n_params, instantiate_base_regressor_subclass
    ):
        """
        Test the `initialize_solver` method with different numbers of initial parameters.
        Check for correct number of parameters.
        """
        X, _, model, true_params = instantiate_base_regressor_subclass[:4]

        model_name = model.__class__.__name__
        y = np.zeros(DEFAULT_OBS_SHAPE[model_name])
        expectation = (
            pytest.raises(
                ValueError,
                match="Params must have length.|GLM-HMM requires three parameters",
            )
            if n_params != INIT_PARAM_LENGTH[model_name]
            else does_not_raise()
        )

        if n_params < INIT_PARAM_LENGTH[model_name]:
            init_params = true_params[:n_params]
        else:
            init_params = true_params + (true_params[0],) * (
                n_params - INIT_PARAM_LENGTH[model_name]
            )
        with expectation:
            params = model.initialize_params(X, y, init_params=init_params)
            # check that params are set
            init_state = model.initialize_state(X, y, params)
            assert init_state.velocity == params

    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (-1, pytest.raises(ValueError, match="X must be two-dimensional")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="X must be two-dimensional")),
        ],
    )
    def test_initialize_solver_x_dimensionality(
        self, delta_dim, expectation, instantiate_base_regressor_subclass
    ):
        """
        Test the `initialize_solver` method with X input data of different dimensionalities.

        Ensure correct dimensionality for X.
        """
        X, _, model, true_params = instantiate_base_regressor_subclass[:4]
        y = np.zeros(DEFAULT_OBS_SHAPE[model.__class__.__name__])
        if delta_dim == -1:
            X = np.zeros((X.shape[0],))
        elif delta_dim == 1:
            X = np.zeros((X.shape[0], 1, X.shape[1]))
        with expectation:
            params = model.initialize_params(X, y, init_params=true_params)
            # check that params are set
            init_state = model.initialize_state(X, y, params)
            assert init_state.velocity == params

    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (-1, pytest.raises(ValueError, match="y must be ...-dimensional")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="y must be ...-dimensional")),
        ],
    )
    def test_initialize_solver_y_dimensionality(
        self, delta_dim, expectation, instantiate_base_regressor_subclass
    ):
        """
        Test the `initialize_solver` method with y target data of different dimensionalities.

        Ensure correct dimensionality for y.
        """
        X, _, model, true_params = instantiate_base_regressor_subclass[:4]
        y = np.zeros(DEFAULT_OBS_SHAPE[model.__class__.__name__])
        if "Population" in model.__class__.__name__:
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
            assert init_state.velocity == params

    @pytest.mark.parametrize(
        "delta_n_features, expectation",
        [
            (-1, pytest.raises(ValueError, match="Inconsistent number of features")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="Inconsistent number of features")),
        ],
    )
    def test_initialize_solver_n_feature_consistency_x(
        self, delta_n_features, expectation, instantiate_base_regressor_subclass
    ):
        """
        Test the `initialize_solver` method for inconsistencies between data features and model's expectations.
        Ensure the number of features in X aligns.
        """
        X, _, model, true_params = instantiate_base_regressor_subclass[:4]
        y = np.zeros(DEFAULT_OBS_SHAPE[model.__class__.__name__])
        if delta_n_features == 1:
            X = jnp.concatenate((X, jnp.zeros((X.shape[0], 1))), axis=1)
        elif delta_n_features == -1:
            X = X[..., :-1]
        with expectation:
            params = model.initialize_params(X, y, init_params=true_params)
            # check that params are set
            init_state = model.initialize_state(X, y, params)
            assert init_state.velocity == params

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
    def test_initialize_solver_time_points_x(
        self, delta_tp, expectation, instantiate_base_regressor_subclass
    ):
        """
        Test the `initialize_solver` method for inconsistencies in time-points in data X.

        Ensure the correct number of time-points.
        """
        X, _, model, true_params = instantiate_base_regressor_subclass[:4]
        y = np.zeros(DEFAULT_OBS_SHAPE[model.__class__.__name__])
        X = jnp.zeros((X.shape[0] + delta_tp,) + X.shape[1:])
        with expectation:
            params = model.initialize_params(X, y, init_params=true_params)
            # check that params are set
            init_state = model.initialize_state(X, y, params)
            assert init_state.velocity == params

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
    def test_initialize_solver_time_points_y(
        self, delta_tp, expectation, instantiate_base_regressor_subclass
    ):
        """
        Test the `initialize_solver` method for inconsistencies in time-points in y.

        Ensure the correct number of time-points.
        """
        X, y, model, true_params = instantiate_base_regressor_subclass[:4]
        shape = DEFAULT_OBS_SHAPE[model.__class__.__name__]
        y = jnp.zeros((shape[0] + delta_tp,) + shape[1:])
        with expectation:
            params = model.initialize_params(X, y, init_params=true_params)
            # check that params are set
            init_state = model.initialize_state(X, y, params)
            assert init_state.velocity == params

    def test_initialize_solver_mask_grouplasso(
        self, instantiate_base_regressor_subclass
    ):
        """Test that the group lasso initialize_solver goes through"""
        X, _, model, params = instantiate_base_regressor_subclass[:4]
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
        params = model.initialize_params(X, y)
        init_state = model.initialize_state(X, y, params)
        assert init_state.velocity == params

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
    def test_initialize_solver_all_invalid_X(
        self, fill_val, expectation, instantiate_base_regressor_subclass
    ):
        X, _, model, true_params = instantiate_base_regressor_subclass[:4]
        y = np.ones(DEFAULT_OBS_SHAPE[model.__class__.__name__])
        X.fill(fill_val)
        with expectation:
            params = model.initialize_params(X, y)
            init_state = model.initialize_state(X, y, params)
            assert init_state.velocity == params


@pytest.mark.parametrize(
    "instantiate_base_regressor_subclass",
    INSTANTIATE_MODEL_ONLY_LINK,
    indirect=True,
)
class TestLinkFunctionModels:
    def test_non_differentiable_inverse_link(self, instantiate_base_regressor_subclass):
        model = instantiate_base_regressor_subclass[2]

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
        model = instantiate_base_regressor_subclass[2]

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
        model = instantiate_base_regressor_subclass[2]

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
        model_cls = instantiate_base_regressor_subclass[2].__class__

        with expectation:
            model_cls(**DEFAULTS[model_cls.__name__]).set_params(
                inverse_link_function=link_function
            )

    @pytest.mark.parametrize("link_function", [jnp.exp, jax.nn.softplus, 1])
    def test_initialization_link_is_callable(
        self, link_function, instantiate_base_regressor_subclass
    ):
        """Check that the observation model initializes when a callable is passed."""
        model_cls = instantiate_base_regressor_subclass[2].__class__
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
        model_cls = instantiate_base_regressor_subclass[2].__class__
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
        model_cls = instantiate_base_regressor_subclass[2].__class__
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
    def test_fit_param_length(self, n_params, instantiate_base_regressor_subclass):
        """
        Test the `fit` method with different numbers of initial parameters.
        Check for correct number of parameters.
        """
        X, y, model, true_params = instantiate_base_regressor_subclass[:4]
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
            init_params = true_params[:n_params]
        else:
            init_params = true_params + (true_params[0],) * (
                n_params - INIT_PARAM_LENGTH[model_name]
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
    def test_fit_x_dimensionality(
        self, delta_dim, expectation, instantiate_base_regressor_subclass
    ):
        """
        Test the `fit` method with X input data of different dimensionalities. Ensure correct dimensionality for X.
        """
        X, y, model, true_params = instantiate_base_regressor_subclass[:4]
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
    def test_fit_y_dimensionality(
        self, delta_dim, expectation, instantiate_base_regressor_subclass
    ):
        """
        Test the `fit` method with y target data of different dimensionalities. Ensure correct dimensionality for y.
        """
        X, y, model, true_params = instantiate_base_regressor_subclass[:4]
        if "Population" in model.__class__.__name__:
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
        X, y, model, true_params = instantiate_base_regressor_subclass[:4]
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
    def test_fit_time_points_x(
        self, delta_tp, expectation, instantiate_base_regressor_subclass
    ):
        """
        Test the `fit` method for inconsistencies in time-points in data X. Ensure the correct number of time-points.
        """
        X, y, model, true_params = instantiate_base_regressor_subclass[:4]
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
    def test_fit_time_points_y(
        self, delta_tp, expectation, instantiate_base_regressor_subclass
    ):
        """
        Test the `fit` method for inconsistencies in time-points in y. Ensure the correct number of time-points.
        """
        X, y, model, true_params = instantiate_base_regressor_subclass[:4]
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
    def test_fit_all_invalid_X(
        self, fill_val, expectation, instantiate_base_regressor_subclass
    ):
        X, y, model, true_params = instantiate_base_regressor_subclass[:4]
        X.fill(fill_val)
        with expectation:
            model.fit(X, y)
