from contextlib import nullcontext as does_not_raise
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import statsmodels.api as sm
from numba import njit
from test_base_regressor_subclasses import (
    INSTANTIATE_MODEL_AND_SIMULATE,
    INSTANTIATE_MODEL_ONLY,
)

import nemos as nmo
from nemos._observation_model_builder import AVAILABLE_OBSERVATION_MODELS
from nemos.typing import FeaturePytree
from nemos.utils import _get_name
from tests.conftest import instantiate_base_regressor_subclass

# FILTER FOR GLM HMM
INSTANTIATE_MODEL_ONLY = INSTANTIATE_MODEL_ONLY.copy()
INSTANTIATE_MODEL_ONLY = [v for v in INSTANTIATE_MODEL_ONLY if "GLMHMM" in v["model"]]
INSTANTIATE_MODEL_AND_SIMULATE = INSTANTIATE_MODEL_AND_SIMULATE.copy()
INSTANTIATE_MODEL_AND_SIMULATE = [
    v for v in INSTANTIATE_MODEL_AND_SIMULATE if "GLMHMM" in v["model"]
]

DEFAULT_GLM_COEF_SHAPE = {
    "GLMHMM": (2, 3),
}


@pytest.mark.parametrize(
    "instantiate_base_regressor_subclass",
    INSTANTIATE_MODEL_AND_SIMULATE,
    indirect=True,
)
def test_get_fit_attrs(instantiate_base_regressor_subclass):
    X, y, model, params = instantiate_base_regressor_subclass[:4]
    expected_state = {
        "coef_": None,
        "glm_params_": None,
        "initial_prob_": None,
        "intercept_": None,
        "transition_prob_": None,
    }
    assert model._get_fit_state() == expected_state
    model.solver_kwargs = {"maxiter": 1}
    model.fit(X, y)
    assert all(val is not None for val in model._get_fit_state().values())
    assert model._get_fit_state().keys() == expected_state.keys()


@pytest.mark.parametrize(
    "instantiate_base_regressor_subclass",
    INSTANTIATE_MODEL_AND_SIMULATE,
    indirect=True,
)
class TestGLMHMM:
    """
    Unit tests for the GLMHMM class that do not depend on the observation model.
    i.e. tests that do not call observation model methods, or tests that do not check the output when
    observation model methods are called (e.g. error testing for input validation)
    """

    # #######################
    # # Test model.fit
    # #######################

    @pytest.fixture
    def fit_weights_dimensionality_expectation(
        self, instantiate_base_regressor_subclass
    ):
        """
        Fixture to define the expected behavior for test_fit_weights_dimensionality based on the type of GLM class.
        """
        model_cls = instantiate_base_regressor_subclass[2].__class__
        if "Population" in model_cls.__name__:
            # FILL IN WHEN POPULATION CLASS IS DEFINED
            # NOTE THAT THE FIXTURE WILL MAKE TESTS FAIL FOR POPULATION, WHICH IS
            # ENOUGH TO REMIND US TO FILL THIS IN
            return {}
        else:
            return {
                0: pytest.raises(
                    ValueError,
                    match=r"params\[0\] \(GLM coefficients\) must be",
                ),
                1: pytest.raises(
                    ValueError,
                    match=r"params\[0\] \(GLM coefficients\) must be",
                ),
                2: does_not_raise(),
                3: pytest.raises(
                    ValueError,
                    match=r"params\[0\] \(GLM coefficients\) must be",
                ),
            }

    @pytest.mark.parametrize("dim_weights", [0, 1, 2, 3])
    def test_fit_weights_dimensionality(
        self,
        dim_weights,
        instantiate_base_regressor_subclass,
        fit_weights_dimensionality_expectation,
    ):
        """
        Test the `fit` method with weight matrices of different dimensionalities.
        Check for correct dimensionality.
        """
        expectation = fit_weights_dimensionality_expectation[dim_weights]
        X, y, model, true_params = instantiate_base_regressor_subclass[:4]
        n_samples, n_features = X.shape

        if dim_weights == 0:
            init_w = jnp.array([])
        elif dim_weights == 1:
            init_w = jnp.zeros((n_features,))
        elif dim_weights == 2:
            init_w = jnp.zeros(DEFAULT_GLM_COEF_SHAPE[model.__class__.__name__])
        else:
            init_w = jnp.zeros(DEFAULT_GLM_COEF_SHAPE[model.__class__.__name__] + (1,) * (dim_weights - 2))
        with expectation:
            model.fit(X, y, init_params=((init_w, true_params[0][1]), *true_params[1:]))

    @pytest.mark.parametrize(
        "dim_intercepts, expectation",
        [
            (0, pytest.raises(ValueError, match=r"params\[1\] \(GLM intercepts\) must be")),
            (1, does_not_raise()),
            (2, pytest.raises(ValueError, match=r"params\[1\] \(GLM intercepts\) must be")),
            (3, pytest.raises(ValueError, match=r"params\[1\] \(GLM intercepts\) must be")),
        ],
    )
    def test_fit_intercepts_dimensionality(
        self,
        dim_intercepts,
        expectation,
        instantiate_base_regressor_subclass
    ):
        """
        Test the `fit` method with intercepts of different dimensionalities. Check for correct dimensionality.
        """
        X, y, model, true_params = instantiate_base_regressor_subclass[:4]
        if dim_intercepts == 1:
            init_b = jnp.ones(DEFAULT_GLM_COEF_SHAPE[model.__class__.__name__][1])
        else:
            init_b = jnp.ones((1,) * dim_intercepts)
        with expectation:
            model.fit(X, y, init_params=((true_params[0][0], init_b), *true_params[1:]))

    """
    Parameterization used by test_fit_init_params_type and test_initialize_solver_init_params_type
    Contains the expected behavior and separate initial parameters for regular and population GLMs
    """
    fit_init_params_type_init_params = (
        "expectation, init_params_glm, init_params_population_glm",
        [
            (
                            does_not_raise(),
                            [jnp.zeros((2, 3)), jnp.zeros((3,))],
                            [jnp.zeros((2, 3, 3)), jnp.zeros((3, 3))],
                        ),
                        (
                            pytest.raises(ValueError, match="The GLM params must be a length"),
                            [[jnp.zeros((1, 2, 3)), jnp.zeros((3,))]],
                            [[jnp.zeros((1, 2, 3)), jnp.zeros((3, 3))]],
                        ),
                        (
                            pytest.raises(KeyError),
                            dict(p1=jnp.zeros((1, 3)), p2=jnp.zeros((1, 3))),
                            dict(p1=jnp.zeros((2, 3, 3)), p2=jnp.zeros((2, 2, 3))),
                        ),
                        (
                            pytest.raises(
                                ValueError, match=r"X and the GLM coefficients must be"
                            ),
                            [dict(p1=jnp.zeros((1, 3)), p2=jnp.zeros((1, 3))), jnp.zeros((3,))],
                            [dict(p1=jnp.zeros((1, 3, 3)), p2=jnp.zeros((1, 3, 3))), jnp.zeros((3, 3))],
                        ),
                        (
                            pytest.raises(
                                ValueError, match=r"X and the GLM coefficients must be"
                            ),
                            [
                                FeaturePytree(p1=jnp.zeros((1, 3)), p2=jnp.zeros((1, 3))),
                                jnp.zeros((3,)),
                            ],
                            [
                                FeaturePytree(p1=jnp.zeros((1, 3, 3)), p2=jnp.zeros((1, 2, 3))),
                                jnp.zeros((3, 3)),
                            ],
                        ),
                        (pytest.raises(ValueError, match="The GLM params must be a length"), 0, 0),
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
    def test_fit_init_glm_params_type(
        self,
        instantiate_base_regressor_subclass,
        expectation,
        init_params_glm,
        init_params_population_glm,
    ):
        """
        Test the `fit` method with various types of initial parameters. Ensure that the provided initial parameters
        are array-like.
        """
        X, y, model, true_params = instantiate_base_regressor_subclass[:4]
        if "Population" in model.__class__.__name__:
            init_params = init_params_population_glm
        else:
            init_params = init_params_glm

        with expectation:
            model.fit(X, y, init_params=(init_params, *true_params[1:]))

    @pytest.mark.parametrize(
        "delta_n_features, expectation",
        [
            (-1, pytest.raises(ValueError, match="Inconsistent number of features")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="Inconsistent number of features")),
        ],
    )
    def test_fit_n_feature_consistency_weights(
        self,
        delta_n_features,
        instantiate_base_regressor_subclass,
        expectation,
    ):
        """
        Test the `fit` method for inconsistencies between data features and initial weights provided.
        Ensure the number of features align.
        """
        X, y, model, true_params = instantiate_base_regressor_subclass[:4]
        if "Population" in model.__class__.__name__:
            raise RuntimeError("Fill in the test case for population glmhmm")
        else:
            init_w = jnp.zeros((X.shape[1] + delta_n_features, 3))
            init_b = jnp.ones(3,)
        with expectation:
            model.fit(X, y, init_params=((init_w, init_b), *true_params[1:]))

    @pytest.fixture
    def initialize_solver_weights_dimensionality_expectation(self, instantiate_base_regressor_subclass):
        name = instantiate_base_regressor_subclass[2].__class__.__name__
        if "Population" in name:
            return {
                0: pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
                1: pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
                2: pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
                3: does_not_raise(),
            }
        else:
            return {
                0: pytest.raises(
                    ValueError,
                    match=r"Inconsistent number of features",
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
    #
    @pytest.mark.parametrize("dim_weights", [0, 1, 2, 3])
    def test_initialize_solver_weights_dimensionality(
        self,
        dim_weights,
        instantiate_base_regressor_subclass,
        fit_weights_dimensionality_expectation,
    ):
        """
        Test the `initialize_solver` method with weight matrices of different dimensionalities.
        Check for correct dimensionality.
        """
        X, y, model, true_params = instantiate_base_regressor_subclass[:4]
        expectation = fit_weights_dimensionality_expectation[dim_weights]
        n_samples, n_features = X.shape

        if dim_weights == 0:
            init_w = jnp.array([])
        elif dim_weights == 1:
            init_w = jnp.zeros((n_features,))
        elif dim_weights == 2:
            init_w = jnp.zeros((n_features, 3))
        elif dim_weights == 3:
            init_w = jnp.zeros((n_features, y.shape[1] if y.ndim > 1 else 1, 3))
        else:
            init_w = jnp.zeros((n_features, 3) + (1,) * (dim_weights - 2))
        with expectation:
            params = model.initialize_params(X, y, init_params=((init_w, true_params[0][1]), *true_params[1:]))
            # check that params are set
            init_state = model.initialize_state(X, y, params)
            assert init_state.velocity == params

    @pytest.mark.parametrize(
        "dim_intercepts", [0,1,2,3],
    )
    def test_initialize_solver_intercepts_dimensionality(
        self,
        dim_intercepts,
        instantiate_base_regressor_subclass,
    ):
        """
        Test the `initialize_solver` method with intercepts of different dimensionalities.

        Check for correct dimensionality.
        """
        X, y, model, true_params = instantiate_base_regressor_subclass[:4]
        n_samples, n_features = X.shape
        is_population = "Population" in model.__class__.__name__
        if (dim_intercepts == 2 and is_population) or (dim_intercepts == 1 and not is_population):
            expectation = does_not_raise()
        else:
            expectation = pytest.raises(ValueError, match=r"params\[1\] \(GLM intercepts\) must be")
        if is_population:
            raise RuntimeError("Fill in the test case for population glmhmm")
        else:
            init_b = jnp.zeros((3,) + (1,) * (dim_intercepts - 1)) if dim_intercepts > 0 else jnp.array([])
            init_w = jnp.zeros((n_features, 3))
        with expectation:
            params = model.initialize_params(X, y, init_params=((init_w, init_b), *true_params[1:]))
            # check that params are set
            init_state = model.initialize_state(X, y, params)
            assert init_state.velocity == params

    @pytest.mark.parametrize(*fit_init_params_type_init_params)
    def test_initialize_solver_init_glm_params_type(
        self,
        instantiate_base_regressor_subclass,
        expectation,
        init_params_glm,
        init_params_population_glm,
    ):
        """
        Test the `initialize_solver` method with various types of initial parameters.

        Ensure that the provided initial parameters are array-like.
        """
        X, y, model, true_params = instantiate_base_regressor_subclass[:4]
        if "Population" in model.__class__.__name__:
            init_params = init_params_population_glm
        else:
            init_params = init_params_glm
        with expectation:
            params = model.initialize_params(X, y, init_params=(init_params, *true_params[1:]))
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
    def test_initialize_solver_n_feature_consistency_glm_coef(
        self,
        delta_n_features,
        expectation,
        instantiate_base_regressor_subclass,
    ):
        """
        Test the `initialize_solver` method for inconsistencies between data features and initial weights provided.
        Ensure the number of features align.
        """
        X, y, model, true_params = instantiate_base_regressor_subclass[:4]
        if "Population" in model.__class__.__name__:
            raise RuntimeError("Fill in the test case for population glmhmm")
        else:
            init_w = jnp.zeros((X.shape[1] + delta_n_features, 3))
            init_b = jnp.ones(3)
        with expectation:
            params = model.initialize_params(X, y, init_params=((init_w, init_b), *true_params[1:]))
            # check that params are set
            init_state = model.initialize_state(X, y, params)
            assert init_state.velocity == params

    # @pytest.mark.parametrize(
    #     "regularizer", ["Ridge", "UnRegularized", "Lasso", "ElasticNet"]
    # )
    # @pytest.mark.parametrize(
    #     "obs_model",
    #     [
    #         "PoissonObservations",
    #         "BernoulliObservations",
    #         "GammaObservations",
    #     ],
    # )
    # @pytest.mark.parametrize(
    #     "solver_name",
    #     [
    #         "GradientDescent",
    #         "BFGS",
    #         "LBFGS",
    #         "NonlinearCG",
    #         "ProximalGradient",
    #         "SVRG",
    #         "ProxSVRG",
    #     ],
    # )
    # @pytest.mark.parametrize(
    #     "model_class, fit_state_attrs",
    #     [
    #         (
    #             nmo.glm.GLM,
    #             {
    #                 "coef_": jnp.zeros(
    #                     3,
    #                 ),
    #                 "intercept_": jnp.array([1.0]),
    #                 "scale_": 2.0,
    #                 "dof_resid_": 3,
    #             },
    #         ),
    #         (
    #             nmo.glm.PopulationGLM,
    #             {
    #                 "coef_": jnp.zeros((3, 1)),
    #                 "intercept_": jnp.array([1.0]),
    #                 "scale_": 2.0,
    #                 "dof_resid_": 3,
    #             },
    #         ),
    #     ],
    # )
    # def test_save_and_load(
    #     self,
    #     regularizer,
    #     obs_model,
    #     solver_name,
    #     tmp_path,
    #     glm_class_type,
    #     fit_state_attrs,
    #     model_class,
    # ):
    #     """
    #     Test saving and loading a model with various observation models and regularizers.
    #     Ensure all parameters are preserved.
    #     """
    #     if (
    #         regularizer == "Lasso"
    #         or regularizer == "GroupLasso"
    #         or regularizer == "ElasticNet"
    #         and solver_name not in ["ProximalGradient", "ProxSVRG"]
    #     ):
    #         pytest.skip(
    #             f"Skipping {solver_name} for Lasso type regularizer; not an approximate solver."
    #         )
    #
    #     kwargs = dict(
    #         observation_model=obs_model,
    #         solver_name=solver_name,
    #         regularizer=regularizer,
    #         regularizer_strength=2.0,
    #         solver_kwargs={"tol": 10**-6},
    #     )
    #
    #     if regularizer == "UnRegularized":
    #         kwargs.pop("regularizer_strength")
    #
    #     model = model_class(**kwargs)
    #
    #     initial_params = model.get_params()
    #     # set fit states
    #     for key, val in fit_state_attrs.items():
    #         setattr(model, key, val)
    #         initial_params[key] = val
    #
    #     # Save
    #     save_path = tmp_path / "test_model.npz"
    #     model.save_params(save_path)
    #
    #     # Load
    #     loaded_model = nmo.load_model(save_path)
    #     loaded_params = loaded_model.get_params()
    #     fit_state = loaded_model._get_fit_state()
    #     fit_state.pop("solver_state_")
    #     loaded_params.update(fit_state)
    #
    #     # Assert matching keys and values
    #     assert (
    #         initial_params.keys() == loaded_params.keys()
    #     ), "Parameter keys mismatch after load."
    #
    #     for key in initial_params:
    #         init_val = initial_params[key]
    #         load_val = loaded_params[key]
    #         if isinstance(init_val, (int, float, str, type(None))):
    #             assert init_val == load_val, f"{key} mismatch: {init_val} != {load_val}"
    #         elif isinstance(init_val, dict):
    #             assert (
    #                 init_val == load_val
    #             ), f"{key} dict mismatch: {init_val} != {load_val}"
    #         elif isinstance(init_val, (np.ndarray, jnp.ndarray)):
    #             assert np.allclose(
    #                 np.array(init_val), np.array(load_val)
    #             ), f"{key} array mismatch"
    #         elif isinstance(init_val, Callable):
    #             assert _get_name(init_val) == _get_name(
    #                 load_val
    #             ), f"{key} function mismatch: {_get_name(init_val)} != {_get_name(load_val)}"
    #
    # @pytest.mark.parametrize("regularizer", ["Ridge"])
    # @pytest.mark.parametrize(
    #     "obs_model",
    #     [
    #         "PoissonObservations",
    #     ],
    # )
    # @pytest.mark.parametrize(
    #     "solver_name",
    #     [
    #         "ProxSVRG",
    #     ],
    # )
    # @pytest.mark.parametrize(
    #     "model_class, fit_state_attrs",
    #     [
    #         (
    #             nmo.glm.GLM,
    #             {
    #                 "coef_": jnp.zeros(
    #                     3,
    #                 ),
    #                 "intercept_": jnp.array([1.0]),
    #                 "scale_": 2.0,
    #                 "dof_resid_": 3,
    #             },
    #         ),
    #         (
    #             nmo.glm.PopulationGLM,
    #             {
    #                 "coef_": jnp.zeros(
    #                     (3, 1),
    #                 ),
    #                 "intercept_": jnp.array([1.0]),
    #                 "scale_": 2.0,
    #                 "dof_resid_": 3,
    #             },
    #         ),
    #     ],
    # )
    # @pytest.mark.parametrize(
    #     "mapping_dict, expectation",
    #     [
    #         ({}, does_not_raise()),
    #         (
    #             {
    #                 "observation_model": nmo.observation_models.GammaObservations,
    #                 "regularizer": nmo.regularizer.Lasso,
    #                 "inverse_link_function": lambda x: x**2,
    #             },
    #             pytest.warns(
    #                 UserWarning, match="The following keys have been replaced"
    #             ),
    #         ),
    #         (
    #             {
    #                 "observation_model": nmo.observation_models.GammaObservations(),  # fails, only class or callable
    #                 "regularizer": nmo.regularizer.Lasso,
    #                 "inverse_link_function": lambda x: x**2,
    #             },
    #             pytest.raises(ValueError, match="Invalid map parameter types detected"),
    #         ),
    #         (
    #             {
    #                 "observation_model": "GammaObservations",  # fails, only class or callable
    #                 "regularizer": nmo.regularizer.Lasso,
    #             },
    #             pytest.raises(ValueError, match="Invalid map parameter types detected"),
    #         ),
    #         (
    #             {
    #                 "regularizer": nmo.regularizer.Lasso,
    #                 "regularizer_strength": 3.0,  # fails, only class or callable
    #             },
    #             pytest.raises(ValueError, match="Invalid map parameter types detected"),
    #         ),
    #         (
    #             {
    #                 "solver_kwargs": {"tol": 10**-1},
    #             },
    #             pytest.raises(ValueError, match="Invalid map parameter types detected"),
    #         ),
    #         (
    #             {
    #                 "some__nested__dictionary": {"tol": 10**-1},
    #             },
    #             pytest.raises(
    #                 ValueError,
    #                 match="Invalid map parameter types detected",
    #             ),
    #         ),
    #         # valid mapping dtype, invalid name
    #         (
    #             {
    #                 "some__nested__dictionary": nmo.regularizer.Ridge,
    #             },
    #             pytest.raises(
    #                 ValueError,
    #                 match="The following keys in your mapping do not match ",
    #             ),
    #         ),
    #     ],
    # )
    # def test_save_and_load_with_custom_mapping(
    #     self,
    #     regularizer,
    #     obs_model,
    #     solver_name,
    #     mapping_dict,
    #     tmp_path,
    #     glm_class_type,
    #     fit_state_attrs,
    #     model_class,
    #     expectation,
    # ):
    #     """
    #     Test saving and loading a model with various observation models and regularizers.
    #     Ensure all parameters are preserved.
    #     """
    #
    #     if (
    #         regularizer == "Lasso"
    #         or regularizer == "GroupLasso"
    #         and solver_name not in ["ProximalGradient", "SVRG", "ProxSVRG"]
    #     ):
    #         pytest.skip(
    #             f"Skipping {solver_name} for Lasso type regularizer; not an approximate solver."
    #         )
    #
    #     model = model_class(
    #         observation_model=obs_model,
    #         solver_name=solver_name,
    #         regularizer=regularizer,
    #         regularizer_strength=2.0,
    #     )
    #
    #     initial_params = model.get_params()
    #     # set fit states
    #     for key, val in fit_state_attrs.items():
    #         setattr(model, key, val)
    #         initial_params[key] = val
    #
    #     # Save
    #     save_path = tmp_path / "test_model.npz"
    #     model.save_params(save_path)
    #
    #     # Load
    #     with expectation:
    #         loaded_model = nmo.load_model(save_path, mapping_dict=mapping_dict)
    #         loaded_params = loaded_model.get_params()
    #         fit_state = loaded_model._get_fit_state()
    #         fit_state.pop("solver_state_")
    #         loaded_params.update(fit_state)
    #
    #         # Assert matching keys and values
    #         assert (
    #             initial_params.keys() == loaded_params.keys()
    #         ), "Parameter keys mismatch after load."
    #
    #         unexpected_keys = set(mapping_dict) - set(initial_params)
    #         raise_exception = bool(unexpected_keys)
    #         if raise_exception:
    #             with pytest.raises(
    #                 ValueError, match="mapping_dict contains unexpected keys"
    #             ):
    #                 raise ValueError(
    #                     f"mapping_dict contains unexpected keys: {unexpected_keys}"
    #                 )
    #
    #         for key in initial_params:
    #             init_val = initial_params[key]
    #             load_val = loaded_params[key]
    #
    #             if key == "observation_model__inverse_link_function":
    #                 if "observation_model" in mapping_dict:
    #                     continue
    #             if key in mapping_dict:
    #                 if key == "observation_model":
    #                     if isinstance(mapping_dict[key], str):
    #                         mapping_obs = instantiate_observation_model(
    #                             mapping_dict[key]
    #                         )
    #                     else:
    #                         mapping_obs = mapping_dict[key]
    #                     assert _get_name(mapping_obs) == _get_name(
    #                         load_val
    #                     ), f"{key} observation model mismatch: {mapping_dict[key]} != {load_val}"
    #                 elif key == "regularizer":
    #                     if isinstance(mapping_dict[key], str):
    #                         mapping_reg = instantiate_regularizer(mapping_dict[key])
    #                     else:
    #                         mapping_reg = mapping_dict[key]
    #                     assert _get_name(mapping_reg) == _get_name(
    #                         load_val
    #                     ), f"{key} regularizer mismatch: {mapping_dict[key]} != {load_val}"
    #                 elif key == "solver_name":
    #                     assert (
    #                         mapping_dict[key] == load_val
    #                     ), f"{key} solver name mismatch: {mapping_dict[key]} != {load_val}"
    #                 elif key == "regularizer_strength":
    #                     assert (
    #                         mapping_dict[key] == load_val
    #                     ), f"{key} regularizer strength mismatch: {mapping_dict[key]} != {load_val}"
    #                 continue
    #
    #         if isinstance(init_val, (int, float, str, type(None))):
    #             assert init_val == load_val, f"{key} mismatch: {init_val} != {load_val}"
    #
    #         elif isinstance(init_val, dict):
    #             assert (
    #                 init_val == load_val
    #             ), f"{key} dict mismatch: {init_val} != {load_val}"
    #
    #         elif isinstance(init_val, (np.ndarray, jnp.ndarray)):
    #             assert np.allclose(
    #                 np.array(init_val), np.array(load_val)
    #             ), f"{key} array mismatch"
    #
    #         elif isinstance(init_val, Callable):
    #             assert _get_name(init_val) == _get_name(
    #                 load_val
    #             ), f"{key} function mismatch: {_get_name(init_val)} != {_get_name(load_val)}"
    #
    # def test_save_and_load_nested_class(
    #     self, nested_regularizer, tmp_path, glm_class_type
    # ):
    #     """Test that save and load works with nested classes."""
    #     model = nmo.glm.GLM(regularizer=nested_regularizer, regularizer_strength=1.0)
    #     save_path = tmp_path / "test_model.npz"
    #     model.save_params(save_path)
    #
    #     mapping_dict = {
    #         "regularizer": nested_regularizer.__class__,
    #         "regularizer__func": jnp.exp,
    #     }
    #     with pytest.warns(UserWarning, match="The following keys have been replaced"):
    #         loaded_model = nmo.load_model(save_path, mapping_dict=mapping_dict)
    #
    #     assert isinstance(loaded_model.regularizer, nested_regularizer.__class__)
    #     assert isinstance(
    #         loaded_model.regularizer.sub_regularizer,
    #         nested_regularizer.sub_regularizer.__class__,
    #     )
    #     assert loaded_model.regularizer.func == mapping_dict["regularizer__func"]
    #
    #     # change mapping
    #     mapping_dict = {
    #         "regularizer": nested_regularizer.__class__,
    #         "regularizer__sub_regularizer": nmo.regularizer.Ridge,
    #         "regularizer__func": lambda x: x**2,
    #     }
    #     with pytest.warns(UserWarning, match="The following keys have been replaced"):
    #         loaded_model = nmo.load_model(save_path, mapping_dict=mapping_dict)
    #     assert isinstance(loaded_model.regularizer, nested_regularizer.__class__)
    #     assert isinstance(
    #         loaded_model.regularizer.sub_regularizer, nmo.regularizer.Ridge
    #     )
    #     assert loaded_model.regularizer.func == mapping_dict["regularizer__func"]
    #
    # @pytest.mark.parametrize(
    #     "fitted_glm_type",
    #     [
    #         "poissonGLM_fitted_model_instantiation",
    #         "population_poissonGLM_fitted_model_instantiation",
    #     ],
    # )
    # def test_save_and_load_fitted_model(
    #     self, request, fitted_glm_type, glm_class_type, tmp_path
    # ):
    #     """
    #     Test saving and loading a fitted model with various observation models and regularizers.
    #     Ensure all parameters are preserved.
    #     """
    #     _, _, fitted_model, _, _ = request.getfixturevalue(fitted_glm_type)
    #
    #     initial_params = fitted_model.get_params()
    #     fit_state = fitted_model._get_fit_state()
    #     fit_state.pop("solver_state_")
    #     initial_params.update(fit_state)
    #
    #     # Save
    #     save_path = tmp_path / "test_model.npz"
    #     fitted_model.save_params(save_path)
    #
    #     # Load
    #     loaded_model = nmo.load_model(save_path)
    #     loaded_params = loaded_model.get_params()
    #     fit_state = loaded_model._get_fit_state()
    #     fit_state.pop("solver_state_")
    #     loaded_params.update(fit_state)
    #
    #     # Assert states are close
    #     for k, v in fit_state.items():
    #         assert np.allclose(initial_params[k], v), f"{k} mismatch after load."
    #
    # @pytest.mark.parametrize(
    #     "fitted_glm_type",
    #     [
    #         "poissonGLM_fitted_model_instantiation",
    #         "population_poissonGLM_fitted_model_instantiation",
    #     ],
    # )
    # @pytest.mark.parametrize(
    #     "param_name, param_value, expectation",
    #     [
    #         # Replace observation model class name  with a string
    #         (
    #             "observation_model__class",
    #             "InvalidObservations",
    #             pytest.raises(
    #                 ValueError, match="The class '[A-z]+' is not a native NeMoS"
    #             ),
    #         ),
    #         # Full path string
    #         (
    #             "observation_model__class",
    #             "nemos.observation_models.InvalidObservations",
    #             pytest.raises(
    #                 ValueError, match="The class '[A-z]+' is not a native NeMoS"
    #             ),
    #         ),
    #         # Replace observation model class name  with an instance
    #         (
    #             "observation_model__class",
    #             nmo.observation_models.GammaObservations(),
    #             pytest.raises(
    #                 ValueError,
    #                 match="Object arrays cannot be loaded when allow_pickle=False",
    #             ),
    #         ),
    #         # Replace observation model class name with class
    #         (
    #             "observation_model__class",
    #             nmo.observation_models.GammaObservations,
    #             pytest.raises(
    #                 ValueError,
    #                 match="Object arrays cannot be loaded when allow_pickle=False",
    #             ),
    #         ),
    #         # Replace link function with another callable
    #         (
    #             "observation_model__params__inverse_link_function",
    #             np.exp,
    #             pytest.raises(
    #                 ValueError,
    #                 match="Object arrays cannot be loaded when allow_pickle=False",
    #             ),
    #         ),
    #         # Unexpected dtype for class name
    #         (
    #             "regularizer__class",
    #             1,
    #             pytest.raises(
    #                 ValueError, match="Parameter ``regularizer`` cannot be initialized"
    #             ),
    #         ),
    #         # Invalid fit parameter
    #         (
    #             "scales_",  # wrong name for the params
    #             1,
    #             pytest.raises(ValueError, match="Unrecognized attribute 'scales_'"),
    #         ),
    #     ],
    # )
    # def test_modified_saved_file_raises(
    #     self,
    #     param_name,
    #     param_value,
    #     expectation,
    #     glm_class_type,
    #     fitted_glm_type,
    #     request,
    #     tmp_path,
    # ):
    #     _, _, fitted_model, _, _ = request.getfixturevalue(fitted_glm_type)
    #     save_path = tmp_path / "test_model.npz"
    #     fitted_model.save_params(save_path)
    #     # load and edit
    #     data = np.load(save_path, allow_pickle=True)
    #     load_data = dict((k, v) for k, v in data.items())
    #     load_data[param_name] = param_value
    #     np.savez(save_path, **load_data, allow_pickle=True)
    #
    #     with expectation:
    #         nmo.load_model(save_path)
    #
    # @pytest.mark.parametrize(
    #     "fitted_glm_type",
    #     [
    #         "poissonGLM_fitted_model_instantiation",
    #         "population_poissonGLM_fitted_model_instantiation",
    #     ],
    # )
    # def test_key_suggestions(self, fitted_glm_type, request, glm_class_type, tmp_path):
    #     _, _, fitted_model, _, _ = request.getfixturevalue(fitted_glm_type)
    #     save_path = tmp_path / "test_model.npz"
    #     fitted_model.save_params(save_path)
    #
    #     invalid_mapping = {
    #         "regulsriaer": nmo.regularizer.Ridge,
    #         "observatino_mdels": nmo.observation_models.GammaObservations,
    #         "inv_link_function": jax.numpy.exp,
    #         "total_nonsense": jax.numpy.exp,
    #     }
    #     match = (
    #         r"The following keys in your mapping do not match any parameters in the loaded model:\n\n"
    #         r"\t- 'regulsriaer', did you mean 'regularizer'\?\n"
    #         r"\t- 'observatino_mdels', did you mean 'observation_model'\?\n"
    #         r"\t- 'inv_link_function', did you mean 'inverse_link_function'\?\n"
    #         r"\t- 'total_nonsense'\n"
    #     )
    #     with pytest.raises(ValueError, match=match):
    #         nmo.load_model(save_path, mapping_dict=invalid_mapping)
