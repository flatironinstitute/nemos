# from contextlib import nullcontext as does_not_raise
#
# import jax.numpy as jnp
# import numpy as np
# import pytest
# import statsmodels.api as sm
# from numba import njit
# from test_base_regressor_subclasses import (
#     INSTANTIATE_MODEL_AND_SIMULATE,
#     INSTANTIATE_MODEL_ONLY,
# )
# import nemos as nmo
# import jax
# from nemos.typing import FeaturePytree
# from typing import Callable
# from nemos.utils import _get_name
#
#
# from nemos._observation_model_builder import AVAILABLE_OBSERVATION_MODELS
#
#
# @pytest.mark.parametrize(
#     "instantiate_glm_hmm",
#     INSTANTIATE_MODEL_AND_SIMULATE,
#     indirect=True,
# )
# def test_get_fit_attrs(instantiate_glm_hmm):
#     X, y, model, params, rates, latents = instantiate_glm_hmm
#     expected_state = {
#         "coef_": None,
#         "glm_params_": None,
#         "initial_prob_": None,
#         "intercept_": None,
#         "transition_prob_": None,
#     }
#     assert model._get_fit_state() == expected_state
#     model.solver_kwargs = {"maxiter": 1}
#     model.fit(X, y)
#     assert all(val is not None for val in model._get_fit_state().values())
#     assert model._get_fit_state().keys() == expected_state.keys()
#
#
# @pytest.mark.parametrize(
#     "instantiate_glm_hmm",
#     INSTANTIATE_MODEL_ONLY,
#     indirect=True,
# )
# def test_validate_lower_dimensional_data_X(instantiate_glm_hmm):
#     """Test behavior with lower-dimensional input data."""
#     model = instantiate_glm_hmm[2]
#     X = jnp.array([1, 2])
#     y = jnp.array([1, 2])
#     with pytest.raises(ValueError, match="X must be two-dimensional"):
#         model._validate(X, y, model._initialize_parameters(X, y))
#
#
# @pytest.mark.parametrize(
#     "instantiate_glm_hmm",
#     INSTANTIATE_MODEL_ONLY,
#     indirect=True,
# )
# def test_preprocess_fit_higher_dimensional_data_y(instantiate_glm_hmm):
#     """Test behavior with higher-dimensional input data."""
#     model = instantiate_glm_hmm[2]
#     X = jnp.array([[[1, 2], [3, 4]]])
#     y = jnp.array([[[1, 2]]])
#     with pytest.raises(ValueError, match="y must be one-dimensional"):
#         model._validate(X, y, model._initialize_parameters(X, y))
#
#
# @pytest.mark.parametrize(
#     "instantiate_glm_hmm",
#     INSTANTIATE_MODEL_ONLY,
#     indirect=True,
# )
# def test_validate_higher_dimensional_data_X(instantiate_glm_hmm):
#     """Test behavior with higher-dimensional input data."""
#     model = instantiate_glm_hmm[2]
#     X = jnp.array([[[[1, 2], [3, 4]]]])
#     y = jnp.array([1, 2])
#     with pytest.raises(ValueError, match="X must be two-dimensional"):
#         model._validate(X, y, model._initialize_parameters(X, y))
#
#
# @pytest.mark.parametrize(
#     "instantiate_glm_hmm",
#     INSTANTIATE_MODEL_ONLY,
#     indirect=True,
# )
# class TestGLMHMM:
#     """
#     Unit tests for the GLMHMM class that do not depend on the observation model.
#     i.e. tests that do not call observation model methods, or tests that do not check the output when
#     observation model methods are called (e.g. error testing for input validation)
#     """
#
#     # #######################
#     # # Test model.fit
#     # #######################
#
#     @pytest.fixture
#     def fit_weights_dimensionality_expectation(self, glm_class_type):
#         """
#         Fixture to define the expected behavior for test_fit_weights_dimensionality based on the type of GLM class.
#         """
#         if "population" in glm_class_type:
#             return {
#                 0: pytest.raises(
#                     ValueError,
#                     match=r"params\[0\] must be an array or .* of shape \(n_features",
#                 ),
#                 1: pytest.raises(
#                     ValueError,
#                     match=r"params\[0\] must be an array or .* of shape \(n_features",
#                 ),
#                 2: does_not_raise(),
#                 3: pytest.raises(
#                     ValueError,
#                     match=r"params\[0\] must be an array or .* of shape \(n_features",
#                 ),
#             }
#         else:
#             return {
#                 0: pytest.raises(
#                     ValueError,
#                     match=r"Inconsistent number of features",
#                 ),
#                 1: does_not_raise(),
#                 2: pytest.raises(
#                     ValueError,
#                     match=r"params\[0\] must be an array or .* of shape \(n_features",
#                 ),
#                 3: pytest.raises(
#                     ValueError,
#                     match=r"params\[0\] must be an array or .* of shape \(n_features",
#                 ),
#             }
#
#     @pytest.mark.parametrize("dim_weights", [0, 1, 2, 3])
#     def test_fit_weights_dimensionality(
#         self,
#         dim_weights,
#         request,
#         glm_class_type,
#         model_instantiation_type,
#         fit_weights_dimensionality_expectation,
#     ):
#         """
#         Test the `fit` method with weight matrices of different dimensionalities.
#         Check for correct dimensionality.
#         """
#         expectation = fit_weights_dimensionality_expectation[dim_weights]
#         X, y, model, true_params, firing_rate = request.getfixturevalue(
#             model_instantiation_type
#         )
#         n_samples, n_features = X.shape
#         if "population" in glm_class_type:
#             n_neurons = 3
#         else:
#             n_neurons = 4
#         if dim_weights == 0:
#             init_w = jnp.array([])
#         elif dim_weights == 1:
#             init_w = jnp.zeros((n_features,))
#         elif dim_weights == 2:
#             init_w = jnp.zeros((n_features, n_neurons))
#         else:
#             init_w = jnp.zeros((n_features, n_neurons) + (1,) * (dim_weights - 2))
#         with expectation:
#             model.fit(X, y, init_params=(init_w, true_params[1]))
#
#     @pytest.mark.parametrize(
#         "dim_intercepts, expectation",
#         [
#             (0, pytest.raises(ValueError, match=r"params\[1\] must be of shape")),
#             (1, does_not_raise()),
#             (2, pytest.raises(ValueError, match=r"params\[1\] must be of shape")),
#             (3, pytest.raises(ValueError, match=r"params\[1\] must be of shape")),
#         ],
#     )
#     def test_fit_intercepts_dimensionality(
#         self,
#         dim_intercepts,
#         expectation,
#         request,
#         glm_class_type,
#         model_instantiation_type,
#     ):
#         """
#         Test the `fit` method with intercepts of different dimensionalities. Check for correct dimensionality.
#         """
#         X, y, model, true_params, firing_rate = request.getfixturevalue(
#             model_instantiation_type
#         )
#         n_samples, n_features = X.shape
#
#         if "population" in glm_class_type:
#             init_b = jnp.zeros((y.shape[1],) * dim_intercepts)
#             init_w = jnp.zeros((n_features, y.shape[1]))
#         else:
#             init_b = jnp.zeros((1,) * dim_intercepts)
#             init_w = jnp.zeros((n_features,))
#
#         with expectation:
#             model.fit(X, y, init_params=(init_w, init_b))
#
#     """
#     Parameterization used by test_fit_init_params_type and test_initialize_solver_init_params_type
#     Contains the expected behavior and separate initial parameters for regular and population GLMs
#     """
#     fit_init_params_type_init_params = (
#         "expectation, init_params_glm, init_params_population_glm",
#         [
#             (
#                 does_not_raise(),
#                 [jnp.zeros((5,)), jnp.zeros((1,))],
#                 [jnp.zeros((5, 3)), jnp.zeros((3,))],
#             ),
#             (
#                 pytest.raises(ValueError, match="Params must have length two."),
#                 [[jnp.zeros((1, 5)), jnp.zeros((1,))]],
#                 [[jnp.zeros((1, 5)), jnp.zeros((3,))]],
#             ),
#             (
#                 pytest.raises(KeyError),
#                 dict(p1=jnp.zeros((5,)), p2=jnp.zeros((1,))),
#                 dict(p1=jnp.zeros((3, 3)), p2=jnp.zeros((3, 2))),
#             ),
#             (
#                 pytest.raises(
#                     TypeError, match=r"X and params\[0\] must be the same type"
#                 ),
#                 [dict(p1=jnp.zeros((5,)), p2=jnp.zeros((1,))), jnp.zeros((1,))],
#                 [dict(p1=jnp.zeros((3, 3)), p2=jnp.zeros((2, 3))), jnp.zeros((3,))],
#             ),
#             (
#                 pytest.raises(
#                     TypeError, match=r"X and params\[0\] must be the same type"
#                 ),
#                 [
#                     FeaturePytree(p1=jnp.zeros((5,)), p2=jnp.zeros((5,))),
#                     jnp.zeros((1,)),
#                 ],
#                 [
#                     FeaturePytree(p1=jnp.zeros((3, 3)), p2=jnp.zeros((3, 2))),
#                     jnp.zeros((3,)),
#                 ],
#             ),
#             (pytest.raises(ValueError, match="Params must have length two."), 0, 0),
#             (
#                 pytest.raises(TypeError, match="Initial parameters must be array-like"),
#                 {0, 1},
#                 {0, 1},
#             ),
#             (
#                 pytest.raises(TypeError, match="Initial parameters must be array-like"),
#                 [jnp.zeros((1, 5)), ""],
#                 [jnp.zeros((1, 5)), ""],
#             ),
#             (
#                 pytest.raises(TypeError, match="Initial parameters must be array-like"),
#                 ["", jnp.zeros((1,))],
#                 ["", jnp.zeros((1,))],
#             ),
#         ],
#     )
#
#     @pytest.mark.parametrize(*fit_init_params_type_init_params)
#     def test_fit_init_params_type(
#         self,
#         request,
#         glm_class_type,
#         model_instantiation_type,
#         expectation,
#         init_params_glm,
#         init_params_population_glm,
#     ):
#         """
#         Test the `fit` method with various types of initial parameters. Ensure that the provided initial parameters
#         are array-like.
#         """
#         if "population" in glm_class_type:
#             init_params = init_params_population_glm
#         else:
#             init_params = init_params_glm
#         X, y, model, true_params, firing_rate = request.getfixturevalue(
#             model_instantiation_type
#         )
#         with expectation:
#             model.fit(X, y, init_params=init_params)
#
#     @pytest.mark.parametrize(
#         "delta_n_features, expectation",
#         [
#             (-1, pytest.raises(ValueError, match="Inconsistent number of features")),
#             (0, does_not_raise()),
#             (1, pytest.raises(ValueError, match="Inconsistent number of features")),
#         ],
#     )
#     def test_fit_n_feature_consistency_weights(
#         self,
#         delta_n_features,
#         expectation,
#         request,
#         glm_class_type,
#         model_instantiation_type,
#     ):
#         """
#         Test the `fit` method for inconsistencies between data features and initial weights provided.
#         Ensure the number of features align.
#         """
#         X, y, model, true_params, firing_rate = request.getfixturevalue(
#             model_instantiation_type
#         )
#         if "population" in glm_class_type:
#             init_w = jnp.zeros((X.shape[1] + delta_n_features, y.shape[1]))
#             init_b = jnp.zeros(
#                 y.shape[1],
#             )
#         else:
#             init_w = jnp.zeros((X.shape[1] + delta_n_features))
#             init_b = jnp.zeros(
#                 1,
#             )
#         with expectation:
#             model.fit(X, y, init_params=(init_w, init_b))
#
#     # ##############################
#     # # Test model.initialize_solver
#     # ##############################
#
#     @pytest.fixture
#     def initialize_solver_weights_dimensionality_expectation(self, glm_class_type):
#         if "population" in glm_class_type:
#             return {
#                 0: pytest.raises(
#                     ValueError,
#                     match=r"params\[0\] must be an array or .* of shape \(n_features",
#                 ),
#                 1: pytest.raises(
#                     ValueError,
#                     match=r"params\[0\] must be an array or .* of shape \(n_features",
#                 ),
#                 2: does_not_raise(),
#                 3: pytest.raises(
#                     ValueError,
#                     match=r"params\[0\] must be an array or .* of shape \(n_features",
#                 ),
#             }
#         else:
#             return {
#                 0: pytest.raises(
#                     ValueError,
#                     match=r"Inconsistent number of features",
#                 ),
#                 1: does_not_raise(),
#                 2: pytest.raises(
#                     ValueError,
#                     match=r"params\[0\] must be an array or .* of shape \(n_features",
#                 ),
#                 3: pytest.raises(
#                     ValueError,
#                     match=r"params\[0\] must be an array or .* of shape \(n_features",
#                 ),
#             }
#
#     @pytest.mark.parametrize("dim_weights", [0, 1, 2, 3])
#     def test_initialize_solver_weights_dimensionality(
#         self,
#         dim_weights,
#         request,
#         glm_class_type,
#         model_instantiation_type,
#         fit_weights_dimensionality_expectation,
#     ):
#         """
#         Test the `initialize_solver` method with weight matrices of different dimensionalities.
#         Check for correct dimensionality.
#         """
#         X, y, model, true_params, firing_rate = request.getfixturevalue(
#             model_instantiation_type
#         )
#         expectation = fit_weights_dimensionality_expectation[dim_weights]
#         n_samples, n_features = X.shape
#         if "population" in glm_class_type:
#             n_neurons = 3
#         else:
#             n_neurons = 4
#         if dim_weights == 0:
#             init_w = jnp.array([])
#         elif dim_weights == 1:
#             init_w = jnp.zeros((n_features,))
#         elif dim_weights == 2:
#             init_w = jnp.zeros((n_features, n_neurons))
#         else:
#             init_w = jnp.zeros((n_features, n_neurons) + (1,) * (dim_weights - 2))
#         with expectation:
#             params = model.initialize_params(X, y, init_params=(init_w, true_params[1]))
#             # check that params are set
#             init_state = model.initialize_state(X, y, params)
#             assert init_state.velocity == params
#
#     @pytest.mark.parametrize(
#         "dim_intercepts, expectation",
#         [
#             (0, pytest.raises(ValueError, match=r"params\[1\] must be of shape")),
#             (1, does_not_raise()),
#             (2, pytest.raises(ValueError, match=r"params\[1\] must be of shape")),
#             (3, pytest.raises(ValueError, match=r"params\[1\] must be of shape")),
#         ],
#     )
#     def test_initialize_solver_intercepts_dimensionality(
#         self,
#         dim_intercepts,
#         expectation,
#         request,
#         glm_class_type,
#         model_instantiation_type,
#     ):
#         """
#         Test the `initialize_solver` method with intercepts of different dimensionalities.
#
#         Check for correct dimensionality.
#         """
#         X, y, model, true_params, firing_rate = request.getfixturevalue(
#             model_instantiation_type
#         )
#         n_samples, n_features = X.shape
#         if "population" in glm_class_type:
#             init_b = jnp.zeros((y.shape[1],) * dim_intercepts)
#             init_w = jnp.zeros((n_features, y.shape[1]))
#         else:
#             init_b = jnp.zeros((1,) * dim_intercepts)
#             init_w = jnp.zeros((n_features,))
#         with expectation:
#             params = model.initialize_params(X, y, init_params=(init_w, init_b))
#             # check that params are set
#             init_state = model.initialize_state(X, y, params)
#             assert init_state.velocity == params
#
#     @pytest.mark.parametrize(*fit_init_params_type_init_params)
#     def test_initialize_solver_init_params_type(
#         self,
#         request,
#         glm_class_type,
#         model_instantiation_type,
#         expectation,
#         init_params_glm,
#         init_params_population_glm,
#     ):
#         """
#         Test the `initialize_solver` method with various types of initial parameters.
#
#         Ensure that the provided initial parameters are array-like.
#         """
#         X, y, model, true_params, firing_rate = request.getfixturevalue(
#             model_instantiation_type
#         )
#         if "population" in glm_class_type:
#             init_params = init_params_population_glm
#         else:
#             init_params = init_params_glm
#         with expectation:
#             params = model.initialize_params(X, y, init_params=init_params)
#             # check that params are set
#             init_state = model.initialize_state(X, y, params)
#             assert init_state.velocity == params
#
#     @pytest.mark.parametrize(
#         "delta_n_features, expectation",
#         [
#             (-1, pytest.raises(ValueError, match="Inconsistent number of features")),
#             (0, does_not_raise()),
#             (1, pytest.raises(ValueError, match="Inconsistent number of features")),
#         ],
#     )
#     def test_initialize_solver_n_feature_consistency_weights(
#         self,
#         delta_n_features,
#         expectation,
#         request,
#         glm_class_type,
#         model_instantiation_type,
#     ):
#         """
#         Test the `initialize_solver` method for inconsistencies between data features and initial weights provided.
#         Ensure the number of features align.
#         """
#         X, y, model, true_params, firing_rate = request.getfixturevalue(
#             model_instantiation_type
#         )
#         if "population" in glm_class_type:
#             init_w = jnp.zeros((X.shape[1] + delta_n_features, y.shape[1]))
#             init_b = jnp.zeros(
#                 y.shape[1],
#             )
#         else:
#             init_w = jnp.zeros((X.shape[1] + delta_n_features))
#             init_b = jnp.zeros(
#                 1,
#             )
#         with expectation:
#             params = model.initialize_params(X, y, init_params=(init_w, init_b))
#             # check that params are set
#             init_state = model.initialize_state(X, y, params)
#             assert init_state.velocity == params
#
#     # #######################################
#     # # Compare with standard implementation
#     # #######################################
#
#
#
#     @pytest.mark.parametrize(
#         "regularizer", ["Ridge", "UnRegularized", "Lasso", "ElasticNet"]
#     )
#     @pytest.mark.parametrize(
#         "obs_model",
#         [
#             "PoissonObservations",
#             "BernoulliObservations",
#             "GammaObservations",
#         ],
#     )
#     @pytest.mark.parametrize(
#         "solver_name",
#         [
#             "GradientDescent",
#             "BFGS",
#             "LBFGS",
#             "NonlinearCG",
#             "ProximalGradient",
#             "SVRG",
#             "ProxSVRG",
#         ],
#     )
#     @pytest.mark.parametrize(
#         "model_class, fit_state_attrs",
#         [
#             (
#                 nmo.glm.GLM,
#                 {
#                     "coef_": jnp.zeros(
#                         3,
#                     ),
#                     "intercept_": jnp.array([1.0]),
#                     "scale_": 2.0,
#                     "dof_resid_": 3,
#                 },
#             ),
#             (
#                 nmo.glm.PopulationGLM,
#                 {
#                     "coef_": jnp.zeros((3, 1)),
#                     "intercept_": jnp.array([1.0]),
#                     "scale_": 2.0,
#                     "dof_resid_": 3,
#                 },
#             ),
#         ],
#     )
#     def test_save_and_load(
#         self,
#         regularizer,
#         obs_model,
#         solver_name,
#         tmp_path,
#         glm_class_type,
#         fit_state_attrs,
#         model_class,
#     ):
#         """
#         Test saving and loading a model with various observation models and regularizers.
#         Ensure all parameters are preserved.
#         """
#         if (
#             regularizer == "Lasso"
#             or regularizer == "GroupLasso"
#             or regularizer == "ElasticNet"
#             and solver_name not in ["ProximalGradient", "ProxSVRG"]
#         ):
#             pytest.skip(
#                 f"Skipping {solver_name} for Lasso type regularizer; not an approximate solver."
#             )
#
#         kwargs = dict(
#             observation_model=obs_model,
#             solver_name=solver_name,
#             regularizer=regularizer,
#             regularizer_strength=2.0,
#             solver_kwargs={"tol": 10**-6},
#         )
#
#         if regularizer == "UnRegularized":
#             kwargs.pop("regularizer_strength")
#
#         model = model_class(**kwargs)
#
#         initial_params = model.get_params()
#         # set fit states
#         for key, val in fit_state_attrs.items():
#             setattr(model, key, val)
#             initial_params[key] = val
#
#         # Save
#         save_path = tmp_path / "test_model.npz"
#         model.save_params(save_path)
#
#         # Load
#         loaded_model = nmo.load_model(save_path)
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
#         for key in initial_params:
#             init_val = initial_params[key]
#             load_val = loaded_params[key]
#             if isinstance(init_val, (int, float, str, type(None))):
#                 assert init_val == load_val, f"{key} mismatch: {init_val} != {load_val}"
#             elif isinstance(init_val, dict):
#                 assert (
#                     init_val == load_val
#                 ), f"{key} dict mismatch: {init_val} != {load_val}"
#             elif isinstance(init_val, (np.ndarray, jnp.ndarray)):
#                 assert np.allclose(
#                     np.array(init_val), np.array(load_val)
#                 ), f"{key} array mismatch"
#             elif isinstance(init_val, Callable):
#                 assert _get_name(init_val) == _get_name(
#                     load_val
#                 ), f"{key} function mismatch: {_get_name(init_val)} != {_get_name(load_val)}"
#
#     @pytest.mark.parametrize("regularizer", ["Ridge"])
#     @pytest.mark.parametrize(
#         "obs_model",
#         [
#             "PoissonObservations",
#         ],
#     )
#     @pytest.mark.parametrize(
#         "solver_name",
#         [
#             "ProxSVRG",
#         ],
#     )
#     @pytest.mark.parametrize(
#         "model_class, fit_state_attrs",
#         [
#             (
#                 nmo.glm.GLM,
#                 {
#                     "coef_": jnp.zeros(
#                         3,
#                     ),
#                     "intercept_": jnp.array([1.0]),
#                     "scale_": 2.0,
#                     "dof_resid_": 3,
#                 },
#             ),
#             (
#                 nmo.glm.PopulationGLM,
#                 {
#                     "coef_": jnp.zeros(
#                         (3, 1),
#                     ),
#                     "intercept_": jnp.array([1.0]),
#                     "scale_": 2.0,
#                     "dof_resid_": 3,
#                 },
#             ),
#         ],
#     )
#     @pytest.mark.parametrize(
#         "mapping_dict, expectation",
#         [
#             ({}, does_not_raise()),
#             (
#                 {
#                     "observation_model": nmo.observation_models.GammaObservations,
#                     "regularizer": nmo.regularizer.Lasso,
#                     "inverse_link_function": lambda x: x**2,
#                 },
#                 pytest.warns(
#                     UserWarning, match="The following keys have been replaced"
#                 ),
#             ),
#             (
#                 {
#                     "observation_model": nmo.observation_models.GammaObservations(),  # fails, only class or callable
#                     "regularizer": nmo.regularizer.Lasso,
#                     "inverse_link_function": lambda x: x**2,
#                 },
#                 pytest.raises(ValueError, match="Invalid map parameter types detected"),
#             ),
#             (
#                 {
#                     "observation_model": "GammaObservations",  # fails, only class or callable
#                     "regularizer": nmo.regularizer.Lasso,
#                 },
#                 pytest.raises(ValueError, match="Invalid map parameter types detected"),
#             ),
#             (
#                 {
#                     "regularizer": nmo.regularizer.Lasso,
#                     "regularizer_strength": 3.0,  # fails, only class or callable
#                 },
#                 pytest.raises(ValueError, match="Invalid map parameter types detected"),
#             ),
#             (
#                 {
#                     "solver_kwargs": {"tol": 10**-1},
#                 },
#                 pytest.raises(ValueError, match="Invalid map parameter types detected"),
#             ),
#             (
#                 {
#                     "some__nested__dictionary": {"tol": 10**-1},
#                 },
#                 pytest.raises(
#                     ValueError,
#                     match="Invalid map parameter types detected",
#                 ),
#             ),
#             # valid mapping dtype, invalid name
#             (
#                 {
#                     "some__nested__dictionary": nmo.regularizer.Ridge,
#                 },
#                 pytest.raises(
#                     ValueError,
#                     match="The following keys in your mapping do not match ",
#                 ),
#             ),
#         ],
#     )
#     def test_save_and_load_with_custom_mapping(
#         self,
#         regularizer,
#         obs_model,
#         solver_name,
#         mapping_dict,
#         tmp_path,
#         glm_class_type,
#         fit_state_attrs,
#         model_class,
#         expectation,
#     ):
#         """
#         Test saving and loading a model with various observation models and regularizers.
#         Ensure all parameters are preserved.
#         """
#
#         if (
#             regularizer == "Lasso"
#             or regularizer == "GroupLasso"
#             and solver_name not in ["ProximalGradient", "SVRG", "ProxSVRG"]
#         ):
#             pytest.skip(
#                 f"Skipping {solver_name} for Lasso type regularizer; not an approximate solver."
#             )
#
#         model = model_class(
#             observation_model=obs_model,
#             solver_name=solver_name,
#             regularizer=regularizer,
#             regularizer_strength=2.0,
#         )
#
#         initial_params = model.get_params()
#         # set fit states
#         for key, val in fit_state_attrs.items():
#             setattr(model, key, val)
#             initial_params[key] = val
#
#         # Save
#         save_path = tmp_path / "test_model.npz"
#         model.save_params(save_path)
#
#         # Load
#         with expectation:
#             loaded_model = nmo.load_model(save_path, mapping_dict=mapping_dict)
#             loaded_params = loaded_model.get_params()
#             fit_state = loaded_model._get_fit_state()
#             fit_state.pop("solver_state_")
#             loaded_params.update(fit_state)
#
#             # Assert matching keys and values
#             assert (
#                 initial_params.keys() == loaded_params.keys()
#             ), "Parameter keys mismatch after load."
#
#             unexpected_keys = set(mapping_dict) - set(initial_params)
#             raise_exception = bool(unexpected_keys)
#             if raise_exception:
#                 with pytest.raises(
#                     ValueError, match="mapping_dict contains unexpected keys"
#                 ):
#                     raise ValueError(
#                         f"mapping_dict contains unexpected keys: {unexpected_keys}"
#                     )
#
#             for key in initial_params:
#                 init_val = initial_params[key]
#                 load_val = loaded_params[key]
#
#                 if key == "observation_model__inverse_link_function":
#                     if "observation_model" in mapping_dict:
#                         continue
#                 if key in mapping_dict:
#                     if key == "observation_model":
#                         if isinstance(mapping_dict[key], str):
#                             mapping_obs = instantiate_observation_model(
#                                 mapping_dict[key]
#                             )
#                         else:
#                             mapping_obs = mapping_dict[key]
#                         assert _get_name(mapping_obs) == _get_name(
#                             load_val
#                         ), f"{key} observation model mismatch: {mapping_dict[key]} != {load_val}"
#                     elif key == "regularizer":
#                         if isinstance(mapping_dict[key], str):
#                             mapping_reg = instantiate_regularizer(mapping_dict[key])
#                         else:
#                             mapping_reg = mapping_dict[key]
#                         assert _get_name(mapping_reg) == _get_name(
#                             load_val
#                         ), f"{key} regularizer mismatch: {mapping_dict[key]} != {load_val}"
#                     elif key == "solver_name":
#                         assert (
#                             mapping_dict[key] == load_val
#                         ), f"{key} solver name mismatch: {mapping_dict[key]} != {load_val}"
#                     elif key == "regularizer_strength":
#                         assert (
#                             mapping_dict[key] == load_val
#                         ), f"{key} regularizer strength mismatch: {mapping_dict[key]} != {load_val}"
#                     continue
#
#             if isinstance(init_val, (int, float, str, type(None))):
#                 assert init_val == load_val, f"{key} mismatch: {init_val} != {load_val}"
#
#             elif isinstance(init_val, dict):
#                 assert (
#                     init_val == load_val
#                 ), f"{key} dict mismatch: {init_val} != {load_val}"
#
#             elif isinstance(init_val, (np.ndarray, jnp.ndarray)):
#                 assert np.allclose(
#                     np.array(init_val), np.array(load_val)
#                 ), f"{key} array mismatch"
#
#             elif isinstance(init_val, Callable):
#                 assert _get_name(init_val) == _get_name(
#                     load_val
#                 ), f"{key} function mismatch: {_get_name(init_val)} != {_get_name(load_val)}"
#
#     def test_save_and_load_nested_class(
#         self, nested_regularizer, tmp_path, glm_class_type
#     ):
#         """Test that save and load works with nested classes."""
#         model = nmo.glm.GLM(regularizer=nested_regularizer, regularizer_strength=1.0)
#         save_path = tmp_path / "test_model.npz"
#         model.save_params(save_path)
#
#         mapping_dict = {
#             "regularizer": nested_regularizer.__class__,
#             "regularizer__func": jnp.exp,
#         }
#         with pytest.warns(UserWarning, match="The following keys have been replaced"):
#             loaded_model = nmo.load_model(save_path, mapping_dict=mapping_dict)
#
#         assert isinstance(loaded_model.regularizer, nested_regularizer.__class__)
#         assert isinstance(
#             loaded_model.regularizer.sub_regularizer,
#             nested_regularizer.sub_regularizer.__class__,
#         )
#         assert loaded_model.regularizer.func == mapping_dict["regularizer__func"]
#
#         # change mapping
#         mapping_dict = {
#             "regularizer": nested_regularizer.__class__,
#             "regularizer__sub_regularizer": nmo.regularizer.Ridge,
#             "regularizer__func": lambda x: x**2,
#         }
#         with pytest.warns(UserWarning, match="The following keys have been replaced"):
#             loaded_model = nmo.load_model(save_path, mapping_dict=mapping_dict)
#         assert isinstance(loaded_model.regularizer, nested_regularizer.__class__)
#         assert isinstance(
#             loaded_model.regularizer.sub_regularizer, nmo.regularizer.Ridge
#         )
#         assert loaded_model.regularizer.func == mapping_dict["regularizer__func"]
#
#     @pytest.mark.parametrize(
#         "fitted_glm_type",
#         [
#             "poissonGLM_fitted_model_instantiation",
#             "population_poissonGLM_fitted_model_instantiation",
#         ],
#     )
#     def test_save_and_load_fitted_model(
#         self, request, fitted_glm_type, glm_class_type, tmp_path
#     ):
#         """
#         Test saving and loading a fitted model with various observation models and regularizers.
#         Ensure all parameters are preserved.
#         """
#         _, _, fitted_model, _, _ = request.getfixturevalue(fitted_glm_type)
#
#         initial_params = fitted_model.get_params()
#         fit_state = fitted_model._get_fit_state()
#         fit_state.pop("solver_state_")
#         initial_params.update(fit_state)
#
#         # Save
#         save_path = tmp_path / "test_model.npz"
#         fitted_model.save_params(save_path)
#
#         # Load
#         loaded_model = nmo.load_model(save_path)
#         loaded_params = loaded_model.get_params()
#         fit_state = loaded_model._get_fit_state()
#         fit_state.pop("solver_state_")
#         loaded_params.update(fit_state)
#
#         # Assert states are close
#         for k, v in fit_state.items():
#             assert np.allclose(initial_params[k], v), f"{k} mismatch after load."
#
#     @pytest.mark.parametrize(
#         "fitted_glm_type",
#         [
#             "poissonGLM_fitted_model_instantiation",
#             "population_poissonGLM_fitted_model_instantiation",
#         ],
#     )
#     @pytest.mark.parametrize(
#         "param_name, param_value, expectation",
#         [
#             # Replace observation model class name  with a string
#             (
#                 "observation_model__class",
#                 "InvalidObservations",
#                 pytest.raises(
#                     ValueError, match="The class '[A-z]+' is not a native NeMoS"
#                 ),
#             ),
#             # Full path string
#             (
#                 "observation_model__class",
#                 "nemos.observation_models.InvalidObservations",
#                 pytest.raises(
#                     ValueError, match="The class '[A-z]+' is not a native NeMoS"
#                 ),
#             ),
#             # Replace observation model class name  with an instance
#             (
#                 "observation_model__class",
#                 nmo.observation_models.GammaObservations(),
#                 pytest.raises(
#                     ValueError,
#                     match="Object arrays cannot be loaded when allow_pickle=False",
#                 ),
#             ),
#             # Replace observation model class name with class
#             (
#                 "observation_model__class",
#                 nmo.observation_models.GammaObservations,
#                 pytest.raises(
#                     ValueError,
#                     match="Object arrays cannot be loaded when allow_pickle=False",
#                 ),
#             ),
#             # Replace link function with another callable
#             (
#                 "observation_model__params__inverse_link_function",
#                 np.exp,
#                 pytest.raises(
#                     ValueError,
#                     match="Object arrays cannot be loaded when allow_pickle=False",
#                 ),
#             ),
#             # Unexpected dtype for class name
#             (
#                 "regularizer__class",
#                 1,
#                 pytest.raises(
#                     ValueError, match="Parameter ``regularizer`` cannot be initialized"
#                 ),
#             ),
#             # Invalid fit parameter
#             (
#                 "scales_",  # wrong name for the params
#                 1,
#                 pytest.raises(ValueError, match="Unrecognized attribute 'scales_'"),
#             ),
#         ],
#     )
#     def test_modified_saved_file_raises(
#         self,
#         param_name,
#         param_value,
#         expectation,
#         glm_class_type,
#         fitted_glm_type,
#         request,
#         tmp_path,
#     ):
#         _, _, fitted_model, _, _ = request.getfixturevalue(fitted_glm_type)
#         save_path = tmp_path / "test_model.npz"
#         fitted_model.save_params(save_path)
#         # load and edit
#         data = np.load(save_path, allow_pickle=True)
#         load_data = dict((k, v) for k, v in data.items())
#         load_data[param_name] = param_value
#         np.savez(save_path, **load_data, allow_pickle=True)
#
#         with expectation:
#             nmo.load_model(save_path)
#
#     @pytest.mark.parametrize(
#         "fitted_glm_type",
#         [
#             "poissonGLM_fitted_model_instantiation",
#             "population_poissonGLM_fitted_model_instantiation",
#         ],
#     )
#     def test_key_suggestions(self, fitted_glm_type, request, glm_class_type, tmp_path):
#         _, _, fitted_model, _, _ = request.getfixturevalue(fitted_glm_type)
#         save_path = tmp_path / "test_model.npz"
#         fitted_model.save_params(save_path)
#
#         invalid_mapping = {
#             "regulsriaer": nmo.regularizer.Ridge,
#             "observatino_mdels": nmo.observation_models.GammaObservations,
#             "inv_link_function": jax.numpy.exp,
#             "total_nonsense": jax.numpy.exp,
#         }
#         match = (
#             r"The following keys in your mapping do not match any parameters in the loaded model:\n\n"
#             r"\t- 'regulsriaer', did you mean 'regularizer'\?\n"
#             r"\t- 'observatino_mdels', did you mean 'observation_model'\?\n"
#             r"\t- 'inv_link_function', did you mean 'inverse_link_function'\?\n"
#             r"\t- 'total_nonsense'\n"
#         )
#         with pytest.raises(ValueError, match=match):
#             nmo.load_model(save_path, mapping_dict=invalid_mapping)
#
