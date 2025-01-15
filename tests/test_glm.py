import inspect
import warnings
from contextlib import nullcontext as does_not_raise
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sklearn
import statsmodels.api as sm
from pynapple import Tsd, TsdFrame
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.model_selection import GridSearchCV

import nemos as nmo
from nemos.pytrees import FeaturePytree
from nemos.tree_utils import pytree_map_and_reduce, tree_l2_norm, tree_slice, tree_sub


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


class TestGLM:
    """
    Unit tests for the PoissonGLM class.
    """

    #######################
    # Test model.__init__
    #######################
    @pytest.mark.parametrize(
        "regularizer, solver_name, expectation",
        [
            (nmo.regularizer.Ridge(), "BFGS", does_not_raise()),
            (
                None,
                None,
                pytest.raises(
                    TypeError, match="The regularizer should be either a string from "
                ),
            ),
            (
                nmo.regularizer.Ridge,
                None,
                pytest.raises(
                    TypeError, match="The regularizer should be either a string from "
                ),
            ),
        ],
    )
    def test_solver_type(self, regularizer, solver_name, expectation, glm_class):
        """
        Test that an error is raised if a non-compatible solver is passed.
        """
        with expectation:
            glm_class(
                regularizer=regularizer, solver_name=solver_name, regularizer_strength=1
            )

    @pytest.mark.parametrize(
        "observation, expectation",
        [
            (nmo.observation_models.PoissonObservations(), does_not_raise()),
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
        self, observation, expectation, glm_class, ridge_regularizer
    ):
        """
        Test initialization with different regularizer names. Check if an appropriate exception is raised
        when the regularizer name is not present in jaxopt.
        """
        with expectation:
            glm_class(
                regularizer=ridge_regularizer,
                regularizer_strength=0.1,
                solver_name="LBFGS",
                observation_model=observation,
            )

    @pytest.mark.parametrize(
        "X, y",
        [
            (jnp.zeros((2, 4)), jnp.zeros((2,))),
            (jnp.zeros((2, 4)), jnp.zeros((2,))),
        ],
    )
    def test_parameter_initialization(self, X, y, poissonGLM_model_instantiation):
        _, _, model, _, _ = poissonGLM_model_instantiation
        coef, inter = model._initialize_parameters(X, y)
        assert coef.shape == (X.shape[1],)
        assert inter.shape == (1,)

    def test_get_params(self):
        """
        Test that get_params() contains expected values.
        """
        expected_keys = {
            "observation_model__inverse_link_function",
            "observation_model",
            "regularizer",
            "regularizer_strength",
            "solver_kwargs",
            "solver_name",
        }

        model = nmo.glm.GLM()

        expected_values = [
            model.observation_model.inverse_link_function,
            model.observation_model,
            model.regularizer,
            model.regularizer_strength,
            model.solver_kwargs,
            model.solver_name,
        ]

        assert set(model.get_params().keys()) == expected_keys
        assert list(model.get_params().values()) == expected_values

        # passing params
        model = nmo.glm.GLM(solver_name="LBFGS", regularizer="UnRegularized")

        expected_values = [
            model.observation_model.inverse_link_function,
            model.observation_model,
            model.regularizer,
            model.regularizer_strength,
            model.solver_kwargs,
            model.solver_name,
        ]

        assert set(model.get_params().keys()) == expected_keys
        assert list(model.get_params().values()) == expected_values

        # changing regularizer
        model.set_params(regularizer="Ridge", regularizer_strength=1.0)

        expected_values = [
            model.observation_model.inverse_link_function,
            model.observation_model,
            model.regularizer,
            model.regularizer_strength,
            model.solver_kwargs,
            model.solver_name,
        ]

        assert set(model.get_params().keys()) == expected_keys
        assert list(model.get_params().values()) == expected_values

        # changing solver
        model.solver_name = "ProximalGradient"

        expected_values = [
            model.observation_model.inverse_link_function,
            model.observation_model,
            model.regularizer,
            model.regularizer_strength,
            model.solver_kwargs,
            model.solver_name,
        ]

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
    def test_fit_param_length(
        self, n_params, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `fit` method with different numbers of initial parameters.
        Check for correct number of parameters.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        if n_params == 0:
            init_params = tuple()
        elif n_params == 1:
            init_params = (true_params[0],)
        else:
            init_params = true_params + (true_params[0],) * (n_params - 2)
        with expectation:
            model.fit(X, y, init_params=init_params)

    @pytest.mark.parametrize(
        "dim_weights, expectation",
        [
            (
                0,
                pytest.raises(
                    ValueError,
                    match=r"Inconsistent number of features",
                ),
            ),
            (
                1,
                does_not_raise(),
            ),
            (
                2,
                pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
            ),
            (
                3,
                pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
            ),
        ],
    )
    def test_fit_weights_dimensionality(
        self, dim_weights, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `fit` method with weight matrices of different dimensionalities.
        Check for correct dimensionality.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_samples, n_features = X.shape
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
    def test_fit_intercepts_dimensionality(
        self, dim_intercepts, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `fit` method with intercepts of different dimensionalities. Check for correct dimensionality.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_samples, n_features = X.shape
        init_b = jnp.zeros((1,) * dim_intercepts)
        init_w = jnp.zeros((n_features,))
        with expectation:
            model.fit(X, y, init_params=(init_w, init_b))

    @pytest.mark.parametrize(
        "init_params, expectation",
        [
            ([jnp.zeros((5,)), jnp.zeros((1,))], does_not_raise()),
            (
                [[jnp.zeros((1, 5)), jnp.zeros((1,))]],
                pytest.raises(ValueError, match="Params must have length two."),
            ),
            (dict(p1=jnp.zeros((5,)), p2=jnp.zeros((1,))), pytest.raises(KeyError)),
            (
                (dict(p1=jnp.zeros((5,)), p2=jnp.zeros((1,))), jnp.zeros((1,))),
                pytest.raises(
                    TypeError, match=r"X and params\[0\] must be the same type"
                ),
            ),
            (
                (
                    FeaturePytree(p1=jnp.zeros((5,)), p2=jnp.zeros((5,))),
                    jnp.zeros((1,)),
                ),
                pytest.raises(
                    TypeError, match=r"X and params\[0\] must be the same type"
                ),
            ),
            (0, pytest.raises(ValueError, match="Params must have length two.")),
            (
                {0, 1},
                pytest.raises(TypeError, match="Initial parameters must be array-like"),
            ),
            (
                [jnp.zeros((1, 5)), ""],
                pytest.raises(TypeError, match="Initial parameters must be array-like"),
            ),
            (
                ["", jnp.zeros((1,))],
                pytest.raises(TypeError, match="Initial parameters must be array-like"),
            ),
        ],
    )
    def test_fit_init_params_type(
        self, init_params, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `fit` method with various types of initial parameters. Ensure that the provided initial parameters
        are array-like.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
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
        self, delta_dim, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `fit` method with X input data of different dimensionalities. Ensure correct dimensionality for X.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        if delta_dim == -1:
            X = np.zeros((X.shape[0],))
        elif delta_dim == 1:
            X = np.zeros((X.shape[0], 1, X.shape[1]))
        with expectation:
            model.fit(X, y, init_params=true_params)

    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (-1, pytest.raises(ValueError, match="y must be one-dimensional")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="y must be one-dimensional")),
        ],
    )
    def test_fit_y_dimensionality(
        self, delta_dim, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `fit` method with y target data of different dimensionalities. Ensure correct dimensionality for y.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
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
    def test_fit_n_feature_consistency_weights(
        self, delta_n_features, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `fit` method for inconsistencies between data features and initial weights provided.
        Ensure the number of features align.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
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
    def test_fit_n_feature_consistency_x(
        self, delta_n_features, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `fit` method for inconsistencies between data features and model's expectations.
        Ensure the number of features in X aligns.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
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
        self, delta_tp, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `fit` method for inconsistencies in time-points in data X. Ensure the correct number of time-points.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
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
        self, delta_tp, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `fit` method for inconsistencies in time-points in y. Ensure the correct number of time-points.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        y = jnp.zeros((y.shape[0] + delta_tp,) + y.shape[1:])
        with expectation:
            model.fit(X, y, init_params=true_params)

    def test_fit_mask_grouplasso(self, group_sparse_poisson_glm_model_instantiation):
        """Test that the group lasso fit goes through"""
        X, y, model, params, rate, mask = group_sparse_poisson_glm_model_instantiation
        model.set_params(
            regularizer_strength=1.0,
            regularizer=nmo.regularizer.GroupLasso(mask=mask),
            solver_name="ProximalGradient",
        )
        model.fit(X, y)

    def test_fit_pytree_equivalence(
        self, poissonGLM_model_instantiation, poissonGLM_model_instantiation_pytree
    ):
        """Check that the glm fit with pytree learns the same parameters."""
        # required for numerical precision of coeffs
        jax.config.update("jax_enable_x64", True)
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        X_tree, _, model_tree, true_params_tree, _ = (
            poissonGLM_model_instantiation_pytree
        )
        # fit both models
        model.fit(X, y, init_params=true_params)
        model_tree.fit(X_tree, y, init_params=true_params_tree)

        # get the flat parameters
        flat_coef = np.concatenate(
            jax.tree_util.tree_flatten(model_tree.coef_)[0], axis=-1
        )

        # assert equivalence of solutions
        assert np.allclose(model.coef_, flat_coef)
        assert np.allclose(model.intercept_, model_tree.intercept_)
        assert np.allclose(model.score(X, y), model_tree.score(X_tree, y))
        assert np.allclose(model.predict(X), model_tree.predict(X_tree))

    def test_fit_pytree_equivalence_gamma(
        self, gammaGLM_model_instantiation, gammaGLM_model_instantiation_pytree
    ):
        """Check that the glm fit with pytree learns the same parameters."""
        # required for numerical precision of coeffs
        jax.config.update("jax_enable_x64", True)
        X, y, model, true_params, firing_rate = gammaGLM_model_instantiation
        X_tree, _, model_tree, true_params_tree, _ = gammaGLM_model_instantiation_pytree
        # fit both models
        model.fit(X, y, init_params=true_params)
        model_tree.fit(X_tree, y, init_params=true_params_tree)

        # get the flat parameters
        flat_coef = np.concatenate(
            jax.tree_util.tree_flatten(model_tree.coef_)[0], axis=0
        )

        # assert equivalence of solutions
        assert np.allclose(model.coef_, flat_coef)
        assert np.allclose(model.intercept_, model_tree.intercept_)
        assert np.allclose(model.score(X, y), model_tree.score(X_tree, y))
        assert np.allclose(model.predict(X), model_tree.predict(X_tree))
        assert np.allclose(model.scale_, model_tree.scale_)

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
        self, fill_val, expectation, poissonGLM_model_instantiation
    ):
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        X.fill(fill_val)
        with expectation:
            model.fit(X, y)

    @pytest.mark.parametrize("inv_link", [jnp.exp, lambda x: 1 / x])
    def test_fit_gamma_glm(self, inv_link, gammaGLM_model_instantiation):
        X, y, model, true_params, firing_rate = gammaGLM_model_instantiation
        model.observation_model.inverse_link_function = inv_link
        model.fit(X, y)

    @pytest.mark.parametrize("inv_link", [jnp.exp, lambda x: 1 / x])
    def test_fit_set_scale(self, inv_link, gammaGLM_model_instantiation):
        X, y, model, true_params, firing_rate = gammaGLM_model_instantiation
        model.observation_model.inverse_link_function = inv_link
        model.fit(X, y)
        assert model.scale_ != 1

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
        self, delta_dim, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `score` method with X input data of different dimensionalities. Ensure correct dimensionality for X.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
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
                    ValueError, match="y must be one-dimensional, with shape"
                ),
            ),
            (0, does_not_raise()),
            (
                1,
                pytest.raises(
                    ValueError, match="y must be one-dimensional, with shape"
                ),
            ),
        ],
    )
    def test_score_y_dimensionality(
        self, delta_dim, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `score` method with y of different dimensionalities.
        Ensure correct dimensionality for y.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
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
        self, delta_n_features, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `score` method for inconsistencies in features of X.
        Ensure the number of features in X aligns with the model params.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        if delta_n_features == 1:
            X = jnp.concatenate((X, jnp.zeros((X.shape[0], 1))), axis=1)
        elif delta_n_features == -1:
            X = X[..., :-1]
        with expectation:
            model.score(X, y)

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
    def test_predict_is_fit(self, is_fit, expectation, poissonGLM_model_instantiation):
        """
        Test the `score` method on models based on their fit status.
        Ensure scoring is only possible on fitted models.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        if is_fit:
            model.fit(X, y)
        with expectation:
            model.predict(X)

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
        self, delta_tp, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `score` method for inconsistencies in time-points in X.
        Ensure that the number of time-points in X and y matches.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
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
        self, delta_tp, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `score` method for inconsistencies in time-points in y.
        Ensure that the number of time-points in X and y matches.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        y = jnp.zeros((y.shape[0] + delta_tp,) + y.shape[1:])
        with expectation:
            model.score(X, y)

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
        self, score_type, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `score` method for unsupported scoring types.
        Ensure only valid score types are used.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        with expectation:
            model.score(X, y, score_type=score_type)

    def test_loglikelihood_against_scipy_stats(self, poissonGLM_model_instantiation):
        """
        Compare the model's log-likelihood computation against `jax.scipy`.
        Ensure consistent and correct calculations.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        # set model coeff
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        # get the rate
        mean_firing = model.predict(X)
        # compute the log-likelihood using jax.scipy
        mean_ll_jax = jax.scipy.stats.poisson.logpmf(y, mean_firing).mean()
        model_ll = model.score(X, y, score_type="log-likelihood")
        if not np.allclose(mean_ll_jax, model_ll):
            raise ValueError(
                "Log-likelihood of PoissonModel does not match" "that of jax.scipy!"
            )

    @pytest.mark.parametrize("inv_link", [jnp.exp, lambda x: 1 / x])
    def test_score_gamma_glm(self, inv_link, gammaGLM_model_instantiation):
        X, y, model, true_params, firing_rate = gammaGLM_model_instantiation
        model.observation_model.inverse_link_function = inv_link
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        model.scale_ = 1.0
        model.score(X, y)

    #######################
    # Test model.predict
    #######################
    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (-1, pytest.raises(ValueError, match="X must be two-dimensional")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="X must be two-dimensional")),
        ],
    )
    def test_predict_x_dimensionality(
        self, delta_dim, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `predict` method with x input data of different dimensionalities.
        Ensure correct dimensionality for x.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
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
        self, delta_n_features, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `predict` method ensuring the number of features in x input data
        is consistent with the model's `model.coef_`.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        if delta_n_features == 1:
            X = jnp.concatenate((X, jnp.zeros((X.shape[0], 1))), axis=1)
        elif delta_n_features == -1:
            X = X[..., :-1]
        with expectation:
            model.predict(X)

    #######################
    # Test model.initialize_solver
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
    def test_initialize_solver_param_length(
        self, n_params, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `initialize_solver` method with different numbers of initial parameters.
        Check for correct number of parameters.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        if n_params == 0:
            init_params = tuple()
        elif n_params == 1:
            init_params = (true_params[0],)
        else:
            init_params = true_params + (true_params[0],) * (n_params - 2)
        with expectation:
            params = model.initialize_params(X, y, init_params=init_params)
            model.initialize_state(X, y, params)

    @pytest.mark.parametrize(
        "inv_link", [jnp.exp, lambda x: jnp.exp(x), jax.nn.softplus, jax.nn.relu]
    )
    def test_high_firing_rate_initialization(
        self, inv_link, example_X_y_high_firing_rates
    ):
        model = nmo.glm.GLM(
            observation_model=nmo.observation_models.PoissonObservations(
                inverse_link_function=inv_link
            )
        )
        X, y = example_X_y_high_firing_rates
        model.initialize_params(X, y[:, 0])

    @pytest.mark.parametrize(
        "dim_weights, expectation",
        [
            (
                0,
                pytest.raises(
                    ValueError,
                    match=r"Inconsistent number of features",
                ),
            ),
            (
                1,
                does_not_raise(),
            ),
            (
                2,
                pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
            ),
            (
                3,
                pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
            ),
        ],
    )
    def test_initialize_solver_weights_dimensionality(
        self, dim_weights, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `initialize_solver` method with weight matrices of different dimensionalities.
        Check for correct dimensionality.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_samples, n_features = X.shape
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
            model.initialize_params(X, y, init_params=(init_w, true_params[1]))

    @pytest.mark.parametrize(
        "dim_intercepts, expectation",
        [
            (0, pytest.raises(ValueError, match=r"params\[1\] must be of shape")),
            (1, does_not_raise()),
            (2, pytest.raises(ValueError, match=r"params\[1\] must be of shape")),
            (3, pytest.raises(ValueError, match=r"params\[1\] must be of shape")),
        ],
    )
    def test_initialize_solver_intercepts_dimensionality(
        self, dim_intercepts, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `initialize_solver` method with intercepts of different dimensionalities.

        Check for correct dimensionality.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_samples, n_features = X.shape
        init_b = jnp.zeros((1,) * dim_intercepts)
        init_w = jnp.zeros((n_features,))
        with expectation:
            params = model.initialize_params(X, y, init_params=(init_w, init_b))
            model.initialize_state(X, y, params)

    @pytest.mark.parametrize(
        "init_params, expectation",
        [
            ([jnp.zeros((5,)), jnp.zeros((1,))], does_not_raise()),
            (
                [[jnp.zeros((1, 5)), jnp.zeros((1,))]],
                pytest.raises(ValueError, match="Params must have length two."),
            ),
            (dict(p1=jnp.zeros((5,)), p2=jnp.zeros((1,))), pytest.raises(KeyError)),
            (
                (dict(p1=jnp.zeros((5,)), p2=jnp.zeros((1,))), jnp.zeros((1,))),
                pytest.raises(
                    TypeError, match=r"X and params\[0\] must be the same type"
                ),
            ),
            (
                (
                    FeaturePytree(p1=jnp.zeros((5,)), p2=jnp.zeros((5,))),
                    jnp.zeros((1,)),
                ),
                pytest.raises(
                    TypeError, match=r"X and params\[0\] must be the same type"
                ),
            ),
            (0, pytest.raises(ValueError, match="Params must have length two.")),
            (
                {0, 1},
                pytest.raises(TypeError, match="Initial parameters must be array-like"),
            ),
            (
                [jnp.zeros((1, 5)), ""],
                pytest.raises(TypeError, match="Initial parameters must be array-like"),
            ),
            (
                ["", jnp.zeros((1,))],
                pytest.raises(TypeError, match="Initial parameters must be array-like"),
            ),
        ],
    )
    def test_initialize_solver_init_params_type(
        self, init_params, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `initialize_solver` method with various types of initial parameters.

        Ensure that the provided initial parameters are array-like.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        with expectation:
            params = model.initialize_params(X, y, init_params=init_params)
            model.initialize_state(X, y, params)

    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (-1, pytest.raises(ValueError, match="X must be two-dimensional")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="X must be two-dimensional")),
        ],
    )
    def test_initialize_solver_x_dimensionality(
        self, delta_dim, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `initialize_solver` method with X input data of different dimensionalities.

        Ensure correct dimensionality for X.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        if delta_dim == -1:
            X = np.zeros((X.shape[0],))
        elif delta_dim == 1:
            X = np.zeros((X.shape[0], 1, X.shape[1]))
        with expectation:
            params = model.initialize_params(X, y, init_params=true_params)
            model.initialize_state(X, y, params)

    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (-1, pytest.raises(ValueError, match="y must be one-dimensional")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="y must be one-dimensional")),
        ],
    )
    def test_initialize_solver_y_dimensionality(
        self, delta_dim, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `initialize_solver` method with y target data of different dimensionalities.

        Ensure correct dimensionality for y.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        if delta_dim == -1:
            y = np.zeros([])
        elif delta_dim == 1:
            y = np.zeros((y.shape[0], 1))
        with expectation:
            params = model.initialize_params(X, y, init_params=true_params)
            model.initialize_state(X, y, params)

    @pytest.mark.parametrize(
        "delta_n_features, expectation",
        [
            (-1, pytest.raises(ValueError, match="Inconsistent number of features")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="Inconsistent number of features")),
        ],
    )
    def test_initialize_solver_n_feature_consistency_weights(
        self, delta_n_features, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `initialize_solver` method for inconsistencies between data features and initial weights provided.
        Ensure the number of features align.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        init_w = jnp.zeros((X.shape[1] + delta_n_features))
        init_b = jnp.zeros(
            1,
        )
        with expectation:
            params = model.initialize_params(X, y, init_params=(init_w, init_b))
            model.initialize_state(X, y, params)

    @pytest.mark.parametrize(
        "delta_n_features, expectation",
        [
            (-1, pytest.raises(ValueError, match="Inconsistent number of features")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="Inconsistent number of features")),
        ],
    )
    def test_initialize_solver_n_feature_consistency_x(
        self, delta_n_features, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `initialize_solver` method for inconsistencies between data features and model's expectations.
        Ensure the number of features in X aligns.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        if delta_n_features == 1:
            X = jnp.concatenate((X, jnp.zeros((X.shape[0], 1))), axis=1)
        elif delta_n_features == -1:
            X = X[..., :-1]
        with expectation:
            params = model.initialize_params(X, y, init_params=true_params)
            model.initialize_state(X, y, params)

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
        self, delta_tp, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `initialize_solver` method for inconsistencies in time-points in data X.

        Ensure the correct number of time-points.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        X = jnp.zeros((X.shape[0] + delta_tp,) + X.shape[1:])
        with expectation:
            params = model.initialize_params(X, y, init_params=true_params)
            model.initialize_state(X, y, params)

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
        self, delta_tp, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `initialize_solver` method for inconsistencies in time-points in y.

        Ensure the correct number of time-points.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        y = jnp.zeros((y.shape[0] + delta_tp,) + y.shape[1:])
        with expectation:
            model.initialize_params(X, y, init_params=true_params)

    def test_initialize_solver_mask_grouplasso(
        self, group_sparse_poisson_glm_model_instantiation
    ):
        """Test that the group lasso initialize_solver goes through"""
        X, y, model, params, rate, mask = group_sparse_poisson_glm_model_instantiation
        model.set_params(
            regularizer=nmo.regularizer.GroupLasso(mask=mask),
            solver_name="ProximalGradient",
            regularizer_strength=1.0,
        )
        params = model.initialize_params(X, y)
        model.initialize_state(X, y, params)

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
        self, fill_val, expectation, poissonGLM_model_instantiation
    ):
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        X.fill(fill_val)
        with expectation:
            params = model.initialize_params(X, y)
            model.initialize_state(X, y, params)

    def test_initializer_solver_set_solver_callable(
        self, poissonGLM_model_instantiation
    ):
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        assert model.solver_init_state is None
        assert model.solver_update is None
        assert model.solver_run is None
        init_params = model.initialize_params(X, y)
        model.initialize_state(X, y, init_params)
        assert isinstance(model.solver_init_state, Callable)
        assert isinstance(model.solver_update, Callable)
        assert isinstance(model.solver_run, Callable)

    #######################
    # Test model.update
    #######################
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
    def test_update_n_samples(
        self, n_samples, expectation, batch_size, poissonGLM_model_instantiation
    ):
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        params = model.initialize_params(X, y)
        state = model.initialize_state(X, y, params)
        with expectation:
            model.update(
                params, state, X[:batch_size], y[:batch_size], n_samples=n_samples
            )

    @pytest.mark.parametrize("batch_size", [1, 10])
    def test_update_params_stored(self, batch_size, poissonGLM_model_instantiation):
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        params = model.initialize_params(X, y)
        state = model.initialize_state(X, y, params)
        assert model.coef_ is None
        assert model.intercept_ is None
        assert model.scale_ is None
        _, _ = model.update(params, state, X[:batch_size], y[:batch_size])
        assert model.coef_ is not None
        assert model.intercept_ is not None
        assert model.scale_ is not None

    @pytest.mark.parametrize("batch_size", [2, 10])
    def test_update_nan_drop_at_jit_comp(
        self, batch_size, poissonGLM_model_instantiation
    ):
        """Test that jit compilation does not affect the update in the presence of nans."""
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        params = model.initialize_params(X, y)
        state = model.initialize_state(X, y, params)

        # extract batch and add nans
        Xnan = X[:batch_size]
        Xnan[: batch_size // 2] = np.nan

        jit_update, _ = model.update(params, state, Xnan, y[:batch_size])
        # make sure there is an update
        assert any(~jnp.allclose(p0, jit_update[k]) for k, p0 in enumerate(params))
        # update without jitting
        with jax.disable_jit(True):
            nojit_update, _ = model.update(params, state, Xnan, y[:batch_size])
        # check for equivalence update
        assert all(jnp.allclose(p0, jit_update[k]) for k, p0 in enumerate(nojit_update))

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
        self, delta_dim, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `simulate` method with input data of different dimensionalities.
        Ensure correct dimensionality for input.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
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
        "input_type, expected_out_type",
        [
            (TsdFrame, Tsd),
            (np.ndarray, jnp.ndarray),
            (jnp.ndarray, jnp.ndarray),
        ],
    )
    def test_simulate_pynapple(
        self, input_type, expected_out_type, poissonGLM_model_instantiation
    ):
        """
        Test that the `simulate` method retturns the expected data type for different allowed inputs.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]

        if input_type == TsdFrame:
            X = TsdFrame(t=np.arange(X.shape[0]), d=X)
        count, rate = model.simulate(
            random_key=jax.random.key(123),
            feedforward_input=X,
        )
        assert isinstance(count, expected_out_type)
        assert isinstance(rate, expected_out_type)

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
    def test_simulate_is_fit(self, is_fit, expectation, poissonGLM_model_instantiation):
        """
        Test if the model raises a ValueError when trying to simulate before it's fitted.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        if is_fit:
            model.coef_ = true_params[0]
            model.intercept_ = true_params[1]
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
        self, delta_features, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `simulate` method ensuring the number of features in `feedforward_input` is
        consistent with the model's expected number of features.

        Notes
        -----
        The total feature number `model.coef_.shape[1]` must be equal to
        `feedforward_input.shape[2] + coupling_basis.shape[1]*n_neurons`
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
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

    def test_simulate_feedforward_glm(self, poissonGLM_model_instantiation):
        """Test that simulate goes through"""
        X, y, model, params, rate = poissonGLM_model_instantiation
        model.coef_ = params[0]
        model.intercept_ = params[1]
        ysim, ratesim = model.simulate(jax.random.key(123), X)
        # check that the expected dimensionality is returned
        assert ysim.ndim == 1
        assert ratesim.ndim == 1
        # check that the rates and spikes has the same shape
        assert ratesim.shape[0] == ysim.shape[0]
        # check the time point number is that expected (same as the input)
        assert ysim.shape[0] == X.shape[0]

    @pytest.mark.parametrize("inv_link", [jnp.exp, lambda x: 1 / x])
    def test_simulate_gamma_glm(self, inv_link, gammaGLM_model_instantiation):
        X, y, model, true_params, firing_rate = gammaGLM_model_instantiation
        model.observation_model.inverse_link_function = inv_link
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        model.scale_ = 1.0
        ysim, ratesim = model.simulate(jax.random.PRNGKey(123), X)
        assert ysim.shape == y.shape
        assert ratesim.shape == y.shape

    #######################################
    # Compare with standard implementation
    #######################################
    def test_deviance_against_statsmodels(self, poissonGLM_model_instantiation):
        """
        Compare fitted parameters to statsmodels.
        Assesses if the model estimates are close to statsmodels' results.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        # set model coeff
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        # get the rate
        dev = sm.families.Poisson().deviance(y, firing_rate)
        dev_model = model.observation_model.deviance(y, firing_rate).sum()
        if not np.allclose(dev, dev_model):
            raise ValueError("Deviance doesn't match statsmodels!")

    def test_compatibility_with_sklearn_cv(self, poissonGLM_model_instantiation):
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        param_grid = {"solver_name": ["BFGS", "GradientDescent"]}
        GridSearchCV(model, param_grid).fit(X, y)

    def test_compatibility_with_sklearn_cv_gamma(self, gammaGLM_model_instantiation):
        X, y, model, true_params, firing_rate = gammaGLM_model_instantiation
        param_grid = {"solver_name": ["BFGS", "GradientDescent"]}
        GridSearchCV(model, param_grid).fit(X, y)

    @pytest.mark.parametrize(
        "regr_setup, glm_class",
        [
            ("poissonGLM_model_instantiation", nmo.glm.GLM),
            ("poissonGLM_model_instantiation_pytree", nmo.glm.GLM),
            ("poisson_population_GLM_model", nmo.glm.PopulationGLM),
            ("poisson_population_GLM_model_pytree", nmo.glm.PopulationGLM),
        ],
    )
    @pytest.mark.parametrize("key", [jax.random.key(0), jax.random.key(19)])
    @pytest.mark.parametrize(
        "regularizer_class, solver_name",
        [
            (nmo.regularizer.UnRegularized, "SVRG"),
            (nmo.regularizer.Ridge, "SVRG"),
            (nmo.regularizer.Lasso, "ProxSVRG"),
            # (nmo.regularizer.GroupLasso, "ProxSVRG"),
        ],
    )
    def test_glm_update_consistent_with_fit_with_svrg(
        self, request, regr_setup, glm_class, key, regularizer_class, solver_name
    ):
        """
        Make sure that calling GLM.update with the rest of the algorithm implemented outside in a naive loop
        is consistent with running the compiled GLM.fit on the same data with the same parameters
        """
        jax.config.update("jax_enable_x64", True)
        X, y, model, true_params, rate = request.getfixturevalue(regr_setup)

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
        glm = glm_class(
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
        glm2 = glm_class(
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
        loss_grad = jax.jit(jax.grad(glm._solver_loss_fun_))

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
    def test_glm_fit_matches_sklearn_poisson(
        self, solver_name, poissonGLM_model_instantiation
    ):
        """Test that different solvers converge to the same solution."""
        jax.config.update("jax_enable_x64", True)
        X, y, _, true_params, firing_rate = poissonGLM_model_instantiation

        model = nmo.glm.GLM(
            regularizer=nmo.regularizer.UnRegularized(),
            observation_model=nmo.observation_models.PoissonObservations(),
            solver_name=solver_name,
            solver_kwargs={"tol": 10**-12},
        )
        # set precision to float64 for accurate matching of the results
        model.data_type = jnp.float64
        model.fit(X, y)

        model_skl = PoissonRegressor(fit_intercept=True, tol=10**-12, alpha=0.0)
        model_skl.fit(X, y)

        match_weights = jnp.allclose(model_skl.coef_, model.coef_, atol=1e-5, rtol=0.0)
        match_intercepts = jnp.allclose(
            model_skl.intercept_, model.intercept_, atol=1e-5, rtol=0.0
        )
        if (not match_weights) or (not match_intercepts):
            raise ValueError("GLM.fit estimate does not match sklearn!")

    @pytest.mark.parametrize("solver_name", ["GradientDescent", "SVRG"])
    def test_glm_fit_matches_sklearn_gamma(
        self, solver_name, gammaGLM_model_instantiation
    ):
        """Test that different solvers converge to the same solution."""
        jax.config.update("jax_enable_x64", True)
        X, y, _, true_params, firing_rate = gammaGLM_model_instantiation

        model = nmo.glm.GLM(
            regularizer=nmo.regularizer.UnRegularized(),
            observation_model=nmo.observation_models.GammaObservations(
                inverse_link_function=jnp.exp
            ),
            solver_name=solver_name,
            solver_kwargs={"tol": 10**-12},
        )
        # set precision to float64 for accurate matching of the results
        model.data_type = jnp.float64
        model.fit(X, y)

        model_skl = GammaRegressor(fit_intercept=True, tol=10**-12, alpha=0.0)
        model_skl.fit(X, y)

        match_weights = jnp.allclose(model_skl.coef_, model.coef_, atol=1e-5, rtol=0.0)
        match_intercepts = jnp.allclose(
            model_skl.intercept_, model.intercept_, atol=1e-5, rtol=0.0
        )

        if (not match_weights) or (not match_intercepts):
            raise ValueError("GLM.fit estimate does not match sklearn!")

    @pytest.mark.parametrize(
        "reg, dof",
        [
            (nmo.regularizer.UnRegularized(), np.array([5])),
            (
                nmo.regularizer.Lasso(),
                np.array([3]),
            ),  # this lasso fit has only 3 coeff of the first neuron
            # surviving
            (nmo.regularizer.Ridge(), np.array([5])),
        ],
    )
    @pytest.mark.parametrize("n_samples", [1, 20])
    def test_estimate_dof_resid(
        self, n_samples, dof, reg, poissonGLM_model_instantiation
    ):
        """
        Test that the dof is an integer.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        strength = None if isinstance(reg, nmo.regularizer.UnRegularized) else 1.0
        model.set_params(regularizer=reg, regularizer_strength=strength)
        model.solver_name = model.regularizer.default_solver
        model.fit(X, y)
        num = model._estimate_resid_degrees_of_freedom(X, n_samples=n_samples)
        assert np.allclose(num, n_samples - dof - 1)

    @pytest.mark.parametrize("reg", ["Ridge", "Lasso", "GroupLasso"])
    def test_warning_solver_reg_str(self, reg):
        # check that a warning is triggered
        # if no param is passed
        with pytest.warns(UserWarning):
            nmo.glm.GLM(regularizer=reg)

        # # check that the warning is not triggered
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = nmo.glm.GLM(regularizer=reg, regularizer_strength=1.0)

        # reset to unregularized
        model.set_params(regularizer="UnRegularized", regularizer_strength=None)
        with pytest.warns(UserWarning):
            nmo.glm.GLM(regularizer=reg)

    @pytest.mark.parametrize("reg", ["Ridge", "Lasso", "GroupLasso"])
    def test_reg_strength_reset(self, reg):
        model = nmo.glm.GLM(regularizer=reg, regularizer_strength=1.0)
        with pytest.warns(
            UserWarning,
            match="Unused parameter `regularizer_strength` for UnRegularized GLM",
        ):
            model.regularizer = "UnRegularized"
        model.regularizer_strength = None
        with pytest.warns(
            UserWarning, match="Caution: regularizer strength has not been set"
        ):
            model.regularizer = "Ridge"

    @pytest.mark.parametrize(
        "params, warns",
        [
            # set regularizer
            (
                {"regularizer": "Ridge"},
                pytest.warns(
                    UserWarning, match="Caution: regularizer strength has not been set"
                ),
            ),
            (
                {"regularizer": "Lasso"},
                pytest.warns(
                    UserWarning, match="Caution: regularizer strength has not been set"
                ),
            ),
            (
                {"regularizer": "GroupLasso"},
                pytest.warns(
                    UserWarning, match="Caution: regularizer strength has not been set"
                ),
            ),
            ({"regularizer": "UnRegularized"}, does_not_raise()),
            # set both None or number
            (
                {"regularizer": "Ridge", "regularizer_strength": None},
                pytest.warns(
                    UserWarning, match="Caution: regularizer strength has not been set"
                ),
            ),
            ({"regularizer": "Ridge", "regularizer_strength": 1.0}, does_not_raise()),
            (
                {"regularizer": "Lasso", "regularizer_strength": None},
                pytest.warns(
                    UserWarning, match="Caution: regularizer strength has not been set"
                ),
            ),
            ({"regularizer": "Lasso", "regularizer_strength": 1.0}, does_not_raise()),
            (
                {"regularizer": "GroupLasso", "regularizer_strength": None},
                pytest.warns(
                    UserWarning, match="Caution: regularizer strength has not been set"
                ),
            ),
            (
                {"regularizer": "GroupLasso", "regularizer_strength": 1.0},
                does_not_raise(),
            ),
            (
                {"regularizer": "UnRegularized", "regularizer_strength": None},
                does_not_raise(),
            ),
            (
                {"regularizer": "UnRegularized", "regularizer_strength": 1.0},
                pytest.warns(
                    UserWarning,
                    match="Unused parameter `regularizer_strength` for UnRegularized GLM",
                ),
            ),
            # set regularizer str only
            (
                {"regularizer_strength": 1.0},
                pytest.warns(
                    UserWarning,
                    match="Unused parameter `regularizer_strength` for UnRegularized GLM",
                ),
            ),
            ({"regularizer_strength": None}, does_not_raise()),
        ],
    )
    def test_reg_set_params(self, params, warns):
        model = nmo.glm.GLM()
        with warns:
            model.set_params(**params)

    @pytest.mark.parametrize(
        "params, warns",
        [
            # set regularizer str only
            ({"regularizer_strength": 1.0}, does_not_raise()),
            (
                {"regularizer_strength": None},
                pytest.warns(
                    UserWarning, match="Caution: regularizer strength has not been set"
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("reg", ["Ridge", "Lasso", "GroupLasso"])
    def test_reg_set_params_reg_str_only(self, params, warns, reg):
        model = nmo.glm.GLM(regularizer=reg, regularizer_strength=1)
        with warns:
            model.set_params(**params)

    @pytest.mark.parametrize(
        "solver_name, reg",
        [
            ("SVRG", "Ridge"),
            ("SVRG", "UnRegularized"),
            ("ProxSVRG", "Ridge"),
            ("ProxSVRG", "UnRegularized"),
            ("ProxSVRG", "Lasso"),
            ("ProxSVRG", "GroupLasso"),
        ],
    )
    @pytest.mark.parametrize(
        "obs",
        [
            nmo.observation_models.PoissonObservations(
                inverse_link_function=jax.nn.softplus
            )
        ],
    )
    @pytest.mark.parametrize("batch_size", [None, 1, 10])
    @pytest.mark.parametrize("stepsize", [None, 0.01])
    def test_glm_optimal_config_set_initial_state(
        self,
        solver_name,
        batch_size,
        stepsize,
        reg,
        obs,
        poissonGLM_model_instantiation,
    ):
        X, y, _, true_params, _ = poissonGLM_model_instantiation
        if reg == "GroupLasso":
            reg = nmo.regularizer.GroupLasso(mask=jnp.ones((1, X.shape[1])))
        model = nmo.glm.GLM(
            solver_name=solver_name,
            solver_kwargs=dict(batch_size=batch_size, stepsize=stepsize),
            observation_model=obs,
            regularizer=reg,
            regularizer_strength=None if reg == "UnRegularized" else 1.0,
        )
        opt_state = model.initialize_state(X, y, true_params)
        solver = inspect.getclosurevars(model._solver_run).nonlocals["solver"]

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
        "solver_name, reg",
        [
            ("SVRG", "Ridge"),
            ("SVRG", "UnRegularized"),
            ("ProxSVRG", "Ridge"),
            ("ProxSVRG", "UnRegularized"),
            ("ProxSVRG", "Lasso"),
        ],
    )
    @pytest.mark.parametrize(
        "obs",
        [
            nmo.observation_models.PoissonObservations(
                inverse_link_function=jax.nn.softplus
            )
        ],
    )
    @pytest.mark.parametrize("batch_size", [None, 1, 10])
    @pytest.mark.parametrize("stepsize", [None, 0.01])
    def test_glm_optimal_config_set_initial_state_pytree(
        self,
        solver_name,
        batch_size,
        stepsize,
        reg,
        obs,
        poissonGLM_model_instantiation_pytree,
    ):
        X, y, _, true_params, _ = poissonGLM_model_instantiation_pytree
        model = nmo.glm.GLM(
            solver_name=solver_name,
            solver_kwargs=dict(batch_size=batch_size, stepsize=stepsize),
            observation_model=obs,
            regularizer=reg,
            regularizer_strength=None if reg == "UnRegularized" else 1.0,
        )
        opt_state = model.initialize_state(X, y, true_params)
        solver = inspect.getclosurevars(model._solver_run).nonlocals["solver"]

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

    @pytest.mark.parametrize("batch_size", [None, 1, 10])
    @pytest.mark.parametrize("stepsize", [None, 0.01])
    @pytest.mark.parametrize(
        "regularizer", ["UnRegularized", "Ridge", "Lasso", "GroupLasso"]
    )
    @pytest.mark.parametrize(
        "solver_name, has_dafaults",
        [
            ("GradientDescent", False),
            ("LBFGS", False),
            ("ProximalGradient", False),
            ("SVRG", True),
            ("ProxSVRG", True),
        ],
    )
    @pytest.mark.parametrize(
        "inv_link, link_has_defaults", [(jax.nn.softplus, True), (jax.numpy.exp, False)]
    )
    @pytest.mark.parametrize(
        "observation_model, obs_has_defaults",
        [
            (nmo.observation_models.PoissonObservations, True),
            (nmo.observation_models.GammaObservations, False),
        ],
    )
    def test_optimize_solver_params(
        self,
        batch_size,
        stepsize,
        regularizer,
        solver_name,
        inv_link,
        observation_model,
        has_dafaults,
        link_has_defaults,
        obs_has_defaults,
        poissonGLM_model_instantiation,
    ):
        """Test the behavior of `optimize_solver_params` for different solver, regularizer, and observation model configurations."""
        obs = observation_model(inverse_link_function=inv_link)
        X, y, _, _, _ = poissonGLM_model_instantiation
        solver_kwargs = dict(stepsize=stepsize, batch_size=batch_size)
        # use glm static methods to check if the solver is batchable
        # if not pop the batch_size kwarg
        try:
            slv_class = nmo.glm.GLM._get_solver_class(solver_name)
            nmo.glm.GLM._check_solver_kwargs(slv_class, solver_kwargs)
        except NameError:
            solver_kwargs.pop("batch_size")

        # if the regularizer is not allowed for the solver type, return
        try:
            model = nmo.glm.GLM(
                regularizer=regularizer,
                solver_name=solver_name,
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
        elif has_dafaults and link_has_defaults and obs_has_defaults:
            # if defaults are available, a batch size is computed
            assert isinstance(kwargs["batch_size"], int) and kwargs["batch_size"] > 0
        elif "batch_size" in solver_kwargs:
            # return None otherwise
            assert isinstance(kwargs["batch_size"], type(None))

        if isinstance(stepsize, float):
            # if stepsize was provided, then it should be returned unchanged
            assert stepsize == kwargs["stepsize"]
        elif has_dafaults and link_has_defaults and obs_has_defaults:
            # if defaults are available, compute a value
            assert isinstance(kwargs["stepsize"], float) and kwargs["stepsize"] > 0
        else:
            # return None otherwise
            assert isinstance(kwargs["stepsize"], type(None))

    @pytest.mark.parametrize("batch_size", [None, 1, 10])
    @pytest.mark.parametrize("stepsize", [None, 0.01])
    @pytest.mark.parametrize(
        "regularizer", ["UnRegularized", "Ridge", "Lasso", "GroupLasso"]
    )
    @pytest.mark.parametrize(
        "solver_name, has_dafaults",
        [
            ("GradientDescent", False),
            ("LBFGS", False),
            ("ProximalGradient", False),
            ("SVRG", True),
            ("ProxSVRG", True),
        ],
    )
    @pytest.mark.parametrize(
        "inv_link, link_has_defaults", [(jax.nn.softplus, True), (jax.numpy.exp, False)]
    )
    @pytest.mark.parametrize(
        "observation_model, obs_has_defaults",
        [
            (nmo.observation_models.PoissonObservations, True),
            (nmo.observation_models.GammaObservations, False),
        ],
    )
    def test_optimize_solver_params_pytree(
        self,
        batch_size,
        stepsize,
        regularizer,
        solver_name,
        inv_link,
        observation_model,
        has_dafaults,
        link_has_defaults,
        obs_has_defaults,
        poissonGLM_model_instantiation_pytree,
    ):
        """Test the behavior of `optimize_solver_params` for different solver, regularizer, and observation model configurations."""
        obs = observation_model(inverse_link_function=inv_link)
        X, y, _, _, _ = poissonGLM_model_instantiation_pytree
        solver_kwargs = dict(stepsize=stepsize, batch_size=batch_size)
        # use glm static methods to check if the solver is batchable
        # if not pop the batch_size kwarg
        try:
            slv_class = nmo.glm.GLM._get_solver_class(solver_name)
            nmo.glm.GLM._check_solver_kwargs(slv_class, solver_kwargs)
        except NameError:
            solver_kwargs.pop("batch_size")

        # if the regularizer is not allowed for the solver type, return
        try:
            model = nmo.glm.GLM(
                regularizer=regularizer,
                solver_name=solver_name,
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
        elif has_dafaults and link_has_defaults and obs_has_defaults:
            # if defaults are available, a batch size is computed
            assert isinstance(kwargs["batch_size"], int) and kwargs["batch_size"] > 0
        elif "batch_size" in solver_kwargs:
            # return None otherwise
            assert isinstance(kwargs["batch_size"], type(None))

        if isinstance(stepsize, float):
            # if stepsize was provided, then it should be returned unchanged
            assert stepsize == kwargs["stepsize"]
        elif has_dafaults and link_has_defaults and obs_has_defaults:
            # if defaults are available, compute a value
            assert isinstance(kwargs["stepsize"], float) and kwargs["stepsize"] > 0
        else:
            # return None otherwise
            assert isinstance(kwargs["stepsize"], type(None))

    def test_repr_out(self, poissonGLM_model_instantiation):
        model = poissonGLM_model_instantiation[2]
        assert (
            repr(model)
            == "GLM(\n    observation_model=PoissonObservations(inverse_link_function=exp),"
            "\n    regularizer=UnRegularized(),\n    solver_name='GradientDescent'\n)"
        )


class TestPopulationGLM:
    """
    Unit tests for the PoissonGLM class.
    """

    #######################
    # Test model.__init__
    #######################
    @pytest.mark.parametrize(
        "regularizer, expectation",
        [
            (nmo.regularizer.Ridge(), does_not_raise()),
            (
                None,
                pytest.raises(
                    TypeError, match="The regularizer should be either a string from "
                ),
            ),
            (
                nmo.regularizer.Ridge,
                pytest.raises(
                    TypeError, match="The regularizer should be either a string from "
                ),
            ),
        ],
    )
    def test_solver_type(self, regularizer, expectation, population_glm_class):
        """
        Test that an error is raised if a non-compatible solver is passed.
        """
        with expectation:
            population_glm_class(regularizer=regularizer, regularizer_strength=1.0)

    def test_get_params(self):
        """
        Test that get_params() contains expected values.
        """
        expected_keys = {
            "feature_mask",
            "observation_model__inverse_link_function",
            "observation_model",
            "regularizer",
            "regularizer_strength",
            "solver_kwargs",
            "solver_name",
        }

        model = nmo.glm.PopulationGLM()

        expected_values = [
            model.feature_mask,
            model.observation_model.inverse_link_function,
            model.observation_model,
            model.regularizer,
            model.regularizer_strength,
            model.solver_kwargs,
            model.solver_name,
        ]

        assert set(model.get_params().keys()) == expected_keys
        assert list(model.get_params().values()) == expected_values

        # passing params
        model = nmo.glm.PopulationGLM(solver_name="LBFGS", regularizer="UnRegularized")

        expected_values = [
            model.feature_mask,
            model.observation_model.inverse_link_function,
            model.observation_model,
            model.regularizer,
            model.regularizer_strength,
            model.solver_kwargs,
            model.solver_name,
        ]

        assert set(model.get_params().keys()) == expected_keys
        assert list(model.get_params().values()) == expected_values

        # changing regularizer
        model.set_params(regularizer="Ridge", regularizer_strength=1.0)

        expected_values = [
            model.feature_mask,
            model.observation_model.inverse_link_function,
            model.observation_model,
            model.regularizer,
            model.regularizer_strength,
            model.solver_kwargs,
            model.solver_name,
        ]

        assert set(model.get_params().keys()) == expected_keys
        assert list(model.get_params().values()) == expected_values

        # changing solver
        model.solver_name = "ProximalGradient"

        expected_values = [
            model.feature_mask,
            model.observation_model.inverse_link_function,
            model.observation_model,
            model.regularizer,
            model.regularizer_strength,
            model.solver_kwargs,
            model.solver_name,
        ]

        assert set(model.get_params().keys()) == expected_keys
        assert list(model.get_params().values()) == expected_values

    @pytest.mark.parametrize(
        "observation, expectation",
        [
            (nmo.observation_models.PoissonObservations(), does_not_raise()),
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
        self, observation, expectation, population_glm_class, ridge_regularizer
    ):
        """
        Test initialization with different regularizer names. Check if an appropriate exception is raised
        when the regularizer name is not present in jaxopt.
        """
        with expectation:
            population_glm_class(
                regularizer=ridge_regularizer,
                observation_model=observation,
                regularizer_strength=1.0,
            )

    @pytest.mark.parametrize(
        "inv_link", [jnp.exp, lambda x: jnp.exp(x), jax.nn.softplus, jax.nn.relu]
    )
    def test_high_firing_rate_initialization(
        self, inv_link, example_X_y_high_firing_rates
    ):
        """Test firing rate regime that would not be initialized correctly in the original implementation."""
        model = nmo.glm.PopulationGLM(
            observation_model=nmo.observation_models.PoissonObservations(
                inverse_link_function=inv_link
            )
        )
        X, y = example_X_y_high_firing_rates
        model.initialize_params(X, y)

    @pytest.mark.parametrize(
        "X, y",
        [
            (jnp.zeros((2, 4)), jnp.ones((2, 2))),
            (jnp.zeros((2, 4)), jnp.ones((2, 3))),
        ],
    )
    def test_parameter_initialization(self, X, y, poisson_population_GLM_model):
        _, _, model, _, _ = poisson_population_GLM_model
        coef, inter = model._initialize_parameters(X, y)
        assert coef.shape == (X.shape[1], y.shape[1])
        assert inter.shape == (y.shape[1],)

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
    def test_fit_param_length(
        self, n_params, expectation, poisson_population_GLM_model
    ):
        """
        Test the `fit` method with different numbers of initial parameters.
        Check for correct number of parameters.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        if n_params == 0:
            init_params = tuple()
        elif n_params == 1:
            init_params = (true_params[0],)
        else:
            init_params = true_params + (true_params[0],) * (n_params - 2)
        with expectation:
            model.fit(X, y, init_params=init_params)

    @pytest.mark.parametrize(
        "reg, dof",
        [
            (nmo.regularizer.UnRegularized(), np.array([5, 5, 5])),
            (
                nmo.regularizer.Lasso(),
                np.array([3, 0, 0]),
            ),  # this lasso fit has only 3 coeff of the first neuron
            # surviving
            (nmo.regularizer.Ridge(), np.array([5, 5, 5])),
        ],
    )
    @pytest.mark.parametrize("n_samples", [1, 20])
    def test_estimate_dof_resid(
        self, n_samples, dof, reg, poisson_population_GLM_model
    ):
        """
        Test that the dof is an integer.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        strength = None if isinstance(reg, nmo.regularizer.UnRegularized) else 1.0
        model.set_params(regularizer=reg, regularizer_strength=strength)
        model.solver_name = model.regularizer.default_solver
        model.fit(X, y)
        num = model._estimate_resid_degrees_of_freedom(X, n_samples=n_samples)
        assert np.allclose(num, n_samples - dof - 1)

    @pytest.mark.parametrize(
        "dim_weights, expectation",
        [
            (
                0,
                pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
            ),
            (
                1,
                pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
            ),
            (
                2,
                does_not_raise(),
            ),
            (
                3,
                pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
            ),
        ],
    )
    def test_fit_weights_dimensionality(
        self, dim_weights, expectation, poisson_population_GLM_model
    ):
        """
        Test the `fit` method with weight matrices of different dimensionalities.
        Check for correct dimensionality.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        n_samples, n_features = X.shape
        n_neurons = 3
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
    def test_fit_intercepts_dimensionality(
        self, dim_intercepts, expectation, poisson_population_GLM_model
    ):
        """
        Test the `fit` method with intercepts of different dimensionalities. Check for correct dimensionality.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        n_samples, n_features = X.shape
        init_b = jnp.zeros((y.shape[1],) * dim_intercepts)
        init_w = jnp.zeros((n_features, y.shape[1]))
        with expectation:
            model.fit(X, y, init_params=(init_w, init_b))

    @pytest.mark.parametrize(
        "init_params, expectation",
        [
            ([jnp.zeros((5, 3)), jnp.zeros((3,))], does_not_raise()),
            (
                [[jnp.zeros((1, 5)), jnp.zeros((3,))]],
                pytest.raises(ValueError, match="Params must have length two."),
            ),
            (dict(p1=jnp.zeros((3, 3)), p2=jnp.zeros((3, 2))), pytest.raises(KeyError)),
            (
                (dict(p1=jnp.zeros((3, 3)), p2=jnp.zeros((2, 3))), jnp.zeros((3,))),
                pytest.raises(
                    TypeError, match=r"X and params\[0\] must be the same type"
                ),
            ),
            (0, pytest.raises(ValueError, match="Params must have length two.")),
            (
                {0, 1},
                pytest.raises(TypeError, match="Initial parameters must be array-like"),
            ),
            (
                [jnp.zeros((1, 5)), ""],
                pytest.raises(TypeError, match="Initial parameters must be array-like"),
            ),
            (
                ["", jnp.zeros((1,))],
                pytest.raises(TypeError, match="Initial parameters must be array-like"),
            ),
        ],
    )
    def test_fit_init_params_type(
        self, init_params, expectation, poisson_population_GLM_model
    ):
        """
        Test the `fit` method with various types of initial parameters. Ensure that the provided initial parameters
        are array-like.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
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
        self, delta_dim, expectation, poisson_population_GLM_model
    ):
        """
        Test the `fit` method with X input data of different dimensionalities. Ensure correct dimensionality for X.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        if delta_dim == -1:
            X = np.zeros((X.shape[0],))
        elif delta_dim == 1:
            X = np.zeros((X.shape[0], 1, X.shape[1]))
        with expectation:
            model.fit(X, y, init_params=true_params)

    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (-1, pytest.raises(ValueError, match="y must be two-dimensional")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="y must be two-dimensional")),
        ],
    )
    def test_fit_y_dimensionality(
        self, delta_dim, expectation, poisson_population_GLM_model
    ):
        """
        Test the `fit` method with y target data of different dimensionalities. Ensure correct dimensionality for y.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        if delta_dim == -1:
            y = y[:, 0]
        elif delta_dim == 1:
            y = np.zeros((*y.shape, 1))
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
    def test_fit_n_feature_consistency_weights(
        self, delta_n_features, expectation, poisson_population_GLM_model
    ):
        """
        Test the `fit` method for inconsistencies between data features and initial weights provided.
        Ensure the number of features align.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        init_w = jnp.zeros((X.shape[1] + delta_n_features, y.shape[1]))
        init_b = jnp.zeros(
            y.shape[1],
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
    def test_fit_n_feature_consistency_x(
        self, delta_n_features, expectation, poisson_population_GLM_model
    ):
        """
        Test the `fit` method for inconsistencies between data features and model's expectations.
        Ensure the number of features in X aligns.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
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
        self, delta_tp, expectation, poisson_population_GLM_model
    ):
        """
        Test the `fit` method for inconsistencies in time-points in data X. Ensure the correct number of time-points.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
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
        self, delta_tp, expectation, poisson_population_GLM_model
    ):
        """
        Test the `fit` method for inconsistencies in time-points in y. Ensure the correct number of time-points.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        y = jnp.zeros((y.shape[0] + delta_tp,) + y.shape[1:])
        with expectation:
            model.fit(X, y, init_params=true_params)

    def test_fit_mask_grouplasso(self, group_sparse_poisson_glm_model_instantiation):
        """Test that the group lasso fit goes through"""
        X, y, model, params, rate, mask = group_sparse_poisson_glm_model_instantiation
        model.set_params(
            regularizer=nmo.regularizer.GroupLasso(mask=mask),
            solver_name="ProximalGradient",
            regularizer_strength=1.0,
        )
        model.fit(X, y)

    def test_fit_pytree_equivalence(
        self, poisson_population_GLM_model, poisson_population_GLM_model_pytree
    ):
        """Check that the glm fit with pytree learns the same parameters."""
        # required for numerical precision of coeffs
        jax.config.update("jax_enable_x64", True)
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        X_tree, _, model_tree, true_params_tree, _ = poisson_population_GLM_model_pytree
        # fit both models
        model.fit(X, y, init_params=true_params)
        model_tree.fit(X_tree, y, init_params=true_params_tree)

        # get the flat parameters
        flat_coef = np.concatenate(
            jax.tree_util.tree_flatten(model_tree.coef_)[0], axis=0
        )

        # assert equivalence of solutions
        assert np.allclose(model.coef_, flat_coef)
        assert np.allclose(model.intercept_, model_tree.intercept_)
        assert np.allclose(model.score(X, y), model_tree.score(X_tree, y))
        assert np.allclose(model.predict(X), model_tree.predict(X_tree))

    def test_fit_pytree_equivalence_gamma(
        self, gamma_population_GLM_model, gamma_population_GLM_model_pytree
    ):
        """Check that the glm fit with pytree learns the same parameters."""
        # required for numerical precision of coeffs
        jax.config.update("jax_enable_x64", True)
        X, y, model, true_params, firing_rate = gamma_population_GLM_model
        X_tree, _, model_tree, true_params_tree, _ = gamma_population_GLM_model_pytree
        # fit both models
        model.fit(X, y, init_params=true_params)
        model_tree.fit(X_tree, y, init_params=true_params_tree)

        # get the flat parameters
        flat_coef = np.concatenate(
            jax.tree_util.tree_flatten(model_tree.coef_)[0], axis=0
        )

        # assert equivalence of solutions
        assert np.allclose(model.coef_, flat_coef)
        assert np.allclose(model.intercept_, model_tree.intercept_)
        assert np.allclose(model.score(X, y), model_tree.score(X_tree, y))
        assert np.allclose(model.predict(X), model_tree.predict(X_tree))
        assert np.allclose(model.scale_, model_tree.scale_)

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
        self, fill_val, expectation, poisson_population_GLM_model
    ):
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        X.fill(fill_val)
        with expectation:
            model.fit(X, y)

    @pytest.mark.parametrize("inv_link", [jnp.exp, lambda x: 1 / x])
    def test_fit_gamma_glm(self, inv_link, gamma_population_GLM_model):
        X, y, model, true_params, firing_rate = gamma_population_GLM_model
        model.observation_model.inverse_link_function = inv_link
        model.fit(X, y)

    @pytest.mark.parametrize("inv_link", [jnp.exp, lambda x: 1 / x])
    def test_fit_set_scale(self, inv_link, gamma_population_GLM_model):
        X, y, model, true_params, firing_rate = gamma_population_GLM_model
        model.observation_model.inverse_link_function = inv_link
        model.fit(X, y)
        assert np.all(model.scale_ != 1)

    def test_fit_scale_array(self, gamma_population_GLM_model):
        X, y, model, true_params, firing_rate = gamma_population_GLM_model
        model.fit(X, y)
        assert model.scale_.size == y.shape[1]

    #######################
    # Test model.initialize_solver
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
    def test_initialize_solver_param_length(
        self, n_params, expectation, poisson_population_GLM_model
    ):
        """
        Test the `initialize_solver` method with different numbers of initial parameters.
        Check for correct number of parameters.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        if n_params == 0:
            init_params = tuple()
        elif n_params == 1:
            init_params = (true_params[0],)
        else:
            init_params = true_params + (true_params[0],) * (n_params - 2)
        with expectation:
            params = model.initialize_params(X, y, init_params=init_params)
            model.initialize_state(X, y, params)

    @pytest.mark.parametrize(
        "dim_weights, expectation",
        [
            (
                0,
                pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
            ),
            (
                1,
                pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
            ),
            (
                2,
                does_not_raise(),
            ),
            (
                3,
                pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
            ),
        ],
    )
    def test_initialize_solver_weights_dimensionality(
        self, dim_weights, expectation, poisson_population_GLM_model
    ):
        """
        Test the `initialize_solver` method with weight matrices of different dimensionalities.
        Check for correct dimensionality.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        n_samples, n_features = X.shape
        n_neurons = 3
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
            model.initialize_state(X, y, params)

    @pytest.mark.parametrize(
        "dim_intercepts, expectation",
        [
            (0, pytest.raises(ValueError, match=r"params\[1\] must be of shape")),
            (1, does_not_raise()),
            (2, pytest.raises(ValueError, match=r"params\[1\] must be of shape")),
            (3, pytest.raises(ValueError, match=r"params\[1\] must be of shape")),
        ],
    )
    def test_initialize_solver_intercepts_dimensionality(
        self, dim_intercepts, expectation, poisson_population_GLM_model
    ):
        """
        Test the `initialize_solver` method with intercepts of different dimensionalities.

        Check for correct dimensionality.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        n_samples, n_features = X.shape
        init_b = jnp.zeros((y.shape[1],) * dim_intercepts)
        init_w = jnp.zeros((n_features, y.shape[1]))
        with expectation:
            params = model.initialize_params(X, y, init_params=(init_w, init_b))
            model.initialize_state(X, y, params)

    @pytest.mark.parametrize(
        "init_params, expectation",
        [
            ([jnp.zeros((5, 3)), jnp.zeros((3,))], does_not_raise()),
            (
                [[jnp.zeros((1, 5)), jnp.zeros((3,))]],
                pytest.raises(ValueError, match="Params must have length two."),
            ),
            (dict(p1=jnp.zeros((3, 3)), p2=jnp.zeros((3, 2))), pytest.raises(KeyError)),
            (
                (dict(p1=jnp.zeros((3, 3)), p2=jnp.zeros((2, 3))), jnp.zeros((3,))),
                pytest.raises(
                    TypeError, match=r"X and params\[0\] must be the same type"
                ),
            ),
            (0, pytest.raises(ValueError, match="Params must have length two.")),
            (
                {0, 1},
                pytest.raises(TypeError, match="Initial parameters must be array-like"),
            ),
            (
                [jnp.zeros((1, 5)), ""],
                pytest.raises(TypeError, match="Initial parameters must be array-like"),
            ),
            (
                ["", jnp.zeros((1,))],
                pytest.raises(TypeError, match="Initial parameters must be array-like"),
            ),
        ],
    )
    def test_initialize_solver_init_params_type(
        self, init_params, expectation, poisson_population_GLM_model
    ):
        """
        Test the `initialize_solver` method with various types of initial parameters.

        Ensure that the provided initial parameters are array-like.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        with expectation:
            params = model.initialize_params(X, y, init_params=init_params)
            model.initialize_state(X, y, params)

    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (-1, pytest.raises(ValueError, match="X must be two-dimensional")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="X must be two-dimensional")),
        ],
    )
    def test_initialize_solver_x_dimensionality(
        self, delta_dim, expectation, poisson_population_GLM_model
    ):
        """
        Test the `initialize_solver` method with X input data of different dimensionalities.

        Ensure correct dimensionality for X.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        if delta_dim == -1:
            X = np.zeros((X.shape[0],))
        elif delta_dim == 1:
            X = np.zeros((X.shape[0], 1, X.shape[1]))
        with expectation:
            params = model.initialize_params(X, y, init_params=true_params)
            model.initialize_state(X, y, params)

    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (-1, pytest.raises(ValueError, match="y must be two-dimensional")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="y must be two-dimensional")),
        ],
    )
    def test_initialize_solver_y_dimensionality(
        self, delta_dim, expectation, poisson_population_GLM_model
    ):
        """
        Test the `initialize_solver` method with y target data of different dimensionalities.

        Ensure correct dimensionality for y.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        if delta_dim == -1:
            y = y[:, 0]
        elif delta_dim == 1:
            y = np.zeros((*y.shape, 1))
        with expectation:
            params = model.initialize_params(X, y, init_params=true_params)
            model.initialize_state(X, y, params)

    @pytest.mark.parametrize(
        "delta_n_features, expectation",
        [
            (-1, pytest.raises(ValueError, match="Inconsistent number of features")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="Inconsistent number of features")),
        ],
    )
    def test_initialize_solver_n_feature_consistency_weights(
        self, delta_n_features, expectation, poisson_population_GLM_model
    ):
        """
        Test the `initialize_solver` method for inconsistencies between data features and initial weights provided.
        Ensure the number of features align.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        init_w = jnp.zeros((X.shape[1] + delta_n_features, y.shape[1]))
        init_b = jnp.zeros(
            y.shape[1],
        )
        with expectation:
            params = model.initialize_params(X, y, init_params=(init_w, init_b))
            model.initialize_state(X, y, params)

    @pytest.mark.parametrize(
        "delta_n_features, expectation",
        [
            (-1, pytest.raises(ValueError, match="Inconsistent number of features")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="Inconsistent number of features")),
        ],
    )
    def test_initialize_solver_n_feature_consistency_x(
        self, delta_n_features, expectation, poisson_population_GLM_model
    ):
        """
        Test the `initialize_solver` method for inconsistencies between data features and model's expectations.
        Ensure the number of features in X aligns.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        if delta_n_features == 1:
            X = jnp.concatenate((X, jnp.zeros((X.shape[0], 1))), axis=1)
        elif delta_n_features == -1:
            X = X[..., :-1]
        with expectation:
            params = model.initialize_params(X, y, init_params=true_params)
            model.initialize_state(X, y, params)

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
        self, delta_tp, expectation, poisson_population_GLM_model
    ):
        """
        Test the `initialize_solver` method for inconsistencies in time-points in data X.

        Ensure the correct number of time-points.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        X = jnp.zeros((X.shape[0] + delta_tp,) + X.shape[1:])
        with expectation:
            params = model.initialize_params(X, y, init_params=true_params)
            model.initialize_state(X, y, params)

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
        self, delta_tp, expectation, poisson_population_GLM_model
    ):
        """
        Test the `initialize_solver` method for inconsistencies in time-points in y.

        Ensure the correct number of time-points.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        y = jnp.zeros((y.shape[0] + delta_tp,) + y.shape[1:])
        with expectation:
            params = model.initialize_params(X, y, init_params=true_params)
            model.initialize_state(X, y, params)

    def test_initialize_solver_mask_grouplasso(
        self, group_sparse_poisson_glm_model_instantiation
    ):
        """Test that the group lasso initialize_solver goes through"""
        X, y, model, params, rate, mask = group_sparse_poisson_glm_model_instantiation
        model.set_params(
            regularizer_strength=1.0,
            regularizer=nmo.regularizer.GroupLasso(mask=mask),
            solver_name="ProximalGradient",
        )
        params = model.initialize_params(X, y)
        model.initialize_state(X, y, params)

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
        self, fill_val, expectation, poisson_population_GLM_model
    ):
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        X.fill(fill_val)
        with expectation:
            params = model.initialize_params(X, y)
            model.initialize_state(X, y, params)

    #######################
    # Test model.update
    #######################
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
    def test_update_n_samples(
        self, n_samples, expectation, batch_size, poisson_population_GLM_model
    ):
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        params = model.initialize_params(X, y)
        state = model.initialize_state(X, y, params)
        with expectation:
            model.update(
                params, state, X[:batch_size], y[:batch_size], n_samples=n_samples
            )

    @pytest.mark.parametrize("batch_size", [1, 10])
    def test_update_params_stored(self, batch_size, poisson_population_GLM_model):
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        params = model.initialize_params(X, y)
        state = model.initialize_state(X, y, params)
        assert model.coef_ is None
        assert model.intercept_ is None
        assert model.scale_ is None
        _, _ = model.update(params, state, X[:batch_size], y[:batch_size])
        assert model.coef_ is not None
        assert model.intercept_ is not None
        assert model.scale_ is not None

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
        self, delta_dim, expectation, poisson_population_GLM_model
    ):
        """
        Test the `score` method with X input data of different dimensionalities. Ensure correct dimensionality for X.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
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
                    ValueError, match="y must be two-dimensional, with shape"
                ),
            ),
            (0, does_not_raise()),
            (
                1,
                pytest.raises(
                    ValueError, match="y must be two-dimensional, with shape"
                ),
            ),
        ],
    )
    def test_score_y_dimensionality(
        self, delta_dim, expectation, poisson_population_GLM_model
    ):
        """
        Test the `score` method with y of different dimensionalities.
        Ensure correct dimensionality for y.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        if delta_dim == -1:
            y = y[:, 0]
        elif delta_dim == 1:
            y = np.zeros((*y.shape, 1))
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
        self, delta_n_features, expectation, poisson_population_GLM_model
    ):
        """
        Test the `score` method for inconsistencies in features of X.
        Ensure the number of features in X aligns with the model params.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        if delta_n_features == 1:
            X = jnp.concatenate((X, jnp.zeros((X.shape[0], 1))), axis=1)
        elif delta_n_features == -1:
            X = X[..., :-1]
        with expectation:
            model.score(X, y)

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
    def test_predict_is_fit_population(
        self, is_fit, expectation, poisson_population_GLM_model
    ):
        """
        Test the `score` method on models based on their fit status.
        Ensure scoring is only possible on fitted models.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        if is_fit:
            model.fit(X, y)
        with expectation:
            model.predict(X)

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
        self, delta_tp, expectation, poisson_population_GLM_model
    ):
        """
        Test the `score` method for inconsistencies in time-points in X.
        Ensure that the number of time-points in X and y matches.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
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
        self, delta_tp, expectation, poisson_population_GLM_model
    ):
        """
        Test the `score` method for inconsistencies in time-points in y.
        Ensure that the number of time-points in X and y matches.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        y = jnp.zeros((y.shape[0] + delta_tp,) + y.shape[1:])
        with expectation:
            model.score(X, y)

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
    def test_score_type_r2(self, score_type, expectation, poisson_population_GLM_model):
        """
        Test the `score` method for unsupported scoring types.
        Ensure only valid score types are used.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        with expectation:
            model.score(X, y, score_type=score_type)

    def test_loglikelihood_against_scipy_stats(self, poisson_population_GLM_model):
        """
        Compare the model's log-likelihood computation against `jax.scipy`.
        Ensure consistent and correct calculations.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        # set model coeff
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        model._initialize_feature_mask(X, y)
        # get the rate
        mean_firing = model.predict(X)
        # compute the log-likelihood using jax.scipy
        mean_ll_jax = jax.scipy.stats.poisson.logpmf(y, mean_firing).mean()
        model_ll = model.score(X, y, score_type="log-likelihood")
        if not np.allclose(mean_ll_jax, model_ll):
            raise ValueError(
                "Log-likelihood of PoissonModel does not match" "that of jax.scipy!"
            )

    @pytest.mark.parametrize("inv_link", [jnp.exp, lambda x: 1 / x])
    def test_score_gamma_glm(self, inv_link, gamma_population_GLM_model):
        X, y, model, true_params, firing_rate = gamma_population_GLM_model
        model.observation_model.inverse_link_function = inv_link
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        model.scale_ = np.ones((y.shape[1]))
        model.score(X, y)

    @pytest.mark.parametrize(
        "score_type", ["log-likelihood", "pseudo-r2-McFadden", "pseudo-r2-Cohen"]
    )
    def test_score_aggregation_ndim(self, score_type, poisson_population_GLM_model):
        """
        Test that the aggregate samples returns the right dimensional object.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
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

    #######################
    # Test model.predict
    #######################
    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (-1, pytest.raises(ValueError, match="X must be two-dimensional")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="X must be two-dimensional")),
        ],
    )
    def test_predict_x_dimensionality(
        self, delta_dim, expectation, poisson_population_GLM_model
    ):
        """
        Test the `predict` method with x input data of different dimensionalities.
        Ensure correct dimensionality for x.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
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
        self, delta_n_features, expectation, poisson_population_GLM_model
    ):
        """
        Test the `predict` method ensuring the number of features in x input data
        is consistent with the model's `model.coef_`.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        model._initialize_feature_mask(X, y)
        if delta_n_features == 1:
            X = jnp.concatenate((X, jnp.zeros((X.shape[0], 1))), axis=1)
        elif delta_n_features == -1:
            X = X[..., :-1]
        with expectation:
            model.predict(X)

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
    def test_predict_is_fit(self, is_fit, expectation, poissonGLM_model_instantiation):
        """
        Test the `score` method on models based on their fit status.
        Ensure scoring is only possible on fitted models.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        if is_fit:
            model.fit(X, y)
        with expectation:
            model.predict(X)

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
        self, delta_dim, expectation, poisson_population_GLM_model
    ):
        """
        Test the `simulate` method with input data of different dimensionalities.
        Ensure correct dimensionality for input.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
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
        "input_type, expected_out_type",
        [
            (TsdFrame, TsdFrame),
            (np.ndarray, jnp.ndarray),
            (jnp.ndarray, jnp.ndarray),
        ],
    )
    def test_simulate_pynapple(
        self, input_type, expected_out_type, poisson_population_GLM_model
    ):
        """
        Test that the `simulate` method retturns the expected data type for different allowed inputs.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        model._initialize_feature_mask(X, y)
        if input_type == TsdFrame:
            X = TsdFrame(t=np.arange(X.shape[0]), d=X)

        count, rate = model.simulate(
            random_key=jax.random.key(123),
            feedforward_input=X,
        )
        assert isinstance(count, expected_out_type)
        assert isinstance(rate, expected_out_type)

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
    def test_simulate_is_fit(self, is_fit, expectation, poisson_population_GLM_model):
        """
        Test if the model raises a ValueError when trying to simulate before it's fitted.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        if is_fit:
            model.coef_ = true_params[0]
            model.intercept_ = true_params[1]
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
        self, delta_features, expectation, poisson_population_GLM_model
    ):
        """
        Test the `simulate` method ensuring the number of features in `feedforward_input` is
        consistent with the model's expected number of features.

        Notes
        -----
        The total feature number `model.coef_.shape[1]` must be equal to
        `feedforward_input.shape[2] + coupling_basis.shape[1]*n_neurons`
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
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

    def test_simulate_feedforward_glm(self, poissonGLM_model_instantiation):
        """Test that simulate goes through"""
        X, y, model, params, rate = poissonGLM_model_instantiation
        model.coef_ = params[0]
        model.intercept_ = params[1]
        ysim, ratesim = model.simulate(jax.random.key(123), X)
        # check that the expected dimensionality is returned
        assert ysim.ndim == 1
        assert ratesim.ndim == 1
        # check that the rates and spikes has the same shape
        assert ratesim.shape[0] == ysim.shape[0]
        # check the time point number is that expected (same as the input)
        assert ysim.shape[0] == X.shape[0]

    @pytest.mark.parametrize("inv_link", [jnp.exp, lambda x: 1 / x])
    def test_simulate_gamma_glm(self, inv_link, gamma_population_GLM_model):
        X, y, model, true_params, firing_rate = gamma_population_GLM_model
        model.observation_model.inverse_link_function = inv_link
        model.feature_mask = jnp.ones((X.shape[1], y.shape[1]))
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        model.scale_ = jnp.ones((y.shape[1]))
        ysim, ratesim = model.simulate(jax.random.PRNGKey(123), X)
        assert ysim.shape == y.shape
        assert ratesim.shape == y.shape

    #######################################
    # Compare with standard implementation
    #######################################
    def test_deviance_against_statsmodels(self, poisson_population_GLM_model):
        """
        Compare fitted parameters to statsmodels.
        Assesses if the model estimates are close to statsmodels' results.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        # set model coeff
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        # get the rate
        dev = sm.families.Poisson().deviance(y, firing_rate)
        dev_model = model.observation_model.deviance(y, firing_rate).sum()
        if not np.allclose(dev, dev_model):
            raise ValueError("Deviance doesn't match statsmodels!")

    def test_compatibility_with_sklearn_cv(self, poisson_population_GLM_model):
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        param_grid = {"solver_name": ["BFGS", "GradientDescent"]}
        GridSearchCV(model, param_grid).fit(X, y)

    def test_compatibility_with_sklearn_cv_gamma(self, gamma_population_GLM_model):
        X, y, model, true_params, firing_rate = gamma_population_GLM_model
        param_grid = {"solver_name": ["BFGS", "GradientDescent"]}
        GridSearchCV(model, param_grid).fit(X, y)

    def test_sklearn_clone(self, poisson_population_GLM_model):
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        model.fit(X, y)
        cloned = sklearn.clone(model)
        assert cloned.feature_mask is None, "cloned GLM shouldn't have feature mask!"
        assert model.feature_mask is not None, "fit GLM should have feature mask!"

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
    def test_feature_mask_setter(self, mask, expectation, poisson_population_GLM_model):
        _, _, model, _, _ = poisson_population_GLM_model
        with expectation:
            model.feature_mask = mask

    @pytest.mark.parametrize(
        "mask, expectation",
        [
            (np.array([0, 1, 1] * 5).reshape(5, 3), does_not_raise()),
            (
                np.array([0, 1, 1] * 4).reshape(4, 3),
                pytest.raises(ValueError, match="Inconsistent number of features"),
            ),
            (
                np.array([0, 1, 1, 1] * 5).reshape(5, 4),
                pytest.raises(ValueError, match="Inconsistent number of neurons"),
            ),
            (
                {"input_1": np.array([0, 1, 0]), "input_2": np.array([1, 0, 1])},
                pytest.raises(
                    TypeError, match="feature_mask and X must have the same structure"
                ),
            ),
            (
                {"input_1": np.array([0, 1, 0, 1]), "input_2": np.array([1, 0, 1, 0])},
                pytest.raises(
                    TypeError, match="feature_mask and X must have the same structure"
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("attr_name", ["fit", "predict", "score"])
    def test_feature_mask_compatibility_fit(
        self, mask, expectation, attr_name, poisson_population_GLM_model
    ):
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        model.feature_mask = mask
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        with expectation:
            if attr_name == "predict":
                getattr(model, attr_name)(X)
            else:
                getattr(model, attr_name)(X, y)

    @pytest.mark.parametrize(
        "mask, expectation",
        [
            (
                np.array([0, 1, 1] * 5).reshape(5, 3),
                pytest.raises(
                    TypeError, match="feature_mask and X must have the same structure"
                ),
            ),
            (
                np.array([0, 1, 1] * 4).reshape(4, 3),
                pytest.raises(
                    TypeError, match="feature_mask and X must have the same structure"
                ),
            ),
            (
                np.array([0, 1, 1, 1] * 5).reshape(5, 4),
                pytest.raises(
                    TypeError, match="feature_mask and X must have the same structure"
                ),
            ),
            (
                {"input_1": np.array([0, 1, 0]), "input_2": np.array([1, 0, 1])},
                does_not_raise(),
            ),
            (
                {"input_1": np.array([0, 1, 0, 1]), "input_2": np.array([1, 0, 1, 0])},
                pytest.raises(ValueError, match="Inconsistent number of neurons"),
            ),
            (
                {"input_1": np.array([0, 1, 0])},
                pytest.raises(
                    TypeError, match="feature_mask and X must have the same structure"
                ),
            ),
            (
                {"input_1": np.array([0, 1, 0, 1])},
                pytest.raises(
                    TypeError, match="feature_mask and X must have the same structure"
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("attr_name", ["fit", "predict", "score"])
    def test_feature_mask_compatibility_fit_tree(
        self, mask, expectation, attr_name, poisson_population_GLM_model_pytree
    ):
        X, y, model, true_params, firing_rate = poisson_population_GLM_model_pytree
        model.feature_mask = mask
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        with expectation:
            if attr_name == "predict":
                getattr(model, attr_name)(X)
            else:
                getattr(model, attr_name)(X, y)

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
            (nmo.regularizer.Ridge(), 1.0, "LBFGS", {"stepsize": 0.1, "tol": 10**-14}),
            (
                nmo.regularizer.Lasso(),
                0.001,
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
    def test_masked_fit_vs_loop(
        self,
        regularizer,
        regularizer_strength,
        solver_name,
        solver_kwargs,
        mask,
        poisson_population_GLM_model,
        poisson_population_GLM_model_pytree,
    ):
        jax.config.update("jax_enable_x64", True)
        if isinstance(mask, dict):
            X, y, _, true_params, firing_rate = poisson_population_GLM_model_pytree

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
            X, y, _, true_params, firing_rate = poisson_population_GLM_model

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

    @pytest.mark.parametrize("reg", ["Ridge", "Lasso", "GroupLasso"])
    def test_waning_solver_reg_str(self, reg):
        # check that a warning is triggered
        # if no param is passed
        with pytest.warns(UserWarning):
            nmo.glm.GLM(regularizer=reg)

        # check that the warning is not triggered
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = nmo.glm.GLM(regularizer=reg, regularizer_strength=1.0)

        # reset to unregularized
        model.set_params(regularizer="UnRegularized", regularizer_strength=None)
        with pytest.warns(UserWarning):
            nmo.glm.GLM(regularizer=reg)

    @pytest.mark.parametrize("reg", ["Ridge", "Lasso", "GroupLasso"])
    def test_reg_strength_reset(self, reg):
        model = nmo.glm.PopulationGLM(regularizer=reg, regularizer_strength=1.0)
        with pytest.warns(
            UserWarning,
            match="Unused parameter `regularizer_strength` for UnRegularized GLM",
        ):
            model.regularizer = "UnRegularized"
        model.regularizer_strength = None
        with pytest.warns(
            UserWarning, match="Caution: regularizer strength has not been set"
        ):
            model.regularizer = "Ridge"

    @pytest.mark.parametrize(
        "solver_name, reg",
        [
            ("SVRG", "Ridge"),
            ("SVRG", "UnRegularized"),
            ("ProxSVRG", "Ridge"),
            ("ProxSVRG", "UnRegularized"),
            ("ProxSVRG", "Lasso"),
            ("ProxSVRG", "GroupLasso"),
        ],
    )
    @pytest.mark.parametrize(
        "obs",
        [
            nmo.observation_models.PoissonObservations(
                inverse_link_function=jax.nn.softplus
            )
        ],
    )
    @pytest.mark.parametrize("batch_size", [None, 1, 10])
    @pytest.mark.parametrize("stepsize", [None, 0.01])
    def test_glm_optimal_config_set_initial_state(
        self, solver_name, batch_size, stepsize, reg, obs, poisson_population_GLM_model
    ):
        X, y, _, true_params, _ = poisson_population_GLM_model

        if reg == "GroupLasso":
            reg = nmo.regularizer.GroupLasso(mask=jnp.ones((1, X.shape[1])))

        model = nmo.glm.PopulationGLM(
            solver_name=solver_name,
            solver_kwargs=dict(batch_size=batch_size, stepsize=stepsize),
            observation_model=obs,
            regularizer=reg,
            regularizer_strength=None if reg == "UnRegularized" else 1.0,
        )
        opt_state = model.initialize_state(X, y, true_params)
        solver = inspect.getclosurevars(model._solver_run).nonlocals["solver"]

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
        "solver_name, reg",
        [
            ("SVRG", "Ridge"),
            ("SVRG", "UnRegularized"),
            ("ProxSVRG", "Ridge"),
            ("ProxSVRG", "UnRegularized"),
            ("ProxSVRG", "Lasso"),
        ],
    )
    @pytest.mark.parametrize(
        "obs",
        [
            nmo.observation_models.PoissonObservations(
                inverse_link_function=jax.nn.softplus
            )
        ],
    )
    @pytest.mark.parametrize("batch_size", [None, 1, 10])
    @pytest.mark.parametrize("stepsize", [None, 0.01])
    def test_glm_optimal_config_set_initial_state_pytree(
        self,
        solver_name,
        batch_size,
        stepsize,
        reg,
        obs,
        poisson_population_GLM_model_pytree,
    ):
        X, y, _, true_params, _ = poisson_population_GLM_model_pytree
        model = nmo.glm.PopulationGLM(
            solver_name=solver_name,
            solver_kwargs=dict(batch_size=batch_size, stepsize=stepsize),
            observation_model=obs,
            regularizer=reg,
            regularizer_strength=None if reg == "UnRegularized" else 1.0,
        )
        opt_state = model.initialize_state(X, y, true_params)
        solver = inspect.getclosurevars(model._solver_run).nonlocals["solver"]

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
        "params, warns",
        [
            # set regularizer
            (
                {"regularizer": "Ridge"},
                pytest.warns(
                    UserWarning, match="Caution: regularizer strength has not been set"
                ),
            ),
            (
                {"regularizer": "Lasso"},
                pytest.warns(
                    UserWarning, match="Caution: regularizer strength has not been set"
                ),
            ),
            (
                {"regularizer": "GroupLasso"},
                pytest.warns(
                    UserWarning, match="Caution: regularizer strength has not been set"
                ),
            ),
            ({"regularizer": "UnRegularized"}, does_not_raise()),
            # set both None or number
            (
                {"regularizer": "Ridge", "regularizer_strength": None},
                pytest.warns(
                    UserWarning, match="Caution: regularizer strength has not been set"
                ),
            ),
            ({"regularizer": "Ridge", "regularizer_strength": 1.0}, does_not_raise()),
            (
                {"regularizer": "Lasso", "regularizer_strength": None},
                pytest.warns(
                    UserWarning, match="Caution: regularizer strength has not been set"
                ),
            ),
            ({"regularizer": "Lasso", "regularizer_strength": 1.0}, does_not_raise()),
            (
                {"regularizer": "GroupLasso", "regularizer_strength": None},
                pytest.warns(
                    UserWarning, match="Caution: regularizer strength has not been set"
                ),
            ),
            (
                {"regularizer": "GroupLasso", "regularizer_strength": 1.0},
                does_not_raise(),
            ),
            (
                {"regularizer": "UnRegularized", "regularizer_strength": None},
                does_not_raise(),
            ),
            (
                {"regularizer": "UnRegularized", "regularizer_strength": 1.0},
                pytest.warns(
                    UserWarning,
                    match="Unused parameter `regularizer_strength` for UnRegularized GLM",
                ),
            ),
            # set regularizer str only
            (
                {"regularizer_strength": 1.0},
                pytest.warns(
                    UserWarning,
                    match="Unused parameter `regularizer_strength` for UnRegularized GLM",
                ),
            ),
            ({"regularizer_strength": None}, does_not_raise()),
        ],
    )
    def test_reg_set_params(self, params, warns):
        model = nmo.glm.PopulationGLM()
        with warns:
            model.set_params(**params)

    @pytest.mark.parametrize(
        "params, warns",
        [
            # set regularizer str only
            ({"regularizer_strength": 1.0}, does_not_raise()),
            (
                {"regularizer_strength": None},
                pytest.warns(
                    UserWarning, match="Caution: regularizer strength has not been set"
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("reg", ["Ridge", "Lasso", "GroupLasso"])
    def test_reg_set_params_reg_str_only(self, params, warns, reg):
        model = nmo.glm.PopulationGLM(regularizer=reg, regularizer_strength=1)
        with warns:
            model.set_params(**params)

    @pytest.mark.parametrize("batch_size", [None, 1, 10])
    @pytest.mark.parametrize("stepsize", [None, 0.01])
    @pytest.mark.parametrize(
        "regularizer", ["UnRegularized", "Ridge", "Lasso", "GroupLasso"]
    )
    @pytest.mark.parametrize(
        "solver_name, has_dafaults",
        [
            ("GradientDescent", False),
            ("LBFGS", False),
            ("ProximalGradient", False),
            ("SVRG", True),
            ("ProxSVRG", True),
        ],
    )
    @pytest.mark.parametrize(
        "inv_link, link_has_defaults", [(jax.nn.softplus, True), (jax.numpy.exp, False)]
    )
    @pytest.mark.parametrize(
        "observation_model, obs_has_defaults",
        [
            (nmo.observation_models.PoissonObservations, True),
            (nmo.observation_models.GammaObservations, False),
        ],
    )
    def test_optimize_solver_params(
        self,
        batch_size,
        stepsize,
        regularizer,
        solver_name,
        inv_link,
        observation_model,
        has_dafaults,
        link_has_defaults,
        obs_has_defaults,
        poisson_population_GLM_model,
    ):
        """Test the behavior of `optimize_solver_params` for different solver, regularizer, and observation model configurations."""
        obs = observation_model(inverse_link_function=inv_link)
        X, y, _, _, _ = poisson_population_GLM_model
        solver_kwargs = dict(stepsize=stepsize, batch_size=batch_size)
        # use glm static methods to check if the solver is batchable
        # if not pop the batch_size kwarg
        try:
            slv_class = nmo.glm.PopulationGLM._get_solver_class(solver_name)
            nmo.glm.PopulationGLM._check_solver_kwargs(slv_class, solver_kwargs)
        except NameError:
            solver_kwargs.pop("batch_size")

        # if the regularizer is not allowed for the solver type, return
        try:
            model = nmo.glm.PopulationGLM(
                regularizer=regularizer,
                solver_name=solver_name,
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
        elif has_dafaults and link_has_defaults and obs_has_defaults:
            # if defaults are available, a batch size is computed
            assert isinstance(kwargs["batch_size"], int) and kwargs["batch_size"] > 0
        elif "batch_size" in solver_kwargs:
            # return None otherwise
            assert isinstance(kwargs["batch_size"], type(None))

        if isinstance(stepsize, float):
            # if stepsize was provided, then it should be returned unchanged
            assert stepsize == kwargs["stepsize"]
        elif has_dafaults and link_has_defaults and obs_has_defaults:
            # if defaults are available, compute a value
            assert isinstance(kwargs["stepsize"], float) and kwargs["stepsize"] > 0
        else:
            # return None otherwise
            assert isinstance(kwargs["stepsize"], type(None))

    @pytest.mark.parametrize("batch_size", [None, 1, 10])
    @pytest.mark.parametrize("stepsize", [None, 0.01])
    @pytest.mark.parametrize(
        "regularizer", ["UnRegularized", "Ridge", "Lasso", "GroupLasso"]
    )
    @pytest.mark.parametrize(
        "solver_name, has_dafaults",
        [
            ("GradientDescent", False),
            ("LBFGS", False),
            ("ProximalGradient", False),
            ("SVRG", True),
            ("ProxSVRG", True),
        ],
    )
    @pytest.mark.parametrize(
        "inv_link, link_has_defaults", [(jax.nn.softplus, True), (jax.numpy.exp, False)]
    )
    @pytest.mark.parametrize(
        "observation_model, obs_has_defaults",
        [
            (nmo.observation_models.PoissonObservations, True),
            (nmo.observation_models.GammaObservations, False),
        ],
    )
    def test_optimize_solver_params_pytree(
        self,
        batch_size,
        stepsize,
        regularizer,
        solver_name,
        inv_link,
        observation_model,
        has_dafaults,
        link_has_defaults,
        obs_has_defaults,
        poisson_population_GLM_model_pytree,
    ):
        """Test the behavior of `optimize_solver_params` for different solver, regularizer, and observation model configurations."""
        obs = observation_model(inverse_link_function=inv_link)
        X, y, _, _, _ = poisson_population_GLM_model_pytree
        solver_kwargs = dict(stepsize=stepsize, batch_size=batch_size)
        # use glm static methods to check if the solver is batchable
        # if not pop the batch_size kwarg
        try:
            slv_class = nmo.glm.PopulationGLM._get_solver_class(solver_name)
            nmo.glm.PopulationGLM._check_solver_kwargs(slv_class, solver_kwargs)
        except NameError:
            solver_kwargs.pop("batch_size")

        # if the regularizer is not allowed for the solver type, return
        try:
            model = nmo.glm.PopulationGLM(
                regularizer=regularizer,
                solver_name=solver_name,
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
        elif has_dafaults and link_has_defaults and obs_has_defaults:
            # if defaults are available, a batch size is computed
            assert isinstance(kwargs["batch_size"], int) and kwargs["batch_size"] > 0
        elif "batch_size" in solver_kwargs:
            # return None otherwise
            assert isinstance(kwargs["batch_size"], type(None))

        if isinstance(stepsize, float):
            # if stepsize was provided, then it should be returned unchanged
            assert stepsize == kwargs["stepsize"]
        elif has_dafaults and link_has_defaults and obs_has_defaults:
            # if defaults are available, compute a value
            assert isinstance(kwargs["stepsize"], float) and kwargs["stepsize"] > 0
        else:
            # return None otherwise
            assert isinstance(kwargs["stepsize"], type(None))

    def test_repr_out(self, poisson_population_GLM_model):
        model = poisson_population_GLM_model[2]
        assert (
            repr(model)
            == "PopulationGLM(\n    observation_model=PoissonObservations(inverse_link_function=exp),\n    regularizer=UnRegularized(),\n    solver_name='GradientDescent'\n)"
        )


@pytest.mark.parametrize(
    "regularizer, expected_type_convexity",
    [
        ("UnRegularized", type(None)),
        ("Lasso", type(None)),
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
    regularizer,
    solver_name,
    inv_link_func,
    expected_type_convexity,
    expected_type_link,
    expected_type_solver,
):
    """Test that 'required_params' is a dictionary."""
    obs = nmo.observation_models.PoissonObservations(
        inverse_link_function=inv_link_func
    )

    # if the regularizer is not allowed for the solver type, return
    try:
        model = nmo.glm.GLM(
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
