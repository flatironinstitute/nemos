from contextlib import nullcontext as does_not_raise

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV

import nemos as nmo
from nemos.pytrees import FeaturePytree


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
        "regularizer, expectation",
        [
            (nmo.regularizer.Ridge("BFGS"), does_not_raise()),
            (
                None,
                pytest.raises(
                    AttributeError, match="The provided `solver` doesn't implement "
                ),
            ),
            (
                nmo.regularizer.Ridge,
                pytest.raises(
                    TypeError, match="The provided `solver` cannot be instantiated"
                ),
            ),
        ],
    )
    def test_solver_type(self, regularizer, expectation, glm_class):
        """
        Test that an error is raised if a non-compatible solver is passed.
        """
        with expectation:
            glm_class(regularizer=regularizer)

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
            glm_class(regularizer=ridge_regularizer, observation_model=observation)

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
        "add_entry, add_to, expectation",
        [
            (0, "X", does_not_raise()),
            (
                np.nan,
                "X",
                pytest.warns(UserWarning, match="The provided trees contain"),
            ),
            (
                np.inf,
                "X",
                pytest.warns(UserWarning, match="The provided trees contain"),
            ),
            (0, "y", does_not_raise()),
            (
                np.nan,
                "y",
                pytest.warns(UserWarning, match="The provided trees contain"),
            ),
            (
                np.inf,
                "y",
                pytest.warns(UserWarning, match="The provided trees contain"),
            ),
        ],
    )
    def test_fit_param_values(
        self, add_entry, add_to, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `fit` method with altered X or y values. Ensure the method raises exceptions for NaN or Inf values.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        if add_to == "X":
            # get an index to be edited
            idx = np.unravel_index(np.random.choice(X.size), X.shape)
            X[idx] = add_entry
        elif add_to == "y":
            idx = np.unravel_index(np.random.choice(y.size), y.shape)
            y = np.asarray(y, dtype=np.float32)
            y[idx] = add_entry
        with expectation:
            model.fit(X, y, init_params=true_params)

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
            regularizer=nmo.regularizer.GroupLasso(
                solver_name="ProximalGradient", mask=mask
            )
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
        assert model.scale != 1

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
        model.scale = 1.0
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

    @pytest.mark.parametrize(
        "insert, expectation",
        [
            (0, does_not_raise()),
            (np.nan, pytest.warns(UserWarning, match=r"The provided trees contain")),
            (np.inf, pytest.warns(UserWarning, match=r"The provided trees contain")),
        ],
    )
    def test_simulate_invalid_feedforward(
        self, insert, expectation, poissonGLM_model_instantiation
    ):
        X, y, model, params, rate = poissonGLM_model_instantiation
        model.coef_ = params[0]
        model.intercept_ = params[1]
        X[0] = insert
        with expectation:
            model.simulate(jax.random.key(123), X)

    @pytest.mark.parametrize("inv_link", [jnp.exp, lambda x: 1 / x])
    def test_simulate_gamma_glm(self, inv_link, gammaGLM_model_instantiation):
        X, y, model, true_params, firing_rate = gammaGLM_model_instantiation
        model.observation_model.inverse_link_function = inv_link
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        model.scale = 1.0
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
        dev_model = model.observation_model.deviance(firing_rate, y).sum()
        if not np.allclose(dev, dev_model):
            raise ValueError("Deviance doesn't match statsmodels!")

    def test_compatibility_with_sklearn_cv(self, poissonGLM_model_instantiation):
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        param_grid = {"regularizer__solver_name": ["BFGS", "GradientDescent"]}
        GridSearchCV(model, param_grid).fit(X, y)

    def test_compatibility_with_sklearn_cv_gamma(self, gammaGLM_model_instantiation):
        X, y, model, true_params, firing_rate = gammaGLM_model_instantiation
        param_grid = {"regularizer__solver_name": ["BFGS", "GradientDescent"]}
        GridSearchCV(model, param_grid).fit(X, y)


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
            (nmo.regularizer.Ridge("BFGS"), does_not_raise()),
            (
                None,
                pytest.raises(
                    AttributeError, match="The provided `solver` doesn't implement "
                ),
            ),
            (
                nmo.regularizer.Ridge,
                pytest.raises(
                    TypeError, match="The provided `solver` cannot be instantiated"
                ),
            ),
        ],
    )
    def test_solver_type(self, regularizer, expectation, population_glm_class):
        """
        Test that an error is raised if a non-compatible solver is passed.
        """
        with expectation:
            population_glm_class(regularizer=regularizer)

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
                regularizer=ridge_regularizer, observation_model=observation
            )

    @pytest.mark.parametrize(
        "X, y",
        [
            (jnp.zeros((2, 4)), jnp.zeros((2, 2))),
            (jnp.zeros((2, 4)), jnp.zeros((2, 3))),
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
        "add_entry, add_to, expectation",
        [
            (0, "X", does_not_raise()),
            (
                np.nan,
                "X",
                pytest.warns(UserWarning, match="The provided trees contain"),
            ),
            (
                np.inf,
                "X",
                pytest.warns(UserWarning, match="The provided trees contain"),
            ),
            (0, "y", does_not_raise()),
            (
                np.nan,
                "y",
                pytest.warns(UserWarning, match="The provided trees contain"),
            ),
            (
                np.inf,
                "y",
                pytest.warns(UserWarning, match="The provided trees contain"),
            ),
        ],
    )
    def test_fit_param_values(
        self, add_entry, add_to, expectation, poisson_population_GLM_model
    ):
        """
        Test the `fit` method with altered X or y values. Ensure the method raises exceptions for NaN or Inf values.
        """
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        if add_to == "X":
            # get an index to be edited
            idx = np.unravel_index(np.random.choice(X.size), X.shape)
            X[idx] = add_entry
        elif add_to == "y":
            idx = np.unravel_index(np.random.choice(y.size), y.shape)
            y = np.asarray(y, dtype=np.float32)
            y[idx] = add_entry
        with expectation:
            model.fit(X, y, init_params=true_params)

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
            regularizer=nmo.regularizer.GroupLasso(
                solver_name="ProximalGradient", mask=mask
            )
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
        assert np.all(model.scale != 1)

    def test_fit_scale_array(self, gamma_population_GLM_model):
        X, y, model, true_params, firing_rate = gamma_population_GLM_model
        model.fit(X, y)
        assert model.scale.size == y.shape[1]

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
    def test_predict_is_fit(self, is_fit, expectation, poisson_population_GLM_model):
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
        model.scale = np.ones((y.shape[1]))
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

    @pytest.mark.parametrize(
        "insert, expectation",
        [
            (0, does_not_raise()),
            (np.nan, pytest.warns(UserWarning, match=r"The provided trees contain")),
            (np.inf, pytest.warns(UserWarning, match=r"The provided trees contain")),
        ],
    )
    def test_simulate_invalid_feedforward(
        self, insert, expectation, poisson_population_GLM_model
    ):
        X, y, model, params, rate = poisson_population_GLM_model
        model.coef_ = params[0]
        model.intercept_ = params[1]
        model._initialize_feature_mask(X, y)
        X[0] = insert
        with expectation:
            model.simulate(jax.random.key(123), X)

    @pytest.mark.parametrize("inv_link", [jnp.exp, lambda x: 1 / x])
    def test_simulate_gamma_glm(self, inv_link, gamma_population_GLM_model):
        X, y, model, true_params, firing_rate = gamma_population_GLM_model
        model.observation_model.inverse_link_function = inv_link
        model.feature_mask = jnp.ones((X.shape[1], y.shape[1]))
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        model.scale = jnp.ones((y.shape[1]))
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
        dev_model = model.observation_model.deviance(firing_rate, y).sum()
        if not np.allclose(dev, dev_model):
            raise ValueError("Deviance doesn't match statsmodels!")

    def test_compatibility_with_sklearn_cv(self, poisson_population_GLM_model):
        X, y, model, true_params, firing_rate = poisson_population_GLM_model
        param_grid = {"regularizer__solver_name": ["BFGS", "GradientDescent"]}
        GridSearchCV(model, param_grid).fit(X, y)

    def test_compatibility_with_sklearn_cv_gamma(self, gamma_population_GLM_model):
        X, y, model, true_params, firing_rate = gamma_population_GLM_model
        param_grid = {"regularizer__solver_name": ["BFGS", "GradientDescent"]}
        GridSearchCV(model, param_grid).fit(X, y)

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
        "regularizer",
        [
            nmo.regularizer.UnRegularized(
                solver_name="LBFGS", solver_kwargs={"stepsize": 0.1, "tol": 10**-14}
            ),
            nmo.regularizer.UnRegularized(
                solver_name="GradientDescent", solver_kwargs={"tol": 10**-14}
            ),
            nmo.regularizer.Ridge(
                solver_name="GradientDescent",
                regularizer_strength=0.001,
                solver_kwargs={"tol": 10**-14},
            ),
            nmo.regularizer.Ridge(
                solver_name="LBFGS", solver_kwargs={"stepsize": 0.1, "tol": 10**-14}
            ),
            nmo.regularizer.Lasso(
                regularizer_strength=0.001, solver_kwargs={"tol": 10**-14}
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
        mask,
        poisson_population_GLM_model,
        poisson_population_GLM_model_pytree,
    ):
        jax.config.update("jax_enable_x64", True)
        if isinstance(mask, dict):
            X, y, model, true_params, firing_rate = poisson_population_GLM_model_pytree

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
            X, y, model, true_params, firing_rate = poisson_population_GLM_model

            def map_neu(k, coef_):
                ind_array = np.where(mask[:, k])[0]
                coef_stack = coef_
                return ind_array, coef_stack

        mask_bool = jax.tree_util.tree_map(lambda x: np.asarray(x.T, dtype=bool), mask)
        # fit pop glm
        model.feature_mask = mask
        model.regularizer = regularizer
        model.fit(X, y)
        coef_vectorized = np.vstack(jax.tree_util.tree_leaves(model.coef_))

        coef_loop = np.zeros((5, 3))
        intercept_loop = np.zeros((3,))
        # loop over neuron
        for k in range(y.shape[1]):
            model_single_neu = nmo.glm.GLM(regularizer=regularizer)
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
