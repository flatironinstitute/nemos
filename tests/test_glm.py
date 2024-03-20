from contextlib import nullcontext as does_not_raise

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV

import nemos as nmo
from nemos.pytrees import FeaturePytree


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
                    match=r"params\[0\] must be an array or .* of shape \(n_neurons, n_features\)",
                ),
            ),
            (
                1,
                pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_neurons, n_features\)",
                ),
            ),
            (2, does_not_raise()),
            (
                3,
                pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_neurons, n_features\)",
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
        n_samples, n_neurons, n_features = X.shape
        if dim_weights == 0:
            init_w = jnp.array([])
        elif dim_weights == 1:
            init_w = jnp.zeros((n_neurons,))
        elif dim_weights == 2:
            init_w = jnp.zeros((n_neurons, n_features))
        else:
            init_w = jnp.zeros((n_neurons, n_features) + (1,) * (dim_weights - 2))
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
        n_samples, n_neurons, n_features = X.shape
        init_b = jnp.zeros((n_neurons,) * dim_intercepts)
        init_w = jnp.zeros((n_neurons, n_features))
        with expectation:
            model.fit(X, y, init_params=(init_w, init_b))

    @pytest.mark.parametrize(
        "init_params, expectation",
        [
            ([jnp.zeros((1, 5)), jnp.zeros((1,))], does_not_raise()),
            (
                iter([jnp.zeros((1, 5)), jnp.zeros((1,))]),
                pytest.raises(ValueError, match="Params must have length two."),
            ),
            (dict(p1=jnp.zeros((1, 5)), p2=jnp.zeros((1,))), pytest.raises(KeyError)),
            (
                (dict(p1=jnp.zeros((1, 5)), p2=jnp.zeros((1, 1))), jnp.zeros((1,))),
                pytest.raises(
                    TypeError, match=r"X and params\[0\] must be the same type"
                ),
            ),
            (
                (
                    FeaturePytree(p1=jnp.zeros((1, 5)), p2=jnp.zeros((1, 1))),
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
        "delta_n_neuron, expectation",
        [
            (
                -1,
                pytest.raises(
                    ValueError, match="Model parameters have inconsistent shapes"
                ),
            ),
            (0, does_not_raise()),
            (
                1,
                pytest.raises(
                    ValueError, match="Model parameters have inconsistent shapes"
                ),
            ),
        ],
    )
    def test_fit_n_neuron_match_baseline_rate(
        self, delta_n_neuron, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `fit` method ensuring The number of neurons in the baseline rate matches the expected number.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_samples, n_neurons, n_features = X.shape
        init_b = jnp.zeros((n_neurons + delta_n_neuron,))
        with expectation:
            model.fit(X, y, init_params=(true_params[0], init_b))

    @pytest.mark.parametrize(
        "delta_n_neuron, expectation",
        [
            (
                -1,
                pytest.raises(
                    ValueError, match="The number of neurons in the model parameters"
                ),
            ),
            (0, does_not_raise()),
            (
                1,
                pytest.raises(
                    ValueError, match="The number of neurons in the model parameters"
                ),
            ),
        ],
    )
    def test_fit_n_neuron_match_x(
        self, delta_n_neuron, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `fit` method ensuring The number of neurons in X matches The number of neurons in the model.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_neurons = X.shape[1]
        X = jnp.repeat(X, n_neurons + delta_n_neuron, axis=1)
        with expectation:
            model.fit(X, y, init_params=true_params)

    @pytest.mark.parametrize(
        "delta_n_neuron, expectation",
        [
            (
                -1,
                pytest.raises(
                    ValueError, match="The number of neurons in the model parameters"
                ),
            ),
            (0, does_not_raise()),
            (
                1,
                pytest.raises(
                    ValueError, match="The number of neurons in the model parameters"
                ),
            ),
        ],
    )
    def test_fit_n_neuron_match_y(
        self, delta_n_neuron, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `fit` method ensuring The number of neurons in y matches The number of neurons in the model.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_neurons = X.shape[1]
        y = jnp.repeat(y, n_neurons + delta_n_neuron, axis=1)
        with expectation:
            model.fit(X, y, init_params=true_params)

    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (-1, pytest.raises(ValueError, match="X must be three-dimensional")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="X must be three-dimensional")),
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
            X = np.zeros((X.shape[0], X.shape[1]))
        elif delta_dim == 1:
            X = np.zeros((X.shape[0], X.shape[1], X.shape[2], 1))
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
        self, delta_dim, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `fit` method with y target data of different dimensionalities. Ensure correct dimensionality for y.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        if delta_dim == -1:
            y = np.zeros(X.shape[0])
        elif delta_dim == 1:
            y = np.zeros((X.shape[0], X.shape[1], 1))
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
        init_w = jnp.zeros((X.shape[1], X.shape[2] + delta_n_features))
        init_b = jnp.zeros(X.shape[1])
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
            X = jnp.concatenate((X, jnp.zeros((X.shape[0], X.shape[1], 1))), axis=2)
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
        flat_coef = np.concatenate(model_tree.coef_.tree_flatten()[0], axis=-1)

        # assert equivalence of solutions
        assert np.allclose(model.coef_, flat_coef)
        assert np.allclose(model.intercept_, model_tree.intercept_)

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

    #######################
    # Test model.score
    #######################
    @pytest.mark.parametrize(
        "delta_n_neuron, expectation",
        [
            (
                -1,
                pytest.raises(
                    ValueError, match="The number of neurons in the model parameters"
                ),
            ),
            (0, does_not_raise()),
            (
                1,
                pytest.raises(
                    ValueError, match="The number of neurons in the model parameters"
                ),
            ),
        ],
    )
    def test_score_n_neuron_match_x(
        self, delta_n_neuron, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `score` method when The number of neurons in X differs. Ensure the correct number of neurons.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        X = jnp.repeat(X, X.shape[1] + delta_n_neuron, axis=1)
        with expectation:
            model.score(X, y)

    @pytest.mark.parametrize(
        "delta_n_neuron, expectation",
        [
            (
                -1,
                pytest.raises(
                    ValueError, match="The number of neurons in the model parameters"
                ),
            ),
            (0, does_not_raise()),
            (
                1,
                pytest.raises(
                    ValueError, match="The number of neurons in the model parameters"
                ),
            ),
        ],
    )
    def test_score_n_neuron_match_y(
        self, delta_n_neuron, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `score` method when The number of neurons in y differs. Ensure the correct number of neurons.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        y = jnp.repeat(y, y.shape[1] + delta_n_neuron, axis=1)
        with expectation:
            model.score(X, y)

    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (-1, pytest.raises(ValueError, match="X must be three-dimensional")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="X must be three-dimensional")),
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
            X = np.zeros((X.shape[0], X.shape[1]))
        elif delta_dim == 1:
            X = np.zeros((X.shape[0], X.shape[1], X.shape[2], 1))
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
            y = np.zeros((X.shape[0],))
        elif delta_dim == 1:
            y = np.zeros((X.shape[0], X.shape[1], 1))
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
            X = jnp.concatenate((X, jnp.zeros((X.shape[0], X.shape[1], 1))), axis=2)
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

    #######################
    # Test model.predict
    #######################
    @pytest.mark.parametrize(
        "delta_n_neuron, expectation",
        [
            (
                -1,
                pytest.raises(
                    ValueError, match="The number of neurons in the model parameters"
                ),
            ),
            (0, does_not_raise()),
            (
                1,
                pytest.raises(
                    ValueError, match="The number of neurons in the model parameters"
                ),
            ),
        ],
    )
    def test_predict_n_neuron_match_x(
        self, delta_n_neuron, expectation, poissonGLM_model_instantiation
    ):
        """
        Test the `predict` method when The number of neurons in X differs.
        Ensure that The number of neurons in X, y and params matches.
        """
        X, _, model, true_params, _ = poissonGLM_model_instantiation
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        X = jnp.repeat(X, X.shape[1] + delta_n_neuron, axis=1)
        with expectation:
            model.predict(X)

    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (-1, pytest.raises(ValueError, match="X must be three-dimensional")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="X must be three-dimensional")),
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
            X = np.zeros((X.shape[0], X.shape[1]))
        elif delta_dim == 1:
            X = np.zeros((X.shape[0], X.shape[1], X.shape[2], 1))
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
            X = jnp.concatenate((X, jnp.zeros((X.shape[0], X.shape[1], 1))), axis=2)
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
        "delta_n_neuron, expectation",
        [
            (
                -1,
                pytest.raises(
                    ValueError, match="The number of neurons in the model parameters"
                ),
            ),
            (0, does_not_raise()),
            (
                1,
                pytest.raises(
                    ValueError, match="The number of neurons in the model parameters"
                ),
            ),
        ],
    )
    def test_simulate_n_neuron_match_input(
        self, delta_n_neuron, expectation, poissonGLM_coupled_model_config_simulate
    ):
        """
        Test the `simulate` method to ensure that The number of neurons in the input
        matches the model's parameters.
        """
        (
            model,
            coupling_basis,
            feedforward_input,
            init_spikes,
            random_key,
        ) = poissonGLM_coupled_model_config_simulate
        if delta_n_neuron != 0:
            feedforward_input = np.zeros(
                (
                    feedforward_input.shape[0],
                    feedforward_input.shape[1] + delta_n_neuron,
                    feedforward_input.shape[2],
                )
            )
        with expectation:
            model.simulate_recurrent(
                random_key=random_key,
                init_y=init_spikes,
                coupling_basis_matrix=coupling_basis,
                feedforward_input=feedforward_input,
            )

    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (-1, pytest.raises(ValueError, match="X must be three-dimensional")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="X must be three-dimensional")),
        ],
    )
    def test_simulate_input_dimensionality(
        self, delta_dim, expectation, poissonGLM_coupled_model_config_simulate
    ):
        """
        Test the `simulate` method with input data of different dimensionalities.
        Ensure correct dimensionality for input.
        """
        (
            model,
            coupling_basis,
            feedforward_input,
            init_spikes,
            random_key,
        ) = poissonGLM_coupled_model_config_simulate
        if delta_dim == -1:
            feedforward_input = np.zeros(feedforward_input.shape[:2])
        elif delta_dim == 1:
            feedforward_input = np.zeros(feedforward_input.shape + (1,))
        with expectation:
            model.simulate_recurrent(
                random_key=random_key,
                init_y=init_spikes,
                coupling_basis_matrix=coupling_basis,
                feedforward_input=feedforward_input,
            )

    @pytest.mark.parametrize(
        "delta_dim, expectation",
        [
            (-1, pytest.raises(ValueError, match="y must be two-dimensional")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="y must be two-dimensional")),
        ],
    )
    def test_simulate_y_dimensionality(
        self, delta_dim, expectation, poissonGLM_coupled_model_config_simulate
    ):
        """
        Test the `simulate` method with init_spikes of different dimensionalities.
        Ensure correct dimensionality for init_spikes.
        """
        (
            model,
            coupling_basis,
            feedforward_input,
            init_spikes,
            random_key,
        ) = poissonGLM_coupled_model_config_simulate
        if delta_dim == -1:
            init_spikes = np.zeros((feedforward_input.shape[0],))
        elif delta_dim == 1:
            init_spikes = np.zeros(
                (feedforward_input.shape[0], feedforward_input.shape[1], 1)
            )
        with expectation:
            model.simulate_recurrent(
                random_key=random_key,
                init_y=init_spikes,
                coupling_basis_matrix=coupling_basis,
                feedforward_input=feedforward_input,
            )

    @pytest.mark.parametrize(
        "delta_n_neuron, expectation",
        [
            (
                -1,
                pytest.raises(
                    ValueError, match="The number of neurons in the model parameters"
                ),
            ),
            (0, does_not_raise()),
            (
                1,
                pytest.raises(
                    ValueError, match="The number of neurons in the model parameters"
                ),
            ),
        ],
    )
    def test_simulate_n_neuron_match_y(
        self, delta_n_neuron, expectation, poissonGLM_coupled_model_config_simulate
    ):
        """
        Test the `simulate` method to ensure that The number of neurons in init_spikes
        matches the model's parameters.
        """
        (
            model,
            coupling_basis,
            feedforward_input,
            init_spikes,
            random_key,
        ) = poissonGLM_coupled_model_config_simulate
        init_spikes = jnp.zeros(
            (init_spikes.shape[0], feedforward_input.shape[1] + delta_n_neuron)
        )
        with expectation:
            model.simulate_recurrent(
                random_key=random_key,
                init_y=init_spikes,
                coupling_basis_matrix=coupling_basis,
                feedforward_input=feedforward_input,
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
        self, is_fit, expectation, poissonGLM_coupled_model_config_simulate
    ):
        """
        Test if the model raises a ValueError when trying to simulate before it's fitted.
        """
        (
            model,
            coupling_basis,
            feedforward_input,
            init_spikes,
            random_key,
        ) = poissonGLM_coupled_model_config_simulate
        if not is_fit:
            model.intercept_ = None
        with expectation:
            model.simulate_recurrent(
                random_key=random_key,
                init_y=init_spikes,
                coupling_basis_matrix=coupling_basis,
                feedforward_input=feedforward_input,
            )

    @pytest.mark.parametrize(
        "delta_tp, expectation",
        [
            (
                -1,
                pytest.raises(ValueError, match="`init_y` and `coupling_basis_matrix`"),
            ),
            (0, does_not_raise()),
            (
                1,
                pytest.raises(ValueError, match="`init_y` and `coupling_basis_matrix`"),
            ),
        ],
    )
    def test_simulate_time_point_match_y(
        self, delta_tp, expectation, poissonGLM_coupled_model_config_simulate
    ):
        """
        Test the `simulate` method to ensure that the time points in init_y
        are consistent with the coupling_basis window size (they must be equal).
        """
        (
            model,
            coupling_basis,
            feedforward_input,
            init_spikes,
            random_key,
        ) = poissonGLM_coupled_model_config_simulate
        init_spikes = jnp.zeros((init_spikes.shape[0] + delta_tp, init_spikes.shape[1]))
        with expectation:
            model.simulate_recurrent(
                random_key=random_key,
                init_y=init_spikes,
                coupling_basis_matrix=coupling_basis,
                feedforward_input=feedforward_input,
            )

    @pytest.mark.parametrize(
        "delta_tp, expectation",
        [
            (
                -1,
                pytest.raises(ValueError, match="`init_y` and `coupling_basis_matrix`"),
            ),
            (0, does_not_raise()),
            (
                1,
                pytest.raises(ValueError, match="`init_y` and `coupling_basis_matrix`"),
            ),
        ],
    )
    def test_simulate_time_point_match_coupling_basis(
        self, delta_tp, expectation, poissonGLM_coupled_model_config_simulate
    ):
        """
        Test the `simulate` method to ensure that the window size in coupling_basis
        is consistent with the time-points in init_spikes (they must be equal).
        """
        (
            model,
            coupling_basis,
            feedforward_input,
            init_spikes,
            random_key,
        ) = poissonGLM_coupled_model_config_simulate
        coupling_basis = jnp.zeros(
            (coupling_basis.shape[0] + delta_tp,) + coupling_basis.shape[1:]
        )
        with expectation:
            model.simulate_recurrent(
                random_key=random_key,
                init_y=init_spikes,
                coupling_basis_matrix=coupling_basis,
                feedforward_input=feedforward_input,
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
        self, delta_features, expectation, poissonGLM_coupled_model_config_simulate
    ):
        """
        Test the `simulate` method ensuring the number of features in `feedforward_input` is
        consistent with the model's expected number of features.

        Notes
        -----
        The total feature number `model.coef_.shape[1]` must be equal to
        `feedforward_input.shape[2] + coupling_basis.shape[1]*n_neurons`
        """
        (
            model,
            coupling_basis,
            feedforward_input,
            init_spikes,
            random_key,
        ) = poissonGLM_coupled_model_config_simulate
        feedforward_input = jnp.zeros(
            (
                feedforward_input.shape[0],
                feedforward_input.shape[1],
                feedforward_input.shape[2] + delta_features,
            )
        )
        with expectation:
            model.simulate_recurrent(
                random_key=random_key,
                init_y=init_spikes,
                coupling_basis_matrix=coupling_basis,
                feedforward_input=feedforward_input,
            )

    @pytest.mark.parametrize(
        "delta_features, expectation",
        [
            (-1, pytest.raises(ValueError, match="Inconsistent number of features")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="Inconsistent number of features")),
        ],
    )
    def test_simulate_feature_consistency_coupling_basis(
        self, delta_features, expectation, poissonGLM_coupled_model_config_simulate
    ):
        """
        Test the `simulate` method ensuring the number of features in `coupling_basis` is
        consistent with the model's expected number of features.

        Notes
        -----
        The total feature number `model.coef_.shape[1]` must be equal to
        `feedforward_input.shape[2] + coupling_basis.shape[1]*n_neurons`
        """
        (
            model,
            coupling_basis,
            feedforward_input,
            init_spikes,
            random_key,
        ) = poissonGLM_coupled_model_config_simulate
        coupling_basis = jnp.zeros(
            (coupling_basis.shape[0], coupling_basis.shape[1] + delta_features)
        )
        with expectation:
            model.simulate_recurrent(
                random_key=random_key,
                init_y=init_spikes,
                coupling_basis_matrix=coupling_basis,
                feedforward_input=feedforward_input,
            )

    def test_simulate_feedforward_GLM_not_fit(self, poissonGLM_model_instantiation):
        X, y, model, params, rate = poissonGLM_model_instantiation
        with pytest.raises(
            nmo.exceptions.NotFittedError, match="This GLM instance is not fitted yet"
        ):
            model.simulate(jax.random.key(123), X)

    def test_simulate_feedforward_GLM(self, poissonGLM_model_instantiation):
        """Test that simulate goes through"""
        X, y, model, params, rate = poissonGLM_model_instantiation
        model.coef_ = params[0]
        model.intercept_ = params[1]
        ysim, ratesim = model.simulate(jax.random.key(123), X)
        # check that the expected dimensionality is returned
        assert ysim.ndim == 2
        assert ratesim.ndim == 2
        # check that the rates and spikes has the same shape
        assert ratesim.shape[0] == ysim.shape[0]
        assert ratesim.shape[1] == ysim.shape[1]
        # check the time point number is that expected (same as the input)
        assert ysim.shape[0] == X.shape[0]
        # check that the number if neurons is respected
        assert ysim.shape[1] == y.shape[1]

    @pytest.mark.parametrize(
        "insert, expectation",
        [
            (0, does_not_raise()),
            (np.nan, pytest.raises(ValueError, match=r"The provided trees contain")),
            (np.inf, pytest.raises(ValueError, match=r"The provided trees contain")),
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

    def test_end_to_end_fit_and_simulate(
        self, poissonGLM_coupled_model_config_simulate
    ):
        (
            model,
            coupling_basis,
            feedforward_input,
            init_spikes,
            random_key,
        ) = poissonGLM_coupled_model_config_simulate
        window_size = coupling_basis.shape[0]
        n_neurons = init_spikes.shape[1]
        n_trials = 1
        n_timepoints = feedforward_input.shape[0]

        # generate spike trains
        spikes, _ = model.simulate_recurrent(
            random_key=random_key,
            init_y=init_spikes,
            coupling_basis_matrix=coupling_basis,
            feedforward_input=feedforward_input,
        )

        # convolve basis and spikes
        # (n_trials, n_timepoints - ws + 1, n_neurons, n_coupling_basis)
        conv_spikes = jnp.asarray(
            nmo.utils.convolve_1d_trials(coupling_basis, [spikes]), dtype=jnp.float32
        )

        # create an individual neuron predictor by stacking the
        # two convolved spike trains in a single feature vector
        # and concatenate the trials.
        conv_spikes = conv_spikes.reshape(
            n_trials * (n_timepoints - window_size + 1), -1
        )

        # replicate for each neuron,
        # (n_trials * (n_timepoints - ws + 1), n_neurons, n_neurons * n_coupling_basis)
        conv_spikes = jnp.tile(conv_spikes, n_neurons).reshape(
            conv_spikes.shape[0], n_neurons, conv_spikes.shape[1]
        )

        # add the feed-forward input to the predictors
        X = jnp.concatenate((conv_spikes[1:], feedforward_input[:-window_size]), axis=2)

        # fit the model
        model.fit(X, spikes[:-window_size])

        # simulate
        model.simulate_recurrent(
            random_key=random_key,
            init_y=init_spikes,
            coupling_basis_matrix=coupling_basis,
            feedforward_input=feedforward_input,
        )
