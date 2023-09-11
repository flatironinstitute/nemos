import jax
import jax.numpy as jnp
import numpy as np
import pytest
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV

import neurostatslib as nsl


class TestGLM:
    """
    Unit tests for the PoissonGLM class.
    """
    #######################
    # Test model.__init__
    #######################
    @pytest.mark.parametrize("solver", [nsl.solver.RidgeSolver("BFGS"), nsl.solver.Solver, 1])
    def test_init_solver_type(self, solver: nsl.solver.Solver, poisson_noise_model):
        """
        Test initialization with different solver names. Check if an appropriate exception is raised
        when the solver name is not present in jaxopt.
        """
        raise_exception = solver.__class__.__name__ not in nsl.solver.__all__
        if raise_exception:
            with pytest.raises(TypeError, match="The provided `solver` should be one of the implemented"):
                nsl.glm.GLM(solver=solver, noise_model=poisson_noise_model)
        else:
            nsl.glm.GLM(solver=solver, noise_model=poisson_noise_model)

    @pytest.mark.parametrize("noise", [nsl.noise_model.PoissonNoiseModel(), nsl.solver.Solver, 1])
    def test_init_noise_type(self, noise: nsl.noise_model.NoiseModel, ridge_solver):
        """
        Test initialization with different solver names. Check if an appropriate exception is raised
        when the solver name is not present in jaxopt.
        """
        raise_exception = noise.__class__.__name__ not in nsl.noise_model.__all__
        if raise_exception:
            with pytest.raises(TypeError, match="The provided `noise_model` should be one of the implemented"):
                nsl.glm.GLM(solver=ridge_solver, noise_model=noise)
        else:
            nsl.glm.GLM(solver=ridge_solver, noise_model=noise)


    #######################
    # Test model.fit
    #######################
    @pytest.mark.parametrize("n_params", [0, 1, 2, 3])
    def test_fit_param_length(self, n_params, poissonGLM_model_instantiation):
        """
        Test the `fit` method with different numbers of initial parameters.
        Check for correct number of parameters.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_samples, n_neurons, n_features = X.shape
        init_w = jnp.zeros((n_neurons, n_features))
        init_b = jnp.log(y.mean(axis=0))
        if n_params == 0:
            init_params = tuple()
        elif n_params == 1:
            init_params = (init_w,)
        elif n_params == 2:
            init_params = (init_w, init_b)
        else:
            init_params = (init_w, init_b) + (init_w,) * (n_params - 2)

        raise_exception = n_params != 2
        if raise_exception:
            with pytest.raises(ValueError, match="Params needs to be array-like of length two."):
                model.fit(X, y, init_params=init_params)
        else:
            model.fit(X, y, init_params=init_params)

    @pytest.mark.parametrize("add_entry", [0, np.nan, np.inf])
    @pytest.mark.parametrize("add_to", ["X", "y"])
    def test_fit_param_values(self, add_entry, add_to, poissonGLM_model_instantiation):
        """
        Test the `fit` method with altered X or y values. Ensure the method raises exceptions for NaN or Inf values.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        if add_to == "X":
            idx = np.unravel_index(np.random.choice(X.size), X.shape)
            X[idx] = add_entry
        elif add_to == "y":
            idx = np.unravel_index(np.random.choice(y.size), y.shape)
            y = np.asarray(y, dtype=np.float32)
            y[idx] = add_entry

        raise_exception = jnp.isnan(add_entry) or jnp.isinf(add_entry)
        if raise_exception:
            with pytest.raises(ValueError, match="Input (X|y) contains a NaNs or Infs"):
                model.fit(X, y, init_params=true_params)
        else:
            model.fit(X, y, init_params=true_params)

    @pytest.mark.parametrize("dim_weights", [0, 1, 2, 3])
    def test_fit_weights_dimensionality(self, dim_weights, poissonGLM_model_instantiation):
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
        init_b = jnp.log(y.mean(axis=0))
        raise_exception = dim_weights != 2
        if raise_exception:
            with pytest.raises(ValueError, match="params\[0\] must be of shape \(n_neurons, n_features\)"):
                model.fit(X, y, init_params=(init_w, init_b))
        else:
            model.fit(X, y, init_params=(init_w, init_b))

    @pytest.mark.parametrize("dim_intercepts", [0, 1, 2, 3])
    def test_fit_intercepts_dimensionality(self, dim_intercepts, poissonGLM_model_instantiation):
        """
        Test the `fit` method with intercepts of different dimensionalities. Check for correct dimensionality.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_samples, n_neurons, n_features = X.shape

        init_b = jnp.zeros((n_neurons,) * dim_intercepts)
        init_w = jnp.zeros((n_neurons, n_features))
        raise_exception = dim_intercepts != 1
        if raise_exception:
            with pytest.raises(ValueError, match="params\[1\] must be of shape"):
                model.fit(X, y, init_params=(init_w, init_b))
        else:
            model.fit(X, y, init_params=(init_w, init_b))

    @pytest.mark.parametrize("init_params",
                             [dict(p1=jnp.zeros((1, 5)), p2=jnp.zeros((1,))),
                              [jnp.zeros((1, 5)), jnp.zeros((1,))],
                              dict(p1=jnp.zeros((1, 5)), p2=np.zeros((1,), dtype='U10')),
                              0,
                              {0, 1},
                              iter([jnp.zeros((1, 5)), jnp.zeros((1,))]),
                              [jnp.zeros((1, 5)), ""],
                              ["", jnp.zeros((1,))]])
    def test_fit_init_params_type(self, init_params, poissonGLM_model_instantiation):
        """
        Test the `fit` method with various types of initial parameters. Ensure that the provided initial parameters
        are array-like.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        # check if parameter can be converted
        try:
            tuple(jnp.asarray(par, dtype=jnp.float32) for par in init_params)
            # ensure that it's an array-like (for example excluding sets and iterators)
            raise_exception = not hasattr(init_params, "__getitem__")
        except(TypeError, ValueError):
            raise_exception = True

        if raise_exception:
            with pytest.raises(TypeError, match="Initial parameters must be array-like"):
                model.fit(X, y, init_params=init_params)
        else:
            model.fit(X, y, init_params=init_params)

    @pytest.mark.parametrize("delta_n_neuron", [-1, 0, 1])
    def test_fit_n_neuron_match_weights(self, delta_n_neuron, poissonGLM_model_instantiation):
        """
        Test the `fit` method ensuring the number of neurons in the weights matches the expected number.
        """
        raise_exception = delta_n_neuron != 0
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_samples, n_neurons, n_features = X.shape
        init_w = jnp.zeros((n_neurons + delta_n_neuron, n_features))
        init_b = jnp.zeros((n_neurons, ))
        # model.basis_coeff_ = init_w
        # model.baseline_link_fr_ = init_b
        if raise_exception:
            with pytest.raises(ValueError, match="Model parameters have inconsistent shapes"):
                model.fit(X, y, init_params=(init_w, init_b))
            # with pytest.raises(ValueError, match="Model parameters have inconsistent shapes"):
            #     model.predict(X)
            # with pytest.raises(ValueError, match="Model parameters have inconsistent shapes"):
            #     model.score(X, y)
        else:
            model.fit(X, y, init_params=(init_w, init_b))
            # model.predict(X)
            # model.score(X, y)

    @pytest.mark.parametrize("delta_n_neuron", [-1, 0, 1])
    def test_fit_n_neuron_match_baseline_rate(self, delta_n_neuron, poissonGLM_model_instantiation):
        """
        Test the `fit` method ensuring the number of neurons in the baseline rate matches the expected number.
        """
        raise_exception = delta_n_neuron != 0
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_samples, n_neurons, n_features = X.shape
        init_w = jnp.zeros((n_neurons, n_features))
        init_b = jnp.zeros((n_neurons + delta_n_neuron,))

        if raise_exception:
            with pytest.raises(ValueError, match="Model parameters have inconsistent shapes"):
                model.fit(X, y, init_params=(init_w, init_b))

        else:
            model.fit(X, y, init_params=(init_w, init_b))

    @pytest.mark.parametrize("delta_n_neuron", [-1, 0, 1])
    def test_fit_n_neuron_match_x(self, delta_n_neuron, poissonGLM_model_instantiation):
        """
        Test the `fit` method ensuring the number of neurons in X matches the number of neurons in the model.
        """
        raise_exception = delta_n_neuron != 0
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_samples, n_neurons, n_features = X.shape
        init_w = jnp.zeros((n_neurons, n_features))
        init_b = jnp.zeros((n_neurons,))
        X = jnp.repeat(X, n_neurons + delta_n_neuron, axis=1)
        if raise_exception:
            with pytest.raises(ValueError, match="The number of neuron in the model parameters"):
                model.fit(X, y, init_params=(init_w, init_b))
        else:
            model.fit(X, y, init_params=(init_w, init_b))

    @pytest.mark.parametrize("delta_n_neuron", [-1, 0, 1])
    def test_fit_n_neuron_match_y(self, delta_n_neuron, poissonGLM_model_instantiation):
        """
        Test the `fit` method ensuring the number of neurons in y matches the number of neurons in the model.
        """
        raise_exception = delta_n_neuron != 0
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_samples, n_neurons, n_features = X.shape
        init_w = jnp.zeros((n_neurons, n_features))
        init_b = jnp.zeros((n_neurons,))
        y = jnp.repeat(y, n_neurons + delta_n_neuron, axis=1)
        if raise_exception:
            with pytest.raises(ValueError, match="The number of neuron in the model parameters"):
                model.fit(X, y, init_params=(init_w, init_b))
        else:
            model.fit(X, y, init_params=(init_w, init_b))

    @pytest.mark.parametrize("delta_dim", [-1, 0, 1])
    def test_fit_x_dimensionality(self, delta_dim, poissonGLM_model_instantiation):
        """
        Test the `fit` method with X input data of different dimensionalities. Ensure correct dimensionality for X.
        """
        raise_exception = delta_dim != 0
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_samples, n_neurons, n_features = X.shape
        init_w = jnp.zeros((n_neurons, n_features))
        init_b = jnp.zeros((n_neurons,))

        if delta_dim == -1:
            # remove a dimension
            X = np.zeros((n_samples, n_neurons))
        elif delta_dim == 1:
            # add a dimension
            X = np.zeros((n_samples, n_neurons, n_features, 1))

        if raise_exception:
            with pytest.raises(ValueError, match="X must be three-dimensional"):
                model.fit(X, y, init_params=(init_w, init_b))
        else:
            model.fit(X, y, init_params=(init_w, init_b))

    @pytest.mark.parametrize("delta_dim", [-1, 0, 1])
    def test_fit_y_dimensionality(self, delta_dim, poissonGLM_model_instantiation):
        """
        Test the `fit` method with y target data of different dimensionalities. Ensure correct dimensionality for y.
        """
        raise_exception = delta_dim != 0
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_samples, n_neurons, n_features = X.shape
        init_w = jnp.zeros((n_neurons, n_features))
        init_b = jnp.zeros((n_neurons,))

        if delta_dim == -1:
            # remove a dimension
            y = np.zeros((n_samples, ))
        elif delta_dim == 1:
            # add a dimension
            y = np.zeros((n_samples, n_neurons, 1))

        if raise_exception:
            with pytest.raises(ValueError, match="y must be two-dimensional"):
                model.fit(X, y, init_params=(init_w, init_b))
        else:
            model.fit(X, y, init_params=(init_w, init_b))

    @pytest.mark.parametrize("delta_n_features", [-1, 0, 1])
    def test_fit_n_feature_consistency_weights(self, delta_n_features, poissonGLM_model_instantiation):
        """
        Test the `fit` method for inconsistencies between data features and initial weights provided.
        Ensure the number of features align.
        """
        raise_exception = delta_n_features != 0
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_samples, n_neurons, n_features = X.shape
        # add/remove a feature from weights
        init_w = jnp.zeros((n_neurons, n_features + delta_n_features))
        init_b = jnp.zeros((n_neurons,))

        if raise_exception:
            with pytest.raises(ValueError, match="Inconsistent number of features"):
                model.fit(X, y, init_params=(init_w, init_b))
        else:
            model.fit(X, y, init_params=(init_w, init_b))

    @pytest.mark.parametrize("delta_n_features", [-1, 0, 1])
    def test_fit_n_feature_consistency_x(self, delta_n_features, poissonGLM_model_instantiation):
        """
        Test the `fit` method for inconsistencies between data features and model's expectations.
        Ensure the number of features in X aligns.
        """
        raise_exception = delta_n_features != 0
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_samples, n_neurons, n_features = X.shape
        init_w = jnp.zeros((n_neurons, n_features))
        init_b = jnp.zeros((n_neurons,))

        if delta_n_features == 1:
            # add a feature
            X = jnp.concatenate((X, jnp.zeros((100, 1, 1))), axis=2)
        elif delta_n_features == -1:
            # remove a feature
            X = X[..., :-1]

        if raise_exception:
            with pytest.raises(ValueError, match="Inconsistent number of features"):
                model.fit(X, y, init_params=(init_w, init_b))
        else:
            model.fit(X, y, init_params=(init_w, init_b))

    @pytest.mark.parametrize("delta_tp", [-1, 0, 1])
    def test_fit_time_points_x(self, delta_tp, poissonGLM_model_instantiation):
        """
        Test the `fit` method for inconsistencies in time-points in data X. Ensure the correct number of time-points.
        """
        raise_exception = delta_tp != 0
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_samples, n_neurons, n_features = X.shape
        init_w = jnp.zeros((n_neurons, n_features))
        init_b = jnp.zeros((n_neurons,))
        X = jnp.zeros((X.shape[0] + delta_tp, ) + X.shape[1:])
        if raise_exception:
            with pytest.raises(ValueError, match="The number of time-points in X and y"):
                model.fit(X, y, init_params=(init_w, init_b))
        else:
            model.fit(X, y, init_params=(init_w, init_b))

    @pytest.mark.parametrize("delta_tp", [-1, 0, 1])
    def test_fit_time_points_y(self, delta_tp, poissonGLM_model_instantiation):
        """
        Test the `fit` method for inconsistencies in time-points in y. Ensure the correct number of time-points.
        """
        raise_exception = delta_tp != 0
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_samples, n_neurons, n_features = X.shape
        init_w = jnp.zeros((n_neurons, n_features))
        init_b = jnp.zeros((n_neurons,))
        y = jnp.zeros((y.shape[0] + delta_tp,) + y.shape[1:])
        if raise_exception:
            with pytest.raises(ValueError, match="The number of time-points in X and y"):
                model.fit(X, y, init_params=(init_w, init_b))
        else:
            model.fit(X, y, init_params=(init_w, init_b))

    @pytest.mark.parametrize("device_spec", ["cpu", "tpu", "gpu", "none", 1])
    def test_fit_device_spec(self, device_spec,
                                   poissonGLM_model_instantiation):
        """
        Test `simulate` across different device specifications.
        Validates if unsupported or absent devices raise exception
        or warning respectively.
        """
        raise_exception = not (device_spec in ["cpu", "tpu", "gpu"])
        raise_warning = all(device_spec != device.device_kind.lower()
                            for device in jax.local_devices())
        raise_warning = raise_warning and (not raise_exception)

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_samples, n_neurons, n_features = X.shape
        init_w = jnp.zeros((n_neurons, n_features))
        init_b = jnp.zeros((n_neurons,))
        if raise_exception:
            with pytest.raises(ValueError, match=f"Invalid device specification: {device_spec}"):
                model.fit(X, y, init_params=(init_w, init_b), device=device_spec)
        elif raise_warning:
            with pytest.warns(UserWarning, match=f"No {device_spec.upper()} found"):
                model.fit(X, y, init_params=(init_w, init_b), device=device_spec)
        else:
            model.fit(X, y, init_params=(init_w, init_b), device=device_spec)

    #######################
    # Test model.score
    #######################
    @pytest.mark.parametrize("delta_n_neuron", [-1, 0, 1])
    def test_score_n_neuron_match_x(self, delta_n_neuron, poissonGLM_model_instantiation):
        """
        Test the `score` method when the number of neurons in X differs. Ensure the correct number of neurons.
        """
        raise_exception = delta_n_neuron != 0
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_samples, n_neurons, n_features = X.shape
        # set model coeff
        model.basis_coeff_ = true_params[0]
        model.baseline_link_fr_ = true_params[1]
        X = jnp.repeat(X, n_neurons + delta_n_neuron, axis=1)
        if raise_exception:
            with pytest.raises(ValueError, match="The number of neuron in the model parameters"):
                model.score(X, y)
        else:
            model.score(X, y)

    @pytest.mark.parametrize("delta_n_neuron", [-1, 0, 1])
    def test_score_n_neuron_match_y(self, delta_n_neuron, poissonGLM_model_instantiation):
        """
        Test the `score` method when the number of neurons in y differs. Ensure the correct number of neurons.
        """
        raise_exception = delta_n_neuron != 0
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_samples, n_neurons, n_features = X.shape
        # set model coeff
        model.basis_coeff_ = true_params[0]
        model.baseline_link_fr_ = true_params[1]
        y = jnp.repeat(y, n_neurons + delta_n_neuron, axis=1)
        if raise_exception:
            with pytest.raises(ValueError, match="The number of neuron in the model parameters"):
                model.score(X, y)
        else:
            model.score(X, y)

    @pytest.mark.parametrize("delta_dim", [-1, 0, 1])
    def test_score_x_dimensionality(self, delta_dim, poissonGLM_model_instantiation):
        """
        Test the `score` method with X input data of different dimensionalities. Ensure correct dimensionality for X.
        """
        raise_exception = delta_dim != 0
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_samples, n_neurons, n_features = X.shape
        # set model coeff
        model.basis_coeff_ = true_params[0]
        model.baseline_link_fr_ = true_params[1]

        if delta_dim == -1:
            # remove a dimension
            X = np.zeros((n_samples, n_neurons))
        elif delta_dim == 1:
            # add a dimension
            X = np.zeros((n_samples, n_neurons, n_features, 1))

        if raise_exception:
            with pytest.raises(ValueError, match="X must be three-dimensional"):
                model.score(X, y)
        else:
            model.score(X, y)

    @pytest.mark.parametrize("delta_dim", [-1, 0, 1])
    def test_score_y_dimensionality(self, delta_dim, poissonGLM_model_instantiation):
        """
        Test the `score` method with y of different dimensionalities.
        Ensure correct dimensionality for y.
        """
        raise_exception = delta_dim != 0
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_samples, n_neurons, _ = X.shape
        # set model coeff
        model.basis_coeff_ = true_params[0]
        model.baseline_link_fr_ = true_params[1]

        if delta_dim == -1:
            # remove a dimension
            y = np.zeros((n_samples,))
        elif delta_dim == 1:
            # add a dimension
            y = np.zeros((n_samples, n_neurons, 1))

        if raise_exception:
            with pytest.raises(ValueError, match="y must be two-dimensional"):
                model.score(X, y)
        else:
            model.score(X, y)

    @pytest.mark.parametrize("delta_n_features", [-1, 0, 1])
    def test_score_n_feature_consistency_x(self, delta_n_features, poissonGLM_model_instantiation):
        """
        Test the `score` method for inconsistencies in features of X.
        Ensure the number of features in X aligns with the model params.
        """
        raise_exception = delta_n_features != 0
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        # set model coeff
        model.basis_coeff_ = true_params[0]
        model.baseline_link_fr_ = true_params[1]
        if delta_n_features == 1:
            # add a feature
            X = jnp.concatenate((X, jnp.zeros((100, 1, 1))), axis=2)
        elif delta_n_features == -1:
            # remove a feature
            X = X[..., :-1]

        if raise_exception:
            with pytest.raises(ValueError, match="Inconsistent number of features"):
                model.score(X, y)
        else:
            model.score(X, y)

    @pytest.mark.parametrize("is_fit", [True, False])
    def test_score_is_fit(self, is_fit, poissonGLM_model_instantiation):
        """
        Test the `score` method on models based on their fit status.
        Ensure scoring is only possible on fitted models.
        """
        raise_exception = not is_fit
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        if is_fit:
            model.fit(X, y)

        if raise_exception:
            with pytest.raises(ValueError, match="This GLM instance is not fitted yet"):
                model.score(X, y)
        else:
            model.score(X, y)


    @pytest.mark.parametrize("delta_tp", [-1, 0, 1])
    def test_score_time_points_x(self, delta_tp, poissonGLM_model_instantiation):
        """
        Test the `score` method for inconsistencies in time-points in X.
        Ensure that the number of time-points in X and y matches.
        """
        raise_exception = delta_tp != 0
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.basis_coeff_ = true_params[0]
        model.baseline_link_fr_ = true_params[1]

        X = jnp.zeros((X.shape[0] + delta_tp,) + X.shape[1:])
        if raise_exception:
            with pytest.raises(ValueError, match="The number of time-points in X and y"):
                model.score(X, y)
        else:
            model.score(X, y)

    @pytest.mark.parametrize("delta_tp", [-1, 0, 1])
    def test_score_time_points_y(self, delta_tp, poissonGLM_model_instantiation):
        """
        Test the `score` method for inconsistencies in time-points in y.
        Ensure that the number of time-points in X and y matches.
        """
        raise_exception = delta_tp != 0
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.basis_coeff_ = true_params[0]
        model.baseline_link_fr_ = true_params[1]

        y = jnp.zeros((y.shape[0] + delta_tp,) + y.shape[1:])
        if raise_exception:
            with pytest.raises(ValueError, match="The number of time-points in X and y"):
                model.score(X, y)
        else:
            model.score(X, y)

    @pytest.mark.parametrize("score_type", ["pseudo-r2", "log-likelihood", "not-implemented"])
    def test_score_type_r2(self, score_type, poissonGLM_model_instantiation):
        """
        Test the `score` method for unsupported scoring types.
        Ensure only valid score types are used.
        """
        raise_exception = score_type not in ["pseudo-r2", "log-likelihood"]
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.basis_coeff_ = true_params[0]
        model.baseline_link_fr_ = true_params[1]

        if raise_exception:
            with pytest.raises(NotImplementedError, match=f"Scoring method {score_type} not implemented"):
                model.score(X, y, score_type=score_type)
        else:
            model.score(X, y, score_type=score_type)

    def test_loglikelihood_against_scipy_stats(self, poissonGLM_model_instantiation):
        """
        Compare the model's log-likelihood computation against `jax.scipy`.
        Ensure consistent and correct calculations.
        """
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        # set model coeff
        model.basis_coeff_ = true_params[0]
        model.baseline_link_fr_ = true_params[1]
        # get the rate
        mean_firing = model.predict(X)
        # compute the log-likelihood using jax.scipy
        mean_ll_jax = jax.scipy.stats.poisson.logpmf(y, mean_firing).mean()
        model_ll = model.score(X, y, score_type="log-likelihood")
        if not np.allclose(mean_ll_jax, model_ll):
            raise ValueError("Log-likelihood of PoissonModel does not match"
                             "that of jax.scipy!")

    
    #######################
    # Test model.predict
    #######################
    @pytest.mark.parametrize("delta_n_neuron", [-1, 0, 1])
    def test_predict_n_neuron_match_x(self, delta_n_neuron, poissonGLM_model_instantiation):
        """
        Test the `predict` method when the number of neurons in X differs.
        Ensure that the number of neurons in X, y and params matches.
        """
        raise_exception = delta_n_neuron != 0
        X, _, model, true_params, _ = poissonGLM_model_instantiation
        n_samples, n_neurons, n_features = X.shape
        # set model coeff
        model.basis_coeff_ = true_params[0]
        model.baseline_link_fr_ = true_params[1]
        X = jnp.repeat(X, n_neurons + delta_n_neuron, axis=1)
        if raise_exception:
            with pytest.raises(ValueError, match="The number of neuron in the model parameters"):
                model.predict(X)
        else:
            model.predict(X)

    @pytest.mark.parametrize("delta_dim", [-1, 0, 1])
    def test_predict_x_dimensionality(self, delta_dim, poissonGLM_model_instantiation):
        """
        Test the `predict` method with x input data of different dimensionalities.
        Ensure correct dimensionality for x.
        """
        raise_exception = delta_dim != 0
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_samples, n_neurons, n_features = X.shape
        # set model coeff
        model.basis_coeff_ = true_params[0]
        model.baseline_link_fr_ = true_params[1]

        if delta_dim == -1:
            # remove a dimension
            X = np.zeros((n_samples, n_neurons))
        elif delta_dim == 1:
            # add a dimension
            X = np.zeros((n_samples, n_neurons, n_features, 1))

        if raise_exception:
            with pytest.raises(ValueError, match="X must be three-dimensional"):
                model.predict(X)
        else:
            model.predict(X)

    @pytest.mark.parametrize("delta_n_features", [-1, 0, 1])
    def test_predict_n_feature_consistency_x(self, delta_n_features, poissonGLM_model_instantiation):
        """
        Test the `predict` method ensuring the number of features in x input data
        is consistent with the model's `model.`basis_coeff_`.
        """
        raise_exception = delta_n_features != 0
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        # set model coeff
        model.basis_coeff_ = true_params[0]
        model.baseline_link_fr_ = true_params[1]
        if delta_n_features == 1:
            # add a feature
            X = jnp.concatenate((X, jnp.zeros((100, 1, 1))), axis=2)
        elif delta_n_features == -1:
            # remove a feature
            X = X[..., :-1]

        if raise_exception:
            with pytest.raises(ValueError, match="Inconsistent number of features"):
                model.predict(X)
        else:
            model.predict(X)

    @pytest.mark.parametrize("is_fit", [True, False])
    def test_score_is_fit(self, is_fit, poissonGLM_model_instantiation):
        """
        Test if the model raises a ValueError when trying to score before it's fitted.
        """
        raise_exception = not is_fit
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        if is_fit:
            model.fit(X, y)

        if raise_exception:
            with pytest.raises(ValueError, match="This GLM instance is not fitted yet"):
                model.predict(X)
        else:
            model.predict(X)

    #######################
    # Test model.simulate
    #######################
    @pytest.mark.parametrize("delta_n_neuron", [-1, 0, 1])
    def test_simulate_n_neuron_match_input(self, delta_n_neuron,
                                       poissonGLM_coupled_model_config_simulate):
        """
        Test the `simulate` method to ensure that the number of neurons in the input
        matches the model's parameters.
        """
        raise_exception = delta_n_neuron != 0
        model, coupling_basis, feedforward_input, init_spikes, random_key = \
            poissonGLM_coupled_model_config_simulate
        n_neurons, n_features = model.basis_coeff_.shape
        n_time_points, _, n_basis_input = feedforward_input.shape
        if delta_n_neuron != 0:
            feedforward_input = np.zeros((n_time_points, n_neurons+delta_n_neuron, n_basis_input))
        if raise_exception:
            with pytest.raises(ValueError, match="The number of neuron in the model parameters"):
                model.simulate(random_key=random_key,
                               init_y=init_spikes,
                               coupling_basis_matrix=coupling_basis,
                               feedforward_input=feedforward_input,
                               device="cpu")
        else:
            model.simulate(random_key=random_key,
                           init_y=init_spikes,
                           coupling_basis_matrix=coupling_basis,
                           feedforward_input=feedforward_input,
                           device="cpu")


    @pytest.mark.parametrize("delta_dim", [-1, 0, 1])
    def test_simulate_input_dimensionality(self, delta_dim,
                                        poissonGLM_coupled_model_config_simulate):
        """
        Test the `simulate` method with input data of different dimensionalities.
        Ensure correct dimensionality for input.
        """
        raise_exception = delta_dim != 0
        model, coupling_basis, feedforward_input, init_spikes, random_key = \
            poissonGLM_coupled_model_config_simulate
        if delta_dim == -1:
            # remove a dimension
            feedforward_input = np.zeros(feedforward_input.shape[:2])
        elif delta_dim == 1:
            # add a dimension
            feedforward_input = np.zeros(feedforward_input.shape + (1,))

        if raise_exception:
            with pytest.raises(ValueError, match="X must be three-dimensional"):
                model.simulate(random_key=random_key,
                               init_y=init_spikes,
                               coupling_basis_matrix=coupling_basis,
                               feedforward_input=feedforward_input,
                               device="cpu")
        else:
            model.simulate(random_key=random_key,
                           init_y=init_spikes,
                           coupling_basis_matrix=coupling_basis,
                           feedforward_input=feedforward_input,
                           device="cpu")

    @pytest.mark.parametrize("delta_dim", [-1, 0, 1])
    def test_simulate_y_dimensionality(self, delta_dim,
                                    poissonGLM_coupled_model_config_simulate):
        """
        Test the `simulate` method with init_spikes of different dimensionalities.
        Ensure correct dimensionality for init_spikes.
        """
        raise_exception = delta_dim != 0
        model, coupling_basis, feedforward_input, init_spikes, random_key = \
            poissonGLM_coupled_model_config_simulate
        n_samples, n_neurons = feedforward_input.shape[:2]
        if delta_dim == -1:
            # remove a dimension
            init_spikes = np.zeros((n_samples,))
        elif delta_dim == 1:
            # add a dimension
            init_spikes = np.zeros((n_samples, n_neurons, 1))

        if raise_exception:
            with pytest.raises(ValueError, match="y must be two-dimensional"):
                model.simulate(random_key=random_key,
                               init_y=init_spikes,
                               coupling_basis_matrix=coupling_basis,
                               feedforward_input=feedforward_input,
                               device="cpu")
        else:
            model.simulate(random_key=random_key,
                           init_y=init_spikes,
                           coupling_basis_matrix=coupling_basis,
                           feedforward_input=feedforward_input,
                           device="cpu")

    @pytest.mark.parametrize("delta_n_neuron", [-1, 0, 1])
    def test_simulate_n_neuron_match_y(self, delta_n_neuron,
                                       poissonGLM_coupled_model_config_simulate):
        """
        Test the `simulate` method to ensure that the number of neurons in init_spikes
        matches the model's parameters.
        """
        raise_exception = delta_n_neuron != 0
        model, coupling_basis, feedforward_input, init_spikes, random_key = \
            poissonGLM_coupled_model_config_simulate
        n_samples, n_neurons = feedforward_input.shape[:2]

        init_spikes = jnp.zeros((init_spikes.shape[0], n_neurons + delta_n_neuron))
        if raise_exception:
            with pytest.raises(ValueError, match="The number of neuron in the model parameters"):
                model.simulate(random_key=random_key,
                               init_y=init_spikes,
                               coupling_basis_matrix=coupling_basis,
                               feedforward_input=feedforward_input,
                               device="cpu")
        else:
            model.simulate(random_key=random_key,
                           init_y=init_spikes,
                           coupling_basis_matrix=coupling_basis,
                           feedforward_input=feedforward_input,
                           device="cpu")

    @pytest.mark.parametrize("is_fit", [True, False])
    def test_simulate_is_fit(self, is_fit,
                             poissonGLM_coupled_model_config_simulate):
        """
        Test if the model raises a ValueError when trying to simulate before it's fitted.
        """
        raise_exception = not is_fit
        model, coupling_basis, feedforward_input, init_spikes, random_key = \
            poissonGLM_coupled_model_config_simulate

        if not is_fit:
            model.baseline_link_fr_ = None

        if raise_exception:
            with pytest.raises(ValueError, match="This GLM instance is not fitted yet"):
                model.simulate(random_key=random_key,
                               init_y=init_spikes,
                               coupling_basis_matrix=coupling_basis,
                               feedforward_input=feedforward_input,
                               device="cpu")
        else:
            model.simulate(random_key=random_key,
                           init_y=init_spikes,
                           coupling_basis_matrix=coupling_basis,
                           feedforward_input=feedforward_input,
                           device="cpu")

    @pytest.mark.parametrize("delta_tp", [-1, 0, 1])
    def test_simulate_time_point_match_y(self, delta_tp,
                             poissonGLM_coupled_model_config_simulate):
        """
        Test the `simulate` method to ensure that the time points in init_y
        are consistent with the coupling_basis window size (they must be equal).
        """
        raise_exception = delta_tp != 0
        model, coupling_basis, feedforward_input, init_spikes, random_key = \
            poissonGLM_coupled_model_config_simulate

        init_spikes = jnp.zeros((init_spikes.shape[0]+delta_tp,
                                     init_spikes.shape[1]))

        if raise_exception:
            with pytest.raises(ValueError, match="`init_y` and `coupling_basis_matrix`"):
                model.simulate(random_key=random_key,
                               init_y=init_spikes,
                               coupling_basis_matrix=coupling_basis,
                               feedforward_input=feedforward_input,
                               device="cpu")
        else:
            model.simulate(random_key=random_key,
                           init_y=init_spikes,
                           coupling_basis_matrix=coupling_basis,
                           feedforward_input=feedforward_input,
                           device="cpu")

    @pytest.mark.parametrize("delta_tp", [-1, 0, 1])
    def test_simulate_time_point_match_coupling_basis(self, delta_tp,
                                         poissonGLM_coupled_model_config_simulate):
        """
        Test the `simulate` method to ensure that the window size in coupling_basis
        is consistent with the time-points in init_spikes (they must be equal).
        """
        raise_exception = delta_tp != 0
        model, coupling_basis, feedforward_input, init_spikes, random_key = \
            poissonGLM_coupled_model_config_simulate

        coupling_basis = jnp.zeros((coupling_basis.shape[0] + delta_tp,) +
                                   coupling_basis.shape[1:])

        if raise_exception:
            with pytest.raises(ValueError, match="`init_y` and `coupling_basis_matrix`"):
                model.simulate(random_key=random_key,
                               init_y=init_spikes,
                               coupling_basis_matrix=coupling_basis,
                               feedforward_input=feedforward_input,
                               device="cpu")
        else:
            model.simulate(random_key=random_key,
                           init_y=init_spikes,
                           coupling_basis_matrix=coupling_basis,
                           feedforward_input=feedforward_input,
                           device="cpu")

    @pytest.mark.parametrize("delta_features", [-1, 0, 1])
    def test_simulate_feature_consistency_input(self, delta_features,
                                           poissonGLM_coupled_model_config_simulate):
        """
        Test the `simulate` method ensuring the number of features in `feedforward_input` is
        consistent with the model's expected number of features.

        Notes
        -----
        The total feature number `model.basis_coeff_.shape[1]` must be equal to
        `feedforward_input.shape[2] + coupling_basis.shape[1]*n_neurons`
        """
        raise_exception = delta_features != 0
        model, coupling_basis, feedforward_input, init_spikes, random_key = \
            poissonGLM_coupled_model_config_simulate
        feedforward_input = jnp.zeros((feedforward_input.shape[0],
                                       feedforward_input.shape[1],
                                       feedforward_input.shape[2]+delta_features))
        if raise_exception:
            with pytest.raises(ValueError, match="Inconsistent number of features"):
                model.simulate(random_key=random_key,
                               init_y=init_spikes,
                               coupling_basis_matrix=coupling_basis,
                               feedforward_input=feedforward_input,
                               device="cpu")
        else:
            model.simulate(random_key=random_key,
                           init_y=init_spikes,
                           coupling_basis_matrix=coupling_basis,
                           feedforward_input=feedforward_input,
                           device="cpu")

    @pytest.mark.parametrize("delta_features", [-1, 0, 1])
    def test_simulate_feature_consistency_coupling_basis(self, delta_features,
                                                poissonGLM_coupled_model_config_simulate):
        """
        Test the `simulate` method ensuring the number of features in `coupling_basis` is
        consistent with the model's expected number of features.

        Notes
        -----
        The total feature number `model.basis_coeff_.shape[1]` must be equal to
        `feedforward_input.shape[2] + coupling_basis.shape[1]*n_neurons`
        """
        raise_exception = delta_features != 0
        model, coupling_basis, feedforward_input, init_spikes, random_key = \
            poissonGLM_coupled_model_config_simulate
        coupling_basis = jnp.zeros((coupling_basis.shape[0],
                                    coupling_basis.shape[1] + delta_features))
        if raise_exception:
            with pytest.raises(ValueError, match="Inconsistent number of features"):
                model.simulate(random_key=random_key,
                               init_y=init_spikes,
                               coupling_basis_matrix=coupling_basis,
                               feedforward_input=feedforward_input,
                               device="cpu")
        else:
            model.simulate(random_key=random_key,
                           init_y=init_spikes,
                           coupling_basis_matrix=coupling_basis,
                           feedforward_input=feedforward_input,
                           device="cpu")

    @pytest.mark.parametrize("device_spec", ["cpu", "tpu", "gpu", "none", 1])
    def test_simulate_device_spec(self, device_spec,
                                   poissonGLM_coupled_model_config_simulate):
        """
        Test `simulate` across different device specifications.
        Validates if unsupported or absent devices raise exception
        or warning respectively.
        """
        raise_exception = not (device_spec in ["cpu", "tpu", "gpu"])
        print(device_spec, raise_exception)
        raise_warning = all(device_spec != device.device_kind.lower()
                            for device in jax.local_devices())
        raise_warning = raise_warning and (not raise_exception)

        model, coupling_basis, feedforward_input, init_spikes, random_key = \
            poissonGLM_coupled_model_config_simulate

        if raise_exception:
            with pytest.raises(ValueError, match=f"Invalid device specification: {device_spec}"):
                model.simulate(random_key=random_key,
                               init_y=init_spikes,
                               coupling_basis_matrix=coupling_basis,
                               feedforward_input=feedforward_input,
                               device=device_spec)
        elif raise_warning:
            with pytest.warns(UserWarning, match=f"No {device_spec.upper()} found"):
                model.simulate(random_key=random_key,
                               init_y=init_spikes,
                               coupling_basis_matrix=coupling_basis,
                               feedforward_input=feedforward_input,
                               device=device_spec)
        else:
            model.simulate(random_key=random_key,
                           init_y=init_spikes,
                           coupling_basis_matrix=coupling_basis,
                           feedforward_input=feedforward_input,
                           device=device_spec)
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
        model.basis_coeff_ = true_params[0]
        model.baseline_link_fr_ = true_params[1]
        # get the rate
        dev = sm.families.Poisson().deviance(y, firing_rate)
        dev_model = model.noise_model.residual_deviance(firing_rate, y).sum()
        if not np.allclose(dev, dev_model):
            raise ValueError("Deviance doesn't match statsmodels!")

    def test_compare_fit_estimate_to_statsmodels(self, poissonGLM_model_instantiation):
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        glm_sm = sm.GLM(endog=y[:, 0],
                        exog=sm.add_constant(X[:, 0]),
                        family=sm.families.Poisson())
        res_sm = glm_sm.fit()
        fit_params_sm = res_sm.params
        # use a second order method for precision, match non-linearity
        model.set_params(noise_model__inverse_link_function=jnp.exp,
                         solver__solver_name="BFGS",
                         solver__solver_kwargs={"tol": 10**-8})
        model.fit(X, y)
        fit_params_model = jnp.hstack((model.baseline_link_fr_,
                                       model.basis_coeff_.flatten()))
        if not np.allclose(fit_params_sm, fit_params_model):
            raise ValueError("Fitted parameters do not match that of statsmodels!")

    def test_compatibility_with_sklearn_cv(self, poissonGLM_model_instantiation):
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        param_grid = {"solver__solver_name": ["BFGS", "GradientDescent"]}
        GridSearchCV(model, param_grid).fit(X, y)

    def test_end_to_end_fit_and_simulate(self,
                                   poissonGLM_coupled_model_config_simulate):
        model, coupling_basis, feedforward_input, init_spikes, random_key = \
            poissonGLM_coupled_model_config_simulate
        window_size = coupling_basis.shape[0]
        n_neurons = init_spikes.shape[1]
        n_trials = 1
        n_timepoints = feedforward_input.shape[0]

        # generate spike trains
        spikes, _ = model.simulate(random_key=random_key,
                       init_y=init_spikes,
                       coupling_basis_matrix=coupling_basis,
                       feedforward_input=feedforward_input,
                       device="cpu")

        # convolve basis and spikes
        # (n_trials, n_timepoints - ws + 1, n_neurons, n_coupling_basis)
        conv_spikes = jnp.asarray(
            nsl.utils.convolve_1d_trials(coupling_basis, [spikes]),
            dtype=jnp.float32
        )

        # create an individual neuron predictor by stacking the
        # two convolved spike trains in a single feature vector
        # and concatenate the trials.
        conv_spikes = conv_spikes.reshape(n_trials * (n_timepoints - window_size + 1), -1)

        # replicate for each neuron,
        # (n_trials * (n_timepoints - ws + 1), n_neurons, n_neurons * n_coupling_basis)
        conv_spikes = jnp.tile(conv_spikes, n_neurons).reshape(conv_spikes.shape[0],
                                                               n_neurons,
                                                               conv_spikes.shape[1])

        # add the feed-forward input to the predictors
        X = jnp.concatenate((conv_spikes[1:],
                             feedforward_input[:-window_size]),
                            axis=2)

        # fit the model
        model.fit(X, spikes[:-window_size])

        # simulate
        model.simulate(random_key=random_key,
                       init_y=init_spikes,
                       coupling_basis_matrix=coupling_basis,
                       feedforward_input=feedforward_input,
                       device="cpu")



