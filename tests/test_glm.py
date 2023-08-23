import pytest

from typing import Literal, Callable

import jax, jaxopt
import jax.numpy as jnp
import numpy as np
import statsmodels.api as sm

import neurostatslib as nsl


class TestPoissonGLM:
    #######################
    # Test model.__init__
    #######################
    @pytest.mark.parametrize("solver_name", ["GradientDescent", "BFGS", "ScipyMinimize", "NotPresent"])
    def test_init_solver_name(self, solver_name: str):
        try:
            getattr(jaxopt, solver_name)
            raise_exception = False
        except:
            raise_exception = True
        if raise_exception:
            with pytest.raises(AttributeError, match="module jaxopt has no attribute"):
                nsl.glm.PoissonGLM(solver_name=solver_name)
        else:
            nsl.glm.PoissonGLM(solver_name=solver_name)

    @pytest.mark.parametrize("solver_name", ["GradientDescent", "BFGS", "ScipyMinimize"])
    @pytest.mark.parametrize("solver_kwargs", [
        {"tol": 1, "verbose": 1, "maxiter": 1},
        {"tol": 1, "maxiter": 1}])
    def test_init_solver_kwargs(self, solver_name, solver_kwargs):
        raise_exception = (solver_name == "ScipyMinimize") & ("verbose" in solver_kwargs)
        if raise_exception:
            with pytest.raises(NameError, match="kwargs {'[a-z]+'} in solver_kwargs not a kwarg"):
                nsl.glm.PoissonGLM(solver_name, solver_kwargs=solver_kwargs)
        else:
            # define glm and instantiate the solver
            nsl.glm.PoissonGLM(solver_name, solver_kwargs=solver_kwargs)
            getattr(jaxopt, solver_name)(fun=lambda x: x, **solver_kwargs)

    @pytest.mark.parametrize("func", [1, "string", lambda x: x, jnp.exp])
    def test_init_callable(self, func: Callable[[jnp.ndarray], jnp.ndarray]):
        if not callable(func):
            with pytest.raises(ValueError, match="inverse_link_function must be a callable"):
                nsl.glm.PoissonGLM("BFGS", inverse_link_function=func)
        else:
            nsl.glm.PoissonGLM("BFGS", inverse_link_function=func)

    @pytest.mark.parametrize("score_type", [1, "ll", "log-likelihood", "pseudo-r2"])
    def test_init_score_type(self, score_type: Literal["log-likelihood", "pseudo-r2"]):
        if score_type not in ["log-likelihood", "pseudo-r2"]:
            with pytest.raises(NotImplementedError, match="Scoring method not implemented."):
                nsl.glm.PoissonGLM("BFGS", score_type=score_type)
        else:
            nsl.glm.PoissonGLM("BFGS", score_type=score_type)

    #######################
    # Test model.fit
    #######################
    @pytest.mark.parametrize("n_params", [0, 1, 2, 3])
    def test_fit_param_length(self, n_params, poissonGLM_model_instantiation):
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

    @pytest.mark.parametrize("dim_weights", [0, 1, 2, 3])
    def test_fit_weights_dimensionality(self, dim_weights, poissonGLM_model_instantiation):
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
            with pytest.raises(ValueError, match="params\[0\] term must be of shape \(n_neurons, n_features\)"):
                model.fit(X, y, init_params=(init_w, init_b))
        else:
            model.fit(X, y, init_params=(init_w, init_b))

    @pytest.mark.parametrize("dim_intercepts", [0, 1, 2, 3])
    def test_fit_intercepts_dimensionality(self, dim_intercepts, poissonGLM_model_instantiation):
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_samples, n_neurons, n_features = X.shape

        init_b = jnp.zeros((n_neurons,) * dim_intercepts)
        init_w = jnp.zeros((n_neurons, n_features))
        raise_exception = dim_intercepts != 1
        if raise_exception:
            with pytest.raises(ValueError, match="params\[1\] term must be of shape"):
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
            with pytest.raises(ValueError, match="spike_data must be two-dimensional"):
                model.fit(X, y, init_params=(init_w, init_b))
        else:
            model.fit(X, y, init_params=(init_w, init_b))

    @pytest.mark.parametrize("delta_n_features", [-1, 0, 1])
    def test_fit_n_feature_consistency_weights(self, delta_n_features, poissonGLM_model_instantiation):
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
        raise_exception = delta_tp != 0
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_samples, n_neurons, n_features = X.shape
        init_w = jnp.zeros((n_neurons, n_features))
        init_b = jnp.zeros((n_neurons,))
        X = jnp.zeros((X.shape[0] + delta_tp, ) + X.shape[1:])
        if raise_exception:
            with pytest.raises(ValueError, match="The number of time-points in X and spike_data"):
                model.fit(X, y, init_params=(init_w, init_b))
        else:
            model.fit(X, y, init_params=(init_w, init_b))

    @pytest.mark.parametrize("delta_tp", [-1, 0, 1])
    def test_fit_time_points_y(self, delta_tp, poissonGLM_model_instantiation):
        raise_exception = delta_tp != 0
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_samples, n_neurons, n_features = X.shape
        init_w = jnp.zeros((n_neurons, n_features))
        init_b = jnp.zeros((n_neurons,))
        y = jnp.zeros((y.shape[0] + delta_tp,) + y.shape[1:])
        if raise_exception:
            with pytest.raises(ValueError, match="The number of time-points in X and spike_data"):
                model.fit(X, y, init_params=(init_w, init_b))
        else:
            model.fit(X, y, init_params=(init_w, init_b))

    #######################
    # Test model.score
    #######################
    @pytest.mark.parametrize("delta_n_neuron", [-1, 0, 1])
    def test_score_n_neuron_match_x(self, delta_n_neuron, poissonGLM_model_instantiation):
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
            with pytest.raises(ValueError, match="spike_data must be two-dimensional"):
                model.score(X, y)
        else:
            model.score(X, y)

    @pytest.mark.parametrize("delta_n_features", [-1, 0, 1])
    def test_score_n_feature_consistency_x(self, delta_n_features, poissonGLM_model_instantiation):
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
    def test_fit_time_points_x(self, delta_tp, poissonGLM_model_instantiation):
        raise_exception = delta_tp != 0
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        n_samples, n_neurons, n_features = X.shape
        init_w = jnp.zeros((n_neurons, n_features))
        init_b = jnp.zeros((n_neurons,))
        X = jnp.zeros((X.shape[0] + delta_tp,) + X.shape[1:])
        if raise_exception:
            with pytest.raises(ValueError, match="The number of time-points in X and spike_data"):
                model.fit(X, y, init_params=(init_w, init_b))
        else:
            model.fit(X, y, init_params=(init_w, init_b))

    @pytest.mark.parametrize("delta_tp", [-1, 0, 1])
    def test_score_time_points_x(self, delta_tp, poissonGLM_model_instantiation):
        raise_exception = delta_tp != 0
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.basis_coeff_ = true_params[0]
        model.baseline_link_fr_ = true_params[1]

        X = jnp.zeros((X.shape[0] + delta_tp,) + X.shape[1:])
        if raise_exception:
            with pytest.raises(ValueError, match="The number of time-points in X and spike_data"):
                model.score(X, y)
        else:
            model.score(X, y)

    @pytest.mark.parametrize("delta_tp", [-1, 0, 1])
    def test_score_time_points_y(self, delta_tp, poissonGLM_model_instantiation):
        raise_exception = delta_tp != 0
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.basis_coeff_ = true_params[0]
        model.baseline_link_fr_ = true_params[1]

        y = jnp.zeros((y.shape[0] + delta_tp,) + y.shape[1:])
        if raise_exception:
            with pytest.raises(ValueError, match="The number of time-points in X and spike_data"):
                model.score(X, y)
        else:
            model.score(X, y)

    def test_loglikelihood_against_scipy_stats(self, poissonGLM_model_instantiation):
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
                               n_timesteps=n_time_points,
                               init_spikes=init_spikes,
                               coupling_basis_matrix=coupling_basis,
                               feedforward_input=feedforward_input,
                               device="cpu")
        else:
            model.simulate(random_key=random_key,
                           n_timesteps=n_time_points,
                           init_spikes=init_spikes,
                           coupling_basis_matrix=coupling_basis,
                           feedforward_input=feedforward_input,
                           device="cpu")


    @pytest.mark.parametrize("delta_dim", [-1, 0, 1])
    def test_simulate_input_dimensionality(self, delta_dim,
                                        poissonGLM_coupled_model_config_simulate):
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
                               n_timesteps=feedforward_input.shape[0],
                               init_spikes=init_spikes,
                               coupling_basis_matrix=coupling_basis,
                               feedforward_input=feedforward_input,
                               device="cpu")
        else:
            model.simulate(random_key=random_key,
                           n_timesteps=feedforward_input.shape[0],
                           init_spikes=init_spikes,
                           coupling_basis_matrix=coupling_basis,
                           feedforward_input=feedforward_input,
                           device="cpu")

    @pytest.mark.parametrize("delta_dim", [-1, 0, 1])
    def test_simulate_y_dimensionality(self, delta_dim,
                                    poissonGLM_coupled_model_config_simulate):
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
            with pytest.raises(ValueError, match="spike_data must be two-dimensional"):
                model.simulate(random_key=random_key,
                               n_timesteps=feedforward_input.shape[0],
                               init_spikes=init_spikes,
                               coupling_basis_matrix=coupling_basis,
                               feedforward_input=feedforward_input,
                               device="cpu")
        else:
            model.simulate(random_key=random_key,
                           n_timesteps=feedforward_input.shape[0],
                           init_spikes=init_spikes,
                           coupling_basis_matrix=coupling_basis,
                           feedforward_input=feedforward_input,
                           device="cpu")

    @pytest.mark.parametrize("delta_n_neuron", [-1, 0, 1])
    def test_simulate_n_neuron_match_y(self, delta_n_neuron,
                                       poissonGLM_coupled_model_config_simulate):
        raise_exception = delta_n_neuron != 0
        model, coupling_basis, feedforward_input, init_spikes, random_key = \
            poissonGLM_coupled_model_config_simulate
        n_samples, n_neurons = feedforward_input.shape[:2]

        init_spikes = jnp.zeros((init_spikes.shape[0], n_neurons + delta_n_neuron))
        if raise_exception:
            with pytest.raises(ValueError, match="The number of neuron in the model parameters"):
                model.simulate(random_key=random_key,
                               n_timesteps=feedforward_input.shape[0],
                               init_spikes=init_spikes,
                               coupling_basis_matrix=coupling_basis,
                               feedforward_input=feedforward_input,
                               device="cpu")
        else:
            model.simulate(random_key=random_key,
                           n_timesteps=feedforward_input.shape[0],
                           init_spikes=init_spikes,
                           coupling_basis_matrix=coupling_basis,
                           feedforward_input=feedforward_input,
                           device="cpu")

    @pytest.mark.parametrize("is_fit", [True, False])
    def test_simulate_is_fit(self, is_fit,
                             poissonGLM_coupled_model_config_simulate):
        raise_exception = not is_fit
        model, coupling_basis, feedforward_input, init_spikes, random_key = \
            poissonGLM_coupled_model_config_simulate

        if not is_fit:
            model.baseline_link_fr_ = None

        if raise_exception:
            with pytest.raises(ValueError, match="This GLM instance is not fitted yet"):
                model.simulate(random_key=random_key,
                               n_timesteps=feedforward_input.shape[0],
                               init_spikes=init_spikes,
                               coupling_basis_matrix=coupling_basis,
                               feedforward_input=feedforward_input,
                               device="cpu")
        else:
            model.simulate(random_key=random_key,
                           n_timesteps=feedforward_input.shape[0],
                           init_spikes=init_spikes,
                           coupling_basis_matrix=coupling_basis,
                           feedforward_input=feedforward_input,
                           device="cpu")

    @pytest.mark.parametrize("delta_tp", [-1, 0, 1])
    def test_simulate_time_point_match_y(self, delta_tp,
                             poissonGLM_coupled_model_config_simulate):
        raise_exception = delta_tp != 0
        model, coupling_basis, feedforward_input, init_spikes, random_key = \
            poissonGLM_coupled_model_config_simulate

        init_spikes = jnp.zeros((init_spikes.shape[0]+delta_tp,
                                     init_spikes.shape[1]))

        if raise_exception:
            with pytest.raises(ValueError, match="`init_spikes` and `coupling_basis_matrix`"):
                model.simulate(random_key=random_key,
                               n_timesteps=feedforward_input.shape[0],
                               init_spikes=init_spikes,
                               coupling_basis_matrix=coupling_basis,
                               feedforward_input=feedforward_input,
                               device="cpu")
        else:
            model.simulate(random_key=random_key,
                           n_timesteps=feedforward_input.shape[0],
                           init_spikes=init_spikes,
                           coupling_basis_matrix=coupling_basis,
                           feedforward_input=feedforward_input,
                           device="cpu")

    @pytest.mark.parametrize("delta_tp", [-1, 0, 1])
    def test_simulate_time_point_match_coupling_basis(self, delta_tp,
                                         poissonGLM_coupled_model_config_simulate):
        raise_exception = delta_tp != 0
        model, coupling_basis, feedforward_input, init_spikes, random_key = \
            poissonGLM_coupled_model_config_simulate

        coupling_basis = jnp.zeros((coupling_basis.shape[0] + delta_tp,) +
                                   coupling_basis.shape[1:])

        if raise_exception:
            with pytest.raises(ValueError, match="`init_spikes` and `coupling_basis_matrix`"):
                model.simulate(random_key=random_key,
                               n_timesteps=feedforward_input.shape[0],
                               init_spikes=init_spikes,
                               coupling_basis_matrix=coupling_basis,
                               feedforward_input=feedforward_input,
                               device="cpu")
        else:
            model.simulate(random_key=random_key,
                           n_timesteps=feedforward_input.shape[0],
                           init_spikes=init_spikes,
                           coupling_basis_matrix=coupling_basis,
                           feedforward_input=feedforward_input,
                           device="cpu")

    @pytest.mark.parametrize("delta_features", [-1, 0, 1])
    def test_simulate_feature_consistency_input(self, delta_features,
                                           poissonGLM_coupled_model_config_simulate):
        raise_exception = delta_features != 0
        model, coupling_basis, feedforward_input, init_spikes, random_key = \
            poissonGLM_coupled_model_config_simulate
        feedforward_input = jnp.zeros((feedforward_input.shape[0],
                                       feedforward_input.shape[1],
                                       feedforward_input.shape[2]+delta_features))
        if raise_exception:
            with pytest.raises(ValueError, match="The number of feed forward input features"):
                model.simulate(random_key=random_key,
                               n_timesteps=feedforward_input.shape[0],
                               init_spikes=init_spikes,
                               coupling_basis_matrix=coupling_basis,
                               feedforward_input=feedforward_input,
                               device="cpu")
        else:
            model.simulate(random_key=random_key,
                           n_timesteps=feedforward_input.shape[0],
                           init_spikes=init_spikes,
                           coupling_basis_matrix=coupling_basis,
                           feedforward_input=feedforward_input,
                           device="cpu")

    @pytest.mark.parametrize("delta_features", [-1, 0, 1])
    def test_simulate_feature_consistency_coupling_basis(self, delta_features,
                                                poissonGLM_coupled_model_config_simulate):
        raise_exception = delta_features != 0
        model, coupling_basis, feedforward_input, init_spikes, random_key = \
            poissonGLM_coupled_model_config_simulate
        coupling_basis = jnp.zeros((coupling_basis.shape[0],
                                    coupling_basis.shape[1] + delta_features))
        if raise_exception:
            with pytest.raises(ValueError, match="The number of feed forward input features"):
                model.simulate(random_key=random_key,
                               n_timesteps=feedforward_input.shape[0],
                               init_spikes=init_spikes,
                               coupling_basis_matrix=coupling_basis,
                               feedforward_input=feedforward_input,
                               device="cpu")
        else:
            model.simulate(random_key=random_key,
                           n_timesteps=feedforward_input.shape[0],
                           init_spikes=init_spikes,
                           coupling_basis_matrix=coupling_basis,
                           feedforward_input=feedforward_input,
                           device="cpu")

    @pytest.mark.parametrize("delta_tp", [-1, 0, 1])
    def test_simulate_input_timepoints(self, delta_tp,
                                      poissonGLM_coupled_model_config_simulate):
        raise_exception = delta_tp != 0
        model, coupling_basis, feedforward_input, init_spikes, random_key = \
            poissonGLM_coupled_model_config_simulate
        n_timesteps = feedforward_input.shape[0]
        feedforward_input = jnp.zeros((feedforward_input.shape[0] + delta_tp,
                                       feedforward_input.shape[1],
                                       feedforward_input.shape[2]))
        if raise_exception:
            with pytest.raises(ValueError, match="`feedforward_input` must be of length"):
                model.simulate(random_key=random_key,
                               n_timesteps=n_timesteps,
                               init_spikes=init_spikes,
                               coupling_basis_matrix=coupling_basis,
                               feedforward_input=feedforward_input,
                               device="cpu")
        else:
            model.simulate(random_key=random_key,
                           n_timesteps=n_timesteps,
                           init_spikes=init_spikes,
                           coupling_basis_matrix=coupling_basis,
                           feedforward_input=feedforward_input,
                           device="cpu")

    #######################################
    # Compare with standard implementation
    #######################################
    def test_deviance_against_statsmodels(self, poissonGLM_model_instantiation):
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        # set model coeff
        model.basis_coeff_ = true_params[0]
        model.baseline_link_fr_ = true_params[1]
        model.set_params(inverse_link_function=jnp.exp)
        # get the rate
        dev = sm.families.Poisson().deviance(y, firing_rate)
        dev_model = model._residual_deviance(firing_rate, y).sum()
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
        model.set_params(inverse_link_function=jnp.exp,
                         solver_name="BFGS",
                         solver_kwargs={"tol": 10**-8})
        model.fit(X, y)
        fit_params_model = jnp.hstack((model.baseline_link_fr_,
                                       model.basis_coeff_.flatten()))
        if not np.allclose(fit_params_sm, fit_params_model):
            raise ValueError("Fitted parameters do not match that of statsmodels!")

