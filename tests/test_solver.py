import jax
import jax.numpy as jnp
import numpy as np
import pytest
from sklearn.linear_model import PoissonRegressor
import statsmodels.api as sm

import neurostatslib as nsl


class TestSolver:
    cls = nsl.solver.Solver
    def test_abstract_nature_of_solver(self):
        """Test that Solver can't be instantiated."""
        with pytest.raises(TypeError, match="TypeError: Can't instantiate abstract class Solver"):
            self.cls("GradientDescent")

class TestRidgeSolver:
    cls = nsl.solver.RidgeSolver

    @pytest.mark.parametrize("solver_name", ["GradientDescent", "BFGS", "ProximalGradient", "AGradientDescent", 1])
    def test_init_solver_name(self, solver_name):
        """Test RidgeSolver acceptable solvers."""
        acceptable_solvers = [
            "GradientDescent",
            "BFGS",
            "LBFGS",
            "ScipyMinimize",
            "NonlinearCG",
            "ScipyBoundedMinimize",
            "LBFGSB"
        ]
        raise_exception = solver_name not in acceptable_solvers
        if raise_exception:
            with pytest.raises(ValueError, match=f"Solver `{solver_name}` not allowed for "):
                self.cls(solver_name)
        else:
            self.cls(solver_name)

    @pytest.mark.parametrize("solver_name", ["GradientDescent", "BFGS"])
    @pytest.mark.parametrize("solver_kwargs", [{"tol": 10**-10}, {"tols": 10**-10}])
    def test_init_solver_kwargs(self, solver_name, solver_kwargs):
        """Test RidgeSolver acceptable kwargs."""

        raise_exception = "tols" in list(solver_kwargs.keys())
        if raise_exception:
            with pytest.raises(NameError, match="kwargs {'tols'} in solver_kwargs not a kwarg"):
                self.cls(solver_name, solver_kwargs=solver_kwargs)
        else:
            self.cls(solver_name, solver_kwargs=solver_kwargs)

    @pytest.mark.parametrize("loss", [jnp.exp, np.exp, 1, None, {}])
    def test_loss_callable(self, loss):
        """Test that the loss function is a callable"""
        raise_exception = not callable(loss)
        if raise_exception:
            with pytest.raises(TypeError, match="The loss function must a Callable"):
                self.cls("GradientDescent").instantiate_solver(loss)
        else:
            self.cls("GradientDescent").instantiate_solver(loss)

    @pytest.mark.parametrize("solver_name", ["GradientDescent", "BFGS"])
    def test_run_solver(self, solver_name, poissonGLM_model_instantiation):
        """Test that the solver runs."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        runner = self.cls("GradientDescent").instantiate_solver(model._score)
        runner((true_params[0]*0., true_params[1]), X, y)

    def test_solver_output_match(self, poissonGLM_model_instantiation):
        """Test that different solvers converge to the same solution."""
        jax.config.update("jax_enable_x64", True)
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        # set precision to float64 for accurate matching of the results
        model.data_type = jnp.float64
        runner_gd = self.cls("GradientDescent", {"tol": 10**-12}).instantiate_solver(model._score)
        runner_bfgs = self.cls("BFGS", {"tol": 10**-12}).instantiate_solver(model._score)
        runner_scipy = self.cls("ScipyMinimize", {"method": "BFGS", "tol": 10**-12}).instantiate_solver(model._score)
        weights_gd, intercepts_gd = runner_gd((true_params[0] * 0., true_params[1]), X, y)[0]
        weights_bfgs, intercepts_bfgs = runner_bfgs((true_params[0] * 0., true_params[1]), X, y)[0]
        weights_scipy, intercepts_scipy = runner_scipy((true_params[0] * 0., true_params[1]), X, y)[0]

        match_weights = np.allclose(weights_gd, weights_bfgs) and \
                        np.allclose(weights_gd, weights_scipy)
        match_intercepts = np.allclose(intercepts_gd, intercepts_bfgs) and \
                           np.allclose(intercepts_gd, intercepts_scipy)
        if (not match_weights) or (not match_intercepts):
            raise ValueError("Convex estimators should converge to the same numerical value.")

    def test_solver_match_sklearn(self, poissonGLM_model_instantiation):
        """Test that different solvers converge to the same solution."""
        jax.config.update("jax_enable_x64", True)
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        # set precision to float64 for accurate matching of the results
        model.data_type = jnp.float64
        solver = self.cls("GradientDescent", {"tol": 10**-12})
        runner_bfgs = solver.instantiate_solver(model._score)
        weights_bfgs, intercepts_bfgs = runner_bfgs((true_params[0] * 0., true_params[1]), X, y)[0]
        model_skl = PoissonRegressor(fit_intercept=True, tol=10**-12, alpha=solver.alpha)
        model_skl.fit(X[:,0], y[:, 0])

        match_weights = np.allclose(model_skl.coef_, weights_bfgs.flatten())
        match_intercepts = np.allclose(model_skl.intercept_, intercepts_bfgs.flatten())
        if (not match_weights) or (not match_intercepts):
            raise ValueError("Ridge GLM solver estimate does not match sklearn!")


class TestLassoSolver:
    cls = nsl.solver.LassoSolver

    @pytest.mark.parametrize("solver_name", ["GradientDescent", "BFGS", "ProximalGradient", "AGradientDescent", 1])
    def test_init_solver_name(self, solver_name):
        """Test RidgeSolver acceptable solvers."""
        acceptable_solvers = [
            "ProximalGradient"
        ]
        raise_exception = solver_name not in acceptable_solvers
        if raise_exception:
            with pytest.raises(ValueError, match=f"Solver `{solver_name}` not allowed for "):
                self.cls(solver_name)
        else:
            self.cls(solver_name)

    @pytest.mark.parametrize("solver_kwargs", [{"tol": 10**-10}, {"tols": 10**-10}])
    def test_init_solver_kwargs(self, solver_kwargs):
        """Test RidgeSolver acceptable kwargs."""
        raise_exception = "tols" in list(solver_kwargs.keys())
        if raise_exception:
            with pytest.raises(NameError, match="kwargs {'tols'} in solver_kwargs not a kwarg"):
                self.cls("ProximalGradient", solver_kwargs=solver_kwargs)
        else:
            self.cls("ProximalGradient", solver_kwargs=solver_kwargs)

    @pytest.mark.parametrize("loss", [jnp.exp, jax.nn.relu, 1, None, {}])
    def test_loss_callable(self, loss):
        """Test that the loss function is a callable"""
        raise_exception = not callable(loss)
        if raise_exception:
            with pytest.raises(TypeError, match="The loss function must a Callable"):
                self.cls("ProximalGradient").instantiate_solver(loss)
        else:
            self.cls("ProximalGradient").instantiate_solver(loss)

    def test_run_solver(self, poissonGLM_model_instantiation):
        """Test that the solver runs."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        runner = self.cls("ProximalGradient").instantiate_solver(model._score)
        runner((true_params[0]*0., true_params[1]), X, y)

    def test_solver_match_sklearn(self, poissonGLM_model_instantiation):
        """Test that different solvers converge to the same solution."""
        jax.config.update("jax_enable_x64", True)
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        # set precision to float64 for accurate matching of the results
        model.data_type = jnp.float64
        solver = self.cls("ProximalGradient", {"tol": 10**-12})
        runner = solver.instantiate_solver(model._score)
        weights, intercepts = runner((true_params[0] * 0., true_params[1]), X, y)[0]

        # instantiate the glm with statsmodels
        glm_sm = sm.GLM(endog=y[:, 0],
                        exog=sm.add_constant(X[:, 0]),
                        family=sm.families.Poisson())

        # regularize everything except intercept
        alpha_sm = np.ones(X.shape[2] + 1) * solver.alpha
        alpha_sm[0] = 0

        # pure lasso = elastic net with L1 weight = 1
        res_sm = glm_sm.fit_regularized(method="elastic_net",
                                        alpha=alpha_sm,
                                        L1_wt=1., cnvrg_tol=10**-12)
        # compare params
        sm_params = res_sm.params
        glm_params = jnp.hstack((intercepts, weights.flatten()))
        match_weights = np.allclose(sm_params, glm_params)
        if not match_weights:
            raise ValueError("Lasso GLM solver estimate does not match statsmodels!")
