import warnings
from contextlib import nullcontext as does_not_raise

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import statsmodels.api as sm
from sklearn.linear_model import GammaRegressor, PoissonRegressor

import nemos as nmo


@pytest.mark.parametrize(
    "regularizer",
    [
        nmo.regularizer.UnRegularized(),
        nmo.regularizer.Ridge(),
        nmo.regularizer.Lasso(),
        nmo.regularizer.GroupLasso("ProximalGradient", np.array([[1.0]])),
    ],
)
def test_get_only_allowed_solvers(regularizer):
    # the error raised by property changed in python 3.11
    with pytest.raises(
        AttributeError,
        match="property 'allowed_solvers' of '.+' object has no setter|can't set attribute",
    ):
        regularizer.allowed_solvers = []


@pytest.mark.parametrize(
    "regularizer",
    [
        nmo.regularizer.UnRegularized(),
        nmo.regularizer.Ridge(),
        nmo.regularizer.Lasso(),
        nmo.regularizer.GroupLasso("ProximalGradient", np.array([[1.0]])),
    ],
)
def test_item_assignment_allowed_solvers(regularizer):
    with pytest.raises(
        TypeError, match="'tuple' object does not support item assignment"
    ):
        regularizer.allowed_solvers[0] = "my-favourite-solver"


class TestUnRegularized:
    cls = nmo.regularizer.UnRegularized

    @pytest.mark.parametrize(
        "solver_name",
        ["GradientDescent", "BFGS", "ProximalGradient", "AGradientDescent", 1],
    )
    def test_init_solver_name(self, solver_name):
        """Test UnRegularized acceptable solvers."""
        acceptable_solvers = [
            "GradientDescent",
            "BFGS",
            "LBFGS",
            "ScipyMinimize",
            "NonlinearCG",
            "ScipyBoundedMinimize",
            "LBFGSB",
        ]
        raise_exception = solver_name not in acceptable_solvers
        if raise_exception:
            with pytest.raises(
                ValueError, match=f"Solver `{solver_name}` not allowed for "
            ):
                self.cls(solver_name)
        else:
            self.cls(solver_name)

    @pytest.mark.parametrize(
        "solver_name",
        ["GradientDescent", "BFGS", "ProximalGradient", "AGradientDescent", 1],
    )
    def test_set_solver_name_allowed(self, solver_name):
        """Test UnRegularized acceptable solvers."""
        acceptable_solvers = [
            "GradientDescent",
            "BFGS",
            "LBFGS",
            "ScipyMinimize",
            "NonlinearCG",
            "ScipyBoundedMinimize",
            "LBFGSB",
        ]
        regularizer = self.cls("GradientDescent")
        raise_exception = solver_name not in acceptable_solvers
        if raise_exception:
            with pytest.raises(
                ValueError, match=f"Solver `{solver_name}` not allowed for "
            ):
                regularizer.set_params(solver_name=solver_name)
        else:
            regularizer.set_params(solver_name=solver_name)

    @pytest.mark.parametrize("solver_name", ["GradientDescent", "BFGS"])
    @pytest.mark.parametrize("solver_kwargs", [{"tol": 10**-10}, {"tols": 10**-10}])
    def test_init_solver_kwargs(self, solver_name, solver_kwargs):
        """Test RidgeSolver acceptable kwargs."""

        raise_exception = "tols" in list(solver_kwargs.keys())
        if raise_exception:
            with pytest.raises(
                NameError, match="kwargs {'tols'} in solver_kwargs not a kwarg"
            ):
                self.cls(solver_name, solver_kwargs=solver_kwargs)
        else:
            self.cls(solver_name, solver_kwargs=solver_kwargs)

    @pytest.mark.parametrize("loss", [lambda a, b, c: 0, 1, None, {}])
    def test_loss_is_callable(self, loss):
        """Test that the loss function is a callable"""
        raise_exception = not callable(loss)
        if raise_exception:
            with pytest.raises(TypeError, match="The `loss` must be a Callable"):
                self.cls("GradientDescent").instantiate_solver(loss)
        else:
            self.cls("GradientDescent").instantiate_solver(loss)

    @pytest.mark.parametrize("solver_name", ["GradientDescent", "BFGS"])
    def test_run_solver(self, solver_name, poissonGLM_model_instantiation):
        """Test that the solver runs."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        runner = self.cls("GradientDescent").instantiate_solver(
            model._predict_and_compute_loss
        )
        runner((true_params[0] * 0.0, true_params[1]), X, y)

    def test_solver_output_match(self, poissonGLM_model_instantiation):
        """Test that different solvers converge to the same solution."""
        jax.config.update("jax_enable_x64", True)
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        # set precision to float64 for accurate matching of the results
        model.data_type = jnp.float64
        runner_gd = self.cls("GradientDescent", {"tol": 10**-12}).instantiate_solver(
            model._predict_and_compute_loss
        )
        runner_bfgs = self.cls("BFGS", {"tol": 10**-12}).instantiate_solver(
            model._predict_and_compute_loss
        )
        runner_scipy = self.cls(
            "ScipyMinimize", {"method": "BFGS", "tol": 10**-12}
        ).instantiate_solver(model._predict_and_compute_loss)
        weights_gd, intercepts_gd = runner_gd(
            (true_params[0] * 0.0, true_params[1]), X, y
        )[0]
        weights_bfgs, intercepts_bfgs = runner_bfgs(
            (true_params[0] * 0.0, true_params[1]), X, y
        )[0]
        weights_scipy, intercepts_scipy = runner_scipy(
            (true_params[0] * 0.0, true_params[1]), X, y
        )[0]

        match_weights = np.allclose(weights_gd, weights_bfgs) and np.allclose(
            weights_gd, weights_scipy
        )
        match_intercepts = np.allclose(intercepts_gd, intercepts_bfgs) and np.allclose(
            intercepts_gd, intercepts_scipy
        )
        if (not match_weights) or (not match_intercepts):
            raise ValueError(
                "Convex estimators should converge to the same numerical value."
            )

    def test_solver_match_sklearn(self, poissonGLM_model_instantiation):
        """Test that different solvers converge to the same solution."""
        jax.config.update("jax_enable_x64", True)
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        # set precision to float64 for accurate matching of the results
        model.data_type = jnp.float64
        regularizer = self.cls("GradientDescent", {"tol": 10**-12})
        runner_bfgs = regularizer.instantiate_solver(model._predict_and_compute_loss)
        weights_bfgs, intercepts_bfgs = runner_bfgs(
            (true_params[0] * 0.0, true_params[1]), X, y
        )[0]
        model_skl = PoissonRegressor(fit_intercept=True, tol=10**-12, alpha=0.0)
        model_skl.fit(X, y)

        match_weights = np.allclose(model_skl.coef_, weights_bfgs)
        match_intercepts = np.allclose(model_skl.intercept_, intercepts_bfgs)
        if (not match_weights) or (not match_intercepts):
            raise ValueError("Ridge GLM regularizer estimate does not match sklearn!")

    def test_solver_match_sklearn_gamma(self, gammaGLM_model_instantiation):
        """Test that different solvers converge to the same solution."""
        jax.config.update("jax_enable_x64", True)
        X, y, model, true_params, firing_rate = gammaGLM_model_instantiation
        # set precision to float64 for accurate matching of the results
        model.data_type = jnp.float64
        model.observation_model.inverse_link_function = jnp.exp
        regularizer = self.cls("GradientDescent", {"tol": 10**-12})
        runner_bfgs = regularizer.instantiate_solver(model._predict_and_compute_loss)
        weights_bfgs, intercepts_bfgs = runner_bfgs(
            (true_params[0] * 0.0, true_params[1]), X, y
        )[0]
        model_skl = GammaRegressor(fit_intercept=True, tol=10**-12, alpha=0.0)
        model_skl.fit(X, y)

        match_weights = np.allclose(model_skl.coef_, weights_bfgs)
        match_intercepts = np.allclose(model_skl.intercept_, intercepts_bfgs)
        if (not match_weights) or (not match_intercepts):
            raise ValueError("Unregularized GLM estimate does not match sklearn!")

    @pytest.mark.parametrize("inv_link_jax, link_sm",
                             [
                                 (jnp.exp, sm.families.links.Log()),
                                 (lambda x: 1/x, sm.families.links.InversePower())
                             ]
                             )
    def test_solver_match_statsmodels_gamma(self, inv_link_jax, link_sm, gammaGLM_model_instantiation):
        """Test that different solvers converge to the same solution."""
        jax.config.update("jax_enable_x64", True)
        X, y, model, true_params, firing_rate = gammaGLM_model_instantiation
        # set precision to float64 for accurate matching of the results
        model.data_type = jnp.float64
        model.observation_model.inverse_link_function = inv_link_jax
        regularizer = self.cls("LBFGS", {"tol": 10**-13})
        runner_bfgs = regularizer.instantiate_solver(model._predict_and_compute_loss)
        weights_bfgs, intercepts_bfgs = runner_bfgs(
            model._initialize_parameters(X, y), X, y
        )[0]

        model_sm = sm.GLM(endog=y, exog=sm.add_constant(X), family=sm.families.Gamma(link=link_sm))

        res_sm = model_sm.fit(cnvrg_tol=10**-12)

        match_weights = np.allclose(res_sm.params[1:], weights_bfgs)
        match_intercepts = np.allclose(res_sm.params[:1], intercepts_bfgs)
        if (not match_weights) or (not match_intercepts):
            raise ValueError("Unregularized GLM estimate does not match statsmodels!")

    @pytest.mark.parametrize("solver_name", ["GradientDescent", "BFGS"])
    @pytest.mark.parametrize(
        "kwargs, expectation",
        [
            ({}, does_not_raise()),
            (
                {"prox": 0},
                pytest.raises(
                    ValueError,
                    match=r"Regularizer of type [A-z]+ does not require a "
                    r"proximal operator!",
                ),
            ),
        ],
    )
    def test_overwritten_proximal_operator(
        self, solver_name, kwargs, expectation, poissonGLM_model_instantiation
    ):
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        with expectation:
            model.regularizer.solver_kwargs = kwargs
            model.fit(X, y)


class TestRidge:
    cls = nmo.regularizer.Ridge

    @pytest.mark.parametrize(
        "solver_name",
        ["GradientDescent", "BFGS", "ProximalGradient", "AGradientDescent", 1],
    )
    def test_init_solver_name(self, solver_name):
        """Test RidgeSolver acceptable solvers."""
        acceptable_solvers = [
            "GradientDescent",
            "BFGS",
            "LBFGS",
            "ScipyMinimize",
            "NonlinearCG",
            "ScipyBoundedMinimize",
            "LBFGSB",
        ]
        raise_exception = solver_name not in acceptable_solvers
        if raise_exception:
            with pytest.raises(
                ValueError, match=f"Solver `{solver_name}` not allowed for "
            ):
                self.cls(solver_name)
        else:
            self.cls(solver_name)

    @pytest.mark.parametrize(
        "solver_name",
        ["GradientDescent", "BFGS", "ProximalGradient", "AGradientDescent", 1],
    )
    def test_set_solver_name_allowed(self, solver_name):
        """Test RidgeSolver acceptable solvers."""
        acceptable_solvers = [
            "GradientDescent",
            "BFGS",
            "LBFGS",
            "ScipyMinimize",
            "NonlinearCG",
            "ScipyBoundedMinimize",
            "LBFGSB",
        ]
        regularizer = self.cls("GradientDescent")
        raise_exception = solver_name not in acceptable_solvers
        if raise_exception:
            with pytest.raises(
                ValueError, match=f"Solver `{solver_name}` not allowed for "
            ):
                regularizer.set_params(solver_name=solver_name)
        else:
            regularizer.set_params(solver_name=solver_name)

    @pytest.mark.parametrize("solver_name", ["GradientDescent", "BFGS"])
    @pytest.mark.parametrize("solver_kwargs", [{"tol": 10**-10}, {"tols": 10**-10}])
    def test_init_solver_kwargs(self, solver_name, solver_kwargs):
        """Test Ridge acceptable kwargs."""

        raise_exception = "tols" in list(solver_kwargs.keys())
        if raise_exception:
            with pytest.raises(
                NameError, match="kwargs {'tols'} in solver_kwargs not a kwarg"
            ):
                self.cls(solver_name, solver_kwargs=solver_kwargs)
        else:
            self.cls(solver_name, solver_kwargs=solver_kwargs)

    @pytest.mark.parametrize("loss", [lambda a, b, c: 0, 1, None, {}])
    def test_loss_is_callable(self, loss):
        """Test that the loss function is a callable"""
        raise_exception = not callable(loss)
        if raise_exception:
            with pytest.raises(TypeError, match="The `loss` must be a Callable"):
                self.cls("GradientDescent").instantiate_solver(loss)
        else:
            self.cls("GradientDescent").instantiate_solver(loss)

    @pytest.mark.parametrize("solver_name", ["GradientDescent", "BFGS"])
    def test_run_solver(self, solver_name, poissonGLM_model_instantiation):
        """Test that the solver runs."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        runner = self.cls("GradientDescent").instantiate_solver(
            model._predict_and_compute_loss
        )
        runner((true_params[0] * 0.0, true_params[1]), X, y)

    def test_solver_output_match(self, poissonGLM_model_instantiation):
        """Test that different solvers converge to the same solution."""
        jax.config.update("jax_enable_x64", True)
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        # set precision to float64 for accurate matching of the results
        model.data_type = jnp.float64
        runner_gd = self.cls("GradientDescent", {"tol": 10**-12}).instantiate_solver(
            model._predict_and_compute_loss
        )
        runner_bfgs = self.cls("BFGS", {"tol": 10**-12}).instantiate_solver(
            model._predict_and_compute_loss
        )
        runner_scipy = self.cls(
            "ScipyMinimize", {"method": "BFGS", "tol": 10**-12}
        ).instantiate_solver(model._predict_and_compute_loss)
        weights_gd, intercepts_gd = runner_gd(
            (true_params[0] * 0.0, true_params[1]), X, y
        )[0]
        weights_bfgs, intercepts_bfgs = runner_bfgs(
            (true_params[0] * 0.0, true_params[1]), X, y
        )[0]
        weights_scipy, intercepts_scipy = runner_scipy(
            (true_params[0] * 0.0, true_params[1]), X, y
        )[0]

        match_weights = np.allclose(weights_gd, weights_bfgs) and np.allclose(
            weights_gd, weights_scipy
        )
        match_intercepts = np.allclose(intercepts_gd, intercepts_bfgs) and np.allclose(
            intercepts_gd, intercepts_scipy
        )
        if (not match_weights) or (not match_intercepts):
            raise ValueError(
                "Convex estimators should converge to the same numerical value."
            )

    def test_solver_match_sklearn(self, poissonGLM_model_instantiation):
        """Test that different solvers converge to the same solution."""
        jax.config.update("jax_enable_x64", True)
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        # set precision to float64 for accurate matching of the results
        model.data_type = jnp.float64
        regularizer = self.cls("GradientDescent", {"tol": 10**-12})
        runner_bfgs = regularizer.instantiate_solver(model._predict_and_compute_loss)
        weights_bfgs, intercepts_bfgs = runner_bfgs(
            (true_params[0] * 0.0, true_params[1]), X, y
        )[0]
        model_skl = PoissonRegressor(
            fit_intercept=True, tol=10**-12, alpha=regularizer.regularizer_strength
        )
        model_skl.fit(X, y)

        match_weights = np.allclose(model_skl.coef_, weights_bfgs)
        match_intercepts = np.allclose(model_skl.intercept_, intercepts_bfgs)
        if (not match_weights) or (not match_intercepts):
            raise ValueError("Ridge GLM solver estimate does not match sklearn!")

    def test_solver_match_sklearn_gamma(self, gammaGLM_model_instantiation):
        """Test that different solvers converge to the same solution."""
        jax.config.update("jax_enable_x64", True)
        X, y, model, true_params, firing_rate = gammaGLM_model_instantiation
        # set precision to float64 for accurate matching of the results
        model.data_type = jnp.float64
        model.observation_model.inverse_link_function = jnp.exp
        regularizer = self.cls("GradientDescent", {"tol": 10 ** -12})
        regularizer.regularizer_strength = 0.1
        runner_bfgs = regularizer.instantiate_solver(model._predict_and_compute_loss)
        weights_bfgs, intercepts_bfgs = runner_bfgs(
            (true_params[0] * 0.0, true_params[1]), X, y
        )[0]
        model_skl = GammaRegressor(fit_intercept=True, tol=10 ** -12, alpha=regularizer.regularizer_strength)
        model_skl.fit(X, y)

        match_weights = np.allclose(model_skl.coef_, weights_bfgs)
        match_intercepts = np.allclose(model_skl.intercept_, intercepts_bfgs)
        if (not match_weights) or (not match_intercepts):
            raise ValueError("Ridge GLM estimate does not match sklearn!")

    @pytest.mark.parametrize("solver_name", ["GradientDescent", "BFGS"])
    @pytest.mark.parametrize(
        "kwargs, expectation",
        [
            ({}, does_not_raise()),
            (
                {"prox": 0},
                pytest.raises(
                    ValueError,
                    match=r"Regularizer of type [A-z]+ does not require a "
                    r"proximal operator!",
                ),
            ),
        ],
    )
    def test_overwritten_proximal_operator(
        self, solver_name, kwargs, expectation, poissonGLM_model_instantiation
    ):
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        with expectation:
            model.regularizer.solver_kwargs = kwargs
            model.fit(X, y)


class TestLasso:
    cls = nmo.regularizer.Lasso

    @pytest.mark.parametrize(
        "solver_name",
        ["GradientDescent", "BFGS", "ProximalGradient", "AGradientDescent", 1],
    )
    def test_init_solver_name(self, solver_name):
        """Test Lasso acceptable solvers."""
        acceptable_solvers = ["ProximalGradient"]
        raise_exception = solver_name not in acceptable_solvers
        if raise_exception:
            with pytest.raises(
                ValueError, match=f"Solver `{solver_name}` not allowed for "
            ):
                self.cls(solver_name)
        else:
            self.cls(solver_name)

    @pytest.mark.parametrize(
        "solver_name",
        ["GradientDescent", "BFGS", "ProximalGradient", "AGradientDescent", 1],
    )
    def test_set_solver_name_allowed(self, solver_name):
        """Test Lasso acceptable solvers."""
        acceptable_solvers = ["ProximalGradient"]
        regularizer = self.cls("ProximalGradient")
        raise_exception = solver_name not in acceptable_solvers
        if raise_exception:
            with pytest.raises(
                ValueError, match=f"Solver `{solver_name}` not allowed for "
            ):
                regularizer.set_params(solver_name=solver_name)
        else:
            regularizer.set_params(solver_name=solver_name)

    @pytest.mark.parametrize("solver_kwargs", [{"tol": 10**-10}, {"tols": 10**-10}])
    def test_init_solver_kwargs(self, solver_kwargs):
        """Test LassoSolver acceptable kwargs."""
        raise_exception = "tols" in list(solver_kwargs.keys())
        if raise_exception:
            with pytest.raises(
                NameError, match="kwargs {'tols'} in solver_kwargs not a kwarg"
            ):
                self.cls("ProximalGradient", solver_kwargs=solver_kwargs)
        else:
            self.cls("ProximalGradient", solver_kwargs=solver_kwargs)

    @pytest.mark.parametrize("loss", [lambda a, b, c: 0, 1, None, {}])
    def test_loss_callable(self, loss):
        """Test that the loss function is a callable"""
        raise_exception = not callable(loss)
        if raise_exception:
            with pytest.raises(TypeError, match="The `loss` must be a Callable"):
                self.cls("ProximalGradient").instantiate_solver(loss)
        else:
            self.cls("ProximalGradient").instantiate_solver(loss)

    def test_run_solver(self, poissonGLM_model_instantiation):
        """Test that the solver runs."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        runner = self.cls("ProximalGradient").instantiate_solver(
            model._predict_and_compute_loss
        )
        runner((true_params[0] * 0.0, true_params[1]), X, y)

    def test_solver_match_statsmodels(self, poissonGLM_model_instantiation):
        """Test that different solvers converge to the same solution."""
        jax.config.update("jax_enable_x64", True)
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        # set precision to float64 for accurate matching of the results
        model.data_type = jnp.float64
        regularizer = self.cls("ProximalGradient", {"tol": 10**-12})
        runner = regularizer.instantiate_solver(model._predict_and_compute_loss)
        weights, intercepts = runner((true_params[0] * 0.0, true_params[1]), X, y)[0]

        # instantiate the glm with statsmodels
        glm_sm = sm.GLM(endog=y, exog=sm.add_constant(X), family=sm.families.Poisson())

        # regularize everything except intercept
        alpha_sm = np.ones(X.shape[1] + 1) * regularizer.regularizer_strength
        alpha_sm[0] = 0

        # pure lasso = elastic net with L1 weight = 1
        res_sm = glm_sm.fit_regularized(
            method="elastic_net", alpha=alpha_sm, L1_wt=1.0, cnvrg_tol=10**-12
        )
        # compare params
        sm_params = res_sm.params
        glm_params = jnp.hstack((intercepts, weights.flatten()))
        match_weights = np.allclose(sm_params, glm_params)
        if not match_weights:
            raise ValueError("Lasso GLM solver estimate does not match statsmodels!")

    @pytest.mark.parametrize("solver_name", ["ProximalGradient"])
    @pytest.mark.parametrize(
        "kwargs, expectation",
        [
            ({}, does_not_raise()),
            (
                {"prox": 0},
                pytest.warns(
                    UserWarning, match=r"Overwritten the user-defined proximal operator"
                ),
            ),
        ],
    )
    def test_overwritten_proximal_operator(
        self, solver_name, kwargs, expectation, poissonGLM_model_instantiation
    ):
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.regularizer = nmo.regularizer.Lasso()
        with expectation:
            model.regularizer.solver_kwargs = kwargs
            model.fit(X, y)

    def test_lasso_pytree(self, poissonGLM_model_instantiation_pytree):
        """Check pytree X can be fit."""
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation_pytree
        model.regularizer = nmo.regularizer.Lasso()
        model.fit(X, y)

    @pytest.mark.parametrize("reg_str", [0.001, 0.01, 0.1, 1, 10])
    def test_lasso_pytree_match(
        self,
        reg_str,
        poissonGLM_model_instantiation_pytree,
        poissonGLM_model_instantiation,
    ):
        """Check pytree and array find same solution."""
        jax.config.update("jax_enable_x64", True)
        X, _, model, _, _ = poissonGLM_model_instantiation_pytree
        X_array, y, model_array, _, _ = poissonGLM_model_instantiation

        model.regularizer = nmo.regularizer.Lasso(regularizer_strength=reg_str)
        model_array.regularizer = nmo.regularizer.Lasso(regularizer_strength=reg_str)
        model.fit(X, y)
        model_array.fit(X_array, y)
        assert np.allclose(
            np.hstack(jax.tree_util.tree_leaves(model.coef_)), model_array.coef_
        )


class TestGroupLasso:
    cls = nmo.regularizer.GroupLasso

    @pytest.mark.parametrize(
        "solver_name",
        ["GradientDescent", "BFGS", "ProximalGradient", "AGradientDescent", 1],
    )
    def test_init_solver_name(self, solver_name):
        """Test GroupLasso acceptable solvers."""
        acceptable_solvers = ["ProximalGradient"]
        raise_exception = solver_name not in acceptable_solvers

        # create a valid mask
        mask = np.zeros((2, 10))
        mask[0, :5] = 1
        mask[1, 5:] = 1
        mask = jnp.asarray(mask)

        if raise_exception:
            with pytest.raises(
                ValueError, match=f"Solver `{solver_name}` not allowed for "
            ):
                self.cls(solver_name, mask)
        else:
            self.cls(solver_name, mask)

    @pytest.mark.parametrize(
        "solver_name",
        ["GradientDescent", "BFGS", "ProximalGradient", "AGradientDescent", 1],
    )
    def test_set_solver_name_allowed(self, solver_name):
        """Test GroupLassoSolver acceptable solvers."""
        acceptable_solvers = ["ProximalGradient"]
        # create a valid mask
        mask = np.zeros((2, 10))
        mask[0, :5] = 1
        mask[1, 5:] = 1
        mask = jnp.asarray(mask)
        regularizer = self.cls("ProximalGradient", mask=mask)
        raise_exception = solver_name not in acceptable_solvers
        if raise_exception:
            with pytest.raises(
                ValueError, match=f"Solver `{solver_name}` not allowed for "
            ):
                regularizer.set_params(solver_name=solver_name)
        else:
            regularizer.set_params(solver_name=solver_name)

    @pytest.mark.parametrize("solver_kwargs", [{"tol": 10**-10}, {"tols": 10**-10}])
    def test_init_solver_kwargs(self, solver_kwargs):
        """Test GroupLasso acceptable kwargs."""
        raise_exception = "tols" in list(solver_kwargs.keys())

        # create a valid mask
        mask = np.zeros((2, 10))
        mask[0, :5] = 1
        mask[0, 1:] = 1
        mask = jnp.asarray(mask)

        if raise_exception:
            with pytest.raises(
                NameError, match="kwargs {'tols'} in solver_kwargs not a kwarg"
            ):
                self.cls("ProximalGradient", mask, solver_kwargs=solver_kwargs)
        else:
            self.cls("ProximalGradient", mask, solver_kwargs=solver_kwargs)

    @pytest.mark.parametrize("loss", [lambda a, b, c: 0, 1, None, {}])
    def test_loss_callable(self, loss):
        """Test that the loss function is a callable"""
        raise_exception = not callable(loss)

        # create a valid mask
        mask = np.zeros((2, 10))
        mask[0, :5] = 1
        mask[1, 5:] = 1
        mask = jnp.asarray(mask)

        if raise_exception:
            with pytest.raises(TypeError, match="The `loss` must be a Callable"):
                self.cls("ProximalGradient", mask).instantiate_solver(loss)
        else:
            self.cls("ProximalGradient", mask).instantiate_solver(loss)

    def test_run_solver(self, poissonGLM_model_instantiation):
        """Test that the solver runs."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        # create a valid mask
        mask = np.zeros((2, X.shape[1]))
        mask[0, :2] = 1
        mask[1, 2:] = 1
        mask = jnp.asarray(mask)

        runner = self.cls("ProximalGradient", mask).instantiate_solver(
            model._predict_and_compute_loss
        )
        runner((true_params[0] * 0.0, true_params[1]), X, y)

    @pytest.mark.parametrize("n_groups_assign", [0, 1, 2])
    def test_mask_validity_groups(
        self, n_groups_assign, group_sparse_poisson_glm_model_instantiation
    ):
        """Test that mask assigns at most 1 group to each weight."""
        raise_exception = n_groups_assign > 1
        (
            X,
            y,
            model,
            true_params,
            firing_rate,
            _,
        ) = group_sparse_poisson_glm_model_instantiation

        # create a valid mask
        mask = np.zeros((2, X.shape[1]))
        mask[0, :2] = 1
        mask[1, 2:] = 1

        # change assignment
        if n_groups_assign == 0:
            mask[:, 3] = 0
        elif n_groups_assign == 2:
            mask[:, 3] = 1

        mask = jnp.asarray(mask)

        if raise_exception:
            with pytest.raises(
                ValueError, match="Incorrect group assignment. " "Some of the features"
            ):
                self.cls("ProximalGradient", mask).instantiate_solver(
                    model._predict_and_compute_loss
                )
        else:
            self.cls("ProximalGradient", mask).instantiate_solver(
                model._predict_and_compute_loss
            )

    @pytest.mark.parametrize("set_entry", [0, 1, -1, 2, 2.5])
    def test_mask_validity_entries(self, set_entry, poissonGLM_model_instantiation):
        """Test that mask is composed of 0s and 1s."""
        raise_exception = set_entry not in {0, 1}
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        # create a valid mask
        mask = np.zeros((2, X.shape[1]))
        mask[0, :2] = 1
        mask[1, 2:] = 1
        # assign an entry
        mask[1, 2] = set_entry
        mask = jnp.asarray(mask, dtype=jnp.float32)

        if raise_exception:
            with pytest.raises(ValueError, match="Mask elements be 0s and 1s"):
                self.cls("ProximalGradient", mask).instantiate_solver(
                    model._predict_and_compute_loss
                )
        else:
            self.cls("ProximalGradient", mask).instantiate_solver(
                model._predict_and_compute_loss
            )

    @pytest.mark.parametrize("n_dim", [0, 1, 2, 3])
    def test_mask_dimension_1(self, n_dim, poissonGLM_model_instantiation):
        """Test that mask is composed of 0s and 1s."""

        raise_exception = n_dim != 2
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        # create a valid mask
        if n_dim == 0:
            mask = np.array([])
        elif n_dim == 1:
            mask = np.ones((1,))
        elif n_dim == 2:
            mask = np.zeros((2, X.shape[1]))
            mask[0, :2] = 1
            mask[1, 2:] = 1
        else:
            mask = np.zeros((2, X.shape[1]) + (1,) * (n_dim - 2))
            mask[0, :2] = 1
            mask[1, 2:] = 1

        mask = jnp.asarray(mask, dtype=jnp.float32)

        if raise_exception:
            with pytest.raises(ValueError, match="`mask` must be 2-dimensional"):
                self.cls("ProximalGradient", mask).instantiate_solver(
                    model._predict_and_compute_loss
                )
        else:
            self.cls("ProximalGradient", mask).instantiate_solver(
                model._predict_and_compute_loss
            )

    @pytest.mark.parametrize("n_groups", [0, 1, 2])
    def test_mask_n_groups(self, n_groups, poissonGLM_model_instantiation):
        """Test that mask has at least 1 group."""
        raise_exception = n_groups < 1
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        # create a mask
        mask = np.zeros((n_groups, X.shape[1]))
        if n_groups > 0:
            for i in range(n_groups - 1):
                mask[i, i : i + 1] = 1
            mask[-1, n_groups - 1 :] = 1

        mask = jnp.asarray(mask, dtype=jnp.float32)

        if raise_exception:
            with pytest.raises(ValueError, match=r"Empty mask provided! Mask has "):
                self.cls("ProximalGradient", mask).instantiate_solver(
                    model._predict_and_compute_loss
                )
        else:
            self.cls("ProximalGradient", mask).instantiate_solver(
                model._predict_and_compute_loss
            )

    def test_group_sparsity_enforcement(
        self, group_sparse_poisson_glm_model_instantiation
    ):
        """Test that group lasso works on a simple dataset."""
        (
            X,
            y,
            model,
            true_params,
            firing_rate,
            _,
        ) = group_sparse_poisson_glm_model_instantiation
        zeros_true = true_params[0].flatten() == 0
        mask = np.zeros((2, X.shape[1]))
        mask[0, zeros_true] = 1
        mask[1, ~zeros_true] = 1
        mask = jnp.asarray(mask, dtype=jnp.float32)

        runner = self.cls("ProximalGradient", mask).instantiate_solver(
            model._predict_and_compute_loss
        )
        params, _ = runner((true_params[0] * 0.0, true_params[1]), X, y)

        zeros_est = params[0] == 0
        if not np.all(zeros_est == zeros_true):
            raise ValueError("GroupLasso failed to zero-out the parameter group!")

    ###########
    # Test mask from set_params
    ###########
    @pytest.mark.parametrize("n_groups_assign", [0, 1, 2])
    def test_mask_validity_groups_set_params(
        self, n_groups_assign, group_sparse_poisson_glm_model_instantiation
    ):
        """Test that mask assigns at most 1 group to each weight."""
        raise_exception = n_groups_assign > 1
        (
            X,
            y,
            model,
            true_params,
            firing_rate,
            _,
        ) = group_sparse_poisson_glm_model_instantiation

        # create a valid mask
        mask = np.zeros((2, X.shape[1]))
        mask[0, :2] = 1
        mask[1, 2:] = 1
        regularizer = self.cls("ProximalGradient", mask)

        # change assignment
        if n_groups_assign == 0:
            mask[:, 3] = 0
        elif n_groups_assign == 2:
            mask[:, 3] = 1

        mask = jnp.asarray(mask)

        if raise_exception:
            with pytest.raises(
                ValueError, match="Incorrect group assignment. " "Some of the features"
            ):
                regularizer.set_params(mask=mask)
        else:
            regularizer.set_params(mask=mask)

    @pytest.mark.parametrize("set_entry", [0, 1, -1, 2, 2.5])
    def test_mask_validity_entries_set_params(
        self, set_entry, poissonGLM_model_instantiation
    ):
        """Test that mask is composed of 0s and 1s."""
        raise_exception = set_entry not in {0, 1}
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        # create a valid mask
        mask = np.zeros((2, X.shape[1]))
        mask[0, :2] = 1
        mask[1, 2:] = 1
        regularizer = self.cls("ProximalGradient", mask)

        # assign an entry
        mask[1, 2] = set_entry
        mask = jnp.asarray(mask, dtype=jnp.float32)

        if raise_exception:
            with pytest.raises(ValueError, match="Mask elements be 0s and 1s"):
                regularizer.set_params(mask=mask)
        else:
            regularizer.set_params(mask=mask)

    @pytest.mark.parametrize("n_dim", [0, 1, 2, 3])
    def test_mask_dimension(self, n_dim, poissonGLM_model_instantiation):
        """Test that mask is composed of 0s and 1s."""

        raise_exception = n_dim != 2
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        valid_mask = np.zeros((2, X.shape[1]))
        valid_mask[0, :1] = 1
        valid_mask[1, 1:] = 1
        regularizer = self.cls("ProximalGradient", valid_mask)

        # create a mask
        if n_dim == 0:
            mask = np.array([])
        elif n_dim == 1:
            mask = np.ones((1,))
        elif n_dim == 2:
            mask = np.zeros((2, X.shape[1]))
            mask[0, :2] = 1
            mask[1, 2:] = 1
        else:
            mask = np.zeros((2, X.shape[1]) + (1,) * (n_dim - 2))
            mask[0, :2] = 1
            mask[1, 2:] = 1

        mask = jnp.asarray(mask, dtype=jnp.float32)

        if raise_exception:
            with pytest.raises(ValueError, match="`mask` must be 2-dimensional"):
                regularizer.set_params(mask=mask)
        else:
            regularizer.set_params(mask=mask)

    @pytest.mark.parametrize("n_groups", [0, 1, 2])
    def test_mask_n_groups_set_params(self, n_groups, poissonGLM_model_instantiation):
        """Test that mask has at least 1 group."""
        raise_exception = n_groups < 1
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        valid_mask = np.zeros((2, X.shape[1]))
        valid_mask[0, :1] = 1
        valid_mask[1, 1:] = 1
        regularizer = self.cls("ProximalGradient", valid_mask)

        # create a mask
        mask = np.zeros((n_groups, X.shape[1]))
        if n_groups > 0:
            for i in range(n_groups - 1):
                mask[i, i : i + 1] = 1
            mask[-1, n_groups - 1 :] = 1

        mask = jnp.asarray(mask, dtype=jnp.float32)

        if raise_exception:
            with pytest.raises(ValueError, match=r"Empty mask provided! Mask has "):
                regularizer.set_params(mask=mask)
        else:
            regularizer.set_params(mask=mask)

    @pytest.mark.parametrize("solver_name", ["ProximalGradient"])
    @pytest.mark.parametrize(
        "kwargs, expectation",
        [
            ({}, does_not_raise()),
            (
                {"prox": 0},
                pytest.warns(
                    UserWarning, match=r"Overwritten the user-defined proximal operator"
                ),
            ),
        ],
    )
    def test_overwritten_proximal_operator(
        self, solver_name, kwargs, expectation, poissonGLM_model_instantiation
    ):
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.regularizer = nmo.regularizer.Lasso()
        with expectation:
            model.regularizer.solver_kwargs = kwargs
            model.fit(X, y)
