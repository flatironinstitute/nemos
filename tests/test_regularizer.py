import copy
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import statsmodels.api as sm
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from statsmodels.tools.sm_exceptions import DomainWarning

import nemos as nmo


@pytest.mark.parametrize(
    "reg_str, reg_type",
    [
        ("UnRegularized", nmo.regularizer.UnRegularized),
        ("Ridge", nmo.regularizer.Ridge),
        ("Lasso", nmo.regularizer.Lasso),
        ("GroupLasso", nmo.regularizer.GroupLasso),
        ("not_valid", None),
    ],
)
def test_regularizer_builder(reg_str, reg_type):
    """Test building a regularizer from a string"""
    raise_exception = reg_str not in nmo._regularizer_builder.AVAILABLE_REGULARIZERS
    if raise_exception:
        with pytest.raises(ValueError, match=f"Unknown regularizer: {reg_str}. "):
            nmo._regularizer_builder.create_regularizer(reg_str)
    else:
        # build a regularizer by string
        regularizer = nmo._regularizer_builder.create_regularizer(reg_str)
        # assert correct type of regularizer is instantiated
        assert isinstance(regularizer, reg_type)
        # create a regularizer of that type
        regularizer2 = reg_type()
        # assert that they have the same attributes
        assert regularizer.__dict__ == regularizer2.__dict__


@pytest.mark.parametrize(
    "expected, reg",
    [
        ("UnRegularized()", nmo.regularizer.UnRegularized()),
        ("Ridge()", nmo.regularizer.Ridge()),
        ("Lasso()", nmo.regularizer.Lasso()),
        ("GroupLasso()", nmo.regularizer.GroupLasso(mask=np.eye(4))),
    ],
)
def test_regularizer_repr(reg, expected):
    assert repr(reg) == expected


def test_regularizer_available():
    for regularizer in nmo._regularizer_builder.AVAILABLE_REGULARIZERS:
        reg = nmo._regularizer_builder.create_regularizer(regularizer)
        assert reg.__class__.__name__ == regularizer


@pytest.mark.parametrize(
    "regularizer_strength, reg_type",
    [
        (0.001, nmo.regularizer.Ridge),
        (1.0, nmo.regularizer.Ridge),
        ("bah", nmo.regularizer.Ridge),
        (0.001, nmo.regularizer.Lasso),
        (1.0, nmo.regularizer.Lasso),
        ("bah", nmo.regularizer.Lasso),
        (0.001, nmo.regularizer.GroupLasso),
        (1.0, nmo.regularizer.GroupLasso),
        ("bah", nmo.regularizer.GroupLasso),
    ],
)
def test_regularizer(regularizer_strength, reg_type):
    if not isinstance(regularizer_strength, float):
        with pytest.raises(
            ValueError,
            match=f"Could not convert the regularizer strength: {regularizer_strength} "
            f"to a float.",
        ):
            nmo.glm.GLM(
                regularizer=reg_type(), regularizer_strength=regularizer_strength
            )
    else:
        nmo.glm.GLM(regularizer=reg_type(), regularizer_strength=regularizer_strength)


@pytest.mark.parametrize(
    "regularizer_strength, regularizer",
    [
        (0.001, nmo.regularizer.Ridge()),
        (1.0, nmo.regularizer.Ridge()),
        ("bah", nmo.regularizer.Ridge()),
        (0.001, nmo.regularizer.Lasso()),
        (1.0, nmo.regularizer.Lasso()),
        ("bah", nmo.regularizer.Lasso()),
        (0.001, nmo.regularizer.GroupLasso()),
        (1.0, nmo.regularizer.GroupLasso()),
        ("bah", nmo.regularizer.GroupLasso()),
    ],
)
def test_regularizer_setter(regularizer_strength, regularizer):
    if not isinstance(regularizer_strength, float):
        with pytest.raises(
            ValueError,
            match=f"Could not convert the regularizer strength: {regularizer_strength} "
            f"to a float.",
        ):

            nmo.glm.GLM(
                regularizer=regularizer, regularizer_strength=regularizer_strength
            )
    else:
        nmo.glm.GLM(regularizer=regularizer, regularizer_strength=regularizer_strength)


@pytest.mark.parametrize(
    "regularizer",
    [
        nmo.regularizer.UnRegularized(),
        nmo.regularizer.Ridge(),
        nmo.regularizer.Lasso(),
        nmo.regularizer.GroupLasso(mask=np.array([[1.0]])),
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
        nmo.regularizer.GroupLasso(mask=np.array([[1.0]])),
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
        [
            "GradientDescent",
            "BFGS",
            "ProximalGradient",
            "AGradientDescent",
            1,
            "SVRG",
            "ProxSVRG",
        ],
    )
    def test_init_solver_name(self, solver_name):
        """Test UnRegularized acceptable solvers."""
        acceptable_solvers = [
            "GradientDescent",
            "BFGS",
            "LBFGSB",
            "NonlinearCG",
            "ProximalGradient",
            "SVRG",
            "ProxSVRG",
        ]

        raise_exception = solver_name not in acceptable_solvers
        if raise_exception:
            with pytest.raises(
                ValueError, match=f"The solver: {solver_name} is not allowed for "
            ):
                nmo.glm.GLM(regularizer=self.cls(), solver_name=solver_name)
        else:
            nmo.glm.GLM(regularizer=self.cls(), solver_name=solver_name)

    @pytest.mark.parametrize(
        "solver_name",
        [
            "GradientDescent",
            "BFGS",
            "ProximalGradient",
            "AGradientDescent",
            1,
            "SVRG",
            "ProxSVRG",
        ],
    )
    def test_set_solver_name_allowed(self, solver_name):
        """Test UnRegularized acceptable solvers."""
        acceptable_solvers = [
            "GradientDescent",
            "BFGS",
            "LBFGS",
            "NonlinearCG",
            "ProximalGradient",
            "SVRG",
            "ProxSVRG",
        ]
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer)
        raise_exception = solver_name not in acceptable_solvers
        if raise_exception:
            with pytest.raises(
                ValueError, match=f"The solver: {solver_name} is not allowed for "
            ):
                model.set_params(solver_name=solver_name)
        else:
            model.set_params(solver_name=solver_name)

    def test_regularizer_strength_none(self):
        """Add test to assert that regularizer strength of UnRegularized model should be `None`"""
        # unregularized should be None
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer)

        assert model.regularizer_strength is None

        # changing to ridge, lasso, or grouplasso should raise UserWarning and set to 1.0
        with pytest.warns(UserWarning):
            model.regularizer = "Lasso"

        assert model.regularizer_strength == 1.0

        with pytest.warns(UserWarning):
            model.regularizer = regularizer
        assert model.regularizer_strength == 1.0

    def test_get_params(self):
        """Test get_params() returns expected values."""
        regularizer = self.cls()

        assert regularizer.get_params() == {}

    @pytest.mark.parametrize(
        "solver_name", ["GradientDescent", "BFGS", "SVRG", "ProxSVRG"]
    )
    @pytest.mark.parametrize("solver_kwargs", [{"tol": 10**-10}, {"tols": 10**-10}])
    def test_init_solver_kwargs(self, solver_name, solver_kwargs):
        """Test RidgeSolver acceptable kwargs."""
        regularizer = self.cls()
        raise_exception = "tols" in list(solver_kwargs.keys())
        if raise_exception:
            with pytest.raises(
                NameError, match="kwargs {'tols'} in solver_kwargs not a kwarg"
            ):
                nmo.glm.GLM(
                    regularizer=regularizer,
                    solver_name=solver_name,
                    solver_kwargs=solver_kwargs,
                )
        else:
            nmo.glm.GLM(
                regularizer=regularizer,
                solver_name=solver_name,
                solver_kwargs=solver_kwargs,
            )

    @pytest.mark.parametrize("loss", [lambda a, b, c: 0, 1, None, {}])
    def test_loss_is_callable(self, loss):
        """Test Unregularized callable loss."""
        raise_exception = not callable(loss)
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer)
        model._predict_and_compute_loss = loss
        if raise_exception:
            with pytest.raises(TypeError, match="The `loss` must be a Callable"):
                nmo.utils.assert_is_callable(model._predict_and_compute_loss, "loss")
        else:
            nmo.utils.assert_is_callable(model._predict_and_compute_loss, "loss")

    @pytest.mark.parametrize(
        "solver_name",
        ["GradientDescent", "BFGS", "ProximalGradient", "SVRG", "ProxSVRG"],
    )
    def test_run_solver(self, solver_name, poissonGLM_model_instantiation):
        """Test that the solver runs."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        # set regularizer and solver name
        model.set_params(regularizer=self.cls())
        model.solver_name = solver_name
        model.instantiate_solver()
        model.solver_run((true_params[0] * 0.0, true_params[1]), X, y)

    @pytest.mark.parametrize(
        "solver_name",
        ["GradientDescent", "BFGS", "ProximalGradient", "SVRG", "ProxSVRG"],
    )
    def test_run_solver_tree(self, solver_name, poissonGLM_model_instantiation_pytree):
        """Test that the solver runs."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation_pytree

        # set regularizer and solver name
        model.set_params(regularizer=self.cls())
        model.solver_name = solver_name
        model.instantiate_solver()
        model.solver_run(
            (jax.tree_util.tree_map(jnp.zeros_like, true_params[0]), true_params[1]),
            X.data,
            y,
        )

    @pytest.mark.parametrize("solver_name", ["GradientDescent", "SVRG"])
    def test_solver_output_match(self, poissonGLM_model_instantiation, solver_name):
        """Test that different solvers converge to the same solution."""
        jax.config.update("jax_enable_x64", True)
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        # set precision to float64 for accurate matching of the results
        model.data_type = jnp.float64
        # set model params
        model.set_params(regularizer=self.cls())
        model.solver_name = solver_name
        model.solver_kwargs = {"tol": 10**-12}
        model.instantiate_solver()

        # update solver name
        model_bfgs = copy.deepcopy(model)
        model_bfgs.solver_name = "BFGS"
        model_bfgs.instantiate_solver()
        weights_gd, intercepts_gd = model.solver_run(
            (true_params[0] * 0.0, true_params[1]), X, y
        )[0]
        weights_bfgs, intercepts_bfgs = model_bfgs.solver_run(
            (true_params[0] * 0.0, true_params[1]), X, y
        )[0]

        match_weights = np.allclose(weights_gd, weights_bfgs)
        match_intercepts = np.allclose(intercepts_gd, intercepts_bfgs)

        if (not match_weights) or (not match_intercepts):
            raise ValueError(
                "Convex estimators should converge to the same numerical value."
            )

    @pytest.mark.parametrize("solver_name", ["GradientDescent", "SVRG"])
    def test_solver_match_sklearn(self, poissonGLM_model_instantiation, solver_name):
        """Test that different solvers converge to the same solution."""
        jax.config.update("jax_enable_x64", True)
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        # set precision to float64 for accurate matching of the results
        model.data_type = jnp.float64
        model.set_params(regularizer=self.cls())
        model.solver_name = solver_name
        model.solver_kwargs = {"tol": 10**-12}
        model.instantiate_solver()
        weights_bfgs, intercepts_bfgs = model.solver_run(
            (true_params[0] * 0.0, true_params[1]), X, y
        )[0]
        model_skl = PoissonRegressor(fit_intercept=True, tol=10**-12, alpha=0.0)
        model_skl.fit(X, y)

        match_weights = np.allclose(model_skl.coef_, weights_bfgs)
        match_intercepts = np.allclose(model_skl.intercept_, intercepts_bfgs)
        if (not match_weights) or (not match_intercepts):
            raise ValueError("UnRegularized GLM estimate does not match sklearn!")

    @pytest.mark.parametrize("solver_name", ["GradientDescent", "SVRG"])
    def test_solver_match_sklearn_gamma(
        self, gammaGLM_model_instantiation, solver_name
    ):
        """Test that different solvers converge to the same solution."""
        jax.config.update("jax_enable_x64", True)
        X, y, model, true_params, firing_rate = gammaGLM_model_instantiation
        # set precision to float64 for accurate matching of the results
        model.data_type = jnp.float64
        model.observation_model.inverse_link_function = jnp.exp
        model.set_params(regularizer=self.cls())
        model.solver_name = solver_name
        model.solver_kwargs = {"tol": 10**-12}
        model.instantiate_solver()
        weights_bfgs, intercepts_bfgs = model.solver_run(
            (true_params[0] * 0.0, true_params[1]), X, y
        )[0]
        model_skl = GammaRegressor(fit_intercept=True, tol=10**-12, alpha=0.0)
        model_skl.fit(X, y)

        match_weights = np.allclose(model_skl.coef_, weights_bfgs)
        match_intercepts = np.allclose(model_skl.intercept_, intercepts_bfgs)
        if (not match_weights) or (not match_intercepts):
            raise ValueError("Unregularized GLM estimate does not match sklearn!")

    @pytest.mark.parametrize(
        "inv_link_jax, link_sm",
        [
            (jnp.exp, sm.families.links.Log()),
            (lambda x: 1 / x, sm.families.links.InversePower()),
        ],
    )
    # @pytest.mark.parametrize("solver_name", ["LBFGS", "GradientDescent", "SVRG"])
    @pytest.mark.parametrize("solver_name", ["LBFGS", "SVRG"])
    def test_solver_match_statsmodels_gamma(
        self, inv_link_jax, link_sm, gammaGLM_model_instantiation, solver_name
    ):
        """Test that different solvers converge to the same solution."""
        jax.config.update("jax_enable_x64", True)
        X, y, model, true_params, firing_rate = gammaGLM_model_instantiation
        # set precision to float64 for accurate matching of the results
        model.data_type = jnp.float64
        model.observation_model.inverse_link_function = inv_link_jax
        model.set_params(regularizer=self.cls())
        model.solver_name = solver_name
        model.solver_kwargs = {"tol": 10**-13}
        model.instantiate_solver()
        weights_bfgs, intercepts_bfgs = model.solver_run(
            model._initialize_parameters(X, y), X, y
        )[0]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The InversePower link function does "
            )
            model_sm = sm.GLM(
                endog=y, exog=sm.add_constant(X), family=sm.families.Gamma(link=link_sm)
            )

        res_sm = model_sm.fit(cnvrg_tol=10**-12)

        match_weights = np.allclose(res_sm.params[1:], weights_bfgs)
        match_intercepts = np.allclose(res_sm.params[:1], intercepts_bfgs)
        if (not match_weights) or (not match_intercepts):
            raise ValueError("Unregularized GLM estimate does not match statsmodels!")

    @pytest.mark.parametrize(
        "solver_name",
        [
            "GradientDescent",
            "BFGS",
            "LBFGS",
            "NonlinearCG",
            "ProximalGradient",
            "SVRG",
            "ProxSVRG",
        ],
    )
    def test_solver_combination(self, solver_name, poissonGLM_model_instantiation):
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.set_params(regularizer=self.cls())
        model.solver_name = solver_name
        model.fit(X, y)


class TestRidge:
    cls = nmo.regularizer.Ridge

    @pytest.mark.parametrize(
        "solver_name",
        [
            "GradientDescent",
            "BFGS",
            "ProximalGradient",
            "AGradientDescent",
            1,
            "SVRG",
            "ProxSVRG",
        ],
    )
    def test_init_solver_name(self, solver_name):
        """Test RidgeSolver acceptable solvers."""
        acceptable_solvers = [
            "GradientDescent",
            "BFGS",
            "LBFGS",
            "NonlinearCG",
            "ProximalGradient",
            "SVRG",
            "ProxSVRG",
        ]
        raise_exception = solver_name not in acceptable_solvers
        if raise_exception:
            with pytest.raises(
                ValueError, match=f"The solver: {solver_name} is not allowed for "
            ):
                nmo.glm.GLM(
                    regularizer=self.cls(),
                    solver_name=solver_name,
                    regularizer_strength=1.0,
                )
        else:
            nmo.glm.GLM(
                regularizer=self.cls(),
                solver_name=solver_name,
                regularizer_strength=1.0,
            )

    @pytest.mark.parametrize(
        "solver_name",
        [
            "GradientDescent",
            "BFGS",
            "ProximalGradient",
            "AGradientDescent",
            1,
            "SVRG",
            "ProxSVRG",
        ],
    )
    def test_set_solver_name_allowed(self, solver_name):
        """Test RidgeSolver acceptable solvers."""
        acceptable_solvers = [
            "GradientDescent",
            "BFGS",
            "LBFGS",
            "LBFGSB",
            "NonlinearCG",
            "ProximalGradient",
            "SVRG",
            "ProxSVRG",
        ]
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer, regularizer_strength=1.0)
        raise_exception = solver_name not in acceptable_solvers
        if raise_exception:
            with pytest.raises(
                ValueError, match=f"The solver: {solver_name} is not allowed for "
            ):
                model.set_params(solver_name=solver_name)
        else:
            model.set_params(solver_name=solver_name)

    @pytest.mark.parametrize("solver_name", ["GradientDescent", "BFGS", "SVRG"])
    @pytest.mark.parametrize("solver_kwargs", [{"tol": 10**-10}, {"tols": 10**-10}])
    def test_init_solver_kwargs(self, solver_name, solver_kwargs):
        """Test Ridge acceptable kwargs."""
        regularizer = self.cls()
        raise_exception = "tols" in list(solver_kwargs.keys())
        if raise_exception:
            with pytest.raises(
                NameError, match="kwargs {'tols'} in solver_kwargs not a kwarg"
            ):
                nmo.glm.GLM(
                    regularizer=regularizer,
                    solver_name=solver_name,
                    solver_kwargs=solver_kwargs,
                    regularizer_strength=1.0,
                )
        else:
            nmo.glm.GLM(
                regularizer=regularizer,
                solver_name=solver_name,
                solver_kwargs=solver_kwargs,
                regularizer_strength=1.0,
            )

    def test_regularizer_strength_none(self):
        """Assert regularizer strength handled appropriately."""
        # if no strength given, should warn and set to 1.0
        with pytest.warns(UserWarning):
            regularizer = self.cls()
            model = nmo.glm.GLM(regularizer=regularizer)

        assert model.regularizer_strength == 1.0

        with pytest.warns(UserWarning):
            # if changed to regularized, is kept to 1.
            model.regularizer = "UnRegularized"
        assert model.regularizer_strength == 1.0

        # if changed back, should warn and set to 1.0
        model.regularizer = "Ridge"

        assert model.regularizer_strength == 1.0

    def test_get_params(self):
        """Test get_params() returns expected values."""
        regularizer = self.cls()

        assert regularizer.get_params() == {}

    @pytest.mark.parametrize("loss", [lambda a, b, c: 0, 1, None, {}])
    def test_loss_is_callable(self, loss):
        """Test Ridge callable loss."""
        raise_exception = not callable(loss)
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer, regularizer_strength=1.0)
        model._predict_and_compute_loss = loss
        if raise_exception:
            with pytest.raises(TypeError, match="The `loss` must be a Callable"):
                nmo.utils.assert_is_callable(model._predict_and_compute_loss, "loss")
        else:
            nmo.utils.assert_is_callable(model._predict_and_compute_loss, "loss")

    @pytest.mark.parametrize(
        "solver_name",
        ["GradientDescent", "BFGS", "ProximalGradient", "SVRG", "ProxSVRG"],
    )
    def test_run_solver(self, solver_name, poissonGLM_model_instantiation):
        """Test that the solver runs."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        # set regularizer and solver name
        model.set_params(regularizer=self.cls(), regularizer_strength=1.0)
        model.solver_name = solver_name
        runner = model.instantiate_solver().solver_run
        runner((true_params[0] * 0.0, true_params[1]), X, y)

    @pytest.mark.parametrize(
        "solver_name",
        ["GradientDescent", "BFGS", "ProximalGradient", "SVRG", "ProxSVRG"],
    )
    def test_run_solver_tree(self, solver_name, poissonGLM_model_instantiation_pytree):
        """Test that the solver runs."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation_pytree

        # set regularizer and solver name
        model.set_params(regularizer=self.cls(), regularizer_strength=1.0)
        model.solver_name = solver_name
        runner = model.instantiate_solver().solver_run
        runner(
            (jax.tree_util.tree_map(jnp.zeros_like, true_params[0]), true_params[1]),
            X.data,
            y,
        )

    @pytest.mark.parametrize("solver_name", ["GradientDescent", "SVRG"])
    def test_solver_output_match(self, poissonGLM_model_instantiation, solver_name):
        """Test that different solvers converge to the same solution."""
        jax.config.update("jax_enable_x64", True)
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        # set precision to float64 for accurate matching of the results
        model.data_type = jnp.float64

        # set model params
        model.set_params(regularizer=self.cls(), regularizer_strength=1.0)
        model.solver_name = solver_name
        model.solver_kwargs = {"tol": 10**-12}

        model_bfgs = copy.deepcopy(model)
        model_bfgs.solver_name = "BFGS"

        runner_gd = model.instantiate_solver().solver_run
        runner_bfgs = model_bfgs.instantiate_solver().solver_run

        weights_gd, intercepts_gd = runner_gd(
            (true_params[0] * 0.0, true_params[1]), X, y
        )[0]
        weights_bfgs, intercepts_bfgs = runner_bfgs(
            (true_params[0] * 0.0, true_params[1]), X, y
        )[0]

        match_weights = np.allclose(weights_gd, weights_bfgs)
        match_intercepts = np.allclose(intercepts_gd, intercepts_bfgs)

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
        model.set_params(regularizer=self.cls(), regularizer_strength=1.0)
        model.solver_kwargs = {"tol": 10**-12}
        model.solver_name = "BFGS"

        runner_bfgs = model.instantiate_solver().solver_run
        weights_bfgs, intercepts_bfgs = runner_bfgs(
            (true_params[0] * 0.0, true_params[1]), X, y
        )[0]
        model_skl = PoissonRegressor(
            fit_intercept=True,
            tol=10**-12,
            alpha=model.regularizer_strength,
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
        model.set_params(regularizer=self.cls(), regularizer_strength=1.0)
        model.solver_kwargs = {"tol": 10**-12}
        model.regularizer_strength = 0.1
        model.solver_name = "BFGS"
        runner_bfgs = model.instantiate_solver().solver_run
        weights_bfgs, intercepts_bfgs = runner_bfgs(
            (true_params[0] * 0.0, true_params[1]), X, y
        )[0]
        model_skl = GammaRegressor(
            fit_intercept=True,
            tol=10**-12,
            alpha=model.regularizer_strength,
        )
        model_skl.fit(X, y)

        match_weights = np.allclose(model_skl.coef_, weights_bfgs)
        match_intercepts = np.allclose(model_skl.intercept_, intercepts_bfgs)
        if (not match_weights) or (not match_intercepts):
            raise ValueError("Ridge GLM estimate does not match sklearn!")

    @pytest.mark.parametrize(
        "solver_name",
        [
            "GradientDescent",
            "BFGS",
            "LBFGS",
            "NonlinearCG",
            "ProximalGradient",
        ],
    )
    def test_solver_combination(self, solver_name, poissonGLM_model_instantiation):
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.set_params(regularizer=self.cls(), regularizer_strength=1.0)
        model.solver_name = solver_name
        model.fit(X, y)


class TestLasso:
    cls = nmo.regularizer.Lasso

    @pytest.mark.parametrize(
        "solver_name",
        [
            "GradientDescent",
            "BFGS",
            "ProximalGradient",
            "AGradientDescent",
            1,
            "SVRG",
            "ProxSVRG",
        ],
    )
    def test_init_solver_name(self, solver_name):
        """Test Lasso acceptable solvers."""
        acceptable_solvers = [
            "ProximalGradient",
            "ProxSVRG",
        ]
        raise_exception = solver_name not in acceptable_solvers
        if raise_exception:
            with pytest.raises(
                ValueError, match=f"The solver: {solver_name} is not allowed for "
            ):
                nmo.glm.GLM(
                    regularizer=self.cls(),
                    solver_name=solver_name,
                    regularizer_strength=1,
                )
        else:
            nmo.glm.GLM(
                regularizer=self.cls(), solver_name=solver_name, regularizer_strength=1
            )

    @pytest.mark.parametrize(
        "solver_name",
        [
            "GradientDescent",
            "BFGS",
            "ProximalGradient",
            "AGradientDescent",
            1,
            "SVRG",
            "ProxSVRG",
        ],
    )
    def test_set_solver_name_allowed(self, solver_name):
        """Test Lasso acceptable solvers."""
        acceptable_solvers = [
            "ProximalGradient",
            "ProxSVRG",
        ]
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer, regularizer_strength=1)
        raise_exception = solver_name not in acceptable_solvers
        if raise_exception:
            with pytest.raises(
                ValueError, match=f"The solver: {solver_name} is not allowed for "
            ):
                model.set_params(solver_name=solver_name)
        else:
            model.set_params(solver_name=solver_name)

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    @pytest.mark.parametrize("solver_kwargs", [{"tol": 10**-10}, {"tols": 10**-10}])
    def test_init_solver_kwargs(self, solver_kwargs, solver_name):
        """Test LassoSolver acceptable kwargs."""
        regularizer = self.cls()
        raise_exception = "tols" in list(solver_kwargs.keys())
        if raise_exception:
            with pytest.raises(
                NameError, match="kwargs {'tols'} in solver_kwargs not a kwarg"
            ):
                nmo.glm.GLM(
                    regularizer=regularizer,
                    solver_name=solver_name,
                    solver_kwargs=solver_kwargs,
                    regularizer_strength=1.0,
                )
        else:
            nmo.glm.GLM(
                regularizer=regularizer,
                solver_name=solver_name,
                solver_kwargs=solver_kwargs,
                regularizer_strength=1.0,
            )

    def test_regularizer_strength_none(self):
        """Assert regularizer strength handled appropriately."""
        # if no strength given, should warn and set to 1.0
        regularizer = self.cls()
        with pytest.warns(UserWarning):
            model = nmo.glm.GLM(regularizer=regularizer)

        assert model.regularizer_strength == 1.0

        # if changed to regularized, should go to None
        model.set_params(regularizer="UnRegularized", regularizer_strength=None)
        assert model.regularizer_strength is None

        # if changed back, should warn and set to 1.0
        with pytest.warns(UserWarning):
            model.regularizer = regularizer

        assert model.regularizer_strength == 1.0

    def test_get_params(self):
        """Test get_params() returns expected values."""
        regularizer = self.cls()

        assert regularizer.get_params() == {}

    @pytest.mark.parametrize("loss", [lambda a, b, c: 0, 1, None, {}])
    def test_loss_callable(self, loss):
        """Test that the loss function is a callable"""
        raise_exception = not callable(loss)
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer, regularizer_strength=1)
        model._predict_and_compute_loss = loss
        if raise_exception:
            with pytest.raises(TypeError, match="The `loss` must be a Callable"):
                nmo.utils.assert_is_callable(model._predict_and_compute_loss, "loss")
        else:
            nmo.utils.assert_is_callable(model._predict_and_compute_loss, "loss")

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    def test_run_solver(self, solver_name, poissonGLM_model_instantiation):
        """Test that the solver runs."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        model.set_params(regularizer=self.cls(), regularizer_strength=1)
        model.solver_name = solver_name
        runner = model.instantiate_solver().solver_run
        runner((true_params[0] * 0.0, true_params[1]), X, y)

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    def test_run_solver_tree(self, solver_name, poissonGLM_model_instantiation_pytree):
        """Test that the solver runs."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation_pytree

        # set regularizer and solver name
        model.set_params(regularizer=self.cls(), regularizer_strength=1)
        model.solver_name = solver_name
        runner = model.instantiate_solver().solver_run
        runner(
            (jax.tree_util.tree_map(jnp.zeros_like, true_params[0]), true_params[1]),
            X.data,
            y,
        )

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    def test_solver_match_statsmodels(
        self, solver_name, poissonGLM_model_instantiation
    ):
        """Test that different solvers converge to the same solution."""
        jax.config.update("jax_enable_x64", True)
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        # set precision to float64 for accurate matching of the results
        model.data_type = jnp.float64
        model.set_params(regularizer=self.cls(), regularizer_strength=1)
        model.solver_name = solver_name
        model.solver_kwargs = {"tol": 10**-12}

        runner = model.instantiate_solver().solver_run
        weights, intercepts = runner((true_params[0] * 0.0, true_params[1]), X, y)[0]

        # instantiate the glm with statsmodels
        glm_sm = sm.GLM(endog=y, exog=sm.add_constant(X), family=sm.families.Poisson())

        # regularize everything except intercept
        alpha_sm = np.ones(X.shape[1] + 1) * model.regularizer_strength
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

    def test_lasso_pytree(self, poissonGLM_model_instantiation_pytree):
        """Check pytree X can be fit."""
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation_pytree
        model.set_params(regularizer=nmo.regularizer.Lasso(), regularizer_strength=1.0)
        model.solver_name = "ProximalGradient"
        model.fit(X, y)

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    @pytest.mark.parametrize("reg_str", [0.001, 0.01, 0.1, 1, 10])
    def test_lasso_pytree_match(
        self,
        reg_str,
        solver_name,
        poissonGLM_model_instantiation_pytree,
        poissonGLM_model_instantiation,
    ):
        """Check pytree and array find same solution."""
        jax.config.update("jax_enable_x64", True)
        X, _, model, _, _ = poissonGLM_model_instantiation_pytree
        X_array, y, model_array, _, _ = poissonGLM_model_instantiation

        model.set_params(
            regularizer=nmo.regularizer.Lasso(), regularizer_strength=reg_str
        )
        model_array.set_params(
            regularizer=nmo.regularizer.Lasso(), regularizer_strength=reg_str
        )
        model.solver_name = solver_name
        model_array.solver_name = solver_name
        model.fit(X, y)
        model_array.fit(X_array, y)
        assert np.allclose(
            np.hstack(jax.tree_util.tree_leaves(model.coef_)), model_array.coef_
        )

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    def test_solver_combination(self, solver_name, poissonGLM_model_instantiation):
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.set_params(regularizer=self.cls(), regularizer_strength=1.0)
        model.solver_name = solver_name
        model.fit(X, y)


class TestGroupLasso:
    cls = nmo.regularizer.GroupLasso

    @pytest.mark.parametrize(
        "solver_name",
        [
            "GradientDescent",
            "BFGS",
            "ProximalGradient",
            "AGradientDescent",
            1,
            "SVRG",
            "ProxSVRG",
        ],
    )
    def test_init_solver_name(self, solver_name):
        """Test GroupLasso acceptable solvers."""
        acceptable_solvers = [
            "ProximalGradient",
            "ProxSVRG",
        ]
        raise_exception = solver_name not in acceptable_solvers

        # create a valid mask
        mask = np.zeros((2, 10))
        mask[0, :5] = 1
        mask[1, 5:] = 1
        mask = jnp.asarray(mask)

        if raise_exception:
            with pytest.raises(
                ValueError, match=f"The solver: {solver_name} is not allowed for "
            ):
                nmo.glm.GLM(
                    regularizer=self.cls(mask=mask),
                    solver_name=solver_name,
                    regularizer_strength=1,
                )
        else:
            nmo.glm.GLM(
                regularizer=self.cls(mask=mask),
                solver_name=solver_name,
                regularizer_strength=1,
            )

    @pytest.mark.parametrize(
        "solver_name",
        [
            "GradientDescent",
            "BFGS",
            "ProximalGradient",
            "AGradientDescent",
            1,
            "SVRG",
            "ProxSVRG",
        ],
    )
    def test_set_solver_name_allowed(self, solver_name):
        """Test GroupLassoSolver acceptable solvers."""
        acceptable_solvers = [
            "ProximalGradient",
            "ProxSVRG",
        ]
        # create a valid mask
        mask = np.zeros((2, 10))
        mask[0, :5] = 1
        mask[1, 5:] = 1
        mask = jnp.asarray(mask)
        regularizer = self.cls(mask=mask)
        raise_exception = solver_name not in acceptable_solvers
        model = nmo.glm.GLM(regularizer=regularizer, regularizer_strength=1)
        if raise_exception:
            with pytest.raises(
                ValueError, match=f"The solver: {solver_name} is not allowed for "
            ):
                model.set_params(solver_name=solver_name)
        else:
            model.set_params(solver_name=solver_name)

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    @pytest.mark.parametrize("solver_kwargs", [{"tol": 10**-10}, {"tols": 10**-10}])
    def test_init_solver_kwargs(self, solver_name, solver_kwargs):
        """Test GroupLasso acceptable kwargs."""
        raise_exception = "tols" in list(solver_kwargs.keys())

        # create a valid mask
        mask = np.zeros((2, 10))
        mask[0, :5] = 1
        mask[0, 1:] = 1
        mask = jnp.asarray(mask)

        regularizer = self.cls(mask=mask)

        if raise_exception:
            with pytest.raises(
                NameError, match="kwargs {'tols'} in solver_kwargs not a kwarg"
            ):
                nmo.glm.GLM(
                    regularizer=regularizer,
                    solver_name=solver_name,
                    solver_kwargs=solver_kwargs,
                    regularizer_strength=1.0,
                )
        else:
            nmo.glm.GLM(
                regularizer=regularizer,
                solver_name=solver_name,
                solver_kwargs=solver_kwargs,
                regularizer_strength=1.0,
            )

    def test_regularizer_strength_none(self):
        """Assert regularizer strength handled appropriately."""
        # if no strength given, should warn and set to 1.0
        regularizer = self.cls()
        with pytest.warns(UserWarning):
            model = nmo.glm.GLM(regularizer=regularizer)

        assert model.regularizer_strength == 1.0

        # if changed to regularized, should go to None
        model.set_params(regularizer="UnRegularized", regularizer_strength=None)

        # if changed back, should warn and set to 1.0
        with pytest.warns(UserWarning):
            model.regularizer = regularizer

        assert model.regularizer_strength == 1.0

    def test_get_params(self):
        """Test get_params() returns expected values."""
        regularizer = self.cls()

        assert regularizer.get_params() == {"mask": None}

    @pytest.mark.parametrize("loss", [lambda a, b, c: 0, 1, None, {}])
    def test_loss_callable(self, loss):
        """Test that the loss function is a callable"""
        raise_exception = not callable(loss)

        # create a valid mask
        mask = np.zeros((2, 10))
        mask[0, :5] = 1
        mask[1, 5:] = 1
        mask = jnp.asarray(mask)

        regularizer = self.cls(mask=mask)
        model = nmo.glm.GLM(regularizer=regularizer, regularizer_strength=1.0)
        model._predict_and_compute_loss = loss

        if raise_exception:
            with pytest.raises(TypeError, match="The `loss` must be a Callable"):
                nmo.utils.assert_is_callable(model._predict_and_compute_loss, "loss")
        else:
            nmo.utils.assert_is_callable(model._predict_and_compute_loss, "loss")

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    def test_run_solver(self, solver_name, poissonGLM_model_instantiation):
        """Test that the solver runs."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        # create a valid mask
        mask = np.zeros((2, X.shape[1]))
        mask[0, :2] = 1
        mask[1, 2:] = 1
        mask = jnp.asarray(mask)

        model.set_params(regularizer=self.cls(mask=mask), regularizer_strength=1.0)
        model.solver_name = solver_name

        model.instantiate_solver()
        model.solver_run((true_params[0] * 0.0, true_params[1]), X, y)

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    def test_init_solver(self, solver_name, poissonGLM_model_instantiation):
        """Test that the solver initialization returns a state."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        # create a valid mask
        mask = np.zeros((2, X.shape[1]))
        mask[0, :2] = 1
        mask[1, 2:] = 1
        mask = jnp.asarray(mask)

        model.set_params(regularizer=self.cls(mask=mask), regularizer_strength=1.0)
        model.solver_name = solver_name

        model.instantiate_solver()
        state = model.solver_init_state(true_params, X, y)
        # asses that state is a NamedTuple by checking tuple type and the availability of some NamedTuple
        # specific namespace attributes
        assert isinstance(state, tuple)
        assert (
            hasattr(state, "_fields")
            and hasattr(state, "_field_defaults")
            and hasattr(state, "_asdict")
        )

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    def test_update_solver(self, solver_name, poissonGLM_model_instantiation):
        """Test that the solver initialization returns a state."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        # create a valid mask
        mask = np.zeros((2, X.shape[1]))
        mask[0, :2] = 1
        mask[1, 2:] = 1
        mask = jnp.asarray(mask)

        model.set_params(regularizer=self.cls(mask=mask), regularizer_strength=1.0)
        model.solver_name = solver_name

        model.instantiate_solver()

        state = model.solver_init_state((true_params[0] * 0.0, true_params[1]), X, y)

        # ProxSVRG needs the full gradient at the anchor point to be initialized
        # so here just set it to xs, which is not correct, but fine shape-wise
        if solver_name == "ProxSVRG":
            state = state._replace(full_grad_at_reference_point=state.reference_point)

        params, state = model.solver_update(true_params, state, X, y)
        # asses that state is a NamedTuple by checking tuple type and the availability of some NamedTuple
        # specific namespace attributes
        assert isinstance(state, tuple)
        assert (
            hasattr(state, "_fields")
            and hasattr(state, "_field_defaults")
            and hasattr(state, "_asdict")
        )
        # check params struct and shapes
        assert jax.tree_util.tree_structure(params) == jax.tree_util.tree_structure(
            true_params
        )
        assert all(
            jax.tree_util.tree_leaves(params)[k].shape == p.shape
            for k, p in enumerate(jax.tree_util.tree_leaves(true_params))
        )

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
                model.set_params(
                    regularizer=self.cls(mask=mask), regularizer_strength=1.0
                )
        else:
            model.set_params(regularizer=self.cls(mask=mask), regularizer_strength=1.0)

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
                model.set_params(
                    regularizer=self.cls(mask=mask), regularizer_strength=1.0
                )
        else:
            model.set_params(regularizer=self.cls(mask=mask), regularizer_strength=1.0)

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
                model.set_params(
                    regularizer=self.cls(mask=mask), regularizer_strength=1.0
                )
        else:
            model.set_params(regularizer=self.cls(mask=mask), regularizer_strength=1.0)

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
                model.set_params(
                    regularizer=self.cls(mask=mask), regularizer_strength=1.0
                )
        else:
            model.set_params(regularizer=self.cls(mask=mask), regularizer_strength=1.0)

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

        model.set_params(regularizer=self.cls(mask=mask), regularizer_strength=1.0)
        model.solver_name = "ProximalGradient"

        runner = model.instantiate_solver().solver_run
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
        regularizer = self.cls(mask=mask)

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
        regularizer = self.cls(mask=mask)

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
        regularizer = self.cls(mask=valid_mask)

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
        regularizer = self.cls(mask=valid_mask)

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

    def test_mask_none(self, poissonGLM_model_instantiation):
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        with pytest.warns(UserWarning):
            model.regularizer = self.cls(mask=np.ones((1, X.shape[1])).astype(float))
            model.solver_name = "ProximalGradient"
            model.fit(X, y)

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    def test_solver_combination(self, solver_name, poissonGLM_model_instantiation):
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.set_params(
            regularizer=self.cls(mask=np.ones((1, X.shape[1])).astype(float)),
            regularizer_strength=(
                None if self.cls == nmo.regularizer.UnRegularized else 1.0
            ),
        )
        model.solver_name = solver_name
        model.fit(X, y)


def test_available_regularizer_match():
    """Test matching of the two regularizer lists."""
    assert set(nmo._regularizer_builder.AVAILABLE_REGULARIZERS) == set(
        nmo.regularizer.__dir__()
    )
