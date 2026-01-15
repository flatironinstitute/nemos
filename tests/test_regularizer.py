import copy
import warnings
from contextlib import nullcontext as does_not_raise

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import statsmodels.api as sm
from scipy.optimize import minimize
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from statsmodels.tools.sm_exceptions import DomainWarning

import nemos as nmo
from nemos.glm.params import GLMParams

# Register every test here as solver-related
pytestmark = pytest.mark.solver_related


@pytest.mark.parametrize(
    "reg_str, reg_type",
    [
        ("UnRegularized", nmo.regularizer.UnRegularized),
        (None, nmo.regularizer.UnRegularized),
        ("Ridge", nmo.regularizer.Ridge),
        ("Lasso", nmo.regularizer.Lasso),
        ("GroupLasso", nmo.regularizer.GroupLasso),
        ("ElasticNet", nmo.regularizer.ElasticNet),
        ("not_valid", None),
        ("nemos.regularizer.UnRegularized", nmo.regularizer.UnRegularized),
        ("nemos.regularizer.Ridge", nmo.regularizer.Ridge),
        ("nemos.regularizer.Lasso", nmo.regularizer.Lasso),
        ("nemos.regularizer.GroupLasso", nmo.regularizer.GroupLasso),
    ],
)
def test_regularizer_builder(reg_str, reg_type):
    """Test building a regularizer from a string"""
    valid_regularizers = nmo._regularizer_builder.AVAILABLE_REGULARIZERS
    raise_exception = reg_str is not None and not (
        reg_str in valid_regularizers
        or any(reg_str == f"nemos.regularizer.{name}" for name in valid_regularizers)
    )
    if raise_exception:
        with pytest.raises(ValueError, match=f"Unknown regularizer: {reg_str}. "):
            nmo._regularizer_builder.instantiate_regularizer(reg_str)
    else:
        # build a regularizer by string
        regularizer = nmo._regularizer_builder.instantiate_regularizer(reg_str)
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
        ("ElasticNet()", nmo.regularizer.ElasticNet()),
    ],
)
def test_regularizer_repr(reg, expected):
    assert repr(reg) == expected


def test_regularizer_available():
    for regularizer in nmo._regularizer_builder.AVAILABLE_REGULARIZERS:
        reg = nmo._regularizer_builder.instantiate_regularizer(regularizer)
        assert reg.__class__.__name__ == regularizer


@pytest.mark.parametrize(
    "regularizer_strength",
    [0.001, 1.0, "bah"],
)
@pytest.mark.parametrize(
    "reg_type",
    [
        nmo.regularizer.Ridge,
        nmo.regularizer.Lasso,
        nmo.regularizer.GroupLasso,
        nmo.regularizer.ElasticNet,
    ],
)
def test_regularizer(regularizer_strength, reg_type):
    if not isinstance(regularizer_strength, float):
        with pytest.raises(
            ValueError,
            match=r"Could not convert the (regularizer strength|regularizer strength and regularizer ratio): {regularizer_strength} "
            r"to a (float|tuple of floats).".format(
                regularizer_strength=regularizer_strength
            ),
        ):
            nmo.glm.GLM(
                regularizer=reg_type(), regularizer_strength=regularizer_strength
            )
    else:
        nmo.glm.GLM(regularizer=reg_type(), regularizer_strength=regularizer_strength)


@pytest.mark.parametrize(
    "regularizer_strength",
    [0.001, 1.0, "bah"],
)
@pytest.mark.parametrize(
    "regularizer",
    [
        nmo.regularizer.Ridge(),
        nmo.regularizer.Lasso(),
        nmo.regularizer.GroupLasso(),
        nmo.regularizer.ElasticNet(),
    ],
)
def test_regularizer_setter(regularizer_strength, regularizer):
    if not isinstance(regularizer_strength, float):
        with pytest.raises(
            ValueError,
            match=r"Could not convert the (regularizer strength|regularizer strength and regularizer ratio): {regularizer_strength} "
            r"to a (float|tuple of floats).".format(
                regularizer_strength=regularizer_strength
            ),
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
        nmo.regularizer.ElasticNet(),
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
        nmo.regularizer.ElasticNet(),
    ],
)
def test_item_assignment_allowed_solvers(regularizer):
    with pytest.raises(
        TypeError, match="'tuple' object does not support item assignment"
    ):
        regularizer.allowed_solvers[0] = "my-favourite-solver"


@pytest.mark.parametrize(
    "regularizer, regularizer_strength",
    [
        ("Ridge", 2.0),
        ("Lasso", 2.0),
        ("GroupLasso", 2.0),
        ("ElasticNet", (2.0, 0.7)),
    ],
)
def test_set_params_order_change_regularizer(regularizer, regularizer_strength):
    """Test that set_params() when changing regularizer and regularizer_strength regardless of order."""
    # start with unregularized
    model = nmo.glm.GLM()
    assert model.regularizer_strength is None
    assert isinstance(model.regularizer, nmo.regularizer.UnRegularized)

    # set regularizer first
    model.set_params(regularizer=regularizer, regularizer_strength=regularizer_strength)
    assert model.regularizer_strength == regularizer_strength
    assert model.regularizer.__class__.__name__ == regularizer

    # set regularizer_strength first
    model.set_params(regularizer="UnRegularized")
    assert model.regularizer_strength is None

    model.set_params(regularizer_strength=regularizer_strength, regularizer=regularizer)
    assert model.regularizer_strength == regularizer_strength
    assert model.regularizer.__class__.__name__ == regularizer


@pytest.mark.parametrize(
    "regularizer, regularizer_strength",
    [
        ("Ridge", 2.0),
        ("Lasso", 2.0),
        ("GroupLasso", 2.0),
        ("ElasticNet", (2.0, 0.7)),
        ("UnRegularized", None),
    ],
)
@pytest.mark.parametrize(
    "regularizer2, regularizer2_default",
    [
        ("Ridge", 1.0),
        ("Lasso", 1.0),
        ("GroupLasso", 1.0),
        ("ElasticNet", (1.0, 0.5)),
        ("UnRegularized", None),
    ],
)
def test_change_regularizer_reset_strength(
    regularizer,
    regularizer_strength,
    regularizer2,
    regularizer2_default,
):
    """Test that set_params() when changing regularizer and regularizer_strength regardless of order."""
    model = nmo.glm.GLM(
        regularizer=regularizer, regularizer_strength=regularizer_strength
    )
    assert model.regularizer_strength == regularizer_strength
    assert model.regularizer.__class__.__name__ == regularizer

    # check that regularizer_strength is reset when changing regularizer
    model.set_params(regularizer=regularizer2)
    assert model.regularizer_strength == regularizer2_default

    # make sure there is no conflict when setting back
    model.set_params(regularizer=regularizer, regularizer_strength=regularizer_strength)
    assert model.regularizer_strength == regularizer_strength


class TestUnRegularized:
    cls = nmo.regularizer.UnRegularized

    @pytest.mark.parametrize(
        "solver_name, expectation",
        [
            ("GradientDescent", does_not_raise()),
            ("BFGS", does_not_raise()),
            ("ProximalGradient", does_not_raise()),
            (
                "AGradientDescent",
                pytest.raises(
                    ValueError,
                    match="The solver: AGradientDescent is not allowed for",
                ),
            ),
            (1, pytest.raises(TypeError, match="solver_name must be a string")),
            ("SVRG", does_not_raise()),
            ("ProxSVRG", does_not_raise()),
        ],
    )
    def test_init_solver_name(self, solver_name, expectation):
        """Test UnRegularized acceptable solvers."""
        with expectation:
            nmo.glm.GLM(regularizer=self.cls(), solver_name=solver_name)

    @pytest.mark.parametrize(
        "solver_name, expectation",
        [
            ("GradientDescent", does_not_raise()),
            ("BFGS", does_not_raise()),
            ("ProximalGradient", does_not_raise()),
            (
                "AGradientDescent",
                pytest.raises(
                    ValueError,
                    match="The solver: AGradientDescent is not allowed for",
                ),
            ),
            (1, pytest.raises(TypeError, match="solver_name must be a string")),
            ("SVRG", does_not_raise()),
            ("ProxSVRG", does_not_raise()),
        ],
    )
    def test_set_solver_name_allowed(self, solver_name, expectation):
        """Test UnRegularized acceptable solvers."""
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer)
        with expectation:
            model.set_params(solver_name=solver_name)

    def test_regularizer_strength_none(self):
        """Add test to assert that regularizer strength of UnRegularized model should be `None`"""
        # unregularized should be None
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer)

        assert model.regularizer_strength is None

        # changing to ridge, lasso, or grouplasso should set to 1.0
        model.regularizer = "Lasso"

        assert model.regularizer_strength == 1.0
        model.regularizer = regularizer
        assert model.regularizer_strength == None

    def test_get_params(self):
        """Test get_params() returns expected values."""
        regularizer = self.cls()

        assert regularizer.get_params() == {}

    @pytest.mark.parametrize(
        "solver_name", ["GradientDescent", "BFGS", "SVRG", "ProxSVRG"]
    )
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
        model._compute_loss = loss
        if raise_exception:
            with pytest.raises(TypeError, match="The `loss` must be a Callable"):
                nmo.utils.assert_is_callable(model._compute_loss, "loss")
        else:
            nmo.utils.assert_is_callable(model._compute_loss, "loss")

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
        params = GLMParams(true_params.coef * 0.0, true_params.intercept)
        model._instantiate_solver(model._compute_loss, params)
        model.solver_run(params, X, y)

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
        params = GLMParams(
            jax.tree_util.tree_map(jnp.zeros_like, true_params.coef),
            true_params.intercept,
        )
        model._instantiate_solver(model._compute_loss, params)
        model.solver_run(
            params,
            X.data,
            y,
        )

    @pytest.mark.parametrize("solver_name", ["GradientDescent", "SVRG"])
    @pytest.mark.requires_x64
    def test_solver_output_match(self, poissonGLM_model_instantiation, solver_name):
        """Test that different solvers converge to the same solution."""
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        # set model params
        model.set_params(regularizer=self.cls())
        model.solver_name = solver_name
        model.solver_kwargs = {"tol": 10**-12}

        init_params = GLMParams(true_params.coef * 0.0, true_params.intercept)
        model._instantiate_solver(model._compute_loss, init_params)

        # update solver name
        model_bfgs = copy.deepcopy(model)
        model_bfgs.solver_name = "BFGS"
        model_bfgs._instantiate_solver(model_bfgs._compute_loss, init_params)
        params_gd = model.solver_run(init_params, X, y)[0]
        params_bfgs = model_bfgs.solver_run(init_params, X, y)[0]

        match_weights = np.allclose(params_gd.coef, params_bfgs.coef)
        match_intercepts = np.allclose(params_gd.intercept, params_bfgs.intercept)

        if (not match_weights) or (not match_intercepts):
            raise ValueError(
                "Convex estimators should converge to the same numerical value."
            )

    @pytest.mark.parametrize("solver_name", ["GradientDescent", "SVRG"])
    @pytest.mark.requires_x64
    def test_solver_match_sklearn(self, poissonGLM_model_instantiation, solver_name):
        """Test that different solvers converge to the same solution."""
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.set_params(regularizer=self.cls())
        model.solver_name = solver_name
        model.solver_kwargs = {"tol": 10**-12}
        init_params = GLMParams(true_params.coef * 0.0, true_params.intercept)
        model._instantiate_solver(model._compute_loss, init_params)
        params = model.solver_run(init_params, X, y)[0]
        model_skl = PoissonRegressor(fit_intercept=True, tol=10**-12, alpha=0.0)
        model_skl.fit(X, y)

        match_weights = np.allclose(model_skl.coef_, params.coef)
        match_intercepts = np.allclose(model_skl.intercept_, params.intercept)
        if (not match_weights) or (not match_intercepts):
            raise ValueError("UnRegularized GLM estimate does not match sklearn!")

    @pytest.mark.parametrize("solver_name", ["GradientDescent", "SVRG"])
    @pytest.mark.requires_x64
    def test_solver_match_sklearn_gamma(
        self, gammaGLM_model_instantiation, solver_name
    ):
        """Test that different solvers converge to the same solution."""
        X, y, model, true_params, firing_rate = gammaGLM_model_instantiation
        model.inverse_link_function = jnp.exp
        model.set_params(regularizer=self.cls())
        model.solver_name = solver_name
        model.solver_kwargs = {"tol": 10**-12}
        init_params = GLMParams(true_params.coef * 0.0, true_params.intercept)
        model._instantiate_solver(model._compute_loss, init_params)
        params = model.solver_run(init_params, X, y)[0]
        model_skl = GammaRegressor(fit_intercept=True, tol=10**-12, alpha=0.0)
        model_skl.fit(X, y)

        match_weights = np.allclose(model_skl.coef_, params.coef)
        match_intercepts = np.allclose(model_skl.intercept_, params.intercept)
        if (not match_weights) or (not match_intercepts):
            raise ValueError("Unregularized GLM estimate does not match sklearn!")

    @pytest.mark.parametrize(
        "inv_link_jax, link_sm",
        [
            (jnp.exp, sm.families.links.Log()),
            (lambda x: 1 / x, sm.families.links.InversePower()),
        ],
    )
    @pytest.mark.parametrize("solver_name", ["LBFGS", "SVRG"])
    @pytest.mark.requires_x64
    def test_solver_match_statsmodels_gamma(
        self, inv_link_jax, link_sm, gammaGLM_model_instantiation, solver_name
    ):
        """Test that different solvers converge to the same solution."""
        X, y, model, true_params, firing_rate = gammaGLM_model_instantiation
        model.inverse_link_function = inv_link_jax
        model.set_params(regularizer=self.cls())
        model.solver_name = solver_name
        model.solver_kwargs = {"tol": 10**-13}
        init_params = model._model_specific_initialization(X, y)
        model._instantiate_solver(model._compute_loss, init_params)
        params = model.solver_run(init_params, X, y)[0]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The InversePower link function does "
            )
            model_sm = sm.GLM(
                endog=y, exog=sm.add_constant(X), family=sm.families.Gamma(link=link_sm)
            )

        res_sm = model_sm.fit(cnvrg_tol=10**-12)

        match_weights = np.allclose(res_sm.params[1:], params.coef)
        match_intercepts = np.allclose(res_sm.params[:1], params.intercept)
        if (not match_weights) or (not match_intercepts):
            raise ValueError("Unregularized GLM estimate does not match statsmodels!")

    @pytest.mark.parametrize(
        "inv_link_jax, link_sm",
        [
            (jnp.exp, sm.families.links.Log()),
        ],
    )
    @pytest.mark.parametrize("solver_name", ["LBFGS", "SVRG", "ProximalGradient"])
    @pytest.mark.requires_x64
    def test_solver_match_statsmodels_negative_binomial(
        self,
        inv_link_jax,
        link_sm,
        negativeBinomialGLM_model_instantiation,
        solver_name,
    ):
        """Test that different solvers converge to the same solution."""
        X, y, model, true_params, firing_rate = negativeBinomialGLM_model_instantiation
        y = y.astype(
            float
        )  # needed since solver.run is called directly, nemos converts.
        # set precision to float64 for accurate matching of the results
        model.data_type = jnp.float64
        model.observation_model.inverse_link_function = inv_link_jax
        model.set_params(regularizer=self.cls())
        model.solver_name = solver_name
        model.solver_kwargs = {"tol": 10**-13}
        init_params = model._model_specific_initialization(X, y)
        model._instantiate_solver(model._compute_loss, init_params)
        params = model.solver_run(init_params, X, y)[0]
        model_sm = sm.GLM(
            endog=y,
            exog=sm.add_constant(X),
            family=sm.families.NegativeBinomial(
                link=link_sm, alpha=model.observation_model.scale
            ),
        )

        res_sm = model_sm.fit(cnvrg_tol=10**-12)

        match_weights = np.allclose(res_sm.params[1:], params.coef, atol=10**-6)
        match_intercepts = np.allclose(res_sm.params[:1], params.intercept, atol=10**-6)
        if (not match_weights) or (not match_intercepts):
            raise ValueError(
                "Unregularized GLM estimate does not match statsmodels!\n"
                f"Intercept difference is: {res_sm.params[:1] - params.intercept}\n"
                f"Coefficient difference is: {res_sm.params[1:] - params.coef}"
            )

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
        "solver_name, expectation",
        [
            ("GradientDescent", does_not_raise()),
            ("BFGS", does_not_raise()),
            ("ProximalGradient", does_not_raise()),
            (
                "AGradientDescent",
                pytest.raises(
                    ValueError,
                    match="The solver: AGradientDescent is not allowed for",
                ),
            ),
            (1, pytest.raises(TypeError, match="solver_name must be a string")),
            ("SVRG", does_not_raise()),
            ("ProxSVRG", does_not_raise()),
        ],
    )
    def test_init_solver_name(self, solver_name, expectation):
        """Test Ridge acceptable solvers."""
        with expectation:
            nmo.glm.GLM(regularizer=self.cls(), solver_name=solver_name)

    @pytest.mark.parametrize(
        "solver_name, expectation",
        [
            ("GradientDescent", does_not_raise()),
            ("BFGS", does_not_raise()),
            ("ProximalGradient", does_not_raise()),
            (
                "AGradientDescent",
                pytest.raises(
                    ValueError,
                    match="The solver: AGradientDescent is not allowed for",
                ),
            ),
            (1, pytest.raises(TypeError, match="solver_name must be a string")),
            ("SVRG", does_not_raise()),
            ("ProxSVRG", does_not_raise()),
        ],
    )
    def test_set_solver_name_allowed(self, solver_name, expectation):
        """Test Ridge acceptable solvers."""
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer, regularizer_strength=1.0)
        with expectation:
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
        # if no strength given, should set to 1.0
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer)

        assert model.regularizer_strength == 1.0
        # if changed to regularized, is set to None.
        model.regularizer = "UnRegularized"
        assert model.regularizer_strength is None

        # if changed back, should set to 1.0
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
        model._compute_loss = loss
        if raise_exception:
            with pytest.raises(TypeError, match="The `loss` must be a Callable"):
                nmo.utils.assert_is_callable(model._compute_loss, "loss")
        else:
            nmo.utils.assert_is_callable(model._compute_loss, "loss")

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
        params = GLMParams(true_params.coef * 0.0, true_params.intercept)
        runner = model._instantiate_solver(model._compute_loss, params).solver_run
        runner(params, X, y)

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
        params = GLMParams(
            jax.tree_util.tree_map(jnp.zeros_like, true_params.coef),
            true_params.intercept,
        )
        runner = model._instantiate_solver(model._compute_loss, params).solver_run
        runner(params, X.data, y)

    @pytest.mark.parametrize("solver_name", ["GradientDescent", "SVRG"])
    @pytest.mark.requires_x64
    def test_solver_output_match(self, poissonGLM_model_instantiation, solver_name):
        """Test that different solvers converge to the same solution."""
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        # set precision to float64 for accurate matching of the results
        model.data_type = jnp.float64

        # set model params
        model.set_params(regularizer=self.cls(), regularizer_strength=1.0)
        model.solver_name = solver_name
        model.solver_kwargs = {"tol": 10**-12}

        model_bfgs = copy.deepcopy(model)
        model_bfgs.solver_name = "BFGS"

        init_params = GLMParams(true_params.coef * 0.0, true_params.intercept)
        runner_gd = model._instantiate_solver(
            model._compute_loss, init_params
        ).solver_run
        runner_bfgs = model_bfgs._instantiate_solver(
            model_bfgs._compute_loss, init_params
        ).solver_run

        params_gd = runner_gd(init_params, X, y)[0]
        params_bfgs = runner_bfgs(init_params, X, y)[0]

        match_weights = np.allclose(params_gd.coef, params_bfgs.coef)
        match_intercepts = np.allclose(params_gd.intercept, params_bfgs.intercept)

        if (not match_weights) or (not match_intercepts):
            raise ValueError(
                "Convex estimators should converge to the same numerical value."
            )

    @pytest.mark.requires_x64
    def test_solver_match_sklearn(self, poissonGLM_model_instantiation):
        """Test that different solvers converge to the same solution."""
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        model.set_params(regularizer=self.cls(), regularizer_strength=1.0)
        model.solver_kwargs = {"tol": 10**-12}
        model.solver_name = "BFGS"

        init_params = GLMParams(true_params.coef * 0.0, true_params.intercept)
        runner_bfgs = model._instantiate_solver(
            model._compute_loss, init_params
        ).solver_run
        params = runner_bfgs(init_params, X, y)[0]
        model_skl = PoissonRegressor(
            fit_intercept=True,
            tol=10**-12,
            alpha=model.regularizer_strength,
        )
        model_skl.fit(X, y)

        match_weights = np.allclose(model_skl.coef_, params.coef)
        match_intercepts = np.allclose(model_skl.intercept_, params.intercept)
        if (not match_weights) or (not match_intercepts):
            raise ValueError("Ridge GLM solver estimate does not match sklearn!")

    @pytest.mark.parametrize("solver_name", ["LBFGS", "ProximalGradient"])
    @pytest.mark.requires_x64
    def test_solver_match_sklearn_gamma(
        self, solver_name, gammaGLM_model_instantiation
    ):
        """Test that different solvers converge to the same solution."""
        X, y, model, true_params, firing_rate = gammaGLM_model_instantiation
        model.inverse_link_function = jnp.exp
        model.set_params(regularizer=self.cls(), regularizer_strength=1.0)
        model.solver_kwargs = {"tol": 10**-12}
        model.regularizer_strength = 0.1
        model.solver_name = solver_name
        init_params = GLMParams(true_params.coef * 0.0, true_params.intercept)
        runner_bfgs = model._instantiate_solver(
            model._compute_loss, init_params
        ).solver_run
        params = runner_bfgs(init_params, X, y)[0]
        model_skl = GammaRegressor(
            fit_intercept=True,
            tol=10**-12,
            alpha=model.regularizer_strength,
        )
        model_skl.fit(X, y)

        match_weights = np.allclose(model_skl.coef_, params.coef)
        match_intercepts = np.allclose(model_skl.intercept_, params.intercept)
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
        "solver_name, expectation",
        [
            ("GradientDescent", pytest.raises(ValueError, match="not allowed for")),
            ("BFGS", pytest.raises(ValueError, match="not allowed for")),
            ("ProximalGradient", does_not_raise()),
            (
                "AGradientDescent",
                pytest.raises(
                    ValueError,
                    match="The solver: AGradientDescent is not allowed for",
                ),
            ),
            (1, pytest.raises(TypeError, match="solver_name must be a string")),
            ("SVRG", pytest.raises(ValueError, match="not allowed for")),
            ("ProxSVRG", does_not_raise()),
        ],
    )
    def test_init_solver_name(self, solver_name, expectation):
        """Test Lasso acceptable solvers."""
        with expectation:
            nmo.glm.GLM(regularizer=self.cls(), solver_name=solver_name)

    @pytest.mark.parametrize(
        "solver_name, expectation",
        [
            ("GradientDescent", pytest.raises(ValueError, match="not allowed for")),
            ("BFGS", pytest.raises(ValueError, match="not allowed for")),
            ("ProximalGradient", does_not_raise()),
            (
                "AGradientDescent",
                pytest.raises(
                    ValueError,
                    match="The solver: AGradientDescent is not allowed for",
                ),
            ),
            (1, pytest.raises(TypeError, match="solver_name must be a string")),
            ("SVRG", pytest.raises(ValueError, match="not allowed for")),
            ("ProxSVRG", does_not_raise()),
        ],
    )
    def test_set_solver_name_allowed(self, solver_name, expectation):
        """Test Lasso acceptable solvers."""
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer, regularizer_strength=1)
        with expectation:
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
        # if no strength given, should set to 1.0
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer)

        assert model.regularizer_strength == 1.0

        # if changed to regularized, should go to None
        model.set_params(regularizer="UnRegularized", regularizer_strength=None)
        assert model.regularizer_strength is None

        # if changed back, should set to 1.0
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
        model._compute_loss = loss
        if raise_exception:
            with pytest.raises(TypeError, match="The `loss` must be a Callable"):
                nmo.utils.assert_is_callable(model._compute_loss, "loss")
        else:
            nmo.utils.assert_is_callable(model._compute_loss, "loss")

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    def test_run_solver(self, solver_name, poissonGLM_model_instantiation):
        """Test that the solver runs."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        model.set_params(regularizer=self.cls(), regularizer_strength=1)
        model.solver_name = solver_name
        params = GLMParams(true_params.coef * 0.0, true_params.intercept)
        runner = model._instantiate_solver(model._compute_loss, params).solver_run
        runner(params, X, y)

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    def test_run_solver_tree(self, solver_name, poissonGLM_model_instantiation_pytree):
        """Test that the solver runs."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation_pytree

        # set regularizer and solver name
        model.set_params(regularizer=self.cls(), regularizer_strength=1)
        model.solver_name = solver_name
        params = GLMParams(
            jax.tree_util.tree_map(jnp.zeros_like, true_params.coef),
            true_params.intercept,
        )
        runner = model._instantiate_solver(model._compute_loss, params).solver_run
        runner(params, X.data, y)

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    @pytest.mark.requires_x64
    def test_solver_match_statsmodels(
        self, solver_name, poissonGLM_model_instantiation
    ):
        """Test that different solvers converge to the same solution."""
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        # set precision to float64 for accurate matching of the results
        model.data_type = jnp.float64
        model.set_params(regularizer=self.cls(), regularizer_strength=1)
        model.solver_name = solver_name
        model.solver_kwargs = {"tol": 10**-12}

        init_params = GLMParams(true_params.coef * 0.0, true_params.intercept)
        runner = model._instantiate_solver(model._compute_loss, init_params).solver_run
        params = runner(init_params, X, y)[0]

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
        glm_params = jnp.hstack((params.intercept, params.coef.flatten()))
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
    @pytest.mark.requires_x64
    def test_lasso_pytree_match(
        self,
        reg_str,
        solver_name,
        poissonGLM_model_instantiation_pytree,
        poissonGLM_model_instantiation,
    ):
        """Check pytree and array find same solution."""
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


class TestElasticNet:
    cls = nmo.regularizer.ElasticNet

    @pytest.mark.parametrize(
        "solver_name, expectation",
        [
            ("GradientDescent", pytest.raises(ValueError, match="not allowed for")),
            ("BFGS", pytest.raises(ValueError, match="not allowed for")),
            ("ProximalGradient", does_not_raise()),
            (
                "AGradientDescent",
                pytest.raises(
                    ValueError,
                    match="The solver: AGradientDescent is not allowed for",
                ),
            ),
            (1, pytest.raises(TypeError, match="solver_name must be a string")),
            ("SVRG", pytest.raises(ValueError, match="not allowed for")),
            ("ProxSVRG", does_not_raise()),
        ],
    )
    def test_init_solver_name(self, solver_name, expectation):
        """Test ElasticNet acceptable solvers."""
        with expectation:
            nmo.glm.GLM(regularizer=self.cls(), solver_name=solver_name)

    @pytest.mark.parametrize(
        "solver_name, expectation",
        [
            ("GradientDescent", pytest.raises(ValueError, match="not allowed for")),
            ("BFGS", pytest.raises(ValueError, match="not allowed for")),
            ("ProximalGradient", does_not_raise()),
            (
                "AGradientDescent",
                pytest.raises(
                    ValueError,
                    match="The solver: AGradientDescent is not allowed for",
                ),
            ),
            (1, pytest.raises(TypeError, match="solver_name must be a string")),
            ("SVRG", pytest.raises(ValueError, match="not allowed for")),
            ("ProxSVRG", does_not_raise()),
        ],
    )
    def test_set_solver_name_allowed(self, solver_name, expectation):
        """Test ElasticNet acceptable solvers."""
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer, regularizer_strength=(1, 0.5))
        with expectation:
            model.set_params(solver_name=solver_name)

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    @pytest.mark.parametrize("solver_kwargs", [{"tol": 10**-10}, {"tols": 10**-10}])
    def test_init_solver_kwargs(self, solver_kwargs, solver_name):
        """Test ElasticNetSolver acceptable kwargs."""
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
                    regularizer_strength=(1.0, 0.5),
                )
        else:
            nmo.glm.GLM(
                regularizer=regularizer,
                solver_name=solver_name,
                solver_kwargs=solver_kwargs,
                regularizer_strength=(1.0, 0.5),
            )

    def test_regularizer_strength_none(self):
        """Assert regularizer strength handled appropriately."""
        # if no strength given, set to (1.0, 0.5)
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer)

        assert model.regularizer_strength == (1.0, 0.5)

        # if changed to regularized, should go to None
        model.set_params(regularizer="UnRegularized", regularizer_strength=None)
        assert model.regularizer_strength is None

        # if changed back, set to (1.0, 0.5)
        model.regularizer = regularizer

        assert model.regularizer_strength == (1.0, 0.5)

    def test_regularizer_strength_float(self):
        """Assert regularizer ratio handled appropriately when only strength provided."""
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer, regularizer_strength=0.6)
        assert model.regularizer_strength == (0.6, 0.5)

    @pytest.mark.parametrize(
        "regularizer_strength, expectation",
        [
            ((1.0, 0.5), does_not_raise()),
            ((1.0, 1.0), does_not_raise()),
            (
                (1.0, 0.0),
                pytest.raises(
                    ValueError,
                    match="Regularization ratio of 0 is not supported. Use Ridge regularization instead.",
                ),
            ),
            (
                (1.0, 1.1),
                pytest.raises(
                    ValueError,
                    match="Regularization ratio must be a number between 0 and 1.",
                ),
            ),
            (
                (1.0, -0.1),
                pytest.raises(
                    ValueError,
                    match="Regularization ratio must be a number between 0 and 1.",
                ),
            ),
            (
                (1.0, "bah"),
                pytest.raises(
                    ValueError,
                    match="Could not convert the regularizer strength and regularizer ratio",
                ),
            ),
            (
                (1.0, 0.5, 0.1),
                pytest.raises(
                    ValueError,
                    match="strength must be a tuple of two floats",
                ),
            ),
        ],
    )
    def test_regularizer_ratio_setter(self, regularizer_strength, expectation):
        """Test that the regularizer ratio setter works as expected."""
        regularizer = self.cls()
        with expectation:
            nmo.glm.GLM(
                regularizer=regularizer, regularizer_strength=regularizer_strength
            )

    def test_get_params(self):
        """Test get_params() returns expected values."""
        regularizer = self.cls()

        assert regularizer.get_params() == {}

    @pytest.mark.parametrize("loss", [lambda a, b, c: 0, 1, None, {}])
    def test_loss_callable(self, loss):
        """Test that the loss function is a callable"""
        raise_exception = not callable(loss)
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer, regularizer_strength=(1, 0.5))
        model._compute_loss = loss
        if raise_exception:
            with pytest.raises(TypeError, match="The `loss` must be a Callable"):
                nmo.utils.assert_is_callable(model._compute_loss, "loss")
        else:
            nmo.utils.assert_is_callable(model._compute_loss, "loss")

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    def test_run_solver(self, solver_name, poissonGLM_model_instantiation):
        """Test that the solver runs."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        model.set_params(regularizer=self.cls(), regularizer_strength=(1, 0.5))
        model.solver_name = solver_name
        params = GLMParams(true_params.coef * 0.0, true_params.intercept)
        runner = model._instantiate_solver(model._compute_loss, params).solver_run
        runner(params, X, y)

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    def test_run_solver_tree(self, solver_name, poissonGLM_model_instantiation_pytree):
        """Test that the solver runs."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation_pytree

        # set regularizer and solver name
        model.set_params(regularizer=self.cls(), regularizer_strength=(1, 0.5))
        model.solver_name = solver_name
        params = GLMParams(
            jax.tree_util.tree_map(jnp.zeros_like, true_params.coef),
            true_params.intercept,
        )
        runner = model._instantiate_solver(model._compute_loss, params).solver_run
        runner(
            params,
            X.data,
            y,
        )

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    @pytest.mark.parametrize("reg_strength", [1.0, 0.5, 0.1])
    @pytest.mark.parametrize("reg_ratio", [1.0, 0.5, 0.2])
    @pytest.mark.requires_x64
    @pytest.mark.filterwarnings("ignore:The fit did not converge:RuntimeWarning")
    def test_solver_match_statsmodels(
        self, solver_name, reg_strength, reg_ratio, poissonGLM_model_instantiation
    ):
        """Test that different solvers converge to the same solution."""
        # with jax.disable_jit():
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        # set precision to float64 for accurate matching of the results
        model.data_type = jnp.float64
        model.set_params(
            regularizer=self.cls(), regularizer_strength=(reg_strength, reg_ratio)
        )
        model.solver_name = solver_name
        model.solver_kwargs = {"tol": 10**-12, "maxiter": 10000}

        init_params = GLMParams(true_params.coef * 0.0, true_params.intercept)
        runner = model._instantiate_solver(model._compute_loss, init_params).solver_run
        params = runner(init_params, X, y)[0]

        model.fit(X, y)
        # instantiate the glm with statsmodels
        glm_sm = sm.GLM(endog=y, exog=sm.add_constant(X), family=sm.families.Poisson())

        # regularize everything except intercept
        alpha_sm = np.ones(X.shape[1] + 1) * reg_strength
        alpha_sm[0] = 0

        # pure lasso = elastic net with L1 weight = 1
        res_sm = glm_sm.fit_regularized(
            method="elastic_net",
            alpha=alpha_sm,
            L1_wt=reg_ratio,
            cnvrg_tol=10**-12,
            zero_tol=1e-1000,
            maxiter=10000,
        )
        # compare params
        sm_params = res_sm.params
        glm_params = jnp.hstack((params.intercept, params.coef.flatten()))
        assert np.allclose(sm_params, glm_params)

    @pytest.mark.requires_x64
    @pytest.mark.filterwarnings("ignore:The fit did not converge:RuntimeWarning")
    def test_loss_convergence(self):
        """Test that penalized loss converges to the same value as statsmodels and the proximal operator."""
        # generate toy data
        np.random.seed(123)
        num_samples, num_features = 1000, 5
        X = np.random.normal(size=(num_samples, num_features))  # design matrix
        w = list(np.random.normal(size=(num_features,)))  # define some weights
        y = np.random.poisson(np.exp(X.dot(w)))  # observed counts

        # instantiate and fit GLM with ProximalGradient
        model_PG = nmo.glm.GLM(
            regularizer="ElasticNet",
            regularizer_strength=(1.0, 0.5),
            solver_name="ProximalGradient",
            solver_kwargs=dict(tol=10**-12, maxiter=10000),
        )
        model_PG.fit(X, y)
        glm_res = np.hstack((model_PG.intercept_, model_PG.coef_))

        # use the penalized loss function to solve optimization via Nelder-Mead
        penalized_loss = lambda p, x, y: model_PG.regularizer.penalized_loss(
            model_PG._compute_loss, model_PG.regularizer_strength, init_params=None
        )(
            GLMParams(
                p[1:],
                p[0].reshape(
                    1,
                ),
            ),
            x,
            y,
        )
        res = minimize(
            penalized_loss,
            [0] + w,
            args=(X, y),
            method="Nelder-Mead",
            tol=10**-12,
            options={"maxiter": 10000},
        )
        # regularize everything except intercept
        alpha_sm = np.ones(X.shape[1] + 1) * model_PG.regularizer_strength[0]
        alpha_sm[0] = 0

        # elastic net with
        glm_sm = sm.GLM(endog=y, exog=sm.add_constant(X), family=sm.families.Poisson())
        res_sm = glm_sm.fit_regularized(
            method="elastic_net",
            alpha=alpha_sm,
            L1_wt=0.5,
            cnvrg_tol=10**-12,
            zero_tol=1e-1000,
            maxiter=10000,
        )
        # assert weights are the same
        assert np.allclose(res.x, glm_res)
        assert np.allclose(res.x, res_sm.params)
        assert np.allclose(glm_res, res_sm.params)

    def test_elasticnet_pytree(self, poissonGLM_model_instantiation_pytree):
        """Check pytree X can be fit."""
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation_pytree
        model.set_params(
            regularizer=nmo.regularizer.ElasticNet(), regularizer_strength=1.0
        )
        model.solver_name = "ProximalGradient"
        model.fit(X, y)

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    @pytest.mark.parametrize("reg_str", [0.001, 0.01, 0.1, 1, 10])
    @pytest.mark.requires_x64
    def test_elasticnet_pytree_match(
        self,
        reg_str,
        solver_name,
        poissonGLM_model_instantiation_pytree,
        poissonGLM_model_instantiation,
    ):
        """Check pytree and array find same solution."""
        X, _, model, _, _ = poissonGLM_model_instantiation_pytree
        X_array, y, model_array, _, _ = poissonGLM_model_instantiation

        model.set_params(
            regularizer=nmo.regularizer.ElasticNet(), regularizer_strength=reg_str
        )
        model_array.set_params(
            regularizer=nmo.regularizer.ElasticNet(), regularizer_strength=reg_str
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
        "solver_name, expectation",
        [
            ("GradientDescent", pytest.raises(ValueError, match="not allowed for")),
            ("BFGS", pytest.raises(ValueError, match="not allowed for")),
            ("ProximalGradient", does_not_raise()),
            (
                "AGradientDescent",
                pytest.raises(
                    ValueError,
                    match="The solver: AGradientDescent is not allowed for",
                ),
            ),
            (1, pytest.raises(TypeError, match="solver_name must be a string")),
            ("SVRG", pytest.raises(ValueError, match="not allowed for")),
            ("ProxSVRG", does_not_raise()),
        ],
    )
    def test_init_solver_name(self, solver_name, expectation):
        """Test GroupLasso acceptable solvers."""
        # create a valid mask
        mask = np.zeros((2, 10))
        mask[0, :5] = 1
        mask[1, 5:] = 1
        mask = jnp.asarray(mask)
        with expectation:
            nmo.glm.GLM(regularizer=self.cls(mask=mask), solver_name=solver_name)

    @pytest.mark.parametrize(
        "solver_name, expectation",
        [
            ("GradientDescent", pytest.raises(ValueError, match="not allowed for")),
            ("BFGS", pytest.raises(ValueError, match="not allowed for")),
            ("ProximalGradient", does_not_raise()),
            (
                "AGradientDescent",
                pytest.raises(
                    ValueError,
                    match="The solver: AGradientDescent is not allowed for",
                ),
            ),
            (1, pytest.raises(TypeError, match="solver_name must be a string")),
            ("SVRG", pytest.raises(ValueError, match="not allowed for")),
            ("ProxSVRG", does_not_raise()),
        ],
    )
    def test_set_solver_name_allowed(self, solver_name, expectation):
        """Test GroupLassoSolver acceptable solvers."""
        # create a valid mask
        mask = np.zeros((2, 10))
        mask[0, :5] = 1
        mask[1, 5:] = 1
        mask = jnp.asarray(mask)
        regularizer = self.cls(mask=mask)
        model = nmo.glm.GLM(regularizer=regularizer, regularizer_strength=1)
        with expectation:
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
        # if no strength given, should set to 1.0
        regularizer = self.cls()
        model = nmo.glm.GLM(regularizer=regularizer)

        assert model.regularizer_strength == 1.0

        # if changed to regularized, should go to None
        model.set_params(regularizer="UnRegularized", regularizer_strength=None)
        assert model.regularizer_strength is None

        # if changed back, should set to 1.0
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
        model._compute_loss = loss

        if raise_exception:
            with pytest.raises(TypeError, match="The `loss` must be a Callable"):
                nmo.utils.assert_is_callable(model._compute_loss, "loss")
        else:
            nmo.utils.assert_is_callable(model._compute_loss, "loss")

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    def test_run_solver(self, solver_name, poissonGLM_model_instantiation):
        """Test that the solver runs."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        # create a valid mask with new PyTree structure
        mask = np.zeros((2, X.shape[1]))
        mask[0, :2] = 1
        mask[1, 2:] = 1
        mask = GLMParams(jnp.asarray(mask), None)

        model.set_params(regularizer=self.cls(mask=mask), regularizer_strength=1.0)
        model.solver_name = solver_name

        init_params = GLMParams(true_params.coef * 0.0, true_params.intercept)
        model._instantiate_solver(model._compute_loss, init_params)
        model.solver_run(init_params, X, y)

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    def test_init_solver(self, solver_name, poissonGLM_model_instantiation):
        """Test that the solver initialization returns a state."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        # create a valid mask with new PyTree structure
        mask = np.zeros((2, X.shape[1]))
        mask[0, :2] = 1
        mask[1, 2:] = 1
        mask = GLMParams(jnp.asarray(mask), None)

        model.set_params(regularizer=self.cls(mask=mask), regularizer_strength=1.0)
        model.solver_name = solver_name

        model._instantiate_solver(model._compute_loss, true_params)
        state = model.solver_init_state(true_params, X, y)
        # asses that state is a NamedTuple by checking tuple type and the availability of some NamedTuple
        # specific namespace attributes
        assert isinstance(state, tuple | eqx.Module)

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    def test_update_solver(self, solver_name, poissonGLM_model_instantiation):
        """Test that the solver initialization returns a state."""

        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        # create a valid mask with new PyTree structure
        mask = np.zeros((2, X.shape[1]))
        mask[0, :2] = 1
        mask[1, 2:] = 1
        mask = GLMParams(jnp.asarray(mask), None)

        model.set_params(regularizer=self.cls(mask=mask), regularizer_strength=1.0)
        model.solver_name = solver_name

        init_params = GLMParams(true_params.coef * 0.0, true_params.intercept)
        model._instantiate_solver(model._compute_loss, init_params)

        state = model.solver_init_state(init_params, X, y)

        # ProxSVRG needs the full gradient at the anchor point to be initialized
        # so here just set it to xs, which is not correct, but fine shape-wise
        if solver_name == "ProxSVRG":
            state = state._replace(full_grad_at_reference_point=state.reference_point)

        params, state, _ = model.solver_update(true_params, state, X, y)
        # asses that state is a NamedTuple by checking tuple type and the availability of some NamedTuple
        # specific namespace attributes
        assert isinstance(state, tuple | eqx.Module)

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
        self, n_groups_assign, poissonGLM_model_instantiation_group_sparse
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
        ) = poissonGLM_model_instantiation_group_sparse

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
                ValueError, match="Incorrect group assignment. Some of the features"
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
        """Test that mask works with PyTree structure."""

        # With PyTree masks, we need proper structure
        raise_exception = n_dim in [0, 1]
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        # create masks with different dimensions
        if n_dim == 0:
            mask = np.array([])
            mask = jnp.asarray(mask, dtype=jnp.float32)
        elif n_dim == 1:
            mask = np.ones((1,))
            mask = jnp.asarray(mask, dtype=jnp.float32)
        elif n_dim == 2:
            # Valid PyTree mask structure
            mask = np.zeros((2, X.shape[1]))
            mask[0, :2] = 1
            mask[1, 2:] = 1
            mask = GLMParams(jnp.asarray(mask, dtype=jnp.float32), None)
        else:
            # 3D mask needs to be wrapped in PyTree
            mask = np.zeros((2, X.shape[1]) + (1,) * (n_dim - 2))
            mask[0, :2] = 1
            mask[1, 2:] = 1
            mask = jnp.asarray(mask, dtype=jnp.float32)

        if raise_exception:
            with pytest.raises(ValueError):
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

        # create a mask with PyTree structure
        mask_array = np.zeros((n_groups, X.shape[1]))
        if n_groups > 0:
            for i in range(n_groups - 1):
                mask_array[i, i : i + 1] = 1
            mask_array[-1, n_groups - 1 :] = 1

        mask = GLMParams(jnp.asarray(mask_array, dtype=jnp.float32), None)

        if raise_exception:
            with pytest.raises(ValueError, match=r"Empty mask provided!"):
                model.set_params(
                    regularizer=self.cls(mask=mask), regularizer_strength=1.0
                )
        else:
            model.set_params(regularizer=self.cls(mask=mask), regularizer_strength=1.0)

    def test_group_sparsity_enforcement(
        self, poissonGLM_model_instantiation_group_sparse
    ):
        """Test that group lasso works on a simple dataset."""
        (
            X,
            y,
            model,
            true_params,
            firing_rate,
            _,
        ) = poissonGLM_model_instantiation_group_sparse
        zeros_true = true_params.coef.flatten() == 0
        mask_array = np.zeros((2, X.shape[1]))
        mask_array[0, zeros_true] = 1
        mask_array[1, ~zeros_true] = 1
        mask = GLMParams(jnp.asarray(mask_array, dtype=jnp.float32), None)

        model.set_params(regularizer=self.cls(mask=mask), regularizer_strength=1.0)
        model.solver_name = "ProximalGradient"

        init_params = GLMParams(true_params.coef * 0.0, true_params.intercept)
        runner = model._instantiate_solver(model._compute_loss, init_params).solver_run
        params, _, _ = runner(init_params, X, y)

        zeros_est = params.coef == 0
        if not np.all(zeros_est == zeros_true):
            raise ValueError("GroupLasso failed to zero-out the parameter group!")

    ###########
    # Test mask from set_params
    ###########
    @pytest.mark.parametrize("n_groups_assign", [0, 1, 2])
    def test_mask_validity_groups_set_params(
        self, n_groups_assign, poissonGLM_model_instantiation_group_sparse
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
        ) = poissonGLM_model_instantiation_group_sparse

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
                ValueError, match="Incorrect group assignment. Some of the features"
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
        """Test that mask works with PyTree structure."""

        raise_exception = n_dim in [0, 1]
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        valid_mask_array = np.zeros((2, X.shape[1]))
        valid_mask_array[0, :1] = 1
        valid_mask_array[1, 1:] = 1
        valid_mask = GLMParams(jnp.asarray(valid_mask_array, dtype=jnp.float32), None)
        regularizer = self.cls(mask=valid_mask)

        # create masks with different dimensions
        if n_dim == 0:
            mask = np.array([])
            mask = jnp.asarray(mask, dtype=jnp.float32)
        elif n_dim == 1:
            mask = np.ones((1,))
            mask = jnp.asarray(mask, dtype=jnp.float32)
        elif n_dim == 2:
            mask_array = np.zeros((2, X.shape[1]))
            mask_array[0, :2] = 1
            mask_array[1, 2:] = 1
            mask = GLMParams(jnp.asarray(mask_array, dtype=jnp.float32), None)
        else:
            mask = np.zeros((2, X.shape[1]) + (1,) * (n_dim - 2))
            mask[0, :2] = 1
            mask[1, 2:] = 1
            mask = jnp.asarray(mask, dtype=jnp.float32)

        if raise_exception:
            with pytest.raises(ValueError):
                regularizer.set_params(mask=mask)
        else:
            regularizer.set_params(mask=mask)

    @pytest.mark.parametrize("n_groups", [0, 1, 2])
    def test_mask_n_groups_set_params(self, n_groups, poissonGLM_model_instantiation):
        """Test that mask has at least 1 group."""
        raise_exception = n_groups < 1
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        valid_mask_array = np.zeros((2, X.shape[1]))
        valid_mask_array[0, :1] = 1
        valid_mask_array[1, 1:] = 1
        valid_mask = GLMParams(jnp.asarray(valid_mask_array, dtype=jnp.float32), None)
        regularizer = self.cls(mask=valid_mask)

        # create a mask with PyTree structure
        mask_array = np.zeros((n_groups, X.shape[1]))
        if n_groups > 0:
            for i in range(n_groups - 1):
                mask_array[i, i : i + 1] = 1
            mask_array[-1, n_groups - 1 :] = 1

        mask = GLMParams(jnp.asarray(mask_array, dtype=jnp.float32), None)

        if raise_exception:
            with pytest.raises(ValueError, match=r"Empty mask provided!"):
                regularizer.set_params(mask=mask)
        else:
            regularizer.set_params(mask=mask)

    def test_mask_none(self, poissonGLM_model_instantiation):
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        # Test with auto-initialized mask (mask=None, initialized during fit)
        model.regularizer = self.cls(mask=None)
        model.solver_name = "ProximalGradient"
        model.fit(X, y)

    @pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
    def test_solver_combination(self, solver_name, poissonGLM_model_instantiation):
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
        mask_array = np.ones((1, X.shape[1])).astype(float)
        mask = GLMParams(jnp.asarray(mask_array), None)
        model.set_params(
            regularizer=self.cls(mask=mask),
            regularizer_strength=(
                None if self.cls == nmo.regularizer.UnRegularized else 1.0
            ),
        )
        model.solver_name = solver_name
        model.fit(X, y)

    @pytest.mark.parametrize(
        "params_factory,expected_type,check_mask_fn",
        [
            # GLMParams single neuron (with regularizable_subtrees)
            (
                lambda: GLMParams(coef=jnp.ones((10, 3)), intercept=jnp.zeros(3)),
                GLMParams,
                lambda mask: (
                    mask.coef is not None
                    and mask.intercept is None
                    and mask.coef.ndim == 3
                    and mask.coef.shape[1:] == (10, 3)
                ),
            ),
            # Plain dict (without regularizable_subtrees)
            (
                lambda: {"spatial": jnp.ones((5, 2)), "temporal": jnp.ones((3, 2))},
                dict,
                lambda mask: (
                    "spatial" in mask
                    and "temporal" in mask
                    and mask["spatial"].ndim == 3
                    and mask["spatial"].shape[1:] == (5, 2)
                    and mask["temporal"].ndim == 3
                    and mask["temporal"].shape[1:] == (3, 2)
                ),
            ),
            # GLMParams multi-neuron (PopulationGLM case)
            (
                lambda: GLMParams(coef=jnp.ones((10, 5)), intercept=jnp.zeros(5)),
                GLMParams,
                lambda mask: (
                    mask.coef is not None
                    and mask.intercept is None
                    and mask.coef.ndim == 3
                    and mask.coef.shape[1:] == (10, 5)
                    and mask.coef.shape[0] == 5  # 5 groups (one per neuron)
                ),
            ),
        ],
    )
    def test_initialize_mask_different_structures(
        self, params_factory, expected_type, check_mask_fn
    ):
        """Test mask initialization for different parameter structures."""
        params = params_factory()
        regularizer = self.cls(mask=None)
        mask = regularizer.initialize_mask(params)

        # Check mask has expected type
        assert isinstance(mask, expected_type)

        # Check structure-specific properties
        assert check_mask_fn(mask)

    def test_apply_operator_dict_structure(self):
        """Test apply_operator with dict-based PyTree parameters."""
        from nemos.regularizer import apply_operator

        # Define a simple operation that doubles values
        def double_func(x):
            return jax.tree_util.tree_map(lambda a: a * 2, x)

        # Test with dict structure (no regularizable_subtrees)
        params = {
            "coef": jnp.ones((5,)),
            "bias": jnp.zeros((1,)),
        }

        result = apply_operator(double_func, params)

        # Check structure preserved
        assert isinstance(result, dict)
        assert set(result.keys()) == {"coef", "bias"}

        # Check operation was applied
        assert jnp.allclose(result["coef"], jnp.ones((5,)) * 2)
        assert jnp.allclose(result["bias"], jnp.zeros((1,)) * 2)

    def test_apply_operator_with_filter_kwargs(self):
        """Test apply_operator with filter_kwargs for routing masks."""
        from nemos.regularizer import apply_operator

        # Create GLMParams with regularizable_subtrees
        params = GLMParams(
            coef=jnp.ones((5,)),
            intercept=jnp.zeros((1,)),
        )

        # Create mask with matching structure
        mask = GLMParams(
            coef=jnp.array([[1, 1, 0, 0, 0], [0, 0, 1, 1, 1]], dtype=float),
            intercept=None,
        )

        # Define a function that uses the mask to check it's passed correctly
        def masked_operation(x, mask=None):
            # Return a marker value to verify mask is passed/not passed
            if mask is None:
                return x * 0  # Return zeros if no mask
            else:
                return x * 2  # Return doubled if mask is present

        result = apply_operator(masked_operation, params, filter_kwargs={"mask": mask})

        # Check that mask was correctly routed to coef but not intercept
        # coef should be doubled (mask was passed)
        assert jnp.allclose(result.coef, params.coef * 2)
        # intercept should be zeros (no mask, returned x * 0)
        assert jnp.allclose(result.intercept, jnp.zeros((1,)))

    def test_penalized_loss_dict_structure(self, poissonGLM_model_instantiation):
        """Test penalized_loss with dict-based PyTree parameters."""
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        # Create dict-based mask (simulating FeaturePytree structure)
        # Split features into two groups
        n_features = X.shape[1]
        mask_dict = {
            "group1": jnp.array([[1] * (n_features // 2)], dtype=float),
            "group2": jnp.array([[1] * (n_features - n_features // 2)], dtype=float),
        }

        # Note: For this test we're just checking that the method can be called
        # with dict structure, not testing actual GLM fitting
        regularizer = self.cls(mask=mask_dict)

        # Create matching dict params
        params_dict = {
            "group1": jnp.ones((n_features // 2,)),
            "group2": jnp.ones((n_features - n_features // 2,)),
        }

        # Test that penalization doesn't crash with dict structure
        filter_kwargs = regularizer._get_filter_kwargs(params_dict)
        penalty = regularizer._penalization(
            params_dict, strength=0.1, filter_kwargs=filter_kwargs
        )

        # Check penalty is a scalar and non-negative
        assert isinstance(penalty, jnp.ndarray)
        assert penalty.ndim == 0
        assert penalty >= 0

    def test_penalized_loss_glmparams_structure(self, poissonGLM_model_instantiation):
        """Test penalized_loss with GLMParams structure."""
        X, y, model, true_params, firing_rate = poissonGLM_model_instantiation

        # Create GLMParams mask
        n_features = X.shape[1]
        mask_array = np.ones((2, n_features), dtype=float)
        mask_array[0, n_features // 2 :] = 0
        mask_array[1, : n_features // 2] = 0
        mask = GLMParams(jnp.asarray(mask_array), None)

        regularizer = self.cls(mask=mask)

        # Create matching GLMParams params
        params = GLMParams(
            coef=jnp.ones((n_features,)),
            intercept=jnp.zeros((1,)),
        )

        # Test that penalization works
        filter_kwargs = regularizer._get_filter_kwargs(params)
        penalty = regularizer._penalization(
            params, strength=0.1, filter_kwargs=filter_kwargs
        )

        # Check penalty is a scalar and non-negative
        assert isinstance(penalty, jnp.ndarray)
        assert penalty.ndim == 0
        assert penalty >= 0


@pytest.mark.parametrize(
    "regularizer",
    [
        nmo.regularizer.UnRegularized(),
        nmo.regularizer.Ridge(),
        nmo.regularizer.Lasso(),
        nmo.regularizer.GroupLasso(mask=GLMParams(jnp.eye(5, dtype=jnp.float32), None)),
        nmo.regularizer.ElasticNet(),
    ],
)
class TestPenalizedLossAuxiliaryVariables:
    """Test that penalized_loss correctly handles auxiliary variables."""

    def test_single_value_return(self, regularizer):
        """Test backward compatibility: loss returning single value."""

        def simple_loss(params, X, y):
            return jnp.mean((y - X @ params.coef - params.intercept) ** 2)

        # ElasticNet requires (strength, ratio) tuple
        reg_strength = (
            (0.1, 0.5) if isinstance(regularizer, nmo.regularizer.ElasticNet) else 0.1
        )
        penalized = regularizer.penalized_loss(
            simple_loss, strength=reg_strength, init_params=None
        )

        params = GLMParams(jnp.ones(5), jnp.array(0.0))
        X = jnp.ones((10, 5))
        y = jnp.ones(10)

        result = penalized(params, X, y)

        # Should return a single scalar value
        assert isinstance(result, jnp.ndarray)
        assert result.shape == ()
        assert jnp.isfinite(result)

    def test_tuple_return_with_aux(self, regularizer):
        """Test that loss returning (loss, aux) preserves auxiliary variable."""

        def loss_with_aux(params, X, y):
            predictions = X @ params.coef + params.intercept
            loss = jnp.mean((y - predictions) ** 2)
            aux = {"predictions": predictions, "mse": loss}
            return loss, aux

        # ElasticNet requires (strength, ratio) tuple
        reg_strength = (
            (0.1, 0.5) if isinstance(regularizer, nmo.regularizer.ElasticNet) else 0.1
        )
        penalized = regularizer.penalized_loss(
            loss_with_aux, strength=reg_strength, init_params=None
        )

        params = GLMParams(jnp.ones(5), jnp.array(0.0))
        X = jnp.ones((10, 5))
        y = jnp.ones(10)

        result = penalized(params, X, y)

        # Should return a tuple (penalized_loss, aux)
        assert isinstance(result, tuple)
        assert len(result) == 2

        penalized_loss_value, aux = result

        # Check that penalized loss is a scalar
        assert isinstance(penalized_loss_value, jnp.ndarray)
        assert penalized_loss_value.shape == ()
        assert jnp.isfinite(penalized_loss_value)

        # Check that auxiliary variable is preserved
        assert isinstance(aux, dict)
        assert "predictions" in aux
        assert "mse" in aux
        assert aux["predictions"].shape == (10,)

        # Check that penalized loss > original loss (penalty added)
        if not isinstance(regularizer, nmo.regularizer.UnRegularized):
            assert penalized_loss_value > aux["mse"]

    def test_invalid_tuple_single_element(self, regularizer):
        """Test that single-element tuple raises error."""

        def bad_loss(params, X, y):
            return (jnp.mean((y - X @ params.coef - params.intercept) ** 2),)

        # ElasticNet requires (strength, ratio) tuple
        reg_strength = (
            (0.1, 0.5) if isinstance(regularizer, nmo.regularizer.ElasticNet) else 0.1
        )
        penalized = regularizer.penalized_loss(
            bad_loss, strength=reg_strength, init_params=None
        )

        params = GLMParams(jnp.ones(5), jnp.array(0.0))
        X = jnp.ones((10, 5))
        y = jnp.ones(10)

        with pytest.raises(
            ValueError,
            match=r"Invalid loss function return.*returns a tuple with 1 value",
        ):
            penalized(params, X, y)

    def test_invalid_tuple_three_elements(self, regularizer):
        """Test that 3+ element tuple raises error."""

        def bad_loss(params, X, y):
            loss = jnp.mean((y - X @ params.coef - params.intercept) ** 2)
            return loss, {"aux": 1}, {"extra": 2}

        # ElasticNet requires (strength, ratio) tuple
        reg_strength = (
            (0.1, 0.5) if isinstance(regularizer, nmo.regularizer.ElasticNet) else 0.1
        )
        penalized = regularizer.penalized_loss(
            bad_loss, strength=reg_strength, init_params=None
        )

        params = GLMParams(jnp.ones(5), jnp.array(0.0))
        X = jnp.ones((10, 5))
        y = jnp.ones(10)

        with pytest.raises(
            ValueError,
            match=r"Invalid loss function return.*returns a tuple with 3 values",
        ):
            penalized(params, X, y)

    def test_penalty_correctly_added_to_loss_with_aux(self, regularizer):
        """Test that penalty is correctly added when aux variables are present."""

        def loss_with_aux(params, X, y):
            predictions = X @ params.coef + params.intercept
            loss = jnp.mean((y - predictions) ** 2)
            return loss, {"predictions": predictions}

        # Get unpenalized loss
        params = GLMParams(jnp.ones(5), jnp.array(0.0))
        X = jnp.ones((10, 5))
        y = jnp.zeros(10)

        unpenalized_loss, _ = loss_with_aux(params, X, y)

        # ElasticNet requires (strength, ratio) tuple
        reg_strength = (
            (1.0, 0.5) if isinstance(regularizer, nmo.regularizer.ElasticNet) else 1.0
        )

        # Get penalized loss
        penalized = regularizer.penalized_loss(
            loss_with_aux, strength=reg_strength, init_params=params
        )
        penalized_loss_value, aux = penalized(params, X, y)

        # Calculate expected penalty
        expected_penalty = regularizer._penalization(
            params, reg_strength, regularizer._get_filter_kwargs(params)
        )

        # Check that penalized loss = unpenalized loss + penalty
        assert jnp.isclose(penalized_loss_value, unpenalized_loss + expected_penalty)


def test_available_regularizer_match():
    """Test matching of the two regularizer lists."""
    assert set(nmo._regularizer_builder.AVAILABLE_REGULARIZERS) == set(
        nmo.regularizer.__dir__()
    )
