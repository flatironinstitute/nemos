from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import nemos as nmo
from conftest import initialize_feature_mask_for_population_glm
from nemos.glm import GLM, PopulationGLM
from nemos.glm.classifier_glm import ClassifierGLM, ClassifierPopulationGLM
from nemos.solvers._hess import (
    BlockDiagonal,
    Full,
    PositiveDefinite,
    PositiveSemiDefinite,
)
from nemos.regularizer import Ridge, UnRegularized
from nemos.solvers._abstract_solver import OptimizationInfo
from nemos.solvers._newton import Newton, NewtonState
from nemos.tree_utils import pytree_map_and_reduce

# Register every test here as solver-related
pytestmark = pytest.mark.solver_related


@pytest.mark.parametrize(
    "regr_setup",
    [
        "linear_regression",
        "ridge_regression",
        "linear_regression_tree",
        "ridge_regression_tree",
    ],
)
@pytest.mark.requires_x64
def test_newton_linear_or_ridge_regression(request, regr_setup):
    X, y, _, params, loss = request.getfixturevalue(regr_setup)

    param_init = jax.tree_util.tree_map(np.zeros_like, params)
    newton_params, state, _ = Newton(
        loss,
        regularizer=UnRegularized(),
        regularizer_strength=0.0,
        has_aux=False,
        tol=10**-12,
        init_params=param_init,
    ).run(param_init, X, y)
    assert pytree_map_and_reduce(
        lambda a, b: np.allclose(a, b, atol=10**-5, rtol=0.0),
        all,
        params,
        newton_params,
    )


@pytest.mark.parametrize(
    "regr_setup, regularizer",
    [
        ("linear_regression", UnRegularized()),
        ("ridge_regression", Ridge()),
        ("linear_regression_tree", UnRegularized()),
        ("ridge_regression_tree", Ridge()),
    ],
)
@pytest.mark.requires_x64
def test_newton_init_state_default(request, regr_setup, regularizer):
    X, y, _, params, loss = request.getfixturevalue(regr_setup)

    param_init = jax.tree_util.tree_map(np.zeros_like, params)
    newton = Newton(
        loss,
        regularizer=regularizer,
        regularizer_strength=0.5,
        has_aux=True,
        tol=10**-12,
        init_params=param_init,
    )
    state = newton.init_state(param_init, X, y)

    assert isinstance(state, NewtonState)
    assert state.grad_norm == jnp.array(jnp.inf)
    assert isinstance(state.stats, OptimizationInfo)
    assert state.stats.num_steps == 0
    assert state.stats.converged == jnp.array(False)
    assert jnp.isnan(state.stats.function_val)
    assert state.stats.converged == jnp.array(False)
    assert state.stats.reached_max_steps == jnp.array(False)
    assert isinstance(state.ls_state, optax.ScaleByBacktrackingLinesearchState)


@pytest.mark.parametrize("regularizer_name", ["Ridge", "UnRegularized"])
@pytest.mark.parametrize("glm_class", [nmo.glm.GLM, nmo.glm.PopulationGLM])
def test_newton_glm_instantiate_solver(regularizer_name, glm_class):
    glm = glm_class(
        regularizer=regularizer_name,
        solver_name="Newton",
        regularizer_strength=None if regularizer_name == "UnRegularized" else 1,
    )
    solver = glm._instantiate_solver(glm._compute_loss, np.zeros(1))

    # currently glm._solver is a Wrapped(Prox)SVRG
    assert glm.solver_name == "Newton"
    assert isinstance(solver, Newton)


@pytest.mark.parametrize("regularizer_name", ["Ridge", "UnRegularized"])
@pytest.mark.parametrize(
    "glm_class",
    [
        nmo.glm.GLM,
        nmo.glm.PopulationGLM,
        nmo.glm.ClassifierGLM,
        nmo.glm.ClassifierPopulationGLM,
    ],
)
def test_newton_glm_passes_solver_kwargs(regularizer_name, glm_class):
    solver_kwargs = {
        "maxiter": np.random.randint(1, 100),
        "jit": False,
        "tol": 1e-6,
    }

    glm = glm_class(
        regularizer=regularizer_name,
        solver_name="Newton",
        solver_kwargs=solver_kwargs,
        regularizer_strength=None if regularizer_name == "UnRegularized" else 1,
    )
    solver = glm._instantiate_solver(glm._compute_loss, np.zeros(1))

    for k, v in solver_kwargs.items():
        assert getattr(solver, k) == v


@pytest.mark.parametrize("regularizer_name", ["Ridge", "UnRegularized"])
@pytest.mark.parametrize("glm_class", [nmo.glm.GLM, nmo.glm.PopulationGLM])
def test_newton_glm_initialize_state(glm_class, regularizer_name, linear_regression):
    X, y, _, _, _ = linear_regression

    if glm_class == nmo.glm.PopulationGLM:
        y = np.expand_dims(y, 1)

    reg_cls = getattr(nmo.regularizer, regularizer_name)
    reg = reg_cls()

    glm = glm_class(
        regularizer=reg,
        solver_name="Newton",
        inverse_link_function=jax.nn.softplus,
        observation_model=nmo.observation_models.PoissonObservations(),
        regularizer_strength=None if regularizer_name == "UnRegularized" else 1,
    )

    init_params = glm.initialize_params(X, y)
    state = glm.initialize_optimizer_and_state(init_params, X, y)

    assert isinstance(state, NewtonState)
    assert state.grad_norm == jnp.array(jnp.inf)
    assert isinstance(state.stats, OptimizationInfo)
    assert state.stats.num_steps == 0
    assert state.stats.converged == jnp.array(False)
    assert jnp.isnan(state.stats.function_val)
    assert state.stats.converged == jnp.array(False)
    assert state.stats.reached_max_steps == jnp.array(False)
    assert isinstance(state.ls_state, optax.ScaleByBacktrackingLinesearchState)


@pytest.mark.requires_x64
@pytest.mark.parametrize("regularizer_name", ["Ridge", "UnRegularized"])
@pytest.mark.parametrize("structure", ["", "_pytree"])
def test_newton_glm_converges(request, regularizer_name, structure):
    """Newton-fitted GLM should converge and return finite parameters."""
    X, y, model, _, _ = request.getfixturevalue(
        "poissonGLM_model_instantiation" + structure
    )
    model.regularizer = regularizer_name
    model.regularizer_strength = 1e-3 if regularizer_name == "Ridge" else None
    model = model.fit(X, y)

    assert model.coef_ is not None
    assert model.intercept_ is not None
    assert bool(model.solver_state_.stats.converged), "Solver did not converge."


@pytest.mark.requires_x64
@pytest.mark.parametrize("regularizer_name", ["Ridge", "UnRegularized"])
@pytest.mark.parametrize("feature_mask", [True, False])
@pytest.mark.parametrize("structure", ["", "_pytree"])
def test_newton_population_glm_converges(
    request, regularizer_name, feature_mask, structure
):
    """Newton-fitted PopulationGLM should converge and return finite parameters."""
    X, y, model, params, _ = request.getfixturevalue(
        "population_poissonGLM_model_instantiation" + structure
    )
    model.regularizer = regularizer_name
    model.regularizer_strength = 1e-3 if regularizer_name == "Ridge" else None

    model = model.fit(X, y)

    assert model.coef_ is not None
    assert model.intercept_ is not None
    assert bool(model.solver_state_.stats.converged), "Solver did not converge."


@pytest.mark.requires_x64
@pytest.mark.parametrize("regularizer_name", ["Ridge", "UnRegularized"])
@pytest.mark.parametrize("structure", ["", "_pytree"])
def test_newton_classifier_glm_converges(request, regularizer_name, structure):
    """Newton-fitted ClassifierGLM should converge and return finite parameters."""
    X, y, model, _, _ = request.getfixturevalue(
        "classifierGLM_model_instantiation" + structure
    )
    model.regularizer = regularizer_name
    model.regularizer_strength = 1e-3 if regularizer_name == "Ridge" else None
    model = model.fit(X, y)

    assert model.coef_ is not None
    assert model.intercept_ is not None
    assert bool(model.solver_state_.stats.converged), "Solver did not converge."


@pytest.mark.requires_x64
@pytest.mark.parametrize("regularizer_name", ["Ridge", "UnRegularized"])
@pytest.mark.parametrize("feature_mask", [True, False])
@pytest.mark.parametrize("structure", ["", "_pytree"])
def test_newton_classifier_population_glm_converges(
    request, regularizer_name, feature_mask, structure
):
    """Newton-fitted ClassifierPopulationGLM should converge and return finite parameters."""
    X, y, model, params, _ = request.getfixturevalue(
        "population_classifierGLM_model_instantiation" + structure
    )
    model.regularizer = regularizer_name
    model.regularizer_strength = 1e-3 if regularizer_name == "Ridge" else None
    if feature_mask:
        feature_mask = initialize_feature_mask_for_population_glm(
            X, y.shape[1], coef=params.coef
        )
        if structure == "_pytree":
            feature_mask["input_1"] = np.zeros_like(feature_mask["input_1"])
        else:
            feature_mask = feature_mask.at[:, 1].set(0)
        model._feature_mask = feature_mask
    model = model.fit(X, y)

    assert model.coef_ is not None
    assert model.intercept_ is not None
    assert bool(model.solver_state_.stats.converged), "Solver did not converge."


@pytest.mark.parametrize(
    "model_instantiation_type",
    [
        "poissonGLM_model_instantiation",
        "population_poissonGLM_model_instantiation",
        "classifierGLM_model_instantiation",
        "population_classifierGLM_model_instantiation",
    ],
)
def test_solver_invalidated_after_regularizer_change(request, model_instantiation_type):
    """Changing regularizer should set _solver to None."""
    X, y, model, true_params, _ = request.getfixturevalue(model_instantiation_type)

    if "class" in model_instantiation_type:
        y = model._label_encoder.encode(y, safe=False)
        y = jax.nn.one_hot(y, model.n_classes)

    model._initialize_optimizer_and_state(true_params, X, y)
    assert model.solver is not None, "Solver should be set after initialization."

    model.regularizer = "UnRegularized"
    assert model._solver is None, "_solver must be None after regularizer change."


@pytest.mark.parametrize(
    "model_instantiation_type",
    [
        "poissonGLM_model_instantiation",
        "population_poissonGLM_model_instantiation",
        "classifierGLM_model_instantiation",
        "population_classifierGLM_model_instantiation",
    ],
)
def test_solver_invalidated_after_strength_change(request, model_instantiation_type):
    """Changing regularizer_strength should set _solver to None."""
    X, y, model, true_params, _ = request.getfixturevalue(model_instantiation_type)

    if "class" in model_instantiation_type:
        y = model._label_encoder.encode(y, safe=False)
        y = jax.nn.one_hot(y, model.n_classes)

    model._initialize_optimizer_and_state(true_params, X, y)
    assert model._solver is not None

    model.regularizer_strength = 0.5
    assert model._solver is None, (
        "_solver must be None after regularizer_strength change."
    )


@pytest.mark.parametrize(
    "model_instantiation_type",
    [
        "poissonGLM_model_instantiation",
        "population_poissonGLM_model_instantiation",
        "classifierGLM_model_instantiation",
        "population_classifierGLM_model_instantiation",
    ],
)
def test_solver_invalidated_after_observation_model_change(
    request, model_instantiation_type
):
    """Changing observation_model should set _solver to None."""
    X, y, model, true_params, _ = request.getfixturevalue(model_instantiation_type)

    if "class" in model_instantiation_type:
        y = model._label_encoder.encode(y, safe=False)
        y = jax.nn.one_hot(y, model.n_classes)

    model._initialize_optimizer_and_state(true_params, X, y)
    assert model._solver is not None

    model.observation_model = "Gaussian"
    assert model._solver is None, "_solver must be None after observation_model change."


@pytest.mark.parametrize(
    "model_instantiation_type",
    [
        "poissonGLM_model_instantiation",
        "population_poissonGLM_model_instantiation",
        "classifierGLM_model_instantiation",
        "population_classifierGLM_model_instantiation",
    ],
)
def test_solver_invalidated_after_solver_name_change(request, model_instantiation_type):
    """Changing solver_name should set _solver to None."""
    X, y, model, true_params, _ = request.getfixturevalue(model_instantiation_type)

    if "class" in model_instantiation_type:
        y = model._label_encoder.encode(y, safe=False)
        y = jax.nn.one_hot(y, model.n_classes)

    model._initialize_optimizer_and_state(true_params, X, y)
    assert model._solver is not None

    model.solver_name = "LBFGS"
    assert model._solver is None, "_solver must be None after solver_name change."


@pytest.mark.parametrize(
    "model_instantiation_type",
    [
        "poissonGLM_model_instantiation",
        "population_poissonGLM_model_instantiation",
        "classifierGLM_model_instantiation",
        "population_classifierGLM_model_instantiation",
    ],
)
def test_solver_invalidated_after_solver_kwargs_change(
    request, model_instantiation_type
):
    """Changing solver_kwargs should set _solver to None."""
    X, y, model, true_params, _ = request.getfixturevalue(model_instantiation_type)

    if "class" in model_instantiation_type:
        y = model._label_encoder.encode(y, safe=False)
        y = jax.nn.one_hot(y, model.n_classes)

    model._initialize_optimizer_and_state(true_params, X, y)
    assert model._solver is not None

    model.solver_kwargs = {"maxiter": 50}
    assert model._solver is None, "_solver must be None after solver_kwargs change."


def test_glm_hess_tag_structure():
    """GLM Hessian should have Full structure."""
    assert GLM._hess_tag.structure is Full


def test_glm_hess_tag_property_unregularized():
    """Unpenalized GLM Hessian is only positive semidefinite."""
    assert GLM._hess_tag.property is PositiveSemiDefinite


def test_population_glm_hess_tag_structure():
    """PopulationGLM Hessian should have BlockDiagonal structure."""
    assert PopulationGLM._hess_tag.structure is BlockDiagonal


def test_population_glm_hess_tag_property():
    """PopulationGLM Hessian should be PositiveDefinite at the class level."""
    assert PopulationGLM._hess_tag.property is PositiveDefinite


def test_population_glm_hess_tag_batch_axes():
    """PopulationGLM _hess_tag should carry per-neuron batch_axes as a GLMParams."""
    from nemos.glm.params import GLMParams

    tag = PopulationGLM._hess_tag
    assert tag.batch_axes is not None
    assert isinstance(tag.batch_axes, GLMParams)


@pytest.mark.parametrize("glm_class", [GLM, PopulationGLM])
def test_hess_property_override(glm_class):
    """Ridge-penalized GLM should override Hessian property to PositiveDefinite."""
    model = glm_class(regularizer="Ridge", regularizer_strength=0.1)
    assert model._hess_property_override() is PositiveDefinite


@pytest.mark.parametrize("glm_class", [ClassifierGLM, ClassifierPopulationGLM])
def test_hess_property_override_classification(glm_class):
    """Ridge-penalized GLM should override Hessian property to PositiveDefinite."""
    model = glm_class(regularizer="Ridge", regularizer_strength=0.1)
    assert model._hess_property_override() is None


@pytest.mark.parametrize(
    "regularizer_name",
    ["UnRegularized", "Lasso", "GroupLasso"],
)
def test_hess_property_override_non_ridge(regularizer_name):
    """Non-Ridge GLMs should return None from _hess_property_override."""
    strength = 0.1 if regularizer_name != "UnRegularized" else None
    model = GLM(regularizer=regularizer_name, regularizer_strength=strength)
    assert model._hess_property_override() is None


@pytest.mark.parametrize(
    "glm_class", [GLM, PopulationGLM, ClassifierGLM, ClassifierPopulationGLM]
)
def test_default_solver_is_newton_for_ridge(glm_class):
    """Ridge-penalized GLM should default to Newton solver."""
    model = glm_class(regularizer="Ridge", regularizer_strength=0.1)
    assert model.solver_name == "Newton"


@pytest.mark.parametrize(
    "glm_class", [GLM, PopulationGLM, ClassifierGLM, ClassifierPopulationGLM]
)
def test_default_solver_is_not_newton_for_unregularized(glm_class):
    """Unregularized GLM should NOT default to Newton."""
    model = glm_class(regularizer="UnRegularized")
    assert model.solver_name != "Newton"


@pytest.mark.parametrize(
    "glm_class", [GLM, PopulationGLM, ClassifierGLM, ClassifierPopulationGLM]
)
def test_solver_name_respected_when_explicitly_set(glm_class):
    """Explicitly setting solver_name='Newton' should be respected."""
    model = glm_class(regularizer="UnRegularized", solver_name="Newton")
    assert model.solver_name == "Newton"


@pytest.mark.parametrize(
    "model_instantiation_type",
    [
        "poissonGLM_model_instantiation",
        "population_poissonGLM_model_instantiation",
        "classifierGLM_model_instantiation",
        "population_classifierGLM_model_instantiation",
    ],
)
def test_newton_solver_type_after_fit(request, model_instantiation_type):
    """After fit(), model._solver should be a Newton instance."""
    X, y, model, _, _ = request.getfixturevalue(model_instantiation_type)
    model.regularizer = "Ridge"
    model.fit(X, y)
    from nemos.solvers._newton import Newton

    assert isinstance(model._solver, Newton)


@pytest.mark.parametrize(
    "model_instantiation_type",
    [
        "poissonGLM_model_instantiation",
        "population_poissonGLM_model_instantiation",
        "classifierGLM_model_instantiation",
        "population_classifierGLM_model_instantiation",
    ],
)
def test_newton_update_increments_step_count(request, model_instantiation_type):
    """Each call to update() should increment the step counter by exactly 1."""
    X, y, model, _, _ = request.getfixturevalue(model_instantiation_type)
    init_params = model.initialize_params(X, y)
    state = model.initialize_optimizer_and_state(init_params, X, y)
    assert state.stats.num_steps == 0

    _, state1 = model.update(init_params, state, X, y)
    assert state1.stats.num_steps == 1

    _, state2 = model.update(model.get_model_params(), state1, X, y)
    assert state2.stats.num_steps == 2


@pytest.mark.parametrize(
    "model_instantiation_type",
    [
        "poissonGLM_model_instantiation",
        "population_poissonGLM_model_instantiation",
        "classifierGLM_model_instantiation",
        "population_classifierGLM_model_instantiation",
    ],
)
def test_newton_maxiter_respected(request, model_instantiation_type):
    """Setting maxiter=1 should bound the solver to at most 1 step."""
    X, y, model, _, _ = request.getfixturevalue(model_instantiation_type)
    model.regularizer = "Ridge"
    model.solver_kwargs = {"maxiter": 1}
    model.fit(X, y)

    n_steps = int(model.solver_state_.stats.num_steps)
    assert n_steps <= 1, f"Expected at most 1 step, got {n_steps}"


@pytest.mark.parametrize(
    "glm_class", [GLM, PopulationGLM, ClassifierGLM, ClassifierPopulationGLM]
)
def test_newton_invalid_kwarg_raises(glm_class):
    """Passing an unrecognised kwarg should raise a NameError immediately."""
    with pytest.raises(NameError, match="not a kwarg"):
        glm_class(
            regularizer="Ridge",
            regularizer_strength=0.1,
            solver_name="Newton",
            solver_kwargs={"totally_fake_kwarg": 99},
        )
