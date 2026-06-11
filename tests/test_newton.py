from contextlib import nullcontext as does_not_raise

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import nemos as nmo
from nemos.glm.params import GLMParams
from nemos.solvers._abstract_solver import OptimizationInfo
from nemos.solvers._newton import Newton, NewtonState, _Newton
from nemos.tree_utils import pytree_map_and_reduce

# Register every test here as solver-related
pytestmark = pytest.mark.solver_related


@pytest.mark.parametrize(
    ("regr_setup", "stepsize"),
    [
        ("linear_regression", 1e-3),
        ("ridge_regression", 1e-4),
        ("linear_regression_tree", 1e-4),
        ("ridge_regression_tree", 1e-4),
    ],
)
@pytest.mark.requires_x64
def test_newton_linear_or_ridge_regression(request, regr_setup, stepsize):
    X, y, _, params, loss = request.getfixturevalue(regr_setup)

    param_init = jax.tree_util.tree_map(np.zeros_like, params)
    newton_params, state, _ = _Newton(
        loss, lambda p, *a: (loss(p, *a), None), tol=10**-12
    ).run(param_init, X, y)
    assert pytree_map_and_reduce(
        lambda a, b: np.allclose(a, b, atol=10**-5, rtol=0.0),
        all,
        params,
        newton_params,
    )


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
def test_newton_init_state_default(request, regr_setup):
    X, y, _, params, loss = request.getfixturevalue(regr_setup)

    param_init = jax.tree_util.tree_map(np.zeros_like, params)
    newton = _Newton(loss, lambda p, *a: (loss(p, *a), None))
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
    assert isinstance(solver._solver, _Newton)


@pytest.mark.parametrize("regularizer_name", ["Ridge", "UnRegularized"])
@pytest.mark.parametrize("glm_class", [nmo.glm.GLM, nmo.glm.PopulationGLM])
def test_newton_glm_passes_solver_kwargs(regularizer_name, glm_class):
    solver_kwargs = {
        "maxiter": np.random.randint(1, 100),
        "jit": False,
        "autodiff": True,
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


@pytest.mark.parametrize("regularizer_name", ["Ridge", "UnRegularized"])
@pytest.mark.parametrize("glm_class", [nmo.glm.GLM, nmo.glm.PopulationGLM])
def test_newton_glm_initialize_hessian(glm_class, regularizer_name, linear_regression):
    X, y, _, _, loss = linear_regression
    if glm_class == nmo.glm.PopulationGLM:
        y = np.expand_dims(y, 1)

    glm = glm_class(regularizer=regularizer_name, solver_name="Newton")
    params = glm.initialize_params(X, y)
    glm.initialize_optimizer_and_state(params, X, y)
    params_tree = GLMParams(*params)

    init = glm.solver._solver._hess_fn(params_tree, X, y)
    expected = glm._get_hess_fn(
        params_tree,
    )(params_tree, X, y)
    np.testing.assert_allclose(init, expected)


@pytest.mark.parametrize("regularizer_name", ["Ridge", "UnRegularized"])
@pytest.mark.parametrize("glm_class", [nmo.glm.GLM, nmo.glm.PopulationGLM])
def test_newton_glm_update(glm_class, regularizer_name, linear_regression):
    X, y, _, _, loss = linear_regression
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
    params, state = glm.update(init_params, state, X, y)

    assert state.stats.num_steps == 1


@pytest.mark.parametrize("regularizer_before", ["Ridge", "UnRegularized"])
@pytest.mark.parametrize("regularizer_after", ["Ridge", "UnRegularized"])
@pytest.mark.parametrize("glm_class", [nmo.glm.GLM, nmo.glm.PopulationGLM])
def test_newton_glm_set_regularizer_update_invalidates(
    glm_class, regularizer_before, regularizer_after, linear_regression
):
    X, y, _, _, loss = linear_regression
    if glm_class == nmo.glm.PopulationGLM:
        y = np.expand_dims(y, 1)

    glm = glm_class(regularizer=regularizer_before, solver_name="Newton")
    params = glm.initialize_params(X, y)
    init_state = glm.initialize_optimizer_and_state(params, X, y)

    # Update regularizer
    glm.regularizer = regularizer_after

    # Verify invalidated solver
    assert glm._solver is None
    assert glm._solver_loss_fun is None
    assert glm._optimizer_init_state is None
    assert glm._optimizer_update is None
    assert glm._optimizer_run is None
    with pytest.raises(
        RuntimeError, match="Attempt at update when solver was in invalid state."
    ):
        glm.update(params, init_state, X, y)


@pytest.mark.parametrize("regularizer_before", ["Ridge", "UnRegularized"])
@pytest.mark.parametrize("regularizer_after", ["Ridge", "UnRegularized"])
@pytest.mark.parametrize("glm_class", [nmo.glm.GLM, nmo.glm.PopulationGLM])
def test_newton_glm_set_regularizer_update_recovers(
    glm_class, regularizer_before, regularizer_after, linear_regression
):
    X, y, _, _, loss = linear_regression
    if glm_class == nmo.glm.PopulationGLM:
        y = np.expand_dims(y, 1)

    glm = glm_class(regularizer=regularizer_before, solver_name="Newton")
    params = glm.initialize_params(X, y)
    init_state = glm.initialize_optimizer_and_state(params, X, y)
    params_tree = GLMParams(*params)
    init = glm.solver._solver._hess_fn(params_tree, X, y)

    # Update regularizer
    glm.regularizer = regularizer_after

    # Verify reinitialisation fixes it
    init_state = glm.initialize_optimizer_and_state(params, X, y)
    with does_not_raise():
        glm.update(params, init_state, X, y)
    after = glm.solver._solver._hess_fn(params_tree, X, y)
    expected = glm._get_hess_fn(
        params_tree,
    )(params_tree, X, y)
    np.testing.assert_allclose(after, expected)
    if regularizer_before == regularizer_after:
        assert np.allclose(after, init)
    else:
        assert not np.allclose(after, init)


@pytest.mark.parametrize("regularizer_name", ["Ridge", "UnRegularized"])
@pytest.mark.parametrize("glm_class", [nmo.glm.GLM, nmo.glm.PopulationGLM])
def test_newton_glm_set_regularizer_strength_invalidates(
    glm_class, regularizer_name, linear_regression
):
    X, y, _, _, loss = linear_regression
    if glm_class == nmo.glm.PopulationGLM:
        y = np.expand_dims(y, 1)

    glm = glm_class(
        regularizer=regularizer_name, solver_name="Newton", regularizer_strength=0.1
    )
    params = glm.initialize_params(X, y)
    init_state = glm.initialize_optimizer_and_state(params, X, y)
    params = GLMParams(*params)

    # Update regularizer strength
    glm.regularizer_strength = 1.0

    # Verify invalidated solver
    assert glm._solver is None
    assert glm._solver_loss_fun is None
    assert glm._optimizer_init_state is None
    assert glm._optimizer_update is None
    assert glm._optimizer_run is None
    with pytest.raises(
        RuntimeError, match="Attempt at update when solver was in invalid state."
    ):
        glm.update(params, init_state, X, y)


@pytest.mark.parametrize("regularizer_name", ["Ridge", "UnRegularized"])
@pytest.mark.parametrize("glm_class", [nmo.glm.GLM, nmo.glm.PopulationGLM])
def test_newton_glm_set_regularizer_strength_recovers(
    glm_class, regularizer_name, linear_regression
):
    X, y, _, _, loss = linear_regression
    if glm_class == nmo.glm.PopulationGLM:
        y = np.expand_dims(y, 1)

    glm = glm_class(
        regularizer=regularizer_name, solver_name="Newton", regularizer_strength=0.1
    )
    params = glm.initialize_params(X, y)
    glm.initialize_optimizer_and_state(params, X, y)
    params_tree = GLMParams(*params)
    init = glm.solver._solver._hess_fn(params_tree, X, y)

    # Update regularizer strength
    glm.regularizer_strength = 1.0

    # Verify reinitialisation fixes it
    init_state = glm.initialize_optimizer_and_state(params, X, y)
    with does_not_raise():
        glm.update(params, init_state, X, y)
    after = glm.solver._solver._hess_fn(params_tree, X, y)
    expected = glm._get_hess_fn(
        params_tree,
    )(params_tree, X, y)
    np.testing.assert_allclose(after, expected)
    if regularizer_name == "Ridge":
        assert not np.allclose(after, init)
    else:
        assert np.allclose(after, init)


@pytest.mark.parametrize(
    "obs_init",
    [
        "PoissonObservations",
        "GammaObservations",
        "GaussianObservations",
        "BernoulliObservations",
    ],
)
@pytest.mark.parametrize(
    "obs_after",
    [
        "PoissonObservations",
        "GammaObservations",
        "GaussianObservations",
        "BernoulliObservations",
    ],
)
@pytest.mark.parametrize("glm_class", [nmo.glm.GLM, nmo.glm.PopulationGLM])
def test_newton_glm_set_observation_model_invalidates(
    glm_class, obs_init, obs_after, linear_regression
):
    X, y, _, _, loss = linear_regression
    if glm_class == nmo.glm.PopulationGLM:
        y = np.expand_dims(y, 1)

    glm = glm_class(solver_name="Newton", observation_model=obs_init)
    params = glm.initialize_params(X, y)
    init_state = glm.initialize_optimizer_and_state(params, X, y)

    # Update observation model
    glm.observation_model = obs_after

    # Verify invalidated solver
    assert glm._solver is None
    assert glm._solver_loss_fun is None
    assert glm._optimizer_init_state is None
    assert glm._optimizer_update is None
    assert glm._optimizer_run is None
    with pytest.raises(
        RuntimeError, match="Attempt at update when solver was in invalid state."
    ):
        glm.update(params, init_state, X, y)


@pytest.mark.parametrize(
    "obs_init",
    [
        "PoissonObservations",
        "GammaObservations",
        "GaussianObservations",
        "BernoulliObservations",
    ],
)
@pytest.mark.parametrize(
    "obs_after",
    [
        "PoissonObservations",
        "GammaObservations",
        "GaussianObservations",
        "BernoulliObservations",
    ],
)
@pytest.mark.parametrize("glm_class", [nmo.glm.GLM, nmo.glm.PopulationGLM])
def test_newton_glm_set_observation_model_recovers(
    glm_class, obs_init, obs_after, linear_regression
):
    X, y, _, _, loss = linear_regression
    if glm_class == nmo.glm.PopulationGLM:
        y = np.expand_dims(y, 1)

    glm = glm_class(solver_name="Newton", observation_model=obs_init)
    params = glm.initialize_params(X, y)
    glm.initialize_optimizer_and_state(params, X, y)
    params_tree = GLMParams(*params)
    init = glm.solver._solver._hess_fn(params_tree, X, y)

    # Update observation model
    glm.observation_model = obs_after

    # Verify reinitialisation fixes it
    init_state = glm.initialize_optimizer_and_state(params, X, y)
    with does_not_raise():
        glm.update(params, init_state, X, y)
    after = glm.solver._solver._hess_fn(params_tree, X, y)
    expected = glm._get_hess_fn(
        params_tree,
    )(params_tree, X, y)
    np.testing.assert_allclose(after, expected)

    if obs_init == obs_after:
        assert np.allclose(after, init)
    else:
        assert not np.allclose(after, init)


@pytest.mark.parametrize(
    "regularizer_strength, diag, structure",
    [
        (0.1, [0.1] * 5, ""),
        (0.1, [0.1] * 5, ""),
        ([0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], ""),
        (
            {"input_1": [0.1, 0.2, 0.3], "input_2": [0.4, 0.5]},
            [0.1, 0.2, 0.3, 0.4, 0.5],
            "_pytree",
        ),
    ],
)
@pytest.mark.parametrize(
    "model_instantiation",
    [
        "poissonGLM",
        "gammaGLM",
        "bernoulliGLM",
        "gaussianGLM",
    ],
)
def test_newton_glm_regularizer_strength(
    regularizer_strength, diag, structure, model_instantiation, request
):
    X, y, model, params, _ = request.getfixturevalue(
        model_instantiation + "_model_instantiation" + structure
    )

    # Get UnRegularized hessian
    model.solver_name = "Newton"
    H = model._get_hess_fn(params)(params, X, y)

    # Get Ridge hessian
    model.regularizer = "Ridge"
    model.regularizer_strength = regularizer_strength
    regularized_hess = model._get_hess_fn(params)(params, X, y)

    # Test not UnRegularized
    assert not np.allclose(H, regularized_hess)
    # Test Ridge
    H = H.at[:-1, :-1].add(jnp.diag(jnp.array(diag)))
    np.testing.assert_allclose(desired=H, actual=regularized_hess)


@pytest.mark.parametrize(
    "regularizer_strength, diag, structure",
    [
        (0.1, [[[0.1] * 5] * 3] * 3, ""),
        (
            [[[0.1, 0.3, 0.2]] * 3] * 5,
            [[[0.1, 0.3, 0.2]] * 5, [[0.1, 0.3, 0.2]] * 5, [[0.1, 0.3, 0.2]] * 5],
            "",
        ),
        (
            {
                "input_1": [[[0.1, 0.3, 0.2]] * 3] * 3,
                "input_2": [[[0.1, 0.3, 0.2]] * 3] * 2,
            },
            [[[0.1, 0.3, 0.2]] * 5, [[0.1, 0.3, 0.2]] * 5, [[0.1, 0.3, 0.2]] * 5],
            "_pytree",
        ),
    ],
)
def test_newton_population_classifier_glm_regularizer_strength(
    regularizer_strength, diag, structure, request
):
    X, y, model, params, _ = request.getfixturevalue(
        "population_classifierGLM_model_instantiation" + structure
    )
    y = model._label_encoder.encode(y, safe=False)
    y = jax.nn.one_hot(y, model.n_classes)

    # Get UnRegularized hessian
    model.solver_name = "Newton"
    H = model._get_hess_fn(params)(params, X, y)

    # Get Ridge hessian
    model.regularizer = "Ridge"
    model.regularizer_strength = regularizer_strength
    regularized_hess = model._get_hess_fn(params)(params, X, y)

    # Test not UnRegularized
    assert not np.allclose(H, regularized_hess)
    # Test Ridge
    diag = jnp.stack([jnp.diag(jnp.array(d).ravel()) for d in diag], axis=0)
    H = H.at[:, :-3, :-3].add(diag)
    np.testing.assert_allclose(desired=H, actual=regularized_hess)


@pytest.mark.parametrize("regularizer ", ["UnRegularized", "Ridge"])
@pytest.mark.parametrize(
    "regularizer_strength, structure",
    [
        (0.1, ""),
        (
            [[[0.1, 0.3, 0.2]] * 3] * 5,
            "",
        ),
        (
            {
                "input_1": [[0.1, 0.3, 0.2]] * 3,
                "input_2": [[0.1, 0.3, 0.2]] * 2,
            },
            "_pytree",
        ),
    ],
)
def test_newton_classifier_glm_regularizer_strength(
    regularizer, regularizer_strength, structure, request
):
    X, y, model, _, _ = request.getfixturevalue(
        "classifierGLM_model_instantiation" + structure
    )
    model.regularizer = regularizer
    model.regularizer_strenth = regularizer_strength

    with does_not_raise():
        model.fit(X, y)

    # ClassifierGLM always uses autodiff so we do not check the hessian


@pytest.mark.parametrize(
    "regularizer_strength, diag, structure",
    [
        (0.1, [[0.1] * 5] * 3, ""),
        (
            [[0.1, 0.3, 0.2]] * 5,
            [[0.1] * 5, [0.3] * 5, [0.2] * 5],
            "",
        ),
        (
            {
                "input_1": [[0.1, 0.3, 0.2], [0.2, 0.2, 0.2], [0.3, 0.1, 0.1]],
                "input_2": [[0.1, 0.3, 0.2], [0.2, 0.2, 0.2]],
            },
            [
                [0.1, 0.2, 0.3, 0.1, 0.2],
                [0.3, 0.2, 0.1, 0.3, 0.2],
                [0.2, 0.2, 0.1, 0.2, 0.2],
            ],
            "_pytree",
        ),
    ],
)
@pytest.mark.parametrize(
    "model_instantiation",
    [
        "_poissonGLM",
        "_gammaGLM",
        "_bernoulliGLM",
        "_gaussianGLM",
    ],
)
def test_newton_population_glm_regularizer_strength(
    regularizer_strength, diag, model_instantiation, structure, request
):
    X, y, model, params, _ = request.getfixturevalue(
        "population" + model_instantiation + "_model_instantiation" + structure
    )

    # Get UnRegularized hessian
    model.solver_name = "Newton"
    H = model._get_hess_fn(params)(params, X, y)

    # Get Ridge hessian
    model.regularizer = "Ridge"
    model.regularizer_strength = regularizer_strength
    regularized_hess = model._get_hess_fn(params)(params, X, y)

    # Test not UnRegularized
    assert not np.allclose(H, regularized_hess)
    # Test Ridge
    diag = jnp.stack([jnp.diag(jnp.array(d)) for d in diag], axis=0)
    H = H.at[:, :-1, :-1].add(diag)
    np.testing.assert_allclose(desired=H, actual=regularized_hess)


@pytest.mark.parametrize(
    "regularizer_strength, structure",
    [
        (0.1, ""),
        (
            [[0.1, 0.3, 0.2]] * 5,
            "",
        ),
        (
            {
                "input_1": [[0.1, 0.3, 0.2], [0.2, 0.2, 0.2], [0.3, 0.1, 0.1]],
                "input_2": [[0.1, 0.3, 0.2], [0.2, 0.2, 0.2]],
            },
            "_pytree",
        ),
    ],
)
@pytest.mark.parametrize(
    "regularizer",
    [
        "UnRegularized",
        "Ridge",
    ],
)
@pytest.mark.parametrize(
    "model_instantiation",
    [
        "_poissonGLM",
        "_gammaGLM",
        "_bernoulliGLM",
        "_gaussianGLM",
    ],
)
def test_newton_population_glm_analytic_v_autodiff(
    regularizer, regularizer_strength, model_instantiation, structure, request
):
    X, y, model, params, _ = request.getfixturevalue(
        "population" + model_instantiation + "_model_instantiation" + structure
    )
    model.solver_name = "Newton"
    model.regularizer = regularizer
    model.regularizer_strength = regularizer_strength

    autodiff = model._get_hess_fn(params, autodiff=True)(params, X, y)
    analytic = model._get_hess_fn(params, autodiff=False)(params, X, y)

    np.testing.assert_allclose(desired=autodiff, actual=analytic, atol=0.00001)


@pytest.mark.parametrize(
    "regularizer_strength, structure",
    [
        (0.1, ""),
        (
            [0.1, 0.3, 0.2, 0.4, 0.3],
            "",
        ),
        (
            {"input_1": [0.1, 0.3, 0.2], "input_2": [0.1, 0.3]},
            "_pytree",
        ),
    ],
)
@pytest.mark.parametrize(
    "regularizer",
    [
        "UnRegularized",
        "Ridge",
    ],
)
@pytest.mark.parametrize(
    "model_instantiation",
    [
        "poissonGLM",
        "gammaGLM",
        "bernoulliGLM",
        "gaussianGLM",
    ],
)
def test_newton_glm_analytic_v_autodiff(
    regularizer, regularizer_strength, model_instantiation, structure, request
):
    X, y, model, params, _ = request.getfixturevalue(
        model_instantiation + "_model_instantiation" + structure
    )
    model.solver_name = "Newton"
    model.regularizer = regularizer
    model.regularizer_strength = regularizer_strength

    autodiff = model._get_hess_fn(params, autodiff=True)(params, X, y)
    analytic = model._get_hess_fn(params, autodiff=False)(params, X, y)

    np.testing.assert_allclose(desired=autodiff, actual=analytic, atol=0.00001)
