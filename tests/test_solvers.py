import jax
import jaxopt
import numpy as np
import pytest

import nemos as nmo
from nemos.regularizer import GroupLasso, Lasso, Ridge, UnRegularized
from nemos.solvers import SVRG, ProxSVRG
from nemos.tree_utils import pytree_map_and_reduce, tree_l2_norm, tree_slice, tree_sub


@pytest.mark.parametrize(
    ("regr_setup", "stepsize"),
    [
        ("linear_regression", 1e-3),
        ("ridge_regression", 1e-4),
        ("linear_regression_tree", 1e-4),
        ("ridge_regression_tree", 1e-4),
    ],
)
def test_svrg_linear_or_ridge_regression(request, regr_setup, stepsize):
    jax.config.update("jax_enable_x64", True)
    X, y, _, params, loss = request.getfixturevalue(regr_setup)

    param_init = jax.tree_util.tree_map(np.zeros_like, params)
    svrg_params, state = SVRG(loss, tol=10**-12, stepsize=stepsize).run(
        param_init, X, y
    )
    assert pytree_map_and_reduce(
        lambda a, b: np.allclose(a, b, atol=10**-5, rtol=0.0), all, params, svrg_params
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
def test_svrg_init_state_default(request, regr_setup):
    jax.config.update("jax_enable_x64", True)
    X, y, _, params, loss = request.getfixturevalue(regr_setup)

    param_init = jax.tree_util.tree_map(np.zeros_like, params)
    svrg = SVRG(loss)
    state = svrg.init_state(param_init, X, y)

    assert state.iter_num == 0
    assert state.key == jax.random.key(0)
    assert state.df_xs is None
    assert state.xs is not None


@pytest.mark.parametrize(
    "regr_setup",
    [
        "linear_regression",
        "ridge_regression",
        "linear_regression_tree",
        "ridge_regression_tree",
    ],
)
def test_svrg_init_state_key(request, regr_setup):
    random_key = jax.random.key(1000)

    jax.config.update("jax_enable_x64", True)
    X, y, _, params, loss = request.getfixturevalue(regr_setup)

    param_init = jax.tree_util.tree_map(np.zeros_like, params)
    svrg = SVRG(loss, key=random_key)
    state = svrg.init_state(param_init, X, y)

    assert state.key == random_key


@pytest.mark.parametrize(
    "regr_setup",
    [
        "linear_regression",
        "ridge_regression",
        "linear_regression_tree",
        "ridge_regression_tree",
    ],
)
def test_svrg_init_state_init_full_gradient(request, regr_setup):
    jax.config.update("jax_enable_x64", True)
    X, y, _, params, loss = request.getfixturevalue(regr_setup)

    param_init = jax.tree_util.tree_map(np.zeros_like, params)
    svrg = SVRG(loss)
    state = svrg.init_state(param_init, X, y, init_full_gradient=True)

    assert state.df_xs is not None


@pytest.mark.parametrize(
    "regr_setup",
    [
        "linear_regression",
        "linear_regression_tree",
    ],
)
@pytest.mark.parametrize(
    "solver_class, prox, prox_lambda",
    [(SVRG, None, None), (ProxSVRG, jaxopt.prox.prox_ridge, 0.1)],
)
def test_svrg_update_needs_df_xs(request, regr_setup, solver_class, prox, prox_lambda):
    jax.config.update("jax_enable_x64", True)
    X, y, _, params, loss = request.getfixturevalue(regr_setup)

    param_init = jax.tree_util.tree_map(np.zeros_like, params)
    if prox_lambda is not None:
        args = (prox_lambda, X, y)
        constr_args = (loss, prox)
    else:
        args = (X, y)
        constr_args = (loss,)

    solver_class = solver_class(*constr_args)
    state = solver_class.init_state(param_init, *args)

    with pytest.raises(
        ValueError,
        match=r"Full gradient at the anchor point \(state\.df_xs\) has to be set",
    ):
        _, _ = solver_class.update(param_init, state, *args)


@pytest.mark.parametrize(
    "regularizer_class, solver_class, mask",
    [
        (Lasso, ProxSVRG, None),
        (GroupLasso, ProxSVRG, np.array([0, 1, 0, 1]).reshape(1, -1).astype(float)),
        (Ridge, SVRG, None),
        (UnRegularized, SVRG, None),
    ],
)
def test_svrg_regularizer_constr(
    regularizer_class, solver_class, mask, linear_regression
):
    _, _, _, _, loss = linear_regression

    # only pass mask if it's not None
    kwargs = {"solver_name": solver_class.__name__}
    if mask is not None:
        kwargs["mask"] = mask

    reg = regularizer_class(**kwargs)
    reg.instantiate_solver(loss)

    assert isinstance(reg._solver, solver_class)


@pytest.mark.parametrize(
    "regularizer_class, solver_class, mask",
    [
        (Lasso, ProxSVRG, None),
        (GroupLasso, ProxSVRG, np.array([0, 1, 0, 1]).reshape(1, -1).astype(float)),
        (Ridge, SVRG, None),
        (UnRegularized, SVRG, None),
    ],
)
def test_svrg_regularizer_passes_solver_kwargs(
    regularizer_class, solver_class, mask, linear_regression
):
    _, _, _, _, loss = linear_regression

    solver_kwargs = {
        "stepsize": np.abs(np.random.randn()),
        "maxiter": np.random.randint(1, 100),
    }

    # only pass mask if it's not None
    kwargs = {
        "solver_name": solver_class.__name__,
        "solver_kwargs": solver_kwargs,
    }
    if mask is not None:
        kwargs["mask"] = mask

    reg = regularizer_class(**kwargs)
    reg.instantiate_solver(loss)

    assert reg._solver.stepsize == solver_kwargs["stepsize"]
    assert reg._solver.maxiter == solver_kwargs["maxiter"]


@pytest.mark.parametrize(
    "regularizer_class, solver_class, mask",
    [
        (Lasso, ProxSVRG, None),
        (GroupLasso, ProxSVRG, np.array([0, 1, 0, 1]).reshape(1, -1).astype(float)),
        (Ridge, SVRG, None),
        (UnRegularized, SVRG, None),
    ],
)
def test_svrg_glm_initialize_solver(
    regularizer_class, solver_class, mask, linear_regression
):
    X, y, _, _, loss = linear_regression
    # make y 2D
    y = np.expand_dims(y, 1)

    # only pass mask if it's not None
    kwargs = {"solver_name": solver_class.__name__}
    if mask is not None:
        kwargs["mask"] = mask

    glm = nmo.glm.PopulationGLM(
        regularizer=regularizer_class(**kwargs),
        observation_model=nmo.observation_models.PoissonObservations(jax.nn.softplus),
    )

    params, state = glm.initialize_solver(X, y)

    assert isinstance(glm.regularizer._solver, solver_class)


@pytest.mark.parametrize(
    "regularizer_class, solver_class, mask",
    [
        (Lasso, ProxSVRG, None),
        (GroupLasso, ProxSVRG, np.array([0, 1, 0]).reshape(1, -1).astype(float)),
        (Ridge, SVRG, None),
        (UnRegularized, SVRG, None),
    ],
)
def test_svrg_glm_update(regularizer_class, solver_class, mask, linear_regression):
    X, y, _, _, loss = linear_regression
    # make y 2D
    y = np.expand_dims(y, 1)

    # only pass mask if it's not None
    kwargs = {"solver_name": solver_class.__name__}
    if mask is not None:
        kwargs["mask"] = mask

    glm = nmo.glm.PopulationGLM(
        regularizer=regularizer_class(**kwargs),
        observation_model=nmo.observation_models.PoissonObservations(jax.nn.softplus),
    )

    params, state = glm.initialize_solver(X, y, init_full_gradient=True)
    params, state = glm.update(params, state, X, y)

    assert state.iter_num == 1


@pytest.mark.parametrize(
    "regularizer_class, solver_class, mask",
    [
        (Lasso, ProxSVRG, None),
        # (
        #    GroupLasso,
        #    ProxSVRG,
        #    np.array([[0, 1, 0], [0, 0, 1]]).reshape(2, -1).astype(float),
        # ),
        (Ridge, SVRG, None),
        (UnRegularized, SVRG, None),
    ],
)
@pytest.mark.parametrize(
    "maxiter",
    [3, 50],
)
def test_svrg_glm_fit(
    regularizer_class, solver_class, mask, linear_regression, maxiter
):
    X, y, true_coef, _, loss = linear_regression
    # make y 2D
    y = np.expand_dims(y, 1)

    # set tolerance to -1 so that doesn't stop the iteration
    solver_kwargs = {
        "maxiter": maxiter,
        "tol": -1.0,
    }

    # only pass mask if it's not None
    kwargs = {"solver_name": solver_class.__name__}
    if mask is not None:
        kwargs["mask"] = mask

    # with jax.disable_jit():
    glm = nmo.glm.PopulationGLM(
        regularizer=regularizer_class(**kwargs, solver_kwargs=solver_kwargs),
        observation_model=nmo.observation_models.PoissonObservations(jax.nn.softplus),
    )

    glm.fit(X, y)

    assert glm.regularizer._solver.maxiter == maxiter
    assert glm.solver_state.iter_num == maxiter


@pytest.mark.parametrize(
    "regularizer_class, solver_class, mask",
    [
        (Lasso, ProxSVRG, None),
        (GroupLasso, ProxSVRG, np.array([0, 1, 0]).reshape(1, -1).astype(float)),
        (Ridge, SVRG, None),
        (UnRegularized, SVRG, None),
    ],
)
def test_svrg_glm_update_needs_df_xs(
    regularizer_class, solver_class, mask, linear_regression
):
    X, y, _, _, loss = linear_regression
    # make y 2D
    y = np.expand_dims(y, 1)

    # only pass mask if it's not None
    kwargs = {"solver_name": solver_class.__name__}
    if mask is not None:
        kwargs["mask"] = mask

    glm = nmo.glm.PopulationGLM(
        regularizer=regularizer_class(**kwargs),
        observation_model=nmo.observation_models.PoissonObservations(jax.nn.softplus),
    )

    with pytest.raises(
        ValueError,
        match=r"Full gradient at the anchor point \(state\.df_xs\) has to be set",
    ):
        params, state = glm.initialize_solver(X, y)
        glm.update(params, state, X, y)


@pytest.mark.parametrize(
    ("regr_setup", "stepsize"),
    [
        ("linear_regression", 1e-3),
        ("ridge_regression", 1e-4),
        ("linear_regression_tree", 1e-4),
        ("ridge_regression_tree", 1e-4),
    ],
)
def test_svrg_update_converges(request, regr_setup, stepsize):
    jax.config.update("jax_enable_x64", True)
    X, y, _, analytical_params, loss = request.getfixturevalue(regr_setup)

    loss_grad = jax.jit(jax.grad(loss))

    N = y.shape[0]
    batch_size = 1
    maxiter = 10_000
    tol = 1e-12
    key = jax.random.key(0)

    m = int((N + batch_size - 1) // batch_size)

    solver = SVRG(loss, stepsize=stepsize, batch_size=batch_size)
    params = jax.tree_util.tree_map(np.zeros_like, analytical_params)
    state = solver.init_state(params, X, y)

    for _ in range(maxiter):
        state = state._replace(
            df_xs=loss_grad(params, X, y),
        )

        prev_params = params
        for _ in range(m):
            key, subkey = jax.random.split(key)
            ind = jax.random.randint(subkey, (batch_size,), 0, N)
            xi, yi = tree_slice(X, ind), y[ind]
            params, state = solver.update(params, state, xi, yi)

        state = state._replace(
            xs=params,
        )

        _error = tree_l2_norm(tree_sub(params, prev_params)) / tree_l2_norm(prev_params)
        if _error < tol:
            break

    assert pytree_map_and_reduce(
        lambda a, b: np.allclose(a, b, atol=10**-5, rtol=0.0),
        all,
        analytical_params,
        params,
    )
