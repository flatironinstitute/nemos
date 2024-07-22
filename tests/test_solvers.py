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
@pytest.mark.parametrize(
    "glm_class",
    [nmo.glm.GLM, nmo.glm.PopulationGLM],
)
def test_svrg_glm_initialize_solver(
    glm_class, regularizer_class, solver_class, mask, linear_regression
):
    X, y, _, _, loss = linear_regression
    if glm_class.__name__ == "PopulationGLM":
        y = np.expand_dims(y, 1)

    # only pass mask if it's not None
    kwargs = {"solver_name": solver_class.__name__}
    if mask is not None:
        kwargs["mask"] = mask

    glm = glm_class(
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
@pytest.mark.parametrize(
    "glm_class",
    [nmo.glm.GLM, nmo.glm.PopulationGLM],
)
def test_svrg_glm_update(
    glm_class, regularizer_class, solver_class, mask, linear_regression
):
    X, y, _, _, loss = linear_regression
    if glm_class.__name__ == "PopulationGLM":
        y = np.expand_dims(y, 1)

    # only pass mask if it's not None
    kwargs = {"solver_name": solver_class.__name__}
    if mask is not None:
        kwargs["mask"] = mask

    glm = glm_class(
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
        (
           GroupLasso,
           ProxSVRG,
           np.array([[0, 1, 0], [0, 0, 1]]).reshape(2, -1).astype(float),
        ),
        (Ridge, SVRG, None),
        (UnRegularized, SVRG, None),
    ],
)
@pytest.mark.parametrize(
    "maxiter",
    [3, 50],
)
@pytest.mark.parametrize(
    "glm_class",
    [nmo.glm.GLM, nmo.glm.PopulationGLM],
)
def test_svrg_glm_fit(
    glm_class, regularizer_class, solver_class, mask, linear_regression, maxiter
):
    X, y, true_coef, _, loss = linear_regression


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
    glm = glm_class(
        regularizer=regularizer_class(**kwargs, solver_kwargs=solver_kwargs),
        observation_model=nmo.observation_models.PoissonObservations(jax.nn.softplus),
    )

    if isinstance(glm, nmo.glm.PopulationGLM):
        y = np.expand_dims(y, 1)

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
@pytest.mark.parametrize(
    "glm_class",
    [nmo.glm.GLM, nmo.glm.PopulationGLM],
)
def test_svrg_glm_update_needs_df_xs(
    glm_class, regularizer_class, solver_class, mask, linear_regression
):
    X, y, _, _, loss = linear_regression
    if glm_class.__name__ == "PopulationGLM":
        y = np.expand_dims(y, 1)

    # only pass mask if it's not None
    kwargs = {"solver_name": solver_class.__name__}
    if mask is not None:
        kwargs["mask"] = mask

    glm = glm_class(
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


@pytest.mark.parametrize(
    "regr_setup, to_tuple",
    [
        ("linear_regression", True),
        ("linear_regression", False),
        ("linear_regression_tree", False),
    ],
)
@pytest.mark.parametrize(
    "prox, prox_lambda",
    [
        (jaxopt.prox.prox_none, None),
        (jaxopt.prox.prox_ridge, 0.1),
        (jaxopt.prox.prox_none, 0.1),
        # (jaxopt.prox.prox_lasso, 0.1),
    ],
)
def test_svrg_xk_update(request, regr_setup, to_tuple, prox, prox_lambda):

    X, y, true_params, ols_coef, loss_arr = request.getfixturevalue(regr_setup)

    # the loss takes an array, but I want to test with tuples as well
    # so make a new loss function that takes a tuple
    if to_tuple:
        true_params = (
            true_params,
            np.zeros(X.shape[1]),
        )
        loss = lambda params, X, y: loss_arr(params[0], X, y)
    else:
        loss = loss_arr

    stepsize = 1e-2
    loss_gradient = jax.jit(jax.grad(loss))

    # set the initial parameters to zero and
    # set the anchor point to a random value that's not just zeros
    init_param = jax.tree_util.tree_map(np.zeros_like, true_params)
    xs = jax.tree_util.tree_map(lambda x: np.random.randn(*x.shape), true_params)
    df_xs = loss_gradient(xs, X, y)

    # sample a mini-batch
    key = jax.random.key(0)
    key, subkey = jax.random.split(key)
    ind = jax.random.randint(subkey, (32,), 0, y.shape[0])
    xi, yi = tree_slice(X, ind), tree_slice(y, ind)

    dfik_xk = loss_gradient(init_param, xi, yi)
    dfik_xs = loss_gradient(xs, xi, yi)

    # update if inputs are arrays
    def _array_update(dfik_xk, dfik_xs, df_xs, init_param, stepsize):
        gk = dfik_xk - dfik_xs + df_xs
        next_xk = init_param - stepsize * gk
        return next_xk

    # update if inputs are a tuple of arrays
    def _tuple_update(dfik_xk, dfik_xs, df_xs, init_param, stepsize):
        return tuple(
            _array_update(a, b, c, d, stepsize)
            for a, b, c, d in zip(dfik_xk, dfik_xs, df_xs, init_param)
        )

    # update if inputs are dicts with either arrays or tuple of arrays as inputs
    # behavior is determined by update_fun
    def _dict_update(dfik_xk, dfik_xs, df_xs, init_param, stepsize, update_fun):
        return {
            k: update_fun(dfik_xk[k], dfik_xs[k], df_xs[k], init_param[k], stepsize)
            for k in dfik_xk.keys()
        }

    if isinstance(true_params, np.ndarray):
        next_xk = _array_update(dfik_xk, dfik_xs, df_xs, init_param, stepsize)
    elif isinstance(true_params, tuple):
        next_xk = _tuple_update(dfik_xk, dfik_xs, df_xs, init_param, stepsize)
    elif isinstance(X, dict):
        assert (
            set(X.keys())
            == set(dfik_xk.keys())
            == set(dfik_xs.keys())
            == set(df_xs.keys())
        )

        if isinstance(list(dfik_xk.values())[0], tuple):
            update_fun = _tuple_update
        else:
            update_fun = _array_update

        next_xk = _dict_update(
            dfik_xk, dfik_xs, df_xs, init_param, stepsize, update_fun
        )
    else:
        raise TypeError

    next_xk = prox(next_xk, prox_lambda, scaling=stepsize)

    if prox_lambda is None:
        assert prox == jaxopt.prox.prox_none
        solver = SVRG(loss)
    else:
        solver = ProxSVRG(loss, prox)
    svrg_next_xk = solver._xk_update(
        init_param, xs, df_xs, stepsize, prox_lambda, xi, yi
    )

    assert pytree_map_and_reduce(
        lambda a, b: np.allclose(a, b, atol=10**-5, rtol=0.0),
        all,
        next_xk,
        svrg_next_xk,
    )
