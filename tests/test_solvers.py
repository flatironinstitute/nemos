import inspect
from contextlib import nullcontext as does_not_raise

import jax
import jaxopt
import numpy as np
import pytest

import nemos as nmo
from nemos.solvers._svrg import SVRG, ProxSVRG, SVRGState
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
    assert state.key == jax.random.key(123)
    assert state.full_grad_at_reference_point is None
    assert state.reference_point is not None


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
        match=r"Full gradient at the anchor point \(state\.full_grad_at_reference_point\) has to be set",
    ):
        _, _ = solver_class.update(param_init, state, *args)


@pytest.mark.parametrize(
    "regularizer_name, solver_class, mask",
    [
        ("Lasso", ProxSVRG, None),
        ("GroupLasso", ProxSVRG, np.array([0, 1, 0, 1]).reshape(1, -1).astype(float)),
        ("Ridge", SVRG, None),
        ("UnRegularized", SVRG, None),
    ],
)
def test_svrg_glm_instantiate_solver(regularizer_name, solver_class, mask):
    solver_name = solver_class.__name__

    # only pass mask if it's not None
    kwargs = {"solver_name": solver_name}
    if mask is not None:
        kwargs["mask"] = mask

    glm = nmo.glm.GLM(
        regularizer=regularizer_name,
        solver_name=solver_name,
        regularizer_strength=None if regularizer_name == "UnRegularized" else 1,
    )
    glm.instantiate_solver()

    solver = inspect.getclosurevars(glm._solver_run).nonlocals["solver"]
    assert glm.solver_name == solver_name
    assert isinstance(solver, solver_class)


@pytest.mark.parametrize(
    "regularizer_name, solver_name, mask",
    [
        ("Lasso", "ProxSVRG", None),
        ("GroupLasso", "ProxSVRG", np.array([0, 1, 0, 1]).reshape(1, -1).astype(float)),
        ("Ridge", "SVRG", None),
        ("UnRegularized", "SVRG", None),
    ],
)
@pytest.mark.parametrize("glm_class", [nmo.glm.GLM, nmo.glm.PopulationGLM])
def test_svrg_glm_passes_solver_kwargs(regularizer_name, solver_name, mask, glm_class):
    solver_kwargs = {
        "stepsize": np.abs(np.random.randn()),
        "maxiter": np.random.randint(1, 100),
    }

    # only pass mask if it's not None
    kwargs = {}
    if mask is not None and glm_class == nmo.glm.PopulationGLM:
        kwargs["feature_mask"] = mask

    glm = glm_class(
        regularizer=regularizer_name,
        solver_name=solver_name,
        solver_kwargs=solver_kwargs,
        regularizer_strength=None if regularizer_name == "UnRegularized" else 1,
        **kwargs,
    )
    glm.instantiate_solver()

    solver = inspect.getclosurevars(glm._solver_run).nonlocals["solver"]
    assert solver.stepsize == solver_kwargs["stepsize"]
    assert solver.maxiter == solver_kwargs["maxiter"]


@pytest.mark.parametrize(
    "regularizer_name, solver_class, mask",
    [
        ("Lasso", ProxSVRG, None),
        (
            "GroupLasso",
            ProxSVRG,
            np.array([[0.0], [0.0], [1.0]]),
        ),
        ("GroupLasso", ProxSVRG, np.array([[1.0], [0.0], [0.0]])),
        ("Ridge", SVRG, None),
        ("UnRegularized", SVRG, None),
    ],
)
@pytest.mark.parametrize(
    "glm_class",
    [nmo.glm.GLM, nmo.glm.PopulationGLM],
)
def test_svrg_glm_initialize_state(
    glm_class, regularizer_name, solver_class, mask, linear_regression
):
    X, y, _, _, _ = linear_regression

    if glm_class == nmo.glm.PopulationGLM:
        y = np.expand_dims(y, 1)

    reg_cls = getattr(nmo.regularizer, regularizer_name)
    if regularizer_name == "GroupLasso":
        reg = reg_cls(mask=mask)
    else:
        reg = reg_cls()

    # only pass mask if it's not None
    kwargs = {}
    if mask is not None and glm_class == nmo.glm.PopulationGLM:
        kwargs["feature_mask"] = mask

    glm = glm_class(
        regularizer=reg,
        solver_name=solver_class.__name__,
        observation_model=nmo.observation_models.PoissonObservations(jax.nn.softplus),
        regularizer_strength=None if regularizer_name == "UnRegularized" else 1,
        **kwargs,
    )

    init_params = glm.initialize_params(X, y)
    state = glm.initialize_state(X, y, init_params)

    assert state.reference_point == init_params

    for f in (glm._solver_init_state, glm._solver_update, glm._solver_run):
        assert isinstance(inspect.getclosurevars(f).nonlocals["solver"], solver_class)
    assert isinstance(state, SVRGState)


@pytest.mark.parametrize(
    "regularizer_name, solver_class, mask",
    [
        ("Lasso", ProxSVRG, None),
        (
            "GroupLasso",
            ProxSVRG,
            np.array([[0.0], [0.0], [1.0]]),
        ),
        ("Ridge", SVRG, None),
        ("UnRegularized", SVRG, None),
    ],
)
@pytest.mark.parametrize(
    "glm_class",
    [nmo.glm.GLM, nmo.glm.PopulationGLM],
)
def test_svrg_glm_update(
    glm_class, regularizer_name, solver_class, mask, linear_regression
):
    X, y, _, _, loss = linear_regression
    if glm_class == nmo.glm.PopulationGLM:
        y = np.expand_dims(y, 1)

    # only pass mask if it's not None
    kwargs = {}
    if glm_class == nmo.glm.PopulationGLM:
        kwargs["feature_mask"] = mask

    reg_cls = getattr(nmo.regularizer, regularizer_name)
    if regularizer_name == "GroupLasso":
        reg = reg_cls(mask=mask)
    else:
        reg = reg_cls()

    glm = glm_class(
        regularizer=reg,
        solver_name=solver_class.__name__,
        observation_model=nmo.observation_models.PoissonObservations(jax.nn.softplus),
        regularizer_strength=None if regularizer_name == "UnRegularized" else 1,
        **kwargs,
    )

    init_params = glm.initialize_params(X, y)
    state = glm.initialize_state(X, y, init_params)

    loss_gradient = jax.jit(jax.grad(glm._solver_loss_fun_))

    # initialize full gradient at the anchor point
    state = state._replace(
        full_grad_at_reference_point=loss_gradient(init_params, X, y),
    )

    params, state = glm.update(init_params, state, X, y)

    assert state.iter_num == 1


@pytest.mark.parametrize(
    "regularizer_name, solver_name, mask",
    [
        ("Lasso", "ProxSVRG", None),
        (
            "GroupLasso",
            "ProxSVRG",
            np.array([[0, 1, 0, 1, 1], [1, 0, 1, 0, 0]]).astype(float),
        ),
        ("GroupLasso", "ProxSVRG", np.array([[1, 1, 1, 1, 1]]).astype(float)),
        (
            "GroupLasso",
            "ProximalGradient",
            np.array([[0, 1, 0, 1, 1], [1, 0, 1, 0, 0]]).astype(float),
        ),
        ("GroupLasso", "ProximalGradient", np.array([[1, 1, 1, 1, 1]]).astype(float)),
        ("Ridge", "SVRG", None),
        ("UnRegularized", "SVRG", None),
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
    glm_class,
    regularizer_name,
    solver_name,
    mask,
    poissonGLM_model_instantiation,
    maxiter,
):
    X, y, model, (w_true, b_true), rate = poissonGLM_model_instantiation

    # set tolerance to -1 so that doesn't stop the iteration
    solver_kwargs = {
        "maxiter": maxiter,
        "tol": -1.0,
    }

    # only pass mask if it's not None
    reg_cls = getattr(nmo.regularizer, regularizer_name)
    if regularizer_name == "GroupLasso":
        reg = reg_cls(mask=mask)
    else:
        reg = reg_cls()

    kwargs = {}
    if glm_class == nmo.glm.PopulationGLM:
        kwargs["feature_mask"] = np.ones((X.shape[1], 1))

    glm = glm_class(
        regularizer=reg,
        solver_name=solver_name,
        observation_model=nmo.observation_models.PoissonObservations(jax.nn.softplus),
        solver_kwargs=solver_kwargs,
        regularizer_strength=None if regularizer_name == "UnRegularized" else 1,
        **kwargs,
    )

    if isinstance(glm, nmo.glm.PopulationGLM):
        y = np.expand_dims(y, 1)

    glm.fit(X, y)

    solver = inspect.getclosurevars(glm._solver_run).nonlocals["solver"]
    assert solver.maxiter == maxiter
    assert glm.solver_state_.iter_num == maxiter


@pytest.mark.parametrize(
    "regularizer_name, solver_class, mask",
    [
        ("Lasso", ProxSVRG, None),
        ("GroupLasso", ProxSVRG, np.array([0, 1, 0]).reshape(-1, 1).astype(float)),
        ("Ridge", SVRG, None),
        ("UnRegularized", SVRG, None),
    ],
)
@pytest.mark.parametrize(
    "glm_class",
    [nmo.glm.GLM, nmo.glm.PopulationGLM],
)
def test_svrg_glm_update_needs_full_grad_at_reference_point(
    glm_class, regularizer_name, solver_class, mask, linear_regression
):
    X, y, _, _, loss = linear_regression
    if glm_class.__name__ == "PopulationGLM":
        y = np.expand_dims(y, 1)

    # only pass mask if it's not None
    reg_cls = getattr(nmo.regularizer, regularizer_name)
    if regularizer_name == "GroupLasso":
        reg = reg_cls(mask=mask)
    else:
        reg = reg_cls()
    kwargs = dict(
        regularizer=reg,
        solver_name=solver_class.__name__,
        observation_model=nmo.observation_models.PoissonObservations(jax.nn.softplus),
        regularizer_strength=None if regularizer_name == "UnRegularized" else 0.1,
    )

    if mask is not None and glm_class == nmo.glm.PopulationGLM:
        kwargs["feature_mask"] = np.array([0, 1, 0]).reshape(-1, 1).astype(float)

    glm = glm_class(**kwargs)

    with pytest.raises(
        ValueError,
        match=r"Full gradient at the anchor point \(state\.full_grad_at_reference_point\) has to be set",
    ):
        params = glm.initialize_params(X, y)
        state = glm.initialize_state(X, y, params)
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
    key = jax.random.key(123)

    m = int((N + batch_size - 1) // batch_size)

    solver = SVRG(loss, stepsize=stepsize, batch_size=batch_size)
    params = jax.tree_util.tree_map(np.zeros_like, analytical_params)
    state = solver.init_state(params, X, y)

    for _ in range(maxiter):
        state = state._replace(
            full_grad_at_reference_point=loss_grad(params, X, y),
        )

        prev_params = params
        for _ in range(m):
            key, subkey = jax.random.split(key)
            ind = jax.random.randint(subkey, (batch_size,), 0, N)
            xi, yi = tree_slice(X, ind), y[ind]
            params, state = solver.update(params, state, xi, yi)

        state = state._replace(
            reference_point=params,
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
        (nmo.proximal_operator.prox_lasso, 0.1),
    ],
)
def test_svrg_xk_update_step(request, regr_setup, to_tuple, prox, prox_lambda):

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
    key = jax.random.key(123)
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
    svrg_next_xk = solver._inner_loop_param_update_step(
        init_param, xs, df_xs, stepsize, prox_lambda, xi, yi
    )

    assert pytree_map_and_reduce(
        lambda a, b: np.allclose(a, b, atol=10**-5, rtol=0.0),
        all,
        next_xk,
        svrg_next_xk,
    )


@pytest.mark.parametrize(
    "shapes, expected_context",
    [
        [
            (10, 10),
            does_not_raise(),
        ],
        [
            (10, 8),
            pytest.raises(
                ValueError,
                match="All arguments must have the same sized first dimension.",
            ),
        ],
    ],
)
def test_svrg_wrong_shapes(shapes, expected_context):
    X = np.random.randn(shapes[0], 3)
    y = np.random.randn(shapes[1], 1)

    init_params = np.random.randn(3, 1)

    def loss_fn(params, X, y):
        return 1.0

    with expected_context:
        svrg = SVRG(loss_fn)
        svrg.run(init_params, X, y)
