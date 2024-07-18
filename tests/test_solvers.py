import jax
import numpy as np
import pytest

from nemos.solvers import SVRG
from nemos.tree_utils import pytree_map_and_reduce


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
