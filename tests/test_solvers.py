import numpy as np
from nemos.solvers import SVRG
import jax
from nemos.tree_utils import pytree_map_and_reduce


def test_svrg_linear_regr(linear_regression):
    with jax.config.update("jax_enable_x64", True):
        X, y, _, params, loss = linear_regression
        svrg_params, state = SVRG(loss, tol=10**-12).run(np.zeros_like(params), X, y)
        assert np.allclose(params, svrg_params)


def test_svrg_linear_regr_tree(linear_regression_tree):
    with jax.config.update("jax_enable_x64", True):
        X, y, _, params, loss = linear_regression_tree
        param_init = jax.tree_util.tree_map(np.zeros_like, params)
        svrg_params, state = SVRG(loss, tol=10**-12, stepsize=0.0001).run(param_init, X, y)
        assert pytree_map_and_reduce(np.allclose, all,params, svrg_params)


def test_svrg_ridge_regr(ridge_regression):
    with jax.config.update("jax_enable_x64", True):
        X, y, _, params, loss = ridge_regression
        svrg_params, state = SVRG(loss, tol=10**-12, stepsize=10**-4).run(np.zeros_like(params), X, y)
        assert np.allclose(params, svrg_params, atol=10**-5, rtol=0.)


def test_svrg_ridge_regr_tree(ridge_regression_tree):
    with jax.config.update("jax_enable_x64", True):
        X, y, _, params, loss = ridge_regression_tree
        param_init = jax.tree_util.tree_map(np.zeros_like, params)
        svrg_params, state = SVRG(loss, tol=10**-12, stepsize=0.0001).run(param_init, X, y)
        assert pytree_map_and_reduce(np.allclose, all,params, svrg_params)
