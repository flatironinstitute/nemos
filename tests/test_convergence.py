"""Tests for making sure that solution reached for proximal operator is that the same as using another
method with just penalized loss."""

import jax
import numpy as np
import pytest
from scipy.optimize import minimize

import nemos as nmo


@pytest.mark.parametrize(
    "solver_names", [("GradientDescent", "ProximalGradient"), ("SVRG", "ProxSVRG")]
)
def test_unregularized_convergence(solver_names):
    """
    Assert that solution found when using GradientDescent vs ProximalGradient with an
    unregularized GLM is the same.
    """
    jax.config.update("jax_enable_x64", True)

    # generate toy data
    np.random.seed(111)
    # random design tensor. Shape (n_time_points, n_features).
    X = 0.5 * np.random.normal(size=(100, 5))

    # log-rates & weights, shape (1, ) and (n_features, ) respectively.
    b_true = np.zeros((1,))
    w_true = np.random.normal(size=(5,))

    # sparsify weights
    w_true[1:4] = 0.0

    # generate counts
    rate = jax.numpy.exp(jax.numpy.einsum("k,tk->t", w_true, X) + b_true)
    y = np.random.poisson(rate)

    # instantiate and fit unregularized GLM with GradientDescent
    model_GD = nmo.glm.GLM(solver_kwargs=dict(tol=10**-12))
    model_GD.fit(X, y)

    # instantiate and fit unregularized GLM with ProximalGradient
    model_PG = nmo.glm.GLM(
        solver_name="ProximalGradient", solver_kwargs=dict(tol=10**-12)
    )
    model_PG.fit(X, y)

    # assert weights are the same
    assert np.allclose(model_GD.coef_, model_PG.coef_)
    assert np.allclose(model_GD.intercept_, model_PG.intercept_)


@pytest.mark.parametrize(
    "solver_names", [("GradientDescent", "ProximalGradient"), ("SVRG", "ProxSVRG")]
)
def test_ridge_convergence(solver_names):
    """
    Assert that solution found when using GradientDescent vs ProximalGradient with an
    ridge GLM is the same.
    """
    jax.config.update("jax_enable_x64", True)
    # generate toy data
    np.random.seed(111)
    # random design tensor. Shape (n_time_points, n_features).
    X = 0.5 * np.random.normal(size=(100, 5))

    # log-rates & weights, shape (1, ) and (n_features, ) respectively.
    b_true = np.zeros((1,))
    w_true = np.random.normal(size=(5,))

    # sparsify weights
    w_true[1:4] = 0.0

    # generate counts
    rate = jax.numpy.exp(jax.numpy.einsum("k,tk->t", w_true, X) + b_true)
    y = np.random.poisson(rate)

    # instantiate and fit ridge GLM with GradientDescent
    model_GD = nmo.glm.GLM(
        regularizer_strength=1.0, regularizer="Ridge", solver_kwargs=dict(tol=10**-12)
    )
    model_GD.fit(X, y)

    # instantiate and fit ridge GLM with ProximalGradient
    model_PG = nmo.glm.GLM(
        regularizer_strength=1.0,
        regularizer="Ridge",
        solver_name="ProximalGradient",
        solver_kwargs=dict(tol=10**-12),
    )
    model_PG.fit(X, y)

    # assert weights are the same
    assert np.allclose(model_GD.coef_, model_PG.coef_)
    assert np.allclose(model_GD.intercept_, model_PG.intercept_)


@pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
def test_lasso_convergence(solver_name):
    """
    Assert that solution found when using ProximalGradient versus Nelder-Mead method using
    lasso GLM is the same.
    """
    jax.config.update("jax_enable_x64", True)
    # generate toy data
    num_samples, num_features, num_groups = 1000, 1, 3
    X = np.random.normal(size=(num_samples, num_features))  # design matrix
    w = [0.5]  # define some weights
    y = np.random.poisson(np.exp(X.dot(w)))  # observed counts

    # instantiate and fit GLM with ProximalGradient
    model_PG = nmo.glm.GLM(
        regularizer="Lasso",
        regularizer_strength=1.0,
        solver_name="ProximalGradient",
        solver_kwargs=dict(tol=10**-12),
    )
    model_PG.regularizer_strength = 0.1
    model_PG.fit(X, y)

    # use the penalized loss function to solve optimization via Nelder-Mead
    penalized_loss = lambda p, x, y: model_PG.regularizer.penalized_loss(
        model_PG._predict_and_compute_loss, model_PG.regularizer_strength
    )(
        (
            p[1:],
            p[0].reshape(
                1,
            ),
        ),
        x,
        y,
    )
    res = minimize(
        penalized_loss, [0] + w, args=(X, y), method="Nelder-Mead", tol=10**-12
    )

    # assert weights are the same
    assert np.allclose(res.x[1:], model_PG.coef_)
    assert np.allclose(res.x[:1], model_PG.intercept_)


@pytest.mark.parametrize("solver_name", ["ProximalGradient", "ProxSVRG"])
def test_group_lasso_convergence(solver_name):
    """
    Assert that solution found when using ProximalGradient versus Nelder-Mead method using
    group lasso GLM is the same.
    """
    jax.config.update("jax_enable_x64", True)
    # generate toy data
    num_samples, num_features, num_groups = 1000, 3, 2
    X = np.random.normal(size=(num_samples, num_features))  # design matrix
    w = [-0.5, 0.25, 0.5]  # define some weights
    y = np.random.poisson(np.exp(X.dot(w)))  # observed counts

    mask = np.zeros((num_groups, num_features))
    mask[0] = [1, 1, 0]  # Group 0 includes features 0 and 1
    mask[1] = [0, 0, 1]  # Group 1 includes features 1

    # instantiate and fit GLM with ProximalGradient
    model_PG = nmo.glm.GLM(
        regularizer=nmo.regularizer.GroupLasso(mask=mask),
        solver_kwargs=dict(tol=10**-14, maxiter=10000),
        regularizer_strength=0.2,
    )
    model_PG.fit(X, y)

    # use the penalized loss function to solve optimization via Nelder-Mead
    penalized_loss = lambda p, x, y: model_PG.regularizer.penalized_loss(
        model_PG._predict_and_compute_loss, model_PG.regularizer_strength
    )(
        (
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
        options=dict(maxiter=1000),
    )

    # assert weights are the same
    assert np.allclose(res.x[1:], model_PG.coef_)
    assert np.allclose(res.x[:1], model_PG.intercept_)
