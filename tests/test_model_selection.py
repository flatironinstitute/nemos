"""Tests for approximate leave-one-out cross-validation (``nemos.model_selection``).

The core correctness tests compare the approximate LOO (single fit + one Newton-step
correction per point) against *exact* LOO computed by refitting the model n times on
the leave-one-out subsets. Because the objective's data term is a per-sample mean, a
plain refit on ``n - 1`` points renormalizes the penalty relative to the data; for the
Ridge test we rescale ``regularizer_strength`` by ``n / (n - 1)`` so the exact refit
targets the same objective as the infinitesimal-jackknife approximation.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import nemos as nmo
from nemos.glm import GLM, PopulationGLM
from nemos.model_selection import ApproximateLOO, approximate_loo

# tight tolerances require float64
pytestmark = pytest.mark.requires_x64


def _fit(cls, X, y, **kwargs):
    kwargs.setdefault("solver_name", "LBFGS")
    kwargs.setdefault("solver_kwargs", dict(tol=1e-12, maxiter=5000))
    return cls(**kwargs).fit(X, y)


def _exact_loo_mean(cls, X, y, ridge_strength=None, **fit_kwargs):
    """Exact LOO predicted means by refitting on each leave-one-out subset."""
    n = X.shape[0]
    mu = np.zeros(y.shape)
    for k in range(n):
        idx = np.arange(n) != k
        kwargs = dict(fit_kwargs)
        if ridge_strength is not None:
            # rescale so the (n-1)-point refit matches the down-weighting ALO target
            kwargs["regularizer_strength"] = ridge_strength * n / (n - 1)
        m = _fit(cls, X[idx], y[idx], **kwargs)
        mu[k] = np.asarray(m.predict(X[k : k + 1]))[0]
    return mu


def _poisson_data(n=50, p=3, seed=0, scale=0.4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    coef = rng.normal(size=p) * scale
    y = rng.poisson(np.exp(X @ coef - 0.5)).astype(float)
    return X, y


# --------------------------------------------------------------------------- #
# correctness vs exact refit-based LOO                                         #
# --------------------------------------------------------------------------- #


def test_approximate_loo_matches_exact_poisson():
    """Unregularized Poisson: approximate LOO ~ exact refit LOO."""
    X, y = _poisson_data(n=50, p=3, seed=0)
    model = _fit(GLM, X, y)
    loo = approximate_loo(model, X, y)
    exact = _exact_loo_mean(GLM, X, y)

    rel_err = np.abs(np.asarray(loo.predicted_mean) - exact) / np.maximum(exact, 1e-8)
    # the one-step-Newton approximation is O(1e-3) on average, worse at high leverage
    assert rel_err.mean() < 1e-2
    assert rel_err.max() < 6e-2


def test_approximate_loo_matches_exact_ridge():
    """Ridge-penalized Poisson: approximate LOO ~ exact refit LOO."""
    X, y = _poisson_data(n=50, p=3, seed=3)
    strength = 1.0
    model = _fit(GLM, X, y, regularizer="Ridge", regularizer_strength=strength)
    loo = approximate_loo(model, X, y)
    exact = _exact_loo_mean(GLM, X, y, ridge_strength=strength, regularizer="Ridge")

    rel_err = np.abs(np.asarray(loo.predicted_mean) - exact) / np.maximum(exact, 1e-8)
    assert rel_err.mean() < 5e-3
    assert rel_err.max() < 3e-2


def test_approximate_loo_gaussian_is_exact():
    """For a Gaussian/identity-link GLM the objective is quadratic, so a single
    Newton step recovers the exact LOO fit (up to solver tolerance)."""
    rng = np.random.default_rng(1)
    n, p = 40, 3
    X = rng.normal(size=(n, p))
    coef = rng.normal(size=p) * 0.5
    y = X @ coef + 0.3 + rng.normal(size=n) * 0.5
    model = _fit(GLM, X, y, observation_model="Gaussian")
    loo = approximate_loo(model, X, y)
    exact = _exact_loo_mean(GLM, X, y, observation_model="Gaussian")
    assert np.allclose(np.asarray(loo.predicted_mean), exact, atol=1e-5, rtol=1e-4)


# --------------------------------------------------------------------------- #
# PopulationGLM                                                                #
# --------------------------------------------------------------------------- #


def test_approximate_loo_population_matches_single_glm():
    """Each PopulationGLM neuron's LOO must equal the corresponding single-GLM LOO,
    since the population objective is separable across neurons (block-diagonal Hessian).
    """
    rng = np.random.default_rng(7)
    n, p, n_neurons = 120, 3, 3
    X = rng.normal(size=(n, p))
    W = rng.normal(size=(p, n_neurons)) * 0.4
    b = rng.normal(size=n_neurons) * 0.2 - 0.5
    y = rng.poisson(np.exp(X @ W + b)).astype(float)

    pop = _fit(PopulationGLM, X, y)
    loo = approximate_loo(pop, X, y)
    assert loo.predicted_mean.shape == (n, n_neurons)

    for j in range(n_neurons):
        single = _fit(GLM, X, y[:, j])
        loo_single = approximate_loo(single, X, y[:, j])
        assert np.allclose(
            np.asarray(loo.predicted_mean[:, j]),
            np.asarray(loo_single.predicted_mean),
            atol=1e-6,
            rtol=1e-5,
        )


# --------------------------------------------------------------------------- #
# structural / diagnostic properties                                          #
# --------------------------------------------------------------------------- #


def test_approximate_loo_return_type_and_shapes():
    X, y = _poisson_data(n=60, p=4, seed=2)
    model = _fit(GLM, X, y)
    loo = approximate_loo(model, X, y)
    assert isinstance(loo, ApproximateLOO)
    for field in (
        "predicted_mean",
        "linear_predictor",
        "log_likelihood",
        "deviance",
        "leverage",
    ):
        assert getattr(loo, field).shape == (60,)


def test_approximate_loo_leverage_in_unit_interval():
    """Hat-matrix diagonals must lie in [0, 1)."""
    X, y = _poisson_data(n=80, p=5, seed=4)
    model = _fit(GLM, X, y)
    loo = approximate_loo(model, X, y)
    h = np.asarray(loo.leverage)
    assert np.all(h >= 0.0)
    assert np.all(h < 1.0)


def test_approximate_loo_method_matches_function():
    """The GLM.approximate_loo method delegates to the module function."""
    X, y = _poisson_data(n=50, p=3, seed=1)
    model = _fit(GLM, X, y)
    from_func = approximate_loo(model, X, y)
    from_method = model.approximate_loo(X, y)
    assert np.array_equal(
        np.asarray(from_func.predicted_mean), np.asarray(from_method.predicted_mean)
    )


def test_approximate_loo_deviance_matches_observation_model():
    """Returned per-observation deviance equals the observation model's deviance at
    the LOO predicted mean."""
    X, y = _poisson_data(n=50, p=3, seed=6)
    model = _fit(GLM, X, y)
    loo = approximate_loo(model, X, y)
    expected = model.observation_model.deviance(jnp.asarray(y), loo.predicted_mean)
    assert np.allclose(np.asarray(loo.deviance), np.asarray(expected))


# --------------------------------------------------------------------------- #
# guards                                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "regularizer, solver_name",
    [
        ("Lasso", "ProximalGradient"),
        ("ElasticNet", "ProximalGradient"),
        ("GroupLasso", "ProximalGradient"),
    ],
)
def test_approximate_loo_raises_for_nonsmooth_regularizer(regularizer, solver_name):
    X, y = _poisson_data(n=40, p=3, seed=0)
    kwargs = dict(
        regularizer=regularizer, regularizer_strength=0.1, solver_name=solver_name
    )
    if regularizer == "GroupLasso":
        kwargs["regularizer"] = nmo.regularizer.GroupLasso(
            mask=np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        )
    model = _fit(GLM, X, y, **kwargs)
    with pytest.raises(NotImplementedError, match="non-smooth"):
        approximate_loo(model, X, y)


def test_approximate_loo_raises_for_missing_variance_function():
    """NegativeBinomial has no variance function registered -> NotImplementedError."""
    X, y = _poisson_data(n=40, p=3, seed=0)
    model = _fit(GLM, X, y, observation_model="NegativeBinomial")
    with pytest.raises(NotImplementedError):
        approximate_loo(model, X, y)


def test_approximate_loo_raises_if_not_fitted():
    X, y = _poisson_data(n=40, p=3, seed=0)
    with pytest.raises(nmo.exceptions.NotFittedError):
        approximate_loo(GLM(), X, y)


def test_approximate_loo_raises_with_feature_mask():
    rng = np.random.default_rng(0)
    n, p, n_neurons = 60, 3, 2
    X = rng.normal(size=(n, p))
    y = rng.poisson(np.exp(X @ rng.normal(size=(p, n_neurons)) * 0.3)).astype(float)
    mask = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    model = _fit(PopulationGLM, X, y, feature_mask=mask)
    with pytest.raises(NotImplementedError, match="feature_mask"):
        approximate_loo(model, X, y)
