from functools import partial

import jax
import numpy as np
import pytest

from nemos.fetch import fetch_data
from nemos.glm_hmm.expectation_maximization import (
    backward_pass,
    compute_xi,
    forward_backward,
    forward_pass,
)
from nemos.observation_models import BernoulliObservations, PoissonObservations


def forward_step_numpy(py_z, new_sess, initial_prob, transition_prob):
    n_time_bins, n_states = py_z.shape
    alphas = np.full((n_time_bins, n_states), np.nan)
    c = np.full(n_time_bins, np.nan)
    for t in range(n_time_bins):
        if new_sess[t]:
            alphas[t] = initial_prob * py_z[t]
        else:
            alphas[t] = py_z[t] * (transition_prob.T @ alphas[t - 1])

        c[t] = np.sum(alphas[t])
        alphas[t] /= c[t]
    return alphas, c


def backward_step_numpy(py_z, c, new_sess, transition_prob):
    n_time_bins, n_states = py_z.shape
    betas = np.full((n_time_bins, n_states), np.nan)
    betas[-1] = np.ones(n_states)

    for t in range(n_time_bins - 2, -1, -1):
        if new_sess[t + 1]:
            betas[t] = np.ones(n_states)
        else:
            betas[t] = transition_prob @ (betas[t + 1] * py_z[t + 1])
            betas[t] /= c[t + 1]
    return betas


# Below are tests for Bernoulli observation 3 states model glm hmm
@pytest.mark.parametrize(
    "decorator",
    [
        lambda x: x,
        partial(jax.jit, static_argnames=["likelihood_func", "inverse_link_function"]),
    ],
)
def test_forward_backward_regression(decorator):
    jax.config.update("jax_enable_x64", True)

    # Fetch the data
    data_path = fetch_data("em_three_states.npz")
    data = np.load(data_path)

    X, y = data["X"], data["y"]
    new_sess = data["new_sess"]

    # assert first column is intercept, otherwise this will break
    # nemos assumption (existence of an intercept term)
    np.testing.assert_array_equal(X[:, 0], 1)

    # E-step initial parameters
    initial_prob = data["initial_prob"]
    intercept, coef = data["projection_weights"][:1], data["projection_weights"][1:]
    transition_prob = data["transition_prob"]

    # E-step output
    xis = data["xis"]
    gammas = data["gammas"]
    ll_orig, ll_norm_orig = data["log_likelihood"], data["ll_norm"]
    alphas, betas = data["alphas"], data["betas"]

    obs = BernoulliObservations()

    likelihood = jax.vmap(
        lambda x, z: obs.likelihood(x, z, aggregate_sample_scores=lambda w: w),
        in_axes=(None, 1),
        out_axes=1,
    )

    decorated_forward_backward = decorator(forward_backward)
    gammas_nemos, xis_nemos, ll_nemos, ll_norm_nemos, alphas_nemos, betas_nemos = (
        decorated_forward_backward(
            X[:, 1:],  # drop intercept
            y,
            initial_prob,
            transition_prob,
            (coef, intercept),
            likelihood_func=likelihood,
            inverse_link_function=obs.default_inverse_link_function,
            is_new_session=new_sess.astype(bool),
        )
    )

    # First testing alphas and betas because they are computed first
    np.testing.assert_almost_equal(alphas_nemos, alphas, decimal=8)
    np.testing.assert_almost_equal(betas_nemos, betas, decimal=8)

    # testing log likelihood and normalized log likelihood
    np.testing.assert_almost_equal(ll_nemos, ll_orig, decimal=8)
    np.testing.assert_almost_equal(ll_norm_nemos, ll_norm_orig, decimal=8)

    # Next testing xis and gammas because they depend on alphas and betas
    # Testing Eq. 13.43 of Bishop
    np.testing.assert_almost_equal(gammas_nemos, gammas, decimal=8)
    # Testing Eq. 13.65 of Bishop
    np.testing.assert_almost_equal(xis_nemos, xis, decimal=8)


def test_for_loop_forward_step():
    jax.config.update("jax_enable_x64", True)

    # Fetch the data
    data_path = fetch_data("em_three_states.npz")
    data = np.load(data_path)

    X, y = data["X"], data["y"]
    new_sess = data["new_sess"]

    # E-step initial parameters
    initial_prob = data["initial_prob"]
    intercept, coef = data["projection_weights"][:1], data["projection_weights"][1:]
    transition_prob = data["transition_prob"]

    obs = BernoulliObservations()

    likelihood = jax.vmap(
        lambda x, z: obs.likelihood(x, z, aggregate_sample_scores=lambda w: w),
        in_axes=(None, 1),
        out_axes=1,
    )

    predicted_rate_given_state = obs.default_inverse_link_function(
        X[:, 1:] @ coef + intercept
    )
    conditionals = likelihood(y, predicted_rate_given_state)

    alphas, normalization = forward_pass(
        initial_prob, transition_prob, conditionals, new_sess
    )

    alphas_numpy, normalization_numpy = forward_step_numpy(
        conditionals, new_sess, initial_prob, transition_prob
    )
    np.testing.assert_almost_equal(alphas_numpy, alphas)
    np.testing.assert_almost_equal(normalization_numpy, normalization)


def test_for_loop_backward_step():
    jax.config.update("jax_enable_x64", True)

    # Fetch the data
    data_path = fetch_data("em_three_states.npz")
    data = np.load(data_path)

    X, y = data["X"], data["y"]
    new_sess = data["new_sess"]

    # E-step initial parameters
    initial_prob = data["initial_prob"]
    intercept, coef = data["projection_weights"][:1], data["projection_weights"][1:]
    transition_prob = data["transition_prob"]

    obs = BernoulliObservations()

    likelihood = jax.vmap(
        lambda x, z: obs.likelihood(x, z, aggregate_sample_scores=lambda w: w),
        in_axes=(None, 1),
        out_axes=1,
    )

    predicted_rate_given_state = obs.default_inverse_link_function(
        X[:, 1:] @ coef + intercept
    )
    conditionals = likelihood(y, predicted_rate_given_state)

    alphas, normalization = forward_pass(
        initial_prob, transition_prob, conditionals, new_sess
    )

    betas = backward_pass(transition_prob, conditionals, normalization, new_sess)
    betas_numpy = backward_step_numpy(
        conditionals, normalization, new_sess, transition_prob
    )
    np.testing.assert_almost_equal(betas_numpy, betas)


def test_single_state_forward_pass():
    """Single state forward pass posteriors reduces to ones (there is a single state)."""
    np.random.seed(42)
    initial_prob = np.ones(1)
    transition_prob = np.ones((1, 1))
    coef, intercept = np.random.randn(2, 1), np.random.randn(1)
    X = np.random.randn(10, 2)
    rate = np.exp(X.dot(coef) + intercept)
    y = np.random.poisson(rate[:, 0])

    obs = PoissonObservations()

    likelihood = jax.vmap(
        lambda x, z: obs.likelihood(x, z, aggregate_sample_scores=lambda w: w),
        in_axes=(None, 1),
        out_axes=1,
    )
    conditionals = likelihood(y, rate)
    new_sess = np.zeros(10)
    new_sess[0] = 1
    alphas, norm = forward_pass(initial_prob, transition_prob, conditionals, new_sess)
    betas = backward_pass(transition_prob, conditionals, norm, new_sess)

    # check that the normalization factor reduces to the p(x_t | z_t)
    np.testing.assert_almost_equal(norm, conditionals[:, 0])

    # Note: alphas * betas is p(z_t | X), so it's automatically ones if the
    # two assertions passes, no need to check explicitly for  p(z_t | X).
    np.testing.assert_almost_equal(np.ones_like(alphas), alphas)
    np.testing.assert_almost_equal(np.ones_like(betas), betas)

    # xis are a sum of the ones over valid entires
    xis = compute_xi(
        alphas,
        betas,
        conditionals,
        norm,
        new_sess,
        transition_prob,
    )
    np.testing.assert_almost_equal(
        np.array([[alphas.shape[0] - sum(new_sess)]]).astype(xis), xis
    )
