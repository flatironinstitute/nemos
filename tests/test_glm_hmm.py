from functools import partial

import jax
import numpy as np
import pytest

from nemos.fetch import fetch_data
from nemos.glm_hmm import forward_backward, run_m_step
from nemos.observation_models import BernoulliObservations


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
    data_path = fetch_data("e_step_three_states.npz")
    data = np.load(data_path)

    X, y = data["X"], data["y"]
    new_sess = data["new_sess"]

    # E-step initial parameters
    initial_prob = data["initial_prob"]
    projection_weights = data["projection_weights"]
    transition_matrix = data["transition_matrix"]

    # E-step output
    xis = data["xis"]
    gammas = data["gammas"]
    ll_orig, ll_norm_orig = data["log_likelihood"], data["ll_norm"]
    alphas, betas = data["alphas"], data["betas"]

    obs = BernoulliObservations()

    log_likelihood = jax.vmap(
        lambda x, z: obs.likelihood(x, z, aggregate_sample_scores=lambda w: w),
        in_axes=(None, 1),
        out_axes=1,
    )

    decorated_forward_backward = decorator(forward_backward)
    gammas_nemos, xis_nemos, ll_nemos, ll_norm_nemos, alphas_nemos, betas_nemos = (
        decorated_forward_backward(
            X,
            y.flatten(),
            initial_prob,
            transition_matrix,
            projection_weights,
            likelihood_func=log_likelihood,
            inverse_link_function=obs.inverse_link_function,
            is_new_session=new_sess.flatten().astype(bool),
        )
    )

    # First testing alphas and betas because they are computed first
    np.testing.assert_almost_equal(alphas.T, alphas_nemos, decimal=4)
    np.testing.assert_almost_equal(betas.T, betas_nemos, decimal=4)
    # testing log likelihood and normalized log likelihood
    np.testing.assert_almost_equal(ll_orig, ll_nemos, decimal=4)
    np.testing.assert_almost_equal(ll_norm_orig, ll_norm_nemos, decimal=4)

    # Next testing xis and gammas because they depend on alphas and betas
    # Equations 13.43
    np.testing.assert_almost_equal(gammas.T, gammas_nemos, decimal=4)
    # testing 13.65 of Bishop
    np.testing.assert_almost_equal(xis, xis_nemos, decimal=4)


@pytest.mark.parametrize(
    "decorator",
    [
        lambda x: x,
        partial(jax.jit, static_argnames=["likelihood_func", "inverse_link_function"]),
    ],
)
def test_m_step_regression(decorator):
    jax.config.update("jax_enable_x64", True)

    # Fetch the data
    data_path = fetch_data("e_step_three_states.npz")
    data = np.load(data_path)

    X, y = data["X"], data["y"]
    new_sess = data["new_sess"]

    # E-step initial parameters
    initial_prob = data["initial_prob"]
    projection_weights = data["projection_weights"]
    transition_matrix = data["transition_matrix"]

    # E-step output
    xis = data["xis"]
    gammas = data["gammas"]
    ll_orig, ll_norm_orig = data["log_likelihood"], data["ll_norm"]
    alphas, betas = data["alphas"], data["betas"]

    obs = BernoulliObservations()

    log_likelihood = jax.vmap(
        lambda x, z: obs.likelihood(x, z, aggregate_sample_scores=lambda w: w),
        in_axes=(None, 1),
        out_axes=1,
    )

    optimal_weights, new_initial_prob, new_transition_prob, _ = run_m_step(
        X,
        y.flatten(),
        gammas.T,
        xis,
        projection_weights,
        inverse_link_function=obs.inverse_link_function,
        log_likelihood_func=log_likelihood,
        is_new_session=new_sess.flatten().astype(bool),
    )
    # np.testing.assert_almost_equal(1, optimal_weights, decimal=4)
