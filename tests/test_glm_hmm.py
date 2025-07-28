from functools import partial

import jax
import numpy as np
import pytest

from nemos.fetch import fetch_data
from nemos.glm_hmm import forward_backward, run_m_step
from nemos.observation_models import BernoulliObservations


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

    likelihood = jax.vmap(
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
            likelihood_func=likelihood,
            inverse_link_function=obs.inverse_link_function,
            is_new_session=new_sess.flatten().astype(bool),
        )
    )

    # First testing alphas and betas because they are computed first
    np.testing.assert_almost_equal(alphas_nemos, alphas.T, decimal=4)
    np.testing.assert_almost_equal(betas_nemos, betas.T, decimal=4)

    # testing log likelihood and normalized log likelihood
    np.testing.assert_almost_equal(ll_nemos, ll_orig, decimal=4)
    np.testing.assert_almost_equal(ll_norm_nemos, ll_norm_orig, decimal=4)

    # Next testing xis and gammas because they depend on alphas and betas
    # Testing Eq. 13.43 of Bishop
    np.testing.assert_almost_equal(gammas_nemos, gammas.T, decimal=4)
    # Testing Eq. 13.65 of Bishop
    np.testing.assert_almost_equal(xis_nemos, xis, decimal=4)


@pytest.mark.parametrize(
    "decorator",
    [
        lambda x: x,
        partial(
            jax.jit,
            static_argnames=["negative_log_likelihood_func", "inverse_link_function"],
        ),
    ],
)
def test_run_m_step_regression(decorator):
    jax.config.update("jax_enable_x64", True)

    # Fetch the data
    # TODO change for fetch after edoardo has uploaded mstep data to server
    # data_path = fetch_data("m_step_three_states.npz")
    # TODO write test for inputs and outputs of log likelihood function
    data_path = "_scripts/data/m_step_three_states.npz"
    data = np.load(data_path)

    # Design matrix and observed choices
    X, y = data["X"], data["y"]

    # M-step input
    gammas = data["gammas"]
    xis = data["xis"]
    projection_weights = data["projection_weights"]
    new_sess = data["new_sess"]

    # M-step output
    optimized_projection_weights = data["optimized_projection_weights"]
    new_initial_prob = data["new_initial_prob"]
    new_transition_prob = data["new_transition_prob"]

    # Initialize nemos observation model
    obs = BernoulliObservations()

    # Define negative log likelihood vmap function
    negative_log_likelihood = jax.vmap(
        lambda x, z: obs._negative_log_likelihood(
            x, z, aggregate_sample_scores=lambda w: w
        ),
        in_axes=(None, 1),
        out_axes=1,
    )

    (
        optimized_projection_weights_nemos,
        new_initial_prob_nemos,
        new_transition_prob_nemos,
        _,
    ) = run_m_step(
        X,
        y.flatten(),
        gammas.T,  # TODO having some data transposed and other untransposed is a bit confusing - probably should change that in the data generating function.
        xis,
        projection_weights,
        inverse_link_function=obs.inverse_link_function,
        negative_log_likelihood_func=negative_log_likelihood,
        is_new_session=new_sess.flatten().astype(
            bool
        ),  # TODO same as above todo comment
        solver_kwargs={"tol": 10**-12},
    )

    # Testing Eq. 13.18 of Bishop
    np.testing.assert_almost_equal(new_initial_prob_nemos, new_initial_prob, decimal=4)
    # Testing Eq. 13.19 of Bishop
    np.testing.assert_almost_equal(
        new_transition_prob_nemos, new_transition_prob, decimal=4
    )
    # Testing output of negative log likelihood
    np.testing.assert_almost_equal(
        optimized_projection_weights_nemos, optimized_projection_weights, decimal=4
    )

@pytest.mark.parametrize(
    "decorator",
    [
        lambda x: x,
        partial(jax.jit, static_argnames=["negative_log_likelihood_func", "inverse_link_function"]),
    ],
)
def test_hmm_negative_log_likelihood_regression(decorator):



    jax.config.update("jax_enable_x64", True)

    # Fetch the data
    # TODO change for fetch after edoardo has uploaded mstep data to server
    #data_path = fetch_data("e_step_three_states.npz")
    # TODO write test for inputs and outputs of log likelihood function
    data_path = "_scripts/data/m_step_three_states.npz"
    data = np.load(data_path)

    # Design matrix and observed choices
    X, y = data["X"], data["y"]

    # M-step input
    gammas = data["gammas"]
    xis = data["xis"]
    projection_weights = data["projection_weights"]
    new_sess = data["new_sess"]

    # Initialize nemos observation model
    obs = BernoulliObservations()

    # Define negative log likelihood vmap function
    negative_log_likelihood = jax.vmap(
        lambda x, z: obs._negative_log_likelihood(x, z, aggregate_sample_scores=lambda w: w),
        in_axes=(None, 1),
        out_axes=1,
    )


