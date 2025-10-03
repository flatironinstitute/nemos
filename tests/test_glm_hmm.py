from functools import partial

import jax
import numpy as np
import pytest

from nemos.fetch import fetch_data
from nemos.glm import GLM
from nemos.glm_hmm.expectation_maximization import (
    em_glm_hmm_simplified,
    forward_backward,
    hmm_negative_log_likelihood,
    prepare_likelihood_func,
    run_m_step,
)
from nemos.observation_models import BernoulliObservations
from nemos.third_party.jaxopt.jaxopt import LBFGS


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

    # E-step initial parameters
    initial_prob = data["initial_prob"]
    projection_weights = data["projection_weights"]
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
            X,
            y,
            initial_prob,
            transition_prob,
            projection_weights,
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
def test_hmm_negative_log_likelihood_regression(decorator):
    jax.config.update("jax_enable_x64", True)

    # Fetch the data
    data_path = fetch_data("em_three_states.npz")
    data = np.load(data_path)

    # Design matrix and observed choices
    X, y = data["X"], data["y"]

    # Likelihood input
    gammas = data["gammas"]
    projection_weights = data["projection_weights_nll"]

    # Negative LL output
    nll_m_step = data["nll_m_step"]

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

    nll_m_step_nemos = hmm_negative_log_likelihood(
        projection_weights,
        X,
        y,
        gammas,
        inverse_link_function=obs.default_inverse_link_function,
        negative_log_likelihood_func=negative_log_likelihood,
    )

    # Testing output of negative log likelihood
    np.testing.assert_almost_equal(nll_m_step_nemos, nll_m_step)


def test_run_m_step_regression():
    jax.config.update("jax_enable_x64", True)

    # Fetch the data
    data_path = fetch_data("em_three_states.npz")
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

    # solver
    def partial_hmm_negative_log_likelihood(
        weights, design_matrix, observations, posterior_prob
    ):
        return hmm_negative_log_likelihood(
            weights,
            X=design_matrix,
            y=observations,
            posteriors=posterior_prob,
            inverse_link_function=obs.default_inverse_link_function,
            negative_log_likelihood_func=negative_log_likelihood,
        )

    solver = LBFGS(partial_hmm_negative_log_likelihood, tol=10**-12)

    (
        optimized_projection_weights_nemos,
        new_initial_prob_nemos,
        new_transition_prob_nemos,
        state,
    ) = run_m_step(
        X,
        y,
        gammas,
        xis,
        projection_weights,
        is_new_session=new_sess.astype(bool),
        solver_run=solver.run,
    )

    n_ll_original = partial_hmm_negative_log_likelihood(
        optimized_projection_weights_nemos,
        X,
        y,
        gammas,
    )
    n_ll_nemos = partial_hmm_negative_log_likelihood(
        optimized_projection_weights,
        X,
        y,
        gammas,
    )

    # Testing Eq. 13.18 of Bishop
    np.testing.assert_almost_equal(new_initial_prob_nemos, new_initial_prob)
    # Testing Eq. 13.19 of Bishop

    np.testing.assert_almost_equal(new_transition_prob_nemos, new_transition_prob)

    # Testing output of negative log likelihood
    np.testing.assert_almost_equal(n_ll_original, n_ll_nemos, decimal=10)
    np.testing.assert_almost_equal(
        optimized_projection_weights, optimized_projection_weights_nemos, decimal=6
    )


@pytest.mark.parametrize("regularization", ["UnRegularized", "Ridge", "Lasso"])
def test_run_em(regularization):
    jax.config.update("jax_enable_x64", True)

    # Fetch the data
    data_path = fetch_data("em_three_states.npz")
    data = np.load(data_path)

    # Design matrix and observed choices
    X, y = data["X"], data["y"]

    # E-step initial parameters
    initial_prob = data["initial_prob"]
    projection_weights = data["projection_weights"]
    transition_prob = data["transition_prob"]

    # M-step input
    gammas = data["gammas"]
    xis = data["xis"]
    projection_weights = data["projection_weights"]
    new_sess = data["new_sess"]

    # Start of the preparatory steps that will be carried out by the GLMHMM class.
    is_population_glm = projection_weights.ndim > 2
    obs = BernoulliObservations()
    likelihood_func, negative_log_likelihood_func = prepare_likelihood_func(
        is_population_glm, obs.log_likelihood, obs._negative_log_likelihood, is_log=True
    )
    inverse_link_function = obs.default_inverse_link_function

    # closure for the static callables
    # NOTE: this is the _predict_and_compute_loss equivalent (aka, what it is used in
    # the numerical M-step).
    def partial_hmm_negative_log_likelihood(
        weights, design_matrix, observations, posterior_prob
    ):
        return hmm_negative_log_likelihood(
            weights,
            X=design_matrix,
            y=observations,
            posteriors=posterior_prob,
            inverse_link_function=inverse_link_function,
            negative_log_likelihood_func=negative_log_likelihood_func,
        )

    # use the BaseRegressor initialize_solver (this will be avaialble also in the GLMHHM class)
    glm = GLM(regularizer=regularization)
    glm.instantiate_solver(partial_hmm_negative_log_likelihood)
    solver_run = glm._solver_run
    em_glm_hmm_simplified(
        X,
        y,
        initial_prob=initial_prob,
        transition_prob=transition_prob,
        projection_weights=projection_weights,
        is_new_session=new_sess.astype(bool),
        inverse_link_function=inverse_link_function,
        likelihood_func=likelihood_func,
        solver_run=solver_run,
    )

    # End of preparatory step.
