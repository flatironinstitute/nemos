from functools import partial

import jax
import numpy as np
import pytest

from nemos.fetch import fetch_data
from nemos.glm_hmm.expectation_maximization import (
    forward_backward,
    hmm_negative_log_likelihood,
    run_m_step,
)
from nemos.observation_models import BernoulliObservations
from nemos.third_party.jaxopt.jaxopt import LBFGS

import pathlib


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
    intercept, coef = projection_weights[:1], projection_weights[1:]

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
        (coef, intercept),
        X[:, 1:],  # drop intercept column
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
    intercept, coef = projection_weights[:1], projection_weights[1:]
    new_sess = data["new_sess"]

    # M-step output
    optimized_projection_weights = data["optimized_projection_weights"]
    opt_intercept, opt_coef = (
        optimized_projection_weights[:1],
        optimized_projection_weights[1:],
    )
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

    solver = LBFGS(partial_hmm_negative_log_likelihood, tol=10**-13)

    (
        optimized_projection_weights_nemos,
        new_initial_prob_nemos,
        new_transition_prob_nemos,
        state,
    ) = run_m_step(
        X[:, 1:],  # drop intercept column
        y,
        gammas,
        xis,
        (coef, intercept),
        is_new_session=new_sess.astype(bool),
        solver_run=solver.run,
    )

    n_ll_nemos = partial_hmm_negative_log_likelihood(
        optimized_projection_weights_nemos,
        X[:, 1:],
        y,
        gammas,
    )
    n_ll_original = partial_hmm_negative_log_likelihood(
        (opt_coef, opt_intercept),
        X[:, 1:],
        y,
        gammas,
    )

    # Testing Eq. 13.18 of Bishop
    np.testing.assert_almost_equal(new_initial_prob_nemos, new_initial_prob)
    # Testing Eq. 13.19 of Bishop

    np.testing.assert_almost_equal(new_transition_prob_nemos, new_transition_prob)

    # Testing output of negative log likelihood
    np.testing.assert_almost_equal(n_ll_original, n_ll_nemos, decimal=10)

    jax.tree_util.tree_map(
        lambda x, y: np.testing.assert_almost_equal(x, y, decimal=6),
        (opt_coef, opt_intercept),
        optimized_projection_weights_nemos,
    )

@pytest.mark.parametrize(
    "data_name",
    [
        "julia_regression_mstep_flat_prior.npz",    # Uniform priors
        "julia_regression_mstep_good_prior.npz",    # Priors coherent with true parameters
        "julia_regression_mstep_no_prior.npz",      # No prior / uninformative prior (alpha = 1)
    ]
)
def test_run_m_step_regression_dirichlet_priors(data_name):
    jax.config.update("jax_enable_x64", True)

    # Fetch data
    ### Change after uploading to osf - this is where the data temporarily lives
    data_path = pathlib.Path(__file__).parent.parent / "_scripts" / "julia/"
    data = np.load(data_path.joinpath(data_name))   
    ###

    #data_path = fetch_data(data_name)
    #data = np.load(data_path)

    # Design matrix and observed choices
    X, y = data["X"], data["y"]

    # Dirichlet priors
    dirichlet_prior_initial_prob = data["dirichlet_prior_initial_prob"]  
    dirichlet_prior_transition_prob = data["dirichlet_prior_transition_prob"] 

    # M-step input
    gammas = data["gammas"]
    xis = data["xis"]
    projection_weights = data["projection_weights"]
    intercept, coef = projection_weights[:1], projection_weights[1:]
    new_sess = data["new_sess"]

    # M-step output
    optimized_projection_weights = data["optimized_projection_weights"]
    opt_intercept, opt_coef = (
        optimized_projection_weights[:1],
        optimized_projection_weights[1:],
    )
    new_initial_prob = data["new_initial_prob"]
    new_transition_prob = data["new_transition_prob"]

    # Initialize nemos observation model
    obs = BernoulliObservations()

    # Define nll vmap function
    # Vectorize the function f(x, z) along the second axis (axis=1) of z, while keeping x the same for all iterations.
    
    # Equivalent:
    # def negative_log_likelihood(x, z):
    #   results = []
    #   for j in range(z.shape[1]):  # loop over the 2nd axis of z
    #       results.append(obs._negative_log_likelihood(x, z[:, j], aggregate_sample_scores=lambda w: w))
    #   return jnp.stack(results, axis=1)

    # Not aggregating scores because aggregating already occurs in hmm_negative_log_likelihood - this is computing the likelihood for each observation. 

    negative_log_likelihood = jax.vmap(
        lambda x, z: obs._negative_log_likelihood(
            x, z, aggregate_sample_scores=lambda w: w
        ),
        in_axes=(None, 1),
        out_axes=1,
    )

    # Solver
    # wrapper function that “freezes in” two arguments of hmm_negative_log_likelihood: the inverse link function, and the (vectorized) negative log-likelihood function.
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
        ) # summing over axis one happening in hmm_negative_log_likelihood
    
    solver = LBFGS(partial_hmm_negative_log_likelihood, tol=10**-13)

    (
        optimized_projection_weights_nemos,
        new_initial_prob_nemos,
        new_transition_prob_nemos,
        state,
    ) = run_m_step(
        X[:, 1:],  # drop intercept column
        y,
        gammas,
        xis,
        (coef, intercept),
        is_new_session=new_sess.astype(bool),
        solver_run=solver.run,
        dirichlet_prior_alphas_init_prob=dirichlet_prior_initial_prob,
        dirichlet_prior_alphas_transition=dirichlet_prior_transition_prob,
    )

    # NLL with nemos input
    n_ll_nemos = partial_hmm_negative_log_likelihood(
        optimized_projection_weights_nemos,
        X[:, 1:],
        y,
        gammas,
    )

    # NLL with simulation input
    n_ll_original = partial_hmm_negative_log_likelihood(
        (opt_coef, opt_intercept),
        X[:, 1:],
        y,
        gammas,
    )

    # Testing Eq. 13.18 of Bishop
    np.testing.assert_almost_equal(new_initial_prob_nemos, new_initial_prob)
    # Testing Eq. 13.19 of Bishop

    np.testing.assert_almost_equal(new_transition_prob_nemos, new_transition_prob, decimal=5) # changed number of decimals

    #print(optimized_projection_weights, optimized_projection_weights_nemos)

    np.testing.assert_almost_equal(optimized_projection_weights[1:,:], optimized_projection_weights_nemos[0], decimal=5) # testing without intercept# changed number of decimals

    # Testing output of negative log likelihood
    np.testing.assert_almost_equal(n_ll_original, n_ll_nemos, decimal=10) 

    jax.tree_util.tree_map(
        lambda x, y: np.testing.assert_almost_equal(x, y, decimal=6),
        (opt_coef, opt_intercept),
        optimized_projection_weights_nemos,
    )