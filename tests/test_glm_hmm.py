from functools import partial

import jax
import numpy as np
import pytest

from nemos.fetch import fetch_data
from nemos.glm import GLM
from nemos.glm_hmm.expectation_maximization import (
    backward_pass,
    compute_xi,
    forward_backward,
    forward_pass,
    hmm_negative_log_likelihood,
    prepare_likelihood_func,
    run_m_step,
)
from nemos.observation_models import BernoulliObservations, PoissonObservations
from nemos.third_party.jaxopt.jaxopt import LBFGS


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
    np.random.seed(44)
    jax.config.update("jax_enable_x64", True)

    # E-step initial parameters
    n_states, n_samples = 5, 100
    initial_prob = np.random.uniform(size=(n_states))
    initial_prob /= np.sum(initial_prob)
    transition_prob = np.random.uniform(size=(n_states, n_states))
    transition_prob /= np.sum(transition_prob, axis=0)
    transition_prob = transition_prob.T
    coef, intercept = np.random.randn(2, n_states), np.random.randn(n_states)

    X = np.random.randn(n_samples, 2)
    y = np.zeros(n_samples)
    for i, k in enumerate(range(0, 100, 10)):
        sl = slice(k, k + 10)
        state = i % n_states
        rate = np.exp(X[sl].dot(coef[:, state]) + intercept[state])
        y[sl] = np.random.poisson(rate)

    new_sess = np.zeros(n_samples)
    new_sess[[0, 10, 90]] = 1

    obs = PoissonObservations()

    likelihood = jax.vmap(
        lambda x, z: obs.likelihood(x, z, aggregate_sample_scores=lambda w: w),
        in_axes=(None, 1),
        out_axes=1,
    )

    predicted_rate_given_state = obs.default_inverse_link_function(X @ coef + intercept)
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
    np.random.seed(43)
    jax.config.update("jax_enable_x64", True)

    # E-step initial parameters
    n_states, n_samples = 5, 100
    initial_prob = np.random.uniform(size=(n_states))
    initial_prob /= np.sum(initial_prob)
    transition_prob = np.random.uniform(size=(n_states, n_states))
    transition_prob /= np.sum(transition_prob, axis=0)
    transition_prob = transition_prob.T
    coef, intercept = np.random.randn(2, n_states), np.random.randn(n_states)

    X = np.random.randn(n_samples, 2)
    y = np.zeros(n_samples)
    for i, k in enumerate(range(0, 100, 10)):
        sl = slice(k, k + 10)
        state = i % n_states
        rate = np.exp(X[sl].dot(coef[:, state]) + intercept[state])
        y[sl] = np.random.poisson(rate)

    new_sess = np.zeros(n_samples)
    new_sess[[0, 10, 90]] = 1

    obs = PoissonObservations()

    likelihood = jax.vmap(
        lambda x, z: obs.likelihood(x, z, aggregate_sample_scores=lambda w: w),
        in_axes=(None, 1),
        out_axes=1,
    )

    predicted_rate_given_state = obs.default_inverse_link_function(X @ coef + intercept)
    conditionals = likelihood(y, predicted_rate_given_state)

    alphas, normalization = forward_pass(
        initial_prob, transition_prob, conditionals, new_sess
    )

    betas = backward_pass(transition_prob, conditionals, normalization, new_sess)
    betas_numpy = backward_step_numpy(
        conditionals, normalization, new_sess, transition_prob
    )
    np.testing.assert_almost_equal(betas_numpy, betas)


def test_single_state_estep():
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
    np.testing.assert_array_almost_equal(norm, conditionals[:, 0])

    # Note: alphas * betas is p(z_t | X), so it's automatically ones if the
    # two assertions passes, no need to check explicitly for  p(z_t | X).
    np.testing.assert_array_almost_equal(np.ones_like(alphas), alphas)
    np.testing.assert_array_almost_equal(np.ones_like(betas), betas)

    # xis are a sum of the ones over valid entires
    xis = compute_xi(
        alphas,
        betas,
        conditionals,
        norm,
        new_sess,
        transition_prob,
    )
    np.testing.assert_array_almost_equal(
        np.array([[alphas.shape[0] - sum(new_sess)]]).astype(xis), xis
    )


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

    # Testing projection weights
    jax.tree_util.tree_map(
        lambda x, y: np.testing.assert_almost_equal(x, y, decimal=6),
        (opt_coef, opt_intercept),
        optimized_projection_weights_nemos,
    )


@pytest.mark.parametrize("regularization", ["UnRegularized", "Ridge", "Lasso"])
@pytest.mark.parametrize("require_new_session", [True, False])
def test_run_em(regularization, require_new_session):
    jax.config.update("jax_enable_x64", True)

    # Fetch the data
    data_path = fetch_data("em_three_states.npz")
    data = np.load(data_path)

    # Design matrix and observed choices
    X, y = data["X"], data["y"]

    # Initial parameters
    initial_prob = data["initial_prob"]
    transition_prob = data["transition_prob"]
    projection_weights = data["projection_weights"]
    intercept, coef = projection_weights[:1], projection_weights[1:]
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
    glm = GLM(observation_model=obs, regularizer=regularization)
    glm.instantiate_solver(partial_hmm_negative_log_likelihood)
    solver_run = glm._solver_run
    # End of preparatory step.
    (
        posteriors,
        joint_posterior,
        learned_initial_prob,
        learned_transition,
        (learned_coef, learned_intercept),
    ) = em_glm_hmm(
        X[:, 1:],
        y,
        initial_prob=initial_prob,
        transition_prob=transition_prob,
        glm_params=(coef, intercept),
        is_new_session=new_sess.astype(bool) if require_new_session else None,
        inverse_link_function=inverse_link_function,
        likelihood_func=likelihood_func,
        solver_run=solver_run,
    )

    (
        _,
        _,
        _,
        log_likelihood_em,
        _,
        _,
    ) = forward_backward(
        X[:, 1:],  # drop intercept
        y,
        learned_initial_prob,
        learned_transition,
        (learned_coef, learned_intercept),
        likelihood_func=likelihood_func,
        inverse_link_function=obs.default_inverse_link_function,
    )
    (
        _,
        _,
        _,
        log_likelihood_true_params,
        _,
        _,
    ) = forward_backward(
        X[:, 1:],  # drop intercept
        y,
        initial_prob,
        transition_prob,
        (coef, intercept),
        likelihood_func=likelihood_func,
        inverse_link_function=obs.default_inverse_link_function,
    )
    assert (
        log_likelihood_true_params < log_likelihood_em
    ), "log-likelihood did not increase."


@pytest.mark.parametrize("n_neurons", [5])
def test_check_em(n_neurons):
    jax.config.update("jax_enable_x64", True)

    # Fetch the data
    data_path = fetch_data(f"glm_hmm_simulation_n_neurons_{n_neurons}_seed_123.npz")
    data = np.load(data_path)

    # Design matrix and observed choices
    X, y = data["design_matrix"], data["counts"]

    # Initial parameters
    initial_prob = data["initial_prob"]
    transition_prob = data["transition_prob"]
    projection_weights = data["projection_weights"]
    intercept, coef = projection_weights[:1], projection_weights[1:]

    # Start of the preparatory steps that will be carried out by the GLMHMM class.
    is_population_glm = n_neurons > 1
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
    glm = GLM(observation_model=obs, solver_name="LBFGS")
    glm.instantiate_solver(partial_hmm_negative_log_likelihood)
    solver_run = glm._solver_run
    # End of preparatory step.

    # add small noise to initial prob & projection weights
    np.random.seed(123)
    init_pb = initial_prob + np.random.uniform(0, 0.1)
    init_pb /= init_pb.sum()
    proj_weights = projection_weights + np.random.randn(*projection_weights.shape)

    # sticky prior (not equal to original)
    transition_pb = np.ones(transition_prob.shape) * 0.05
    transition_pb[np.diag_indices(transition_prob.shape[1])] = 0.9

    (
        posteriors_noisy_params,
        joint_posterior_noisy_params,
        log_likelihood_noisy_params,
        log_likelihood_norm_noisy_params,
        alphas_noisy_params,
        betas_noisy_params,
    ) = forward_backward(
        X[:, 1:],  # drop intercept
        y,
        init_pb,
        transition_pb,
        (proj_weights[1:], proj_weights[:1]),
        likelihood_func=likelihood_func,
        inverse_link_function=obs.default_inverse_link_function,
    )

    latent_states = data["latent_states"]
    corr_matrix_before_em = np.corrcoef(latent_states.T, posteriors_noisy_params.T)[
        : latent_states.shape[1], latent_states.shape[1] :
    ]
    max_corr_before_em = np.max(corr_matrix_before_em, axis=1)

    (
        posteriors,
        joint_posterior,
        learned_initial_prob,
        learned_transition,
        (learned_coef, learned_intercept),
    ) = em_glm_hmm(
        X[:, 1:],
        jax.numpy.squeeze(y),
        initial_prob=init_pb,
        transition_prob=transition_pb,
        glm_params=(proj_weights[1:], proj_weights[:1]),
        inverse_link_function=inverse_link_function,
        likelihood_func=likelihood_func,
        solver_run=solver_run,
        tol=10**-10,
    )
    (
        _,
        _,
        _,
        log_likelihood_em,
        _,
        _,
    ) = forward_backward(
        X[:, 1:],  # drop intercept
        y,
        learned_initial_prob,
        learned_transition,
        (learned_coef, learned_intercept),
        likelihood_func=likelihood_func,
        inverse_link_function=obs.default_inverse_link_function,
    )

    # find state mapping
    corr_matrix = np.corrcoef(latent_states.T, posteriors.T)[
        : latent_states.shape[1], latent_states.shape[1] :
    ]
    max_corr = np.max(corr_matrix, axis=1)
    print("\nMAX CORR", max_corr)
    assert np.all(max_corr > 0.95), "State recovery failed."
    assert np.all(
        max_corr > max_corr_before_em
    ), "Latent state recovery did not improve."
    assert log_likelihood_noisy_params < log_likelihood_em, "Log-likelihood decreased."
