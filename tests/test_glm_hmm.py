import itertools
from functools import partial

import jax
import numpy as np
import pytest
from hmmlearn import hmm

from nemos.fetch import fetch_data
from nemos.glm import GLM
from nemos.glm_hmm.expectation_maximization import (
    backward_pass,
    compute_xi,
    forward_backward,
    forward_pass,
    hmm_negative_log_likelihood,
    max_sum,
    run_m_step,
)
from nemos.observation_models import BernoulliObservations, PoissonObservations
from nemos.third_party.jaxopt.jaxopt import LBFGS


def viterbi_with_hmmlearn(
    log_emission, transition_proba, init_proba, is_new_session=None
):
    """
    Use hmmlearn's Viterbi algorithm with custom log probabilities.

    Parameters
    ----------
    log_emission : np.ndarray, shape (T, K)
        Log emission probabilities for each time step and state
    transition_proba : np.ndarray, shape (K, K)
        Log transition probability matrix [from_state, to_state]
    init_proba : np.ndarray, shape (K,)
        Log initial state probabilities
    is_new_session:
        Either None or array of shape (T,) of 0s and 1s, where 1s mark
        the beginning of a new session.

    Returns
    -------
    state_sequence : np.ndarray, shape (T,)
        Most likely state sequence (as integer indices 0 to K-1)
    """
    K = log_emission.shape[1]
    # Create a CategoricalHMM (this is what replaced MultinomialHMM)
    model = hmm.CategoricalHMM(n_components=K)

    # Set the HMM parameters (convert from log space)
    model.startprob_ = init_proba
    model.transmat_ = transition_proba
    # Set dummy emission probabilities (required but will be overridden)
    model.emissionprob_ = np.ones((K, 2)) / 2
    if is_new_session is None:
        slices = [slice(None)]
    else:
        session_start = np.where(is_new_session)[0]
        session_end = np.concatenate([session_start[1:], [len(is_new_session)]])
        slices = [slice(s, e) for s, e in zip(session_start, session_end)]

    map_path = []
    for sl in slices:
        map_path.append(single_session_viterbi_with_hmmlearn(model, log_emission[sl]))
    return np.concatenate(map_path)


def single_session_viterbi_with_hmmlearn(model, log_emission):
    """
    Use hmmlearn's Viterbi algorithm with custom log probabilities.

    Parameters
    ----------
    model:
        The hmm model to be patched.
    log_emission : np.ndarray, shape (T, K)
        Log emission probabilities for each time step and state
    log_transition : np.ndarray, shape (K, K)
        Log transition probability matrix [from_state, to_state]
    log_init : np.ndarray, shape (K,)
        Log initial state probabilities

    Returns
    -------
    state_sequence : np.ndarray, shape (T,)
        Most likely state sequence (as integer indices 0 to K-1)
    """
    T, K = log_emission.shape

    # Create dummy observations
    X = np.zeros((T, 1), dtype=np.int32)

    # Override the log-likelihood computation to use our custom emissions
    original_method = model._compute_log_likelihood
    model._compute_log_likelihood = lambda X: log_emission

    try:
        # Run Viterbi decoding
        _, state_sequence = model.decode(X, algorithm="viterbi")
        return state_sequence
    finally:
        # Restore original method
        model._compute_log_likelihood = original_method


def prepare_solver_for_m_step_single_neuron(
    X, y, initial_prob, transition_prob, glm_params, new_sess, obs
):
    (coef, intercept) = glm_params
    likelihood = jax.vmap(
        lambda x, z: obs.likelihood(x, z, aggregate_sample_scores=lambda w: w),
        in_axes=(None, 1),
        out_axes=1,
    )
    gammas, xis, _, _, _, _ = forward_backward(
        X,
        y,
        initial_prob,
        transition_prob,
        (coef, intercept),
        likelihood_func=likelihood,
        inverse_link_function=obs.default_inverse_link_function,
        is_new_session=new_sess.astype(bool),
    )

    # Define negative log likelihood vmap function


def prepare_partial_hmm_nll_single_neuron(obs):
    # Define nll vmap function
    negative_log_likelihood = jax.vmap(
        lambda x, z: obs._negative_log_likelihood(
            x, z, aggregate_sample_scores=lambda w: w
        ),
        in_axes=(None, 1),
        out_axes=1,
    )

    # Solver
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

    return partial_hmm_negative_log_likelihood, solver


def prepare_gammas_and_xis_for_m_step_single_neuron(
    X, y, initial_prob, transition_prob, glm_params, new_sess, obs
):
    (coef, intercept) = glm_params
    likelihood = jax.vmap(
        lambda x, z: obs.likelihood(x, z, aggregate_sample_scores=lambda w: w),
        in_axes=(None, 1),
        out_axes=1,
    )
    gammas, xis, _, _, _, _ = forward_backward(
        X,
        y,
        initial_prob,
        transition_prob,
        (coef, intercept),
        likelihood_func=likelihood,
        inverse_link_function=obs.default_inverse_link_function,
        is_new_session=new_sess.astype(bool),
    )

    return gammas, xis


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


@pytest.fixture(scope="module")
def generate_data_multi_state():
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
    return new_sess, initial_prob, transition_prob, coef, intercept, X, y


@pytest.fixture(scope="module")
def generate_data_multi_state_population():
    np.random.seed(44)
    jax.config.update("jax_enable_x64", True)

    # E-step initial parameters
    n_states, n_neurons, n_samples = 5, 3, 100
    initial_prob = np.random.uniform(size=(n_states))
    initial_prob /= np.sum(initial_prob)
    transition_prob = np.random.uniform(size=(n_states, n_states))
    transition_prob /= np.sum(transition_prob, axis=0)
    transition_prob = transition_prob.T
    coef, intercept = np.random.randn(2, n_neurons, n_states), np.random.randn(
        n_neurons, n_states
    )

    X = np.random.randn(n_samples, 2)
    y = np.zeros((n_samples, n_neurons))
    for i, k in enumerate(range(0, 100, 10)):
        sl = slice(k, k + 10)
        state = i % n_states
        rate = np.exp(X[sl].dot(coef[..., state]) + intercept[..., state])
        y[sl] = np.random.poisson(rate)

    new_sess = np.zeros(n_samples)
    new_sess[[0, 10, 90]] = 1
    return new_sess, initial_prob, transition_prob, coef, intercept, X, y


def test_for_loop_forward_step(generate_data_multi_state):
    new_sess, initial_prob, transition_prob, coef, intercept, X, y = (
        generate_data_multi_state
    )

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


def test_for_loop_backward_step(generate_data_multi_state):
    new_sess, initial_prob, transition_prob, coef, intercept, X, y = (
        generate_data_multi_state
    )
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


@pytest.fixture
def single_state_inputs():
    np.random.seed(42)
    initial_prob = np.ones(1)
    transition_prob = np.ones((1, 1))
    coef, intercept = np.random.randn(2, 1), np.random.randn(1)
    X = np.random.randn(10, 2)
    rate = np.exp(X.dot(coef) + intercept)
    y = np.random.poisson(rate[:, 0])
    return initial_prob, transition_prob, coef, intercept, X, rate, y


def test_single_state_estep(single_state_inputs):
    """Single state forward pass posteriors reduces to ones (there is a single state)."""

    initial_prob, transition_prob, coef, intercept, X, rate, y = single_state_inputs
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


def expected_log_likelihood_wrt_transitions(
    transition_prob, xis, dirichlet_alphas=None
):
    likelihood = jax.numpy.sum(xis * jax.numpy.log(transition_prob))
    if dirichlet_alphas is None:
        return likelihood
    prior = (jax.numpy.log(transition_prob) * (dirichlet_alphas - 1)).sum()
    return likelihood + prior


def expected_log_likelihood_wrt_initial_prob(
    initial_prob, gammas, dirichlet_alphas=None
):
    likelihood = jax.numpy.sum(gammas * jax.numpy.log(initial_prob))
    if dirichlet_alphas is None:
        return likelihood
    prior = (jax.numpy.log(initial_prob) * (dirichlet_alphas - 1)).sum()
    return likelihood + prior


def lagrange_mult_loss(param, args, loss, **kwargs):
    proba, lam = param
    n_states = proba.shape[0]
    if proba.ndim == 2:
        constraint = proba.sum(axis=1) - jax.numpy.ones(n_states)
    else:
        constraint = proba.sum() - 1
    lagrange_mult_term = (lam * constraint).sum()
    return loss(proba, args, **kwargs) + lagrange_mult_term


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

    partial_hmm_negative_log_likelihood, solver = prepare_partial_hmm_nll_single_neuron(
        obs
    )

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

    # check maximization analytical
    # Transition Probability:
    # 1) Compute the lagrange multiplier at the maximum
    #    This is the grad of the loss at the probabilities.
    lagrange_multiplier = -jax.grad(expected_log_likelihood_wrt_transitions)(
        new_transition_prob, xis
    ).mean(
        axis=1
    )  # note that the lagrange mult makes the gradient all the same for each prob.
    # 2) Check that the gradient of the loss is zero
    grad_objective = jax.grad(lagrange_mult_loss)
    (grad_at_transition, grad_at_lagr) = grad_objective(
        (new_transition_prob, lagrange_multiplier),
        xis,
        expected_log_likelihood_wrt_transitions,
    )
    np.testing.assert_array_almost_equal(
        grad_at_transition, np.zeros_like(new_transition_prob)
    )
    np.testing.assert_array_almost_equal(
        grad_at_lagr, np.zeros_like(lagrange_multiplier)
    )
    # Initial probability:
    sum_gammas = np.sum(gammas[np.where(new_sess)[0]], axis=0)
    lagrange_multiplier = -jax.grad(expected_log_likelihood_wrt_initial_prob)(
        new_initial_prob, sum_gammas
    ).mean()  # note that the lagrange mult makes the gradient all the same for each prob.
    grad_objective = jax.grad(lagrange_mult_loss)
    (grad_at_init, grad_at_lagr) = grad_objective(
        (new_initial_prob, lagrange_multiplier),
        sum_gammas,
        expected_log_likelihood_wrt_initial_prob,
    )
    np.testing.assert_array_almost_equal(grad_at_init, np.zeros_like(new_initial_prob))
    np.testing.assert_array_almost_equal(
        grad_at_lagr, np.zeros_like(lagrange_multiplier)
    )


def test_single_state_mstep(single_state_inputs):
    """Single state forward pass posteriors reduces to a GLM."""
    initial_prob, transition_prob, coef, intercept, X, rate, y = single_state_inputs
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

    # xis are a sum of the ones over valid entires
    xis = compute_xi(
        alphas,
        betas,
        conditionals,
        norm,
        new_sess,
        transition_prob,
    )

    partial_hmm_negative_log_likelihood, solver = prepare_partial_hmm_nll_single_neuron(
        obs
    )

    (
        optimized_projection_weights_nemos,
        new_initial_prob_nemos,
        new_transition_prob_nemos,
        state,
    ) = run_m_step(
        X,
        y,
        alphas * betas,
        xis,
        (np.zeros_like(coef), np.zeros_like(intercept)),
        is_new_session=new_sess.astype(bool),
        solver_run=solver.run,
    )
    glm = GLM(
        observation_model=obs, solver_name="LBFGS", solver_kwargs={"tol": 10**-13}
    )
    glm.fit(X, y)
    # test that the glm coeff and intercept matches with the m-step output
    np.testing.assert_array_almost_equal(
        glm.coef_, optimized_projection_weights_nemos[0].flatten()
    )
    np.testing.assert_array_almost_equal(
        glm.intercept_, optimized_projection_weights_nemos[1].flatten()
    )

    # test that the transition and initial probabilities are all ones.
    np.testing.assert_array_equal(new_initial_prob_nemos, np.ones_like(initial_prob))
    np.testing.assert_array_equal(
        new_transition_prob_nemos, np.ones_like(new_transition_prob_nemos)
    )

    # check expected shapes
    assert new_transition_prob_nemos.shape == (1, 1)
    assert new_initial_prob_nemos.shape == (1,)
    assert optimized_projection_weights_nemos[0].shape == (2, 1)
    assert optimized_projection_weights_nemos[1].shape == (1,)


def test_maximization_with_prior(generate_data_multi_state):
    new_sess, initial_prob, transition_prob, coef, intercept, X, y = (
        generate_data_multi_state
    )

    obs = PoissonObservations()
    _, solver = prepare_partial_hmm_nll_single_neuron(obs)

    gammas, xis = prepare_gammas_and_xis_for_m_step_single_neuron(
        X, y, initial_prob, transition_prob, (coef, intercept), new_sess, obs
    )

    alphas_transition = np.random.uniform(1, 3, size=transition_prob.shape)
    alphas_init = np.random.uniform(1, 3, size=initial_prob.shape)

    (
        optimized_projection_weights_nemos,
        new_initial_prob,
        new_transition_prob,
        state,
    ) = run_m_step(
        X,
        y,
        gammas,
        xis,
        (np.zeros_like(coef), np.zeros_like(intercept)),
        is_new_session=new_sess.astype(bool),
        solver_run=solver.run,
        dirichlet_prior_alphas_transition=alphas_transition,
        dirichlet_prior_alphas_init_prob=alphas_init,
    )

    lagrange_multiplier = -jax.grad(expected_log_likelihood_wrt_transitions)(
        new_transition_prob, xis, dirichlet_alphas=alphas_transition
    ).mean(
        axis=1
    )  # note that the lagrange mult makes the gradient all the same for each prob.
    # 2) Check that the gradient of the loss is zero
    grad_objective = jax.grad(lagrange_mult_loss)
    (grad_at_transition, grad_at_lagr) = grad_objective(
        (new_transition_prob, lagrange_multiplier),
        xis,
        expected_log_likelihood_wrt_transitions,
        dirichlet_alphas=alphas_transition,
    )
    np.testing.assert_array_almost_equal(
        grad_at_transition, np.zeros_like(new_transition_prob)
    )
    np.testing.assert_array_almost_equal(
        grad_at_lagr, np.zeros_like(lagrange_multiplier)
    )
    # Initial probabilities:
    sum_gammas = np.sum(gammas[np.where(new_sess)[0]], axis=0)
    lagrange_multiplier = -jax.grad(expected_log_likelihood_wrt_initial_prob)(
        new_initial_prob, sum_gammas, dirichlet_alphas=alphas_init
    ).mean()  # note that the lagrange mult makes the gradient all the same for each prob.
    # 2) Check that the gradient of the loss is zero
    grad_objective = jax.grad(lagrange_mult_loss)
    (grad_at_init, grad_at_lagr) = grad_objective(
        (new_initial_prob, lagrange_multiplier),
        sum_gammas,
        expected_log_likelihood_wrt_initial_prob,
        dirichlet_alphas=alphas_init,
    )
    np.testing.assert_array_almost_equal(grad_at_init, np.zeros_like(new_initial_prob))
    np.testing.assert_array_almost_equal(
        grad_at_lagr, np.zeros_like(lagrange_multiplier)
    )


def test_e_and_m_step_for_population(generate_data_multi_state_population):
    """Run E and M step fitting a population."""
    new_sess, initial_prob, transition_prob, coef, intercept, X, y = (
        generate_data_multi_state_population
    )
    obs = PoissonObservations()

    # Wrap likelihood_func to avoid aggregating over samples
    def likelihood_per_sample(x, z):
        return obs.log_likelihood(x, z, aggregate_sample_scores=lambda s: s)

    def negative_log_likelihood_per_sample(x, z):
        return obs._negative_log_likelihood(x, z, aggregate_sample_scores=lambda s: s)

    # Vectorize over the states axis
    state_axes = 2
    likelihood_per_sample = jax.vmap(
        likelihood_per_sample,
        in_axes=(None, state_axes),
        out_axes=state_axes,
    )

    def likelihood(y, rate):
        log_like = likelihood_per_sample(y, rate)
        # Multi-neuron case: sum log-likelihoods across neurons
        log_like = log_like.sum(axis=1)
        return jax.numpy.exp(log_like)

    gammas, xis, _, _, _, _ = forward_backward(
        X,
        y,
        initial_prob,
        transition_prob,
        (coef, intercept),
        likelihood_func=likelihood,
        inverse_link_function=obs.default_inverse_link_function,
        is_new_session=new_sess.astype(bool),
    )

    vmap_nll = jax.vmap(
        negative_log_likelihood_per_sample,
        in_axes=(None, state_axes),
        out_axes=state_axes,
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
            negative_log_likelihood_func=vmap_nll,
        )

    alphas_transition = np.random.uniform(1, 3, size=transition_prob.shape)
    alphas_init = np.random.uniform(1, 3, size=initial_prob.shape)
    solver = LBFGS(partial_hmm_negative_log_likelihood, tol=10**-13)
    (
        optimized_projection_weights_nemos,
        new_initial_prob,
        new_transition_prob,
        state,
    ) = run_m_step(
        X,
        y,
        gammas,
        xis,
        (np.zeros_like(coef), np.zeros_like(intercept)),
        is_new_session=new_sess.astype(bool),
        solver_run=solver.run,
        dirichlet_prior_alphas_transition=alphas_transition,
        dirichlet_prior_alphas_init_prob=alphas_init,
    )


@pytest.mark.parametrize("state_idx", range(5))
def test_m_step_set_alpha_init_to_inf(generate_data_multi_state, state_idx):
    new_sess, initial_prob, transition_prob, coef, intercept, X, y = (
        generate_data_multi_state
    )

    obs = PoissonObservations()

    _, solver = prepare_partial_hmm_nll_single_neuron(obs)

    gammas, xis = prepare_gammas_and_xis_for_m_step_single_neuron(
        X, y, initial_prob, transition_prob, (coef, intercept), new_sess, obs
    )

    alphas_transition = np.random.uniform(1, 3, size=transition_prob.shape)
    alphas_init = np.random.uniform(1, 3, size=initial_prob.shape)
    alphas_init[state_idx] = 10**20

    (
        optimized_projection_weights_nemos,
        new_initial_prob,
        new_transition_prob,
        state,
    ) = run_m_step(
        X,
        y,
        gammas,
        xis,
        (np.zeros_like(coef), np.zeros_like(intercept)),
        is_new_session=new_sess.astype(bool),
        solver_run=solver.run,
        dirichlet_prior_alphas_transition=alphas_transition,
        dirichlet_prior_alphas_init_prob=alphas_init,
    )
    np.testing.assert_array_almost_equal(
        new_initial_prob, np.eye(new_initial_prob.shape[0])[state_idx]
    )


@pytest.mark.parametrize("row, col", itertools.product(range(3), range(3)))
def test_m_step_set_alpha_transition_to_inf(generate_data_multi_state, row, col):
    new_sess, initial_prob, transition_prob, coef, intercept, X, y = (
        generate_data_multi_state
    )

    obs = PoissonObservations()

    _, solver = prepare_partial_hmm_nll_single_neuron(obs)

    gammas, xis = prepare_gammas_and_xis_for_m_step_single_neuron(
        X, y, initial_prob, transition_prob, (coef, intercept), new_sess, obs
    )

    alphas_transition = np.random.uniform(1, 3, size=transition_prob.shape)
    alphas_init = np.random.uniform(1, 3, size=initial_prob.shape)
    alphas_transition[row, col] = 10**20

    (
        optimized_projection_weights_nemos,
        new_initial_prob,
        new_transition_prob,
        state,
    ) = run_m_step(
        X,
        y,
        gammas,
        xis,
        (np.zeros_like(coef), np.zeros_like(intercept)),
        is_new_session=new_sess.astype(bool),
        solver_run=solver.run,
        dirichlet_prior_alphas_transition=alphas_transition,
        dirichlet_prior_alphas_init_prob=alphas_init,
    )
    np.testing.assert_array_almost_equal(
        new_transition_prob[row, :], np.eye(new_initial_prob.shape[0])[col]
    )


def test_m_step_set_alpha_init_to_1(generate_data_multi_state):
    new_sess, initial_prob, transition_prob, coef, intercept, X, y = (
        generate_data_multi_state
    )

    obs = PoissonObservations()
    _, solver = prepare_partial_hmm_nll_single_neuron(obs)

    gammas, xis = prepare_gammas_and_xis_for_m_step_single_neuron(
        X, y, initial_prob, transition_prob, (coef, intercept), new_sess, obs
    )

    alphas_transition = np.random.uniform(1, 3, size=transition_prob.shape)
    alphas_init = np.ones(initial_prob.shape)

    (
        prior_optimized_projection_weights_nemos,
        prior_initial_prob,
        prior_transition_prob,
        state,
    ) = run_m_step(
        X,
        y,
        gammas,
        xis,
        (np.zeros_like(coef), np.zeros_like(intercept)),
        is_new_session=new_sess.astype(bool),
        solver_run=solver.run,
        dirichlet_prior_alphas_transition=alphas_transition,
        dirichlet_prior_alphas_init_prob=alphas_init,
    )
    (
        optimized_projection_weights_nemos,
        no_prior_initial_prob,
        no_prior_transition_prob,
        state,
    ) = run_m_step(
        X,
        y,
        gammas,
        xis,
        (np.zeros_like(coef), np.zeros_like(intercept)),
        is_new_session=new_sess.astype(bool),
        solver_run=solver.run,
        dirichlet_prior_alphas_transition=alphas_transition,
        dirichlet_prior_alphas_init_prob=None,
    )
    np.testing.assert_array_almost_equal(no_prior_initial_prob, prior_initial_prob)
    np.testing.assert_array_almost_equal(
        no_prior_transition_prob, prior_transition_prob
    )
    np.testing.assert_array_almost_equal(
        optimized_projection_weights_nemos[0],
        prior_optimized_projection_weights_nemos[0],
    )
    np.testing.assert_array_almost_equal(
        optimized_projection_weights_nemos[1],
        prior_optimized_projection_weights_nemos[1],
    )


def test_m_step_set_alpha_transition_to_1(generate_data_multi_state):
    new_sess, initial_prob, transition_prob, coef, intercept, X, y = (
        generate_data_multi_state
    )

    obs = PoissonObservations()
    _, solver = prepare_partial_hmm_nll_single_neuron(obs)

    gammas, xis = prepare_gammas_and_xis_for_m_step_single_neuron(
        X, y, initial_prob, transition_prob, (coef, intercept), new_sess, obs
    )
    alphas_transition = np.ones(transition_prob.shape)
    alphas_init = np.random.uniform(1, 3, size=initial_prob.shape)

    (
        prior_optimized_projection_weights_nemos,
        prior_initial_prob,
        prior_transition_prob,
        state,
    ) = run_m_step(
        X,
        y,
        gammas,
        xis,
        (np.zeros_like(coef), np.zeros_like(intercept)),
        is_new_session=new_sess.astype(bool),
        solver_run=solver.run,
        dirichlet_prior_alphas_transition=alphas_transition,
        dirichlet_prior_alphas_init_prob=alphas_init,
    )
    (
        optimized_projection_weights_nemos,
        no_prior_initial_prob,
        no_prior_transition_prob,
        state,
    ) = run_m_step(
        X,
        y,
        gammas,
        xis,
        (np.zeros_like(coef), np.zeros_like(intercept)),
        is_new_session=new_sess.astype(bool),
        solver_run=solver.run,
        dirichlet_prior_alphas_transition=None,
        dirichlet_prior_alphas_init_prob=alphas_init,
    )
    np.testing.assert_array_almost_equal(no_prior_initial_prob, prior_initial_prob)
    np.testing.assert_array_almost_equal(
        no_prior_transition_prob, prior_transition_prob
    )
    np.testing.assert_array_almost_equal(
        optimized_projection_weights_nemos[0],
        prior_optimized_projection_weights_nemos[0],
    )
    np.testing.assert_array_almost_equal(
        optimized_projection_weights_nemos[1],
        prior_optimized_projection_weights_nemos[1],
    )


@pytest.mark.parametrize(
    "data_name",
    [
        "julia_regression_mstep_flat_prior.npz",  # Uniform priors
        "julia_regression_mstep_good_prior.npz",  # Priors coherent with true parameters
        "julia_regression_mstep_no_prior.npz",  # No prior / uninformative prior (alpha = 1)
    ],
)
def test_run_m_step_regression_priors_simulation(data_name):
    jax.config.update("jax_enable_x64", True)

    # Fetch the data
    data_path = fetch_data(data_name)
    data = np.load(data_path)

    # Design matrix and observed choices
    X, y = data["X"], data["y"]

    # Dirichlet priors
    dirichlet_prior_initial_prob = data[
        "dirichlet_prior_initial_prob"
    ]  # storing alphas (NOT alphas + 1)
    dirichlet_prior_transition_prob = data[
        "dirichlet_prior_transition_prob"
    ]  # storing alphas (NOT alphas + 1)

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

    # Prepare nll function & solver
    partial_hmm_negative_log_likelihood, solver = prepare_partial_hmm_nll_single_neuron(
        obs
    )

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

    np.testing.assert_almost_equal(
        new_transition_prob_nemos, new_transition_prob, decimal=5
    )  # changed number of decimals

    # Testing output of negative log likelihood
    np.testing.assert_almost_equal(n_ll_original, n_ll_nemos, decimal=10)

    # Testing intercept and optimized projection weights
    jax.tree_util.tree_map(
        lambda x, y: np.testing.assert_almost_equal(x, y, decimal=6),
        (opt_coef, opt_intercept),
        optimized_projection_weights_nemos,
    )


@pytest.mark.parametrize("use_new_sess", [True, False])
def test_viterbi_against_hmmlearn(use_new_sess):
    data = np.load(fetch_data("em_three_states.npz"))
    initial_prob = data["initial_prob"]
    transition_prob = data["transition_prob"]
    X, y = data["X"], data["y"]
    new_session = data["new_sess"] if use_new_sess else None
    intercept, coef = data["projection_weights"][:1], data["projection_weights"][1:]

    obs = BernoulliObservations()
    inverse_link_function = obs.default_inverse_link_function
    predicted_rate_given_state = inverse_link_function(X[..., 1:] @ coef + intercept)

    log_like_func = jax.vmap(
        lambda x, z: obs.log_likelihood(x, z, aggregate_sample_scores=lambda w: w),
        in_axes=(None, 1),
        out_axes=1,
    )
    log_emission_array = log_like_func(y, predicted_rate_given_state)
    map_path = max_sum(
        X[:, 1:],
        y,
        initial_prob,
        transition_prob,
        (coef, intercept),
        inverse_link_function,
        log_like_func,
        is_new_session=new_session,
        return_index=True,
    )
    hmmlearn_map_path = viterbi_with_hmmlearn(
        log_emission_array, transition_prob, initial_prob, is_new_session=new_session
    )
    np.testing.assert_array_equal(
        map_path.astype(np.int32), hmmlearn_map_path.astype(np.int32)
    )


@pytest.mark.parametrize("use_new_sess", [True, False])
@pytest.mark.parametrize("return_index", [True, False])
def test_viterbi_return_index(use_new_sess, return_index):
    data = np.load(fetch_data("em_three_states.npz"))
    initial_prob = data["initial_prob"]
    transition_prob = data["transition_prob"]
    X, y = data["X"], data["y"]
    new_session = data["new_sess"][:100] if use_new_sess else None
    intercept, coef = data["projection_weights"][:1], data["projection_weights"][1:]

    obs = BernoulliObservations()
    inverse_link_function = obs.default_inverse_link_function

    n_states = initial_prob.shape[0]

    log_like_func = jax.vmap(
        lambda x, z: obs.log_likelihood(x, z, aggregate_sample_scores=lambda w: w),
        in_axes=(None, 1),
        out_axes=1,
    )
    map_path = max_sum(
        X[:100, 1:],
        y[:100],
        initial_prob,
        transition_prob,
        (coef, intercept),
        inverse_link_function,
        log_like_func,
        is_new_session=new_session,
        return_index=return_index,
    )
    if return_index:
        assert map_path.shape == (100,)
        set(np.unique(map_path).astype(np.int32)).issubset(np.arange(n_states))
    else:
        assert map_path.shape == (100, n_states)
        np.testing.assert_array_equal(
            np.unique(map_path).astype(np.int32), np.array([0, 1])
        )
