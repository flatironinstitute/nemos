import itertools
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hmmlearn import hmm
from numba.cpython.mathimpl import log_impl

from nemos.fetch import fetch_data
from nemos.glm import GLM
from nemos.glm_hmm.expectation_maximization import (
    GLMHMMState,
    backward_pass,
    check_log_likelihood_increment,
    em_glm_hmm,
    forward_backward,
    forward_pass,
    hmm_negative_log_likelihood,
    max_sum,
    prepare_likelihood_func,
    run_m_step, compute_xi_log,
)
from nemos.observation_models import BernoulliObservations, PoissonObservations
from nemos.third_party.jaxopt.jaxopt import LBFGS
from scripts.generate_simulation_glm_hmm_behavioral import log_likelihoods


def viterbi_with_hmmlearn(
    log_emission, transition_proba, init_proba, is_new_session=None
):
    """
    Use hmmlearn's Viterbi algorithm with custom log probabilities.

    Parameters
    ----------
    log_emission
        Log emission probabilities for each time step and state.
    transition_proba
        Log transition probability matrix [from_state, to_state].
    init_proba
        Log initial state probabilities.
    is_new_session
        Either None or array of shape (T,) of 0s and 1s, where 1s mark
        the beginning of a new session.

    Returns
    -------
    state_sequence
        Most likely state sequence (as integer indices 0 to K-1).
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
    Use hmmlearn's Viterbi algorithm for a single session.

    Parameters
    ----------
    model
        The HMM model to be patched.
    log_emission
        Log emission probabilities for each time step and state.

    Returns
    -------
    state_sequence
        Most likely state sequence (as integer indices 0 to K-1).
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
    """
    Prepare solver for M-step optimization for single neuron.

    Parameters
    ----------
    X
        Design matrix.
    y
        Observations.
    initial_prob
        Initial state probabilities.
    transition_prob
        State transition probability matrix.
    glm_params
        Tuple of (coefficients, intercept).
    new_sess
        Binary array indicating new session starts.
    obs
        Observation model instance.

    Returns
    -------
    gammas
        Posterior state probabilities.
    xis
        Joint posterior probabilities for consecutive states.
    """
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
        log_likelihood_func=likelihood,
        inverse_link_function=obs.default_inverse_link_function,
        is_new_session=new_sess.astype(bool),
    )


def prepare_partial_hmm_nll_single_neuron(obs):
    """
    Prepare partial HMM negative log-likelihood function and solver.

    Parameters
    ----------
    obs
        Observation model instance.

    Returns
    -------
    partial_hmm_negative_log_likelihood
        Partial negative log-likelihood function for HMM.
    solver
        LBFGS solver instance configured for the HMM likelihood.
    """
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

    solver = LBFGS(partial_hmm_negative_log_likelihood, tol=10**-8)

    return partial_hmm_negative_log_likelihood, solver


def prepare_gammas_and_xis_for_m_step_single_neuron(
    X, y, initial_prob, transition_prob, glm_params, new_sess, obs
):
    """
    Compute gammas and xis for M-step using forward-backward algorithm.

    Parameters
    ----------
    X
        Design matrix.
    y
        Observations.
    initial_prob
        Initial state probabilities.
    transition_prob
        State transition probability matrix.
    glm_params
        Tuple of (coefficients, intercept).
    new_sess
        Binary array indicating new session starts.
    obs
        Observation model instance.

    Returns
    -------
    gammas
        Posterior state probabilities.
    xis
        Joint posterior probabilities for consecutive states.
    """
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
        log_likelihood_func=likelihood,
        inverse_link_function=obs.default_inverse_link_function,
        is_new_session=new_sess.astype(bool),
    )

    return gammas, xis


def forward_step_numpy(py_z, new_sess, initial_prob, transition_prob):
    """
    Numpy implementation of forward algorithm for testing.

    Parameters
    ----------
    py_z
        Emission probabilities for each time and state.
    new_sess
        Binary array indicating new session starts.
    initial_prob
        Initial state probabilities.
    transition_prob
        State transition probability matrix.

    Returns
    -------
    alphas
        Forward probabilities (filtered state estimates).
    c
        Normalization constants at each time step.
    """
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
    """
    Numpy implementation of backward algorithm for testing.

    Parameters
    ----------
    py_z
        Emission probabilities for each time and state.
    c
        Normalization constants from forward pass.
    new_sess
        Binary array indicating new session starts.
    transition_prob
        State transition probability matrix.

    Returns
    -------
    betas
        Backward probabilities.
    """
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


@pytest.fixture(scope="module")
def generate_data_multi_state():
    """
    Generate synthetic multi-state HMM data for testing.

    Returns
    -------
    new_sess
    Session indicator array.
    initial_prob
    Initial state probabilities.
    transition_prob
    State transition matrix.
    coef
    GLM coefficients for each state.
    intercept
    GLM intercepts for each state.
    X
    Design matrix.
    y
    Observations.
    """
    np.random.seed(44)

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
    """Generate synthetic multi-state population HMM data for testing."""
    np.random.seed(44)

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


@pytest.fixture(scope="module")
def single_state_inputs():
    """Generate single-state HMM data for testing."""
    np.random.seed(42)
    initial_prob = np.ones(1)
    transition_prob = np.ones((1, 1))
    coef, intercept = np.random.randn(2, 1), np.random.randn(1)
    X = np.random.randn(10, 2)
    rate = np.exp(X.dot(coef) + intercept)
    y = np.random.poisson(rate[:, 0])
    return initial_prob, transition_prob, coef, intercept, X, rate, y


def expected_log_likelihood_wrt_transitions(
    transition_prob, xis, dirichlet_alphas=None
):
    """
    Compute expected log-likelihood with respect to transition probabilities.

    Parameters
    ----------
    transition_prob
        State transition probability matrix.
    xis
        Joint posterior probabilities for consecutive states.
    dirichlet_alphas
        Dirichlet prior hyperparameters (optional).

    Returns
    -------
    log_likelihood
        Expected log-likelihood (with prior if specified).
    """
    likelihood = jnp.sum(xis * jnp.log(transition_prob))
    if dirichlet_alphas is None:
        return likelihood
    prior = (jnp.log(transition_prob) * (dirichlet_alphas - 1)).sum()
    return likelihood + prior


def expected_log_likelihood_wrt_initial_prob(
    initial_prob, gammas, dirichlet_alphas=None
):
    """
    Compute expected log-likelihood with respect to initial probabilities.

    Parameters
    ----------
    initial_prob
        Initial state probabilities.
    gammas
        Posterior state probabilities.
    dirichlet_alphas
        Dirichlet prior hyperparameters (optional).

    Returns
    -------
    log_likelihood
        Expected log-likelihood (with prior if specified).
    """
    likelihood = jnp.sum(gammas * jnp.log(initial_prob))
    if dirichlet_alphas is None:
        return likelihood
    prior = (jnp.log(initial_prob) * (dirichlet_alphas - 1)).sum()
    return likelihood + prior


def lagrange_mult_loss(param, args, loss, **kwargs):
    """
    Lagrange multiplier loss for constrained optimization.

    Parameters
    ----------
    param
        Tuple of (probabilities, lagrange_multiplier).
    args
        Additional arguments passed to the loss function.
    loss
        Loss function to optimize.
    **kwargs
        Additional keyword arguments passed to the loss function.

    Returns
    -------
    constrained_loss
        Loss augmented with Lagrange multiplier constraint term.
    """
    proba, lam = param
    n_states = proba.shape[0]
    if proba.ndim == 2:
        constraint = proba.sum(axis=1) - jnp.ones(n_states)
    else:
        constraint = proba.sum() - 1
    lagrange_mult_term = (lam * constraint).sum()
    return loss(proba, args, **kwargs) + lagrange_mult_term


class TestForwardBackward:
    """Tests for forward-backward algorithm and related E-step computations."""

    @pytest.mark.parametrize(
        "decorator",
        [
            lambda x: x,
            partial(
                jax.jit,
                static_argnames=["log_likelihood_func", "inverse_link_function"],
            ),
        ],
    )
    @pytest.mark.requires_x64
    def test_forward_backward_regression(self, decorator):
        """
        Test forward-backward algorithm against reference implementation.

        Validates alphas, betas, gammas, xis, and log-likelihoods
        computed by forward_backward match expected results.
        """

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

        log_likelihood = jax.vmap(
            lambda x, z: obs.log_likelihood(x, z, aggregate_sample_scores=lambda w: w),
            in_axes=(None, 1),
            out_axes=1,
        )

        decorated_forward_backward = decorator(forward_backward)
        (
            gammas_nemos,
            xis_nemos,
            ll_nemos,
            ll_norm_nemos,
            log_alphas_nemos,
            log_betas_nemos,
        ) = decorated_forward_backward(
            X[:, 1:],  # drop intercept
            y,
            jnp.log(initial_prob),
            jnp.log(transition_prob),
            (coef, intercept),
            log_likelihood_func=log_likelihood,
            inverse_link_function=obs.default_inverse_link_function,
            is_new_session=new_sess.astype(bool),
        )

        # First testing alphas and betas because they are computed first
        np.testing.assert_almost_equal(log_alphas_nemos, np.log(alphas), decimal=8)
        np.testing.assert_almost_equal(log_betas_nemos, np.log(betas), decimal=8)

        # testing log likelihood and normalized log likelihood
        np.testing.assert_almost_equal(ll_nemos, ll_orig, decimal=8)
        np.testing.assert_almost_equal(ll_norm_nemos, ll_norm_orig, decimal=8)

        # Next testing xis and gammas because they depend on alphas and betas
        # Testing Eq. 13.43 of Bishop
        np.testing.assert_almost_equal(gammas_nemos, gammas, decimal=8)
        # Testing Eq. 13.65 of Bishop
        np.testing.assert_almost_equal(xis_nemos, xis, decimal=8)

    @pytest.mark.requires_x64
    def test_for_loop_forward_step(self, generate_data_multi_state):
        """
        Test forward pass implementation against numpy for-loop version.

        Ensures that the JAX vectorized forward pass produces
        identical results to a simple numpy loop implementation.
        """
        new_sess, initial_prob, transition_prob, coef, intercept, X, y = (
            generate_data_multi_state
        )

        obs = PoissonObservations()

        log_likelihood = jax.vmap(
            lambda x, z: obs.log_likelihood(x, z, aggregate_sample_scores=lambda w: w),
            in_axes=(None, 1),
            out_axes=1,
        )

        predicted_rate_given_state = obs.default_inverse_link_function(
            X @ coef + intercept
        )
        log_conditionals = log_likelihood(y, predicted_rate_given_state)

        log_alphas, log_normalization = forward_pass(
            np.log(initial_prob), np.log(transition_prob), log_conditionals, new_sess
        )

        alphas_numpy, normalization_numpy = forward_step_numpy(
            np.exp(log_conditionals), new_sess, initial_prob, transition_prob
        )
        np.testing.assert_almost_equal(np.log(alphas_numpy), log_alphas)
        np.testing.assert_almost_equal(np.log(normalization_numpy), log_normalization)

    @pytest.mark.requires_x64
    def test_for_loop_backward_step(self, generate_data_multi_state):
        """
        Test backward pass implementation against numpy for-loop version.

        Ensures that the JAX vectorized backward pass produces
        identical results to a simple numpy loop implementation.
        """
        new_sess, initial_prob, transition_prob, coef, intercept, X, y = (
            generate_data_multi_state
        )
        obs = PoissonObservations()

        log_likelihood = jax.vmap(
            lambda x, z: obs.log_likelihood(x, z, aggregate_sample_scores=lambda w: w),
            in_axes=(None, 1),
            out_axes=1,
        )

        predicted_rate_given_state = obs.default_inverse_link_function(
            X @ coef + intercept
        )
        log_conditionals = log_likelihood(y, predicted_rate_given_state)

        log_alphas, log_normalization = forward_pass(
            np.log(initial_prob), np.log(transition_prob), log_conditionals, new_sess
        )

        log_betas = backward_pass(
            np.log(transition_prob), log_conditionals, log_normalization, new_sess
        )
        betas_numpy = backward_step_numpy(
            np.exp(log_conditionals),
            np.exp(log_normalization),
            new_sess,
            transition_prob,
        )
        np.testing.assert_almost_equal(np.log(betas_numpy), log_betas)

    def test_single_state_estep(self, single_state_inputs):
        """
        Test single-state HMM E-step reduces to trivial case.

        When there's only one state, all posteriors should be 1
        and normalization should equal emission probabilities.
        """
        initial_prob, transition_prob, coef, intercept, X, rate, y = single_state_inputs
        obs = PoissonObservations()

        log_likelihood = jax.vmap(
            lambda x, z: obs.log_likelihood(x, z, aggregate_sample_scores=lambda w: w),
            in_axes=(None, 1),
            out_axes=1,
        )
        log_conditionals = log_likelihood(y, rate)
        new_sess = np.zeros(10)
        new_sess[0] = 1
        log_alphas, log_norm = forward_pass(
            np.log(initial_prob), np.log(transition_prob), log_conditionals, new_sess
        )
        log_betas = backward_pass(
            np.log(transition_prob), log_conditionals, log_norm, new_sess
        )

        # check that the normalization factor reduces to the log p(x_t | z_t)
        np.testing.assert_array_almost_equal(log_norm, log_conditionals[:, 0])

        np.testing.assert_array_almost_equal(np.zeros_like(log_alphas), log_alphas)
        np.testing.assert_array_almost_equal(np.zeros_like(log_betas), log_betas)

        # xis are a sum of the ones over valid entries
        log_xis = compute_xi_log(
            log_alphas,
            log_betas,
            log_conditionals,
            log_norm,
            new_sess,
            np.log(transition_prob),
        )
        xis = jnp.exp(log_xis)
        np.testing.assert_array_almost_equal(
            np.array([[log_alphas.shape[0] - sum(new_sess)]]).astype(xis), xis
        )


class TestLikelihood:
    """Tests for HMM negative log-likelihood computation."""

    @pytest.mark.parametrize(
        "decorator",
        [
            lambda x: x,
            partial(
                jax.jit,
                static_argnames=[
                    "negative_log_likelihood_func",
                    "inverse_link_function",
                ],
            ),
        ],
    )
    @pytest.mark.requires_x64
    def test_hmm_negative_log_likelihood_regression(self, decorator):
        """
        Test HMM negative log-likelihood against reference implementation.

        Validates that computed negative log-likelihood matches expected
        values from reference implementation.
        """

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


class TestMStep:
    @pytest.mark.requires_x64
    def test_run_m_step_regression(self):

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

        partial_hmm_negative_log_likelihood, solver = (
            prepare_partial_hmm_nll_single_neuron(obs)
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

        # Testing projection weights
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
        np.testing.assert_array_almost_equal(
            grad_at_init, np.zeros_like(new_initial_prob)
        )
        np.testing.assert_array_almost_equal(
            grad_at_lagr, np.zeros_like(lagrange_multiplier)
        )

    def test_single_state_mstep(self, single_state_inputs):
        """Single state forward pass posteriors reduces to a GLM."""
        initial_prob, transition_prob, coef, intercept, X, rate, y = single_state_inputs
        obs = PoissonObservations()

        log_likelihood = jax.vmap(
            lambda x, z: obs.log_likelihood(x, z, aggregate_sample_scores=lambda w: w),
            in_axes=(None, 1),
            out_axes=1,
        )
        log_conditionals = log_likelihood(y, rate)
        new_sess = np.zeros(10)
        new_sess[0] = 1
        log_alphas, log_norm = forward_pass(
            np.log(initial_prob), np.log(transition_prob), log_conditionals, new_sess
        )
        log_betas = backward_pass(np.log(transition_prob), log_conditionals, log_norm, new_sess)

        # xis are a sum of the ones over valid entires
        log_xis = compute_xi_log(
            log_alphas,
            log_betas,
            log_conditionals,
            log_norm,
            new_sess,
            np.log(transition_prob),
        )
        xis = np.exp(log_xis)
        partial_hmm_negative_log_likelihood, solver = (
            prepare_partial_hmm_nll_single_neuron(obs)
        )

        (
            optimized_projection_weights_nemos,
            new_initial_prob_nemos,
            new_transition_prob_nemos,
            state,
        ) = run_m_step(
            X,
            y,
            np.exp(log_alphas + log_betas),
            xis,
            (np.zeros_like(coef), np.zeros_like(intercept)),
            is_new_session=new_sess.astype(bool),
            solver_run=solver.run,
        )
        glm = GLM(
            observation_model=obs, solver_name="LBFGS", solver_kwargs={"tol": 10**-8}
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
        np.testing.assert_array_equal(
            new_initial_prob_nemos, np.ones_like(initial_prob)
        )
        np.testing.assert_array_equal(
            new_transition_prob_nemos, np.ones_like(new_transition_prob_nemos)
        )

        # check expected shapes
        assert new_transition_prob_nemos.shape == (1, 1)
        assert new_initial_prob_nemos.shape == (1,)
        assert optimized_projection_weights_nemos[0].shape == (2, 1)
        assert optimized_projection_weights_nemos[1].shape == (1,)

    @pytest.mark.requires_x64
    def test_m_step_with_prior(self, generate_data_multi_state):
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
        np.testing.assert_array_almost_equal(
            grad_at_init, np.zeros_like(new_initial_prob)
        )
        np.testing.assert_array_almost_equal(
            grad_at_lagr, np.zeros_like(lagrange_multiplier)
        )

    @pytest.mark.parametrize("state_idx", range(5))
    @pytest.mark.requires_x64
    def test_m_step_set_alpha_init_to_inf(self, generate_data_multi_state, state_idx):
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
    @pytest.mark.requires_x64
    def test_m_step_set_alpha_transition_to_inf(
        self, generate_data_multi_state, row, col
    ):
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

    @pytest.mark.requires_x64
    def test_m_step_set_alpha_init_to_1(self, generate_data_multi_state):
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

    @pytest.mark.requires_x64
    def test_m_step_set_alpha_transition_to_1(self, generate_data_multi_state):
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
    @pytest.mark.requires_x64
    def test_run_m_step_regression_priors_simulation(self, data_name):

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
        partial_hmm_negative_log_likelihood, solver = (
            prepare_partial_hmm_nll_single_neuron(obs)
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


class TestEMAlgorithm:
    """Tests for Expectation-Maximization algorithm for GLM-HMM."""

    @pytest.mark.parametrize("regularization", ["UnRegularized", "Ridge", "Lasso"])
    @pytest.mark.parametrize("require_new_session", [True, False])
    @pytest.mark.requires_x64
    def test_run_em(self, regularization, require_new_session):
        """
        Test EM algorithm increases log-likelihood.

        Validates that EM algorithm improves log-likelihood compared
        to initial parameters, with different regularization schemes.
        """

        # Fetch the data
        data_path = fetch_data("em_three_states.npz")
        data = np.load(data_path)

        # Design matrix and observed choices
        X, y = data["X"][:100], data["y"][:100]
        X = X.astype(float)
        y = y.astype(float)

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
            is_population_glm,
            obs.log_likelihood,
            obs._negative_log_likelihood,
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
        solver_name = "ProximalGradient" if "Lasso" in regularization else "LBFGS"
        glm = GLM(
            observation_model=obs, regularizer=regularization, solver_name=solver_name
        )
        glm.instantiate_solver(partial_hmm_negative_log_likelihood)
        solver_run = glm._solver_run
        # End of preparatory step.
        (
            posteriors,
            joint_posterior,
            learned_initial_prob,
            learned_transition,
            (learned_coef, learned_intercept),
            _,
        ) = em_glm_hmm(
            X[:, 1:],
            y,
            initial_prob=initial_prob,
            transition_prob=transition_prob,
            glm_params=(coef, intercept),
            is_new_session=(
                new_sess.astype(bool)[: X.shape[0]] if require_new_session else None
            ),
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
            log_likelihood_func=likelihood_func,
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
            log_likelihood_func=likelihood_func,
            inverse_link_function=obs.default_inverse_link_function,
        )
        assert (
            log_likelihood_true_params < log_likelihood_em
        ), "log-likelihood did not increase."

    @pytest.mark.parametrize("n_neurons", [5])
    @pytest.mark.requires_x64
    def test_check_em(self, n_neurons):
        """
        Test EM algorithm recovers true latent states.

        Validates that EM improves state recovery from noisy initial
        parameters for multi-neuron population data.
        """

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
            is_population_glm,
            obs.log_likelihood,
            obs._negative_log_likelihood,
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
            log_alphas_noisy_params,
            log_betas_noisy_params,
        ) = forward_backward(
            X[:, 1:],  # drop intercept
            y,
            init_pb,
            transition_pb,
            (proj_weights[1:], proj_weights[:1]),
            log_likelihood_func=likelihood_func,
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
            state,
        ) = em_glm_hmm(
            X[:, 1:],
            jnp.squeeze(y),
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
            log_likelihood_func=likelihood_func,
            inverse_link_function=obs.default_inverse_link_function,
        )

        # find state mapping
        corr_matrix = np.corrcoef(latent_states.T, posteriors.T)[
            : latent_states.shape[1], latent_states.shape[1] :
        ]
        max_corr = np.max(corr_matrix, axis=1)
        print("\nMAX CORR", max_corr)
        assert np.all(max_corr > 0.9), "State recovery failed."
        assert np.all(
            max_corr > max_corr_before_em
        ), "Latent state recovery did not improve."
        assert (
            log_likelihood_noisy_params < log_likelihood_em
        ), "Log-likelihood decreased."


@pytest.mark.requires_x64
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
        return jnp.exp(log_like)

    gammas, xis, _, _, _, _ = forward_backward(
        X,
        y,
        initial_prob,
        transition_prob,
        (coef, intercept),
        log_likelihood_func=likelihood,
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


class TestViterbi:
    @pytest.mark.parametrize("use_new_sess", [True, False])
    def test_viterbi_against_hmmlearn(self, use_new_sess):
        data = np.load(fetch_data("em_three_states.npz"))
        initial_prob = data["initial_prob"]
        transition_prob = data["transition_prob"]
        X, y = data["X"], data["y"]
        new_session = data["new_sess"] if use_new_sess else None
        intercept, coef = data["projection_weights"][:1], data["projection_weights"][1:]

        obs = BernoulliObservations()
        inverse_link_function = obs.default_inverse_link_function
        predicted_rate_given_state = inverse_link_function(
            X[..., 1:] @ coef + intercept
        )

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
            log_emission_array,
            transition_prob,
            initial_prob,
            is_new_session=new_session,
        )
        np.testing.assert_array_equal(
            map_path.astype(np.int32), hmmlearn_map_path.astype(np.int32)
        )

    @pytest.mark.parametrize("use_new_sess", [True, False])
    @pytest.mark.parametrize("return_index", [True, False])
    def test_viterbi_return_index(self, use_new_sess, return_index):
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


class TestConvergence:
    """Tests for EM convergence checking and early stopping."""

    def test_check_log_likelihood_increment_converged(self):
        """Test that convergence checker detects convergence with small likelihood change."""

        state = GLMHMMState(
            log_initial_prob=jnp.array([0.5, 0.5]),
            log_transition_matrix=jnp.eye(2),
            glm_params=(jnp.zeros((2, 2)), jnp.zeros(2)),
            data_log_likelihood=-0.0,
            previous_data_log_likelihood=-0.0001,  # Very small change
            log_likelihood_history=jnp.zeros(1),
            iterations=5,
        )

        # Should converge with loose tolerance
        assert check_log_likelihood_increment(state, tol=1e-3)

        # Should not converge with very tight tolerance
        assert not check_log_likelihood_increment(state, tol=1e-6)

    def test_check_log_likelihood_increment_not_converged(self):
        """Test that convergence checker detects non-convergence with large likelihood change."""

        state = GLMHMMState(
            log_initial_prob=jnp.array([0.5, 0.5]),
            log_transition_matrix=jnp.eye(2),
            glm_params=(jnp.zeros((2, 2)), jnp.zeros(2)),
            data_log_likelihood=-100.0,
            previous_data_log_likelihood=-130.0,  # Large change
            log_likelihood_history=jnp.zeros(1),
            iterations=5,
        )

        # Should not converge even with loose tolerance
        assert not check_log_likelihood_increment(state, tol=20.0)

    def test_check_log_likelihood_increment_first_iteration(self):
        """Test convergence checker behavior on first iteration."""

        state = GLMHMMState(
            log_initial_prob=jnp.array([0.5, 0.5]),
            log_transition_matrix=jnp.eye(2),
            glm_params=(jnp.zeros((2, 2)), jnp.zeros(2)),
            data_log_likelihood=-jnp.inf,
            previous_data_log_likelihood=-jnp.inf,
            log_likelihood_history=jnp.zeros(1),
            iterations=0,
        )

        # First iteration with -inf should not trigger convergence
        # (since abs(-inf - (-inf)) = nan, and nan < tol = False)
        result = check_log_likelihood_increment(state, tol=1e-8)
        # This should return False (not converged) or handle gracefully
        assert isinstance(result, jnp.ndarray)

    @pytest.mark.requires_x64
    def test_custom_convergence_checker(self):
        """Test that custom convergence functions work with EM."""

        def always_converge(state, tol):
            """Convergence checker that always returns True."""
            return jnp.array(True)

        # Fetch minimal test data
        data_path = fetch_data("em_three_states.npz")
        data = np.load(data_path)

        X, y = data["X"], data["y"]
        initial_prob = data["initial_prob"]
        transition_prob = data["transition_prob"]
        projection_weights = data["projection_weights"]
        intercept, coef = projection_weights[:1], projection_weights[1:]

        obs = BernoulliObservations()
        likelihood_func, negative_log_likelihood_func = prepare_likelihood_func(
            False, obs.log_likelihood, obs._negative_log_likelihood,
        )

        def partial_hmm_negative_log_likelihood(
            weights, design_matrix, observations, posterior_prob
        ):
            return hmm_negative_log_likelihood(
                weights,
                X=design_matrix,
                y=observations,
                posteriors=posterior_prob,
                inverse_link_function=obs.default_inverse_link_function,
                negative_log_likelihood_func=negative_log_likelihood_func,
            )

        glm = GLM(observation_model=obs, solver_name="LBFGS")
        glm.instantiate_solver(partial_hmm_negative_log_likelihood)

        # Run EM with custom checker - should stop after 1 iteration
        result = em_glm_hmm(
            X[:, 1:],
            y,
            initial_prob=initial_prob,
            transition_prob=transition_prob,
            glm_params=(coef, intercept),
            inverse_link_function=obs.default_inverse_link_function,
            likelihood_func=likelihood_func,
            solver_run=glm._solver_run,
            check_convergence=always_converge,
            maxiter=100,
            tol=1e-8,
        )

        final_state = result[-1]

        # Check it stopped very early (within first few iterations)
        assert final_state.iterations == 0, (
            f"EM should stop after 0 iterations with always_converge, "
            f"but ran for {final_state.iterations}"
        )

    @pytest.mark.requires_x64
    def test_never_converge_checker(self):
        """Test that EM runs to maxiter when convergence is never reached."""

        def never_converge(state, tol):
            """Convergence checker that always returns False."""
            return jnp.array(False)

        # Fetch minimal test data
        data_path = fetch_data("em_three_states.npz")
        data = np.load(data_path)

        X, y = data["X"][:50], data["y"][:50]  # Use subset for speed
        initial_prob = data["initial_prob"]
        transition_prob = data["transition_prob"]
        projection_weights = data["projection_weights"]
        intercept, coef = projection_weights[:1], projection_weights[1:]

        obs = BernoulliObservations()
        likelihood_func, negative_log_likelihood_func = prepare_likelihood_func(
            False, obs.log_likelihood, obs._negative_log_likelihood,
        )

        def partial_hmm_negative_log_likelihood(
            weights, design_matrix, observations, posterior_prob
        ):
            return hmm_negative_log_likelihood(
                weights,
                X=design_matrix,
                y=observations,
                posteriors=posterior_prob,
                inverse_link_function=obs.default_inverse_link_function,
                negative_log_likelihood_func=negative_log_likelihood_func,
            )

        glm = GLM(observation_model=obs, solver_name="LBFGS")
        glm.instantiate_solver(partial_hmm_negative_log_likelihood)

        maxiter = 10
        result = em_glm_hmm(
            X[:, 1:],
            y,
            initial_prob=initial_prob,
            transition_prob=transition_prob,
            glm_params=(coef, intercept),
            inverse_link_function=obs.default_inverse_link_function,
            likelihood_func=likelihood_func,
            solver_run=glm._solver_run,
            check_convergence=never_converge,
            maxiter=maxiter,
            tol=1e-8,
        )

        final_state = result[-1]

        # Should run exactly maxiter iterations
        assert final_state.iterations == maxiter, (
            f"EM should run for exactly {maxiter} iterations with never_converge, "
            f"but ran for {final_state.iterations}"
        )

    @pytest.mark.requires_x64
    def test_em_stops_when_converged(self):
        """Test that EM stops early when convergence criterion is met."""

        data_path = fetch_data("em_three_states.npz")
        data = np.load(data_path)

        X, y = data["X"], data["y"]
        initial_prob = data["initial_prob"]
        transition_prob = data["transition_prob"]
        projection_weights = data["projection_weights"]
        intercept, coef = projection_weights[:1], projection_weights[1:]

        obs = BernoulliObservations()
        likelihood_func, negative_log_likelihood_func = prepare_likelihood_func(
            False, obs.log_likelihood, obs._negative_log_likelihood
        )

        def partial_hmm_negative_log_likelihood(
            weights, design_matrix, observations, posterior_prob
        ):
            return hmm_negative_log_likelihood(
                weights,
                X=design_matrix,
                y=observations,
                posteriors=posterior_prob,
                inverse_link_function=obs.default_inverse_link_function,
                negative_log_likelihood_func=negative_log_likelihood_func,
            )

        glm = GLM(observation_model=obs, solver_name="LBFGS")
        glm.instantiate_solver(partial_hmm_negative_log_likelihood)

        maxiter = 1000
        tol = 1e-3  # Loose tolerance for faster convergence

        result = em_glm_hmm(
            X[:, 1:],
            y,
            initial_prob=initial_prob,
            transition_prob=transition_prob,
            glm_params=(coef, intercept),
            inverse_link_function=obs.default_inverse_link_function,
            likelihood_func=likelihood_func,
            solver_run=glm._solver_run,
            maxiter=maxiter,
            tol=tol,
        )

        final_state = result[-1]

        # Should not use all iterations
        assert final_state.iterations < maxiter, (
            f"EM should converge before {maxiter} iterations, "
            f"but used all {final_state.iterations}"
        )

        # Should have actually converged according to the criterion
        assert check_log_likelihood_increment(
            final_state, tol=tol
        ), "EM stopped but did not meet convergence criterion"

    @pytest.mark.requires_x64
    def test_em_nan_diagnostics_after_convergence(self):
        """Test that NaN likelihoods after convergence don't break the algorithm."""

        data_path = fetch_data("em_three_states.npz")
        data = np.load(data_path)

        X, y = data["X"], data["y"]
        initial_prob = data["initial_prob"]
        transition_prob = data["transition_prob"]
        projection_weights = data["projection_weights"]
        intercept, coef = projection_weights[:1], projection_weights[1:]

        obs = BernoulliObservations()
        likelihood_func, negative_log_likelihood_func = prepare_likelihood_func(
            False, obs.log_likelihood, obs._negative_log_likelihood
        )

        def partial_hmm_negative_log_likelihood(
            weights, design_matrix, observations, posterior_prob
        ):
            return hmm_negative_log_likelihood(
                weights,
                X=design_matrix,
                y=observations,
                posteriors=posterior_prob,
                inverse_link_function=obs.default_inverse_link_function,
                negative_log_likelihood_func=negative_log_likelihood_func,
            )

        glm = GLM(observation_model=obs, solver_name="LBFGS")
        glm.instantiate_solver(partial_hmm_negative_log_likelihood)

        maxiter = 100
        tol = 1e-6

        (
            posteriors,
            joint_posterior,
            final_initial_prob,
            final_transition_prob,
            final_glm_params,
            final_state,
        ) = em_glm_hmm(
            X[:100, 1:],
            y[:100],
            initial_prob=initial_prob,
            transition_prob=transition_prob,
            glm_params=(coef, intercept),
            inverse_link_function=obs.default_inverse_link_function,
            likelihood_func=likelihood_func,
            solver_run=glm._solver_run,
            maxiter=maxiter,
            tol=tol,
        )

        # Final state should have valid likelihood
        assert jnp.isfinite(
            final_state.data_log_likelihood
        ), "Final state has non-finite log-likelihood"

        # Final state should have valid previous likelihood
        assert jnp.isfinite(
            final_state.previous_data_log_likelihood
        ), "Final state has non-finite previous log-likelihood"

        # All outputs should be valid
        assert jnp.all(jnp.isfinite(posteriors)), "Posteriors contain non-finite values"
        assert jnp.all(
            jnp.isfinite(joint_posterior)
        ), "Joint posteriors contain non-finite values"
        assert jnp.all(
            jnp.isfinite(final_initial_prob)
        ), "Final initial_prob contains non-finite values"
        assert jnp.all(
            jnp.isfinite(final_transition_prob)
        ), "Final transition_prob contains non-finite values"

    @pytest.mark.requires_x64
    def test_convergence_with_different_tolerances(self):
        """Test that different tolerance values produce expected iteration counts."""

        data_path = fetch_data("em_three_states.npz")
        data = np.load(data_path)

        X, y = data["X"], data["y"]
        n_states = data["initial_prob"].shape[0]
        initial_prob = np.random.uniform(size=n_states)
        initial_prob /= np.sum(initial_prob)
        transition_prob = np.random.uniform(size=(n_states, n_states))
        transition_prob = transition_prob / np.sum(transition_prob, axis=1)[:, None]
        projection_weights = data["projection_weights"]
        intercept, coef = projection_weights[:1], projection_weights[1:]

        obs = BernoulliObservations()
        likelihood_func, negative_log_likelihood_func = prepare_likelihood_func(
            False, obs.log_likelihood, obs._negative_log_likelihood,
        )

        def partial_hmm_negative_log_likelihood(
            weights, design_matrix, observations, posterior_prob
        ):
            return hmm_negative_log_likelihood(
                weights,
                X=design_matrix,
                y=observations,
                posteriors=posterior_prob,
                inverse_link_function=obs.default_inverse_link_function,
                negative_log_likelihood_func=negative_log_likelihood_func,
            )

        glm = GLM(observation_model=obs, solver_name="LBFGS")
        glm.instantiate_solver(partial_hmm_negative_log_likelihood)

        tolerances = [1e-2, 1e-4, 1e-6]
        iteration_counts = []

        for tol in tolerances:
            result = em_glm_hmm(
                X[:, 1:],
                y,
                initial_prob=initial_prob,
                transition_prob=transition_prob,
                glm_params=(coef, intercept),
                inverse_link_function=obs.default_inverse_link_function,
                likelihood_func=likelihood_func,
                solver_run=glm._solver_run,
                maxiter=10,
                tol=tol,
            )
            iteration_counts.append(result[-1].iterations)

        # Tighter tolerance should require more iterations
        assert iteration_counts[0] <= iteration_counts[1] <= iteration_counts[2], (
            f"Expected increasing iterations with tighter tolerance, "
            f"got {iteration_counts}"
        )
        print("\nn of fb comp", forward_backward._cache_size())

    @pytest.mark.requires_x64
    def test_convergence_checker_with_iteration_limit(self):
        """Test custom convergence checker that combines likelihood and iteration limit."""

        def converge_after_n_iterations(state: GLMHMMState, tol: float, n: int = 5):
            """Stop after n iterations OR when likelihood converges."""
            likelihood_converged = check_log_likelihood_increment(state, tol)
            iteration_limit_reached = state.iterations >= n
            return likelihood_converged | iteration_limit_reached

        # Create a partial with n=5
        check_conv_5_iter = lambda state, tol: converge_after_n_iterations(
            state, tol, n=5
        )

        data_path = fetch_data("em_three_states.npz")
        data = np.load(data_path)

        X, y = data["X"][:50], data["y"][:50]  # Small subset
        initial_prob = data["initial_prob"]
        transition_prob = data["transition_prob"]
        projection_weights = data["projection_weights"]
        intercept, coef = projection_weights[:1], projection_weights[1:]

        obs = BernoulliObservations()
        likelihood_func, negative_log_likelihood_func = prepare_likelihood_func(
            False, obs.log_likelihood, obs._negative_log_likelihood,
        )

        def partial_hmm_negative_log_likelihood(
            weights, design_matrix, observations, posterior_prob
        ):
            return hmm_negative_log_likelihood(
                weights,
                X=design_matrix,
                y=observations,
                posteriors=posterior_prob,
                inverse_link_function=obs.default_inverse_link_function,
                negative_log_likelihood_func=negative_log_likelihood_func,
            )

        glm = GLM(observation_model=obs, solver_name="LBFGS")
        glm.instantiate_solver(partial_hmm_negative_log_likelihood)

        result = em_glm_hmm(
            X[:, 1:],
            y,
            initial_prob=initial_prob,
            transition_prob=transition_prob,
            glm_params=(coef, intercept),
            inverse_link_function=obs.default_inverse_link_function,
            likelihood_func=likelihood_func,
            solver_run=glm._solver_run,
            check_convergence=check_conv_5_iter,
            maxiter=100,
            tol=1e-10,  # Very tight tolerance, but will stop at 5 iterations
        )

        final_state = result[-1]

        # Should stop at or just after 5 iterations
        assert (
            final_state.iterations <= 6
        ), f"EM should stop around 5 iterations, but ran for {final_state.iterations}"


class TestCompilation:
    """Tests for JIT compilation behavior."""

    @pytest.mark.requires_x64
    def test_m_step_compiling(self, generate_data_multi_state):
        new_sess, initial_prob, transition_prob, coef, intercept, X, y = (
            generate_data_multi_state
        )

        obs = PoissonObservations()
        _, solver = prepare_partial_hmm_nll_single_neuron(obs)

        gammas, xis = prepare_gammas_and_xis_for_m_step_single_neuron(
            X, y, initial_prob, transition_prob, (coef, intercept), new_sess, obs
        )

        init_cache = run_m_step._cache_size()

        # call with no prior
        _ = run_m_step(
            X,
            y,
            gammas,
            xis,
            (np.zeros_like(coef), np.zeros_like(intercept)),
            is_new_session=new_sess.astype(bool),
            solver_run=solver.run,
            dirichlet_prior_alphas_transition=None,
            dirichlet_prior_alphas_init_prob=None,
        )

        first_call_cache = run_m_step._cache_size()
        assert init_cache + 1 == first_call_cache

        # second call with no prior
        _ = run_m_step(
            X,
            y,
            gammas,
            xis,
            (np.zeros_like(coef), np.zeros_like(intercept)),
            is_new_session=new_sess.astype(bool),
            solver_run=solver.run,
            dirichlet_prior_alphas_transition=None,
            dirichlet_prior_alphas_init_prob=None,
        )
        second_call_cache = run_m_step._cache_size()
        assert first_call_cache == second_call_cache, "None prior not cached!"

        # second call with prior
        _ = run_m_step(
            X,
            y,
            gammas,
            xis,
            (np.zeros_like(coef), np.zeros_like(intercept)),
            is_new_session=new_sess.astype(bool),
            solver_run=solver.run,
            dirichlet_prior_alphas_transition=np.ones(transition_prob.shape),
            dirichlet_prior_alphas_init_prob=np.ones(initial_prob.shape),
        )
        third_call_cache = run_m_step._cache_size()
        assert second_call_cache + 1 == third_call_cache

        # 4th call with prior
        _ = run_m_step(
            X,
            y,
            gammas,
            xis,
            (np.zeros_like(coef), np.zeros_like(intercept)),
            is_new_session=new_sess.astype(bool),
            solver_run=solver.run,
            dirichlet_prior_alphas_transition=2 * np.ones(transition_prob.shape),
            dirichlet_prior_alphas_init_prob=2 * np.ones(initial_prob.shape),
        )
        forth_call_cache = run_m_step._cache_size()
        assert third_call_cache == forth_call_cache, "Array prior not cached!"

    @pytest.mark.parametrize("solver_name", ["LBFGS", "ProximalGradient"])
    @pytest.mark.requires_x64
    def test_em_glm_hmm_compiles_once(self, generate_data_multi_state, solver_name):
        """
        Test that em_glm_hmm compiles only once for repeated calls.

        Ensures no unnecessary recompilation occurs when calling
        with same static arguments and array shapes.
        """
        new_sess, initial_prob, transition_prob, coef, intercept, X, y = (
            generate_data_multi_state
        )

        obs = PoissonObservations()
        _, solver = prepare_partial_hmm_nll_single_neuron(obs)

        obs = BernoulliObservations()
        likelihood_func, negative_log_likelihood_func = prepare_likelihood_func(
            False, obs.log_likelihood, obs._negative_log_likelihood
        )

        def partial_hmm_negative_log_likelihood(
            weights, design_matrix, observations, posterior_prob
        ):
            return hmm_negative_log_likelihood(
                weights,
                X=design_matrix,
                y=observations,
                posteriors=posterior_prob,
                inverse_link_function=obs.default_inverse_link_function,
                negative_log_likelihood_func=negative_log_likelihood_func,
            )

        glm = GLM(observation_model=obs, solver_name=solver_name)
        glm.instantiate_solver(partial_hmm_negative_log_likelihood)

        # Clear compilation cache
        initial_cache_size = em_glm_hmm._cache_size()

        # First call - should compile
        _ = em_glm_hmm(
            X,
            y,
            initial_prob=initial_prob,
            transition_prob=transition_prob,
            glm_params=(coef, intercept),
            inverse_link_function=obs.default_inverse_link_function,
            likelihood_func=likelihood_func,
            solver_run=glm._solver_run,
            maxiter=5,
            tol=1e-8,
        )

        # Check that compilation happened
        after_first_call = em_glm_hmm._cache_size()
        assert after_first_call == initial_cache_size + 1, (
            f"Expected 1 compilation, but cache went from {initial_cache_size} "
            f"to {after_first_call}"
        )

        # Second call with SAME arguments - should NOT recompile
        _ = em_glm_hmm(
            X,
            y,
            initial_prob=initial_prob,
            transition_prob=transition_prob,
            glm_params=(coef, intercept),
            inverse_link_function=obs.default_inverse_link_function,
            likelihood_func=likelihood_func,
            solver_run=glm._solver_run,
            maxiter=5,
            tol=1e-8,
        )

        # Cache should not have grown
        after_second_call = em_glm_hmm._cache_size()
        assert after_second_call == after_first_call, (
            f"Unexpected recompilation: cache grew from {after_first_call} "
            f"to {after_second_call}"
        )

        # Third call with DIFFERENT data (same shape) - should NOT recompile
        X_new = (X + np.random.randn(*X.shape) * 0.1).astype(X.dtype)
        y_new = y.copy()
        initial_prob_new = np.ones_like(initial_prob) / len(initial_prob)
        transition_prob_new = np.ones_like(transition_prob) / len(initial_prob)
        coef_new = coef * np.random.randn(*coef.shape)
        intercept_new = intercept * np.random.randn(*intercept.shape)
        _ = em_glm_hmm(
            X_new,
            y_new,
            initial_prob=initial_prob_new,
            transition_prob=transition_prob_new,
            glm_params=(coef_new, intercept_new),
            inverse_link_function=obs.default_inverse_link_function,
            likelihood_func=likelihood_func,
            solver_run=glm._solver_run,
            maxiter=5,
            tol=1e-8,
        )

        after_third_call = em_glm_hmm._cache_size()
        assert after_third_call == after_second_call, (
            f"Recompiled on different data values (same shape): "
            f"cache grew from {after_second_call} to {after_third_call}"
        )

    @pytest.mark.requires_x64
    def test_forward_backward_compiles_once(self, generate_data_multi_state):
        """
        Test that forward_backward is not recompiled on each EM iteration.

        forward_backward should compile once and be reused across all EM steps.
        """
        new_sess, initial_prob, transition_prob, coef, intercept, X, y = (
            generate_data_multi_state
        )

        obs = BernoulliObservations()
        likelihood_func, negative_log_likelihood_func = prepare_likelihood_func(
            False, obs.log_likelihood, obs._negative_log_likelihood,
        )

        def partial_hmm_negative_log_likelihood(
            weights, design_matrix, observations, posterior_prob
        ):
            return hmm_negative_log_likelihood(
                weights,
                X=design_matrix,
                y=observations,
                posteriors=posterior_prob,
                inverse_link_function=obs.default_inverse_link_function,
                negative_log_likelihood_func=negative_log_likelihood_func,
            )

        glm = GLM(observation_model=obs, solver_name="LBFGS")
        glm.instantiate_solver(partial_hmm_negative_log_likelihood)

        _ = forward_backward(
            X,  # drop intercept
            y,
            initial_prob,
            transition_prob,
            (coef, intercept),
            log_likelihood_func=likelihood_func,
            inverse_link_function=obs.default_inverse_link_function,
            is_new_session=new_sess.astype(bool),
        )
        initial_fb_cache = forward_backward._cache_size()
        # second call with new data (same shape and size)
        X_new = (X + np.random.randn(*X.shape) * 0.1).astype(X.dtype)
        y_new = y.copy()
        initial_prob_new = np.ones_like(initial_prob) / len(initial_prob)
        transition_prob_new = np.ones_like(transition_prob) / len(initial_prob)
        coef_new = coef * np.random.randn(*coef.shape)
        intercept_new = intercept * np.random.randn(*intercept.shape)
        _ = forward_backward(
            X_new,  # drop intercept
            y_new,
            initial_prob_new,
            transition_prob_new,
            (coef_new, intercept_new),
            log_likelihood_func=likelihood_func,
            inverse_link_function=obs.default_inverse_link_function,
            is_new_session=new_sess.astype(bool),
        )

        final_fb_cache = forward_backward._cache_size()

        # forward_backward should compile at most once during entire EM
        # (It's called multiple times but with same shapes)
        compilations = final_fb_cache - initial_fb_cache
        assert compilations <= 1, (
            f"forward_backward compiled {compilations} times, "
            f"expected at most 1 compilation"
        )


class TestPytreeSupport:
    """Test that GLM-HMM algorithms support pytree inputs."""

    @pytest.mark.requires_x64
    def test_forward_backward_with_pytree(self, generate_data_multi_state):
        """Test forward_backward accepts pytree inputs for X and coef."""
        new_sess, initial_prob, transition_prob, coef, intercept, X, y = (
            generate_data_multi_state
        )

        # Split X and coef into dictionaries
        n_features = X.shape[1]
        X_tree = {
            "feature_a": X[:, :1],
            "feature_b": X[:, 1:],
        }
        coef_tree = {
            "feature_a": coef[:1, :],
            "feature_b": coef[1:, :],
        }

        obs = PoissonObservations()
        likelihood_func, _ = prepare_likelihood_func(
            is_population_glm=False,
            log_likelihood_func=obs.log_likelihood,
            negative_log_likelihood_func=obs._negative_log_likelihood,
        )

        # Test with standard arrays (reference)
        (
            posteriors_ref,
            joint_posterior_ref,
            ll_ref,
            ll_norm_ref,
            alphas_ref,
            betas_ref,
        ) = forward_backward(
            X,
            y,
            initial_prob,
            transition_prob,
            (coef, intercept),
            obs.default_inverse_link_function,
            likelihood_func,
            new_sess.astype(bool),
        )

        # Test with pytrees
        posteriors, joint_posterior, ll, ll_norm, alphas, betas = forward_backward(
            X_tree,
            y,
            initial_prob,
            transition_prob,
            (coef_tree, intercept),
            obs.default_inverse_link_function,
            likelihood_func,
            new_sess.astype(bool),
        )

        # Results should be identical
        np.testing.assert_allclose(posteriors, posteriors_ref)
        np.testing.assert_allclose(joint_posterior, joint_posterior_ref)
        np.testing.assert_allclose(ll, ll_ref)
        np.testing.assert_allclose(ll_norm, ll_norm_ref)
        np.testing.assert_allclose(alphas, alphas_ref)
        np.testing.assert_allclose(betas, betas_ref)

    @pytest.mark.requires_x64
    def test_hmm_negative_log_likelihood_with_pytree(self, generate_data_multi_state):
        """Test hmm_negative_log_likelihood accepts pytree inputs."""
        new_sess, initial_prob, transition_prob, coef, intercept, X, y = (
            generate_data_multi_state
        )

        # Split X and coef into dictionaries
        X_tree = {
            "feature_a": X[:, :1],
            "feature_b": X[:, 1:],
        }
        coef_tree = {
            "feature_a": coef[:1, :],
            "feature_b": coef[1:, :],
        }

        obs = PoissonObservations()

        # Create vmapped negative log likelihood
        negative_log_likelihood = jax.vmap(
            lambda x, z: obs._negative_log_likelihood(
                x, z, aggregate_sample_scores=lambda w: w
            ),
            in_axes=(None, 1),
            out_axes=1,
        )

        # Create some dummy posteriors
        n_states = initial_prob.shape[0]
        n_samples = X.shape[0]
        posteriors = np.random.uniform(size=(n_samples, n_states))
        posteriors /= posteriors.sum(axis=1, keepdims=True)

        # Test with standard arrays (reference)
        nll_ref = hmm_negative_log_likelihood(
            (coef, intercept),
            X,
            y,
            posteriors,
            obs.default_inverse_link_function,
            negative_log_likelihood,
        )

        # Test with pytrees
        nll = hmm_negative_log_likelihood(
            (coef_tree, intercept),
            X_tree,
            y,
            posteriors,
            obs.default_inverse_link_function,
            negative_log_likelihood,
        )

        # Results should be identical
        np.testing.assert_allclose(nll, nll_ref)

    @pytest.mark.requires_x64
    def test_em_glm_hmm_with_pytree(self, generate_data_multi_state):
        """Test em_glm_hmm accepts pytree inputs for X and coef."""
        new_sess, initial_prob, transition_prob, coef, intercept, X, y = (
            generate_data_multi_state
        )

        # Split X and coef into dictionaries
        X_tree = {
            "feature_a": X[:, :1],
            "feature_b": X[:, 1:],
        }
        coef_tree = {
            "feature_a": coef[:1, :],
            "feature_b": coef[1:, :],
        }

        obs = PoissonObservations()
        likelihood_func, vmap_nll = prepare_likelihood_func(
            is_population_glm=False,
            likelihood_func=obs.log_likelihood,
            negative_log_likelihood_func=obs._negative_log_likelihood,
            is_log=True,
        )

        # Create solver using GLM class
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

        glm = GLM(observation_model=obs, solver_name="LBFGS")
        glm.instantiate_solver(partial_hmm_negative_log_likelihood)
        solver_run = glm._solver_run

        # Run EM with pytrees (just a few iterations)
        (
            posteriors,
            joint_posterior,
            final_init,
            final_trans,
            final_params,
            final_state,
        ) = em_glm_hmm(
            X_tree,
            y,
            initial_prob,
            transition_prob,
            (coef_tree, intercept),
            obs.default_inverse_link_function,
            likelihood_func,
            solver_run,
            new_sess.astype(bool),
            maxiter=3,
            tol=1e-8,
        )

        # Just verify it runs and returns valid outputs
        assert posteriors.shape == (X.shape[0], initial_prob.shape[0])
        assert joint_posterior.shape == (initial_prob.shape[0], initial_prob.shape[0])
        assert final_init.shape == initial_prob.shape
        assert final_trans.shape == transition_prob.shape
        assert isinstance(final_params, tuple)
        assert isinstance(final_params[0], dict)  # coef should be a dict
        assert final_state.iterations > 0
