import itertools
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hmmlearn import hmm

from nemos.fetch import fetch_data
from nemos.glm import GLM
from nemos.glm.params import GLMParams
from nemos.glm_hmm.algorithm_configs import (
    get_analytical_scale_update,
    posterior_weighted_glm_negative_log_likelihood,
    prepare_estep_log_likelihood,
    prepare_mstep_nll_for_analytical_scale,
    prepare_mstep_nll_objective_param,
    prepare_mstep_nll_objective_scale,
)
from nemos.glm_hmm.expectation_maximization import (
    GLMHMMState,
    backward_pass,
    check_log_likelihood_increment,
    compute_rate_per_state,
    compute_xi_log,
    em_glm_hmm,
    forward_backward,
    forward_pass,
    max_sum,
    run_m_step,
)
from nemos.glm_hmm.m_step_analytical_updates import (
    _analytical_m_step_log_initial_prob,
    _analytical_m_step_log_transition_prob,
)
from nemos.glm_hmm.params import GLMHMMParams, GLMScale, HMMParams
from nemos.observation_models import (
    BernoulliObservations,
    GammaObservations,
    GaussianObservations,
    Observations,
    PoissonObservations,
)
from nemos.regularizer import UnRegularized
from nemos.solvers import solver_registry


def setup_solver(
    objective, init_params, tol=1e-12, reg_strength=0.0, reg=UnRegularized()
):
    lbfgs_class = solver_registry["LBFGS"]
    solver = lbfgs_class(
        objective,
        init_params=init_params,
        regularizer=reg,
        regularizer_strength=reg_strength,
        has_aux=False,
        tol=tol,
    )
    return solver


def _add_prior_logspace(log_val: jnp.ndarray, offset: jnp.ndarray):
    """Add prior offset in log-space (current implementation)."""
    result = jnp.where(
        offset > 0,
        jnp.logaddexp(log_val, jnp.log(jnp.maximum(offset, 1e-10))),
        log_val,
    )
    return result


_vmap_add_prior = jax.vmap(_add_prior_logspace)


# ============================================================================
# LOG-SPACE IMPLEMENTATIONS (Current)
# ============================================================================


def m_step_initial_logspace(
    log_posteriors: jnp.ndarray,
    is_new_session: jnp.ndarray,
    dirichlet_prior_alphas: jnp.ndarray | None = None,
):
    """M-step for initial probabilities in log-space.

    Underflow safe update (slow and overkill, just for testing against our implementation).
    """
    # Mask out non-session-start time points
    masked_log_posteriors = jnp.where(
        is_new_session[:, jnp.newaxis], log_posteriors, -jnp.inf
    )

    # Sum over time in log-space
    log_tmp_initial_prob = jax.scipy.special.logsumexp(masked_log_posteriors, axis=0)

    if dirichlet_prior_alphas is not None:
        prior_offset = dirichlet_prior_alphas - 1
        log_numerator = _vmap_add_prior(log_tmp_initial_prob, prior_offset)
    else:
        log_numerator = log_tmp_initial_prob

    # Normalize in log-space
    log_sum = jax.scipy.special.logsumexp(log_numerator)
    log_initial_prob = log_numerator - log_sum

    return log_initial_prob


def m_step_transition_logspace(
    log_joint_posterior: jnp.ndarray,
    dirichlet_prior_alphas: jnp.ndarray | None = None,
):
    """M-step for transition probabilities in log-space.

    Underflow safe update (slow and overkill, just for testing against our implementation).
    """
    if dirichlet_prior_alphas is not None:
        prior_offset = dirichlet_prior_alphas - 1
        log_numerator = _vmap_add_prior(log_joint_posterior, prior_offset)
    else:
        log_numerator = log_joint_posterior

    # Normalize each row in log-space
    log_row_sums = jax.scipy.special.logsumexp(log_numerator, axis=1, keepdims=True)
    log_transition_prob = log_numerator - log_row_sums

    return log_transition_prob


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


def prepare_partial_hmm_nll_single_neuron(obs, init_params):
    """
    Prepare partial HMM negative log-likelihood function and solver.

    Parameters
    ----------
    obs
        Observation model instance.
    init_params
        Initial parameters.

    Returns
    -------
    partial_posterior_weighted_glm_negative_log_likelihood
        Partial negative log-likelihood function for HMM.
    solver
        LBFGS solver instance configured for the HMM likelihood.
    """
    partial_posterior_weighted_glm_negative_log_likelihood = (
        prepare_mstep_nll_objective_param(
            False,
            observation_model=obs,
            inverse_link_function=obs.default_inverse_link_function,
        )
    )
    solver = setup_solver(
        partial_posterior_weighted_glm_negative_log_likelihood,
        init_params=init_params,
        tol=1e-8,
    )

    return partial_posterior_weighted_glm_negative_log_likelihood, solver


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
    coef, intercept = glm_params
    likelihood = prepare_estep_log_likelihood(y.ndim > 1, obs)
    gammas, xis, _, _, _, _ = forward_backward(
        X,
        y,
        initial_prob,
        transition_prob,
        GLMParams(coef, intercept),
        glm_scale=GLMScale(jnp.zeros(initial_prob.shape[0])),
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
def generate_data_multi_state(request):
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
    obs_model: Observations = request.param["observations"]
    scale: float = request.param["scale"]
    inv_link = request.param.get("inv_link", None)
    if inv_link is None:
        inv_link = obs_model.default_inverse_link_function

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
    key = jax.random.PRNGKey(42)
    for i, k in enumerate(range(0, 100, 10)):
        sl = slice(k, k + 10)
        state = i % n_states
        rate = inv_link(X[sl].dot(coef[:, state]) + intercept[state])
        key, subkey = jax.random.split(key)
        y[sl] = obs_model.sample_generator(subkey, rate, scale)

    new_sess = np.zeros(n_samples)
    new_sess[[0, 10, 90]] = 1
    return (
        new_sess,
        initial_prob,
        transition_prob,
        coef,
        intercept,
        X,
        y,
        obs_model,
        scale,
        inv_link,
    )


@pytest.fixture(scope="module")
def generate_data_multi_state_population(request):
    """Generate synthetic multi-state population HMM data for testing."""
    np.random.seed(44)
    obs_model: Observations = request.param["observations"]
    scale: float = request.param["scale"]
    inv_link = request.param.get("inv_link", None)
    if inv_link is None:
        inv_link = obs_model.default_inverse_link_function

    # E-step initial parameters
    n_states, n_neurons, n_samples = 5, 3, 100
    initial_prob = np.random.uniform(size=(n_states))
    initial_prob /= np.sum(initial_prob)
    transition_prob = np.random.uniform(size=(n_states, n_states))
    transition_prob /= np.sum(transition_prob, axis=0)
    transition_prob = transition_prob.T
    coef, intercept = (
        np.random.randn(2, n_neurons, n_states),
        np.random.randn(n_neurons, n_states),
    )

    X = np.random.randn(n_samples, 2)
    y = np.zeros((n_samples, n_neurons))
    key = jax.random.PRNGKey(42)
    for i, k in enumerate(range(0, 100, 10)):
        sl = slice(k, k + 10)
        state = i % n_states
        rate = np.exp(X[sl].dot(coef[..., state]) + intercept[..., state])
        key, subkey = jax.random.split(key)
        y[sl] = obs_model.sample_generator(subkey, rate, scale)

    new_sess = np.zeros(n_samples)
    new_sess[[0, 10, 90]] = 1
    return (
        new_sess,
        initial_prob,
        transition_prob,
        coef,
        intercept,
        X,
        y,
        obs_model,
        scale,
        inv_link,
    )


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
        intercept = intercept.squeeze()
        transition_prob = data["transition_prob"]

        # E-step output
        xis = data["xis"]
        gammas = data["gammas"]
        ll_orig, ll_norm_orig = data["log_likelihood"], data["ll_norm"]
        alphas, betas = data["alphas"], data["betas"]

        obs = BernoulliObservations()

        log_likelihood = prepare_estep_log_likelihood(
            is_population_glm=y.ndim > 1, observation_model=obs
        )

        decorated_forward_backward = decorator(forward_backward)
        (
            log_gammas_nemos,
            log_xis_nemos,
            ll_nemos,
            ll_norm_nemos,
            log_alphas_nemos,
            log_betas_nemos,
        ) = decorated_forward_backward(
            X[:, 1:],  # drop intercept
            y,
            jnp.log(initial_prob),
            jnp.log(transition_prob),
            GLMParams(coef, intercept),
            glm_scale=GLMScale(jnp.zeros_like(intercept)),
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
        np.testing.assert_almost_equal(log_gammas_nemos, np.log(gammas), decimal=8)
        # Testing Eq. 13.65 of Bishop
        np.testing.assert_almost_equal(log_xis_nemos, np.log(xis), decimal=8)

    @pytest.mark.requires_x64
    @pytest.mark.parametrize(
        "generate_data_multi_state",
        [{"observations": PoissonObservations(), "scale": 1.0}],
        indirect=True,
    )
    def test_for_loop_forward_step(self, generate_data_multi_state):
        """
        Test forward pass implementation against numpy for-loop version.

        Ensures that the JAX vectorized forward pass produces
        identical results to a simple numpy loop implementation.
        """
        (
            new_sess,
            initial_prob,
            transition_prob,
            coef,
            intercept,
            X,
            y,
            obs,
            scale,
            inv_link,
        ) = generate_data_multi_state
        log_likelihood = jax.vmap(
            lambda x, z: obs.log_likelihood(x, z, aggregate_sample_scores=lambda w: w),
            in_axes=(None, 1),
            out_axes=1,
        )

        predicted_rate_given_state = inv_link(X @ coef + intercept)
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
    @pytest.mark.parametrize(
        "generate_data_multi_state",
        [{"observations": PoissonObservations(), "scale": 1.0}],
        indirect=True,
    )
    def test_for_loop_backward_step(self, generate_data_multi_state):
        """
        Test backward pass implementation against numpy for-loop version.

        Ensures that the JAX vectorized backward pass produces
        identical results to a simple numpy loop implementation.
        """
        (
            new_sess,
            initial_prob,
            transition_prob,
            coef,
            intercept,
            X,
            y,
            obs,
            scale,
            inv_link,
        ) = generate_data_multi_state
        log_likelihood = jax.vmap(
            lambda x, z: obs.log_likelihood(x, z, aggregate_sample_scores=lambda w: w),
            in_axes=(None, 1),
            out_axes=1,
        )

        predicted_rate_given_state = inv_link(X @ coef + intercept)
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
    def test_posterior_weighted_glm_negative_log_likelihood_regression(self, decorator):
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
        intercept = intercept.squeeze()

        # Negative LL output
        nll_m_step = data["nll_m_step"]

        # Initialize nemos observation model
        obs = BernoulliObservations()

        # Define negative log likelihood vmap function
        log_likelihood = prepare_mstep_nll_objective_param(
            y.ndim > 1,
            observation_model=obs,
            inverse_link_function=obs.default_inverse_link_function,
        )
        nll_m_step_nemos = log_likelihood(
            GLMParams(coef, intercept),
            X[:, 1:],  # drop intercept column
            y,
            gammas,
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
        intercept = intercept.squeeze()

        new_sess = data["new_sess"]

        # M-step output
        optimized_projection_weights = data["optimized_projection_weights"]
        opt_intercept, opt_coef = (
            optimized_projection_weights[:1].squeeze(),
            optimized_projection_weights[1:],
        )
        new_initial_prob = data["new_initial_prob"]
        new_transition_prob = data["new_transition_prob"]

        # Initialize nemos observation model
        obs = BernoulliObservations()

        partial_posterior_weighted_glm_negative_log_likelihood, solver = (
            prepare_partial_hmm_nll_single_neuron(
                obs, init_params=GLMParams(coef, intercept)
            )
        )
        params = GLMHMMParams(
            hmm_params=HMMParams(None, None),
            glm_params=GLMParams(coef, intercept),
            glm_scale=GLMScale(jnp.zeros_like(intercept)),
        )
        (
            new_params,
            state,
        ) = run_m_step(
            params,
            X[:, 1:],  # drop intercept column
            y,
            np.log(gammas),
            np.log(xis),
            is_new_session=new_sess.astype(bool),
            m_step_fn_glm_params=solver.run,
            m_step_fn_glm_scale=None,
            inverse_link_function=obs.default_inverse_link_function,
        )

        # Convert back to probability space for comparison with reference
        new_initial_prob_nemos = np.exp(new_params.hmm_params.log_initial_prob)
        new_transition_prob_nemos = np.exp(new_params.hmm_params.log_transition_prob)

        n_ll_nemos = partial_posterior_weighted_glm_negative_log_likelihood(
            new_params.glm_params,
            X[:, 1:],
            y,
            gammas,
        )
        n_ll_original = partial_posterior_weighted_glm_negative_log_likelihood(
            GLMParams(opt_coef, opt_intercept),
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
            GLMParams(opt_coef, opt_intercept),
            new_params.glm_params,
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
        grad_at_transition, grad_at_lagr = grad_objective(
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
        grad_at_init, grad_at_lagr = grad_objective(
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

    @pytest.mark.requires_x64
    def test_single_state_mstep(self, single_state_inputs):
        """Single state forward pass posteriors reduces to a GLM."""
        initial_prob, transition_prob, coef, intercept, X, rate, y = single_state_inputs
        obs = PoissonObservations()

        log_likelihood = prepare_estep_log_likelihood(
            is_population_glm=y.ndim > 1, observation_model=obs
        )
        log_conditionals = log_likelihood(y, rate, jnp.ones(coef.shape[-1]))
        new_sess = np.zeros(10)
        new_sess[0] = 1
        log_alphas, log_norm = forward_pass(
            np.log(initial_prob), np.log(transition_prob), log_conditionals, new_sess
        )
        log_betas = backward_pass(
            np.log(transition_prob), log_conditionals, log_norm, new_sess
        )

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
        partial_posterior_weighted_glm_negative_log_likelihood, solver = (
            prepare_partial_hmm_nll_single_neuron(
                obs, init_params=GLMParams(coef, intercept)
            )
        )

        params = GLMHMMParams(
            hmm_params=HMMParams(None, None),
            glm_params=GLMParams(jnp.zeros_like(coef), jnp.zeros_like(intercept)),
            glm_scale=GLMScale(jnp.zeros_like(intercept)),
        )
        (
            new_params,
            state,
        ) = run_m_step(
            params,
            X,
            y,
            log_alphas + log_betas,
            np.log(xis),
            is_new_session=new_sess.astype(bool),
            m_step_fn_glm_params=solver.run,
            m_step_fn_glm_scale=None,
            inverse_link_function=obs.default_inverse_link_function,
        )
        glm = GLM(
            observation_model=obs, solver_name="LBFGS", solver_kwargs={"tol": 10**-8}
        )
        glm.fit(X, y)
        # test that the glm coeff and intercept matches with the m-step output
        np.testing.assert_array_almost_equal(
            glm.coef_, new_params.glm_params.coef.flatten()
        )
        np.testing.assert_array_almost_equal(
            glm.intercept_, new_params.glm_params.intercept.flatten()
        )

        # test that the transition and initial probabilities are all ones (log(1) = 0).
        np.testing.assert_array_equal(
            new_params.hmm_params.log_initial_prob, np.zeros_like(initial_prob)
        )
        np.testing.assert_array_equal(
            new_params.hmm_params.log_transition_prob, np.zeros_like(transition_prob)
        )

        # check expected shapes
        assert new_params.hmm_params.log_transition_prob.shape == (1, 1)
        assert new_params.hmm_params.log_initial_prob.shape == (1,)
        assert new_params.glm_params.coef.shape == (2, 1)
        assert new_params.glm_params.intercept.shape == (1,)

    @pytest.mark.requires_x64
    @pytest.mark.parametrize(
        "generate_data_multi_state",
        [{"observations": PoissonObservations(), "scale": 1.0}],
        indirect=True,
    )
    def test_m_step_with_prior(self, generate_data_multi_state):
        """Test M-step with Dirichlet priors (alpha > 1) using Lagrange multipliers.

        This test uses gradient-based Lagrange multiplier optimality conditions,
        which work well for interior solutions (alpha > 1).
        """
        (
            new_sess,
            initial_prob,
            transition_prob,
            coef,
            intercept,
            X,
            y,
            obs,
            scale,
            inv_link,
        ) = generate_data_multi_state
        _, solver = prepare_partial_hmm_nll_single_neuron(
            obs, init_params=GLMParams(coef, intercept)
        )

        log_gammas, log_xis = prepare_gammas_and_xis_for_m_step_single_neuron(
            X, y, initial_prob, transition_prob, (coef, intercept), new_sess, obs
        )

        alphas_transition = np.random.uniform(1, 3, size=transition_prob.shape)
        alphas_init = np.random.uniform(1, 3, size=initial_prob.shape)

        params = GLMHMMParams(
            hmm_params=HMMParams(None, None),
            glm_params=GLMParams(jnp.zeros_like(coef), jnp.zeros_like(intercept)),
            glm_scale=GLMScale(jnp.zeros_like(intercept)),
        )

        (
            new_params,
            state,
        ) = run_m_step(
            params,
            X,
            y,
            log_gammas,
            log_xis,
            is_new_session=new_sess.astype(bool),
            m_step_fn_glm_params=solver.run,
            m_step_fn_glm_scale=None,
            inverse_link_function=inv_link,
            dirichlet_prior_alphas_transition=alphas_transition,
            dirichlet_prior_alphas_init_prob=alphas_init,
        )

        # Convert back to probability space for gradient checks
        new_initial_prob = np.exp(new_params.hmm_params.log_initial_prob)
        new_transition_prob = np.exp(new_params.hmm_params.log_transition_prob)

        lagrange_multiplier = -jax.grad(expected_log_likelihood_wrt_transitions)(
            new_transition_prob, np.exp(log_xis), dirichlet_alphas=alphas_transition
        ).mean(
            axis=1
        )  # note that the lagrange mult makes the gradient all the same for each prob.
        # 2) Check that the gradient of the loss is zero
        grad_objective = jax.grad(lagrange_mult_loss)
        grad_at_transition, grad_at_lagr = grad_objective(
            (new_transition_prob, lagrange_multiplier),
            np.exp(log_xis),
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
        sum_gammas = np.sum(np.exp(log_gammas)[np.where(new_sess)[0]], axis=0)
        lagrange_multiplier = -jax.grad(expected_log_likelihood_wrt_initial_prob)(
            new_initial_prob, sum_gammas, dirichlet_alphas=alphas_init
        ).mean()  # note that the lagrange mult makes the gradient all the same for each prob.
        # 2) Check that the gradient of the loss is zero
        grad_objective = jax.grad(lagrange_mult_loss)
        grad_at_init, grad_at_lagr = grad_objective(
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
    @pytest.mark.parametrize(
        "generate_data_multi_state",
        [{"observations": PoissonObservations(), "scale": 1.0}],
        indirect=True,
    )
    def test_m_step_set_alpha_init_to_inf(self, generate_data_multi_state, state_idx):
        (
            new_sess,
            initial_prob,
            transition_prob,
            coef,
            intercept,
            X,
            y,
            obs,
            scale,
            inv_link,
        ) = generate_data_multi_state
        _, solver = prepare_partial_hmm_nll_single_neuron(
            obs, init_params=GLMParams(coef, intercept)
        )

        log_gammas, log_xis = prepare_gammas_and_xis_for_m_step_single_neuron(
            X, y, initial_prob, transition_prob, (coef, intercept), new_sess, obs
        )

        alphas_transition = np.random.uniform(1, 3, size=transition_prob.shape)
        alphas_init = np.random.uniform(1, 3, size=initial_prob.shape)
        alphas_init[state_idx] = 10**20

        params = GLMHMMParams(
            hmm_params=HMMParams(None, None),
            glm_params=GLMParams(jnp.zeros_like(coef), jnp.zeros_like(intercept)),
            glm_scale=GLMScale(jnp.zeros_like(intercept)),
        )

        (
            new_params,
            state,
        ) = run_m_step(
            params,
            X,
            y,
            log_gammas,
            log_xis,
            is_new_session=new_sess.astype(bool),
            m_step_fn_glm_params=solver.run,
            m_step_fn_glm_scale=None,
            inverse_link_function=inv_link,
            dirichlet_prior_alphas_transition=alphas_transition,
            dirichlet_prior_alphas_init_prob=alphas_init,
        )
        # Convert back to probability space
        new_initial_prob = np.exp(new_params.hmm_params.log_initial_prob)
        np.testing.assert_array_almost_equal(
            new_initial_prob,
            np.eye(new_params.hmm_params.log_initial_prob.shape[0])[state_idx],
        )

    @pytest.mark.parametrize("row, col", itertools.product(range(3), range(3)))
    @pytest.mark.requires_x64
    @pytest.mark.parametrize(
        "generate_data_multi_state",
        [{"observations": PoissonObservations(), "scale": 1.0}],
        indirect=True,
    )
    def test_m_step_set_alpha_transition_to_inf(
        self, generate_data_multi_state, row, col
    ):
        (
            new_sess,
            initial_prob,
            transition_prob,
            coef,
            intercept,
            X,
            y,
            obs,
            scale,
            inv_link,
        ) = generate_data_multi_state
        _, solver = prepare_partial_hmm_nll_single_neuron(
            obs, init_params=GLMParams(coef, intercept)
        )

        log_gammas, log_xis = prepare_gammas_and_xis_for_m_step_single_neuron(
            X, y, initial_prob, transition_prob, (coef, intercept), new_sess, obs
        )

        alphas_transition = np.random.uniform(1, 3, size=transition_prob.shape)
        alphas_init = np.random.uniform(1, 3, size=initial_prob.shape)
        alphas_transition[row, col] = 10**20

        params = GLMHMMParams(
            hmm_params=HMMParams(None, None),
            glm_params=GLMParams(jnp.zeros_like(coef), jnp.zeros_like(intercept)),
            glm_scale=GLMScale(jnp.zeros_like(intercept)),
        )

        (
            new_params,
            state,
        ) = run_m_step(
            params,
            X,
            y,
            log_gammas,
            log_xis,
            is_new_session=new_sess.astype(bool),
            m_step_fn_glm_params=solver.run,
            m_step_fn_glm_scale=None,
            inverse_link_function=inv_link,
            dirichlet_prior_alphas_transition=alphas_transition,
            dirichlet_prior_alphas_init_prob=alphas_init,
        )
        # Convert back to probability space
        new_initial_prob = np.exp(new_params.hmm_params.log_initial_prob)
        new_transition_prob = np.exp(new_params.hmm_params.log_transition_prob)
        np.testing.assert_array_almost_equal(
            new_transition_prob[row, :], np.eye(new_initial_prob.shape[0])[col]
        )

    @pytest.mark.requires_x64
    @pytest.mark.parametrize(
        "generate_data_multi_state",
        [{"observations": PoissonObservations(), "scale": 1.0}],
        indirect=True,
    )
    def test_m_step_set_alpha_init_to_1(self, generate_data_multi_state):
        (
            new_sess,
            initial_prob,
            transition_prob,
            coef,
            intercept,
            X,
            y,
            obs,
            scale,
            inv_link,
        ) = generate_data_multi_state
        _, solver = prepare_partial_hmm_nll_single_neuron(
            obs, init_params=GLMParams(coef, intercept)
        )

        log_gammas, log_xis = prepare_gammas_and_xis_for_m_step_single_neuron(
            X, y, initial_prob, transition_prob, (coef, intercept), new_sess, obs
        )

        alphas_transition = np.random.uniform(1, 3, size=transition_prob.shape)
        alphas_init = np.ones(initial_prob.shape)

        params = GLMHMMParams(
            hmm_params=HMMParams(None, None),
            glm_params=GLMParams(jnp.zeros_like(coef), jnp.zeros_like(intercept)),
            glm_scale=GLMScale(jnp.zeros_like(intercept)),
        )

        (
            new_params,
            state,
        ) = run_m_step(
            params,
            X,
            y,
            log_gammas,
            log_xis,
            is_new_session=new_sess.astype(bool),
            m_step_fn_glm_params=solver.run,
            m_step_fn_glm_scale=None,
            inverse_link_function=inv_link,
            dirichlet_prior_alphas_transition=alphas_transition,
            dirichlet_prior_alphas_init_prob=alphas_init,
        )
        (
            new_params_no_prior,
            state,
        ) = run_m_step(
            params,
            X,
            y,
            log_gammas,
            log_xis,
            is_new_session=new_sess.astype(bool),
            m_step_fn_glm_params=solver.run,
            m_step_fn_glm_scale=None,
            inverse_link_function=inv_link,
            dirichlet_prior_alphas_transition=alphas_transition,
            dirichlet_prior_alphas_init_prob=None,
        )
        np.testing.assert_array_almost_equal(
            new_params_no_prior.hmm_params.log_initial_prob,
            new_params.hmm_params.log_initial_prob,
        )
        np.testing.assert_array_almost_equal(
            new_params_no_prior.hmm_params.log_transition_prob,
            new_params.hmm_params.log_transition_prob,
        )
        np.testing.assert_array_almost_equal(
            new_params_no_prior.glm_params.coef,
            new_params.glm_params.coef,
        )
        np.testing.assert_array_almost_equal(
            new_params_no_prior.glm_params.intercept,
            new_params.glm_params.intercept,
        )

    @pytest.mark.requires_x64
    @pytest.mark.parametrize(
        "generate_data_multi_state",
        [{"observations": PoissonObservations(), "scale": 1.0}],
        indirect=True,
    )
    def test_m_step_set_alpha_transition_to_1(self, generate_data_multi_state):
        (
            new_sess,
            initial_prob,
            transition_prob,
            coef,
            intercept,
            X,
            y,
            obs,
            scale,
            inv_link,
        ) = generate_data_multi_state
        _, solver = prepare_partial_hmm_nll_single_neuron(
            obs, init_params=GLMParams(coef, intercept)
        )

        log_gammas, log_xis = prepare_gammas_and_xis_for_m_step_single_neuron(
            X, y, initial_prob, transition_prob, (coef, intercept), new_sess, obs
        )
        alphas_transition = np.ones(transition_prob.shape)
        alphas_init = np.random.uniform(1, 3, size=initial_prob.shape)

        params = GLMHMMParams(
            hmm_params=HMMParams(None, None),
            glm_params=GLMParams(jnp.zeros_like(coef), jnp.zeros_like(intercept)),
            glm_scale=GLMScale(jnp.zeros_like(intercept)),
        )

        (
            new_params,
            state,
        ) = run_m_step(
            params,
            X,
            y,
            log_gammas,
            log_xis,
            is_new_session=new_sess.astype(bool),
            m_step_fn_glm_params=solver.run,
            m_step_fn_glm_scale=None,
            inverse_link_function=inv_link,
            dirichlet_prior_alphas_transition=alphas_transition,
            dirichlet_prior_alphas_init_prob=alphas_init,
        )
        (
            new_params_no_prior,
            state,
        ) = run_m_step(
            params,
            X,
            y,
            log_gammas,
            log_xis,
            is_new_session=new_sess.astype(bool),
            m_step_fn_glm_params=solver.run,
            m_step_fn_glm_scale=None,
            inverse_link_function=inv_link,
            dirichlet_prior_alphas_transition=None,
            dirichlet_prior_alphas_init_prob=alphas_init,
        )
        np.testing.assert_array_almost_equal(
            new_params_no_prior.hmm_params.log_initial_prob,
            new_params.hmm_params.log_initial_prob,
        )
        np.testing.assert_array_almost_equal(
            new_params_no_prior.hmm_params.log_transition_prob,
            new_params.hmm_params.log_transition_prob,
        )
        np.testing.assert_array_almost_equal(
            new_params_no_prior.glm_params.coef,
            new_params.glm_params.coef,
        )
        np.testing.assert_array_almost_equal(
            new_params_no_prior.glm_params.intercept,
            new_params.glm_params.intercept,
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
        intercept = intercept.squeeze()
        new_sess = data["new_sess"]

        # M-step output
        optimized_projection_weights = data["optimized_projection_weights"]
        opt_intercept, opt_coef = (
            optimized_projection_weights[:1].squeeze(),
            optimized_projection_weights[1:],
        )
        new_initial_prob = data["new_initial_prob"]
        new_transition_prob = data["new_transition_prob"]

        # Initialize nemos observation model
        obs = BernoulliObservations()

        # Prepare nll function & solver
        partial_posterior_weighted_glm_negative_log_likelihood, solver = (
            prepare_partial_hmm_nll_single_neuron(
                obs, init_params=GLMParams(coef, intercept)
            )
        )

        params = GLMHMMParams(
            hmm_params=HMMParams(None, None),
            glm_params=GLMParams(jnp.zeros_like(coef), jnp.zeros_like(intercept)),
            glm_scale=GLMScale(jnp.zeros_like(intercept)),
        )

        (
            new_params,
            state,
        ) = run_m_step(
            params,
            X[:, 1:],  # drop intercept column
            y,
            np.log(gammas),
            np.log(xis),
            is_new_session=new_sess.astype(bool),
            m_step_fn_glm_params=solver.run,
            m_step_fn_glm_scale=None,
            inverse_link_function=obs.default_inverse_link_function,
            dirichlet_prior_alphas_init_prob=dirichlet_prior_initial_prob,
            dirichlet_prior_alphas_transition=dirichlet_prior_transition_prob,
        )

        # Convert back to probability space for comparison with reference
        new_initial_prob_nemos = np.exp(new_params.hmm_params.log_initial_prob)
        new_transition_prob_nemos = np.exp(new_params.hmm_params.log_transition_prob)

        # NLL with nemos input
        n_ll_nemos = partial_posterior_weighted_glm_negative_log_likelihood(
            new_params.glm_params,
            X[:, 1:],
            y,
            gammas,
        )

        # NLL with simulation input
        n_ll_original = partial_posterior_weighted_glm_negative_log_likelihood(
            GLMParams(opt_coef, opt_intercept),
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
            GLMParams(opt_coef, opt_intercept),
            new_params.glm_params,
        )

    @pytest.mark.parametrize("underflow_scheme", ["one_of_three", "two_of_three"])
    @pytest.mark.requires_x64
    def test_m_step_underflow_logspace_controlled(self, underflow_scheme):
        """Test run_m_step with controlled underflow in log-posteriors."""

        n_timesteps = 10000
        n_states = 3
        n_sessions = 10

        key = jax.random.PRNGKey(0)

        # Generate baseline log-posteriors
        key, subkey = jax.random.split(key)
        log_posteriors = jax.random.normal(subkey, (n_timesteps, n_states))

        # Apply controlled underflow
        if underflow_scheme == "one_of_three":
            # For each row, pick 1 of 3 to underflow
            key, subkey = jax.random.split(key)
            selected = jax.random.randint(subkey, (n_timesteps,), 0, n_states)
            log_posteriors = log_posteriors.at[jnp.arange(n_timesteps), selected].set(
                -200.0
            )
        elif underflow_scheme == "two_of_three":
            # For each row, pick 2 of 3 to underflow
            key, subkey = jax.random.split(key)
            choices = jax.random.permutation(subkey, n_states)[:2]  # pick two indices
            log_posteriors = log_posteriors.at[:, choices].set(-200.0)

        # Normalize in log-space per row
        log_posteriors = log_posteriors - jax.scipy.special.logsumexp(
            log_posteriors, axis=1, keepdims=True
        )

        print(
            "\nEffective underflow log_posterior fraction :",
            np.mean(log_posteriors < -40),
        )
        # Session starts
        is_new_session = np.zeros(n_timesteps, dtype=bool)
        session_starts = np.linspace(0, n_timesteps - 1, n_sessions, dtype=int)
        is_new_session[session_starts] = True
        is_new_session = jnp.array(is_new_session)

        # Log joint posterior for transitions
        key, subkey = jax.random.split(key)
        log_joint_posterior = jax.random.normal(subkey, (n_states, n_states))
        if underflow_scheme == "one_of_three":
            for i in range(n_states):
                key, subkey = jax.random.split(key)
                choices = jax.random.permutation(subkey, n_states)[:1]
                log_joint_posterior = log_joint_posterior.at[i, choices].set(-200.0)
        elif underflow_scheme == "two_of_three":
            for i in range(n_states):
                key, subkey = jax.random.split(key)
                choices = jax.random.permutation(subkey, n_states)[:2]
                log_joint_posterior = log_joint_posterior.at[i, choices].set(-200.0)

        # Normalize each row in log-space to make them valid log-probabilities
        log_joint_posterior = log_joint_posterior - jax.scipy.special.logsumexp(
            log_joint_posterior, axis=1, keepdims=True
        )
        print(
            "\nEffective underflow log_joint_posterior fraction :",
            np.mean(log_joint_posterior < -40),
        )

        # Dirichlet priors
        alphas_init = jnp.ones(n_states) * 1.5
        alphas_trans = jnp.ones((n_states, n_states)) * 1.5

        # Compute reference log-space M-step
        log_init_ref = m_step_initial_logspace(
            log_posteriors, is_new_session, alphas_init
        )
        log_trans_ref = m_step_transition_logspace(log_joint_posterior, alphas_trans)

        # Dummy GLM parameters
        dummy_coef = jnp.zeros((n_states, 1))
        dummy_intercept = jnp.zeros((1,))
        dummy_aux = None
        X_dummy = jnp.ones((n_timesteps, n_states))
        y_dummy = jnp.ones((n_timesteps,))

        obs = PoissonObservations()

        params = GLMHMMParams(
            hmm_params=HMMParams(None, None),
            glm_params=GLMParams(dummy_coef, dummy_intercept),
            glm_scale=GLMScale(jnp.zeros_like(dummy_intercept)),
        )

        # Run M-step
        new_params, _ = run_m_step(
            params,
            X_dummy,
            y_dummy,
            log_posteriors,
            log_joint_posterior,
            is_new_session=is_new_session,
            m_step_fn_glm_params=lambda *a, **kw: (
                GLMParams(
                    dummy_coef,
                    dummy_intercept,
                ),
                None,
                dummy_aux,
            ),
            m_step_fn_glm_scale=None,
            inverse_link_function=obs.default_inverse_link_function,
            dirichlet_prior_alphas_init_prob=alphas_init,
            dirichlet_prior_alphas_transition=alphas_trans,
        )

        # Compare to reference
        np.testing.assert_allclose(
            new_params.hmm_params.log_initial_prob, log_init_ref, rtol=1e-12, atol=0
        )
        np.testing.assert_allclose(
            new_params.hmm_params.log_transition_prob, log_trans_ref, rtol=1e-12, atol=0
        )

        # Shapes
        assert new_params.hmm_params.log_initial_prob.shape == (n_states,)
        assert new_params.hmm_params.log_transition_prob.shape == (n_states, n_states)

    @pytest.mark.requires_x64
    @pytest.mark.parametrize(
        "generate_data_multi_state",
        [
            {"observations": BernoulliObservations(), "scale": 1.0, "inv_link": None},
            {"observations": GaussianObservations(), "scale": 1.0, "inv_link": None},
            {"observations": GaussianObservations(), "scale": 2.0, "inv_link": None},
            {"observations": GammaObservations(), "scale": 1.0, "inv_link": jnp.exp},
            {"observations": GammaObservations(), "scale": 2.0, "inv_link": jnp.exp},
        ],
        indirect=True,
    )
    def test_likelihood_increases_at_each_update(self, generate_data_multi_state):
        (
            new_sess,
            initial_prob,
            transition_prob,
            coef,
            intercept,
            X,
            y,
            obs,
            scale,
            inv_link,
        ) = generate_data_multi_state
        new_sess = jnp.asarray(new_sess, dtype=bool)
        ll_func = prepare_estep_log_likelihood(False, obs)
        log_posteriors, log_joint_posterior, _, initial_log_like, _, _ = (
            forward_backward(
                X,
                y,
                jnp.log(initial_prob),
                jnp.log(transition_prob),
                glm_params=GLMParams(coef, intercept),
                glm_scale=GLMScale(jnp.zeros_like(intercept)),
                inverse_link_function=inv_link,
                log_likelihood_func=ll_func,
                is_new_session=new_sess,
            )
        )

        # apply update:
        # Update Initial state probability Eq. 13.18
        posteriors = jnp.exp(log_posteriors)
        new_log_initial_prob = _analytical_m_step_log_initial_prob(
            log_posteriors,
            is_new_session=new_sess,
        )
        _, _, _, updated_log_like, _, _ = forward_backward(
            X,
            y,
            new_log_initial_prob,
            jnp.log(transition_prob),
            glm_params=GLMParams(coef, intercept),
            glm_scale=GLMScale(jnp.zeros_like(intercept)),
            inverse_link_function=inv_link,
            log_likelihood_func=ll_func,
            is_new_session=new_sess,
        )
        assert (
            updated_log_like > initial_log_like
        ), "M-step for initial prob did not increase likelihood"

        initial_log_like = updated_log_like
        new_log_transition_prob = _analytical_m_step_log_transition_prob(
            log_joint_posterior
        )
        _, _, _, updated_log_like, _, _ = forward_backward(
            X,
            y,
            new_log_initial_prob,
            new_log_transition_prob,
            glm_params=GLMParams(coef, intercept),
            glm_scale=GLMScale(jnp.zeros_like(intercept)),
            inverse_link_function=inv_link,
            log_likelihood_func=ll_func,
            is_new_session=new_sess,
        )
        assert (
            updated_log_like > initial_log_like
        ), "M-step for transition prob did not increase likelihood"

        # Minimize negative log-likelihood to update GLM weights
        initial_log_like = updated_log_like
        init_glm_params = GLMParams(coef, intercept)
        objective = prepare_mstep_nll_objective_param(False, obs, inv_link)

        solver = setup_solver(objective, init_params=init_glm_params, tol=1e-8)

        new_glm_prams, state, _ = solver.run(init_glm_params, X, y, posteriors)
        _, _, _, updated_log_like, _, _ = forward_backward(
            X,
            y,
            new_log_initial_prob,
            new_log_transition_prob,
            glm_params=new_glm_prams,
            glm_scale=GLMScale(jnp.zeros_like(intercept)),
            inverse_link_function=inv_link,
            log_likelihood_func=ll_func,
            is_new_session=new_sess,
        )
        assert (
            updated_log_like > initial_log_like
        ), "M-step for GLMParams prob did not increase likelihood"

        initial_log_like = updated_log_like
        predicted_rate = compute_rate_per_state(
            X, new_glm_prams, inverse_link_function=inv_link
        )
        objective_scale = prepare_mstep_nll_objective_scale(False, obs)
        init_scale = GLMScale(jnp.zeros_like(intercept))
        solver = setup_solver(objective_scale, init_params=init_scale, tol=1e-8)

        new_scale, _, _ = solver.run(init_scale, y, predicted_rate, posteriors)
        if not isinstance(obs, (PoissonObservations, BernoulliObservations)):
            _, _, _, updated_log_like, _, _ = forward_backward(
                X,
                y,
                new_log_initial_prob,
                new_log_transition_prob,
                glm_params=new_glm_prams,
                glm_scale=new_scale,
                inverse_link_function=inv_link,
                log_likelihood_func=ll_func,
                is_new_session=new_sess,
            )
            assert (
                updated_log_like > initial_log_like
            ), "M-step for GLM scale prob did not increase likelihood"
        else:
            np.testing.assert_array_equal(
                new_scale.log_scale, jnp.zeros_like(intercept)
            )

        params = GLMHMMParams(
            hmm_params=HMMParams(None, None),
            glm_params=GLMParams(coef, intercept),
            glm_scale=GLMScale(jnp.zeros_like(intercept)),
        )

        (
            new_params,
            _,
        ) = run_m_step(
            params,
            X,
            y,
            log_posteriors=log_posteriors,
            log_joint_posterior=log_joint_posterior,
            inverse_link_function=inv_link,
            is_new_session=new_sess,
            m_step_fn_glm_scale=setup_solver(
                objective_scale, params.glm_scale, tol=10**-12
            ).run,
            m_step_fn_glm_params=setup_solver(
                objective, init_params=params.glm_params
            ).run,
        )

        jax.tree_util.tree_map(
            np.testing.assert_allclose, new_params.glm_params, new_glm_prams
        )
        np.testing.assert_allclose(new_params.glm_scale.log_scale, new_scale.log_scale)
        np.testing.assert_allclose(
            jnp.exp(new_params.hmm_params.log_initial_prob),
            jnp.exp(new_log_initial_prob),
        )
        np.testing.assert_allclose(
            jnp.exp(new_params.hmm_params.log_transition_prob),
            jnp.exp(new_log_transition_prob),
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
        likelihood_func = prepare_estep_log_likelihood(
            y.ndim > 1, observation_model=obs
        )

        inverse_link_function = obs.default_inverse_link_function
        partial_posterior_weighted_glm_negative_log_likelihood = (
            prepare_mstep_nll_objective_param(
                y.ndim > 1,
                observation_model=obs,
                inverse_link_function=inverse_link_function,
            )
        )

        # use the BaseRegressor initialize_solver (this will be avaialble also in the GLMHHM class)
        solver_name = "ProximalGradient" if "Lasso" in regularization else "LBFGS"
        glm = GLM(
            observation_model=obs, regularizer=regularization, solver_name=solver_name
        )
        glm._instantiate_solver(
            partial_posterior_weighted_glm_negative_log_likelihood,
            GLMParams(coef, intercept),
        )
        solver_run = glm._solver_run
        # End of preparatory step.

        # Create initial parameters
        params = GLMHMMParams(
            glm_params=GLMParams(coef, intercept),
            glm_scale=GLMScale(jnp.zeros(coef.shape[-1])),
            hmm_params=HMMParams(jnp.log(initial_prob), jnp.log(transition_prob)),
        )

        learned_params, state = em_glm_hmm(
            params=params,
            X=X[:, 1:],
            y=y,
            is_new_session=(
                new_sess.astype(bool)[: X.shape[0]] if require_new_session else None
            ),
            inverse_link_function=inverse_link_function,
            log_likelihood_func=likelihood_func,
            m_step_fn_glm_params=solver_run,
            m_step_fn_glm_scale=None,
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
            learned_params.hmm_params.log_initial_prob,
            learned_params.hmm_params.log_transition_prob,
            learned_params.glm_params,
            learned_params.glm_scale,
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
            jnp.log(initial_prob),
            jnp.log(transition_prob),
            GLMParams(coef, intercept),
            glm_scale=GLMScale(jnp.zeros(coef.shape[-1])),
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
        likelihood_func = prepare_estep_log_likelihood(is_population_glm, obs)
        negative_log_likelihood_func = prepare_mstep_nll_for_analytical_scale(
            is_population_glm, obs
        )
        inverse_link_function = obs.default_inverse_link_function

        # closure for the static callables
        # NOTE: this is the _predict_and_compute_loss equivalent (aka, what it is used in
        # the numerical M-step).
        def partial_posterior_weighted_glm_negative_log_likelihood(
            weights, design_matrix, observations, posterior_prob
        ):
            return posterior_weighted_glm_negative_log_likelihood(
                weights,
                X=design_matrix,
                y=observations,
                posteriors=posterior_prob,
                inverse_link_function=inverse_link_function,
                negative_log_likelihood_func=negative_log_likelihood_func,
            )

        # use the BaseRegressor initialize_solver (this will be avaialble also in the GLMHHM class)
        glm = GLM(observation_model=obs, solver_name="LBFGS")
        glm._instantiate_solver(
            partial_posterior_weighted_glm_negative_log_likelihood,
            GLMParams(intercept, coef),
        )
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
            log_posteriors_noisy_params,
            log_joint_posterior_noisy_params,
            log_likelihood_noisy_params,
            log_likelihood_norm_noisy_params,
            log_alphas_noisy_params,
            log_betas_noisy_params,
        ) = forward_backward(
            X[:, 1:],  # drop intercept
            y,
            jnp.log(init_pb),
            jnp.log(transition_pb),
            GLMParams(proj_weights[1:], proj_weights[:1]),
            glm_scale=GLMScale(
                jnp.zeros((y.shape[1], initial_prob.shape[0]))
                if is_population_glm
                else jnp.zeros(initial_prob.shape[0])
            ),
            log_likelihood_func=likelihood_func,
            inverse_link_function=obs.default_inverse_link_function,
        )

        latent_states = data["latent_states"]
        corr_matrix_before_em = np.corrcoef(
            latent_states.T, np.exp(log_posteriors_noisy_params).T
        )[: latent_states.shape[1], latent_states.shape[1] :]
        max_corr_before_em = np.max(corr_matrix_before_em, axis=1)

        # Create initial parameters
        noisy_params = GLMHMMParams(
            glm_params=GLMParams(proj_weights[1:], proj_weights[:1]),
            glm_scale=GLMScale(
                jnp.zeros((y.shape[1], transition_prob.shape[0]))
                if is_population_glm
                else jnp.zeros(transition_prob.shape[0])
            ),
            hmm_params=HMMParams(jnp.log(init_pb), jnp.log(transition_pb)),
        )

        learned_params, state = em_glm_hmm(
            params=noisy_params,
            X=X[:, 1:],
            y=jnp.squeeze(y),
            inverse_link_function=inverse_link_function,
            log_likelihood_func=likelihood_func,
            m_step_fn_glm_params=solver_run,
            m_step_fn_glm_scale=None,
            tol=10**-10,
        )
        (
            log_posteriors_em,
            _,
            _,
            log_likelihood_em,
            _,
            _,
        ) = forward_backward(
            X[:, 1:],  # drop intercept
            y,
            learned_params.hmm_params.log_initial_prob,
            learned_params.hmm_params.log_transition_prob,
            learned_params.glm_params,
            glm_scale=learned_params.glm_scale,
            log_likelihood_func=likelihood_func,
            inverse_link_function=obs.default_inverse_link_function,
        )

        # find state mapping
        posteriors = jnp.exp(log_posteriors_em)
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
@pytest.mark.parametrize(
    "generate_data_multi_state_population",
    [{"observations": PoissonObservations(), "scale": 1.0}],
    indirect=True,
)
def test_e_and_m_step_for_population(generate_data_multi_state_population):
    """Run E and M step fitting a population."""
    (
        new_sess,
        initial_prob,
        transition_prob,
        coef,
        intercept,
        X,
        y,
        obs,
        scale,
        inv_link,
    ) = generate_data_multi_state_population

    likelihood = prepare_estep_log_likelihood(True, observation_model=obs)
    init_glm_params = GLMParams(coef, intercept)
    init_scale = GLMScale(jnp.zeros_like(intercept))
    log_gammas, log_xis, _, _, _, _ = forward_backward(
        X,
        y,
        jnp.log(initial_prob),
        jnp.log(transition_prob),
        init_glm_params,
        glm_scale=init_scale,
        log_likelihood_func=likelihood,
        inverse_link_function=inv_link,
        is_new_session=new_sess.astype(bool),
    )

    partial_posterior_weighted_glm_negative_log_likelihood = (
        prepare_mstep_nll_objective_param(
            True,
            observation_model=obs,
            inverse_link_function=inv_link,
        )
    )
    alphas_transition = np.random.uniform(1, 3, size=transition_prob.shape)
    alphas_init = np.random.uniform(1, 3, size=initial_prob.shape)
    solver = setup_solver(
        partial_posterior_weighted_glm_negative_log_likelihood,
        init_params=init_glm_params,
        tol=1e-13,
    )

    nll_scale = prepare_mstep_nll_objective_scale(True, obs)
    solver_scale = setup_solver(nll_scale, init_params=init_scale, tol=1e-13)

    params = GLMHMMParams(
        glm_params=GLMParams(np.zeros_like(coef), np.zeros_like(intercept)),
        glm_scale=GLMScale(jnp.zeros(intercept.shape)),
        hmm_params=HMMParams(None, None),  # Will be set by run_m_step
    )

    new_params, state = run_m_step(
        params,
        X,
        y,
        log_gammas,
        log_xis,
        is_new_session=new_sess.astype(bool),
        m_step_fn_glm_params=solver.run,
        m_step_fn_glm_scale=solver_scale.run,
        dirichlet_prior_alphas_transition=alphas_transition,
        dirichlet_prior_alphas_init_prob=alphas_init,
        inverse_link_function=inv_link,
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
            GLMParams(coef, intercept),
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
            GLMParams(coef, intercept),
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
        likelihood_func = prepare_estep_log_likelihood(False, obs)
        negative_log_likelihood_func = prepare_mstep_nll_for_analytical_scale(
            False, obs
        )

        def partial_posterior_weighted_glm_negative_log_likelihood(
            weights, design_matrix, observations, posterior_prob
        ):
            return posterior_weighted_glm_negative_log_likelihood(
                weights,
                X=design_matrix,
                y=observations,
                posteriors=posterior_prob,
                inverse_link_function=obs.default_inverse_link_function,
                negative_log_likelihood_func=negative_log_likelihood_func,
            )

        glm = GLM(observation_model=obs, solver_name="LBFGS")
        glm._instantiate_solver(
            partial_posterior_weighted_glm_negative_log_likelihood,
            GLMParams(coef, intercept),
        )

        # Create initial parameters
        params = GLMHMMParams(
            glm_params=GLMParams(coef, intercept),
            glm_scale=GLMScale(jnp.zeros(transition_prob.shape[0])),
            hmm_params=HMMParams(jnp.log(initial_prob), jnp.log(transition_prob)),
        )

        # Run EM with custom checker - should stop after 1 iteration
        learned_params, final_state = em_glm_hmm(
            params=params,
            X=X[:, 1:],
            y=y,
            inverse_link_function=obs.default_inverse_link_function,
            log_likelihood_func=likelihood_func,
            m_step_fn_glm_params=glm._solver_run,
            m_step_fn_glm_scale=None,
            check_convergence=always_converge,
            maxiter=100,
            tol=1e-8,
        )

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
        likelihood_func = prepare_estep_log_likelihood(False, obs)
        negative_log_likelihood_func = prepare_mstep_nll_for_analytical_scale(
            False, obs
        )

        def partial_posterior_weighted_glm_negative_log_likelihood(
            weights, design_matrix, observations, posterior_prob
        ):
            return posterior_weighted_glm_negative_log_likelihood(
                weights,
                X=design_matrix,
                y=observations,
                posteriors=posterior_prob,
                inverse_link_function=obs.default_inverse_link_function,
                negative_log_likelihood_func=negative_log_likelihood_func,
            )

        glm = GLM(observation_model=obs, solver_name="LBFGS")
        glm._instantiate_solver(
            partial_posterior_weighted_glm_negative_log_likelihood,
            GLMParams(coef, intercept),
        )

        # Create initial parameters
        params = GLMHMMParams(
            glm_params=GLMParams(coef, intercept),
            glm_scale=GLMScale(jnp.zeros(transition_prob.shape[0])),
            hmm_params=HMMParams(jnp.log(initial_prob), jnp.log(transition_prob)),
        )

        maxiter = 10
        learned_params, final_state = em_glm_hmm(
            params=params,
            X=X[:, 1:],
            y=y,
            inverse_link_function=obs.default_inverse_link_function,
            log_likelihood_func=likelihood_func,
            m_step_fn_glm_params=glm._solver_run,
            m_step_fn_glm_scale=None,
            check_convergence=never_converge,
            maxiter=maxiter,
            tol=1e-8,
        )

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
        likelihood_func = prepare_estep_log_likelihood(False, obs)
        negative_log_likelihood_func = prepare_mstep_nll_for_analytical_scale(
            False, obs
        )

        def partial_posterior_weighted_glm_negative_log_likelihood(
            weights, design_matrix, observations, posterior_prob
        ):
            return posterior_weighted_glm_negative_log_likelihood(
                weights,
                X=design_matrix,
                y=observations,
                posteriors=posterior_prob,
                inverse_link_function=obs.default_inverse_link_function,
                negative_log_likelihood_func=negative_log_likelihood_func,
            )

        glm = GLM(observation_model=obs, solver_name="LBFGS")
        glm._instantiate_solver(
            partial_posterior_weighted_glm_negative_log_likelihood,
            GLMParams(coef, intercept),
        )

        # Create initial parameters
        params = GLMHMMParams(
            glm_params=GLMParams(coef, intercept),
            glm_scale=GLMScale(jnp.zeros(transition_prob.shape[0])),
            hmm_params=HMMParams(jnp.log(initial_prob), jnp.log(transition_prob)),
        )

        maxiter = 1000
        tol = 1e-3  # Loose tolerance for faster convergence

        learned_params, final_state = em_glm_hmm(
            params=params,
            X=X[:, 1:],
            y=y,
            inverse_link_function=obs.default_inverse_link_function,
            log_likelihood_func=likelihood_func,
            m_step_fn_glm_params=glm._solver_run,
            m_step_fn_glm_scale=None,
            maxiter=maxiter,
            tol=tol,
        )

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
        intercept, coef = projection_weights[:1].squeeze(), projection_weights[1:]

        obs = BernoulliObservations()
        likelihood_func = prepare_estep_log_likelihood(False, obs)
        negative_log_likelihood_func = prepare_mstep_nll_for_analytical_scale(
            False, obs
        )

        def partial_posterior_weighted_glm_negative_log_likelihood(
            weights, design_matrix, observations, posterior_prob
        ):
            return posterior_weighted_glm_negative_log_likelihood(
                weights,
                X=design_matrix,
                y=observations,
                posteriors=posterior_prob,
                inverse_link_function=obs.default_inverse_link_function,
                negative_log_likelihood_func=negative_log_likelihood_func,
            )

        glm = GLM(observation_model=obs, solver_name="LBFGS")
        glm._instantiate_solver(
            partial_posterior_weighted_glm_negative_log_likelihood,
            GLMParams(coef, intercept),
        )

        # Create initial parameters
        params = GLMHMMParams(
            glm_params=GLMParams(coef, intercept),
            glm_scale=GLMScale(jnp.zeros(transition_prob.shape[0])),
            hmm_params=HMMParams(jnp.log(initial_prob), jnp.log(transition_prob)),
        )

        maxiter = 100
        tol = 1e-6
        learned_params, final_state = em_glm_hmm(
            params=params,
            X=X[:100, 1:],
            y=y[:100],
            inverse_link_function=obs.default_inverse_link_function,
            log_likelihood_func=likelihood_func,
            m_step_fn_glm_params=glm._solver_run,
            m_step_fn_glm_scale=None,
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

        # All learned parameters should be valid
        assert jnp.all(
            jnp.isfinite(learned_params.hmm_params.log_initial_prob)
        ), "Final log_initial_prob contains non-finite values"
        assert jnp.all(
            jnp.isfinite(learned_params.hmm_params.log_transition_prob)
        ), "Final log_transition_prob contains non-finite values"
        assert jnp.all(
            jnp.isfinite(learned_params.glm_scale.log_scale)
        ), "Final log_scale contains non-finite values"

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
        likelihood_func = prepare_estep_log_likelihood(False, obs)
        negative_log_likelihood_func = prepare_mstep_nll_for_analytical_scale(
            False, obs
        )

        def partial_posterior_weighted_glm_negative_log_likelihood(
            weights, design_matrix, observations, posterior_prob
        ):
            return posterior_weighted_glm_negative_log_likelihood(
                weights,
                X=design_matrix,
                y=observations,
                posteriors=posterior_prob,
                inverse_link_function=obs.default_inverse_link_function,
                negative_log_likelihood_func=negative_log_likelihood_func,
            )

        glm = GLM(observation_model=obs, solver_name="LBFGS")
        glm._instantiate_solver(
            partial_posterior_weighted_glm_negative_log_likelihood,
            GLMParams(coef, intercept),
        )

        # Create initial parameters
        params = GLMHMMParams(
            glm_params=GLMParams(coef, intercept),
            glm_scale=GLMScale(jnp.zeros(transition_prob.shape[0])),
            hmm_params=HMMParams(jnp.log(initial_prob), jnp.log(transition_prob)),
        )

        tolerances = [1e-2, 1e-4, 1e-6]
        iteration_counts = []

        for tol in tolerances:
            _, final_state = em_glm_hmm(
                params=params,
                X=X[:, 1:],
                y=y,
                inverse_link_function=obs.default_inverse_link_function,
                log_likelihood_func=likelihood_func,
                m_step_fn_glm_params=glm._solver_run,
                m_step_fn_glm_scale=None,
                maxiter=10,
                tol=tol,
            )
            iteration_counts.append(final_state.iterations)

        # Tighter tolerance should require more iterations
        assert iteration_counts[0] <= iteration_counts[1] <= iteration_counts[2], (
            f"Expected increasing iterations with tighter tolerance, "
            f"got {iteration_counts}"
        )

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
        likelihood_func = prepare_estep_log_likelihood(False, obs)
        negative_log_likelihood_func = prepare_mstep_nll_for_analytical_scale(
            False, obs
        )

        def partial_posterior_weighted_glm_negative_log_likelihood(
            weights, design_matrix, observations, posterior_prob
        ):
            return posterior_weighted_glm_negative_log_likelihood(
                weights,
                X=design_matrix,
                y=observations,
                posteriors=posterior_prob,
                inverse_link_function=obs.default_inverse_link_function,
                negative_log_likelihood_func=negative_log_likelihood_func,
            )

        glm = GLM(observation_model=obs, solver_name="LBFGS")
        glm._instantiate_solver(
            partial_posterior_weighted_glm_negative_log_likelihood,
            GLMParams(coef, intercept),
        )

        # Create initial parameters
        params = GLMHMMParams(
            glm_params=GLMParams(coef, intercept),
            glm_scale=GLMScale(jnp.zeros(transition_prob.shape[0])),
            hmm_params=HMMParams(jnp.log(initial_prob), jnp.log(transition_prob)),
        )

        _, final_state = em_glm_hmm(
            params=params,
            X=X[:, 1:],
            y=y,
            inverse_link_function=obs.default_inverse_link_function,
            log_likelihood_func=likelihood_func,
            m_step_fn_glm_params=glm._solver_run,
            m_step_fn_glm_scale=None,
            check_convergence=check_conv_5_iter,
            maxiter=100,
            tol=1e-10,  # Very tight tolerance, but will stop at 5 iterations
        )

        # Should stop at or just after 5 iterations
        assert (
            final_state.iterations <= 6
        ), f"EM should stop around 5 iterations, but ran for {final_state.iterations}"


class TestCompilation:
    """Tests for JIT compilation behavior.

    These tests verify that JAX correctly caches compiled functions and doesn't
    recompile unnecessarily. We use a counter inside a jitted function to track
    compilations - the counter only increments during JAX tracing (compilation),
    not during cached execution.
    """

    @pytest.mark.requires_x64
    @pytest.mark.parametrize(
        "generate_data_multi_state",
        [{"observations": PoissonObservations(), "scale": 1.0}],
        indirect=True,
    )
    def test_m_step_compiling(self, generate_data_multi_state):
        """Test that run_m_step caches correctly for None vs array priors."""
        (
            new_sess,
            initial_prob,
            transition_prob,
            coef,
            intercept,
            X,
            y,
            obs,
            scale,
            inv_link,
        ) = generate_data_multi_state

        obs = PoissonObservations()
        _, solver = prepare_partial_hmm_nll_single_neuron(
            obs, init_params=GLMParams(coef, intercept)
        )

        log_gammas, log_xis = prepare_gammas_and_xis_for_m_step_single_neuron(
            X, y, initial_prob, transition_prob, (coef, intercept), new_sess, obs
        )

        # Create a tracked version of run_m_step with compilation counter
        compilation_counter = {"n_compilations": 0}

        @partial(
            jax.jit,
            static_argnames=[
                "m_step_fn_glm_params",
                "inverse_link_function",
                "m_step_fn_glm_scale",
            ],
        )
        def tracked_run_m_step(
            params,
            X,
            y,
            log_gammas,
            log_xis,
            is_new_session,
            m_step_fn_glm_params,
            m_step_fn_glm_scale,
            inverse_link_function,
            dirichlet_prior_alphas_init_prob=None,
            dirichlet_prior_alphas_transition=None,
        ):
            # This increment only runs during tracing (compilation)
            compilation_counter["n_compilations"] += 1

            new_params, new_state = run_m_step(
                params,
                X,
                y,
                log_gammas,
                log_xis,
                is_new_session=is_new_session,
                m_step_fn_glm_params=m_step_fn_glm_params,
                m_step_fn_glm_scale=None,
                inverse_link_function=inverse_link_function,
                dirichlet_prior_alphas_transition=dirichlet_prior_alphas_transition,
                dirichlet_prior_alphas_init_prob=dirichlet_prior_alphas_init_prob,
            )

            return new_params, new_state

        params = GLMHMMParams(
            glm_params=GLMParams(np.zeros_like(coef), np.zeros_like(intercept)),
            glm_scale=GLMScale(jnp.zeros(intercept.shape[-1])),
            hmm_params=HMMParams(None, None),
        )

        # call with no prior
        _ = tracked_run_m_step(
            params,
            X,
            y,
            log_gammas,
            log_xis,
            is_new_session=new_sess.astype(bool),
            m_step_fn_glm_params=solver.run,
            m_step_fn_glm_scale=None,
            inverse_link_function=inv_link,
            dirichlet_prior_alphas_transition=None,
            dirichlet_prior_alphas_init_prob=None,
        )
        assert compilation_counter["n_compilations"] == 1, "First call should compile"

        # second call with no prior
        _ = tracked_run_m_step(
            params,
            X,
            y,
            log_gammas,
            log_xis,
            is_new_session=new_sess.astype(bool),
            m_step_fn_glm_params=solver.run,
            m_step_fn_glm_scale=None,
            inverse_link_function=inv_link,
            dirichlet_prior_alphas_transition=None,
            dirichlet_prior_alphas_init_prob=None,
        )
        assert compilation_counter["n_compilations"] == 1, "None prior not cached!"

        # third call with prior (array)
        _ = tracked_run_m_step(
            params,
            X,
            y,
            log_gammas,
            log_xis,
            is_new_session=new_sess.astype(bool),
            m_step_fn_glm_params=solver.run,
            m_step_fn_glm_scale=None,
            inverse_link_function=obs.default_inverse_link_function,
            dirichlet_prior_alphas_transition=np.ones(transition_prob.shape),
            dirichlet_prior_alphas_init_prob=np.ones(initial_prob.shape),
        )
        assert (
            compilation_counter["n_compilations"] == 2
        ), "None -> array should recompile"

        # 4th call with prior (different values, same shape)
        _ = tracked_run_m_step(
            params,
            X,
            y,
            log_gammas,
            log_xis,
            is_new_session=new_sess.astype(bool),
            m_step_fn_glm_params=solver.run,
            m_step_fn_glm_scale=None,
            inverse_link_function=inv_link,
            dirichlet_prior_alphas_transition=2 * np.ones(transition_prob.shape),
            dirichlet_prior_alphas_init_prob=2 * np.ones(initial_prob.shape),
        )
        assert compilation_counter["n_compilations"] == 2, "Array prior not cached!"

    @pytest.mark.parametrize("solver_name", ["LBFGS", "ProximalGradient"])
    @pytest.mark.requires_x64
    @pytest.mark.parametrize(
        "generate_data_multi_state",
        [{"observations": PoissonObservations(), "scale": 1.0}],
        indirect=True,
    )
    def test_em_glm_hmm_compiles_once(self, generate_data_multi_state, solver_name):
        """
        Test that em_glm_hmm compiles only once for repeated calls.

        Ensures no unnecessary recompilation occurs when calling
        with same static arguments and array shapes.
        """
        (
            new_sess,
            initial_prob,
            transition_prob,
            coef,
            intercept,
            X,
            y,
            obs,
            scale,
            inv_link,
        ) = generate_data_multi_state
        _, solver = prepare_partial_hmm_nll_single_neuron(
            obs, init_params=GLMParams(coef, intercept)
        )

        obs = BernoulliObservations()
        likelihood_func = prepare_estep_log_likelihood(False, obs)
        negative_log_likelihood_func = prepare_mstep_nll_for_analytical_scale(
            False, obs
        )

        def partial_posterior_weighted_glm_negative_log_likelihood(
            weights, design_matrix, observations, posterior_prob
        ):
            return posterior_weighted_glm_negative_log_likelihood(
                weights,
                X=design_matrix,
                y=observations,
                posteriors=posterior_prob,
                inverse_link_function=inv_link,
                negative_log_likelihood_func=negative_log_likelihood_func,
            )

        glm = GLM(observation_model=obs, solver_name=solver_name)
        glm._instantiate_solver(
            partial_posterior_weighted_glm_negative_log_likelihood,
            GLMParams(coef, intercept),
        )

        # Create tracked version with compilation counter
        compilation_counter = {"n_compilations": 0}

        @partial(
            jax.jit,
            static_argnames=[
                "inverse_link_function",
                "likelihood_func",
                "m_step_fn_glm_params",
                "m_step_fn_glm_scale",
                "maxiter",
                "check_convergence",
                "tol",
            ],
        )
        def tracked_em_glm_hmm(
            params,
            X,
            y,
            likelihood_func,
            m_step_fn_glm_params,
            inverse_link_function,
            is_new_session=None,
            m_step_fn_glm_scale=None,
            maxiter=10**3,
            tol=1e-8,
            check_convergence=check_log_likelihood_increment,
        ):
            # This increment only runs during tracing (compilation)
            compilation_counter["n_compilations"] += 1

            p, s = em_glm_hmm(
                params=params,
                X=X,
                y=y,
                inverse_link_function=inverse_link_function,
                log_likelihood_func=likelihood_func,
                m_step_fn_glm_params=m_step_fn_glm_params,
                m_step_fn_glm_scale=m_step_fn_glm_scale,
                maxiter=maxiter,
                is_new_session=is_new_session,
                tol=tol,
                check_convergence=check_convergence,
            )

            return p, s

        params = GLMHMMParams(
            glm_params=GLMParams(coef, intercept),
            glm_scale=GLMScale(jnp.zeros(transition_prob.shape[0])),
            hmm_params=HMMParams(jnp.log(initial_prob), jnp.log(transition_prob)),
        )

        # First call - should compile
        _ = tracked_em_glm_hmm(
            params,
            X,
            y,
            inverse_link_function=obs.default_inverse_link_function,
            likelihood_func=likelihood_func,
            m_step_fn_glm_params=glm._solver_run,
            m_step_fn_glm_scale=None,
            maxiter=5,
            tol=1e-8,
        )
        assert compilation_counter["n_compilations"] == 1, "First call should compile"

        # Second call with SAME arguments - should NOT recompile
        _ = tracked_em_glm_hmm(
            params,
            X,
            y,
            inverse_link_function=obs.default_inverse_link_function,
            likelihood_func=likelihood_func,
            m_step_fn_glm_params=glm._solver_run,
            m_step_fn_glm_scale=None,
            maxiter=5,
            tol=1e-8,
        )
        assert (
            compilation_counter["n_compilations"] == 1
        ), "Second call should use cache"

        # Third call with DIFFERENT data (same shape) - should NOT recompile
        X_new = (X + np.random.randn(*X.shape) * 0.1).astype(X.dtype)
        y_new = y.copy()
        initial_prob_new = np.ones_like(initial_prob) / len(initial_prob)
        transition_prob_new = np.ones_like(transition_prob) / len(initial_prob)
        coef_new = coef * np.random.randn(*coef.shape)
        intercept_new = intercept * np.random.randn(*intercept.shape)
        params_new = GLMHMMParams(
            glm_params=GLMParams(coef_new, intercept_new),
            glm_scale=GLMScale(jnp.zeros(transition_prob.shape[0])),
            hmm_params=HMMParams(
                jnp.log(initial_prob_new), jnp.log(transition_prob_new)
            ),
        )

        _ = tracked_em_glm_hmm(
            params_new,
            X_new,
            y_new,
            inverse_link_function=obs.default_inverse_link_function,
            likelihood_func=likelihood_func,
            m_step_fn_glm_params=glm._solver_run,
            m_step_fn_glm_scale=None,
            maxiter=5,
            tol=1e-8,
        )
        assert (
            compilation_counter["n_compilations"] == 1
        ), "Different data (same shape) should use cache"

    @pytest.mark.requires_x64
    @pytest.mark.parametrize(
        "generate_data_multi_state",
        [{"observations": PoissonObservations(), "scale": 1.0}],
        indirect=True,
    )
    def test_forward_backward_compiles_once(self, generate_data_multi_state):
        """
        Test that forward_backward is not recompiled on each EM iteration.

        forward_backward should compile once and be reused across all EM steps.
        """
        (
            new_sess,
            initial_prob,
            transition_prob,
            coef,
            intercept,
            X,
            y,
            obs,
            scale,
            inv_link,
        ) = generate_data_multi_state
        likelihood_func = prepare_estep_log_likelihood(False, obs)
        negative_log_likelihood_func = prepare_mstep_nll_for_analytical_scale(
            False, obs
        )

        def partial_posterior_weighted_glm_negative_log_likelihood(
            weights, design_matrix, observations, posterior_prob
        ):
            return posterior_weighted_glm_negative_log_likelihood(
                weights,
                X=design_matrix,
                y=observations,
                posteriors=posterior_prob,
                inverse_link_function=inv_link,
                negative_log_likelihood_func=negative_log_likelihood_func,
            )

        glm = GLM(observation_model=obs, solver_name="LBFGS")
        glm._instantiate_solver(
            partial_posterior_weighted_glm_negative_log_likelihood,
            GLMParams(coef, intercept),
        )

        # Create tracked version with compilation counter
        compilation_counter = {"n_compilations": 0}

        @partial(
            jax.jit, static_argnames=["inverse_link_function", "log_likelihood_func"]
        )
        def tracked_forward_backward(
            X,
            y,
            log_initial_prob,
            log_transition_prob,
            glm_params,
            glm_scale,
            inverse_link_function,
            log_likelihood_func,
            is_new_session=None,
        ):
            # This increment only runs during tracing (compilation)
            compilation_counter["n_compilations"] += 1

            return forward_backward(
                X,  # drop intercept
                y,
                log_initial_prob,
                log_transition_prob,
                glm_params,
                glm_scale=glm_scale,
                log_likelihood_func=log_likelihood_func,
                inverse_link_function=inverse_link_function,
                is_new_session=is_new_session,
            )

        _ = tracked_forward_backward(
            X,
            y,
            jnp.log(initial_prob),
            jnp.log(transition_prob),
            GLMParams(coef, intercept),
            glm_scale=GLMScale(jnp.zeros(initial_prob.shape[0])),
            log_likelihood_func=likelihood_func,
            inverse_link_function=inv_link,
            is_new_session=new_sess.astype(bool),
        )
        assert compilation_counter["n_compilations"] == 1, "First call should compile"

        # second call with new data (same shape and size)
        X_new = (X + np.random.randn(*X.shape) * 0.1).astype(X.dtype)
        y_new = y.copy()
        initial_prob_new = np.ones_like(initial_prob) / len(initial_prob)
        transition_prob_new = np.ones_like(transition_prob) / len(initial_prob)
        coef_new = coef * np.random.randn(*coef.shape)
        intercept_new = intercept * np.random.randn(*intercept.shape)
        _ = tracked_forward_backward(
            X_new,
            y_new,
            jnp.log(initial_prob_new),
            jnp.log(transition_prob_new),
            GLMParams(coef_new, intercept_new),
            glm_scale=GLMScale(jnp.zeros(initial_prob.shape[0])),
            log_likelihood_func=likelihood_func,
            inverse_link_function=inv_link,
            is_new_session=new_sess.astype(bool),
        )

        assert compilation_counter["n_compilations"] == 1, (
            f"forward_backward compiled {compilation_counter['n_compilations']} times, "
            f"expected 1 compilation"
        )


class TestPytreeSupport:
    """Test that GLM-HMM algorithms support pytree inputs."""

    @pytest.mark.requires_x64
    @pytest.mark.parametrize(
        "generate_data_multi_state",
        [{"observations": PoissonObservations(), "scale": 1.0}],
        indirect=True,
    )
    def test_forward_backward_with_pytree(self, generate_data_multi_state):
        """Test forward_backward accepts pytree inputs for X and coef."""
        (
            new_sess,
            initial_prob,
            transition_prob,
            coef,
            intercept,
            X,
            y,
            obs,
            scale,
            inv_link,
        ) = generate_data_multi_state
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

        likelihood_func = prepare_estep_log_likelihood(
            is_population_glm=False,
            observation_model=obs,
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
            jnp.log(initial_prob),
            jnp.log(transition_prob),
            GLMParams(coef, intercept),
            glm_scale=GLMScale(jnp.zeros(initial_prob.shape[0])),
            log_likelihood_func=likelihood_func,
            inverse_link_function=inv_link,
            is_new_session=new_sess.astype(bool),
        )

        # Test with pytrees
        posteriors, joint_posterior, ll, ll_norm, alphas, betas = forward_backward(
            X_tree,
            y,
            jnp.log(initial_prob),
            jnp.log(transition_prob),
            GLMParams(coef_tree, intercept),
            glm_scale=GLMScale(jnp.zeros(initial_prob.shape[0])),
            log_likelihood_func=likelihood_func,
            inverse_link_function=inv_link,
            is_new_session=new_sess.astype(bool),
        )

        # Results should be identical (allow small numerical errors from floating point)
        np.testing.assert_allclose(posteriors, posteriors_ref, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(
            joint_posterior, joint_posterior_ref, rtol=1e-13, atol=1e-14
        )
        np.testing.assert_allclose(ll, ll_ref, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(ll_norm, ll_norm_ref, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(alphas, alphas_ref, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(betas, betas_ref, rtol=1e-13, atol=1e-14)

    @pytest.mark.requires_x64
    @pytest.mark.parametrize(
        "generate_data_multi_state",
        [{"observations": PoissonObservations(), "scale": 1.0}],
        indirect=True,
    )
    def test_posterior_weighted_glm_negative_log_likelihood_with_pytree(
        self, generate_data_multi_state
    ):
        """Test posterior_weighted_glm_negative_log_likelihood accepts pytree inputs."""
        (
            new_sess,
            initial_prob,
            transition_prob,
            coef,
            intercept,
            X,
            y,
            obs,
            scale,
            inv_link,
        ) = generate_data_multi_state
        # Split X and coef into dictionaries
        X_tree = {
            "feature_a": X[:, :1],
            "feature_b": X[:, 1:],
        }
        coef_tree = {
            "feature_a": coef[:1, :],
            "feature_b": coef[1:, :],
        }

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
        nll_ref = posterior_weighted_glm_negative_log_likelihood(
            GLMParams(coef, intercept),
            X,
            y,
            posteriors,
            inv_link,
            negative_log_likelihood,
        )

        # Test with pytrees
        nll = posterior_weighted_glm_negative_log_likelihood(
            GLMParams(coef_tree, intercept),
            X_tree,
            y,
            posteriors,
            inv_link,
            negative_log_likelihood,
        )

        # Results should be identical
        np.testing.assert_allclose(nll, nll_ref)

    @pytest.mark.requires_x64
    @pytest.mark.parametrize(
        "generate_data_multi_state",
        [{"observations": PoissonObservations(), "scale": 1.0}],
        indirect=True,
    )
    def test_em_glm_hmm_with_pytree(self, generate_data_multi_state):
        """Test em_glm_hmm accepts pytree inputs for X and coef."""
        (
            new_sess,
            initial_prob,
            transition_prob,
            coef,
            intercept,
            X,
            y,
            obs,
            scale,
            inv_link,
        ) = generate_data_multi_state
        # Split X and coef into dictionaries
        X_tree = {
            "feature_a": X[:, :1],
            "feature_b": X[:, 1:],
        }
        coef_tree = {
            "feature_a": coef[:1, :],
            "feature_b": coef[1:, :],
        }

        likelihood_func = prepare_estep_log_likelihood(
            is_population_glm=False,
            observation_model=obs,
        )
        vmap_nll = prepare_mstep_nll_for_analytical_scale(
            is_population_glm=False,
            observation_model=obs,
        )

        # Create solver using GLM class
        def partial_posterior_weighted_glm_negative_log_likelihood(
            weights, design_matrix, observations, posterior_prob
        ):
            return posterior_weighted_glm_negative_log_likelihood(
                weights,
                X=design_matrix,
                y=observations,
                posteriors=posterior_prob,
                inverse_link_function=inv_link,
                negative_log_likelihood_func=vmap_nll,
            )

        glm = GLM(observation_model=obs, solver_name="LBFGS")
        glm._instantiate_solver(
            partial_posterior_weighted_glm_negative_log_likelihood,
            GLMParams(coef_tree, intercept),
        )
        solver_run = glm._solver_run

        # Create initial parameters
        params = GLMHMMParams(
            glm_params=GLMParams(coef_tree, intercept),
            glm_scale=GLMScale(jnp.zeros(initial_prob.shape[0])),
            hmm_params=HMMParams(jnp.log(initial_prob), jnp.log(transition_prob)),
        )

        # Run EM with pytrees (just a few iterations)
        final_params, final_state = em_glm_hmm(
            params=params,
            X=X_tree,
            y=y,
            inverse_link_function=inv_link,
            log_likelihood_func=likelihood_func,
            m_step_fn_glm_params=solver_run,
            m_step_fn_glm_scale=None,
            is_new_session=new_sess.astype(bool),
            maxiter=3,
            tol=1e-8,
        )

        # Just verify it runs and returns valid outputs
        assert final_params.hmm_params.log_initial_prob.shape == initial_prob.shape
        assert (
            final_params.hmm_params.log_transition_prob.shape == transition_prob.shape
        )
        assert isinstance(final_params, GLMHMMParams)
        assert isinstance(final_params.glm_params.coef, dict)  # coef should be a dict
        assert final_state.iterations > 0


@pytest.mark.requires_x64
class TestEMScaleOptimization:
    """
    Integration tests for EM algorithm with scale parameter optimization.

    This test class uses dedicated fixtures rather than the existing
    `generate_data_multi_state` fixture for the following reasons:

    1. **Tailored data generation**: These integration tests require specific
       configurations (different numbers of states, sample sizes) optimized for
       testing convergence behavior rather than general algorithm correctness.

    2. **Access to ground truth**: The fixtures return dictionaries containing
       true parameters (`true_scale`, `true_coef`, `true_intercept`) which are
       essential for validating parameter recovery and convergence quality.

    3. **Test independence**: Using `scope="function"` fixtures ensures each test
       has independent data, avoiding interference between tests that compare
       different optimization strategies (e.g., with vs without scale optimization).

    4. **Flexibility**: Different tests need different observation models (Gaussian
       with analytical updates, Gamma with numerical updates) and both single-neuron
       and population configurations, which would require extensive parametrization
       of the existing fixtures.

    The existing `generate_data_multi_state` fixtures remain appropriate for unit
    tests and algorithm correctness checks where ground truth parameters are not
    needed and module-scoped fixtures improve test performance.
    """

    @pytest.fixture
    def gaussian_data_single_neuron(self):
        """Generate synthetic Gaussian GLM-HMM data for single neuron."""
        np.random.seed(42)
        n_samples, n_features, n_states = 200, 3, 4

        # True parameters
        true_coef = np.random.randn(n_features, n_states) * 0.5
        true_intercept = np.random.randn(n_states)
        true_scale = np.random.uniform(
            0.5, 2.0, n_states
        )  # Different variance per state

        # HMM parameters
        initial_prob = np.ones(n_states) / n_states
        transition_prob = np.eye(n_states) * 0.9 + (1 - np.eye(n_states)) * 0.1 / (
            n_states - 1
        )

        # Generate data
        X = np.random.randn(n_samples, n_features)
        states = np.zeros(n_samples, dtype=int)
        states[0] = np.random.choice(n_states, p=initial_prob)
        for t in range(1, n_samples):
            states[t] = np.random.choice(n_states, p=transition_prob[states[t - 1]])

        # Generate observations
        rates = X @ true_coef + true_intercept
        y = rates[np.arange(n_samples), states] + np.random.randn(n_samples) * np.sqrt(
            true_scale[states]
        )

        return {
            "X": X,
            "y": y,
            "true_coef": true_coef,
            "true_intercept": true_intercept,
            "true_scale": true_scale,
            "initial_prob": initial_prob,
            "transition_prob": transition_prob,
            "n_states": n_states,
        }

    @pytest.fixture
    def gaussian_data_population(self):
        """Generate synthetic Gaussian GLM-HMM data for population."""
        np.random.seed(123)
        n_samples, n_features, n_neurons, n_states = 150, 2, 3, 3

        # True parameters
        true_coef = np.random.randn(n_features, n_neurons, n_states) * 0.3
        true_intercept = np.random.randn(n_neurons, n_states)
        true_scale = np.random.uniform(0.3, 1.5, (n_neurons, n_states))

        # HMM parameters
        initial_prob = np.ones(n_states) / n_states
        transition_prob = np.eye(n_states) * 0.85 + (1 - np.eye(n_states)) * 0.15 / (
            n_states - 1
        )

        # Generate data
        X = np.random.randn(n_samples, n_features)
        states = np.zeros(n_samples, dtype=int)
        states[0] = np.random.choice(n_states, p=initial_prob)
        for t in range(1, n_samples):
            states[t] = np.random.choice(n_states, p=transition_prob[states[t - 1]])

        # Generate observations
        rates = np.einsum("tf,fnk->tnk", X, true_coef) + true_intercept
        y = np.zeros((n_samples, n_neurons))
        for t in range(n_samples):
            y[t] = rates[t, :, states[t]] + np.random.randn(n_neurons) * np.sqrt(
                true_scale[:, states[t]]
            )

        return {
            "X": X,
            "y": y,
            "true_coef": true_coef,
            "true_intercept": true_intercept,
            "true_scale": true_scale,
            "initial_prob": initial_prob,
            "transition_prob": transition_prob,
            "n_states": n_states,
        }

    @pytest.fixture
    def gamma_data_single_neuron(self):
        """Generate synthetic Gamma GLM-HMM data for single neuron."""
        np.random.seed(99)
        n_samples, n_features, n_states = 200, 3, 3

        # True parameters
        true_coef = np.random.randn(n_features, n_states) * 0.3
        true_intercept = (
            np.random.randn(n_states) - 1.0
        )  # Negative to ensure positive rates after exp
        true_scale = np.random.uniform(1.0, 3.0, n_states)

        # HMM parameters
        initial_prob = np.ones(n_states) / n_states
        transition_prob = np.eye(n_states) * 0.9 + (1 - np.eye(n_states)) * 0.1 / (
            n_states - 1
        )

        # Generate data
        X = np.random.randn(n_samples, n_features)
        states = np.zeros(n_samples, dtype=int)
        states[0] = np.random.choice(n_states, p=initial_prob)
        for t in range(1, n_samples):
            states[t] = np.random.choice(n_states, p=transition_prob[states[t - 1]])

        # Generate observations (Gamma with rate parameterization)
        rates = np.exp(X @ true_coef + true_intercept)
        y = np.zeros(n_samples)
        for t in range(n_samples):
            # Gamma: shape = scale, rate = scale / mean  =>  mean = rate
            shape = true_scale[states[t]]
            rate_param = shape / rates[t, states[t]]
            y[t] = np.random.gamma(shape, 1.0 / rate_param)

        return {
            "X": X,
            "y": y,
            "true_coef": true_coef,
            "true_intercept": true_intercept,
            "true_scale": true_scale,
            "initial_prob": initial_prob,
            "transition_prob": transition_prob,
            "n_states": n_states,
            "states": states,
        }

    def test_em_gaussian_analytical_scale_single_neuron(
        self, gaussian_data_single_neuron
    ):
        """
        Test #1: Full EM with Gaussian observations using analytical scale update (single neuron).

        Verifies that:
        1. EM converges with analytical scale optimization
        2. Scale parameters improve over iterations
        3. Final likelihood is better than initial
        """
        data = gaussian_data_single_neuron
        obs = GaussianObservations()

        # Initialize parameters (intentionally poor initialization)
        init_coef = np.random.randn(*data["true_coef"].shape) * 0.1
        init_intercept = np.random.randn(data["n_states"]) * 0.1
        init_scale = jnp.ones(data["n_states"])  # Start with all scales = 1

        # Prepare EM components
        likelihood_func = prepare_estep_log_likelihood(False, obs)
        nll_params = prepare_mstep_nll_objective_param(False, obs, lambda x: x)
        scale_update_fn = get_analytical_scale_update(obs, is_population_glm=False)

        solver = setup_solver(
            nll_params, init_params=GLMParams(init_coef, init_intercept), tol=10**-6
        )

        # Create initial parameters
        params = GLMHMMParams(
            glm_params=GLMParams(init_coef, init_intercept),
            glm_scale=GLMScale(jnp.log(init_scale)),
            hmm_params=HMMParams(
                jnp.log(data["initial_prob"]), jnp.log(data["transition_prob"])
            ),
        )

        # Run EM with scale optimization
        final_params, final_state = em_glm_hmm(
            params=params,
            X=data["X"],
            y=data["y"],
            inverse_link_function=lambda x: x,  # Identity link for Gaussian
            log_likelihood_func=likelihood_func,
            m_step_fn_glm_params=solver.run,
            m_step_fn_glm_scale=scale_update_fn,
            maxiter=50,
            tol=1e-5,
        )

        # Verify convergence
        assert final_state.iterations > 0, "EM should have run at least one iteration"

        # Verify scale was updated (should differ from initialization)
        final_scale = jnp.exp(final_params.glm_scale.log_scale)
        assert not jnp.allclose(
            final_scale, init_scale, atol=0.1
        ), "Scale should have been updated from initialization"

        # Verify scale is positive
        assert jnp.all(final_scale > 0), "All scale parameters should be positive"

        # Verify shapes
        assert final_scale.shape == (data["n_states"],)

    def test_em_gaussian_scale_improves_likelihood(self, gaussian_data_single_neuron):
        """
        Test #3: Compare EM with and without scale optimization for Gaussian.

        Verifies that:
        1. EM with scale optimization achieves higher or equal likelihood
        2. Both configurations converge successfully
        """
        data = gaussian_data_single_neuron
        obs = GaussianObservations()

        # Shared initialization (use fixed seed for reproducibility)
        np.random.seed(999)
        init_coef = np.random.randn(*data["true_coef"].shape) * 0.1
        init_intercept = np.random.randn(data["n_states"]) * 0.1
        init_scale = jnp.ones(data["n_states"])

        # Prepare components
        likelihood_func = prepare_estep_log_likelihood(False, obs)
        nll_params = prepare_mstep_nll_objective_param(False, obs, lambda x: x)
        scale_update_fn = get_analytical_scale_update(obs, is_population_glm=False)

        solver = setup_solver(
            nll_params,
            init_params=GLMParams(init_coef.copy(), init_intercept.copy()),
            tol=10**-6,
        )

        # Create initial parameters
        params_no_scale = GLMHMMParams(
            glm_params=GLMParams(init_coef.copy(), init_intercept.copy()),
            glm_scale=GLMScale(jnp.log(init_scale)),
            hmm_params=HMMParams(
                jnp.log(data["initial_prob"]), jnp.log(data["transition_prob"])
            ),
        )

        # Run EM WITHOUT scale optimization
        _, state_no_scale = em_glm_hmm(
            params=params_no_scale,
            X=data["X"],
            y=data["y"],
            inverse_link_function=lambda x: x,
            log_likelihood_func=likelihood_func,
            m_step_fn_glm_params=solver.run,
            m_step_fn_glm_scale=None,  # No scale optimization
            maxiter=50,
            tol=1e-5,
        )

        # Create initial parameters for with-scale run
        params_with_scale = GLMHMMParams(
            glm_params=GLMParams(init_coef.copy(), init_intercept.copy()),
            glm_scale=GLMScale(jnp.log(init_scale)),
            hmm_params=HMMParams(
                jnp.log(data["initial_prob"]), jnp.log(data["transition_prob"])
            ),
        )

        # Run EM WITH scale optimization
        final_params, state_with_scale = em_glm_hmm(
            params=params_with_scale,
            X=data["X"],
            y=data["y"],
            inverse_link_function=lambda x: x,
            log_likelihood_func=likelihood_func,
            m_step_fn_glm_params=solver.run,
            m_step_fn_glm_scale=scale_update_fn,
            maxiter=50,
            tol=1e-5,
        )
        final_scale = jnp.exp(final_params.glm_scale.log_scale)

        # Verify both converged
        assert state_no_scale.iterations > 0
        assert state_with_scale.iterations > 0

        # The version with scale optimization should achieve equal or better likelihood
        # Note: We compare final log-likelihoods stored in the state
        # Since we're optimizing scale, it should at minimum match the fixed scale version
        assert (
            state_with_scale.data_log_likelihood
            >= state_no_scale.data_log_likelihood - 1e-3
        ), "EM with scale optimization should achieve at least as good likelihood"

        # Verify scale changed
        assert not jnp.allclose(
            final_scale, init_scale, atol=0.1
        ), "Scale parameters should have been optimized"

    def test_em_gamma_numerical_scale_single_neuron(self, gamma_data_single_neuron):
        """
        Test #2: Full EM with Gamma observations using numerical scale update.

        Verifies that:
        1. Numerical scale optimization works for Gamma
        2. EM converges
        3. Scale parameters are updated appropriately
        """
        data = gamma_data_single_neuron
        obs = GammaObservations()

        # Initialize parameters
        init_coef = np.random.randn(*data["true_coef"].shape) * 0.1
        init_intercept = np.random.randn(data["n_states"]) - 1.0
        init_scale = jnp.ones(data["n_states"]) * np.std(
            data["y"]
        )  # Initialize to moderate value

        # Prepare EM components
        likelihood_func = prepare_estep_log_likelihood(False, obs)
        nll_params = prepare_mstep_nll_objective_param(False, obs, jnp.exp)
        nll_scale = prepare_mstep_nll_objective_scale(False, obs)

        solver_params = setup_solver(
            nll_params, init_params=GLMParams(init_coef, init_intercept), tol=10**-6
        )
        solver_scale = setup_solver(
            nll_scale, init_params=GLMScale(jnp.log(init_scale)), tol=10**-6
        )

        # Create initial parameters
        params = GLMHMMParams(
            glm_params=GLMParams(init_coef, init_intercept),
            glm_scale=GLMScale(jnp.log(init_scale)),
            hmm_params=HMMParams(
                jnp.log(data["initial_prob"]), jnp.log(data["transition_prob"])
            ),
        )

        # Run EM with numerical scale optimization
        final_params, final_state = em_glm_hmm(
            params=params,
            X=data["X"],
            y=data["y"],
            inverse_link_function=jnp.exp,
            log_likelihood_func=likelihood_func,
            m_step_fn_glm_params=solver_params.run,
            m_step_fn_glm_scale=solver_scale.run,  # Numerical optimization
            maxiter=50,
            tol=1e-5,
        )

        # Verify convergence
        assert final_state.iterations > 0, "EM should have run at least one iteration"

        # Verify scale was updated
        final_scale = jnp.exp(final_params.glm_scale.log_scale)
        assert not jnp.allclose(
            final_scale, init_scale, atol=0.1
        ), "Scale should have been updated from initialization"

        # Verify shapes
        assert final_scale.shape == (data["n_states"],)

    def test_em_gaussian_analytical_scale_population(self, gaussian_data_population):
        """
        Test #5: Full EM for population GLM with Gaussian observations and analytical scale.

        Verifies:
        1. Population case works with analytical scale update
        2. Scale has shape (n_neurons, n_states)
        3. EM converges
        """
        data = gaussian_data_population
        obs = GaussianObservations()

        # Initialize parameters
        init_coef = np.random.randn(*data["true_coef"].shape) * 0.1
        init_intercept = np.random.randn(*data["true_intercept"].shape) * 0.1
        init_scale = jnp.ones(data["true_scale"].shape)

        # Prepare EM components
        likelihood_func = prepare_estep_log_likelihood(
            True, obs
        )  # is_population_glm=True
        nll_params = prepare_mstep_nll_objective_param(True, obs, lambda x: x)
        scale_update_fn = get_analytical_scale_update(obs, is_population_glm=True)

        solver = setup_solver(
            nll_params, init_params=GLMParams(init_coef, init_intercept), tol=10**-6
        )

        # Create initial parameters
        params = GLMHMMParams(
            glm_params=GLMParams(init_coef, init_intercept),
            glm_scale=GLMScale(jnp.log(init_scale)),
            hmm_params=HMMParams(
                jnp.log(data["initial_prob"]), jnp.log(data["transition_prob"])
            ),
        )

        # Run EM with scale optimization
        final_params, final_state = em_glm_hmm(
            params=params,
            X=data["X"],
            y=data["y"],
            inverse_link_function=lambda x: x,
            log_likelihood_func=likelihood_func,
            m_step_fn_glm_params=solver.run,
            m_step_fn_glm_scale=scale_update_fn,
            maxiter=50,
            tol=1e-5,
        )

        # Verify convergence
        assert final_state.iterations > 0, "EM should have run at least one iteration"

        # Verify scale shape (n_neurons, n_states)
        final_scale = jnp.exp(final_params.glm_scale.log_scale)
        n_neurons = data["y"].shape[1]
        assert final_scale.shape == (
            n_neurons,
            data["n_states"],
        ), f"Expected scale shape ({n_neurons}, {data['n_states']}), got {final_scale.shape}"

        # Verify scale was updated
        assert not jnp.allclose(
            final_scale, init_scale, atol=0.1
        ), "Scale should have been updated from initialization"

        # Verify all scales are positive
        assert jnp.all(final_scale > 0), "All scale parameters should be positive"
