"""Forward backward pass for a GLM-HMM."""

from functools import partial
from typing import Any, Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from numpy.typing import NDArray

from ..tree_utils import pytree_map_and_reduce

Array = NDArray | jax.numpy.ndarray


class GLMHMMState(eqx.Module):
    """State class for the GLMHHM EM-algorithm."""

    log_initial_prob: Array
    log_transition_matrix: Array
    glm_params: Tuple[Array, Array]  # (coef, intercept)
    data_log_likelihood: float | Array
    previous_data_log_likelihood: float | Array
    log_likelihood_history: Array
    iterations: int


def _analytical_m_step_initial_prob(
    posteriors: jnp.ndarray,
    is_new_session: jnp.ndarray,
    dirichlet_prior_alphas: Optional[jnp.ndarray] = None,
):
    """
    Compute the M-step update for initial state probabilities.

    Analytically computes the maximum-likelihood (or MAP with Dirichlet prior)
    estimate of the initial state distribution. Computation is performed in
    probability space for efficiency.

    Parameters
    ----------
    posteriors :
        Posterior probabilities over latent states, shape ``(n_time_bins, n_states)``.
    is_new_session :
        Boolean array indicating session start time bins, shape ``(n_time_bins,)``.
        Only these positions contribute to the initial state estimate.
    dirichlet_prior_alphas :
        Dirichlet prior parameters for the initial distribution,
        shape ``(n_states,)``. If None, uses uniform prior.
        **Note**: All alpha values must be >= 1.

    Returns
    -------
    initial_prob :
        Initial state probabilities, shape ``(n_states,)``.
        Normalized to sum to 1.

    Notes
    -----
    The current implementation requires Dirichlet prior parameters alpha >= 1.
    Support for sparse priors (0 < alpha < 1) may be added in a future version
    using alternative optimization methods.
    """
    # Mask and sum
    masked_posteriors = jnp.where(is_new_session[:, jnp.newaxis], posteriors, 0.0)
    counts = jnp.sum(masked_posteriors, axis=0)

    # Add prior
    if dirichlet_prior_alphas is not None:
        numerator = counts + (dirichlet_prior_alphas - 1)
    else:
        numerator = counts

    # Normalize
    initial_prob = numerator / jnp.sum(numerator)

    return initial_prob


def _analytical_m_step_transition_prob(
    joint_posterior: jnp.ndarray,
    dirichlet_prior_alphas: Optional[jnp.ndarray] = None,
):
    """
    Compute the M-step update for the transition probability matrix.

    Analytically computes the maximum-likelihood (or MAP with Dirichlet prior)
    estimate of the transition matrix using expected transition counts.
    Computation is performed in probability space for efficiency.

    Parameters
    ----------
    joint_posterior :
        Expected transition counts from state i to j (in probability space),
        shape ``(n_states, n_states)``.
    dirichlet_prior_alphas :
        Dirichlet prior parameters for each row of the transition matrix,
        shape ``(n_states, n_states)``. If None, uses uniform prior.
        **Note**: All alpha values must be >= 1.

    Returns
    -------
    transition_prob :
        Transition probability matrix, shape ``(n_states, n_states)``.
        Each row is normalized to sum to 1.

    Notes
    -----
    The current implementation requires Dirichlet prior parameters alpha >= 1.
    Support for sparse priors (0 < alpha < 1) may be added in a future version
    using alternative optimization methods.
    """

    if dirichlet_prior_alphas is not None:
        numerator = joint_posterior + (dirichlet_prior_alphas - 1)
    else:
        numerator = joint_posterior

    # Normalize rows
    row_sums = jnp.sum(numerator, axis=1, keepdims=True)
    transition_prob = numerator / row_sums

    return transition_prob


def compute_xi_log(
    log_alphas,
    log_betas,
    log_conditional_prob,
    log_normalization,
    is_new_session,
    log_transition_prob,
):
    """
    Compute the sum of the joint posterior (xi) over consecutive latent states in log-space.

    Compute the sum of the joint posterior (xis, eqn. 13.14 of [1]_) over samples, implementing
    the summation required in the eqn. 13.19 of [1]_.

    Parameters
    ----------
    log_alphas :
        Log forward messages, shape ``(n_time_bins, n_states)``
    log_betas :
        Log backward messages, shape ``(n_time_bins, n_states)``.
    log_conditional_prob :
        Log observation likelihoods log p(y_t | z_t), shape ``(n_time_bins, n_states)``.
    log_normalization :
        Log normalization constants from forward pass, shape ``(n_time_bins,)``.
    is_new_session :
        Boolean array, True at start of new sessions, shape ``(n_time_bins,)``.
    log_transition_prob :
        Log transition probability matrix, shape ``(n_states, n_states)``.

    Returns
    -------
    :
        Log expected joint posteriors between time steps, shape ``(n_states, n_states)``.

    References
    ----------
    .. [1] Bishop, C. M. (2006). Pattern recognition and machine learning. Springer.

    """
    # shift alpha so that alpha[t-1] aligns with beta[t]
    norm_log_alpha = log_alphas[:-1] - log_normalization[1:, jnp.newaxis]

    # mask out steps where t is a new session
    norm_log_alpha = jnp.where(
        is_new_session[1:, jnp.newaxis], -jnp.inf, norm_log_alpha
    )

    # Compute xi sum in one matmul
    log_xi_sum = jax.scipy.special.logsumexp(
        norm_log_alpha.T[..., jnp.newaxis]
        + (log_conditional_prob[1:] + log_betas[1:])[jnp.newaxis],
        axis=1,
    )

    return log_xi_sum + log_transition_prob


def forward_pass(
    log_initial_prob: Array,
    log_transition_prob: Array,
    log_conditional_prob: Array,
    is_new_session: Array,
) -> Tuple[Array, Array]:
    """
    Forward pass of an HMM in log-space.

    This function performs the recursive forward pass over time using ``jax.lax.scan``,
    computing the filtered log-probabilities (``log alpha``, eqn. 13.34 and 13.36 of [1]_) at each time step.
    At the start of a new session, the recursion is reset using the initial state distribution.
    All computations are performed in log-space for numerical stability.

    Parameters
    ----------
    log_initial_prob :
        Initial state log-probability distribution, array of shape ``(n_states,)``.

    log_transition_prob :
        Log-transition matrix of shape ``(n_states, n_states)``, where entry ``log T[i, j]`` is the
        log-probability of transitioning from state ``i`` to state ``j``.

    log_conditional_prob :
        Array of shape ``(n_time_bins, n_states)``, representing the observation log-likelihood
        ``log p(y_t | z_t)`` at each time step for each state.

    is_new_session :
        Boolean array of shape ``(n_time_bins,)`` indicating the start of new sessions. When
        ``is_new_session[t]`` is True, the recursion at time ``t`` is reset using ``log_initial_prob``.

    Returns
    -------
    log_alphas :
        Array of shape ``(n_time_bins, n_states)``, containing the filtered log-probabilities
        at each time step. ``log_alphas[t]`` corresponds to the log forward message at time ``t``.

    log_normalizers :
        Array of shape ``(n_time_bins,)`` containing the log-normalization constants at each
        time step. The sum of these values gives the log-likelihood of the sequence.

    Notes
    -----
    All operations are performed in log-space to avoid numerical underflow/overflow.
    Normalization is performed at each time step using ``logsumexp`` for numerical stability.

    Equivalent pseudocode in standard Python (log-space version):

    .. code-block:: python

        n_time_bins, n_states = log_py_z.shape
        log_alphas = np.full((n_time_bins, n_states), -np.inf)
        log_c = np.full(n_time_bins, -np.inf)

        for t in range(n_time_bins):
            if new_sess[t]:
                log_alphas[t] = log_initial_prob + log_py_z[t]
            else:
                log_alphas[t] = log_py_z[t] + logsumexp(
                    log_transition_prob + log_alphas[t - 1][None, :],
                    axis=1
                )

            log_c[t] = logsumexp(log_alphas[t])
            log_alphas[t] -= log_c[t]

    References
    ----------
    .. [1] Bishop, C. M. (2006). Pattern recognition and machine learning. Springer.

    """

    def initial_compute(log_posterior, _):
        return log_posterior + log_initial_prob

    def transition_compute(log_posterior, log_alpha_previous):
        log_exp_transition = jax.scipy.special.logsumexp(
            log_transition_prob + log_alpha_previous[None, :],  # Broadcasting
            axis=1,  # Sum over second axis (after transpose)
        )
        return log_posterior + log_exp_transition

    def body_fn(carry, xs):
        log_alpha_previous = carry
        log_posterior, is_new_session = xs

        log_alpha = jax.lax.cond(
            is_new_session,
            initial_compute,
            transition_compute,
            log_posterior,
            log_alpha_previous,
        )
        log_const = jax.scipy.special.logsumexp(log_alpha)
        log_alpha = log_alpha - log_const
        return log_alpha, (log_alpha, log_const)

    init = jnp.full_like(log_conditional_prob[0], -jnp.inf)  # log(0)
    log_transition_prob = log_transition_prob.T
    _, (log_alphas, log_normalizers) = jax.lax.scan(
        body_fn, init, (log_conditional_prob, is_new_session)
    )
    return log_alphas, log_normalizers


def backward_pass(
    log_transition_prob: Array,
    log_conditional_prob: Array,
    log_normalizers: Array,
    is_new_session: Array,
):
    """
    Run the backward pass of the HMM inference algorithm to compute log-beta messages.

    This function performs the backward recursion step of the forward–backward algorithm,
    using ``jax.lax.scan`` in reverse to compute log-beta messages at each time step, computing
    the ``log beta`` parameters, see eqn. 13.35 and 13.38 of [1]_.
    It handles session boundaries by resetting the beta messages when a new session starts.
    All computations are performed in log-space for numerical stability.

    Parameters
    ----------
    log_transition_prob :
        Log-transition matrix of shape ``(n_states, n_states)``, where entry ``log T[i, j]`` is the
        log-probability of transitioning from state ``i`` to state ``j``.

    log_conditional_prob :
        Array of shape ``(n_time_bins, n_states)``, representing the observation log-likelihoods
        ``log p(y_t | z_t)`` at each time step for each state.

    log_normalizers :
        Array of shape ``(n_time_bins,)`` containing the log-normalization constants from the forward
        pass. These are used to normalize the backward recursion in log-space.

    is_new_session :
        Boolean array of shape ``(n_time_bins,)`` indicating the start of new sessions. When
        ``is_new_session[t]`` is True, the backward message at time ``t`` is reset to a vector of zeros
        (corresponding to log(1) for each state).

    Returns
    -------
    log_betas :
        Array of shape ``(n_time_bins, n_states)``, containing the log-beta messages at each time step.
        The indexing is aligned with the forward pass, such that ``log_betas[t]`` corresponds to the
        log backward message at time ``t``.

    Notes
    -----
    This implementation follows the standard HMM backward equations (Bishop, 2006, Eq. 13.38–13.39),
    adapted to log-space, including reinitialization for segmented sequences.

    Equivalent pseudocode in standard Python (log-space version):

    .. code-block:: python

        n_time_bins, n_states = log_py_z.shape
        log_betas = np.full((n_time_bins, n_states), -np.inf)
        log_betas[-1] = np.zeros(n_states)  # log(1) = 0

        for t in range(n_time_bins - 2, -1, -1):
            if new_sess[t + 1]:
                log_betas[t] = np.zeros(n_states)
            else:
                log_betas[t] = logsumexp(
                    log_transition_prob + (log_betas[t + 1] + log_py_z[t + 1])[None, :],
                    axis=1
                ) - log_c[t + 1]

    References
    ----------
    .. [1] Bishop, C. M. (2006). Pattern recognition and machine learning. Springer.

    """
    init = jnp.zeros_like(log_conditional_prob[0])

    def initial_compute(log_posterior, *_):
        # Initialize with log(ones) = zeros
        return jnp.zeros_like(log_posterior)

    def backward_step(log_posterior, log_betas, log_normalization):
        # Normalize (log of Equation 13.62)
        return (
            jax.scipy.special.logsumexp(
                log_transition_prob + (log_posterior + log_betas)[None, :], axis=1
            )
            - log_normalization
        )

    def body_fn(carry, xs):
        log_posterior, log_norm, is_new_sess = xs
        log_beta = jax.lax.cond(
            is_new_sess,
            initial_compute,
            backward_step,
            log_posterior,
            carry,
            log_norm,
        )
        return log_beta, carry

    # Keeping the output betas because I am interested in
    # all outputs, including the last one.
    _, log_betas = jax.lax.scan(
        body_fn,
        init,
        (log_conditional_prob, log_normalizers, is_new_session),
        reverse=True,
    )
    return log_betas


def initialize_new_session(n_samples, is_new_session):
    """Initialize new session indicator."""
    # Revise if the data is one single session or multiple sessions.
    # If new_sess is not provided, assume one session
    if is_new_session is None:
        # default: all False, but first time bin must be True
        is_new_session = jax.lax.dynamic_update_index_in_dim(
            jnp.zeros(n_samples, dtype=bool), True, 0, axis=0
        )
    else:
        # use the user-provided tree, but force the first time bin to be True
        is_new_session = jax.lax.dynamic_update_index_in_dim(
            jnp.asarray(is_new_session, dtype=bool), True, 0, axis=0
        )

    return is_new_session


def compute_rate_per_state(
    X: Any, glm_params: Any, inverse_link_function: Callable
) -> Array:
    """Compute the GLM mean per state."""
    coef, intercept = glm_params

    # Predicted y
    if jax.tree_util.tree_leaves(coef)[0].ndim > 2:
        lin_comb = pytree_map_and_reduce(
            lambda x, w: jnp.einsum("ik, kjw->ijw", x, w), sum, X, coef
        )
    else:
        lin_comb = pytree_map_and_reduce(lambda x, w: jnp.matmul(x, w), sum, X, coef)
    predicted_rate_given_state = inverse_link_function(lin_comb + intercept)
    return predicted_rate_given_state


@partial(jax.jit, static_argnames=["inverse_link_function", "log_likelihood_func"])
def forward_backward(
    X: Array,
    y: Array,
    log_initial_prob: Array,
    log_transition_prob: Array,
    glm_params: Tuple[Array, Array],
    inverse_link_function: Callable,
    log_likelihood_func: Callable[[Array, Array], Array],
    is_new_session: Array | None = None,
):
    """
    Run the forward-backward Baum-Welch algorithm.

    Run the forward-backward Baum-Welch algorithm [1]_ that compute a posterior distribution over latent
    states. It handles session boundaries by resetting the ``alpha`` and ``beta`` messages when a new
    session starts.

    Parameters
    ----------
    X :
        Design matrix, pytree with leaves of shape ``(n_time_bins, n_features)``.

    y :
        Observations, pytree with leaves of shape ``(n_time_bins,)``.

    log_initial_prob :
        Log of the initial latent state probability, pytree with leaves of shape ``(n_states, 1)``.

    log_transition_prob :
        Latent state log-transition matrix, pytree with leaves of shape ``(n_states, n_states)``.
        ``transition_prob[i, j]`` is the probability of transitioning from state ``i`` to state ``j``.

    glm_params :
        Length two tuple with the GLM coefficients of shape ``(n_features, n_states)``
        and intercept of shape ``(n_states,)``.

    inverse_link_function :
        Function mapping linear predictors to the mean of the observation distribution
        (e.g., exp for Poisson, sigmoid for Bernoulli).

    log_likelihood_func :
        Function computing the elementwise log-likelihood of observations given predicted mean values.
        Must return an array of shape ``(n_time_bins, n_states)``.

    is_new_session :
        Boolean array marking the start of a new session.
        If unspecified or empty, treats the full set of trials as a single session.

    Returns
    -------
    log_posteriors :
        Marginal log-posterior distribution over latent states, shape ``(n_time_bins, n_states)``.

    log_joint_posterior :
        Joint log-posterior distribution between consecutive time steps summed
        over samples, shape ``(n_states, n_states)``.

    log_likelihood :
        Total log-likelihood of the observation sequence under the model.

    log_likelihood_norm :
        The normalized total likelihood.

    log_alphas :
        Log forward messages (log alpha values), shape ``(n_time_bins, n_states)``.

    log_betas :
        Log backward messages (log beta values), shape ``(n_time_bins, n_states)``.

    References
    ----------
    .. [1] Bishop, C. M. (2006). *Pattern recognition and machine learning*. Springer.
    """
    # Initialize variables
    n_time_bins = y.shape[0]
    is_new_session = initialize_new_session(y.shape[0], is_new_session)
    predicted_rate_given_state = compute_rate_per_state(
        X, glm_params, inverse_link_function
    )

    # Compute likelihood given the fixed weights
    # Data likelihood p(y|z) from emissions model
    # NOTE:
    # For N neurons and S samples, we want the total likelihood
    # across neurons for each sample and latent state.

    # Example helper:
    # def combined_likelihood(log_likelihood_func, y, rate):
    #     # log_likelihood_func takes (y, rate) and returns log-likelihood per neuron:
    #     #   y:    shape (S, N)
    #     #   rate: shape (S, N, K)  # K = number of latent states
    #
    #     assert y.ndim == 2      # (samples, neurons)
    #     assert rate.ndim == 3   # (samples, neurons, states)
    #
    #     # vmap over the state axis: apply log_likelihood_func for each state
    #     # Result: shape (S, N, K)
    #     log_like = jax.vmap(log_likelihood_func, in_axes=(None, 2), out_axes=2)(y, rate)
    #
    #     # Combine neurons assuming conditional independence:
    #     # sum log-likelihoods over neurons (axis=1), then exponentiate for stability
    #     # Final shape: (S, K)
    #     return jnp.exp(jnp.sum(log_like, axis=1))
    #
    # Here, log_likelihood_func is the ``log_likelihood`` method from
    # nemos.observation_models.Observations with ``aggregate_sample_scores = lambda x:x``

    log_conditionals = log_likelihood_func(y, predicted_rate_given_state)

    # Compute forward pass
    log_alphas, log_normalization = forward_pass(
        log_initial_prob, log_transition_prob, log_conditionals, is_new_session
    )  # these are equivalent to the forward pass with python loop

    # Compute backward pass
    log_betas = backward_pass(
        log_transition_prob, log_conditionals, log_normalization, is_new_session
    )

    log_likelihood = jnp.sum(
        log_normalization
    )  # Store log-likelihood, log of Equation 13.63

    likelihood_norm = jnp.exp(log_likelihood / n_time_bins)  # Normalize

    # Posteriors
    # ----------
    # Compute posterior distributions
    # Gamma - Equations 13.32, 13.64 from [1]
    log_posteriors = log_alphas + log_betas

    # xis Equations 13.43 and 13.65 from [1]
    # Posterior over consecutive states summed across time steps
    log_joint_posterior = compute_xi_log(
        log_alphas,
        log_betas,
        log_conditionals,
        log_normalization,
        is_new_session,
        log_transition_prob,
    )
    return (
        log_posteriors,
        log_joint_posterior,
        log_likelihood,
        likelihood_norm,
        log_alphas,
        log_betas,
    )


@partial(
    jax.jit, static_argnames=["inverse_link_function", "negative_log_likelihood_func"]
)
def hmm_negative_log_likelihood(
    glm_params: Array,
    X: Array,
    y: Array,
    posteriors: Array,
    inverse_link_function: Callable,
    negative_log_likelihood_func: Callable,
):
    """
    Compute the posterior-weighted negative log-likelihood for GLM parameters.

    Computes the expected negative log-likelihood as a function of the GLM
    projection weights, where the expectation is taken over the posterior
    distribution over states.

    This is the objective function minimized during the M-step to update
    GLM parameters.

    Parameters
    ----------
    glm_params:
        Projection coefficients and intercept for the GLM.
    X:
        Design matrix of observations.
    y:
        Target responses.
    posteriors:
        Posterior probabilities over states, shape (n_time_bins, n_states).
    inverse_link_function:
        Function mapping linear predictors to rates.
    negative_log_likelihood_func:
        Function to compute the negative log-likelihood.

    Returns
    -------
    :
        Scalar negative log-likelihood weighted by posteriors:
        sum_t sum_k posterior[t,k] * nll[t,k]
    """
    predicted_rate = compute_rate_per_state(X, glm_params, inverse_link_function)
    nll = negative_log_likelihood_func(
        y,
        predicted_rate,
    )
    if nll.ndim > 2:
        nll = nll.sum(axis=1)  # sum over neurons

    # Compute dot products between log-likelihood terms and gammas
    return jnp.sum(nll * posteriors)


@partial(jax.jit, static_argnames=["m_step_fn_glm_params"])
def run_m_step(
    X: Array,
    y: Array,
    log_posteriors: Array,
    log_joint_posterior: Array,
    glm_params: Tuple[Array, Array],
    is_new_session: Array,
    m_step_fn_glm_params: Callable[[Tuple[Array, Array], Array, Array, Array], Array],
    dirichlet_prior_alphas_init_prob: Array | None = None,
    dirichlet_prior_alphas_transition: Array | None = None,
) -> Tuple[Tuple[Array, Array], Array, Array, Any]:
    r"""
    Perform the M-step of the EM algorithm for GLM-HMM.

    Parameters
    ----------
    X:
        Design matrix of observations, shape (n_samples, n_features).
    y:
        Target responses, shape ``(n_samples,)`` or ``(n_samples, n_neurons)``.
    log_posteriors:
        Log-posterior probabilities over states, shape ``(n_samples, n_states)``.
    log_joint_posterior:
        Log joint posterior probabilities over pairs of states summed over samples. Shape ``(n_states, n_states)``.
        :math:`\sum_t P(z_{t-1}, z_t \mid X, y, \theta_{\text{old}})`.
    glm_params:
        Current GLM coefficients and intercept terms. Coefficients have shape ``(n_features, n_states)`` for
        single observation fits and ``(n_features, n_neurons, n_states)`` for population fits. Intercepts have
        shape ``(n_states,)`` for single observation fits and ``(n_states, n_neurons)`` for population fits.
    is_new_session:
        Boolean mask marking the first observation of each session. Shape ``(n_samples,)``.
    m_step_fn_glm_params:
        Callable that performs the M-step update for GLM parameters (coefficients and intercepts).
        Should have signature: ``f(glm_params, X, y, posteriors) -> (updated_params, state)``.
        The regularizer/prior for the GLM parameters should be configured within this callable.
    dirichlet_prior_alphas_init_prob:
        Prior for the initial states, shape ``(n_states,)``.
    dirichlet_prior_alphas_transition:
        Prior for the transition probabilities, shape ``(n_states, n_states)``.

    Returns
    -------
    optimized_projection_weights:
        Updated projection weights after optimization.
    log_initial_prob:
        Updated initial state distribution in log-space.
    log_transition_prob:
        Updated transition matrix in log-space.
    state:
        State returned by the solver.

    Notes
    -----
    The current implementation requires all Dirichlet prior parameters alpha >= 1.
    Support for sparse priors (0 < alpha < 1) may be added in a future version
    using alternative optimization methods such as proximal gradient descent.
    """
    posteriors = jnp.exp(log_posteriors)
    joint_posterior = jnp.exp(log_joint_posterior)

    # Update Initial state probability Eq. 13.18
    initial_prob = _analytical_m_step_initial_prob(
        posteriors,
        is_new_session=is_new_session,
        dirichlet_prior_alphas=dirichlet_prior_alphas_init_prob,
    )
    transition_prob = _analytical_m_step_transition_prob(
        joint_posterior, dirichlet_prior_alphas=dirichlet_prior_alphas_transition
    )

    # Minimize negative log-likelihood to update GLM weights
    optimized_projection_weights, state = m_step_fn_glm_params(
        glm_params, X, y, posteriors
    )

    return (
        optimized_projection_weights,
        jnp.log(initial_prob),
        jnp.log(transition_prob),
        state,
    )


def prepare_likelihood_func(
    is_population_glm: bool,
    log_likelihood_func: Callable,
    negative_log_likelihood_func: Callable,
) -> Tuple[Callable, Callable]:
    """
    Prepare a likelihood function for use in the EM algorithm.

    Parameters
    ----------
    is_population_glm:
        Bool, true if it is a population GLM likelihood.
    log_likelihood_func:
        Function computing the log-likelihood.
    negative_log_likelihood_func
        Function computing the negative log-likelihood.

    Returns
    -------
    likelihood:
        Likelihood function.
    vmap_nll:
        Vectorized negative log-likelihood function.
    """

    # Wrap likelihood_func to avoid aggregating over samples
    def log_likelihood_per_sample(x, z):
        return log_likelihood_func(x, z, aggregate_sample_scores=lambda s: s)

    def negative_log_likelihood_per_sample(x, z):
        return negative_log_likelihood_func(x, z, aggregate_sample_scores=lambda s: s)

    # Vectorize over the states axis
    state_axes = 2 if is_population_glm else 1
    log_likelihood_per_sample = jax.vmap(
        log_likelihood_per_sample,
        in_axes=(None, state_axes),
        out_axes=state_axes,
    )

    def log_likelihood(y, rate):
        log_like = log_likelihood_per_sample(y, rate)
        if is_population_glm:
            # Multi-neuron case: sum log-likelihoods across neurons
            log_like = log_like.sum(axis=1)
        return log_like

    vmap_nll = jax.vmap(
        negative_log_likelihood_per_sample,
        in_axes=(None, state_axes),
        out_axes=state_axes,
    )
    return log_likelihood, vmap_nll


def _em_step(
    carry: GLMHMMState,
    X: Array,
    y: Array,
    inverse_link_function: Callable,
    likelihood_func: Callable,
    m_step_fn_glm_params: Callable,
    is_new_session: Array,
) -> GLMHMMState:
    """Single EM iteration step."""
    previous_state = carry

    (log_posteriors, log_joint_posterior, _, new_log_like, _, _) = forward_backward(
        X,
        y,
        previous_state.log_initial_prob,
        previous_state.log_transition_matrix,
        previous_state.glm_params,
        inverse_link_function,
        likelihood_func,
        is_new_session,
    )

    glm_params_update, log_init_prob, log_trans_matrix, _ = run_m_step(
        X,
        y,
        log_posteriors=log_posteriors,
        log_joint_posterior=log_joint_posterior,
        glm_params=previous_state.glm_params,
        is_new_session=is_new_session,
        m_step_fn_glm_params=m_step_fn_glm_params,
    )

    new_state = GLMHMMState(
        log_initial_prob=log_init_prob,
        log_transition_matrix=log_trans_matrix,
        glm_params=glm_params_update,
        iterations=previous_state.iterations + 1,
        data_log_likelihood=new_log_like,
        previous_data_log_likelihood=previous_state.data_log_likelihood,
        log_likelihood_history=previous_state.log_likelihood_history.at[
            previous_state.iterations
        ].set(new_log_like),
    )

    return new_state


def check_log_likelihood_increment(state: GLMHMMState, tol: float) -> Array:
    """
    Check EM convergence using absolute tolerance on log-likelihood.

    Parameters
    ----------
    state :
        Current EM state containing likelihood history.
    tol :
        Absolute tolerance threshold.

    Returns
    -------
    : Array
        Boolean indicating convergence.
    """
    delta = jnp.abs(state.data_log_likelihood - state.previous_data_log_likelihood)
    return delta < tol


@partial(
    jax.jit,
    static_argnames=[
        "inverse_link_function",
        "likelihood_func",
        "m_step_fn_glm_params",
        "maxiter",
        "check_convergence",
        "tol",
    ],
)
def em_glm_hmm(
    X: Array,
    y: Array,
    initial_prob: Array,
    transition_prob: Array,
    glm_params: Tuple[Array, Array],
    inverse_link_function: Callable,
    likelihood_func: Callable,
    m_step_fn_glm_params: Callable,
    is_new_session: Optional[Array] = None,
    maxiter: int = 10**3,
    tol: float = 1e-8,
    check_convergence: Callable = check_log_likelihood_increment,
) -> Tuple[Array, Array, Array, Array, Tuple[Array, Array], GLMHMMState]:
    """
    Perform EM optimization for a GLM-HMM.

    Uses equinox while_loop for efficient early stopping when convergence
    criteria are met.

    Parameters
    ----------
    X:
        Design matrix of observations.
    y:
        Target responses.
    initial_prob:
        Initial state distribution.
    transition_prob:
        Initial transition matrix.
    glm_params:
        Initial projection coefficients and intercept for the GLM, shape ``(n_features, n_states)``
        and ``(n_states,)``, respectively.
    inverse_link_function:
        Elementwise function mapping linear predictors to rates.
    likelihood_func:
        Function computing the log-likelihood.
    m_step_fn_glm_params:
        Callable that performs the M-step update for GLM parameters (coefficients and intercepts).
        Should have signature: ``f(glm_params, X, y, posteriors) -> (updated_params, state)``.
        Typically created by configuring a solver with the appropriate regularizer/prior.
    is_new_session:
        Boolean mask for the first observation of each session.
    maxiter:
        Maximum number of EM iterations.
    tol:
        The tolerance for the convergence criterion.
    check_convergence:
        Callable receiving the state and computing the convergence.

    Returns
    -------
    posteriors:
        Posterior probabilities over states for each observation.
    joint_posterior:
        Joint posterior probabilities over pairs of states.
    final_initial_prob:
        Final estimate of the initial state distribution.
    final_transition_prob:
        Final estimate of the transition matrix.
    final_projection_weights:
        Final optimized projection weights.
    final_state:
        Final GLMHMMState containing all parameters and diagnostics.
    """
    is_new_session = initialize_new_session(y.shape[0], is_new_session)

    state = GLMHMMState(
        log_initial_prob=jnp.log(initial_prob),
        log_transition_matrix=jnp.log(transition_prob),
        glm_params=glm_params,
        data_log_likelihood=-jnp.array(jnp.inf),
        previous_data_log_likelihood=-jnp.array(jnp.inf),
        log_likelihood_history=jnp.full(maxiter, jnp.nan),
        iterations=0,
    )

    em_step_fn_while = eqx.Partial(
        lambda *args, **kwargs: _em_step(*args, **kwargs),
        X=X,
        y=y,
        inverse_link_function=inverse_link_function,
        likelihood_func=likelihood_func,
        m_step_fn_glm_params=m_step_fn_glm_params,
        is_new_session=is_new_session,
    )

    def stopping_condition_while(carry):
        new_state = carry
        return ~check_convergence(
            new_state,
            tol,
        )

    state = eqx.internal.while_loop(
        stopping_condition_while, em_step_fn_while, state, max_steps=maxiter, kind="lax"
    )

    # final posterior calculation
    (log_posteriors, log_joint_posterior, _, _, _, _) = forward_backward(
        X,
        y,
        state.log_initial_prob,
        state.log_transition_matrix,
        state.glm_params,
        inverse_link_function,
        likelihood_func,
        is_new_session,
    )

    return (
        jnp.exp(log_posteriors),
        jnp.exp(log_joint_posterior),
        state.log_initial_prob,
        state.log_transition_matrix,
        state.glm_params,
        state,
    )


@partial(
    jax.jit,
    static_argnames=["inverse_link_function", "log_likelihood_func", "return_index"],
)
def max_sum(
    X: Array,
    y: Array,
    initial_prob: Array,
    transition_prob: Array,
    glm_params: Tuple[Array, Array],
    inverse_link_function: Callable,
    log_likelihood_func: Callable[[Array, Array], Array],
    is_new_session: Array | None = None,
    return_index: bool = False,
):
    """
    Find maximum a posteriori (MAP) state path via the max-sum algorithm.

    This function implements the max-sum algorithm for a GLM-HMM, also known as Viterbi algorithm.

    Parameters
    ----------
    X :
        Design matrix, pytree with leaves of shape ``(n_time_bins, n_features)``.

    y :
        Observations, pytree with leaves of shape ``(n_time_bins,)``.

    initial_prob :
        Initial latent state probability, pytree with leaves of shape ``(n_states, 1)``.

    transition_prob :
        Latent state transition matrix, pytree with leaves of shape ``(n_states, n_states)``.
        ``transition_prob[i, j]`` is the probability of transitioning from state ``i`` to state ``j``.

    glm_params :
        Length two tuple with the GLM coefficients of shape ``(n_features, n_states)``
        and intercept of shape ``(n_states,)``.

    inverse_link_function :
        Function mapping linear predictors to the mean of the observation distribution
        (e.g., exp for Poisson, sigmoid for Bernoulli).

    is_new_session :
        Boolean array marking the start of a new session.
        If unspecified or empty, treats the full set of trials as a single session.

    return_index:
        If False, return 1-hot encoded map states, if True, return map state indices.

    Returns
    -------
    map_path:
        The MAP state path.

    """
    is_new_session = initialize_new_session(y.shape[0], is_new_session)
    predicted_rate_given_state = compute_rate_per_state(
        X, glm_params, inverse_link_function
    )
    log_emission = log_likelihood_func(y, predicted_rate_given_state)

    log_transition = jnp.log(transition_prob)
    log_init = jnp.log(initial_prob)
    n_states = initial_prob.shape[0]

    def forward_max_sum(omega_prev, xs):
        log_em, is_new_sess = xs

        def reset_chain(omega_prev, log_em):
            # New session: reset to initial distribution
            omega = log_init + log_em
            max_prob_state = jnp.full(n_states, -1)  # Boundary marker
            return omega, max_prob_state

        def continue_chain(omega_prev, log_em):
            # Continue existing session: Viterbi step
            step = log_em[None, :] + log_transition + omega_prev[:, None]
            max_prob_state = jnp.argmax(step, axis=0)
            omega = step[max_prob_state, jnp.arange(n_states)]
            return omega, max_prob_state

        omega, max_prob_state = jax.lax.cond(
            is_new_sess,
            reset_chain,
            continue_chain,
            omega_prev,
            log_em,
        )

        return omega, (omega, max_prob_state)

    init_omega = log_init + log_emission[0]
    _, (omegas, max_prob_states) = jax.lax.scan(
        forward_max_sum, init_omega, (log_emission[1:], is_new_session[1:])
    )

    # Backward pass
    best_final_state = jnp.argmax(omegas[-1])
    # Prepend initial omega and exclude last one, which is already considered.
    omegas = jnp.concatenate([init_omega[None, :], omegas[:-1]], axis=0)

    def backward_max_sum(current_state_idx, xs):
        max_prob_st, omega_t = xs

        def session_boundary(state_idx, max_prob, omega):
            # Hit a session start, pick best state at this boundary
            return jnp.argmax(omega)

        def continue_backward(state_idx, max_prob, omega):
            # Normal backtracking
            return max_prob[state_idx]

        is_boundary = max_prob_st[current_state_idx] == -1

        prev_state_idx = jax.lax.cond(
            is_boundary,
            session_boundary,
            continue_backward,
            current_state_idx,
            max_prob_st,
            omega_t,
        )

        return prev_state_idx, prev_state_idx

    _, map_path = jax.lax.scan(
        backward_max_sum, best_final_state, (max_prob_states, omegas), reverse=True
    )

    # Append the final state
    map_path = jnp.concatenate([map_path, jnp.array([best_final_state])])

    if not return_index:
        map_path = jax.nn.one_hot(map_path, n_states, dtype=jnp.int32)

    return map_path
