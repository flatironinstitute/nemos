"""Forward backward pass for a HMM."""

from functools import partial
from typing import Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp

from ..typing import Aux, ModelParamsT, SolverState
from .m_step_analytical_updates import (
    _analytical_m_step_log_initial_prob,
    _analytical_m_step_log_transition_prob,
)
from .params import HMMParams
from .utils import Array, initialize_is_new_session


class EMState(eqx.Module):
    """State class for the HMM EM-algorithm."""

    data_log_likelihood: float | Array
    previous_data_log_likelihood: float | Array
    log_likelihood_history: Array
    iterations: int
    converged: bool


EMCarry = Tuple[ModelParamsT, EMState]


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


def _forward_pass(
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


@partial(jax.jit, static_argnames=["log_likelihood_func"])
def forward_pass(
    params: ModelParamsT,
    X: Array,
    y: Array,
    log_likelihood_func: Callable[[Array, Array, Array], Array],
    is_new_session: Array | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute filtering probabilities (forward messages) for an HMM.

    Performs the forward pass of the forward-backward algorithm, computing the
    filtered state probabilities p(z_t | y_1:t) at each time point. These represent
    the probability distribution over states conditioned on observations up to time t.

    This is the public API for computing forward messages, useful for:
    - Online/causal state estimation (filtering)
    - Computing filter_proba in the HMM class
    - One-step-ahead prediction

    Parameters
    ----------
    params :
        Parameter container for an HMM model.
        It must include the attribute `hmm_params`, which is a `ModelParams` subclass with attributes
        `log_initial_prob` and `log_transition_prob`, as well as the attribute `model_params`, also a
        `ModelParams` subclass containing model-dependent parameters used in the log-likelihood function.
    X :
        Design matrix, shape ``(n_time_bins, n_features)``.
    y :
        Observations, shape ``(n_time_bins,)`` or ``(n_time_bins, n_neurons)``.
    log_likelihood_func :
        Function computing observation log-likelihoods per sample, i.e. no aggregation
        should be performed across samples.
    is_new_session :
        Boolean array of shape ``(n_time_bins,)`` marking session starts.
        If None, treats all data as a single continuous session.

    Returns
    -------
    log_alphas :
        Normalized log forward messages, shape ``(n_time_bins, n_states)``.
        Entry ``[t, k]`` is the log filtered probability log p(z_t=k | y_1:t).
        Each row is normalized: ``exp(log_alphas[t]).sum() == 1``.
    log_normalizers :
        Array of shape ``(n_time_bins,)`` containing the log-normalization constants at each
        time step. The sum of these values gives the log-likelihood of the sequence.

    See Also
    --------
    :func:`~nemos.hmm.forward_backward` : Computes both forward and backward messages for smoothing.

    Notes
    -----
    - Forward messages provide causal state estimates (no future information)
    - Smoothing (forward + backward) provides better estimates using all data
    - Log-space computation ensures numerical stability
    - Session boundaries reset the recursion using initial state distribution

    """
    # unpack parameters
    model_params = params.model_params
    log_initial_prob = params.hmm_params.log_initial_prob
    log_transition_prob = params.hmm_params.log_transition_prob

    # Initialize variables
    is_new_session = initialize_is_new_session(y.shape[0], is_new_session)

    # Compute log-likelihoods
    log_conditionals = log_likelihood_func(X, y, model_params)

    # Compute forward pass
    log_alphas, log_normalizers = _forward_pass(
        log_initial_prob, log_transition_prob, log_conditionals, is_new_session
    )  # these are equivalent to the forward pass with python loop
    return log_alphas, log_normalizers


def _backward_pass(
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


@partial(jax.jit, static_argnames=["log_likelihood_func"])
def forward_backward(
    params: ModelParamsT,
    X: Array,
    y: Array,
    log_likelihood_func: Callable[[Array, Array, Array], Array],
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

    params :
        The HMM and additional model parameters.
        It must include the attribute `hmm_params`, which is a `ModelParams` subclass with attributes
        `log_initial_prob` and `log_transition_prob`, as well as the attribute `model_params`, also a
        `ModelParams` subclass containing model-dependent parameters used in the log-likelihood function.

    log_likelihood_func :
        Function computing the elementwise log-likelihood of observations.
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
    # unpack parameters
    model_params = params.model_params
    log_initial_prob = params.hmm_params.log_initial_prob
    log_transition_prob = params.hmm_params.log_transition_prob

    # Initialize variables
    n_time_bins = y.shape[0]
    is_new_session = initialize_is_new_session(y.shape[0], is_new_session)

    # Compute log-likelihoods
    log_conditionals = log_likelihood_func(X, y, model_params)

    # Compute forward pass
    log_alphas, log_normalization = _forward_pass(
        log_initial_prob, log_transition_prob, log_conditionals, is_new_session
    )  # these are equivalent to the forward pass with python loop

    # Compute backward pass
    log_betas = _backward_pass(
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
    jax.jit,
    static_argnames=[
        "m_step_fn_model_params",
    ],
)
def run_m_step(
    params: ModelParamsT,
    X: Array,
    y: Array,
    log_posteriors: Array,
    log_joint_posterior: Array,
    is_new_session: Array,
    m_step_fn_model_params: Callable[
        [ModelParamsT, Array, Array, Array], Tuple[ModelParamsT, SolverState, Aux]
    ],
    dirichlet_prior_alphas_init_prob: Array | None = None,
    dirichlet_prior_alphas_transition: Array | None = None,
) -> Tuple[ModelParamsT, SolverState]:
    r"""
    Perform the M-step of the EM algorithm for an HMM.

    Parameters
    ----------
    params :
        The current model parameters.
        It must include the attribute `hmm_params`, which is a `ModelParams` subclass with attributes
        `log_initial_prob` and `log_transition_prob`, as well as the attribute `model_params`, also a
        `ModelParams` subclass containing model-dependent parameters used in the log-likelihood function.
    X :
        Design matrix of observations, shape (n_samples, n_features).
    y :
        Target responses, shape ``(n_samples,)`` or ``(n_samples, n_neurons)``.
    log_posteriors :
        Log-posterior probabilities over states, shape ``(n_samples, n_states)``.
    log_joint_posterior :
        Log joint posterior probabilities over pairs of states summed over samples. Shape ``(n_states, n_states)``.
        :math:`\sum_t P(z_{t-1}, z_t \mid X, y, \theta_{\text{old}})`.
    is_new_session :
        Boolean mask marking the first observation of each session. Shape ``(n_samples,)``.
    m_step_fn_model_params :
        Callable that performs the M-step update for model parameters (e.g., coefficients and intercepts for a GLM).
        Should have signature: ``f(model_params, X, y, posteriors) -> (updated_params, state, aux)``.
        The regularizer/prior for the parameters should be configured within this callable.
    dirichlet_prior_alphas_init_prob :
        Prior for the initial states, shape ``(n_states,)``.
    dirichlet_prior_alphas_transition :
        Prior for the transition probabilities, shape ``(n_states, n_states)``.

    Returns
    -------
    params :
        The updated model parameters.
    state :
        State returned by the solver.

    Notes
    -----
    The current implementation requires all Dirichlet prior parameters alpha >= 1.
    Support for sparse priors (0 < alpha < 1) may be added in a future version
    using alternative optimization methods such as proximal gradient descent.
    """
    posteriors = jnp.exp(log_posteriors)

    # Update Initial state probability Eq. 13.18
    log_initial_prob = _analytical_m_step_log_initial_prob(
        log_posteriors,
        is_new_session=is_new_session,
        dirichlet_prior_alphas=dirichlet_prior_alphas_init_prob,
    )
    log_transition_prob = _analytical_m_step_log_transition_prob(
        log_joint_posterior, dirichlet_prior_alphas=dirichlet_prior_alphas_transition
    )

    # Minimize negative log-likelihood to update model parameters
    optimized_model_params, state, _ = m_step_fn_model_params(
        params.model_params, X, y, posteriors
    )

    params = params.initialize_params(
        hmm_params=HMMParams(log_initial_prob, log_transition_prob),
        model_params=optimized_model_params,
    )
    return (
        params,
        state,
    )


def _em_step(
    carry: EMCarry,
    X: Array,
    y: Array,
    log_likelihood_func: Callable[[Array, Array, Array], Array],
    m_step_fn_model_params: Callable[
        [ModelParamsT, Array, Array, Array], Tuple[ModelParamsT, SolverState, Aux]
    ],
    is_new_session: Array,
) -> EMCarry:
    """
    Execute a single EM iteration combining E-step and M-step.

    Performs one complete EM cycle: computes posterior distributions over
    latent states (E-step), then updates all model parameters to maximize
    the expected complete-data log-likelihood (M-step).

    Parameters
    ----------
    carry :
        Tuple of current parameters and state:
        ``((log_init_prob, log_trans_matrix, model_params), previous_state)``
    X :
        Design matrix of observations.
    y :
        Target responses.
    log_likelihood_func :
        Log-likelihood function for the E-step.
    m_step_fn_model_params :
        M-step update function for GLM coefficients and intercepts.
    is_new_session :
        Boolean array marking session boundaries.

    Returns
    -------
    carry :
        Updated tuple of parameters and state with new log-likelihood values:
        ``((log_init_prob, log_trans_matrix, model_params), new_state)``

    Notes
    -----
    This function is designed to be called repeatedly by ``eqx.internal.while_loop``
    until convergence criteria are met. The carry structure allows JAX to efficiently
    compile and execute the EM loop.
    """

    params, previous_state = carry

    log_posteriors, log_joint_posterior, _, new_log_like, _, _ = forward_backward(
        params,
        X,
        y,
        log_likelihood_func,
        is_new_session,
    )

    new_params, _ = run_m_step(
        params,
        X,
        y,
        log_posteriors=log_posteriors,
        log_joint_posterior=log_joint_posterior,
        is_new_session=is_new_session,
        m_step_fn_model_params=m_step_fn_model_params,
    )

    new_state = EMState(
        iterations=previous_state.iterations + 1,
        data_log_likelihood=new_log_like,
        previous_data_log_likelihood=previous_state.data_log_likelihood,
        log_likelihood_history=previous_state.log_likelihood_history.at[
            previous_state.iterations
        ].set(new_log_like),
        converged=previous_state.converged,
    )

    return new_params, new_state


def em_step(
    params: ModelParamsT,
    state: EMState,
    X: Array,
    y: Array,
    log_likelihood_func: Callable,
    m_step_fn_model_params: Callable,
    is_new_session: Array,
) -> Tuple[ModelParamsT, EMState]:
    """
    Perform a single EM iteration step for an HMM.

    This function provides a clean public API for running a single EM iteration,
    compatible with the optimization API pattern used by solvers. It wraps the
    internal `_em_step` function which operates on EMCarry tuples.

    Parameters
    ----------
    params :
        Current HMM and model parameters.
        It must include the attribute `hmm_params`, which is a `ModelParams` subclass with attributes
        `log_initial_prob` and `log_transition_prob`, as well as the attribute `model_params`, also a
        `ModelParams` subclass containing model-dependent parameters used in the log-likelihood function.
    state :
        Current EM algorithm state containing iteration count and log-likelihood history.
    X :
        Design matrix of observations.
    y :
        Target responses.
    log_likelihood_func :
        Function computing the log-likelihood or log emissions probability.
    m_step_fn_model_params :
        Callable that performs the M-step update for model parameters.
    is_new_session :
        Boolean mask for the first observation of each session.

    Returns
    -------
    updated_params :
        Updated parameters after one EM iteration.
    updated_state :
        Updated state after one EM iteration.
    """
    # Pack params and state into EMCarry format (log-space for HMM params)
    carry = params, state

    # Run the internal EM step
    params, state = _em_step(
        carry,
        X=X,
        y=y,
        log_likelihood_func=log_likelihood_func,
        m_step_fn_model_params=m_step_fn_model_params,
        is_new_session=is_new_session,
    )

    return params, state


def check_log_likelihood_increment(state: EMState, tol: float) -> Array:
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
        "log_likelihood_func",
        "m_step_fn_model_params",
        "maxiter",
        "check_convergence",
        "tol",
    ],
)
def em_hmm(
    params: ModelParamsT,
    X: Array,
    y: Array,
    log_likelihood_func: Callable,
    m_step_fn_model_params: Callable,
    is_new_session: Optional[Array] = None,
    maxiter: int = 10**3,
    tol: float = 1e-8,
    check_convergence: Callable = check_log_likelihood_increment,
) -> Tuple[ModelParamsT, EMState]:
    """
    Perform EM optimization for a HMM model (i.e. HMM or GLMHMM).

    Uses equinox while_loop for efficient early stopping when convergence
    criteria are met.

    Parameters
    ----------
    params :
        Initial HMM and model parameters. This includes:
        - the HMM initial probabilities, shape ``(n_states,)``.
        - the HMM transition probabilities, shape ``(n_states, n_states)``.
        - any parameters associated with the paired model, e.g. the GLM coef,
          shape ``(n_features, n_states)`` or ``(n_features, n_neurons, n_states)``
          or GLM intercept, shape  ``(n_states, )`` or ``(n_neurons, n_states)`` .
    X :
        Design matrix of observations.
    y :
        Target responses.
    log_likelihood_func :
        Function computing the log-likelihood or log emissions probabilities.
    m_step_fn_model_params :
        Callable that performs the M-step update for the model parameters.
        Should have signature: ``f(model_params, X, y, posteriors) -> (updated_params, state)``.
        Typically created by configuring a solver with the appropriate regularizer/prior.
    is_new_session :
        Boolean mask for the first observation of each session.
    maxiter :
        Maximum number of EM iterations.
    tol :
        The tolerance for the convergence criterion.
    check_convergence :
        Callable receiving the state and computing the convergence.

    Returns
    -------
    params :
        The fitted HMM and model parameters.
    state :
        Final EMState containing all parameters and diagnostics.
    """
    is_new_session = initialize_is_new_session(y.shape[0], is_new_session)

    state = EMState(
        data_log_likelihood=-jnp.array(jnp.inf),
        previous_data_log_likelihood=-jnp.array(jnp.inf),
        log_likelihood_history=jnp.full(maxiter, jnp.nan),
        iterations=0,
        converged=False,
    )

    em_step_fn_while = eqx.Partial(
        lambda *args, **kwargs: _em_step(*args, **kwargs),
        X=X,
        y=y,
        log_likelihood_func=log_likelihood_func,
        m_step_fn_model_params=m_step_fn_model_params,
        is_new_session=is_new_session,
    )

    def stopping_condition_while(carry):
        _, new_state = carry
        return ~check_convergence(new_state, tol)

    init_carry = params, state
    params, state = eqx.internal.while_loop(
        stopping_condition_while,
        em_step_fn_while,
        init_carry,
        max_steps=maxiter,
        kind="lax",
    )
    # update converged parameter
    state = EMState(
        data_log_likelihood=state.data_log_likelihood,
        previous_data_log_likelihood=state.previous_data_log_likelihood,
        iterations=state.iterations,
        log_likelihood_history=state.log_likelihood_history,
        converged=check_convergence(state, tol),
    )
    return params, state


@partial(
    jax.jit,
    static_argnames=["log_likelihood_func", "return_index"],
)
def max_sum(
    params: ModelParamsT,
    X: Array,
    y: Array,
    log_likelihood_func: Callable[[Array, Array, Array], Array],
    is_new_session: Array | None = None,
    return_index: bool = False,
):
    """
    Find maximum a posteriori (MAP) state path via the max-sum algorithm.

    This function implements the max-sum algorithm for an HMM, also known as Viterbi algorithm.

    Parameters
    ----------
    params :
        Current HMM and model parameters.
        It must include the attribute `hmm_params`, which is a `ModelParams` subclass with attributes
        `log_initial_prob` and `log_transition_prob`, as well as the attribute `model_params`, also a
        `ModelParams` subclass containing model-dependent parameters used in the log-likelihood function.

    X :
        Design matrix, pytree with leaves of shape ``(n_time_bins, n_features)``.

    y :
        Observations, pytree with leaves of shape ``(n_time_bins,)``.

    log_likelihood_func :
        Function computing log p(y | model_parameters) for the emissions model.

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
    # unpack parameters
    model_params = params.model_params
    log_transition = params.hmm_params.log_transition_prob
    log_init = params.hmm_params.log_initial_prob

    n_states = log_init.shape[0]

    # initialize new session
    is_new_session = initialize_is_new_session(y.shape[0], is_new_session)

    log_emission = log_likelihood_func(X, y, model_params)

    # Forward pass: similar to forward-backward, scan over all time points
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

    # Initialize with dummy value (won't be used since is_new_session[0] = True)
    init_omega = jnp.full(n_states, -jnp.inf)

    _, (omegas, max_prob_states) = jax.lax.scan(
        forward_max_sum, init_omega, (log_emission, is_new_session)
    )

    # Backward pass
    best_final_state = jnp.argmax(omegas[-1])

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

    # Scan backwards over times 1..T-1 using omegas[0..T-2] and max_prob_states[1..T-1]
    _, map_path = jax.lax.scan(
        backward_max_sum,
        best_final_state,
        (max_prob_states[1:], omegas[:-1]),
        reverse=True,
    )

    # Append the final state
    map_path = jnp.concatenate([map_path, jnp.array([best_final_state])])

    if not return_index:
        map_path = jax.nn.one_hot(map_path, n_states, dtype=jnp.int32)

    return map_path
