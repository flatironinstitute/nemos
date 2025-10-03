"""Forward backward pass for a GLM-HMM."""

from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from numpy.typing import NDArray

Array = NDArray | jax.numpy.ndarray


def compute_xi(
    alphas, betas, conditionals, normalization, is_new_session, transition_prob
):
    """
    Compute the sum of the joint posterior (xi) over consecutive latent states.

    Compute the sum of the joint posterior (xis, eqn. 13.14 of [1]_) over samples, implementing
    the summation required in the eqn. 13.19 of [1]_.

    Parameters
    ----------
    alphas :
        Forward messages, shape ``(n_time_bins, n_states)``
    betas :
        Backward messages, shape ``(n_time_bins, n_states)``.
    conditionals :
        Observation likelihoods p(y_t | z_t), shape ``(n_time_bins, n_states)``.
    normalization :
        Normalization constants from forward pass, shape ``(n_time_bins,)``.
    is_new_session :
        Boolean array, True at start of new sessions, shape ``(n_time_bins,)``.
    transition_prob :
        Transition probability matrix, shape ``(n_states, n_states)``.

    Returns
    -------
    :
        Expected joint posteriors between time steps, shape ``(n_states, n_states)``.

    References
    ----------
    .. [1] Bishop, C. M. (2006). Pattern recognition and machine learning. Springer.

    """
    # shift alpha so that alpha[t-1] aligns with beta[t]
    norm_alpha = alphas[:-1] / normalization[1:, jnp.newaxis]

    # mask out steps where t is a new session
    norm_alpha = jnp.where(is_new_session[1:, jnp.newaxis], 0.0, norm_alpha)

    # Compute xi sum in one matmul
    xi_sum = norm_alpha.T @ (conditionals[1:] * betas[1:])

    return xi_sum * transition_prob


def forward_pass(
    initial_prob: Array,
    transition_prob: Array,
    conditional_prob: Array,
    is_new_session: Array,
) -> Tuple[Array, Array]:
    """
    Forward pass of an HMM.

    This function performs the recursive forward pass over time using ``jax.lax.scan``,
    computing the filtered probabilities (``alpha``, eqn. 13.34 and 13.36 of [1]_) at each time step.
    At the start of a new session, the recursion is reset using the initial state distribution.

    Parameters
    ----------
    initial_prob :
        Initial state probability distribution, array of shape ``(n_states,)``.

    transition_prob :
        Transition matrix of shape ``(n_states, n_states)``, where entry ``T[i, j]`` is the
        probability of transitioning from state ``i`` to state ``j``.

    conditional_prob :
        Array of shape ``(n_time_bins, n_states)``, representing the observation likelihood
        ``p(y_t | z_t)`` at each time step for each state.

    is_new_session :
        Boolean array of shape ``(n_time_bins,)`` indicating the start of new sessions. When
        ``is_new_session[t]`` is True, the recursion at time ``t`` is reset using ``initial_prob``.

    Returns
    -------
    alphas :
        Array of shape ``(n_time_bins, n_states)``, containing the filtered probabilities
        at each time step. ``alphas[t]`` corresponds to the forward message at time ``t``.

    normalizers :
        Array of shape ``(n_time_bins,)`` containing the normalization constants at each
        time step. These values can be used to compute the log-likelihood of the sequence.

    Notes
    -----
    Normalization is performed at each time step to avoid numerical underflow, guarding
    against divide-by-zero errors.

    Equivalent pseudocode in standard Python:

    .. code-block:: python

        alphas = np.full((n_states, n_time_bins), np.nan)
        c = np.full(n_time_bins, np.nan)

        for t in range(n_time_bins):
            if new_sess[t]:
                alphas[:, t] = initial_prob * py_z.T[:, t]
            else:
                alphas[:, t] = py_z.T[:, t] * (transition_prob.T @ alphas[:, t - 1])

            c[t] = np.sum(alphas[:, t])
            alphas[:, t] /= c[t]

    References
    ----------
    .. [1] Bishop, C. M. (2006). Pattern recognition and machine learning. Springer.

    """

    def initial_compute(posterior, _):
        # Equation 13.37. Reinitialize for new sessions
        return posterior * initial_prob

    def transition_compute(posterior, alpha_previous):
        # Equation 13.36
        exp_transition = jnp.matmul(transition_prob, alpha_previous)
        return posterior * exp_transition

    def body_fn(carry, xs):
        alpha_previous = carry
        posterior, is_new_session = xs
        # if it is a new session, run initial_compute
        # else, run transition_compute
        # for both functions, the inputs are posterior and alpha_previous
        alpha = jax.lax.cond(
            is_new_session,
            initial_compute,
            transition_compute,
            posterior,
            alpha_previous,
        )
        const = jnp.sum(alpha)  # Store marginal likelihood

        # Safe divide implementation so we don't divide over 0
        const = jnp.where(const > 0, const, 1.0)

        alpha = alpha / const  # Normalize - Equation 13.59
        return alpha, (alpha, const)

    init = jnp.zeros_like(conditional_prob[0])
    transition_prob = transition_prob.T
    _, (alphas, normalizers) = jax.lax.scan(
        body_fn, init, (conditional_prob, is_new_session)
    )
    return alphas, normalizers


def backward_pass(
    transition_prob: Array,
    conditional_prob: Array,
    normalizers: Array,
    is_new_session: Array,
):
    """
    Run the backward pass of the HMM inference algorithm to compute beta messages.

    This function performs the backward recursion step of the forward–backward algorithm,
    using ``jax.lax.scan`` in reverse to compute beta messages at each time step, computing
    the ``beta`` parameters, see eqn. 13.35 and 13.38 of [1]_.
    It handles session boundaries by resetting the beta messages when a new session starts.

    Parameters
    ----------
    transition_prob :
        Transition matrix of shape ``(n_states, n_states)``, where entry ``T[i, j]`` is the
        probability of transitioning from state ``i`` to state ``j``.

    conditional_prob :
        Array of shape ``(n_time_bins, n_states)``, representing the observation likelihoods
        ``p(y_t | z_t)`` at each time step for each state.

    normalizers :
        Array of shape ``(n_time_bins,)`` containing the normalization constants from the forward
        pass (e.g., sums of alpha messages). These are used to normalize the backward recursion.

    is_new_session :
        Boolean array of shape ``(n_time_bins,)`` indicating the start of new sessions. When
        ``is_new_session[t]`` is True, the backward message at time ``t`` is reset to a vector of ones.

    Returns
    -------
    betas :
        Array of shape ``(n_time_bins, n_states)``, containing the beta messages at each time step.
        The indexing is aligned with the forward pass, such that ``betas[t]`` corresponds to the
        backward message at time ``t``.

    Notes
    -----
    This implementation follows the standard HMM backward equations (Bishop, 2006, Eq. 13.38–13.39),
    including reinitialization for segmented sequences.

    Equivalent pseudocode in standard Python:

    .. code-block:: python

        betas = np.full((n_states, n_time_bins), np.nan)
        betas[:, -1] = np.ones(n_states)

        for t in range(n_time_bins - 2, -1, -1):
            if new_sess[t + 1]:
                betas[:, t] = np.ones(
                    n_states
                )
            else:
                betas[:, t] = transition_prob @ (
                    betas[:, t + 1] * py_z.T[:, t + 1]
                )
                betas[:, t] /= c[t + 1]

    References
    ----------
    .. [1] Bishop, C. M. (2006). Pattern recognition and machine learning. Springer.

    """
    init = jnp.ones_like(conditional_prob[0])

    def initial_compute(posterior, *_):
        # Initialize
        return jnp.ones_like(posterior)

    def backward_step(posterior, beta, normalization):
        # Normalize (Equation 13.62)
        return jnp.matmul(transition_prob, posterior * beta) / normalization

    def body_fn(carry, xs):
        posterior, norm, is_new_sess = xs
        beta = jax.lax.cond(
            is_new_sess,
            initial_compute,
            backward_step,
            posterior,
            carry,
            norm,
        )
        return beta, carry

    # Keeping the carrys because I am interested in
    # all outputs, including the last one.
    _, betas = jax.lax.scan(
        body_fn, init, (conditional_prob, normalizers, is_new_session), reverse=True
    )
    return betas


@partial(jax.jit, static_argnames=["inverse_link_function", "likelihood_func"])
def forward_backward(
    X: Array,
    y: Array,
    initial_prob: Array,
    transition_prob: Array,
    glm_params: Tuple[Array, Array],
    inverse_link_function: Callable,
    likelihood_func: Callable[[Array, Array], Array],
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

    likelihood_func :
        Function computing the elementwise likelihood of observations given predicted mean values.
        Must return an array of shape ``(n_time_bins, n_states)``.

    is_new_session :
        Boolean array marking the start of a new session.
        If unspecified or empty, treats the full set of trials as a single session.

    Returns
    -------
    posteriors :
        Marginal posterior distribution over latent states, shape ``(n_time_bins, n_states)``.

    joint_posterior :
        Joint posterior distribution between consecutive time steps summed
        over samples, shape ``(n_states, n_states)``.

    log_likelihood :
        Total log-likelihood of the observation sequence under the model.

    log_likelihood_norm :
        The normalized total likelihood.

    alphas :
        Forward messages (alpha values), shape ``(n_time_bins, n_states)``.

    betas :
        Backward messages (beta values), shape ``(n_time_bins, n_states)``.

    References
    ----------
    .. [1] Bishop, C. M. (2006). *Pattern recognition and machine learning*. Springer.
    """
    coef, intercept = glm_params
    # Initialize variables
    n_time_bins = X.shape[0]

    # Revise if the data is one single session or multiple sessions.
    # If new_sess is not provided, assume one session
    if is_new_session is None:
        # default: all False, but first time bin must be True
        is_new_session = jax.lax.dynamic_update_index_in_dim(
            jnp.zeros(y.shape[0], dtype=bool), True, 0, axis=0
        )
    else:
        # use the user-provided tree, but force the first time bin to be True
        is_new_session = jax.lax.dynamic_update_index_in_dim(
            jnp.asarray(is_new_session, dtype=bool), True, 0, axis=0
        )

    # Predicted y
    if coef.ndim > 2:
        predicted_rate_given_state = inverse_link_function(
            jnp.einsum("ik, kjw->ijw", X, coef) + intercept
        )
    else:
        predicted_rate_given_state = inverse_link_function(X @ coef + intercept)

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

    conditionals = likelihood_func(y, predicted_rate_given_state)

    # Compute forward pass
    alphas, normalization = forward_pass(
        initial_prob, transition_prob, conditionals, is_new_session
    )  # these are equivalent to the forward pass with python loop

    # Compute backward pass
    betas = backward_pass(transition_prob, conditionals, normalization, is_new_session)

    log_likelihood = jnp.sum(
        jnp.log(normalization)
    )  # Store log-likelihood, log of Equation 13.63

    likelihood_norm = jnp.exp(log_likelihood / n_time_bins)  # Normalize

    # Posteriors
    # ----------
    # Compute posterior distributions
    # Gamma - Equations 13.32, 13.64 from [1]
    posteriors = alphas * betas

    # xis Equations 13.43 and 13.65 from [1]
    # Posterior over consecutive states summed across time steps
    joint_posterior = compute_xi(
        alphas,
        betas,
        conditionals,
        normalization,
        is_new_session,
        transition_prob,
    )
    return (
        posteriors,
        joint_posterior,
        log_likelihood,
        likelihood_norm,
        alphas,
        betas,
    )
