"""Forward backward pass for a GLM-HMM."""

from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from numpy.typing import NDArray

from .third_party.jaxopt.jaxopt import LBFGS
from .tree_utils import pytree_map_and_reduce
from .typing import Pytree

Array = NDArray | jax.numpy.ndarray


def compute_xi(
    alphas, betas, conditionals, normalization, is_new_session, transition_prob
):
    """
    Compute the expected joint posterior (xi) over consecutive latent states.

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
    xi_all : (T, n_states, n_states)
        Expected joint posteriors between time steps.
    """
    # shift alpha so that alpha[t-1] aligns with beta[t]
    norm_alpha = alphas[:-1] / normalization[1:, jnp.newaxis]

    # mask out steps where t is a new session
    norm_alpha = jnp.where(is_new_session[1:, jnp.newaxis], 0.0, norm_alpha)

    # Compute xi sum in one matmul
    xi_sum = norm_alpha.T @ (conditionals[1:] * betas[1:])

    return xi_sum * transition_prob


def forward_pass(
    initial_prob: Pytree,
    transition_prob: Pytree,
    posterior_prob: Pytree,
    new_session: Pytree,
) -> Tuple[Pytree, Pytree]:
    """
    Forward pass of a HMM.

    This function performs a recursive forward pass over time using JAX's `lax.scan`,
    updating filtered probabilities (`alpha`) based on either an initial distribution
    or a transition matrix, depending on whether a new session starts at a given time step.

    Parameters
    ----------
    initial_prob :
        Initial probability distribution for each leaf in the tree. Each leaf is typically
        a 1D array of shape `(n_states,)`.

    transition_prob :
        Transition matrix or tree of transition matrices, where each entry `T[i, j]`
        represents the probability of transitioning from state `j` to state `i`.
        The transition matrix should be compatible with `jnp.matmul` applied to the `alpha` vector.

    posterior_prob :
        A tree of arrays of shape `(n_time_bins, n_states)`, representing the observation likelihood
        at each time step for each state.

    new_session :
        A tree of boolean arrays of shape `(n_time_bins,)`, where each element indicates
        whether a new session starts at that time step (in which case `initial_prob` is used
        instead of a transition update).

    Returns
    -------
    alphas :
        A tree of arrays of shape `(n_time_bins, n_states)`, containing the filtered
        probabilities at each time step. Represents the joint probability of observing all
        of the given data up to time n (first dimension) and the value n_state (second dimension).

    normalizers :
        A tree of arrays of shape `(n_time_bins,)`, representing the normalization constant
        (sum of probabilities) at each time step. Can be used for log-likelihood computation.

    Notes
    -----
    The function assumes all PyTrees (`initial_prob`, `posterior_prob`, etc.) have matching
    structures and compatible shapes. Normalization at each time step is done safely by
    guarding against divide-by-zero errors.

    If this was computed in regular python code, it would look something like:

    .. code-block:: python

        alphas = np.full((n_states, n_time_bins), np.nan)
        c = np.full(n_time_bins, np.nan)

        for t in range(n_time_bins):
            if new_sess[t]:
                alphas[:, t] = (
                    initial_prob * py_z.T[:, t]
                )
            else:
                alphas[:, t] = py_z.T[:, t] * (
                    transition_prob.T @ alphas[:, t - 1]
                )

            c[t] = np.sum(alphas[:, t])  # Store marginal likelihood
            alphas[:, t] /= c[t]
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

    init = jnp.zeros_like(posterior_prob[0])
    transition_prob = transition_prob.T
    _, (alphas, normalizers) = jax.lax.scan(
        body_fn, init, (posterior_prob, new_session)
    )
    return alphas, normalizers


def backward_pass(
    transition_prob: Array,
    posterior_prob: Array,
    normalizers: Array,
    new_session: Array,
):
    """
    Run the backward pass of the HMM inference algorithm to compute beta messages.

    This function performs the backward recursion step of the forward–backward algorithm,
    using ``jax.lax.scan`` in reverse to compute beta messages at each time step. It handles
    session boundaries by resetting the beta messages when a new session starts.

    Parameters
    ----------
    transition_prob :
        Transition matrix of shape ``(n_states, n_states)``, where entry ``T[i, j]`` is the
        probability of transitioning from state ``j`` to state ``i``.

    posterior_prob :
        Array of shape ``(n_time_bins, n_states)``, representing the observation likelihoods
        ``p(y_t | z_t)`` at each time step for each state.

    normalizers :
        Array of shape ``(n_time_bins,)`` containing the normalization constants from the forward
        pass (e.g., sums of alpha messages). These are used to normalize the backward recursion.

    new_session :
        Boolean array of shape ``(n_time_bins,)`` indicating the start of new sessions. When
        ``new_session[t]`` is True, the backward message at time ``t`` is reset to a vector of ones.

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
    """
    init = jnp.ones_like(posterior_prob[0])

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
        body_fn, init, (posterior_prob, normalizers, new_session), reverse=True
    )
    return betas


def forward_backward(
    X: Array,
    y: Array,
    initial_prob: Array,
    transition_prob: Array,
    projection_weights: Array,
    inverse_link_function: Callable,
    likelihood_func: Callable[[Array, Array], Array],
    is_new_session: Array | None = None,
):
    """
    Run the forward-backward Baum-Welch algorithm.

    Run the forward-backward Baum-Welch algorithm [1]_ that compute a posterior distribution over latent
    states.

    Parameters
    ----------
    X :
        Design matrix, pytree with leaves of shape ``(n_time_bins, n_features)``.

    y :
        Observations, pytree with leaves of shape ``(n_time_bins,)``.

    initial_prob :
        Initial latent state probability, pytree with leaves of shape ``(``n_states, 1)``.

    transition_prob :
        Latent state transition matrix, pytree with leaves of shape ``(n_states x n_states)``.

    projection_weights :
        Latent state GLM weights, pytree with leaves of shape ``(n_features, n_states)``.

    inverse_link_function :
        Function mapping linear predictors to the mean of the observation distribution
        (e.g., ``jnp.exp`` for Poisson, sigmoid for Bernoulli).

    likelihood_func :
        Function computing the elementwise likelihood of observations given predicted mean values.
        Must return an array of shape ``(n_time_bins, n_states)``.

    is_new_session :
        Boolean array marking the start of a new session.
        If unspecified or empty, treats the full set of trials as a single session.

    Returns
    -------
    gammas :
        Marginal posterior distribution over latent states, shape ``(n_states, n_time_bins)``.

    xis :
        Joint posterior distribution between consecutive time steps, shape ``(n_states, n_states, n_time_bins)``.

    log_likelihood :
        Total log-likelihood of the observation sequence under the model.

    log_likelihood_norm :
        A vmapped function that computes the elementwise log-likelihood between observed
        and predicted values. Must return an array of shape ``(n_time_bins, n_states)``.
        The vmapping over states must be performed by the caller, outside this function,
        using `jax.vmap` or equivalent, so that the passed function is already fully
        vectorized over the state dimension.

    alphas :
        Forward messages (alpha values), shape ``(n_states, n_time_bins)``.

    betas :
        Backward messages (beta values), shape ``(n_states, n_time_bins)``.

    References
    ----------
    .. [1] Bishop, C. M. (2006). *Pattern recognition and machine learning*. Springer.
    """
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
    predicted_rate_given_state = inverse_link_function(X @ projection_weights)

    # Compute likelihood given the fixed weights
    # Data likelihood p(y|z) from emissions model
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

    log_likelihood_norm = jnp.exp(
        log_likelihood / n_time_bins
    )  # Normalize - where did this come from?

    # Posteriors
    # ----------
    # Compute posterior distributions
    # Gamma - Equations 13.32, 13.64 from [1]
    gammas = alphas * betas

    # Equations 13.43 and 13.65 from [1]
    # Xi summed across time steps
    xis = compute_xi(
        alphas,
        betas,
        conditionals,
        normalization,
        is_new_session,
        transition_prob,
    )
    return gammas, xis, log_likelihood, log_likelihood_norm, alphas, betas


def func_to_minimize(
    projection_weights,
    n_states,
    y,
    X,
    gammas,
    inverse_link_function,
    log_likelihood_func,
):
    """Minimize expected log-likelihood."""
    # Reshape flat weights into tree of (n_features, n_states)
    projection_weights = jax.tree_util.tree_map(
        lambda w: w.reshape(-1, n_states), projection_weights
    )

    # Predict mean from each feature block
    tmpy = jax.tree_util.tree_map(
        lambda x, w: inverse_link_function(x @ w), X, projection_weights
    )

    # Compute dot products between log-likelihood terms and gammas
    def tree_dot(a):
        return pytree_map_and_reduce(lambda x, y: jnp.sum(x * y), sum, a, gammas)

    # TODO:  probably vmap outside the call so that this function works
    log_likelihood_func = partial(log_likelihood_func, aggregate_sample_scores=tree_dot)

    nll = log_likelihood_func(
        y,
        tmpy,
    )

    return nll


def run_m_step(
    y: Array,
    X: Array,
    gammas: Array,
    projection_weights: Array,
    inverse_link_function,
    log_likelihood_func,
    solver_kwargs: dict | None = None,
):
    """Run M-step."""
    if solver_kwargs is None:
        solver_kwargs = {}

    n_states = projection_weights.shape[1]

    objective = partial(
        func_to_minimize,
        n_states=n_states,
        y=y,
        X=X,
        gammas=gammas,
        inverse_link_function=inverse_link_function,
        log_likelihood_func=log_likelihood_func,
    )

    # Minimize negative log-likelihood to update GLM weights
    solver = LBFGS(objective, **solver_kwargs)
    opt_param, state = solver.run(projection_weights)

    return opt_param, state
