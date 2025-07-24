"""Filtering and smoothing step of an HMM."""

from typing import Tuple

import jax
import jax.numpy as jnp

from .typing import Pytree


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
        A tree of arrays of shape `(n_states, n_time_bins)`, representing the observation likelihood
        at each time step for each state.

    new_session :
        A tree of boolean arrays of shape `(n_time_bins,)`, where each element indicates
        whether a new session starts at that time step (in which case `initial_prob` is used
        instead of a transition update).

    Returns
    -------
    alphas :
        A tree of arrays of shape `(n_time_bins, n_states)`, containing the filtered
        probabilities at each time step. Represents the joint probability of observing all of the given data up to time n (first dimension) and the value n_state (second dimension)

    normalizers :
        A tree of arrays of shape `(n_time_bins,)`, representing the normalization constant
        (sum of probabilities) at each time step. Can be used for log-likelihood computation.

    Notes
    -----
    The function assumes all PyTrees (`initial_prob`, `posterior_prob`, etc.) have matching
    structures and compatible shapes. Normalization at each time step is done safely by
    guarding against divide-by-zero errors.

    If this was computed in regular python code, it would look something like:
    ```
     # Initialize variables
    alphas = np.full((n_states, n_time_bins), np.nan)  # forward pass alphas
    c = np.full(n_time_bins, np.nan)  # variable to store marginal likelihood

    for t in range(n_time_bins):
        if new_sess[t]:
            alphas[:, t] = (
                initial_prob * py_z.T[:, t]
            )  # Initial alpha. Equation 13.37. Reinitialize for new sessions
        else:
            alphas[:, t] = py_z.T[:, t] * (
                transition_prob.T @ alphas[:, t - 1]
            )  # Equation 13.36

        c[t] = np.sum(alphas[:, t])  # Store marginal likelihood
        if (
            c[t] == 0
        ):  # This should not happen, but if it does, raise an error if weights are out of control
            raise ValueError(
                f"Zero marginal likelihood at time {t} - Weights may be out of control"
            )
        alphas[:, t] /= c[t]  # Normalize (Equation 13.59)

    ```
    """

    def initial_compute(posterior, _):
        # Equation 13.37. Reinitialize for new sessions
        return jax.tree_util.tree_map(lambda a, b: a * b, posterior, initial_prob)

    def transition_compute(posterior, alpha_previous):
        # Equation 13.36
        exp_transition = jax.tree_util.tree_map(
            jnp.matmul, transition_prob, alpha_previous
        )
        return jax.tree_util.tree_map(lambda a, b: a * b, posterior, exp_transition)

    def body_fn(carry, xs):
        alpha_previous = carry
        posterior, is_new_session = xs
        # if its a new session, run initial_compute
        # else, run transition_compute
        # for both functions, the inputs are posterior and alpha_previous
        alpha = jax.lax.cond(
            is_new_session,
            initial_compute,
            transition_compute,
            posterior,
            alpha_previous,
        )
        const = jnp.sum(alpha) # Store marginal likelihood

        # Safe divide implementation so we don't divide over 0
        const = jnp.where(const > 0, const, 1.0)
        
        alpha = alpha / const  # Normalize - Equation 13.59
        return alpha, (alpha, const)

    init = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x[0]), posterior_prob)
    transition_prob = jax.tree_util.tree_map(lambda x: x.T, transition_prob)
    _, (alphas, normalizers) = jax.lax.scan(
        body_fn, init, (posterior_prob, new_session)
    )
    return alphas, normalizers


def backward_pass(
    transition_prob: Pytree,
    posterior_prob: Pytree,
    normalizers: Pytree,
    new_session: Pytree,
):
    """
    Run the backward pass of the HMM inference algorithm to compute beta messages.

    This function performs a backward recursion (using `jax.lax.scan` in reverse) to compute
    the beta messages for a probabilistic model. It supports PyTree-structured state spaces
    and handles session boundaries by resetting the recursion when a new session starts.

    Parameters
    ----------
    transition_prob :
        A PyTree containing transition matrices of shape `(n_states, n_states)`
        where `T[i, j]` is the probability of transitioning from state `j` to state `i`.
        The matrix is internally transposed to match the recursion logic.

    posterior_prob :
        A PyTree of arrays, each of shape `(n_time_bins, n_states)`, representing the observation
        likelihoods (posterior probabilities) at each time step for each state.

    normalizers :
        A PyTree of arrays, each of shape `(n_time_bins,)`, representing the normalization
        constants from the forward pass (e.g., `alpha.sum()` at each time step). These are used
        to normalize the beta recursion.

    new_session :
        A PyTree of boolean arrays of shape `(n_time_bins,)` indicating the start of new sessions.
        When `new_session[t]` is True, the backward message is reset to a vector of ones at that time step.

    Returns
    -------
    betas :
        A PyTree of arrays, each of shape `(n_time_bins, n_states)`, representing the beta messages
        at each time step. The output is aligned such that `betas[t]` corresponds to the backward
        message at time `t`, matching the forward time indexing used in standard HMM inference.

    Notes
    -----
    This implementation assumes all PyTrees share the same structure and compatible shapes.
    It follows the standard HMM backward equations (e.g., Bishop Eq. 13.38–13.39), including
    reinitialization for segmented sequences.

    If this was regular python code, it would look similar to:
    ```
    betas = np.full((n_states, n_time_bins), np.nan)  # backward pass betas
    betas[:, -1] = np.ones(n_states)  # initial beta (Equation 13.39)

    # Solve for remaining betas
    t0 = perf_counter()
    for t in range(n_time_bins - 2, -1, -1):
        if new_sess[t + 1]:
            betas[:, t] = np.ones(
                n_states
            )  # Reinitialize backward pass if end of session
        else:
            betas[:, t] = transition_prob @ (
                betas[:, t + 1] * py_z.T[:, t + 1]
            )  # Equation 13.38
            betas[:, t] /= c[t + 1]  # Normalize (Equation 13.62)
    ```
    """
    init = jax.tree_util.tree_map(lambda x: jnp.ones_like(x[0]), posterior_prob)

    def initial_compute(posterior, *_):
        # Initialize
        return jax.tree_util.tree_map(lambda x: jnp.ones_like(x), posterior)

    def backward_step(posterior, beta, normalization):
        # Normalize (Equation 13.62)
        return jax.tree_util.tree_map(
            lambda m, x, y, z: jnp.matmul(m, x * y) / z,
            transition_prob,
            posterior,
            beta,
            normalization,
        )

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
