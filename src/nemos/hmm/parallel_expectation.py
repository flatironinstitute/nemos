"""Parallel implementation of the forward-backward algorithm."""

from typing import Tuple

import jax
import jax.numpy as jnp

from .utils import Array


def condition(log_w: Array, M: Array) -> Tuple[Array, Array, Array]:
    """
    Stable computation of (log(l), L).

    Conditioning step described in the notes docs/developers_notes/09-associative_estep_hmm.md

    Parameters
    ----------
    log_w:
        Log of either :math:`p(y_t | z_t=i)` (when computing initial elements) or
        :math:`p(y_{v:t} | z_{v-1}=k)` (during the scan). Shape (n_samples, n_states).
    M:
        Either :math:`p(z_t=i | z_{t-1}=j)` (when computing initial elements) or
        :math:`p(z_{v-1}=k | z_{u-1}=j, y_{u:v-1})$` (during the scan). Shape (n_samples, n_states, n_states).

    Returns
    -------
    log_l:
        The log of :math:`p(y_t | z_{t-1}=j)` (when computing initial elements) or
        :math:`p(y_{v:t} | z_{u-1}=j, y_{u:v-1})` (during the scan).
    L:
        Either :math:`p(z_t=i | z_{t-1}=j, y_t)` (when computing initial elements)
        or :math:`p(z_{v-1}=k | z_{u-1}=j, y_{u:t})` (during the scan).
    max_log_w:
        The max of log_w over states.

    Notes
    -----
    The conditioning operation is described in section **Stable Parametrization** in
    the notes docs/developers_notes/09-associative_estep_hmm.md.

    The actual stable computation of the conditioning implemented here is described in
    the **Scan, Step by Step** section, subsection **Conditioning on exit weights**.
    """
    # safe exp
    max_log_w = jnp.max(log_w, axis=-1, keepdims=True)
    exp_cond = jnp.exp(log_w - max_log_w)
    L_unnorm = M * exp_cond[..., None, :]
    lin = L_unnorm.sum(axis=-1)
    log_l = jnp.log(lin) + max_log_w
    L = L_unnorm / lin[..., None]
    return log_l, L, max_log_w


def combine(x1: Tuple[Array, Array], x2: Tuple[Array, Array]) -> Tuple[Array, Array]:
    r"""Combine in the associative scan.

    The combination implements the :math:`\oplus` operator described
    in section **Get (log(l), L) via scan**  of the note
    the notes docs/developers_notes/09-associative_estep_hmm.md.

    The stable implementation is described from section **Combine** to
    section **Dropping the accumulated scale**.
    """
    log_l1, L1 = x1
    log_l2, L2 = x2
    log_r, L_prime, max_log_l2 = condition(log_l2, L1)
    # subtracting max_log_l2 is what drops the accumulated
    # scale keeping log_l O(1)
    log_l = log_l1 + log_r - max_log_l2
    return log_l, L_prime @ L2


def normalizers(
    initial_prob: Array,
    transition_prob: Array,
    log_conditional_prob: Array,
    session_starts: Array,
    filtered_probs: Array,
) -> Array:
    r"""
    Recompute the per-step log normalizers from the filtered probabilities.

    The scan drops the accumulated scale, so ``log c_t`` cannot be read off its log
    output. It is recomputed here from the filtered probabilities as
    :math:`c_t = ((\hat{\alpha}_{t-1} A) \odot b_t) \mathbf{1}` outside a session start and
    :math:`c_t = (\pi \odot b_t) \mathbf{1}` at one, for all ``t`` at once. See section
    **Per-step normalizers** of docs/developers_notes/09-associative_estep_hmm.md.

    Parameters
    ----------
    initial_prob:
        Initial state distribution, shape (n_states,).
    transition_prob:
        Transition matrix, shape (n_states, n_states), indexed ``[from, to]``.
    log_conditional_prob:
        Log-emissions :math:`\log p(y_t | z_t=i)`, shape (n_samples, n_states).
    session_starts:
        Boolean array marking the start of a new session, shape (n_samples,).
    filtered_probs:
        Filtered probabilities :math:`\hat{\alpha}_t`, shape (n_samples, n_states).

    Returns
    -------
    :
        Log normalizers :math:`\log c_t`, shape (n_samples,).
    """
    # alpha_hat_{t-1}; the row at t=0 is discarded by the where below, index 0 being
    # always a session start, and is duplicated only to line the shapes up.
    filtered_prev = jnp.concatenate([filtered_probs[:1], filtered_probs[:-1]])
    predicted_prob = jnp.where(
        session_starts[:, None], initial_prob, filtered_prev @ transition_prob
    )
    # b_t is the one factor that cannot be carried in probability space: it is a sum
    # over neurons in the log, so exp(log_conditional_prob) underflows on its own.
    max_log_b = jnp.max(log_conditional_prob, axis=-1, keepdims=True)
    lin = jnp.sum(predicted_prob * jnp.exp(log_conditional_prob - max_log_b), axis=-1)
    return jnp.log(lin) + max_log_b[:, 0]


def _forward_pass_assoc(
    initial_prob: Array,
    transition_prob: Array,
    log_conditional_prob: Array,
    session_starts: Array,
) -> Tuple[Array, Array]:
    r"""
    Forward pass of an HMM by associative scan.

    The scan composes the one-step elements :math:`\phi(F_{t:t})` of
    docs/developers_notes/09-associative_estep_hmm.md with :func:`combine`, so it runs at
    depth :math:`O(\log T)` instead of the :math:`O(T)` of the sequential recursion. Every
    matrix in the scan is row-stochastic and carried in probability space; the only
    quantities kept in the log are the emissions and the segment scales.

    At a session start the element is built from the initial distribution rather than
    from the transition matrix, which is all the session handling the forward pass
    needs: index 0 is a session start, its element has equal rows, and both the
    matrix product and the row rescaling in :func:`combine` preserve equal rows, so
    every cumulative matrix has all its rows equal to the filtered distribution.

    Parameters
    ----------
    initial_prob :
        Initial state distribution, shape (n_states,).
    transition_prob :
        Transition matrix, shape (n_states, n_states), where entry ``[j, i]`` is
        :math:`p(z_t = i | z_{t-1} = j)`.
    log_conditional_prob :
        Log-emissions :math:`\log p(y_t | z_t=i)`, shape (n_time_bins, n_states).
    session_starts :
        Boolean array of shape (n_time_bins,) marking the start of a new session.

    Returns
    -------
    filtered_probs :
        Filtered probabilities :math:`\hat{\alpha}_t`, shape (n_time_bins, n_states),
        in probability space.
    log_normalizers :
        Log normalizers :math:`\log c_t`, shape (n_time_bins,). Their sum is the
        log-likelihood of the observations.

    References
    ----------
    .. [1] Bishop, C. M. (2006). Pattern recognition and machine learning. Springer.
    """
    n_time_bins, n_states = log_conditional_prob.shape

    # compute the scan elements log_l, L = phi(F_{t:t})
    base = jnp.where(
        session_starts[:, None, None],
        jnp.broadcast_to(initial_prob, (n_time_bins, n_states, n_states)),
        transition_prob,
    )
    log_l, L, _ = condition(log_conditional_prob, base)
    # the log output of the scan is discarded: combine subtracts out the max that
    # would otherwise accumulate, so it is no longer log(p(y_0:t)).
    _, L_cum = jax.lax.associative_scan(combine, (log_l, L))
    filtered_probs = L_cum[:, 0, :]
    return filtered_probs, normalizers(
        initial_prob,
        transition_prob,
        log_conditional_prob,
        session_starts,
        filtered_probs,
    )
