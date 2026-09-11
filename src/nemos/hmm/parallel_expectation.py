"""Parallel implementation of the forward-backward algorithm."""

from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from ..typing import ModelParamsT
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
    L_unnorm = M * exp_cond[..., jnp.newaxis, :]
    lin = L_unnorm.sum(axis=-1)
    log_l = jnp.log(lin) + max_log_w
    L = L_unnorm / lin[..., jnp.newaxis]
    return log_l, L, max_log_w


def combine_forward(
    x1: Tuple[Array, Array], x2: Tuple[Array, Array]
) -> Tuple[Array, Array]:
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
        session_starts[:, jnp.newaxis], initial_prob, filtered_prev @ transition_prob
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
        session_starts[:, jnp.newaxis, jnp.newaxis],
        jnp.broadcast_to(initial_prob, (n_time_bins, n_states, n_states)),
        transition_prob,
    )
    log_l, L, _ = condition(log_conditional_prob, base)
    # the log output of the scan is discarded: combine subtracts out the max that
    # would otherwise accumulate, so it is no longer log(p(y_0:t)).
    _, L_cum = jax.lax.associative_scan(combine_forward, (log_l, L))
    filtered_probs = L_cum[:, 0, :]
    return filtered_probs, normalizers(
        initial_prob,
        transition_prob,
        log_conditional_prob,
        session_starts,
        filtered_probs,
    )


@partial(jax.jit, static_argnames=["log_likelihood_func"])
def forward_pass_assoc(
    params: ModelParamsT,
    X: Array,
    y: Array,
    log_likelihood_func: Callable[[Array, Array, Array], Array],
    session_starts: Array | None = None,
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
    session_starts :
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
    initial_prob = jnp.exp(params.hmm_params.log_initial_prob)
    transition_prob = jnp.exp(params.hmm_params.log_transition_prob)

    # Initialize variables
    session_starts = (
        session_starts
        if session_starts is not None
        else jnp.zeros(y.shape[0], dtype=bool).at[0].set(1)
    )

    # Compute log-likelihoods
    log_conditionals = log_likelihood_func(model_params, X, y)

    # Compute forward pass
    alphas, log_normalizers = _forward_pass_assoc(
        initial_prob, transition_prob, log_conditionals, session_starts
    )  # these are equivalent to the forward pass with python loop
    return jnp.log(alphas), log_normalizers


def _combine_backward(
    x: Tuple[Array, Array], y: Tuple[Array, Array]
) -> Tuple[Array, Array]:
    r"""Compose backward transfer matrices, keeping their scale.

    Nothing is renormalized here, unlike in :func:`combine`: the backward messages are
    not distributions over the states, they carry the :math:`1/c_t` factors so that
    ``log_alphas + log_betas`` is the log posterior, so there is no row-stochastic
    invariant to restore. Nothing needs to be either, the :math:`1/c_t` folded into each
    element being what keeps the products O(1).

    With ``reverse=True`` the element at the current index arrives as ``y`` and the
    accumulation over later times as ``x``, so the product is ordered
    :math:`N_t (N_{t+1} \cdots N_{T-1})`. A reset in ``y`` truncates the block, a session
    start at the current index making the message independent of everything after it, and
    the flag is propagated so that a block containing a reset shadows any later block it
    is composed with.
    """
    N_x, reset_x = x
    N_y, reset_y = y
    composed = jnp.where(reset_y[:, None, None], N_y, N_y @ N_x)
    return composed, reset_x | reset_y


def _backward_pass_assoc(
    transition_prob: Array,
    log_conditional_prob: Array,
    log_normalizers: Array,
    session_starts: Array,
) -> Array:
    r"""
    Backward pass of an HMM by associative scan.

    The recursion that :func:`~nemos.hmm.expectation_maximization._backward_pass`
    evaluates, indexed by the later time :math:`t`, is

    .. math::
        \hat{\beta}_{t-1}[j] = \frac{1}{c_t} \sum_i A[j, i] \, b_t[i] \, \hat{\beta}_t[i],

    with :math:`\hat{\beta}_{t-1} = \mathbf{1}` when :math:`t` starts a session and
    :math:`\hat{\beta}_{T-1} = \mathbf{1}`. Its transfer matrix is therefore
    :math:`N_t = c_t^{-1} A \operatorname{diag}(b_t)`, the one-step element
    :math:`F_{t:t}` of Proposition 1 of
    docs/developers_notes/09-associative_estep_hmm.md divided by the forward normalizer,
    and

    .. math::
        \hat{\beta}_{t-1} = N_t N_{t+1} \cdots N_{T-1} \mathbf{1},

    a suffix product of matrices, which is the scan map of section **Scan Map** of the
    note run in reverse.

    Two things differ from the forward pass, both noted in the closing paragraph of the
    note, which does not derive this pass. The messages are not normalized, so the
    equal-rows argument that makes sessions free in :func:`_forward_pass_assoc` does not
    apply and the elements carry an explicit reset flag. And the scale is kept rather
    than dropped, since a per-element factor would corrupt messages whose absolute scale
    is what makes ``log_alphas + log_betas`` a posterior.

    Parameters
    ----------
    transition_prob :
        Transition matrix, shape (n_states, n_states), where entry ``[j, i]`` is
        :math:`p(z_t = i | z_{t-1} = j)`.
    log_conditional_prob :
        Log-emissions :math:`\log p(y_t | z_t=i)`, shape (n_time_bins, n_states).
    log_normalizers :
        Log normalizers :math:`\log c_t` from the forward pass, shape (n_time_bins,).
    session_starts :
        Boolean array of shape (n_time_bins,) marking the start of a new session.

    Returns
    -------
    :
        Backward messages :math:`\hat{\beta}_t`, shape (n_time_bins, n_states), in
        probability space, normalized as in eqn. 13.62 of [1]_ so that their product
        with the filtered probabilities is the marginal posterior.

    References
    ----------
    .. [1] Bishop, C. M. (2006). Pattern recognition and machine learning. Springer.
    """
    n_states = log_conditional_prob.shape[1]

    # b_t / c_t is the pairing that stays O(1): each is of the order of the per-bin
    # likelihood, which underflows on its own for a population model, while their ratio
    # is a ratio of two averages of the same numbers.
    # The reset element R satisfies R 1 = 1, that is, beta_hat = 1 at a session end.
    elements = jnp.where(
        session_starts[:, None, None],
        1.0 / n_states,
        transition_prob[None]
        * jnp.exp(log_conditional_prob - log_normalizers[:, None])[:, None, :],
    )
    cumulative, _ = jax.lax.associative_scan(
        _combine_backward, (elements, session_starts), reverse=True
    )
    # cumulative[t] maps beta_{T-1} = 1 to beta_{t-1}, so the messages are its row sums.
    # Index 0 is dropped, there being no beta_{-1}, and it cannot pollute the others
    # since cumulative[t] folds only the indices >= t.
    betas = cumulative[1:].sum(axis=2)
    return jnp.concatenate([betas, jnp.ones((1, n_states), betas.dtype)])


@partial(jax.jit, static_argnames=["log_likelihood_func"])
def forward_backward_assoc(
    params: ModelParamsT,
    X: Array,
    y: Array,
    log_likelihood_func: Callable[[Array, Array, Array], Array],
    session_starts: Array | None = None,
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

    session_starts :
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
    initial_prob = jnp.exp(params.hmm_params.log_initial_prob)
    transition_prob = jnp.exp(params.hmm_params.log_transition_prob)

    # Initialize variables
    n_time_bins = y.shape[0]
    session_starts = (
        session_starts
        if session_starts is not None
        else jnp.zeros(y.shape[0], dtype=bool).at[0].set(1)
    )

    # Compute log-likelihoods
    log_conditionals = log_likelihood_func(model_params, X, y)

    # Compute forward pass
    alphas, log_normalization = _forward_pass_assoc(
        initial_prob, transition_prob, log_conditionals, session_starts
    )  # these are equivalent to the forward pass with python loop

    # Compute backward pass
    betas = _backward_pass_assoc(
        transition_prob, log_conditionals, log_normalization, session_starts
    )

    log_likelihood = jnp.sum(
        log_normalization
    )  # Store log-likelihood, log of Equation 13.63

    likelihood_norm = jnp.exp(log_likelihood / n_time_bins)  # Normalize

    # Posteriors
    # ----------
    # Compute posterior distributions
    # Gamma - Equations 13.32, 13.64 from [1]
    log_alphas, log_betas = jnp.log(alphas), jnp.log(betas)
    log_posteriors = log_alphas + log_betas

    # xis Equations 13.43 and 13.65 from [1]
    # Posterior over consecutive states summed across time steps
    # b_t / c_t is the O(1) pairing: alpha_hat / c_t overflows and b_t * beta_hat
    # underflows separately, while their product is the xi probability.
    weights = jnp.exp(log_conditionals[1:] - log_normalization[1:, jnp.newaxis])
    weights = jnp.where(session_starts[1:, jnp.newaxis], 0.0, weights)
    # (n_states, n_time_bins - 1) @ (n_time_bins - 1, n_states), summing over time
    # without materializing the (n_time_bins, n_states, n_states) intermediate.
    sum_xis = alphas[:-1].T @ (betas[1:] * weights)
    log_joint_posterior = jnp.log(sum_xis * transition_prob)
    return (
        log_posteriors,
        log_joint_posterior,
        log_likelihood,
        likelihood_norm,
        log_alphas,
        log_betas,
    )
