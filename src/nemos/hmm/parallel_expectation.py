r"""Parallel implementation of the forward-backward algorithm.

The recursions of :mod:`nemos.hmm.expectation_maximization` are prefix problems over a
sequence of per-step transfer matrices, so they can be evaluated by
``jax.lax.associative_scan`` at depth :math:`O(\log T)` rather than by a scan of length
:math:`T`. The derivation is in docs/developers_notes/09-associative_estep_hmm.md;
:func:`forward_backward_assoc` returns the same 6-tuple as
:func:`~nemos.hmm.expectation_maximization.forward_backward` and is interchangeable
with it.

Conventions
-----------
**entry and exit states.** Every matrix here is indexed ``[entry, exit]``: the row is
the state the segment is entered in, the column the state it is left in. This follows
``log_transition_prob[j, i] = log p(z_t = i | z_{t-1} = j)``, the one-step case. A
weight carried per column is therefore a weight per *exit* state, which is what
``log_exit_weights`` means in :func:`_condition_on`.

**earlier and later.** Composing two segments is a matrix product: the earlier segment is
always the left factor, ``log_matmul(earlier,later)``, with the shared boundary state contracted away.
The scan's operands are named for their position in time rather than in the argument list, because the two do not
always agree: ``associative_scan`` passes ``(earlier, later)`` going forwards, but with
``reverse=True`` it passes them in reverse time, so :func:`_combine_backward` receives
``(later, earlier)`` and still puts the earlier segment on the left of the product.

**the two halves of a forward element.** A forward element is a pair
``(log_l, log_L)``: a scale, which grows with the length of the segment it spans, and a
row-stochastic matrix whose rows ``logsumexp`` to zero. They are kept apart on purpose.
Carrying them in one number is the obvious alternative and it is wrong -- the O(1)
message would then have to be recovered by cancelling numbers of the magnitude of
``log p(y)``, an error growing linearly in ``T``. :func:`combine_forward` drops the
scale at every step, which is why ``log c_t`` cannot be read off the scan and is
recomputed by :func:`_compute_log_normalizers`.

**the backward has no such split.** Its messages carry the ``1 / c_t`` factors so that
``log_alphas + log_betas`` is the log posterior, so their absolute scale is meaningful
and cannot be dropped. It does not need to be: the ``1 / c_t`` folded into each element
keeps the products O(1). The cost is that sessions need an explicit reset flag there,
whereas the forward gets them for free because the rows of a reset element are constant.
"""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from ..typing import EStepOutput, LogLikelihoodFn, ModelParamsT
from .expectation_maximization import compute_xi_log
from .utils import Array


def log_matmul(log_earlier: Array, log_later: Array) -> Array:
    r"""Log-semiring product of two segments, earlier on the left.

    :math:`\text{out}[i,j] = \text{logsumexp}_k(\text{earlier}[i,k] + \text{later}[k,j])`,
    so ``i`` indexes the entry state of the earlier segment and ``j`` the exit state
    of the later one, with the shared boundary state ``k`` contracted away.

    Shifted per row of ``log_earlier`` and per column of ``log_later``, then run as a GEMM. The
    broadcast form would materialize a ``(..., K, K, K)`` intermediate, which at
    ``T = 1e6``, ``K = 20`` is 8e9 elements; exponentiating and calling into GEMM is
    ``O(T K^2)`` memory instead. The shifts make the result exact except for entries
    more than ~700 decades (f64) below their row's or column's maximum.
    """
    row_max = jnp.max(log_earlier, axis=-1, keepdims=True)
    col_max = jnp.max(log_later, axis=-2, keepdims=True)
    prod = jnp.exp(log_earlier - row_max) @ jnp.exp(log_later - col_max)
    return jnp.log(prod) + row_max + col_max


def _condition_on(
    log_exit_weights: Array, log_row_stochastic: Array
) -> Tuple[Array, Array]:
    """
    Stable computation of (log(l), log(L)).

    Conditioning step described in the notes docs/developers_notes/09-associative_estep_hmm.md

    Parameters
    ----------
    log_exit_weights:
        Log of either :math:`p(y_t | z_t=i)` (when computing initial elements) or
        :math:`p(y_{v:t} | z_{v-1}=k)` (during the scan). Shape (n_samples, n_states).
    log_row_stochastic:
        Log of either :math:`p(z_t=i | z_{t-1}=j)` (when computing initial elements) or
        :math:`p(z_{v-1}=k | z_{u-1}=j, y_{u:v-1})` (during the scan). Shape (n_samples, n_states, n_states).

    Returns
    -------
    log_l:
        The log of :math:`p(y_t | z_{t-1}=j)` (when computing initial elements) or
        :math:`p(y_{v:t} | z_{u-1}=j, y_{u:v-1})` (during the scan).
    log_L:
        Log of either :math:`p(z_t=i | z_{t-1}=j, y_t)` (when computing initial elements)
        or :math:`p(z_{v-1}=k | z_{u-1}=j, y_{u:t})` (during the scan). Its rows
        logsumexp to zero.

    Notes
    -----
    The conditioning operation is described in section **Stable Parametrization** in
    the notes docs/developers_notes/09-associative_estep_hmm.md.

    The log-space form implemented here is described in the section
    **The Same Scan in Log Space**. Against the probability-space form of the section
    above it, the explicit max shift is absorbed into the ``logsumexp`` that replaces
    the row sum, and the row normalization becomes a subtraction. Keeping ``log_L`` in
    the log rather than ``L`` in probability space is what removes the floor under an
    individual message, which a population model reaches in single precision.
    """
    log_L_unnorm = log_row_stochastic + log_exit_weights[..., jnp.newaxis, :]
    log_l = jax.scipy.special.logsumexp(log_L_unnorm, axis=-1)
    return log_l, log_L_unnorm - log_l[..., jnp.newaxis]


def combine_forward(
    earlier: Tuple[Array, Array], later: Tuple[Array, Array]
) -> Tuple[Array, Array]:
    r"""Combine in the associative scan.

    The combination implements the :math:`\oplus` operator described
    in section **Get (log(l), L) via scan** of the note
    the notes docs/developers_notes/09-associative_estep_hmm.md.

    The stable implementation is described from section **Combine** to
    section **Dropping the accumulated scale**.
    """
    log_l1, log_L1 = earlier
    log_l2, log_L2 = later
    log_r, log_L_prime = _condition_on(log_l2, log_L1)
    log_l = log_l1 + log_r
    # subtracting the max is what drops the accumulated scale, keeping log_l O(1).
    # Without it log_l reaches the magnitude of log p(y_0:t) and the O(1) message is
    # recovered by cancelling numbers that large, an error growing linearly with T.
    log_l = log_l - jnp.max(log_l, axis=-1, keepdims=True)
    return log_l, log_matmul(log_L_prime, log_L2)


def _compute_log_normalizers(
    log_initial_prob: Array,
    log_transition_prob: Array,
    log_conditional_prob: Array,
    session_starts: Array,
    log_alphas: Array,
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
    log_initial_prob:
        Log initial state distribution, shape (n_states,).
    log_transition_prob:
        Log transition matrix, shape (n_states, n_states), indexed ``[from, to]``.
    log_conditional_prob:
        Log-emissions :math:`\log p(y_t | z_t=i)`, shape (n_samples, n_states).
    session_starts:
        Boolean array marking the start of a new session, shape (n_samples,).
    log_alphas:
        Log filtered probabilities :math:`\log \hat{\alpha}_t`, shape (n_samples, n_states).

    Returns
    -------
    :
        Log normalizers :math:`\log c_t`, shape (n_samples,).
    """
    # log alpha_hat_{t-1}; the row at t=0 is discarded by the where below, index 0
    # being always a session start, and is duplicated only to line the shapes up.
    log_alphas_prev = jnp.concatenate([log_alphas[:1], log_alphas[:-1]])
    log_transitioned = jax.scipy.special.logsumexp(
        log_alphas_prev[..., :, jnp.newaxis] + log_transition_prob[jnp.newaxis],
        axis=-2,
    )
    log_predicted = jnp.where(
        session_starts[:, jnp.newaxis], log_initial_prob, log_transitioned
    )
    return jax.scipy.special.logsumexp(log_conditional_prob + log_predicted, axis=-1)


def _forward_pass_assoc(
    log_initial_prob: Array,
    log_transition_prob: Array,
    log_conditional_prob: Array,
    session_starts: Array,
) -> Tuple[Array, Array]:
    r"""
    Forward pass of an HMM by associative scan.

    The scan composes the one-step elements :math:`\phi(F_{t:t})` of
    docs/developers_notes/09-associative_estep_hmm.md with :func:`combine_forward`, so it
    runs at depth :math:`O(\log T)` instead of the :math:`O(T)` of the sequential
    recursion. Every matrix in the scan is row-stochastic and carried in the log, its
    rows summing to zero under ``logsumexp``, so no individual message has a floor
    under it; the scale rides alongside as ``log_l`` and is dropped at every combine.

    At a session start the element is built from the initial distribution rather than
    from the transition matrix, which is all the session handling the forward pass
    needs: index 0 is a session start, its element has equal rows, and both the
    matrix product and the row rescaling in :func:`combine_forward` preserve equal rows, so
    every cumulative matrix has all its rows equal to the filtered distribution.

    Parameters
    ----------
    log_initial_prob :
        Log initial state distribution, shape (n_states,).
    log_transition_prob :
        Log transition matrix, shape (n_states, n_states), where entry ``[j, i]`` is
        :math:`\log p(z_t = i | z_{t-1} = j)`.
    log_conditional_prob :
        Log-emissions :math:`\log p(y_t | z_t=i)`, shape (n_time_bins, n_states).
    session_starts :
        Boolean array of shape (n_time_bins,) marking the start of a new session.

    Returns
    -------
    log_alphas :
        Log filtered probabilities :math:`\log \hat{\alpha}_t`, shape
        (n_time_bins, n_states), each row summing to zero under ``logsumexp``.
    log_normalizers :
        Log normalizers :math:`\log c_t`, shape (n_time_bins,). Their sum is the
        log-likelihood of the observations.

    References
    ----------
    .. [1] Bishop, C. M. (2006). Pattern recognition and machine learning. Springer.
    """
    n_time_bins, n_states = log_conditional_prob.shape

    # compute the scan elements log_l, log_L = phi(F_{t:t})
    log_base = jnp.where(
        session_starts[:, jnp.newaxis, jnp.newaxis],
        jnp.broadcast_to(log_initial_prob, (n_time_bins, n_states, n_states)),
        log_transition_prob,
    )
    elements = _condition_on(log_conditional_prob, log_base)
    # the log output of the scan is discarded: combine subtracts out the max that
    # would otherwise accumulate, so it is no longer log(p(y_0:t)).
    _, log_L_cum = jax.lax.associative_scan(combine_forward, elements)
    log_alphas = log_L_cum[:, 0, :]
    return log_alphas, _compute_log_normalizers(
        log_initial_prob,
        log_transition_prob,
        log_conditional_prob,
        session_starts,
        log_alphas,
    )


@partial(jax.jit, static_argnames=["log_likelihood_func"])
def forward_pass_assoc(
    params: ModelParamsT,
    X: Array,
    y: Array,
    log_likelihood_func: LogLikelihoodFn,
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
    :func:`~nemos.hmm.forward_backward_assoc` : Computes both forward and backward messages for smoothing.

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
    session_starts = (
        session_starts
        if session_starts is not None
        else jnp.zeros(y.shape[0], dtype=bool).at[0].set(True)
    )

    # Compute log-likelihoods
    log_conditionals = log_likelihood_func(model_params, X, y)

    # Compute forward pass
    log_alphas, log_normalizers = _forward_pass_assoc(
        log_initial_prob, log_transition_prob, log_conditionals, session_starts
    )  # these are equivalent to the forward pass with python loop
    return log_alphas, log_normalizers


def _combine_backward(
    later: Tuple[Array, Array], earlier: Tuple[Array, Array]
) -> Tuple[Array, Array]:
    r"""Compose backward transfer matrices, keeping their scale.

    Nothing is renormalized here, unlike in :func:`combine_forward`: the backward messages are
    not distributions over the states, they carry the :math:`1/c_t` factors so that
    ``log_alphas + log_betas`` is the log posterior, so there is no row-stochastic
    invariant to restore. Nothing needs to be either, the :math:`1/c_t` folded into each
    element being what keeps the products O(1).

    Note the argument order: ``reverse=True`` hands the operands to the combine in
    reverse time, so ``later`` arrives first, and the product below still places the
    earlier segment on the left, :math:`N_t (N_{t+1} \cdots N_{T-1})`. A reset in
    ``earlier`` truncates the block, a session start at that index making the message
    independent of everything after it, and the flag is propagated so that a block
    containing a reset shadows any later block it is composed with.
    """
    log_N_later, reset_later = later
    log_N_earlier, reset_earlier = earlier
    composed = jnp.where(
        reset_earlier[:, jnp.newaxis, jnp.newaxis],
        log_N_earlier,
        log_matmul(log_N_earlier, log_N_later),
    )
    return composed, reset_later | reset_earlier


def _backward_pass_assoc(
    log_transition_prob: Array,
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
    log_transition_prob :
        Log transition matrix, shape (n_states, n_states), where entry ``[j, i]`` is
        :math:`\log p(z_t = i | z_{t-1} = j)`.
    log_conditional_prob :
        Log-emissions :math:`\log p(y_t | z_t=i)`, shape (n_time_bins, n_states).
    log_normalizers :
        Log normalizers :math:`\log c_t` from the forward pass, shape (n_time_bins,).
    session_starts :
        Boolean array of shape (n_time_bins,) marking the start of a new session.

    Returns
    -------
    :
        Log backward messages :math:`\log \hat{\beta}_t`, shape (n_time_bins, n_states),
        normalized as in eqn. 13.62 of [1]_ so that their sum with the log filtered
        probabilities is the log marginal posterior.

    References
    ----------
    .. [1] Bishop, C. M. (2006). Pattern recognition and machine learning. Springer.
    """
    n_states = log_conditional_prob.shape[1]

    # log b_t - log c_t is the pairing that stays O(1): each is of the order of the
    # per-bin log-likelihood, large and negative for a population model, while their
    # difference is a ratio of two averages of the same numbers.
    # The reset element R satisfies R 1 = 1, that is, beta_hat = 1 at a session end,
    # which in the log is a row-constant matrix of -log K.
    log_elements = jnp.where(
        session_starts[:, jnp.newaxis, jnp.newaxis],
        -jnp.log(jnp.asarray(n_states, log_conditional_prob.dtype)),
        log_transition_prob[jnp.newaxis]
        + (log_conditional_prob - log_normalizers[:, jnp.newaxis])[:, jnp.newaxis, :],
    )
    log_cumulative, _ = jax.lax.associative_scan(
        _combine_backward, (log_elements, session_starts), reverse=True
    )
    # cumulative[t] maps beta_{T-1} = 1 to beta_{t-1}, so the messages are its row
    # sums, here a logsumexp. Index 0 is dropped, there being no beta_{-1}, and it
    # cannot pollute the others since cumulative[t] folds only the indices >= t.
    log_betas = jax.scipy.special.logsumexp(log_cumulative[1:], axis=2)
    return jnp.concatenate([log_betas, jnp.zeros((1, n_states), log_betas.dtype)])


@partial(jax.jit, static_argnames=["log_likelihood_func"])
def forward_backward_assoc(
    params: ModelParamsT,
    X: Array,
    y: Array,
    log_likelihood_func: LogLikelihoodFn,
    session_starts: Array | None = None,
) -> EStepOutput:
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
    log_initial_prob = params.hmm_params.log_initial_prob
    log_transition_prob = params.hmm_params.log_transition_prob

    # Initialize variables
    n_time_bins = y.shape[0]
    session_starts = (
        session_starts
        if session_starts is not None
        else jnp.zeros(y.shape[0], dtype=bool).at[0].set(True)
    )

    # Compute log-likelihoods
    log_conditionals = log_likelihood_func(model_params, X, y)

    # Compute forward pass
    log_alphas, log_normalization = _forward_pass_assoc(
        log_initial_prob, log_transition_prob, log_conditionals, session_starts
    )  # these are equivalent to the forward pass with python loop

    # Compute backward pass
    log_betas = _backward_pass_assoc(
        log_transition_prob, log_conditionals, log_normalization, session_starts
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
    # Posterior over consecutive states summed across time steps. Both passes return
    # the same log-space messages as the sequential ones, so this is the sequential
    # E-step's own function rather than a reimplementation of it.
    log_joint_posterior = compute_xi_log(
        log_alphas,
        log_betas,
        log_conditionals,
        log_normalization,
        session_starts,
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


def max_plus_matmul(log_earlier: Array, log_later: Array) -> Array:
    r"""Max-plus product of two segments, earlier on the left.

    :math:`\text{out}[i,j] = \max_k(\text{earlier}[i,k] + \text{later}[k,j])`, the
    tropical-semiring counterpart of :func:`log_matmul`. Max-plus is a semiring, so
    this product is associative and the Viterbi recursion is a prefix problem in
    the same way the forward pass is.
    """
    return jnp.max(
        log_earlier[..., :, :, jnp.newaxis] + log_later[..., jnp.newaxis, :, :],
        axis=-2,
    )


def _compose_backpointers(later: Array, earlier: Array) -> Array:
    """Compose two backpointer maps, ``(earlier o later)[i] = earlier[later[i]]``.

    Backtracking is a chain of maps from the state at one step to the state at the
    previous one, and composing maps is associative, so the backtrack is a scan too.
    Without it the parallel forward would be paired with a sequential pointer chase
    of the same length, which is the cost the scan exists to remove.

    Session boundaries need no flag here: a reset makes the map constant, and a
    constant map absorbs everything composed after it, which is the statement that
    the state before a session start does not depend on the state after it.

    As in :func:`_combine_backward`, ``reverse=True`` hands the operands over in
    reverse time, so ``later`` arrives first while the composition still applies the
    earlier map last.
    """
    return jnp.take_along_axis(earlier, later, axis=-1)


@partial(
    jax.jit,
    static_argnames=["log_likelihood_func", "return_index"],
)
def max_sum_assoc(
    params: ModelParamsT,
    X: Array,
    y: Array,
    log_likelihood_func: LogLikelihoodFn,
    session_starts: Array | None = None,
    return_index: bool = False,
):
    r"""
    Find maximum a posteriori (MAP) state path via the max-sum algorithm.

    Associative-scan counterpart of
    :func:`~nemos.hmm.expectation_maximization.max_sum`, returning the same path.
    Both halves run at depth :math:`O(\\log T)`: the scores by a max-plus scan over
    the same one-step elements the forward pass uses, and the backtrack by composing
    backpointer maps.

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

    session_starts :
        Boolean array marking the start of a new session.
        If unspecified or empty, treats the full set of trials as a single session.

    return_index:
        If False, return 1-hot encoded map states, if True, return map state indices.

    Returns
    -------
    map_path:
        The MAP state path.

    Notes
    -----
    The scan accumulates the score of the sessions already closed, so its scores
    differ from those of the sequential recursion by a constant per time bin. Nothing
    reads them except through an ``argmax`` over states, which that constant leaves
    untouched, so unlike the forward pass there is no scale to drop.
    """
    # unpack parameters
    model_params = params.model_params
    log_transition = params.hmm_params.log_transition_prob
    log_init = params.hmm_params.log_initial_prob

    n_states = log_init.shape[0]

    # initialize new session
    session_starts = (
        session_starts
        if session_starts is not None
        else jnp.zeros(y.shape[0], dtype=bool).at[0].set(True)
    )

    log_emission = log_likelihood_func(model_params, X, y)

    # the elements of the scan are those of the forward pass, read in the tropical
    # semiring: at a session start the row-constant initial distribution, elsewhere
    # the transition matrix, both conditioned on that step's emissions.
    log_base = jnp.where(
        session_starts[:, jnp.newaxis, jnp.newaxis], log_init, log_transition
    )
    cumulative = jax.lax.associative_scan(
        max_plus_matmul, log_base + log_emission[:, jnp.newaxis, :]
    )
    omegas = cumulative[:, 0, :]

    # Backpointers, recomputed from the scores rather than carried through the scan,
    # the same trade _compute_log_normalizers makes: one batched argmax over all t.
    backpointers = jnp.argmax(
        omegas[:-1, :, jnp.newaxis] + log_transition[jnp.newaxis], axis=1
    )
    # at a session start the previous state is not reachable from the current one, so
    # the map is the constant that closes the previous session at its own best state.
    backpointers = jnp.where(
        session_starts[1:, jnp.newaxis],
        jnp.argmax(omegas[:-1], axis=-1)[:, jnp.newaxis],
        backpointers,
    )

    # Backward pass
    best_final_state = jnp.argmax(omegas[-1])
    composed = jax.lax.associative_scan(
        _compose_backpointers, backpointers, reverse=True
    )
    map_path = jnp.concatenate(
        [composed[:, best_final_state], jnp.array([best_final_state])]
    )

    if not return_index:
        map_path = jax.nn.one_hot(map_path, n_states, dtype=jnp.int32)

    return map_path
