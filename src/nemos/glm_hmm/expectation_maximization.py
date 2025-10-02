"""Forward backward pass for a GLM-HMM."""

from functools import partial
from typing import Any, Callable, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from numpy.typing import NDArray

from ..third_party.jaxopt.jaxopt import LBFGS

Array = NDArray | jax.numpy.ndarray


class GLMHMMState(eqx.Module):
    """State class for the GLMHHM EM-algorithm."""

    initial_prob: Array
    transition_matrix: Array
    projection_weights: Array
    data_log_likelihood: float | Array
    iterations: int


def _analytical_m_step_initial_prob(
    posteriors: jnp.ndarray,
    is_new_session: jnp.ndarray,
    dirichlet_prior_alphas: Optional[jnp.ndarray] = None,
):
    """
    Calculate the M-step for initial probabilities.

    Parameters
    ----------
    posteriors:
        The posterior distribution over latent states, shape ``(n_time_bins, n_states)``.
    dirichlet_prior_alphas:
        The parameters of the Dirichlet prior, if available. Flat prior otherwise.

    Returns
    -------
        Updated initial parameters.
    """
    tmp_initial_prob = jnp.sum(posteriors, axis=0, where=is_new_session[:, jnp.newaxis])
    if dirichlet_prior_alphas is not None:
        tmp_initial_prob += dirichlet_prior_alphas - 1

    new_initial_prob = tmp_initial_prob / jnp.sum(tmp_initial_prob)
    return new_initial_prob


def _analytical_m_step_transition_prob(
    joint_posterior: jnp.ndarray, dirichlet_prior_alphas: Optional[jnp.ndarray] = None
):
    if dirichlet_prior_alphas is not None:
        new_transition_prob = joint_posterior + dirichlet_prior_alphas - 1
    else:
        new_transition_prob = joint_posterior
    new_transition_prob /= jnp.sum(new_transition_prob, axis=0)[jnp.newaxis]
    return new_transition_prob


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
    :
        Expected joint posteriors between time steps, shape ``(n_states, n_states)``.
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
    computing the filtered probabilities (``alpha``) at each time step. At the start of
    a new session, the recursion is reset using the initial state distribution.

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
    using ``jax.lax.scan`` in reverse to compute beta messages at each time step. It handles
    session boundaries by resetting the beta messages when a new session starts.

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
        Latent state transition matrix, pytree with leaves of shape ``(n_states, n_states)``.
        ``transition_prob[i, j]`` is the probability of transitioning from state ``i`` to state ``j``.

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
    posteriors :
        Marginal posterior distribution over latent states, shape ``(n_time_bins, n_states)``.

    joint_posterior :
        Joint posterior distribution between consecutive time steps summed
        over samples, shape ``(n_states, n_states)``.

    log_likelihood :
        Total log-likelihood of the observation sequence under the model.

    log_likelihood_norm :
        A vmapped function that computes the elementwise log-likelihood between observed
        and predicted values. Must return an array of shape ``(n_time_bins, n_states)``.
        The vmapping over states must be performed by the caller, outside this function,
        using `jax.vmap` or equivalent, so that the passed function is already fully
        vectorized over the state dimension.

    alphas :
        Forward messages (alpha values), shape ``(n_time_bins, n_states)``.

    betas :
        Backward messages (beta values), shape ``(n_time_bins, n_states)``.

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
    if projection_weights.ndim > 2:
        predicted_rate_given_state = inverse_link_function(
            jnp.einsum("ik, kjw->ijw", X, projection_weights)
        )
    else:
        predicted_rate_given_state = inverse_link_function(X @ projection_weights)

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

    log_likelihood_norm = jnp.exp(log_likelihood / n_time_bins)  # Normalize

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
        log_likelihood_norm,
        alphas,
        betas,
    )


@partial(
    jax.jit, static_argnames=["inverse_link_function", "negative_log_likelihood_func"]
)
def hmm_negative_log_likelihood(
    projection_weights: Array,
    X: Array,
    y: Array,
    posteriors: Array,
    inverse_link_function: Callable,
    negative_log_likelihood_func: Callable,
):
    """
    Compute the negative log-likelihood of the GLM-HMM.

    Compute the negative log-likelihood as a function of the projection weights.

    Parameters
    ----------
    projection_weights:
        Projection weights for the GLM.
    X:
        Design matrix of observations.
    y:
        Target responses.
    posteriors:
        Posterior probabilities over states.
    inverse_link_function:
        Function mapping linear predictors to rates.
    negative_log_likelihood_func:
        Function to compute the negative log-likelihood.

    Returns
    -------
    nll:
        The scalar negative log-likelihood weighted by the posteriors.
    """

    if projection_weights.ndim > 2:
        predicted_rate = inverse_link_function(
            jnp.einsum("ik, kjw->ijw", X, projection_weights)
        )
        nll = negative_log_likelihood_func(
            y,
            predicted_rate,
        ).sum(axis=1)
    else:
        predicted_rate = inverse_link_function(X @ projection_weights)
        nll = negative_log_likelihood_func(
            y,
            predicted_rate,
        )

    # Compute dot products between log-likelihood terms and gammas
    nll = jnp.sum(nll * posteriors)

    return nll


@partial(jax.jit, static_argnames=["solver_run"])
def run_m_step(
    X: Array,
    y: Array,
    posteriors: Array,
    joint_posterior: Array,
    projection_weights: Array,
    is_new_session: Array,
    solver_run: Callable[[Array, Array, Array, Array], Array],
    dirichlet_prior_alphas_init_prob: Array | None = None,
    dirichlet_prior_alphas_transition: Array | None = None,
) -> Tuple[Array, Array, Array, Any]:
    r"""
    Perform the M-step of the EM algorithm for GLM-HMM.

    Parameters
    ----------
    X:
        Design matrix of observations.
    y:
        Target responses.
    posteriors:
        Posterior probabilities over states.
    joint_posterior:
        Joint posterior probabilities over pairs of states
        :math:`P(z_{t-1}, z_t \mid X, y, \theta_{\text{old}})`.
    projection_weights:
        Current projection weights.
    is_new_session:
        Boolean mask for the first observation of each session.
    solver_run:
        Callable performing a full optimization loop for the GLM weights.
        Note that the prior for the projection weights is baked in the solver run.
    dirichlet_prior_alphas_init_prob:
        Prior for the initial states.
    dirichlet_prior_alphas_transition:
        Prior for the transition probabilities.

    Returns
    -------
    optimized_projection_weights:
        Updated projection weights after optimization.
    new_initial_prob:
        Updated initial state distribution.
    new_transition_prob:
        Updated transition matrix.
    state:
        State returned by the solver.
    """

    # Update Initial state probability Eq. 13.18
    new_transition_prob = _analytical_m_step_initial_prob(
        posteriors,
        is_new_session=is_new_session,
        dirichlet_prior_alphas=dirichlet_prior_alphas_init_prob,
    )
    new_initial_prob = _analytical_m_step_transition_prob(
        joint_posterior, dirichlet_prior_alphas=dirichlet_prior_alphas_transition
    )

    # Minimize negative log-likelihood to update GLM weights
    optimized_projection_weights, state = solver_run(
        projection_weights, X, y, posteriors
    )

    return optimized_projection_weights, new_initial_prob, new_transition_prob, state


def prepare_likelihood_func(
    projection_weights: Array,
    likelihood_func: Callable,
    negative_log_likelihood_func: Callable,
    is_log: bool = True,
) -> Tuple[Callable, Callable]:
    """
    Prepare a likelihood function for use in the EM algorithm.

    Parameters
    ----------
    projection_weights:
        Initial projection weights for the GLM.
    likelihood_func:
        Function computing the log-likelihood.
    negative_log_likelihood_func
        Function computing the negative log-likelihood.
    is_log:
        Whether the likelihood function returns log-likelihood values.

    Returns
    -------
    likelihood:
        Likelihood function.
    vmap_nll:
        Vectorized negative log-likelihood function.
    """

    if not is_log and projection_weights.ndim > 2:
        raise ValueError(
            "Population GLM-HMM requires log-likelihood for numerical stability."
        )

    # Wrap likelihood_func to avoid aggregating over samples
    def likelihood_per_sample(x, z):
        return likelihood_func(x, z, aggregate_sample_scores=lambda s: s)

    def negative_log_likelihood_per_sample(x, z):
        return negative_log_likelihood_func(x, z, aggregate_sample_scores=lambda s: s)

    # Vectorize over the states axis
    state_axes = 2 if projection_weights.ndim > 2 else 1
    likelihood_per_sample = jax.vmap(
        likelihood_per_sample,
        in_axes=(None, state_axes),
        out_axes=state_axes,
    )

    def likelihood(y, rate):
        log_like = likelihood_per_sample(y, rate)
        if projection_weights.ndim > 2:
            # Multi-neuron case: sum log-likelihoods across neurons
            log_like = log_like.sum(axis=1)
        return jnp.exp(log_like) if is_log else log_like

    vmap_nll = jax.vmap(
        negative_log_likelihood_per_sample,
        in_axes=(None, state_axes),
        out_axes=state_axes,
    )
    return likelihood, vmap_nll


def em_glm_hmm(
    X: Array,
    y: Array,
    initial_prob: Array,
    transition_prob: Array,
    projection_weights: Array,
    is_new_session: Array,
    inverse_link_function: Callable,
    likelihood_func: Callable,
    negative_log_likelihood_func: Callable,
    maxiter: int = 10**3,
    tol: float = 1e-6,
    is_log: bool = True,
    solver_kwargs: Optional[dict] = None,
) -> Tuple[Array, Array, Array, Array, Array]:
    """
    Perform EM optimization for a GLM-HMM.

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
    projection_weights:
        Initial projection weights for the GLM.
    is_new_session:
        Boolean mask for the first observation of each session.
    inverse_link_function:
        Elementwise function mapping linear predictors to rates.
    likelihood_func:
        Function computing the log-likelihood, usually either:

        - ``nemos.observation_models.Observations.log_likelihood``, if ``is_log==True``.
        - ``nemos.observation_models.Observations.likelihood``, if ``is_log==False``.

    negative_log_likelihood_func:
        Function computing the negative log-likelihood, usually
        ``nemos.observation_models.Observations._negative_log_likelihood``.
    maxiter:
        Maximum number of EM iterations.
    tol:
        Convergence tolerance on the log-likelihood.
    is_log:
        Whether the likelihood function returns log-likelihood values.
    solver_kwargs:
        Additional keyword arguments for the solver.

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
    """
    likelihood_func, negative_log_likelihood_func = prepare_likelihood_func(
        projection_weights, likelihood_func, negative_log_likelihood_func, is_log=is_log
    )

    # closure for the static callables
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

    if solver_kwargs is None:
        solver_kwargs = {}
    # define a solver
    solver = LBFGS(partial_hmm_negative_log_likelihood, **solver_kwargs)

    state = GLMHMMState(
        initial_prob=initial_prob,
        transition_matrix=transition_prob,
        projection_weights=projection_weights,
        data_log_likelihood=-jnp.array(jnp.inf),
        iterations=0,
    )

    def em_step(carry, xs):
        _, previous_state = carry
        (
            posteriors,
            joint_posterior,
            log_likelihood,
            log_likelihood_norm,
            alphas,
            betas,
        ) = forward_backward(
            X,
            y,
            previous_state.initial_prob,
            previous_state.transition_matrix,
            previous_state.projection_weights,
            inverse_link_function,
            likelihood_func,
            is_new_session,
        )

        # alphas[-1] is p(y_1,...,y_n, z_n), see 13.34 Bishop
        # marginalizing over z_n we have the data likelihood:
        # p(y_1,...,y_n) = sum_{z_n} p(y_1,...,y_n, z_n)

        new_log_like = jnp.log(alphas[-1].sum())

        proj_weights, init_prob, trans_matrix, _ = run_m_step(
            X,
            y,
            posteriors=posteriors,
            joint_posterior=joint_posterior,
            projection_weights=previous_state.projection_weights,
            is_new_session=is_new_session,
            solver_run=solver.run,
        )

        new_state = GLMHMMState(
            initial_prob=init_prob,
            transition_matrix=trans_matrix,
            projection_weights=proj_weights,
            iterations=previous_state.iterations + 1,
            data_log_likelihood=new_log_like,
        )
        return (previous_state.data_log_likelihood, new_state), new_log_like

    def stopping_condition(carry, _):
        old_likelihood, new_state = carry
        return jnp.abs(new_state.data_log_likelihood - old_likelihood) < tol

    def body_fn(carry, xs):
        return jax.lax.cond(
            stopping_condition(carry, xs),
            lambda c, _: (c, jnp.array(jnp.nan)),
            em_step,
            carry,
            xs,
        )

    (_, state), likelihoods = jax.lax.scan(
        body_fn, (jnp.array(-jnp.inf), state), length=maxiter
    )
    # final posterior calculation
    (posteriors, joint_posterior, _, _, _, _) = forward_backward(
        X,
        y,
        state.initial_prob,
        state.transition_matrix,
        state.projection_weights,
        inverse_link_function,
        likelihood_func,
        is_new_session,
    )
    return (
        posteriors,
        joint_posterior,
        state.initial_prob,
        state.transition_matrix,
        state.projection_weights,
    )
