"""PP-GLM core log-likelihood computation."""

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from pynapple import IntervalSet

from . import utils
from .data import MCSamplePPGLM, PredictorsPPGLM, SpikesPPGLM
from .params import GLMParams, PPGLMParamsWithKey


def _eval_point(
    t: jnp.ndarray,
    idx: jnp.ndarray,
    X: PredictorsPPGLM,
    eval_function: Callable,
    max_window: int,
    n_predictors: int,
) -> jnp.ndarray:
    """
    Build the feature vector for a single evaluation time point.

    Slices the max_window events preceding the eval time point, evaluates the basis
    at the resulting lag times, and accumulates the basis values per presynaptic neuron
    with a segment sum. The result is the row of the design matrix for this time
    point for all features.

    Parameters
    ----------
    t :
        Timestamp for evaluation.
    idx :
        Index of the evaluation point into the event time array X.
    X :
        Preprocessed predictors with fields ``times`` (event timestamps) and
        ``predictor_ids``.
    eval_function :
        Basis evaluation function mapping lag times to basis values.
    max_window :
        Number of past events to include in the history window.
    n_predictors :
        Number of predictors that defines the number of segments.

    Returns
    -------
    :
        Feature vector. Shape (n_predictors * n_basis_funcs,).
    """
    ts = utils.slice_array(X.times, idx, max_window)
    ids = utils.slice_array(X.predictor_ids, idx, max_window)

    fx = eval_function(t - ts)  # shape (max_window, n_basis_funcs)

    # shape (n_predictors * n_basis_funcs)
    return jax.ops.segment_sum(fx, ids, num_segments=n_predictors).reshape(-1)


def _compute_design_matrix(
    timestamps: jnp.ndarray,
    timestamp_idx: jnp.ndarray,
    X: PredictorsPPGLM,
    eval_function: Callable,
    max_window: int,
    n_predictors: int,
) -> jnp.ndarray:
    """
    Build the design matrix for a chunk of evaluation time points.

    Maps `_eval_point` over the chunk, so that every row is the feature
    vector at one evaluation point.

    Parameters
    ----------
    timestamps :
        Evaluation timestamps for this chunk. Shape (chunk_size,).
    timestamp_idx :
        Indices of the evaluation points into the event time array X.
        Shape (chunk_size,).
    X :
        Preprocessed predictors with fields ``times`` and ``predictor_ids``.
    eval_function :
        Basis evaluation function mapping lag times to basis values.
    max_window :
        Number of past events to include in the history window.
    n_predictors :
        Number of predictors that defines the number of segments.

    Returns
    -------
    :
        Design matrix. Shape (chunk_size, n_predictors * n_basis_funcs).
    """
    eval_point = partial(
        _eval_point,
        X=X,
        eval_function=eval_function,
        max_window=max_window,
        n_predictors=n_predictors,
    )
    return jax.vmap(eval_point, in_axes=(0, 0))(timestamps, timestamp_idx)


def _compute_log_lambda_y(
    X: PredictorsPPGLM,
    y: SpikesPPGLM,
    weights: jnp.ndarray,
    bias: jnp.ndarray,
    inverse_link_function: Callable,
    eval_function: Callable,
    max_window: int,
    n_predictors: int,
    chunk_size: int,
) -> jnp.ndarray:
    """
    Compute the log-firing rates at the observed spike times.

    Scans over chunks of spikes in y. For each chunk, builds the feature matrix
    from recent history events, computes and sums log firing rates selecting the
    neuron that spiked.

    Parameters
    ----------
    X :
        Preprocessed predictors with fields ``times`` and ``predictor_ids``.
    y :
        Preprocessed spikes with fields ``times``, ``neuron_ids`` and ``timestamp_idx``.
    weights :
        Model coefficients. Shape (n_predictors * n_basis_funcs, n_neurons).
    bias :
        Intercept for each target neuron. Shape (n_neurons,).
    inverse_link_function :
        Maps the linear predictor to a firing rate.
    eval_function :
        Basis evaluation function.
    max_window :
        Number of past events to include in the history window.
    n_predictors :
        Number of predictors that defines the number of segments.
    chunk_size :
        Number of evaluation points processed per scan.

    Returns
    -------
    :
        Sum of log-firing rates at observed spike times.
    """
    chunked, valid = utils._reshape_and_pad_eval_points(y, chunk_size)

    def body(lam_sum, chunk):
        spikes, is_valid = chunk

        A = _compute_design_matrix(
            spikes.times,
            spikes.timestamp_idx,
            X,
            eval_function,
            max_window,
            n_predictors,
        )
        lam_tilde = A @ weights + bias

        # select rate of the neuron that actually fired in each row
        log_lam = jnp.log(
            inverse_link_function(lam_tilde[jnp.arange(chunk_size), spikes.neuron_ids])
        )
        return lam_sum + jnp.sum(jnp.where(is_valid, log_lam, 0.0)), None

    init = jnp.zeros((), dtype=X.times.dtype)
    log_lambda_y, _ = jax.lax.scan(body, init, (chunked, valid))

    return log_lambda_y


def _compute_mc_estimate(
    X: PredictorsPPGLM,
    mc_samples: MCSamplePPGLM,
    weights: jnp.ndarray,
    bias: jnp.ndarray,
    inverse_link_function: Callable,
    eval_function: Callable,
    max_window: int,
    n_predictors: int,
    chunk_size: int,
) -> jnp.ndarray:
    """
    Compute the firing rates at the Monte Carlo sample points.

    Scans over chunks of sample points. For each chunk, builds the feature matrix
    from recent history events, computes and sums firing rates across all neurons.

    Parameters
    ----------
    X :
        Preprocessed predictors with fields ``times`` and ``predictor_ids``.
    mc_samples :
        Monte Carlo samples with fields ``times`` and ``timestamp_idx``.
    weights :
        Model coefficients. Shape (n_predictors * n_basis_funcs, n_neurons).
    bias :
        Model intercepts. Shape (n_neurons,).
    inverse_link_function :
        Maps the linear predictor to a firing rate.
    eval_function :
        Basis evaluation function.
    max_window :
        Number of past events to include in the history window.
    n_predictors :
        Number of predictors that defines the number of segments.
    chunk_size :
        Number of evaluation points processed per scan.

    Returns
    -------
    :
        Sum of firing rates at the Monte Carlo sample points.
    """
    chunked, valid = utils._reshape_and_pad_eval_points(mc_samples, chunk_size)

    def body(lam_sum, chunk):
        samples, is_valid = chunk

        A = _compute_design_matrix(
            samples.times,
            samples.timestamp_idx,
            X,
            eval_function,
            max_window,
            n_predictors,
        )
        lam = inverse_link_function(A @ weights + bias)

        return lam_sum + jnp.sum(jnp.where(is_valid, jnp.sum(lam, axis=-1), 0.0)), None

    init = jnp.zeros((), dtype=X.times.dtype)
    mc_estimate, _ = jax.lax.scan(body, init, (chunked, valid))

    return mc_estimate


def _draw_mc_sample(
    X: PredictorsPPGLM,
    random_key: jnp.ndarray,
    M_samples: int,
    T: float,
    M_grid,
) -> MCSamplePPGLM:
    """
    Draw stratified sample time points for Monte Carlo estimate of the conditional intensity function.

    Adds uniform random jitter to the deterministic M_grid, then finds
    the corresponding indices into the event time array X.

    Parameters
    ----------
    X :
        Preprocessed predictors with fields ``times`` (event timestamps) and
        ``predictor_ids``.
    random_key :
        JAX PRNG key for sampling the jitter.

    Returns
    -------
    mc_sample_pts :
        Monte Carlo samples with fields ``times`` (sampled timestamps) and
        ``timestamp_idx`` (indices into event times).
    """
    dt = T / M_samples
    epsilon_m = jax.random.uniform(
        random_key, shape=(M_samples,), minval=0.0, maxval=dt
    )
    tau_m = M_grid + epsilon_m
    tau_m_idx = jnp.searchsorted(X.times, tau_m)
    mc_sample_pts = MCSamplePPGLM(times=tau_m, timestamp_idx=tau_m_idx)

    return mc_sample_pts


def _negative_log_likelihood(
    params: GLMParams,
    X: PredictorsPPGLM,
    y: SpikesPPGLM,
    random_key: jnp.ndarray,
    inverse_link_function: Callable,
    M_samples: int,
    M_grid: jnp.ndarray,
    recording_time: IntervalSet,
    n_basis_funcs: int,
    scan_size: int,
    max_window: int,
    eval_function: Callable,
) -> jnp.ndarray:
    r"""
    Compute the Poisson point process negative log-likelihood with a Monte Carlo estimate of the CIF integral.

    Evaluates:

    $\sum_{k=1}^K \log \lambda(y_k) - \frac{T}{M} \sum_{m=1}^M \lambda(\tau_m)$

    where the first term sums log-firing rates at observed spike times, y, and the second
    term is the MC estimate of $\int_0^T \lambda(t) dt$.

    Parameters
    ----------
    X :
        Preprocessed predictors with fields ``times`` (event timestamps) and ``predictor_ids``.
    y :
        Preprocessed spikes with fields ``times`` (spike timestamps), ``neuron_ids``
        (postsynaptic neuron indices), and ``timestamp_idx`` (indices into event times).
    params :
        GLMParams containing the basis coefficients and bias terms.
    random_key :
        JAX PRNG key used to jitter the MC integration grid.

    === all arguments below will be model attributes ===
    inverse_link_function :
        A function that maps the linear combination of predictors to a firing rate.
    M_samples :
        Number of Monte Carlo samples for the integral estimate.
    M_grid :
        Stratified grid for MC integration. Shape (M_samples,).
    recording_time :
        pynapple IntervalSet defining the recording epochs.
    n_basis_funcs :
        Number of basis functions.
    scan_size :
        Number of evaluation points processed per loop iteration (chunk size).
    max_window :
        The maximum number of events falling within the history window.
    eval_function :
        A function evaluating basis at lag times.

    Returns
    -------
    :
        Scalar negative log-likelihood.
    """
    weights = utils._reshape_2d_coef(params.coef)  # (n_pred * n_basis, n_neurons)
    bias = params.intercept  # (n_neurons,)
    n_predictors = weights.shape[0] // n_basis_funcs

    log_lambda_y = _compute_log_lambda_y(
        X,
        y,
        weights,
        bias,
        inverse_link_function,
        eval_function,
        max_window,
        n_predictors,
        scan_size,
    )

    mc_samples = _draw_mc_sample(
        X,
        random_key,
        M_samples,
        recording_time.tot_length(),
        M_grid,
    )

    mc_estimate = _compute_mc_estimate(
        X,
        mc_samples,
        weights,
        bias,
        inverse_link_function,
        eval_function,
        max_window,
        n_predictors,
        scan_size,
    )

    nll_sum = ((recording_time.tot_length() / M_samples) * mc_estimate) - log_lambda_y

    return nll_sum / y.times.shape[0]


def _compute_loss(
    params_with_key: PPGLMParamsWithKey,
    X: PredictorsPPGLM,
    y: SpikesPPGLM,
    *args,
    **kwargs,
) -> jnp.ndarray:
    """
    Compute the negative log-likelihood loss for stochastic optimization.

    Splits the PRNG key before calling the nll function.

    Parameters
    ----------
    params_with_key :
        PPGLMParamsWithKey instance combining model params (coef, intercept) and
        a random key used for MC sampling.
    X :
        Preprocessed predictors with fields ``times`` (event timestamps) and ``ids`` (predictor neuron indices).
    y :
        Preprocessed spikes with fields ``times`` (spike timestamps), ``ids``
        (postsynaptic neuron indices), and ``idx`` (indices into event times).

    Returns
    -------
    :
        The model negative log-likelihood. Shape (1,).
    """
    key = params_with_key.random_key.astype(jnp.uint32)

    new_key, _ = jax.random.split(key)

    neg_ll = _negative_log_likelihood(
        params_with_key.params, X, y, new_key, *args, **kwargs
    )

    return neg_ll
