from typing import Callable, Dict, Union

from functools import partial

import jax
import jax.numpy as jnp

from pynapple import IntervalSet

from ..typing import DESIGN_INPUT_TYPE
from . import utils
from .params import PPGLMParams, PPGLMParamsWithKey

jax.config.update("jax_enable_x64", True)


def _compute_lam_tilde(
    dts: jnp.ndarray,
    weights: Union[jnp.ndarray, Dict],
    bias: jnp.ndarray,
    eval_function,
) -> jnp.ndarray:
    """
    Evaluate the non-rectified firing rate (lambda tilde) at a single time point.

     Selects the coefficients for the predictors present in the history
     window, evaluates the basis functions at the lag times, and accumulates
     their weighted sum plus the bias for the target neuron(s).

    Parameters
    ----------
    dts :
        Lag times between the reference time point and each event in the history
        window. Shape (max_window,).
    weights :
        Model coefficients selected for neurons present in the history window.
        Shape (max_window, n_basis_funcs, n_neurons).
    bias :
        Intercept for each target neuron. Shape (n_neurons,).

    Returns
    -------
    :
        Non-rectified firing rate for all target neurons. Shape (n_neurons,).
    """
    fx = eval_function(dts)  # shape (max_window, n_basis_funcs)

    return jnp.einsum('hjn,hj->n', weights, fx) + bias


def _draw_mc_sample(
    X: DESIGN_INPUT_TYPE,
    random_key: jnp.ndarray,
    M_samples,
    recording_time,
    M_grid,
) -> tuple:
    """
    Draw stratified sample time points for Monte Carlo estimate
    of the conditional intensity function.

    Adds uniform random jitter to the deterministic M_grid, then finds
    the corresponding indices into the event time array X.

    Parameters
    ----------
    X :
         Padded event time series: (times: float (n_events,), predictor_ids: int (n_events,)).
    random_key :
        JAX PRNG key for sampling the jitter.

    Returns
    -------
    :
        Tuple of jnp.ndarray. Contains:
            - sample time points. Shape (M_samples,).
            - indices into X. Shape (M_samples,).
    """
    dt = recording_time.tot_length() / M_samples
    epsilon_m = jax.random.uniform(
        random_key, shape=(M_samples,), minval=0.0, maxval=dt
    )
    tau_m = M_grid + epsilon_m
    tau_m_idx = jnp.searchsorted(X[0], tau_m)
    mc_spikes = (tau_m, tau_m_idx)

    return mc_spikes


def _scan_fn_log_lam_y(
    lam_sum: jnp.ndarray,
    i: tuple,
    X: DESIGN_INPUT_TYPE,
    weights: jnp.ndarray,
    bias: jnp.ndarray,
    eval_function: Callable,
    inverse_link_function: Callable,
    max_window: int,
):
    """
    Scan body for accumulating log-firing rates at observed spike times.

    Intended to be partially applied over the static arguments before passing
    to jax.lax.scan via _log_likelihood_scan.

    Parameters
    ----------
    lam_sum :
        Running scalar sum of log-firing rates (scan carry).
    i :
        Current eval point. Entries are (spike_time, spike_neuron_id, preceding_event_idx).

    Closed over via ``functools.partial``
    --------------------------------------
    X :
         Padded event time series: (times: float (n_events,), predictor_ids: int (n_events,)).
    weights :
        Reshaped basis coefficients. Shape (n_predictors, n_basis_funcs, n_neurons).
    bias :
        Bias terms. Shape (n_neurons,).
    eval_function :
        Basis evaluation function.
    inverse_link_function :
        Maps lam_tilde to firing rate.
    max_window :
        Number of past events to include in history window.

    Returns
    -------
    lam_sum : jnp.ndarray
        Updated scalar sum.
    None
        No concatenated per-step output (required by jax.lax.scan).
    """
    spk_in_window = utils.slice_array(X[0], i[-1], max_window)
    ids_in_window = utils.slice_array(X[1], i[-1], max_window)
    dts = i[0] - spk_in_window
    lam_tilde = _compute_lam_tilde(
        dts,
        weights[ids_in_window, :, i[1], None],
        bias[i[1]],
        eval_function,
    )
    lam_sum += jnp.log(inverse_link_function(lam_tilde)).sum()
    return lam_sum, None


def _scan_fn_mc_est(
    lam_sum: jnp.ndarray,
    i: tuple,
    X: DESIGN_INPUT_TYPE,
    weights: jnp.ndarray,
    bias: jnp.ndarray,
    eval_function: Callable,
    inverse_link_function: Callable,
    max_window: int,
):
    """
    Scan body for accumulating firing rates at Monte Carlo sample points to compute
    an estimate of the firing rate integral over the recording time.

    Intended to be partially applied over the static arguments before passing
    to jax.lax.scan via _log_likelihood_scan.

    Parameters
    ----------
    lam_sum :
        Running scalar sum of log-firing rates (scan carry).
    i :
        Current eval point. Entries are (sample_time, preceding_event_idx).

    Closed over via ``functools.partial``
    --------------------------------------
    X :
         Padded event time series: (times: float (n_events,), predictor_ids: int (n_events,)).
    weights :
        Reshaped basis coefficients. Shape (n_predictors, n_basis_funcs, n_neurons).
    bias :
        Bias terms. Shape (n_neurons,).
    eval_function :
        Basis evaluation function.
    inverse_link_function :
        Maps lam_tilde to firing rate.
    max_window :
        Number of past events to include in history window.

    Returns
    -------
    lam_sum : jnp.ndarray
        Updated scalar sum.
    None
        No concatenated per-step output (required by jax.lax.scan).
    """
    spk_in_window = utils.slice_array(X[0], i[-1], max_window)
    ids_in_window = utils.slice_array(X[1], i[-1], max_window)
    dts = i[0] - spk_in_window
    lam_tilde = _compute_lam_tilde(
        dts,
        weights[ids_in_window],
        bias,
        eval_function,
    )
    lam_sum += inverse_link_function(lam_tilde).sum()
    return lam_sum, None


def _log_likelihood_scan(
    X: DESIGN_INPUT_TYPE,
    eval_pts: tuple,
    params: PPGLMParams,
    scan_function: Callable,
    inverse_link_function,
    n_basis_funcs,
    max_window,
    scan_size,
    eval_function,
) -> jnp.ndarray:
    """
    Compute the sum of log-firing rates (or firing rates) at a set of time points
    using parallelized JAX scans over batches of events.

    Iterates over time_points in y, selects the recent history events form X, evaluates
    the linear combination of predictors via lam_tilde_function, and accumulates the (log-)firing
    rates. Padding added by reshape_input_for_scan is subtracted out at the end.

    Parameters
    ----------
    X :
        Padded event time series: (times: float (n_events,), predictor_ids: int (n_events,)).
    eval_pts :
        Observed spike time series or MC sample points. Either (times: float (n_spikes,), neuron_ids: int (n_spikes,),
        event_indices: int (n_spikes,)) or (times: float (M_samples,), event_indices: int (M_samples,)), respectively.
    params :
        PPGLMParams containing the basis coefficients and bias terms.
    scan_function :
        Either _scan_fn_log_lam_y for the first NLL term or _scan_fn_mc_est for the second term .

    Returns
    -------
    :
        Scalar sum of log-firing rates (or firing rates) over all eval points,
        with padding contribution subtracted.
    """

    weights, bias = params.coef, params.intercept
    weights = utils.reshape_coef_for_scan(weights, n_basis_funcs)

    scan_body = partial(
        scan_function,
        X=X,
        weights=weights,
        bias=bias,
        eval_function=eval_function,
        inverse_link_function=inverse_link_function,
        max_window=max_window,
    )

    scan_vmap = jax.vmap(
        lambda pts: jax.lax.scan(scan_body, jnp.array(0), pts), in_axes=0
    )

    reshaped_spikes_array, padding_val, padding_len = utils.reshape_input_for_scan(
        eval_pts, scan_size
    )
    out, _ = scan_vmap(reshaped_spikes_array)  # shape (n_scans,)

    # compute padding contribution separately to subtract it
    padding_contrib = scan_body(jnp.array(0.0), padding_val)[0] * padding_len

    return jnp.sum(out) - padding_contrib


def _negative_log_likelihood(
    params: PPGLMParams,
    X: DESIGN_INPUT_TYPE,
    y: tuple,
    random_key: jnp.ndarray,
    inverse_link_function: Callable,
    M_samples: int,
    M_grid: jnp.ndarray,
    recording_time: IntervalSet,
    n_basis_funcs: int,
    scan_size: int,
    max_window: int,
    eval_function: Callable,
    aggregate_sample_scores: Callable = lambda l, y: l / y.shape[0],
) -> jnp.ndarray:
    r"""
    Compute the Poisson point process negative log-likelihood with a Monte Carlo
    estimate of the conditional intensity function (CIF).

    Evaluates:

    $\sum_{k=1}^K \log \lambda(y_k) - \frac{T}{M} \sum_{m=1}^M \lambda(\tau_m)$

    where the first term sums log-firing rates at observed spike times, y, and the second
    term is the MC estimate of $\int_0^T \lambda(t) dt$.

    Parameters
    ----------
    X :
        Padded event time series: (times: float (n_events,), predictor_ids: int (n_events,)).
    y :
         Spike time series: (times: float (n_spikes,), neuron_ids: int (n_spikes,),
        event_indices: int (n_spikes,)).
    params :
        PPGLMParams containing the basis coefficients and bias terms.
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
        Number of time points processed per scan.
    max_window :
        The maximum number of events falling within the history window.
    eval_function :
        A function evaluating basis at lag times.

    Returns
    -------
    :
        Scalar negative log-likelihood.
    """

    log_lambda_y = _log_likelihood_scan(
        X,
        y,
        params,
        _scan_fn_log_lam_y,
        inverse_link_function,
        n_basis_funcs,
        max_window,
        scan_size,
        eval_function,
    )

    mc_samples = _draw_mc_sample(
        X,
        random_key,
        M_samples,
        recording_time,
        M_grid,
    )

    mc_estimate = _log_likelihood_scan(
        X,
        mc_samples,
        params,
        _scan_fn_mc_est,
        inverse_link_function,
        n_basis_funcs,
        max_window,
        scan_size,
        eval_function,
    )

    nll_sum = ((recording_time.tot_length() / M_samples) * mc_estimate) - log_lambda_y

    return aggregate_sample_scores(nll_sum, y[0])


def _compute_loss(
    params_with_key: PPGLMParamsWithKey,
    X: DESIGN_INPUT_TYPE,
    y: tuple,
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
        Padded event time series for all predictors (spikes, stimuli, etc.).
        Stored as (times, predictor_ids) where:
            - times : jnp.ndarray of float, shape (n_events,) — event timestamps
            - prededictor_ids : jnp.ndarray of int, shape (n_events,) — predictor ids(neuron ids, stimulus
            ids, etc.)
    y :
        Spike time series for the neurons being modeled.
        Stored as (times, neuron_ids, event_indices) where:
            - times : jnp.ndarray of float, shape (n_spikes,) — spike timestamps
            - neuron_ids : jnp.ndarray of int, shape (n_spikes,) — neuron indices
            - event_indices : jnp.ndarray of int, shape (n_spikes,) — indices into X.times

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
