import jax
import jax.numpy as jnp

from ..typing import DESIGN_INPUT_TYPE

from . import utils
from .params import PPGLMParamsWithKey, PPGLMParams


def compute_lam_tilde_single(dts, i, event_ids, weights, bias, eval_function):
    """
        Evaluate the non-rectified firing rate (lambda tilde) for a single
        target neurons at a single time point.

        Selects the coefficients for the predictors present in the history
        window, evaluates the basis functions at the lag times, and accumulates
        their weighted sum plus the bias for the target neuron indexed by i.

        Parameters
        ----------
        dts :
            Lag times between the reference time point and each event in the history
            window. Shape (max_window,).
        i :
            Target neuron id. Scalar int.
        event_ids :
            An array of predictor ids corresponding to all events in the history window.
        weights :
            Reshaped model coefficients. Shape (n_predictors, n_basis_funcs, n_neurons).
        bias :
            Intercept for each target neuron. Shape (n_neurons,).

        Returns
        -------
        :
            Non-rectified firing rate for all target neurons. Shape (1,).
    """
    w = weights[event_ids, :, i]       # shape (max_window, n_basis_funcs)
    fx = eval_function(dts)            # shape (max_window, n_basis_funcs)

    return jnp.sum(fx * w, axis=(0, 1)) + bias[i]

def compute_lam_tilde_all(dts, i, event_ids, weights, bias, eval_function):
    """
        Evaluate the non-rectified firing rate (lambda tilde) for all target neurons
        at a single time point.

        Used during the CIF integral computation, where the intensity must
        be evaluated for all target neurons.

        Parameters
        ----------
        dts :
            Lag times between the reference time point and each event in the history
            window. Shape (max_window,).
        i :
            Target neuron id (unused in this function; retained for a consistent signature
            with compute_lam_tilde_single).
        event_ids :
            An array of event ids corresponding to all events in the history window.
            Used to select the corresponding coefficients (weights).
        weights :
            Reshaped model coefficients. Shape (n_predictors, n_basis_funcs, n_neurons).
        bias :
            Intercept for each target neuron. Shape (n_neurons,).

        Returns
        -------
        :
            Non-rectified firing rate for all target neurons. Shape (n_neurons,).
    """
    w = weights[event_ids]          # shape (max_window, n_basis_funcs, n_neurons)
    fx = eval_function(dts)         # shape (max_window, n_basis_funcs)

    return jnp.sum(fx[:, :, None] * w, axis=(0, 1)) + bias

def draw_mc_sample(
        X: DESIGN_INPUT_TYPE,
        M_samples,
        random_key,
        recording_time,
        M_grid,
):
    """
        Draw stratified sample time points for Monte Carlo estimate
        of the conditional intensity function.

        Adds uniform random jitter to the deterministic M_grid, then finds
        the corresponding indices into the event time array X.

        Parameters
        ----------
        X :
            Padded event time series. Shape (2, n_events).
        random_key :
            JAX PRNG key for sampling the jitter.

        Returns
        -------
        :
            Marked array of sample time points and their indices into X.
            Shape (2, M_samples).
    """
    dt = recording_time.tot_length() / M_samples
    epsilon_m = jax.random.uniform(random_key, shape=(M_samples,), minval=0.0, maxval=dt)
    tau_m = M_grid + epsilon_m
    tau_m_idx = jnp.searchsorted(X[0], tau_m)
    mc_spikes = jnp.vstack((tau_m, tau_m_idx))

    return mc_spikes

def log_likelihood_scan(
        X,
        eval_pts,
        params,
        inverse_link_function,
        n_basis_funcs,
        max_window,
        scan_size,
        eval_function,
        lam_tilde_function,
        log=False,
):
    """
        Compute the sum of log-firing rates (or firing rates) at a set of time points
        using parallelized JAX scans over batches of events.

        Iterates over time_points in y, selects the recent history events form X, evaluates
        the linear combination of predictors via lam_tilde_function, and accumulates the (log-)firing
        rates. Padding added by reshape_input_for_scan is subtracted out at the end.

        Parameters
        ----------
        X :
            Padded event time series. Shape (2, n_events).
        eval_pts :
            Observed spike time series or MC sample points. Shape (n_channels, n_time_points).
        params :
            PPGLMParams containing the basis coefficients and bias terms.
        lam_tilde_function :
            Either compute_lam_tilde_single or compute_lam_tilde_all, depending on
            whether intensity is needed for one target neuron or all.
        log :
            If True, accumulate log-firing rates; if False, accumulate firing rates directly.
            Default False.

        Returns
        -------
        :
            Scalar sum of log-firing rates (or firing rates) over all eval points,
            with padding contribution subtracted.
    """
    optional_log = jnp.log if log else lambda x: x

    weights, bias = params.coef, params.intercept
    weights = utils.reshape_coef_for_scan(weights, n_basis_funcs)

    # body of the scan function
    def scan_fn(lam_s, i):
        spk_in_window = utils.slice_array(
            X, i[-1].astype(int), max_window
        )
        dts = i[0] - spk_in_window[0]
        lam_tilde = lam_tilde_function(
            dts,
            i[1].astype(int),
            spk_in_window[1].astype(int),
            weights,
            bias,
            eval_function
        )
        lam_s += optional_log(inverse_link_function(lam_tilde)).sum()
        return lam_s, None

    scan_vmap = jax.vmap(lambda idxs: jax.lax.scan(scan_fn, jnp.array(0), idxs), in_axes=0)

    reshaped_spikes_array, padding_val, padding_len = utils.reshape_input_for_scan(eval_pts, scan_size)
    out, _ = scan_vmap(reshaped_spikes_array)

    # compute padding contribution separately to subtract it
    padding_lam = scan_fn(jnp.array(0.), padding_val)[0] * padding_len

    return jnp.sum(out) - padding_lam


def _negative_log_likelihood(
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        params: PPGLMParams,
        random_key,
        inverse_link_function,
        M_samples,
        M_grid,
        recording_time,
        n_basis_funcs,
        scan_size,
        max_window,
        eval_function,

):
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
            Padded event time series. Shape (2, n_events).
        y :
            Observed spike time series. Shape (3, n_spikes).
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

    log_lambda_y = log_likelihood_scan(
        X,
        y,
        params,
        inverse_link_function,
        n_basis_funcs,
        max_window,
        scan_size,
        eval_function,
        compute_lam_tilde_single,
        log=True,
    )

    mc_samples = draw_mc_sample(
        X,
        M_samples,
        random_key,
        recording_time,
        M_grid,
    )
    mc_estimate = log_likelihood_scan(
        X,
        mc_samples,
        params,
        inverse_link_function,
        n_basis_funcs,
        max_window,
        scan_size,
        eval_function,
        compute_lam_tilde_all,
        log=False,
    )

    return ((recording_time.tot_length() / M_samples) * mc_estimate) - log_lambda_y

def _compute_loss(
        params_with_key: PPGLMParamsWithKey,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
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
            Row 0 contains event times; row 1 contains predictor ids (neuron ids, stimulus
            ids, etc.). Shape (2, n_events).
        y :
            Marked spike time series for the neurons being modeled. Row 0 contains spike
            times; row 1 contains neuron ids; row 2 contains integer indices into X.
            Shape (3, n_spikes).

        Returns
        -------
        :
            The model negative log-likelihood. Shape (1,).
    """

    key = params_with_key.random_key.astype(jnp.uint32)

    new_key, _ = jax.random.split(key)

    neg_ll = _negative_log_likelihood(X, y, params_with_key.params, new_key, *args, **kwargs)

    return neg_ll
