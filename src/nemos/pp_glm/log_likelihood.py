import jax
import jax.numpy as jnp

from ..typing import DESIGN_INPUT_TYPE

from . import utils
from .params import PPGLMParamsWithKey


def compute_lam_tilde_single(dts, i, spk_in_window, weights, bias, eval_function):
    w = weights[spk_in_window[1].astype(int), :, i[1].astype(int)]
    fx = eval_function(dts)
    return jnp.sum(fx * w, axis=(0, 1)) + bias[i[1].astype(int)]

def compute_lam_tilde_all(dts, i, spk_in_window, weights, bias, eval_function):
    # i is not used
    w = weights[spk_in_window[1].astype(int)]
    fx = eval_function(dts)
    return jnp.sum(fx[:, :, None] * w, axis=(0, 1)) + bias

def draw_mc_sample(
        X: DESIGN_INPUT_TYPE,
        M_samples,
        random_key,
        recording_time,
        M_grid,
):
    """draw sample time points for a Monte Carlo estimate of /int_0^T(lambda(t))dt"""
    dt = recording_time.tot_length() / M_samples
    epsilon_m = jax.random.uniform(random_key, shape=(M_samples,), minval=0.0, maxval=dt)
    tau_m = M_grid + epsilon_m
    tau_m_idx = jnp.searchsorted(X[0], tau_m)
    mc_spikes = jnp.vstack((tau_m, tau_m_idx))

    return mc_spikes

def log_likelihood_scan(
        X,
        y,
        params,
        inverse_link_function,
        n_basis_funcs,
        max_window,
        scan_size,
        eval_function,
        lam_tilde_function,
        log=True,
):
    optional_log = jnp.log if log else lambda x: x

    weights, bias = params.coef, params.intercept
    weights = utils.reshape_coef_for_scan(weights, n_basis_funcs)

    # body of the scan function
    def scan_fn(lam_s, i):
        spk_in_window = utils.slice_array(
            X, i[-1].astype(int), max_window
        )
        dts = i[0] - spk_in_window[0]
        lam_tilde = lam_tilde_function(dts, i, spk_in_window, weights, bias, eval_function)
        lam_s += optional_log(inverse_link_function(lam_tilde)).sum()
        return lam_s, None

    scan_vmap = jax.vmap(lambda idxs: jax.lax.scan(scan_fn, jnp.array(0), idxs), in_axes=0)

    reshaped_spikes_array, padding_val, padding_len = utils.reshape_input_for_scan(y, scan_size)
    out, _ = scan_vmap(reshaped_spikes_array)

    # compute padding contribution separately to subtract it
    padding_lam = scan_fn(jnp.array(0.), padding_val)[0] * padding_len

    return jnp.sum(out) - padding_lam


def _negative_log_likelihood(
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        params,
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
    computes Poisson point process negative log-likelihood with MC estimate of the CIF

    $\sum_{k=1}^K \log \lambda(y_k) - \frac{T}{M} \sum_{m=1}^M \lambda(\tau_m)$
    """

    log_lam_y = log_likelihood_scan(
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

    return ((recording_time.tot_length() / M_samples) * mc_estimate) - log_lam_y

def _compute_loss(
        params_with_key: PPGLMParamsWithKey,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        inverse_link_function,
        *args,
        **kwargs,
) -> jnp.ndarray:
    """Loss function for a given model to be optimized over."""

    key = params_with_key.random_key.astype(jnp.uint32)

    new_key, _ = jax.random.split(key)

    neg_ll = _negative_log_likelihood(X, y, params_with_key.params, new_key, inverse_link_function, *args, **kwargs)

    return neg_ll
