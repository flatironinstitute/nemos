import jax
import jax.numpy as jnp

from ..typing import DESIGN_INPUT_TYPE

from . import utils
from .params import PPGLMParams
from .validation import to_pp_glm_params

import numpy as np
import pynapple as nap
from .basis import RaisedCosineLogEval

def compute_lam_tilde(dts, weights, bias, eval_function):
    fx = eval_function(dts)
    return jnp.sum(fx[:, :, None] * weights, axis=(0, 1)) + bias

def _compute_event_ll(
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        params,
        inverse_link_function,
        n_basis_funcs,
        max_window,
        scan_size,
        eval_function,
):
    r"""
        compute the first PP log-likelihood term:

        \log\Phi(\sum_t \sum_{substack{t_s \in \\ \cX(t, W)}} \mathbf{w}_{n}^\top \mbphi(t - t_s))

        Performs a parallelized scan over all spike times in y and computes their contributions
        to the log likelihood.
    """

    weights, bias = params
    weights = utils.reshape_w(weights, n_basis_funcs)

    # body of the scan function
    def scan_fn(lam_s, i):
        spk_in_window = utils.slice_array(
            X, i[-1].astype(int), max_window
        )
        dts = i[0] - spk_in_window[0]
        lam_tilde = compute_lam_tilde(dts, weights[spk_in_window[1].astype(int), :, i[1].astype(int), None],
                                           bias[i[1].astype(int)], eval_function)
        lam_s += jnp.log(inverse_link_function(lam_tilde)).sum()
        return lam_s, None

    scan_vmap = jax.vmap(lambda idxs: jax.lax.scan(scan_fn, jnp.array(0), idxs), in_axes=0)

    shifted_spikes_array, padding = utils.reshape_for_vmap(y, scan_size)
    out, _ = scan_vmap(shifted_spikes_array)
    sub, _ = scan_vmap(padding[None, :])
    return jnp.sum(out) - jnp.sum(sub)

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

def _compute_mc_est(
        X: DESIGN_INPUT_TYPE,
        mc_samples: jnp.ndarray,
        params,
        inverse_link_function,
        n_basis_funcs,
        max_window,
        scan_size,
        eval_function,
):
    weights, bias = params
    weights = utils.reshape_w(weights, n_basis_funcs)

    def scan_fn(lam_s, i):
        spk_in_window = utils.slice_array(
            X, i[-1].astype(int), max_window
        )
        dts = i[0] - spk_in_window[0]
        lam_tilde = compute_lam_tilde(dts, weights[spk_in_window[1].astype(int)], bias, eval_function)
        lam_s += inverse_link_function(lam_tilde).sum()
        return lam_s, None

    scan_vmap = jax.vmap(lambda idxs: jax.lax.scan(scan_fn, jnp.array(0), idxs), in_axes=0)

    shifted_spikes_array, padding = utils.reshape_for_vmap(mc_samples, scan_size)
    out, _ = scan_vmap(shifted_spikes_array)
    sub, _ = scan_vmap(padding[None, :])
    return jnp.sum(out) - jnp.sum(sub)

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

    log_lam_y = _compute_event_ll(
        X,
        y,
        params,
        inverse_link_function,
        n_basis_funcs,
        max_window,
        scan_size,
        eval_function,
    )

    mc_samples = draw_mc_sample(
        X,
        M_samples,
        random_key,
        recording_time,
        M_grid,
    )
    mc_estimate = _compute_mc_est(
        X,
        mc_samples,
        params,
        inverse_link_function,
        n_basis_funcs,
        max_window,
        scan_size,
        eval_function,
    )

    return ((recording_time.tot_length() / M_samples) * mc_estimate) - log_lam_y

def _compute_loss(
        params_with_key: PPGLMParams,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        inverse_link_function,
        *args,
        **kwargs,
) -> jnp.ndarray:
    """Loss function for a given model to be optimized over."""

    params = (params_with_key.coef, params_with_key.intercept)
    key = params_with_key.random_key.astype(jnp.uint32)

    new_key, _ = jax.random.split(key)

    neg_ll = _negative_log_likelihood(X, y, params, new_key, inverse_link_function, *args, **kwargs)

    return neg_ll


n_neurons = 5
recording_time = nap.IntervalSet(0, 10)
M_samples = 100
inverse_link_function = jnp.exp
n_basis_funcs = 4
history_window = 0.01
scan_size = 1

M_grid = utils.build_sampling_grid(recording_time, M_samples)
eval_function = RaisedCosineLogEval(n_basis_funcs, history_window)

spike_times = np.sort(np.random.uniform(0, 10, 10000))
spike_ids = np.random.choice(np.arange(n_neurons), 10000)

X = jnp.vstack((spike_times, spike_ids))
y = jnp.vstack((X, jnp.zeros(spike_times.size)))
max_window = int(utils.compute_max_window_size(jnp.array([-history_window, 0]), X[0], X[0]))
X, y = utils.adjust_indices_and_spike_times(X, history_window, max_window, y)

params = (
    jnp.ones((n_neurons*n_basis_funcs, n_neurons)),
    jnp.zeros(n_neurons),
    jax.random.PRNGKey(0).astype(jnp.float64),
)
params = to_pp_glm_params(params)

loss = _compute_loss(
    params,
    X,
    y,
    inverse_link_function,
    M_samples=M_samples,
    M_grid=M_grid,
    recording_time=recording_time,
    n_basis_funcs=n_basis_funcs,
    scan_size=scan_size,
    max_window=max_window,
    eval_function=eval_function,
)