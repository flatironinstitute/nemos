from functools import partial

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def raised_cosine_log_eval(x, n_basis_funcs, history_window, width=2.0, time_scaling=50.0):
    """jax only raised cosine log."""
    last_peak = 1 - width / (n_basis_funcs + width - 1)
    peaks = jnp.linspace(0, last_peak, n_basis_funcs)
    delta = peaks[1] - peaks[0]

    x = x / history_window

    # this makes sure that the out of range are set to zero
    x = jnp.where(jnp.abs(x) > 1, 1, x)

    x = jnp.log(time_scaling * x + 1) / jnp.log(time_scaling + 1)

    basis_funcs = 0.5 * (
        jnp.cos(
            jnp.clip(
                jnp.pi * (x[:, None] - peaks[None]) / (delta * width),
                -jnp.pi,
                jnp.pi,
            )
        )
        + 1
    )
    return basis_funcs

def RaisedCosineLogEval(n_basis_funcs, history_window, width=2.0, time_scaling=50.0):
    eval_func = lambda dts: raised_cosine_log_eval(dts, n_basis_funcs, history_window, width, time_scaling)
    return eval_func