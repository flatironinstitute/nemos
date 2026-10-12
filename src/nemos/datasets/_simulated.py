from typing import Callable

import jax
import jax.numpy as jnp


def _tuned_rnd(
    rnd_fn: Callable,
    inv_link: Callable,
    *args,
    n_neurons: int = 10,
    n_samples: int = 100,
):
    peak = jnp.linspace(0, 1, n_neurons)
    sub, key = jax.random.split(jax.random.key(123))
    samples = jax.random.uniform(key, shape=(n_samples,))
    tun = inv_link(-10 * (samples[:, None] - peak[None]) ** 2)
    return samples[:, None], rnd_fn(sub, tun, *args).astype(float)


def tuned_poisson(n_neurons: int = 10, n_samples: int = 100):
    return _tuned_rnd(
        jax.random.poisson, jnp.exp, n_neurons=n_neurons, n_samples=n_samples
    )


def tuned_gaussian(n_neurons: int = 10, n_samples: int = 100):
    return _tuned_rnd(
        jax.random.normal, lambda x: x, n_neurons=n_neurons, n_samples=n_samples
    )


def tuned_gamma(n_neurons: int = 10, n_samples: int = 100):
    return _tuned_rnd(
        jax.random.gamma, jnp.exp, n_neurons=n_neurons, n_samples=n_samples
    )


def tuned_bernoulli(n_neurons: int = 10, n_samples: int = 100):
    return _tuned_rnd(
        jax.random.bernoulli,
        jax.lax.logistic,
        n_neurons=n_neurons,
        n_samples=n_samples,
    )
