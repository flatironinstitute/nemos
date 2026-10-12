from typing import Callable

import jax
import jax.numpy as jnp

from ..basis import BSplineEval


def _tuned_rnd(
    rnd_fn: Callable,
    inv_link: Callable,
    random_key: jax.random.PRNGKey,
    *args,
    n_neurons: int = 1,
    sample_shape: int = (1000,),
):
    peak = jnp.linspace(0, 1, n_neurons) if n_neurons > 1 else jnp.array([0.5])
    sub, key = jax.random.split(random_key)
    samples = jax.random.uniform(key, shape=sample_shape)
    exp_axis = (0, *(i + 1 for i in range(1, len(sample_shape))))
    peak = jnp.expand_dims(peak, exp_axis)
    tun = inv_link(-10 * (samples[:, None] - peak) ** 2)
    X = BSplineEval(8, bounds=(0, 1)).compute_features(samples)
    y = rnd_fn(sub, tun, *args).astype(float)
    squeeze_axis = [i for i in range(1, y.ndim) if y.shape[i] == 1]
    y = jnp.squeeze(y, squeeze_axis)
    return X, y


def tuned_poisson(
    random_key=jax.random.PRNGKey(0), n_neurons: int = 1, n_samples: int = 100
):
    return _tuned_rnd(
        jax.random.poisson,
        lambda x: 10 * jnp.exp(x),
        random_key=random_key,
        n_neurons=n_neurons,
        sample_shape=(n_samples,),
    )


def tuned_gaussian(
    random_key=jax.random.PRNGKey(0), n_neurons: int = 1, n_samples: int = 100
):
    def gauss_fn(key, x, *args):
        return jax.random.normal(key, shape=(n_samples, 1)) + x

    return _tuned_rnd(
        gauss_fn,
        lambda x: x,
        random_key=random_key,
        n_neurons=n_neurons,
        sample_shape=(n_samples,),
    )


def tuned_gamma(
    random_key=jax.random.PRNGKey(0), n_neurons: int = 1, n_samples: int = 100
):
    return _tuned_rnd(
        jax.random.gamma,
        jnp.exp,
        random_key=random_key,
        n_neurons=n_neurons,
        sample_shape=(n_samples,),
    )


def tuned_bernoulli(
    random_key=jax.random.PRNGKey(0), n_neurons: int = 1, n_samples: int = 100
):
    return _tuned_rnd(
        jax.random.bernoulli,
        jax.lax.logistic,
        n_neurons=n_neurons,
        sample_shape=(n_samples,),
        random_key=random_key,
    )


def tuned_categorical(
    random_key=jax.random.PRNGKey(0),
    n_neurons: int = 1,
    n_samples: int = 100,
    n_classes=3,
):
    return _tuned_rnd(
        jax.random.categorical,
        jax.nn.softplus,
        random_key=random_key,
        n_neurons=n_neurons,
        sample_shape=(n_samples, n_classes),
    )
