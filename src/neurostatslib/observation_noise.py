import abc
from typing import Tuple

import jax
import jax.numpy as jnp
import jaxopt

from .base_class import _Base


class NoiseModel(_Base, abc.ABC):
    def __init__(self, inverse_link_function, **kwargs):
        super().__init__(**kwargs)
        self.inverse_link_function = inverse_link_function

    @abc.abstractmethod
    def log_likelihood(self, params, X, y):
        pass

    @abc.abstractmethod
    def emission_probability(self, rate, **kwargs):
        pass

    def _predict(self, params, X):
        Ws, bs = params
        return self.inverse_link_function(jnp.einsum("ik,tik->ti", Ws, X) + bs[None, :])


class PoissonNoiseModel(_Base, NoiseModel):
    def __init__(self, inverse_link_function, **kwargs):
        super().__init__(inverse_link_function=inverse_link_function, **kwargs)

    def log_likelihood(
        self,
        params: Tuple[jnp.ndarray, jnp.ndarray],
        X: jnp.ndarray,
        y: jnp.ndarray,
    ):
        predicted_firing_rates = jnp.clip(self._predict(params, X), a_min=10**-10)
        x = y * jnp.log(predicted_firing_rates)
        # see above for derivation of this.
        return jnp.mean(predicted_firing_rates - x)

    @staticmethod
    def emission_probability(
        key: jax.random.PRNGKey, firing_rate: jnp.ndarray
    ) -> jnp.ndarray:
        return jax.random.poisson(key, firing_rate)
