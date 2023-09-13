import abc
from typing import Union

import jax
import jax.numpy as jnp

from .base_class import _Base

KeyArray = Union[jnp.ndarray, jax.random.PRNGKeyArray]

__all__ = ["PoissonNoiseModel"]


def __dir__():
    return __all__


class NoiseModel(_Base, abc.ABC):
    FLOAT_EPS = jnp.finfo(jnp.float32).eps

    def __init__(self, inverse_link_function, **kwargs):
        super().__init__(**kwargs)
        if not callable(inverse_link_function):
            raise ValueError("inverse_link_function must be a callable!")
        self.inverse_link_function = inverse_link_function
        self._scale = None

    @abc.abstractmethod
    def negative_log_likelihood(self, firing_rate, y):
        pass

    @staticmethod
    @abc.abstractmethod
    def emission_probability(
        key: KeyArray, predicted_rate: jnp.ndarray, **kwargs
    ) -> jnp.ndarray:
        pass

    @abc.abstractmethod
    def residual_deviance(self, predicted_rate: jnp.ndarray, spike_counts: jnp.ndarray):
        pass

    def pseudo_r2(self, predicted_rate: jnp.ndarray, y: jnp.ndarray):
        r"""Pseudo-R^2 calculation for a GLM.

        The Pseudo-R^2 metric gives a sense of how well the model fits the data,
        relative to a null (or baseline) model.

        Parameters
        ----------
        predicted_rate:
            The mean neural activity.
        y:
            The neural activity.

        Returns
        -------
        :
            The pseudo-$R^2$ of the model. A value closer to 1 indicates a better model fit,
            whereas a value closer to 0 suggests that the model doesn't improve much over the null model.

        """
        res_dev_t = self.residual_deviance(predicted_rate, y)
        resid_deviance = jnp.sum(res_dev_t**2)

        null_mu = jnp.ones(y.shape, dtype=jnp.float32) * y.mean()
        null_dev_t = self.residual_deviance(null_mu, y)
        null_deviance = jnp.sum(null_dev_t**2)

        return (null_deviance - resid_deviance) / null_deviance


class PoissonNoiseModel(NoiseModel):
    def __init__(self, inverse_link_function=jnp.exp):
        super().__init__(inverse_link_function=inverse_link_function)
        self._scale = 1

    def negative_log_likelihood(
        self,
        predicted_rate: jnp.ndarray,
        y: jnp.ndarray,
    ):
        predicted_firing_rates = jnp.clip(predicted_rate, a_min=self.FLOAT_EPS)
        x = y * jnp.log(predicted_firing_rates)
        # see above for derivation of this.
        return jnp.mean(predicted_firing_rates - x)

    @staticmethod
    def emission_probability(
        key: KeyArray, predicted_rate: jnp.ndarray, **kwargs
    ) -> jnp.ndarray:
        return jax.random.poisson(key, predicted_rate)

    def residual_deviance(
        self, predicted_rate: jnp.ndarray, spike_counts: jnp.ndarray
    ) -> jnp.ndarray:
        r"""Compute the residual deviance for a Poisson model.

        Parameters
        ----------
        predicted_rate:
            The predicted firing rates.
        spike_counts:
            The spike counts.

        Returns
        -------
        :
            The residual deviance of the model.

        Notes
        -----
        Deviance is a measure of the goodness of fit of a statistical model.
        For a Poisson model, the residual deviance is computed as:

        $$
        \begin{aligned}
            D(y\_{tn}, \hat{y}\_{tn}) &= 2 \left[ y\_{tn} \log\left(\frac{y\_{tn}}{\hat{y}\_{tn}}\right)
            - (y\_{tn} - \hat{y}\_{tn}) \right]\\\
            &= -2 \left( \text{LL}\left(y\_{tn} | \hat{y}\_{tn}\right) - \text{LL}\left(y\_{tn} | y\_{tn}\right)\right)
        \end{aligned}
        $$
        where $ y $ is the observed data, $ \hat{y} $ is the predicted data, and $\text{LL}$ is the model
        log-likelihood. Lower values of deviance indicate a better fit.

        """
        # this takes care of 0s in the log
        ratio = jnp.clip(spike_counts / predicted_rate, self.FLOAT_EPS, jnp.inf)
        resid_dev = 2 * (
            spike_counts * jnp.log(ratio) - (spike_counts - predicted_rate)
        )
        return resid_dev
