"""Noise model classes for GLMs."""

import abc
from typing import Callable, Union

import jax
import jax.numpy as jnp

from .base_class import _Base

KeyArray = Union[jnp.ndarray, jax.random.PRNGKeyArray]

__all__ = ["PoissonNoiseModel"]


def __dir__():
    return __all__


class NoiseModel(_Base, abc.ABC):
    """
    Abstract noise model class for neural data processing.

    This is an abstract base class used to implement noise models for neural data.
    Specific noise models that inherit from this class should define their versions
    of the abstract methods: negative_log_likelihood, emission_probability, and
    residual_deviance.

    Attributes
    ----------
    FLOAT_EPS :
        A small value used to ensure numerical stability. Set to the machine epsilon for float32.
    inverse_link_function :
        A function that transforms a set of predictors to the domain of the model parameter.

    See Also
    --------
    [PoissonNoiseModel](./#neurostatslib.noise_model.PoissonNoiseModel) : A specific implementation of a
    noise model using the Poisson distribution.
    """

    FLOAT_EPS = jnp.finfo(jnp.float32).eps

    def __init__(self, inverse_link_function: Callable, **kwargs):
        super().__init__(**kwargs)
        self._check_inverse_link_function(inverse_link_function)
        self._inverse_link_function = inverse_link_function
        self._scale = 1.

    @property
    def inverse_link_function(self):
        """Getter for the inverse link function for the model."""
        return self._inverse_link_function

    @inverse_link_function.setter
    def inverse_link_function(self, inverse_link_function: Callable):
        """Setter for the inverse link function for the model."""
        self._check_inverse_link_function(inverse_link_function)
        self._inverse_link_function = inverse_link_function

    @property
    def scale(self):
        """Getter for the scale parameter of the model."""
        return self._scale

    @scale.setter
    def scale(self, value: Union[int, float]):
        """Setter for the scale parameter of the model."""
        if not isinstance(value, (int, float)):
            raise ValueError("The `scale` parameter must be of numeric type.")
        self._scale = value

    @staticmethod
    def _check_inverse_link_function(inverse_link_function):
        if not callable(inverse_link_function):
            raise TypeError("The `inverse_link_function` function must be a Callable!")
        # check that the callable is in the jax namespace
        if not hasattr(inverse_link_function, "__module__"):
            raise TypeError(
                "The `inverse_link_function` must be from the `jax` namespace!"
            )
        elif not getattr(inverse_link_function, "__module__").startswith("jax"):
            raise TypeError(
                "The `inverse_link_function` must be from the `jax` namespace!"
            )

    @abc.abstractmethod
    def negative_log_likelihood(self, predicted_rate, y):
        r"""Compute the noise model negative log-likelihood.

        This computes the negative log-likelihood of the predicted rates
        for the observed neural activity up to a constant.

        Parameters
        ----------
        predicted_rate :
            The predicted rate of the current model. Shape (n_time_bins, n_neurons).
        y :
            The target activity to compare against. Shape (n_time_bins, n_neurons).

        Returns
        -------
        :
            The Poisson negative log-likehood. Shape (1,).
        """
        pass

    @abc.abstractmethod
    def emission_probability(
        self, key: KeyArray, predicted_rate: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calculate the emission of the noise model.

        This method generates random numbers from the desired distribution based on the given
        `predicted_rate`.

        Parameters
        ----------
        key :
            Random key used for the generation of random numbers in JAX.
        predicted_rate :
            Expected rate of the distribution. Shape (n_time_bins, n_neurons).

        Returns
        -------
        jnp.ndarray
            Random numbers generated from the desired distribution based on the `predicted_rate` scale parameter
            if needed.
        """
        pass

    @abc.abstractmethod
    def residual_deviance(self, predicted_rate: jnp.ndarray, spike_counts: jnp.ndarray):
        r"""Compute the residual deviance for a Poisson model.

        Parameters
        ----------
        predicted_rate:
            The predicted firing rates. Shape (n_time_bins, n_neurons).
        spike_counts:
            The spike counts. Shape (n_time_bins, n_neurons).

        Returns
        -------
        :
            The residual deviance of the model.
        """
        pass

    @abc.abstractmethod
    def estimate_scale(self, predicted_rate: jnp.ndarray) -> float:
        """Estimate the scale parameter for the model."""
        pass

    def pseudo_r2(self, predicted_rate: jnp.ndarray, y: jnp.ndarray):
        r"""Pseudo-R^2 calculation for a GLM.

        The Pseudo-R^2 metric gives a sense of how well the model fits the data,
        relative to a null (or baseline) model.

        Parameters
        ----------
        predicted_rate:
            The mean neural activity. Expected shape: (n_time_bins, n_neurons)
        y:
            The neural activity. Expected shape: (n_time_bins, n_neurons)

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
    """
    Poisson Noise Model class for spike count data.

    The PoissonNoiseModel is designed to model the observed spike counts based on a Poisson distribution
    with a given rate. It provides methods for computing the negative log-likelihood, emission probability,
    and residual deviance for the given spike count data.

    Attributes
    ----------
    inverse_link_function :
        A function that maps the predicted rate to the domain of the Poisson parameter. Defaults to jnp.exp.

    See Also
    --------
    [NoiseModel](./#neurostatslib.noise_model.NoiseModel) : Base class for noise models.
    """

    def __init__(self, inverse_link_function=jnp.exp):
        super().__init__(inverse_link_function=inverse_link_function)
        self._scale = 1

    def negative_log_likelihood(
        self,
        predicted_rate: jnp.ndarray,
        y: jnp.ndarray,
    ) -> jnp.ndarray:
        r"""Compute the Poisson negative log-likelihood.

        This computes the Poisson negative log-likelihood of the predicted rates
        for the observed spike counts up to a constant.

        The formula for the Poisson mean log-likelihood is the following,

        $$
        \begin{aligned}
        \text{LL}(\hat{\lambda} | y) &= \frac{1}{T \cdot N} \sum_{n=1}^{N} \sum_{t=1}^{T}
        [y\_{tn} \log(\hat{\lambda}\_{tn}) - \hat{\lambda}\_{tn} - \log({y\_{tn}!})] \\\
        &= \frac{1}{T \cdot N} \sum_{n=1}^{N} \sum_{t=1}^{T} [y\_{tn} \log(\hat{\lambda}\_{tn}) -
        \hat{\lambda}\_{tn} - \Gamma({y\_{tn}+1})] \\\
        &= \frac{1}{T \cdot N} \sum_{n=1}^{N} \sum_{t=1}^{T} [y\_{tn} \log(\hat{\lambda}\_{tn}) -
        \hat{\lambda}\_{tn}] + \\text{const}
        \end{aligned}
        $$

        Because $\Gamma(k+1)=k!$, see [wikipedia](https://en.wikipedia.org/wiki/Gamma_function) for example.

        Parameters
        ----------
        predicted_rate :
            The predicted rate of the current model. Shape (n_time_bins, n_neurons).
        y :
            The target spikes to compare against. Shape (n_time_bins, n_neurons).

        Returns
        -------
        :
            The Poisson negative log-likehood. Shape (1,).

        Notes
        -----
        The $\log({y\_{tn}!})$ term is not a function of the parameters and can be disregarded
        when computing the loss-function. This is why we incorporated it into the `const` term.
        """
        predicted_firing_rates = jnp.clip(predicted_rate, a_min=self.FLOAT_EPS)
        x = y * jnp.log(predicted_firing_rates)
        # see above for derivation of this.
        return jnp.mean(predicted_firing_rates - x)

    def emission_probability(
        self, key: KeyArray, predicted_rate: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Calculate the emission probability using a Poisson distribution.

        This method generates random numbers from a Poisson distribution based on the given
        `predicted_rate`.

        Parameters
        ----------
        key :
            Random key used for the generation of random numbers in JAX.
        predicted_rate :
            Expected rate (lambda) of the Poisson distribution. Shape (n_time_bins, n_neurons).

        Returns
        -------
        jnp.ndarray
            Random numbers generated from the Poisson distribution based on the `predicted_rate`.
        """
        return jax.random.poisson(key, predicted_rate)

    def residual_deviance(
        self, predicted_rate: jnp.ndarray, spike_counts: jnp.ndarray
    ) -> jnp.ndarray:
        r"""Compute the residual deviance for a Poisson model.

        Parameters
        ----------
        predicted_rate:
            The predicted firing rates. Shape (n_time_bins, n_neurons).
        spike_counts:
            The spike counts. Shape (n_time_bins, n_neurons).

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

    def estimate_scale(self, predicted_rate: jnp.ndarray):
        """Assign 1 to the scale parameter of the Poisson model."""
        self.scale = 1.

