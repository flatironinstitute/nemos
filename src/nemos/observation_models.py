"""Observation model classes for GLMs."""

import abc
from typing import Callable, Literal, Union

import jax
import jax.numpy as jnp
from numpy.typing import NDArray

from . import utils
from .base_class import Base

__all__ = ["PoissonObservations", "GammaObservations"]


def __dir__():
    return __all__


class Observations(Base, abc.ABC):
    """
    Abstract observation model class for neural data processing.

    This is an abstract base class used to implement observation models for neural data.
    Specific observation models that inherit from this class should define their versions
    of the abstract methods such as :meth:`~nemos.observation_models.Observations.log_likelihood`,
    :meth:`~nemos.observation_models.Observations.sample_generator`, and
    :meth:`~nemos.observation_models.Observations.deviance`.

    Attributes
    ----------
    inverse_link_function :
        A function that transforms a set of predictors to the domain of the model parameter.

    See Also
    --------
    :class:`~nemos.observation_models.PoissonObservations`
        A specific implementation of a observation model using the Poisson distribution.
    :class:`~nemos.observation_models.GammaObservations`
        A specific implementation of a observation model using the Gamma distribution.
    """

    def __init__(self, inverse_link_function: Callable, **kwargs):
        super().__init__(**kwargs)
        self.inverse_link_function = inverse_link_function
        self.scale = 1.0

    def __repr__(self):
        return utils.format_repr(self, use_name_keys=["inverse_link_function"])

    @property
    def inverse_link_function(self):
        """Getter for the inverse link function for the model."""
        return self._inverse_link_function

    @inverse_link_function.setter
    def inverse_link_function(self, inverse_link_function: Callable):
        """Setter for the inverse link function for the model."""
        self.check_inverse_link_function(inverse_link_function)
        self._inverse_link_function = inverse_link_function

    @property
    def scale(self):
        """Getter for the scale parameter of the model."""
        return self._scale

    @scale.setter
    def scale(self, value: Union[int, float, jnp.ndarray]):
        """Setter for the scale parameter of the model."""
        try:
            self._scale = float(value)
        except Exception:
            raise ValueError("The `scale` parameter must be of numeric type.")

    @staticmethod
    def check_inverse_link_function(inverse_link_function: Callable):
        """
        Check if the provided inverse_link_function is usable.

        This function verifies if the inverse link function:

        1. Is callable
        2. Returns a jax.numpy.ndarray
        3. Is differentiable (via jax)

        Parameters
        ----------
        inverse_link_function :
            The function to be checked.

        Raises
        ------
        TypeError
            If the function is not callable, does not return a jax.numpy.ndarray,
            or is not differentiable.
        """
        # check that it's callable
        if not callable(inverse_link_function):
            raise TypeError("The `inverse_link_function` function must be a Callable!")

        # check if the function returns a jax array for a 1D array
        array_out = inverse_link_function(jnp.array([1.0, 2.0, 3.0]))
        if not isinstance(array_out, jnp.ndarray):
            raise TypeError(
                "The `inverse_link_function` must return a jax.numpy.ndarray!"
            )

        # Optionally: Check for scalar input
        scalar_out = inverse_link_function(1.0)
        if not isinstance(scalar_out, (jnp.ndarray, float, int)):
            raise TypeError(
                "The `inverse_link_function` must handle scalar inputs correctly and return a scalar or a "
                "jax.numpy.ndarray!"
            )

        # check for autodiff
        try:
            gradient_fn = jax.grad(inverse_link_function)
            gradient_fn(1.0)
        except Exception as e:
            raise TypeError(
                f"The `inverse_link_function` function cannot be differentiated. Error: {e}"
            )

    @abc.abstractmethod
    def _negative_log_likelihood(
        self, y, predicted_rate, aggregate_sample_scores: Callable = jnp.mean
    ):
        r"""Compute the observation model negative log-likelihood.

        This computes the negative log-likelihood of the predicted rates
        for the observed neural activity up to a constant.

        Parameters
        ----------
        y :
            The target activity to compare against. Shape (n_time_bins, ), or (n_time_bins, n_neurons)..
        predicted_rate :
            The predicted rate of the current model. Shape (n_time_bins, ), or (n_time_bins, n_neurons)..

        Returns
        -------
        :
            The negative log-likehood. Shape (1,).
        """
        pass

    @abc.abstractmethod
    def log_likelihood(
        self,
        y: jnp.ndarray,
        predicted_rate: jnp.ndarray,
        scale: Union[float, jnp.ndarray] = 1.0,
        aggregate_sample_scores: Callable = jnp.mean,
    ):
        r"""Compute the observation model log-likelihood.

        This computes the log-likelihood of the predicted rates
        for the observed neural activity including the normalization constant

        Parameters
        ----------
        y :
            The target activity to compare against. Shape (n_time_bins, ), or (n_time_bins, n_neurons).
        predicted_rate :
            The predicted rate of the current model. Shape (n_time_bins, ), or (n_time_bins, n_neurons).
        scale :
            The scale parameter of the model
        aggregate_sample_scores :
            Function that aggregates the log-likelihood of each sample.

        Returns
        -------
        :
            The log-likehood. Shape (1,).
        """
        pass

    @abc.abstractmethod
    def sample_generator(
        self,
        key: jax.Array,
        predicted_rate: jnp.ndarray,
        scale: Union[float, jnp.ndarray] = 1.0,
    ) -> jnp.ndarray:
        """
        Sample from the estimated distribution.

        This method generates random numbers from the desired distribution based on the given
        `predicted_rate`.

        Parameters
        ----------
        key :
            Random key used for the generation of random numbers in JAX.
        predicted_rate :
            Expected rate of the distribution. Shape (n_time_bins, ), or (n_time_bins, n_neurons)..
        scale:
            Scale parameter for the distribution.

        Returns
        -------
        :
            Random numbers generated from the observation model with `predicted_rate`.
        """
        pass

    @abc.abstractmethod
    def deviance(
        self,
        spike_counts: jnp.ndarray,
        predicted_rate: jnp.ndarray,
        scale: Union[float, jnp.ndarray] = 1.0,
    ):
        r"""Compute the residual deviance for the observation model.

        Parameters
        ----------
        spike_counts:
            The spike counts. Shape ``(n_time_bins, )`` or ``(n_time_bins, n_neurons)`` for population models.
        predicted_rate:
            The predicted firing rates. Shape ``(n_time_bins, )`` or ``(n_time_bins, n_neurons)`` for population models.
        scale:
            Scale parameter of the model.

        Returns
        -------
        :
            The residual deviance of the model.
        """
        pass

    @abc.abstractmethod
    def estimate_scale(
        self,
        y: jnp.ndarray,
        predicted_rate: jnp.ndarray,
        dof_resid: Union[float, jnp.ndarray],
    ) -> Union[float, jnp.ndarray]:
        r"""Estimate the scale parameter for the model.

        This method estimates the scale parameter, often denoted as :math:`\phi`, which determines the dispersion
        of an exponential family distribution. The probability density function (pdf) for such a distribution
        is generally expressed as
        :math:`f(x; \theta, \phi) \propto \exp \left(a(\phi)\left(  y\theta - \mathcal{k}(\theta) \right)\right)`.

        The relationship between variance and the scale parameter is given by:

        .. math::
           \text{var}(Y) = \frac{V(\mu)}{a(\phi)}.

        The scale parameter, :math:`\phi`, is necessary for capturing the variance of the data accurately.

        Parameters
        ----------
        y :
            Observed activity.
        predicted_rate :
            The predicted rate values.
        dof_resid :
            The DOF of the residual.
        """
        pass

    def pseudo_r2(
        self,
        y: jnp.ndarray,
        predicted_rate: jnp.ndarray,
        score_type: Literal[
            "pseudo-r2-McFadden", "pseudo-r2-Cohen"
        ] = "pseudo-r2-McFadden",
        scale: Union[float, jnp.ndarray, NDArray] = 1.0,
        aggregate_sample_scores: Callable = jnp.mean,
    ) -> jnp.ndarray:
        r"""Pseudo-:math:`R^2` calculation for a GLM.

        Compute the pseudo-:math:`R^2` metric for the GLM, as defined by McFadden et al. [1]_
        or by Cohen et al. [2]_.

        This metric evaluates the goodness-of-fit of the model relative to a null (baseline) model that assumes a
        constant mean for the observations. While the pseudo-:math:`R^2` is bounded between 0 and 1 for the
        training set, it can yield negative values on out-of-sample data, indicating potential over-fitting.

        Parameters
        ----------
        y:
            The neural activity. Expected shape: ``(n_time_bins, )``
        predicted_rate:
            The mean neural activity. Expected shape: ``(n_time_bins, )``
        score_type:
            The pseudo-:math:`R^2` type.
        scale:
            The scale parameter of the model.

        Returns
        -------
        :
            The pseudo-:math:`R^2` of the model. A value closer to 1 indicates a better model fit,
            whereas a value closer to 0 suggests that the model doesn't improve much over the null model.

        Notes
        -----
        - The McFadden pseudo-:math:`R^2` is given by:

          .. math::
                R^2_{\text{mcf}} = 1 - \frac{\log(L_{M})}{\log(L_0)}.

          *Equivalent to statsmodels*
          `GLMResults.pseudo_rsquared(kind='mcf') <https://www.statsmodels.org/dev/generated/statsmodels.genmod.
          generalized_linear_model.GLMResults.pseudo_rsquared.html>`_ .

        - The Cohen pseudo-:math:`R^2` is given by:

          .. math::
               \begin{aligned}
               R^2_{\text{Cohen}} &= \frac{D_0 - D_M}{D_0} \\\
               &= 1 - \frac{\log(L_s) - \log(L_M)}{\log(L_s)-\log(L_0)},
               \end{aligned}

          where :math:`L_M`, :math:`L_0` and :math:`L_s` are the likelihood of the fitted model, the null model (a
          model with only the intercept term), and the saturated model (a model with one parameter per
          sample, i.e. the maximum value that the likelihood could possibly achieve). :math:`D_M` and :math:`D_0` are
          the model and the null deviance, :math:`D_i = -2 \left[ \log(L_s) - \log(L_i) \right]` for :math:`i=M,0`.

        References
        ----------
        .. [1] McFadden D (1979). Quantitative methods for analysing travel behavior of individuals: Some recent
               developments. In D. A. Hensher & P. R. Stopher (Eds.), *Behavioural travel modelling* (pp. 279-318).
               London: Croom Helm.

        .. [2] Jacob Cohen, Patricia Cohen, Steven G. West, Leona S. Aiken.
               *Applied Multiple Regression/Correlation Analysis for the Behavioral Sciences*.
               3rd edition. Routledge, 2002. p.502. ISBN 978-0-8058-2223-6. (May 2012)
        """
        if score_type == "pseudo-r2-McFadden":
            pseudo_r2 = self._pseudo_r2_mcfadden(
                y,
                predicted_rate,
                scale=scale,
                aggregate_sample_scores=aggregate_sample_scores,
            )
        elif score_type == "pseudo-r2-Cohen":
            pseudo_r2 = self._pseudo_r2_cohen(
                y, predicted_rate, aggregate_sample_scores=aggregate_sample_scores
            )
        else:
            raise NotImplementedError(f"Score {score_type} not implemented!")
        return pseudo_r2

    def _pseudo_r2_cohen(
        self,
        y: jnp.ndarray,
        predicted_rate: jnp.ndarray,
        aggregate_sample_scores: Callable = jnp.mean,
    ) -> jnp.ndarray:
        r"""Cohen's pseudo-:math:`R^2`.

        Compute the pseudo-:math:`R^2` metric as defined by Cohen et al. (2002). See
        :meth:`nemos.observation_models.Observations.pseudo_r2` for additional information.

        Parameters
        ----------
        y:
            The neural activity. Expected shape: ``(n_time_bins, )``.
        predicted_rate:
            The mean neural activity. Expected shape: ``(n_time_bins, )``

        Returns
        -------
        :
            The pseudo-:math:`R^2` of the model. A value closer to 1 indicates a better model fit,
            whereas a value closer to 0 suggests that the model doesn't improve much over the null model.
        """
        model_dev_t = self.deviance(y, predicted_rate)
        model_deviance = aggregate_sample_scores(model_dev_t)

        null_mu = jnp.ones(y.shape, dtype=jnp.float32) * jnp.mean(y, axis=0)
        null_dev_t = self.deviance(y, null_mu)
        null_deviance = aggregate_sample_scores(null_dev_t)
        return (null_deviance - model_deviance) / null_deviance

    def _pseudo_r2_mcfadden(
        self,
        y: jnp.ndarray,
        predicted_rate: jnp.ndarray,
        scale: Union[float, jnp.ndarray] = 1.0,
        aggregate_sample_scores: Callable = jnp.mean,
    ):
        """
        McFadden's pseudo-:math:`R^2`.

        Compute the pseudo-:math:`R^2` metric as defined by McFadden et al. (1979). See
        :meth:`nemos.observation_models.Observations.pseudo_r2` for additional information.

        Parameters
        ----------
        y:
            The neural activity. Expected shape: (n_time_bins, ), or (n_time_bins, n_neurons).
        predicted_rate:
            The mean neural activity. Expected shape: (n_time_bins, ), or (n_time_bins, n_neurons).
        scale:
            The scale parameter of the model.

        Returns
        -------
        :
            The pseudo-:math:`R^2` of the model. A value closer to 1 indicates a better model fit,
            whereas a value closer to 0 suggests that the model doesn't improve much over the null model.
        """
        mean_y = jnp.ones(y.shape) * y.mean(axis=0)
        ll_null = self.log_likelihood(
            y, mean_y, scale=scale, aggregate_sample_scores=aggregate_sample_scores
        )
        ll_model = self.log_likelihood(
            y,
            predicted_rate,
            scale=scale,
            aggregate_sample_scores=aggregate_sample_scores,
        )
        return 1 - ll_model / ll_null


class PoissonObservations(Observations):
    """
    Model observations as Poisson random variables.

    The PoissonObservations is designed to model the observed spike counts based on a Poisson distribution
    with a given rate. It provides methods for computing the negative log-likelihood, generating samples,
    and computing the residual deviance for the given spike count data.

    Attributes
    ----------
    inverse_link_function :
        A function that maps the predicted rate to the domain of the Poisson parameter. Defaults to ``jax.numpy.exp``.

    """

    def __init__(self, inverse_link_function=jnp.exp):
        super().__init__(inverse_link_function=inverse_link_function)
        self.scale = 1.0

    def _negative_log_likelihood(
        self,
        y: jnp.ndarray,
        predicted_rate: jnp.ndarray,
        aggregate_sample_scores: Callable = jnp.mean,
    ) -> jnp.ndarray:
        r"""Compute the Poisson negative log-likelihood.

        This computes the Poisson negative log-likelihood of the predicted rates
        for the observed spike counts up to a constant.

        Parameters
        ----------
        y :
            The target spikes to compare against. Shape (n_time_bins, ), or (n_time_bins, n_neurons).
        predicted_rate :
            The predicted rate of the current model. Shape (n_time_bins, ), or (n_time_bins, n_neurons).

        Returns
        -------
        :
            The Poisson negative log-likehood. Shape (1,).

        Notes
        -----
        The formula for the Poisson mean log-likelihood is the following,

        .. math::
        \begin{aligned}
        \text{LL}(\hat{\lambda} | y) &= \frac{1}{T \cdot N} \sum_{n=1}^{N} \sum_{t=1}^{T}
        [y_{tn} \log(\hat{\lambda}_{tn}) - \hat{\lambda}_{tn} - \log({y_{tn}!})] \\\
        &= \frac{1}{T \cdot N} \sum_{n=1}^{N} \sum_{t=1}^{T} [y_{tn} \log(\hat{\lambda}_{tn}) -
        \hat{\lambda}_{tn} - \Gamma({y_{tn}+1})] \\\
        &= \frac{1}{T \cdot N} \sum_{n=1}^{N} \sum_{t=1}^{T} [y_{tn} \log(\hat{\lambda}_{tn}) -
        \hat{\lambda}_{tn}] + \\text{const}
        \end{aligned}

        Because :math:`\Gamma(k+1)=k!`, see `wikipedia <https://en.wikipedia.org/wiki/Gamma_function>` for explanation.

        The :math:`\log({y_{tn}!})` term is not a function of the parameters and can be disregarded
        when computing the loss-function. This is why we incorporated it into the `const` term.
        """
        predicted_rate = jnp.clip(
            predicted_rate, min=jnp.finfo(predicted_rate.dtype).eps
        )
        x = y * jnp.log(predicted_rate)
        # see above for derivation of this.
        return aggregate_sample_scores(predicted_rate - x)

    def log_likelihood(
        self,
        y: jnp.ndarray,
        predicted_rate: jnp.ndarray,
        scale: Union[float, jnp.ndarray] = 1.0,
        aggregate_sample_scores: Callable = jnp.mean,
    ):
        r"""Compute the Poisson negative log-likelihood.

        This computes the Poisson negative log-likelihood of the predicted rates
        for the observed spike counts up to a constant.

        Parameters
        ----------
        y :
            The target spikes to compare against. Shape ``(n_time_bins, )``, or ``(n_time_bins, n_neurons)``.
        predicted_rate :
            The predicted rate of the current model. Shape ``(n_time_bins, )``, or ``(n_time_bins, n_neurons)``.
        scale :
            The scale parameter of the model.
        aggregate_sample_scores :
            Function that aggregates the log-likelihood of each sample.

        Returns
        -------
        :
            The Poisson negative log-likehood. Shape (1,).

        Notes
        -----
        The formula for the Poisson mean log-likelihood is the following,

        .. math::
            \begin{aligned}
            \text{LL}(\hat{\lambda} | y) &= \frac{1}{T \cdot N} \sum_{n=1}^{N} \sum_{t=1}^{T}
            [y_{tn} \log(\hat{\lambda}_{tn}) - \hat{\lambda}_{tn} - \log({y_{tn}!})] \\\
            &= \frac{1}{T \cdot N} \sum_{n=1}^{N} \sum_{t=1}^{T} [y_{tn} \log(\hat{\lambda}_{tn}) -
            \hat{\lambda}_{tn} - \Gamma({y_{tn}+1})] \\\
            &= \frac{1}{T \cdot N} \sum_{n=1}^{N} \sum_{t=1}^{T} [y_{tn} \log(\hat{\lambda}_{tn}) -
            \hat{\lambda}_{tn}] + \text{const}
            \end{aligned}


        Because :math:`\Gamma(k+1)=k!`, see `wikipedia <https://en.wikipedia.org/wiki/Gamma_function>`_ for explanation.

        The :math:`\log({y_{tn}!})` term is not a function of the parameters and can be disregarded
        when computing the loss-function. This is why we incorporated it into the `const` term.
        """
        nll = self._negative_log_likelihood(y, predicted_rate, aggregate_sample_scores)
        return -nll - aggregate_sample_scores(jax.scipy.special.gammaln(y + 1))

    def sample_generator(
        self,
        key: jax.Array,
        predicted_rate: jnp.ndarray,
        scale: Union[float, jnp.ndarray] = 1.0,
    ) -> jnp.ndarray:
        """
        Sample from the Poisson distribution.

        This method generates random numbers from a Poisson distribution based on the given
        `predicted_rate`.

        Parameters
        ----------
        key :
            Random key used for the generation of random numbers in JAX.
        predicted_rate :
            Expected rate (lambda) of the Poisson distribution. Shape ``(n_time_bins, )``, or
            ``(n_time_bins, n_neurons)``.
        scale :
            Scale parameter. For Poisson should be equal to 1.

        Returns
        -------
        jnp.ndarray
            Random numbers generated from the Poisson distribution based on the `predicted_rate`.
        """
        return jax.random.poisson(key, predicted_rate)

    def deviance(
        self,
        spike_counts: jnp.ndarray,
        predicted_rate: jnp.ndarray,
        scale: Union[float, jnp.ndarray] = 1.0,
    ) -> jnp.ndarray:
        r"""Compute the residual deviance for a Poisson model.

        Parameters
        ----------
        spike_counts:
            The spike counts. Shape ``(n_time_bins, )`` or ``(n_time_bins, n_neurons)`` for population models.
        predicted_rate:
            The predicted firing rates. Shape ``(n_time_bins, )``  or ``(n_time_bins, n_neurons)`` for
            population models.
        scale:
            Scale parameter of the model.

        Returns
        -------
        :
            The residual deviance of the model.

        Notes
        -----
        The deviance is a measure of the goodness of fit of a statistical model.
        For a Poisson model, the residual deviance is computed as:

        .. math::
            \begin{aligned}
                D(y_{tn}, \hat{y}_{tn}) &= 2 \left[ y_{tn} \log\left(\frac{y_{tn}}{\hat{y}_{tn}}\right)
                - (y_{tn} - \hat{y}_{tn}) \right]\\\
                &= 2 \left( \text{LL}\left(y_{tn} | y_{tn}\right) - \text{LL}\left(y_{tn} | \hat{y}_{tn}\right)\right)
            \end{aligned}

        where :math:`y` is the observed data, :math:`\hat{y}` is the predicted data, and :math:`\text{LL}` is
        the model log-likelihood. Lower values of deviance indicate a better fit.
        """
        # this takes care of 0s in the log
        ratio = jnp.clip(
            spike_counts / predicted_rate, jnp.finfo(predicted_rate.dtype).eps, jnp.inf
        )
        deviance = 2 * (spike_counts * jnp.log(ratio) - (spike_counts - predicted_rate))
        return deviance

    def estimate_scale(
        self,
        y: jnp.ndarray,
        predicted_rate: jnp.ndarray,
        dof_resid: Union[float, jnp.ndarray],
    ) -> Union[float, jnp.ndarray]:
        r"""
        Assign 1 to the scale parameter of the Poisson model.

        For the Poisson exponential family distribution, the scale parameter :math:`\phi` is always 1.
        This property is consistent with the fact that the variance equals the mean in a Poisson distribution.
        As given in the general exponential family expression:

        .. math::
            \text{var}(Y) = \frac{V(\mu)}{a(\phi)},

        for the Poisson family, it simplifies to :math:`\text{var}(Y) = \mu` since :math:`a(\phi) = 1`
        and :math:`V(\mu) = \mu`.

        Parameters
        ----------
        y :
            Observed spike counts.
        predicted_rate :
            The predicted rate values. This is not used in the Poisson model for estimating scale,
            but is retained for compatibility with the abstract method signature.
        dof_resid :
            The DOF of the residuals.
        """
        return jnp.ones_like(jnp.atleast_1d(y[0]))


class GammaObservations(Observations):
    """
    Model observations as Gamma random variables.

    The GammaObservations is designed to model the observed spike counts based on a Gamma distribution
    with a given rate. It provides methods for computing the negative log-likelihood, generating samples,
    and computing the residual deviance for the given spike count data.

    Attributes
    ----------
    inverse_link_function :
        A function that maps the predicted rate to the domain of the Poisson parameter. Defaults to jnp.exp.

    """

    def __init__(self, inverse_link_function=lambda x: jnp.power(x, -1)):
        super().__init__(inverse_link_function=inverse_link_function)
        self.scale = 1.0

    def _negative_log_likelihood(
        self,
        y: jnp.ndarray,
        predicted_rate: jnp.ndarray,
        aggregate_sample_scores: Callable = jnp.mean,
    ) -> jnp.ndarray:
        r"""Compute the Gamma negative log-likelihood.

        This computes the Gamma negative log-likelihood of the predicted rates
        for the observed neural activity up to a constant.

        Parameters
        ----------
        y :
            The target activity to compare against. Shape (n_time_bins, ), or (n_time_bins, n_neurons).
        predicted_rate :
            The predicted rate of the current model. Shape (n_time_bins, ), or (n_time_bins, n_neurons).
        aggregate_sample_scores :
            Function that aggregates the log-likelihood of each sample.

        Returns
        -------
        :
            The Gamma negative log-likelihood. Shape (1,).

        """
        predicted_rate = jnp.clip(
            predicted_rate, min=jnp.finfo(predicted_rate.dtype).eps
        )
        x = jnp.power(-predicted_rate, -1)
        # see above for derivation of this.
        return -aggregate_sample_scores(y * x + jnp.log(-x))

    def log_likelihood(
        self,
        y: jnp.ndarray,
        predicted_rate: jnp.ndarray,
        scale: Union[float, jnp.ndarray] = 1.0,
        aggregate_sample_scores: Callable = jnp.mean,
    ):
        r"""Compute the Gamma negative log-likelihood.

        This computes the Gamma negative log-likelihood of the predicted rates
        for the observed neural activity including the normalization constant.

        Parameters
        ----------
        y :
            The target activity to compare against. Shape (n_time_bins, ) or (n_time_bins, n_neurons).
        predicted_rate :
            The predicted rate of the current model. Shape (n_time_bins, ) or (n_time_bins, n_neurons).
        scale :
            The scale parameter of the model.
        aggregate_sample_scores :
            Function that aggregates the log-likelihood of each sample.

        Returns
        -------
        :
            The Gamma negative log-likelihood. Shape (1,).

        """
        k = 1 / scale
        norm = (
            (k - 1) * jnp.mean(jnp.log(y))
            + k * jnp.log(k)
            - jax.scipy.special.gammaln(k)
        )
        return aggregate_sample_scores(
            norm - k * self._negative_log_likelihood(y, predicted_rate, lambda x: x)
        )

    def sample_generator(
        self,
        key: jax.Array,
        predicted_rate: jnp.ndarray,
        scale: Union[float, jnp.ndarray] = 1.0,
    ) -> jnp.ndarray:
        """
        Sample from the Gamma distribution.

        This method generates random numbers from a Gamma distribution based on the given
        `predicted_rate` and `scale`.

        Parameters
        ----------
        key :
            Random key used for the generation of random numbers in JAX.
        predicted_rate :
            Expected rate (lambda) of the Poisson distribution. Shape (n_time_bins, ), or (n_time_bins, n_neurons)..
        scale:
            The scale parameter for the distribution.

        Returns
        -------
        jnp.ndarray
            Random numbers generated from the Gamma distribution based on the `predicted_rate` and the `scale`.
        """
        return jax.random.gamma(key, predicted_rate / scale) * scale

    def deviance(
        self,
        neural_activity: jnp.ndarray,
        predicted_rate: jnp.ndarray,
        scale: Union[float, jnp.ndarray] = 1.0,
    ) -> jnp.ndarray:
        r"""Compute the residual deviance for a Gamma model.

        Parameters
        ----------
        neural_activity:
            The spike coun activity. Shape (n_time_bins, ) or (n_time_bins, n_neurons) for population models.
        predicted_rate:
            The predicted firing rates. Shape (n_time_bins, ) or (n_time_bins, n_neurons) for population models.
        scale:
            Scale parameter of the model.

        Returns
        -------
        :
            The residual deviance of the model.

        Notes
        -----
        The deviance is a measure of the goodness of fit of a statistical model.
        For a Gamma model, the residual deviance is computed as:

        .. math::
            \begin{aligned}
                D(y_{tn}, \hat{y}_{tn}) &=  2 \left[ -\log \frac{ y_{tn}}{\hat{y}_{tn}} +  \frac{y_{tn} -
                \hat{y}_{tn}}{\hat{y}_{tn}}\right]\\\
                &= 2 \left( \text{LL}\left(y_{tn} | y_{tn}\right) - \text{LL}\left(y_{tn} | \hat{y}_{tn}\right) \right)
            \end{aligned}

        where :math:`y` is the observed data, :math:`\hat{y}` is the predicted data, and :math:`\text{LL}` is the model
        log-likelihood. Lower values of deviance indicate a better fit.

        """
        y_mu = jnp.clip(neural_activity / predicted_rate, min=jnp.finfo(float).eps)
        resid_dev = 2 * (
            -jnp.log(y_mu) + (neural_activity - predicted_rate) / predicted_rate
        )
        return resid_dev / scale

    def estimate_scale(
        self,
        y: jnp.ndarray,
        predicted_rate: jnp.ndarray,
        dof_resid: Union[float, jnp.ndarray],
    ) -> Union[float, jnp.ndarray]:
        r"""
        Estimate the scale of the model based on the GLM residuals.

        For :math:`y \sim \Gamma` the scale is equal to,

        .. math::
            \Phi = \frac{\text{Var(y)}}{V(\mu)}

        with :math:`V(\mu) = \mu^2`.

        Therefore, the scale can be estimated as the ratio of the sample variance to the squared rate.

        Parameters
        ----------
        y :
            Observed neural activity.
        predicted_rate :
            The predicted rate values. This is not used in the Poisson model for estimating scale,
            but is retained for compatibility with the abstract method signature.
        dof_resid :
            The DOF of the residuals.

        Returns
        -------
        :
            The scale parameter. If predicted_rate is ``(n_samples, n_neurons)``, this method will return a
            scale for each neuron.
        """
        predicted_rate = jnp.clip(
            predicted_rate, min=jnp.finfo(predicted_rate.dtype).eps
        )
        resid = jnp.power(y - predicted_rate, 2)
        return (
            jnp.sum(resid * jnp.power(predicted_rate, -2), axis=0) / dof_resid
        )  # pearson residuals


def check_observation_model(observation_model):
    r"""
    Check the attributes of an observation model for compliance.

    This function ensures that the observation model has the required attributes and that each
    attribute is a callable function. Additionally, it checks if these functions return
    jax.numpy.ndarray objects, and in the case of 'inverse_link_function', whether it is
    differentiable.

    Parameters
    ----------
    observation_model : object
        An instance of an observation model that should have specific attributes.

    Raises
    ------
    AttributeError
        If the `observation_model` does not have one of the required attributes.

    TypeError
        If an attribute is not a callable function.
    TypeError
        If a function does not return a jax.numpy.ndarray.
    TypeError
        If 'inverse_link_function' is not differentiable.

    Examples
    --------
    >>> class MyObservationModel:
    ...     def inverse_link_function(self, x):
    ...         return jax.scipy.special.expit(x)
    ...     def _negative_log_likelihood(self, params, y_true, aggregate_sample_scores=jnp.mean):
    ...         return -aggregate_sample_scores(y_true * jax.scipy.special.logit(params) + \
    ...                 (1 - y_true) * jax.scipy.special.logit(1 - params))
    ...     def pseudo_r2(self, params, y_true, aggregate_sample_scores=jnp.mean):
    ...         return 1 - (self._negative_log_likelihood(y_true, params, aggregate_sample_scores) /
    ...                     jnp.sum((y_true - y_true.mean()) ** 2))
    ...     def sample_generator(self, key, params, scale=1.):
    ...         return jax.random.bernoulli(key, params)
    >>> model = MyObservationModel()
    >>> check_observation_model(model)  # Should pass without error if the model is correctly implemented.
    """
    # Define the checks to be made on each attribute
    checks = {
        "inverse_link_function": {
            "input": [jnp.array([1.0, 1.0, 1.0])],
            "test_differentiable": True,
            "test_preserve_shape": False,
        },
        "_negative_log_likelihood": {
            "input": [0.5 * jnp.array([1.0, 1.0, 1.0]), jnp.array([1.0, 1.0, 1.0])],
            "test_scalar_func": True,
        },
        "pseudo_r2": {
            "input": [0.5 * jnp.array([1.0, 1.0, 1.0]), jnp.array([1.0, 1.0, 1.0])],
            "test_scalar_func": True,
        },
        "sample_generator": {
            "input": [jax.random.key(123), 0.5 * jnp.array([1.0, 1.0, 1.0]), 1],
            "test_preserve_shape": True,
        },
    }

    # Perform checks for each attribute
    for attr_name, check_info in checks.items():
        # check if the observation model has the attribute
        utils.assert_has_attribute(observation_model, attr_name)

        # check if the attribute is a callable
        func = getattr(observation_model, attr_name)
        utils.assert_is_callable(func, attr_name)

        # check that the callable returns an array
        utils.assert_returns_ndarray(func, check_info["input"], attr_name)

        if check_info.get("test_differentiable"):
            utils.assert_differentiable(func, attr_name)

        if "test_preserve_shape" in check_info:
            index = int(check_info["test_preserve_shape"])
            utils.assert_preserve_shape(
                func, check_info["input"], attr_name, input_index=index
            )

        if check_info.get("test_scalar_func"):
            utils.assert_scalar_func(func, check_info["input"], attr_name)
