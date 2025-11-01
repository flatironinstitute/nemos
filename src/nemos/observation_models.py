"""Observation model classes for GLMs."""

import abc
from typing import Callable, Literal, Union

import jax
import jax.numpy as jnp
from numpy.typing import NDArray

from . import utils
from .base_class import Base

__all__ = [
    "PoissonObservations",
    "GammaObservations",
    "BernoulliObservations",
    "NegativeBinomialObservations",
]


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

    See Also
    --------
    :class:`~nemos.observation_models.PoissonObservations`
        A specific implementation of an observation model using the Poisson distribution.
    :class:`~nemos.observation_models.GammaObservations`
        A specific implementation of an observation model using the Gamma distribution.
    :class:`~nemos.observation_models.BernoulliObservations`
        A specific implementation of an observation model using the Bernoulli distribution.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scale = 1.0

    def __repr__(self):
        return utils.format_repr(self)

    @property
    @abc.abstractmethod
    def default_inverse_link_function(self):
        pass

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

    def likelihood(
        self,
        y: jnp.ndarray,
        predicted_rate: jnp.ndarray,
        scale: Union[float, jnp.ndarray] = 1.0,
        aggregate_sample_scores: Callable = jnp.mean,
    ):
        r"""Compute the observation model likelihood.

        This computes the likelihood of the predicted rates
        for the observed neural activity including the normalization constant.

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
            The likelihood. Shape (1,).
        """
        return jnp.exp(
            self.log_likelihood(y, predicted_rate, scale, aggregate_sample_scores)
        )

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

        Compute the pseudo-:math:`R^2` metric for the GLM, as defined by McFadden et al. [2]_
        or by Cohen et al. [3]_.

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
        .. [2] McFadden D (1979). Quantitative methods for analysing travel behavior of individuals: Some recent
               developments. In D. A. Hensher & P. R. Stopher (Eds.), *Behavioural travel modelling* (pp. 279-318).
               London: Croom Helm.

        .. [3] Jacob Cohen, Patricia Cohen, Steven G. West, Leona S. Aiken.
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
        # ruff: noqa D403
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

    """

    def __init__(self):
        super().__init__()
        self.scale = 1.0

    @property
    def default_inverse_link_function(self):
        return jnp.exp

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

    """

    def __init__(
        self,
    ):
        super().__init__()
        self.scale = 1.0

    @property
    def default_inverse_link_function(self):
        return utils.one_over_x

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


class BernoulliObservations(Observations):
    """
    Model observations as Bernoulli random variables.

    The BernoulliObservations is designed to model an observed binary variable based on a Bernoulli distribution
    with a given success probability. When using a logit link function (i.e. a logistic inverse link function),
    this is equivalent to Logistic Regression. It provides methods for computing the negative log-likelihood,
    generating samples, and computing the residual deviance for the given binary observations.

    """

    def __init__(self):
        super().__init__()
        self.scale = 1.0

    @property
    def default_inverse_link_function(self):
        return jax.lax.logistic

    def _negative_log_likelihood(
        self,
        y: jnp.ndarray,
        predicted_rate: jnp.ndarray,
        aggregate_sample_scores: Callable = jnp.mean,
    ) -> jnp.ndarray:
        r"""Compute the Bernoulli negative log-likelihood.

        This computes the Bernoulli negative log-likelihood of the predicted success probability (predicted rates)
        for the observations up to a constant.

        Parameters
        ----------
        y :
            The target observation to compare against. Shape (n_time_bins, ), or (n_time_bins, n_observations).
        predicted_rate :
            The predicted rate (success probability) of the current model.
            Shape (n_time_bins, ), or (n_time_bins, n_observations).
        aggregate_sample_scores :
            Function that aggregates the log-likelihood of each sample.

        Returns
        -------
        :
            The Bernoulli negative log-likelihood. Shape (1,).

        Notes
        -----
        The formula for the Bernoulli mean log-likelihood is the following,

        .. math::
            \text{LL}(p | y) &= \frac{1}{T \cdot N} \sum_{n=1}^{N} \sum_{t=1}^{T}
            [y_{tn} \log(p_{tn}) + (1 - y_{tn}) \log(1 - p_{tn})]

        where :math:`p` is the predicted success probability, given by the inverse link function, and :math:`y` is
        the observed binary variable.
        """
        predicted_rate = jnp.clip(
            predicted_rate,
            min=jnp.finfo(predicted_rate.dtype).eps,
            max=1.0 - jnp.finfo(predicted_rate.dtype).eps,
        )

        x = y * jnp.log(predicted_rate) + (1 - y) * jnp.log1p(-predicted_rate)
        return -aggregate_sample_scores(x)

    def log_likelihood(
        self,
        y: jnp.ndarray,
        predicted_rate: jnp.ndarray,
        scale: Union[float, jnp.ndarray] = 1.0,
        aggregate_sample_scores: Callable = jnp.mean,
    ):
        r"""Compute the Bernoulli negative log-likelihood.

        This computes the Bernoulli negative log-likelihood of the predicted success probability (predicted rates)
        for the observations up to a constant.

        Parameters
        ----------
        y :
            The target observation to compare against. Shape (n_time_bins, ), or (n_time_bins, n_observations).
        predicted_rate :
            The predicted rate (success probability) of the current model. Shape (n_time_bins, ),
            or (n_time_bins, n_observations).
        scale :
            The scale parameter of the model.
        aggregate_sample_scores :
            Function that aggregates the log-likelihood of each sample.

        Returns
        -------
        :
            The Bernoulli negative log-likelihood. Shape (1,).

        Notes
        -----
        The formula for the Bernoulli mean log-likelihood is the following,

        .. math::
            \text{LL}(p | y) = \frac{1}{T \cdot N} \sum_{n=1}^{N} \sum_{t=1}^{T}
            [y_{tn} \log(p_{tn}) + (1 - y_{tn}) \log(1 - p_{tn})]

        where :math:`p` is the predicted success probability, given by the inverse link function, and :math:`y` is the
        observed binary variable.
        """
        nll = self._negative_log_likelihood(y, predicted_rate, aggregate_sample_scores)
        return -nll

    def sample_generator(
        self,
        key: jax.Array,
        predicted_rate: jnp.ndarray,
        scale: Union[float, jnp.ndarray] = 1.0,
    ) -> jnp.ndarray:
        """
        Sample from the Bernoulli distribution.

        This method generates random numbers from a Bernoulli distribution based on the given
        `predicted_rate`.

        Parameters
        ----------
        key :
            Random key used for the generation of random numbers in JAX.
        predicted_rate :
            Expected rate (success probability) of the Poisson distribution. Shape ``(n_time_bins, )``, or
            ``(n_time_bins, n_observations)``.
        scale :
            Scale parameter. For Bernoulli should be equal to 1.

        Returns
        -------
        jnp.ndarray
            Random numbers generated from the Bernoulli distribution based on the `predicted_rate`.
        """
        return jax.random.bernoulli(key, predicted_rate)

    def deviance(
        self,
        observations: jnp.ndarray,
        predicted_rate: jnp.ndarray,
        scale: Union[float, jnp.ndarray] = 1.0,
    ) -> jnp.ndarray:
        r"""Compute the residual deviance for a Bernoulli model.

        Parameters
        ----------
        observations:
            The binary observations. Shape ``(n_time_bins, )`` or ``(n_time_bins, n_observations)`` for population
            models (i.e. multiple observations).
        predicted_rate:
            The predicted rate (success probability). Shape ``(n_time_bins, )``  or ``(n_time_bins, n_observations)``
            for population models (i.e. multiple observations).
        scale:
            Scale parameter of the model. For Bernoulli should be equal to 1.

        Returns
        -------
        :
            The residual deviance of the model.

        Notes
        -----
        The deviance is a measure of the goodness of fit of a statistical model.
        For a Bernoulli model, the residual deviance is computed as:

        .. math::
            \begin{aligned}
                D(y_{tn}, \hat{y}_{tn}) &= 2 \left( \text{LL}\left(y_{tn} | y_{tn}\right) - \text{LL}\left(y_{tn}
                  | \hat{y}_{tn}\right)\right) \\\
                &= 2 \left[ y_{tn} \log\left(\frac{y_{tn}}{\hat{y}_{tn}}\right) + (1 - y_{tn}) \log\left(\frac{1
                  - y_{tn}}{1 - \hat{y}_{tn}}\right) \right]
            \end{aligned}

        where :math:`y` is the observed data, :math:`\hat{y}` is the predicted data, and :math:`\text{LL}` is
        the model log-likelihood. Lower values of deviance indicate a better fit.
        """
        # this takes care of 0s in the log
        ratio1 = jnp.clip(
            observations / predicted_rate, jnp.finfo(predicted_rate.dtype).eps, jnp.inf
        )
        ratio2 = jnp.clip(
            (1 - observations) / (1 - predicted_rate),
            jnp.finfo(predicted_rate.dtype).eps,
            jnp.inf,
        )
        deviance = 2 * (
            observations * jnp.log(ratio1) + (1 - observations) * jnp.log(ratio2)
        )
        return deviance

    def estimate_scale(
        self,
        y: jnp.ndarray,
        predicted_rate: jnp.ndarray,
        dof_resid: Union[float, jnp.ndarray],
    ) -> Union[float, jnp.ndarray]:
        r"""
        Assign 1 to the scale parameter of the Bernoulli model.

        For the Binomial exponential family distribution (to which the Bernoulli belongs), the scale parameter
        :math:`\phi` is always 1.

        Parameters
        ----------
        y :
            Observed spike counts.
        predicted_rate :
            The predicted rate values (success probabilities). This is not used in the Bernoulli model for estimating
            scale, but is retained for compatibility with the abstract method signature.
        dof_resid :
            The DOF of the residuals.
        """
        return jnp.ones_like(jnp.atleast_1d(y[0]))

    def likelihood(
        self,
        y: jnp.ndarray,
        predicted_rate: jnp.ndarray,
        scale: Union[float, jnp.ndarray] = 1.0,
        aggregate_sample_scores: Callable = lambda x: jnp.exp(jnp.mean(jnp.log(x))),
    ):
        r"""Compute the Binomial model likelihood.

        This computes the likelihood of the predicted rates
        for the observed neural activity including the normalization constant.

        Parameters
        ----------
        y :
            The target activity to compare against. Shape (n_time_bins, ), or (n_time_bins, n_neurons).
        predicted_rate :
            The predicted rate of the current model. Shape (n_time_bins, ), or (n_time_bins, n_neurons).
        scale :
            The scale parameter of the model
        aggregate_sample_scores :
            Function that aggregates the likelihood of each sample.

        Returns
        -------
        :
            The likelihood. Shape (1,).
        """
        predicted_rate = jnp.clip(
            predicted_rate,
            min=jnp.finfo(predicted_rate.dtype).eps,
            max=1.0 - jnp.finfo(predicted_rate.dtype).eps,
        )
        # convenient formulation that works for specifically for the Bernoulli
        # y can be only 0 or 1 and the likelihood
        # (predicted_rate) ** y * (1 - predicted_rate) ** (1-y)
        # is equal to the computation below, easily shown by plugging y=0,1
        return aggregate_sample_scores(
            y * predicted_rate + (1 - y) * (1 - predicted_rate)
        )


class NegativeBinomialObservations(Observations):
    r"""
    A Negative Binomial model for overdispersed count data using mean-dispersion parameterization.

    This model represents a Negative Binomial distribution [4]_ commonly used to model overdispersed
    count data [5]_ [6]_ (i.e., data where the variance exceeds the mean), which cannot be captured by a
    standard Poisson model. The distribution is parameterized by the predicted mean rate
    (:math:`\mu`) and a fixed dispersion parameter (:math:`\phi`) or `scale` of the model.

    **Important:** the scale parameter must be estimated from the data for accurately capturing the
    dispersion. In the context of NeMoS GLM, estimation can be achieved by cross-validating the
    ``scale`` parameter. One may use scikit-learn :ref:`GridSearchCV <sklearn-how-to>` for example.

    The variance of the Negative Binomial distribution under this parameterization is:

    .. math::

        \mathrm{Var}(Y) = \mu + \phi \mu^2

    where :math:`\mu` is the predicted mean, and :math:`\phi` is the dispersion parameter. This
    formulation corresponds to the Negative Binomial as a Gamma–Poisson mixture.

    The scale parameter :math:`\phi` is related to the canonical Negative Binomial
    shape parameter `r` as:

    .. math::

        r = \frac{1}{\phi}

    As :math:`\phi \to 0` (equivalently, :math:`r \to \infty`), the distribution approaches a
    Poisson distribution. This makes the model flexible for handling both equidispersed
    (Poisson-like) and overdispersed data.

    Parameters
    ----------
    scale :
        The dispersion parameter :math:`\phi`. Lower values correspond to lower overdispersion, and as
        :math:`\phi \to 0`, the model behaves like a Poisson. The shape parameter of
        the Negative Binomial is given by `r = 1 / scale`.

    References
    ----------
    .. [4] https://en.wikipedia.org/wiki/Negative_binomial_distribution

    .. [5] Pillow, Jonathan, and James Scott. "Fully Bayesian inference for neural models
        with negative-binomial spiking." Advances in neural information processing systems 25 (2012).

    .. [6] Wei, Ganchao, et al. "Calibrating Bayesian decoders of neural spiking activity."
        Journal of Neuroscience 44.18 (2024).
    """

    def __init__(
        self,
        scale=1.0,
    ):
        super().__init__()
        self.scale = scale

    @property
    def default_inverse_link_function(self):
        return jnp.exp

    def _negative_log_likelihood(
        self,
        y: jnp.ndarray,
        predicted_rate: jnp.ndarray,
        aggregate_sample_scores: Callable = jnp.mean,
    ) -> jnp.ndarray:
        r"""Compute the Negative Binomial negative log-likelihood.

        This computes the Negative Binomial negative log-likelihood of the
        predicted mean rate for the observed counts.

        Parameters
        ----------
        y :
            Observed count data. Shape ``(n_time_bins,)`` or ``(n_time_bins, n_observations)``.
        predicted_rate :
            The predicted mean of the Negative Binomial distribution. Shape
            ``(n_time_bins,)`` or ``(n_time_bins, n_observations)``.
        aggregate_sample_scores :
            Function that aggregates the log-likelihood across samples (e.g., ``jnp.mean`` or
            ``jnp.sum``).

        Returns
        -------
        :
            The Negative Binomial negative log-likelihood. Shape ``(1,)``.

        Notes
        -----
        The Negative Binomial distribution models overdispersed count data.
        The likelihood assumes the mean-parameterized form with dispersion `r = 1 / scale`.

        The log-likelihood is computed (up to a constant) using:

        .. math::
            \log p(y | \mu, r) = \log \Gamma(y + r) - \log \Gamma(r) - \log y!
            + r \log\left(\frac{r}{r + \mu}\right)
            + y \log\left(\frac{\mu}{r + \mu}\right)

        """
        if self.scale is None:
            self.estimate_scale(y, predicted_rate, aggregate_sample_scores)
        predicted_rate = jnp.clip(
            predicted_rate, min=jnp.finfo(predicted_rate.dtype).eps
        )
        factor = 1 / (self.scale * predicted_rate + 1)
        return -aggregate_sample_scores(
            y * jnp.log(1 - factor) + jnp.log(factor) / self.scale
        )

    def log_likelihood(
        self,
        y: jnp.ndarray,
        predicted_rate: jnp.ndarray,
        scale: Union[float, jnp.ndarray, None] = None,
        aggregate_sample_scores: Callable = jnp.mean,
    ):
        r"""Compute the Negative Binomial log-likelihood.

        This computes the Negative Binomial log-likelihood of the predicted mean
        rate for the observed counts.

        Parameters
        ----------
        y :
            Observed count data. Shape ``(n_time_bins,)`` or ``(n_time_bins, n_observations)``.
        predicted_rate :
            The predicted mean of the Negative Binomial distribution. Shape ``(n_time_bins,)`` or
            ``(n_time_bins, n_observations)``.
        scale :
            The scale (dispersion) parameter of the distribution. It is related to the shape ``r`` as ``r = 1 / scale``.
            Default is the scale provided at initialization ``self.scale``.
        aggregate_sample_scores :
            Function that aggregates the log-likelihood across samples (e.g., jnp.mean or jnp.sum).

        Returns
        -------
        :
            The log-likelihood of the Negative Binomial model. Shape ``(1,)``.
        """
        scale = self.scale if scale is None else scale
        ll_unnormalized = -self._negative_log_likelihood(
            y, predicted_rate, aggregate_sample_scores
        )
        norm = aggregate_sample_scores(
            jax.scipy.special.gammaln(y + 1 / scale)
            - jax.scipy.special.gammaln(y + 1)
            - jax.scipy.special.gammaln(1 / scale)
        )
        return ll_unnormalized + norm

    def sample_generator(
        self,
        key: jax.Array,
        predicted_rate: jnp.ndarray,
        scale: Union[float, jnp.ndarray, None] = None,
    ) -> jnp.ndarray:
        r"""
        Sample from the Negative Binomial distribution.

        This method generates random count data from the Negative Binomial distribution based on the predicted rate
        and scale (dispersion).

        Parameters
        ----------
        key :
            Random key used for number generation in JAX.
        predicted_rate :
            The predicted mean of the Negative Binomial distribution. Shape ``(n_time_bins,)`` or
            ``(n_time_bins, n_observations)``.
        scale :
            Dispersion parameter of the distribution. Smaller values imply higher variance.

        Returns
        -------
        :
            Samples drawn from the Negative Binomial distribution. Same shape as ``predicted_rate``.

        Notes
        -----
        This method uses the Gamma--Poisson mixture representation of the Negative Binomial distribution:

        .. math::

            Y \sim \text{Poisson}(\lambda), \quad \lambda \sim \text{Gamma}(r, \beta)

        where :math:`r = 1 / \phi` and :math:`\beta = r / \mu`.

        For more information, see the `Negative Binomial distribution on Wikipedia
        <https://en.wikipedia.org/wiki/Negative_binomial_distribution#Gamma%E2%80%93Poisson_mixture>`_.
        """
        scale = self.scale if scale is None else scale
        r = 1.0 / scale
        gamma_key, poisson_key = jax.random.split(key)

        # Gamma with shape=r, rate=r/mu → scale=mu/r
        gamma_sample = jax.random.gamma(gamma_key, r, shape=predicted_rate.shape) * (
            predicted_rate / r
        )

        return jax.random.poisson(poisson_key, gamma_sample)

    def deviance(
        self,
        observations: jnp.ndarray,
        predicted_rate: jnp.ndarray,
        scale: Union[float, jnp.ndarray, None] = None,
    ) -> jnp.ndarray:
        r"""Compute the residual deviance for a Negative Binomial model.

        The deviance measures how well a statistical model fits the data by
        quantifying the difference between the observed values and the values
        predicted by the model. Lower values of deviance indicate a better fit.

        Parameters
        ----------
        observations:
            Observed count data. Shape ``(n_time_bins,)`` or ``(n_time_bins, n_observations).``
        predicted_rate:
            Predicted mean count of the Negative Binomial distribution. Shape matches `observations`.
        scale:
            Dispersion parameter of the distribution.

        Returns
        -------
        :
            The residual deviance of the model. Shape matches ``observations``.

        Notes
        -----
        The deviance is a measure of the goodness of fit of a statistical model.
        For a Negative Binomial model, the residual deviance is computed as:

        .. math::
            \begin{aligned}
                D(y_{tn}, \hat{y}_{tn}) &= 2 \left( \text{LL}\left(y_{tn} | y_{tn}\right) - \text{LL}\left(y_{tn}
                  | \hat{y}_{tn}\right)\right) \\\
                &= 2 \left[ y_{tn} \log\left(\frac{y_{tn}}{\hat{y}_{tn}}\right) + (1 - y_{tn}) \log\left(\frac{1
                  - y_{tn}}{1 - \hat{y}_{tn}}\right) \right]
            \end{aligned}

        where :math:`y` is the observed data, :math:`\hat{y}` is the predicted data, and :math:`\text{LL}` is
        the model log-likelihood.
        """
        scale = self.scale if scale is None else scale
        factor = (predicted_rate * scale + 1) / (observations * scale + 1)
        y_mu = observations / predicted_rate
        y_mu = jnp.clip(y_mu, min=jnp.finfo(predicted_rate.dtype).eps)
        term1 = observations * jnp.log(y_mu * factor)
        term2 = jnp.log(factor) / scale
        return 2 * (term1 + term2)

    def estimate_scale(
        self,
        y: jnp.ndarray,
        predicted_rate: jnp.ndarray,
        dof_resid: Union[float, jnp.ndarray],
    ) -> Union[float, jnp.ndarray]:
        r"""
        Return the scale parameter of the distribution.

        The ``scale`` parameter of the Negative Binomial distribution is set
        at initialization and affect the likelihood landscape. This implies
        that the ``scale`` parameter cannot be estimated post-hoc without
        re-fitting a model.

        Note that the arguments of this method are not used but are kept for
        API consistency—i.e., all ``Observations.estimate_scale`` methods
        have the same signature.

        Parameters
        ----------
        y :
            Observed spike counts.
        predicted_rate :
            The predicted mean of the distribution.
        dof_resid :
            The DOF of the residuals.

        Notes
        -----
        NeMoS currently does not support joint estimation of scale and mean for the negative binomial.
        For alternatives, see the R package MASS
        `glm.nb <https://www.rdocumentation.org/packages/MASS/versions/7.3-65/topics/glm.nb>`_
        for more details.
        """
        return jnp.array(self.scale)


def check_observation_model(observation_model, force_checks=False):
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
    force_checks:
        If true, always checks. This is intended for testing purposes, to make sure
        that the check passes for native nemos observation models.

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
    >>> import jax
    >>> import jax.numpy as jnp
    >>> class MyObservationModel:
    ...     @property
    ...     def default_inverse_link_function(self):
    ...         return jax.scipy.special.expit
    ...     def _negative_log_likelihood(self, y, rate, aggregate_sample_scores=jnp.mean):
    ...         return -aggregate_sample_scores(y * jax.scipy.special.logit(rate) + \
    ...                                         (1 - y) * jax.scipy.special.logit(1 - rate))
    ...     def pseudo_r2(self, y, rate, aggregate_sample_scores=jnp.mean):
    ...         return 1 - (self._negative_log_likelihood(y, rate, aggregate_sample_scores) /
    ...                     jnp.sum((y - y.mean()) ** 2))
    ...     def sample_generator(self, key, rate, scale=1.):
    ...         return jax.random.bernoulli(key, rate)
    ...     def estimate_scale(self, y, predicted_rate, dof_resid):
    ...         return jnp.array(1.)
    ...     def log_likelihood(self, y, rate, aggregate_sample_scores=jnp.mean):
    ...         return -self._negative_log_likelihood(y, rate, aggregate_sample_scores)
    ...     def likelihood(self, y, rate, aggregate_sample_scores=jnp.mean):
    ...         return jnp.exp(self.log_likelihood(y, rate, aggregate_sample_scores))
    ...     def deviance(self, y, rate, scale=1.0):
    ...         identity = lambda x: x
    ...         return 2 * (
    ...                 self.log_likelihood(y, rate, identity) -
    ...                 self.log_likelihood(y.mean()*jnp.ones_like(y), rate, identity))
    >>> model = MyObservationModel()
    >>> check_observation_model(model)  # Should pass without error if the model is correctly implemented.
    """
    # Define the checks to be made on each attribute

    is_nemos = isinstance(
        observation_model,
        (
            PoissonObservations,
            GammaObservations,
            BernoulliObservations,
            NegativeBinomialObservations,
        ),
    )

    checks = {}
    if not is_nemos or force_checks:
        checks.update(
            {
                "_negative_log_likelihood": {
                    "input": [
                        0.5 * jnp.array([1.0, 1.0, 1.0]),
                        jnp.array([1.0, 1.0, 1.0]),
                    ],
                    "test_scalar_func": True,
                },
                "pseudo_r2": {
                    "input": [
                        0.5 * jnp.array([1.0, 1.0, 1.0]),
                        jnp.array([1.0, 1.0, 1.0]),
                    ],
                    "test_scalar_func": True,
                },
                "log_likelihood": {
                    "input": [
                        0.5 * jnp.array([1.0, 1.0, 1.0]),
                        jnp.array([1.0, 1.0, 1.0]),
                    ],
                    "test_scalar_func": True,
                },
                "likelihood": {
                    "input": [
                        0.5 * jnp.array([1.0, 1.0, 1.0]),
                        jnp.array([1.0, 1.0, 1.0]),
                    ],
                    "test_scalar_func": True,
                },
                "deviance": {
                    "input": [
                        0.5 * jnp.array([1.0, 1.0, 1.0]),
                        jnp.array([1.0, 1.0, 1.0]),
                    ],
                    "test_scalar_func": False,
                },
                "estimate_scale": {
                    "input": [
                        0.5 * jnp.array([1.0, 1.0, 1.0]),
                        jnp.array([1.0, 1.0, 1.0]),
                        1,
                    ],
                    "test_scalar_func": False,
                },
                "sample_generator": {
                    "input": [jax.random.key(123), 0.5 * jnp.array([1.0, 1.0, 1.0]), 1],
                    "test_preserve_shape": True,
                },
                "default_inverse_link_function": {
                    "input": [jnp.array([1.0, 1.0, 1.0])],
                    "test_preserve_shape": False,
                },
            }
        )

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
