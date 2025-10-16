import math
import warnings
from contextlib import nullcontext as does_not_raise

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy as sp
import scipy.stats as sts
import statsmodels.api as sm

import nemos as nmo
from nemos._observation_model_builder import (
    AVAILABLE_OBSERVATION_MODELS,
    instantiate_observation_model,
)
from nemos.glm.initialize_parameters import initialize_intercept_matching_mean_rate
from nemos.glm.inverse_link_function_utils import LINK_NAME_TO_FUNC


@pytest.fixture
def observation_model_rate_and_samples(observation_model_string, shape=None):
    """
    Fixture that returns rate and samples for each observation model.
    """
    if shape is None:
        shape = (10,)
    obs = instantiate_observation_model(observation_model_string)
    rate = jax.random.uniform(
        jax.random.PRNGKey(122), shape=shape, minval=0.1, maxval=10
    )
    if observation_model_string == "Poisson":
        y = jax.random.poisson(jax.random.PRNGKey(123), rate)
    elif observation_model_string == "Gamma":
        theta = 3
        y = jax.random.gamma(jax.random.PRNGKey(123), rate / theta) * theta
    elif observation_model_string == "Bernoulli":
        rate = rate / (1 + jnp.max(rate))
        y = jax.random.bernoulli(jax.random.PRNGKey(123), rate)
    elif observation_model_string == "NegativeBinomial":
        r = 1.0
        gamma_key, poisson_key = jax.random.split(jax.random.PRNGKey(123))
        gamma_sample = jax.random.gamma(gamma_key, r, shape=rate.shape) * (rate / r)
        y = jax.random.poisson(poisson_key, gamma_sample)
    else:
        raise ValueError(f"Unknown observation model {observation_model_string}.")
    return obs, y, rate


@pytest.fixture()
def poisson_observations():
    return nmo.observation_models.PoissonObservations


@pytest.fixture()
def gamma_observations():
    return nmo.observation_models.GammaObservations


@pytest.fixture()
def bernoulli_observations():
    return nmo.observation_models.BernoulliObservations


@pytest.fixture()
def negative_binomial_observations():
    return nmo.observation_models.NegativeBinomialObservations


@pytest.mark.parametrize(
    "obs_model_string, expectation",
    [
        ("Poisson", does_not_raise()),
        ("Gamma", does_not_raise()),
        ("Bernoulli", does_not_raise()),
        ("NegativeBinomial", does_not_raise()),
        (
            "invalid",
            pytest.raises(ValueError, match="Unknown observation model: invalid"),
        ),
    ],
)
@pytest.mark.parametrize("glm_class", [nmo.glm.GLM, nmo.glm.PopulationGLM])
def test_glm_instantiation_from_string_at_init(
    obs_model_string, glm_class, expectation
):
    with expectation:
        glm_class(observation_model=obs_model_string)


@pytest.mark.parametrize(
    "obs_model_string, expectation",
    [
        ("Poisson", does_not_raise()),
        ("Gamma", does_not_raise()),
        ("Bernoulli", does_not_raise()),
        ("nemos.observation_models.PoissonObservations", does_not_raise()),
        ("nemos.observation_models.GammaObservations", does_not_raise()),
        ("nemos.observation_models.BernoulliObservations", does_not_raise()),
        ("NegativeBinomial", does_not_raise()),
        ("nemos.observation_models.NegativeBinomial", does_not_raise()),
        (
            "invalid",
            pytest.raises(ValueError, match="Unknown observation model: invalid"),
        ),
    ],
)
@pytest.mark.parametrize("glm_class", [nmo.glm.GLM, nmo.glm.PopulationGLM])
def test_glm_setter_observation_model(obs_model_string, glm_class, expectation):
    """Test that the observation model can be set after the init providing a string."""
    if obs_model_string != "Poisson":
        obs = nmo.observation_models.PoissonObservations()
    else:
        obs = nmo.observation_models.GammaObservations()
    model = glm_class(observation_model=obs)
    with expectation:
        model.observation_model = obs_model_string
    if (
        obs_model_string == "Gamma"
        or obs_model_string == "nemos.observation_models.GammaObservations"
    ):
        assert isinstance(
            model.observation_model, nmo.observation_models.GammaObservations
        )
    elif (
        obs_model_string == "Poisson"
        or obs_model_string == "nemos.observation_models.PoissonObservations"
    ):
        assert isinstance(
            model.observation_model, nmo.observation_models.PoissonObservations
        )
    elif (
        obs_model_string == "Bernoulli"
        or obs_model_string == "nemos.observation_models.BernoulliObservations"
    ):
        assert isinstance(
            model.observation_model, nmo.observation_models.BernoulliObservations
        )
    elif obs_model_string == "NegativeBinomial":
        assert isinstance(
            model.observation_model, nmo.observation_models.NegativeBinomialObservations
        )


@pytest.mark.parametrize(
    "obs_model_string, expectation",
    [
        ("Poisson", does_not_raise()),
        ("Gamma", does_not_raise()),
        ("Bernoulli", does_not_raise()),
        ("NegativeBinomial", does_not_raise()),
        ("NB", pytest.raises(ValueError, match="Unknown observation model: NB")),
    ],
)
def test_instantiate_observation_model(obs_model_string, expectation):
    """Test instantiation of observation model with a link function."""
    with expectation:
        obs_model = instantiate_observation_model(
            obs_model_string,
        )


class TestPoissonObservations:

    def test_get_params(self, poisson_observations):
        """Test get_params() returns expected values."""
        observation_model = poisson_observations()

        assert observation_model.get_params() == {}

    def test_deviance_against_statsmodels(self, poissonGLM_model_instantiation):
        """
        Compare fitted parameters to statsmodels.
        Assesses if the model estimates are close to statsmodels' results.
        """
        _, y, model, _, firing_rate = poissonGLM_model_instantiation
        dev = sm.families.Poisson().deviance(y, firing_rate)
        dev_model = model.observation_model.deviance(y, firing_rate).sum()
        if not np.allclose(dev, dev_model):
            raise ValueError("Deviance doesn't match statsmodels!")

    def test_loglikelihood_against_scipy(self, poissonGLM_model_instantiation):
        """
        Compare log-likelihood to scipy.
        Assesses if the model estimates are close to statsmodels' results.
        """
        _, y, model, _, firing_rate = poissonGLM_model_instantiation
        ll_model = model.observation_model.log_likelihood(y, firing_rate)
        ll_scipy = sts.poisson(firing_rate).logpmf(y).mean()
        if not np.allclose(ll_model, ll_scipy):
            raise ValueError("Log-likelihood doesn't match scipy!")

    def test_emission_probability(selfself, poissonGLM_model_instantiation):
        """
        Test the poisson emission probability.

        Check that the emission probability is set to jax.random.poisson.
        """
        _, _, model, _, _ = poissonGLM_model_instantiation
        key_array = jax.random.key(123)
        counts = model.observation_model.sample_generator(key_array, np.arange(1, 11))
        if not jnp.all(counts == jax.random.poisson(key_array, np.arange(1, 11))):
            raise ValueError(
                "The emission probability should output the results of a call to jax.random.poisson."
            )

    @pytest.mark.parametrize(
        "score_type, expectation",
        [
            ("pseudo-r2-McFadden", does_not_raise()),
            (
                "not-implemented",
                pytest.raises(
                    NotImplementedError, match="Score not-implemented not implemented"
                ),
            ),
        ],
    )
    def test_not_implemented_score(
        self, score_type, expectation, poissonGLM_model_instantiation
    ):
        _, y, model, _, firing_rate = poissonGLM_model_instantiation
        with expectation:
            model.observation_model.pseudo_r2(y, firing_rate, score_type)

    @pytest.mark.parametrize(
        "scale, expectation",
        [
            (1, does_not_raise()),
            (
                "invalid",
                pytest.raises(
                    ValueError, match="The `scale` parameter must be of numeric type"
                ),
            ),
        ],
    )
    def test_scale_setter(self, scale, expectation, poissonGLM_model_instantiation):
        _, _, model, _, firing_rate = poissonGLM_model_instantiation
        with expectation:
            model.observation_model.scale = scale

    def test_scale_getter(self, poissonGLM_model_instantiation):
        _, _, model, _, firing_rate = poissonGLM_model_instantiation
        assert model.observation_model.scale == 1

    @pytest.mark.parametrize(
        "test_value",
        [
            jnp.array([25.0]),
            jnp.array([7]),
            jnp.array([0.0]),
            jnp.array([1.0]),
            jnp.array([0.5]),
        ],
    )
    def test_custom_inverse_link_function(
        self,
        test_value,
        poissonGLM_model_instantiation,
    ):
        """
        Test that custom inverse link function can be inverted and works correctly.
        """
        # Initialize model
        _, _, model, _, _ = poissonGLM_model_instantiation
        model.observation_model.inverse_link_function = lambda x: jnp.power(x, 2)

        # Validate custom link function
        expected_output = jnp.sqrt(test_value)
        result = initialize_intercept_matching_mean_rate(
            model.observation_model.inverse_link_function, test_value
        )

        assert np.allclose(
            result, expected_output
        ), f"Inverse link function result mismatch: expected {expected_output}, got {result}"

    def test_pseudo_r2_vs_statsmodels(self, poissonGLM_model_instantiation):
        """
        Compare log-likelihood to scipy.
        Assesses if the model estimates are close to statsmodels' results.
        """
        X, y, model, _, firing_rate = poissonGLM_model_instantiation

        # statsmodels mcfadden
        mdl = sm.GLM(y, sm.add_constant(X), family=sm.families.Poisson()).fit()
        pr2_sms = mdl.pseudo_rsquared("mcf")

        # set params
        pr2_model = model.observation_model.pseudo_r2(
            y, mdl.mu, score_type="pseudo-r2-McFadden"
        )

        if not np.allclose(pr2_model, pr2_sms):
            raise ValueError("Log-likelihood doesn't match statsmodels!")

    def test_repr_out(self):
        obs = nmo.observation_models.PoissonObservations()
        assert repr(obs) == f"PoissonObservations()"


class TestGammaObservations:

    def test_get_params(self, gamma_observations):
        """Test get_params() returns expected values."""
        observation_model = gamma_observations()

        assert observation_model.get_params() == {}

    def test_deviance_against_statsmodels(self, gammaGLM_model_instantiation):
        """
        Compare fitted parameters to statsmodels.
        Assesses if the model estimates are close to statsmodels' results.
        """
        _, y, model, _, firing_rate = gammaGLM_model_instantiation
        dev = sm.families.Gamma().deviance(y, firing_rate)
        dev_model = model.observation_model.deviance(y, firing_rate).sum()
        if not np.allclose(dev, dev_model):
            raise ValueError("Deviance doesn't match statsmodels!")

    def test_loglikelihood_against_statsmodels(self, gammaGLM_model_instantiation):
        """
        Compare log-likelihood to scipy.
        Assesses if the model estimates are close to statsmodels' results.
        """
        _, y, model, _, firing_rate = gammaGLM_model_instantiation
        ll_model = model.observation_model.log_likelihood(y, firing_rate)
        ll_sms = sm.families.Gamma().loglike(y, firing_rate) / y.shape[0]
        if not np.allclose(ll_model, ll_sms):
            raise ValueError("Log-likelihood doesn't match statsmodels!")

    def test_emission_probability(self, gammaGLM_model_instantiation):
        """
        Test the gamma emission probability.

        Check that the emission probability is set to jax.random.gamma.
        """
        _, _, model, _, _ = gammaGLM_model_instantiation
        key_array = jax.random.key(123)
        counts = model.observation_model.sample_generator(key_array, np.arange(1, 11))
        if not jnp.all(counts == jax.random.gamma(key_array, np.arange(1, 11))):
            raise ValueError(
                "The emission probability should output the results of a call to jax.random.gamma."
            )

    def test_pseudo_r2_vs_statsmodels(self, gammaGLM_model_instantiation):
        """
        Compare log-likelihood to scipy.
        Assesses if the model estimates are close to statsmodels' results.
        """
        jax.config.update("jax_enable_x64", True)
        X, y, model, _, firing_rate = gammaGLM_model_instantiation

        # statsmodels mcfadden
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The InversePower link function does"
            )
            mdl = sm.GLM(y, sm.add_constant(X), family=sm.families.Gamma()).fit()
        pr2_sms = mdl.pseudo_rsquared("mcf")

        # set params
        pr2_model = model.observation_model.pseudo_r2(
            y, mdl.mu, score_type="pseudo-r2-McFadden", scale=mdl.scale
        )

        if not np.allclose(pr2_model, pr2_sms):
            raise ValueError("Log-likelihood doesn't match statsmodels!")

    def test_repr_out(self):
        obs = nmo.observation_models.GammaObservations()
        assert repr(obs) == f"GammaObservations()"


class TestBernoulliObservations:

    def test_get_params(self, bernoulli_observations):
        """Test get_params() returns expected values."""
        observation_model = bernoulli_observations()

        assert observation_model.get_params() == {}

    def test_deviance_against_statsmodels(self, bernoulliGLM_model_instantiation):
        """
        Compare fitted parameters to statsmodels.
        Assesses if the model estimates are close to statsmodels' results.
        """
        _, y, model, _, firing_rate = bernoulliGLM_model_instantiation
        dev = sm.families.Binomial().deviance(y, firing_rate)
        dev_model = model.observation_model.deviance(y, firing_rate).sum()
        if not np.allclose(dev, dev_model):
            raise ValueError("Deviance doesn't match statsmodels!")

    def test_loglikelihood_against_scipy(self, bernoulliGLM_model_instantiation):
        """
        Compare log-likelihood to scipy.
        Assesses if the model estimates are close to statsmodels' results.
        """
        _, y, model, _, firing_rate = bernoulliGLM_model_instantiation
        ll_model = model.observation_model.log_likelihood(y, firing_rate)
        ll_scipy = sts.bernoulli(firing_rate).logpmf(y).mean()
        if not np.allclose(ll_model, ll_scipy):
            raise ValueError("Log-likelihood doesn't match scipy!")

    def test_emission_probability(self, bernoulliGLM_model_instantiation):
        """
        Test the poisson emission probability.

        Check that the emission probability is set to jax.random.poisson.
        """
        _, _, model, _, _ = bernoulliGLM_model_instantiation
        key_array = jax.random.key(123)
        p = np.random.rand(10)
        counts = model.observation_model.sample_generator(key_array, p)
        if not jnp.all(counts == jax.random.bernoulli(key_array, p)):
            raise ValueError(
                "The emission probability should output the results of a call to jax.random.poisson."
            )

    def test_pseudo_r2_vs_statsmodels(self, bernoulliGLM_model_instantiation):
        """
        Compare log-likelihood to scipy.
        Assesses if the model estimates are close to statsmodels' results.
        """
        X, y, model, _, firing_rate = bernoulliGLM_model_instantiation

        # statsmodels mcfadden
        mdl = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial()).fit()
        pr2_sms = mdl.pseudo_rsquared("mcf")

        # set params
        pr2_model = model.observation_model.pseudo_r2(
            y, mdl.mu, score_type="pseudo-r2-McFadden"
        )

        if not np.allclose(pr2_model, pr2_sms):
            raise ValueError("Log-likelihood doesn't match statsmodels!")

    def test_repr_out(self):
        obs = nmo.observation_models.BernoulliObservations()
        assert repr(obs) == f"BernoulliObservations()"


class TestNegativeBinomialObservations:

    def test_get_params(self, negative_binomial_observations):
        observation_model = negative_binomial_observations()
        assert observation_model.get_params() == {
            "scale": 1.0,
        }

    def test_deviance_against_statsmodels(
        self, negativeBinomialGLM_model_instantiation
    ):
        jax.config.update("jax_enable_x64", True)
        _, y, model, _, firing_rate = negativeBinomialGLM_model_instantiation
        dev = sm.families.NegativeBinomial(
            alpha=model.observation_model.scale
        ).deviance(y, firing_rate)
        dev_model = model.observation_model.deviance(y, firing_rate).sum()
        if not np.allclose(dev, dev_model):
            raise ValueError("Deviance doesn't match statsmodels!")

    def test_loglikelihood_against_scipy(self, negativeBinomialGLM_model_instantiation):
        _, y, model, _, firing_rate = negativeBinomialGLM_model_instantiation
        r = 1.0 / model.observation_model.scale
        p = r / (r + firing_rate)
        ll_model = model.observation_model.log_likelihood(y, firing_rate)
        ll_scipy = sts.nbinom.logpmf(y, r, p).mean()
        if not np.allclose(ll_model, ll_scipy, atol=1e-5):
            raise ValueError("Log-likelihood doesn't match scipy!")

    def test_emission_probability(self, negativeBinomialGLM_model_instantiation):
        _, _, model, _, _ = negativeBinomialGLM_model_instantiation
        key_array = jax.random.key(123)
        gkey, pkey = jax.random.split(key_array)
        p = np.random.rand(10)
        counts = model.observation_model.sample_generator(
            key_array, p, scale=model.observation_model.scale
        )
        r = 1.0 / model.observation_model.scale
        gamma_sample = jax.random.gamma(gkey, r, shape=p.shape) * (p / r)
        expected_counts = jax.random.poisson(pkey, gamma_sample)
        if not jnp.allclose(counts, expected_counts):
            raise ValueError(
                "The emission probability doesn't match expected NB sampling."
            )

    def test_pseudo_r2_vs_statsmodels(self, negativeBinomialGLM_model_instantiation):
        X, y, model, _, firing_rate = negativeBinomialGLM_model_instantiation
        mdl = sm.GLM(
            y,
            sm.add_constant(X),
            family=sm.families.NegativeBinomial(alpha=model.observation_model.scale),
        ).fit()
        pr2_sms = mdl.pseudo_rsquared("mcf")
        pr2_model = model.observation_model.pseudo_r2(
            y,
            mdl.mu,
            scale=model.observation_model.scale,
            score_type="pseudo-r2-McFadden",
        )
        if not np.allclose(pr2_model, pr2_sms, atol=1e-5):
            raise ValueError("Pseudo-r2 doesn't match statsmodels!")

    def test_repr_out(self):
        obs = nmo.observation_models.NegativeBinomialObservations()
        assert repr(obs) == f"NegativeBinomialObservations(scale=1.0)"


@pytest.mark.parametrize("observation_model_string", AVAILABLE_OBSERVATION_MODELS)
class TestCommonObservationModels:

    @pytest.mark.parametrize("shape", [(10,), (10, 5), (10, 5, 2)])
    def test_likelihood_matching(
        self, shape, observation_model_string, observation_model_rate_and_samples
    ):
        jax.config.update("jax_enable_x64", True)
        obs, y, rate = observation_model_rate_and_samples
        like1 = jnp.exp(
            obs.log_likelihood(y, rate, aggregate_sample_scores=lambda x: x)
        )
        like2 = obs.likelihood(y, rate, aggregate_sample_scores=lambda x: x)
        assert jnp.allclose(like1, like2)

        like1 = jnp.exp(obs.log_likelihood(y, rate))
        like2 = obs.likelihood(y, rate)
        assert jnp.allclose(like1, like2)

    @pytest.mark.parametrize("shape", [(10,), (10, 5), (10, 5, 2)])
    def test_aggregation_score_mcfadden(
        self, shape, observation_model_string, observation_model_rate_and_samples
    ):
        obs, y, rate = observation_model_rate_and_samples
        sm = obs._pseudo_r2_mcfadden(y, rate, aggregate_sample_scores=jnp.sum)
        mn = obs._pseudo_r2_mcfadden(y, rate, aggregate_sample_scores=jnp.mean)
        assert np.allclose(sm, mn)

    @pytest.mark.parametrize("shape", [(10,), (10, 5), (10, 5, 2)])
    def test_aggregation_score_choen(
        self, shape, observation_model_string, observation_model_rate_and_samples
    ):
        obs, y, rate = observation_model_rate_and_samples
        sm = obs._pseudo_r2_cohen(y, rate, aggregate_sample_scores=jnp.sum)
        mn = obs._pseudo_r2_cohen(y, rate, aggregate_sample_scores=jnp.mean)
        assert np.allclose(sm, mn)

    @pytest.mark.parametrize("score_type", ["pseudo-r2-McFadden", "pseudo-r2-Cohen"])
    @pytest.mark.parametrize("shape", [(10,), (10, 5), (10, 5, 2)])
    def test_aggregation_score_pr2(
        self,
        score_type,
        shape,
        observation_model_string,
        observation_model_rate_and_samples,
    ):
        obs, y, rate = observation_model_rate_and_samples
        sm = obs.pseudo_r2(
            y, rate, score_type=score_type, aggregate_sample_scores=jnp.sum
        )
        mn = obs.pseudo_r2(
            y, rate, score_type=score_type, aggregate_sample_scores=jnp.mean
        )
        assert np.allclose(sm, mn)

    @pytest.mark.parametrize("shape", [(10,), (10, 5), (10, 5, 2)])
    def test_aggregation_score_ll(
        self, shape, observation_model_string, observation_model_rate_and_samples
    ):
        obs, y, rate = observation_model_rate_and_samples
        sm = obs.log_likelihood(y, rate, aggregate_sample_scores=jnp.sum)
        mn = obs.log_likelihood(y, rate, aggregate_sample_scores=jnp.mean)
        assert np.allclose(sm, mn * math.prod(y.shape))

    @pytest.mark.parametrize("shape", [(10,), (10, 5), (10, 5, 2)])
    def test_aggregation_score_neg_ll(
        self, shape, observation_model_string, observation_model_rate_and_samples
    ):
        obs, y, rate = observation_model_rate_and_samples
        sm = obs._negative_log_likelihood(y, rate, jnp.sum)
        mn = obs._negative_log_likelihood(y, rate, jnp.mean)
        assert np.allclose(sm, mn * math.prod(y.shape))

    def test_scale_getter(
        self, observation_model_string, observation_model_rate_and_samples
    ):
        obs, y, rate = observation_model_rate_and_samples
        assert obs.scale == 1

    @pytest.mark.parametrize(
        "scale, expectation",
        [
            (1, does_not_raise()),
            (
                "invalid",
                pytest.raises(
                    ValueError, match="The `scale` parameter must be of numeric type"
                ),
            ),
        ],
    )
    def test_scale_setter(
        self,
        scale,
        expectation,
        observation_model_string,
        observation_model_rate_and_samples,
    ):
        obs, y, rate = observation_model_rate_and_samples
        with expectation:
            obs.scale = scale

    @pytest.mark.parametrize(
        "score_type, expectation",
        [
            ("pseudo-r2-McFadden", does_not_raise()),
            (
                "not-implemented",
                pytest.raises(
                    NotImplementedError, match="Score not-implemented not implemented"
                ),
            ),
        ],
    )
    def test_not_implemented_score(
        self,
        score_type,
        expectation,
        observation_model_string,
        observation_model_rate_and_samples,
    ):
        obs, y, rate = observation_model_rate_and_samples
        with expectation:
            obs.pseudo_r2(y, rate, score_type)

    @pytest.mark.parametrize("score_type", ["pseudo-r2-Cohen", "pseudo-r2-McFadden"])
    def test_pseudo_r2_mean(
        self, score_type, observation_model_string, observation_model_rate_and_samples
    ):
        """
        Check that the pseudo-r2 of the null model is 0.
        """
        obs, y, rate = observation_model_rate_and_samples
        pseudo_r2 = obs.pseudo_r2(y, y.mean(), score_type=score_type)
        if not np.allclose(pseudo_r2, 0, atol=10**-7, rtol=0.0):
            raise ValueError(
                f"pseudo-r2 of {pseudo_r2} for the null model. Should be equal to 0!"
            )

    @pytest.mark.parametrize("score_type", ["pseudo-r2-Cohen", "pseudo-r2-McFadden"])
    def test_pseudo_r2_range(
        self, score_type, observation_model_string, observation_model_rate_and_samples
    ):
        """
        Compute the pseudo-r2 and check that is < 1.
        """
        obs, y, rate = observation_model_rate_and_samples
        pseudo_r2 = obs.pseudo_r2(y, rate, score_type=score_type)
        if (pseudo_r2 > 1) or (pseudo_r2 < 0):
            raise ValueError(f"pseudo-r2 of {pseudo_r2} outside the [0,1] range!")


class ConfigurableObservationModel:
    """
    A configurable observation model for testing.

    Parameters
    ----------
    exclude_methods : list of str, optional
        Methods to exclude from the model (simulates missing attributes)
    override_returns : dict, optional
        Dictionary mapping method names to return values.
        If a method is in this dict, it will return the specified value instead of the default.
    non_callable_methods : list of str, optional
        Methods that should be non-callable (assigned as strings or other non-callable types)
    """

    def __init__(
        self, exclude_methods=None, override_returns=None, non_callable_methods=None
    ):
        self.exclude_methods = exclude_methods or []
        self.override_returns = override_returns or {}
        self.non_callable_methods = non_callable_methods or []

        # Set non-callable attributes
        for method_name in self.non_callable_methods:
            setattr(self, method_name, "not_a_function")

    def __getattribute__(self, name):
        # Use object's __getattribute__ to avoid infinite recursion
        exclude_methods = object.__getattribute__(self, "exclude_methods")

        # If method is excluded, raise AttributeError
        if name in exclude_methods:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        return object.__getattribute__(self, name)

    @property
    def default_inverse_link_function(self):
        if "default_inverse_link_function" in self.override_returns:
            return lambda x: self.override_returns["default_inverse_link_function"]
        return jax.scipy.special.expit

    def _negative_log_likelihood(self, y, rate, aggregate_sample_scores=jnp.mean):
        if "_negative_log_likelihood" in self.override_returns:
            return self.override_returns["_negative_log_likelihood"]
        return -aggregate_sample_scores(
            y * jax.scipy.special.logit(rate)
            + (1 - y) * jax.scipy.special.logit(1 - rate)
        )

    def pseudo_r2(self, y, rate, aggregate_sample_scores=jnp.mean):
        if "pseudo_r2" in self.override_returns:
            return self.override_returns["pseudo_r2"]
        return 1 - (
            self._negative_log_likelihood(y, rate, aggregate_sample_scores)
            / jnp.sum((y - y.mean()) ** 2)
        )

    def sample_generator(self, key, rate, scale=1.0):
        if "sample_generator" in self.override_returns:
            return self.override_returns["sample_generator"]
        return jax.random.bernoulli(key, rate)

    def estimate_scale(self, y, predicted_rate, dof_resid):
        if "estimate_scale" in self.override_returns:
            return self.override_returns["estimate_scale"]
        return jnp.array(1.0)

    def log_likelihood(self, y, rate, aggregate_sample_scores=jnp.mean):
        if "log_likelihood" in self.override_returns:
            return self.override_returns["log_likelihood"]
        return -self._negative_log_likelihood(y, rate, aggregate_sample_scores)

    def likelihood(self, y, rate, aggregate_sample_scores=jnp.mean):
        if "likelihood" in self.override_returns:
            return self.override_returns["likelihood"]
        return jnp.exp(self.log_likelihood(y, rate, aggregate_sample_scores))

    def deviance(self, y, rate, scale=1.0):
        if "deviance" in self.override_returns:
            return self.override_returns["deviance"]
        identity = lambda x: x
        return 2 * (
            self.log_likelihood(y, rate, identity)
            - self.log_likelihood(y.mean() * np.ones_like(y), rate, identity)
        )


@pytest.fixture
def valid_observation_model():
    """A valid observation model that passes all checks."""
    return ConfigurableObservationModel()


def test_valid_observation_model(valid_observation_model):
    """Test that a valid observation model passes all checks."""
    nmo.observation_models.check_observation_model(valid_observation_model)


@pytest.mark.parametrize("observation", AVAILABLE_OBSERVATION_MODELS)
def test_nemos_model_pass_check(observation):
    """Test that a valid observation model passes all checks."""
    obs = instantiate_observation_model(observation)
    nmo.observation_models.check_observation_model(obs, force_checks=True)


@pytest.mark.parametrize(
    "missing_method",
    [
        "_negative_log_likelihood",
        "pseudo_r2",
        "log_likelihood",
        "likelihood",
        "deviance",
        "sample_generator",
        "estimate_scale",
        "default_inverse_link_function",
    ],
)
def test_missing_method(missing_method):
    """Test that missing required methods raise AttributeError."""
    model = ConfigurableObservationModel(exclude_methods=[missing_method])

    with pytest.raises(AttributeError, match="does not have the required"):
        nmo.observation_models.check_observation_model(model)


@pytest.mark.parametrize(
    "non_callable_method",
    [
        "_negative_log_likelihood",
        "pseudo_r2",
        "log_likelihood",
        "likelihood",
        "deviance",
        "sample_generator",
        "estimate_scale",
    ],
)
def test_non_callable_method(non_callable_method):
    """Test that non-callable methods raise TypeError."""
    model = ConfigurableObservationModel(non_callable_methods=[non_callable_method])

    with pytest.raises(TypeError, match="must be a Callable"):
        nmo.observation_models.check_observation_model(model)


@pytest.mark.parametrize(
    "method_name,return_value",
    [
        ("_negative_log_likelihood", "not an array"),
        ("pseudo_r2", [1, 2, 3]),
        ("log_likelihood", {"key": "value"}),
        ("likelihood", 42),
        ("deviance", "string"),
        ("sample_generator", [1.0, 2.0, 3.0]),
        ("estimate_scale", None),
    ],
)
def test_method_not_returning_array(method_name, return_value):
    """Test that methods not returning ndarray raise TypeError."""
    model = ConfigurableObservationModel(override_returns={method_name: return_value})

    with pytest.raises(TypeError, match="must return a"):
        nmo.observation_models.check_observation_model(model)


@pytest.mark.parametrize(
    "method_name,return_value",
    [
        ("_negative_log_likelihood", jnp.array([1.0, 2.0, 3.0])),
        ("pseudo_r2", jnp.array([0.5, 0.6, 0.7])),
        ("log_likelihood", jnp.array([[1.0], [2.0]])),
        ("likelihood", jnp.array([1.0, 1.0])),
    ],
)
def test_method_not_returning_scalar(method_name, return_value):
    """Test that methods marked as scalar functions return scalars, not vectors."""
    model = ConfigurableObservationModel(override_returns={method_name: return_value})

    with pytest.raises(TypeError, match="should return a scalar"):
        nmo.observation_models.check_observation_model(model)


@pytest.mark.parametrize(
    "return_value,description",
    [
        (jnp.array([1.0, 2.0]), "too few elements"),
        (jnp.array([1.0, 2.0, 3.0, 4.0]), "too many elements"),
        (jnp.array([[1.0, 2.0, 3.0]]), "wrong dimensionality"),
    ],
)
def test_sample_generator_wrong_shape(return_value, description):
    """Test that sample_generator preserves input shape."""
    model = ConfigurableObservationModel(
        override_returns={"sample_generator": return_value}
    )

    with pytest.raises(ValueError, match="preserve the input array"):
        nmo.observation_models.check_observation_model(model)
