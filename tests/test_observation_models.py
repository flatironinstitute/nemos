import warnings
from contextlib import nullcontext as does_not_raise

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy as sp
import scipy.stats as sts
import statsmodels.api as sm
from numba import njit

import nemos as nmo


@pytest.fixture()
def poisson_observations():
    return nmo.observation_models.PoissonObservations


@pytest.fixture()
def gamma_observations():
    return nmo.observation_models.GammaObservations


@pytest.fixture()
def bernoulli_observations():
    return nmo.observation_models.BernoulliObservations


@pytest.mark.parametrize(
    "obs_model_string, expectation",
    [
        ("Poisson", does_not_raise()),
        ("Gamma", does_not_raise()),
        ("Bernoulli", does_not_raise()),
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
    if obs_model_string == "Gamma":
        assert isinstance(
            model.observation_model, nmo.observation_models.GammaObservations
        )
    elif obs_model_string == "Poisson":
        assert isinstance(
            model.observation_model, nmo.observation_models.PoissonObservations
        )
    elif obs_model_string == "Bernoulli":
        assert isinstance(
            model.observation_model, nmo.observation_models.BernoulliObservations
        )


class TestPoissonObservations:
    @pytest.mark.parametrize("link_function", [jnp.exp, jax.nn.softplus, 1])
    def test_initialization_link_is_callable(self, link_function, poisson_observations):
        """Check that the observation model initializes when a callable is passed."""
        raise_exception = not callable(link_function)
        if raise_exception:
            with pytest.raises(
                TypeError,
                match="The `inverse_link_function` function must be a Callable",
            ):
                poisson_observations(link_function)
        else:
            poisson_observations(link_function)

    @pytest.mark.parametrize(
        "link_function", [jnp.exp, np.exp, lambda x: x, sm.families.links.Log()]
    )
    def test_initialization_link_is_jax(self, link_function, poisson_observations):
        """Check that the observation model initializes when a callable is passed."""
        raise_exception = isinstance(link_function, np.ufunc) | isinstance(
            link_function, sm.families.links.Link
        )
        if raise_exception:
            with pytest.raises(
                TypeError,
                match="The `inverse_link_function` must return a jax.numpy.ndarray",
            ):
                poisson_observations(link_function)
        else:
            poisson_observations(link_function)

    @pytest.mark.parametrize("link_function", [jnp.exp, jax.nn.softplus, 1])
    def test_initialization_link_is_callable_set_params(
        self, link_function, poisson_observations
    ):
        """Check that the observation model initializes when a callable is passed."""
        observation_model = poisson_observations()
        raise_exception = not callable(link_function)
        if raise_exception:
            with pytest.raises(
                TypeError,
                match="The `inverse_link_function` function must be a Callable",
            ):
                observation_model.set_params(inverse_link_function=link_function)
        else:
            observation_model.set_params(inverse_link_function=link_function)

    @pytest.mark.parametrize(
        "link_function", [jnp.exp, np.exp, lambda x: x, sm.families.links.Log()]
    )
    def test_initialization_link_is_jax_set_params(
        self, link_function, poisson_observations
    ):
        """Check that the observation model initializes when a callable is passed."""
        raise_exception = isinstance(link_function, np.ufunc) | isinstance(
            link_function, sm.families.links.Link
        )
        observation_model = poisson_observations()
        if raise_exception:
            with pytest.raises(
                TypeError,
                match="The `inverse_link_function` must return a jax.numpy.ndarray!",
            ):
                observation_model.set_params(inverse_link_function=link_function)
        else:
            observation_model.set_params(inverse_link_function=link_function)

    @pytest.mark.parametrize(
        "link_function",
        [
            jnp.exp,
            lambda x: jnp.exp(x) if isinstance(x, jnp.ndarray) else "not a number",
        ],
    )
    def test_initialization_link_returns_scalar(
        self, link_function, poisson_observations
    ):
        """Check that the observation model initializes when a callable is passed."""
        raise_exception = not isinstance(link_function(1.0), (jnp.ndarray, float))
        observation_model = poisson_observations()
        if raise_exception:
            with pytest.raises(
                TypeError,
                match="The `inverse_link_function` must handle scalar inputs correctly",
            ):
                observation_model.set_params(inverse_link_function=link_function)
        else:
            observation_model.set_params(inverse_link_function=link_function)

    def test_get_params(self, poisson_observations):
        """Test get_params() returns expected values."""
        observation_model = poisson_observations()

        assert observation_model.get_params() == {
            "inverse_link_function": observation_model.inverse_link_function
        }

    def test_deviance_against_statsmodels(self, poissonGLM_model_instantiation):
        """
        Compare fitted parameters to statsmodels.
        Assesses if the model estimates are close to statsmodels' results.
        """
        _, y, model, _, firing_rate = poissonGLM_model_instantiation
        dev = sm.families.Poisson().deviance(y, firing_rate)
        with model.observation_model._unlinked_rate():
            dev_model = model.observation_model.deviance(y, firing_rate).sum()
        if not np.allclose(dev, dev_model):
            raise ValueError("Deviance doesn't match statsmodels!")

    def test_loglikelihood_against_scipy(self, poissonGLM_model_instantiation):
        """
        Compare log-likelihood to scipy.
        Assesses if the model estimates are close to statsmodels' results.
        """
        _, y, model, _, firing_rate = poissonGLM_model_instantiation
        with model.observation_model._unlinked_rate():
            ll_model = model.observation_model.log_likelihood(y, firing_rate)
        ll_scipy = sts.poisson(firing_rate).logpmf(y).mean()
        if not np.allclose(ll_model, ll_scipy):
            raise ValueError("Log-likelihood doesn't match scipy!")

    @pytest.mark.parametrize("score_type", ["pseudo-r2-Cohen", "pseudo-r2-McFadden"])
    def test_pseudo_r2_range(self, score_type, poissonGLM_model_instantiation):
        """
        Compute the pseudo-r2 and check that is < 1.
        """
        _, y, model, _, firing_rate = poissonGLM_model_instantiation
        with model.observation_model._unlinked_rate():
            pseudo_r2 = model.observation_model.pseudo_r2(
                y, firing_rate, score_type=score_type
            )
        if (pseudo_r2 > 1) or (pseudo_r2 < 0):
            raise ValueError(f"pseudo-r2 of {pseudo_r2} outside the [0,1] range!")

    @pytest.mark.parametrize("score_type", ["pseudo-r2-Cohen", "pseudo-r2-McFadden"])
    def test_pseudo_r2_mean(self, score_type, poissonGLM_model_instantiation):
        """
        Check that the pseudo-r2 of the null model is 0.
        """
        _, y, model, _, _ = poissonGLM_model_instantiation
        with model.observation_model._unlinked_rate():
            pseudo_r2 = model.observation_model.pseudo_r2(
                y, y.mean(), score_type=score_type
            )
        if not np.allclose(pseudo_r2, 0, atol=10**-7, rtol=0.0):
            raise ValueError(
                f"pseudo-r2 of {pseudo_r2} for the null model. Should be equal to 0!"
            )

    def test_emission_probability(selfself, poissonGLM_model_instantiation):
        """
        Test the poisson emission probability.

        Check that the emission probability is set to jax.random.poisson.
        """
        _, _, model, _, _ = poissonGLM_model_instantiation
        key_array = jax.random.key(123)
        with model.observation_model._unlinked_rate():
            counts = model.observation_model.sample_generator(
                key_array, np.arange(1, 11)
            )
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

    def test_non_differentiable_inverse_link(self, poissonGLM_model_instantiation):
        _, _, model, _, _ = poissonGLM_model_instantiation

        # define a jax non-diff function
        non_diff = lambda y: jnp.asarray(njit(lambda x: x)(np.atleast_1d(y)))

        with pytest.raises(
            TypeError,
            match="The `inverse_link_function` function cannot be differentiated",
        ):
            model.observation_model.inverse_link_function = non_diff

    def test_pseudo_r2_vs_statsmodels(self, poissonGLM_model_instantiation):
        """
        Compare log-likelihood to scipy.
        Assesses if the model estimates are close to statsmodels' results.
        """
        X, y, model, _, firing_rate = poissonGLM_model_instantiation

        # statsmodels mcfadden
        mdl = sm.GLM(y, sm.add_constant(X), family=sm.families.Poisson()).fit()
        pr2_sms = mdl.pseudo_rsquared("mcf")

        # assume link is provided
        with model.observation_model._unlinked_rate():
            pr2_model = model.observation_model.pseudo_r2(
                y, mdl.mu, score_type="pseudo-r2-McFadden"
            )

        if not np.allclose(pr2_model, pr2_sms):
            raise ValueError("Log-likelihood doesn't match statsmodels!")

    def test_aggregation_score_neg_ll(self, poissonGLM_model_instantiation):
        X, y, model, _, firing_rate = poissonGLM_model_instantiation
        sm = model.observation_model._negative_log_likelihood(y, firing_rate, jnp.sum)
        mn = model.observation_model._negative_log_likelihood(y, firing_rate, jnp.mean)
        assert np.allclose(sm, mn * y.shape[0])

    def test_aggregation_score_ll(self, poissonGLM_model_instantiation):
        X, y, model, _, firing_rate = poissonGLM_model_instantiation
        sm = model.observation_model.log_likelihood(
            y, firing_rate, aggregate_sample_scores=jnp.sum
        )
        mn = model.observation_model.log_likelihood(
            y, firing_rate, aggregate_sample_scores=jnp.mean
        )
        assert np.allclose(sm, mn * y.shape[0])

    @pytest.mark.parametrize("score_type", ["pseudo-r2-McFadden", "pseudo-r2-Cohen"])
    def test_aggregation_score_pr2(self, score_type, poissonGLM_model_instantiation):
        X, y, model, _, firing_rate = poissonGLM_model_instantiation
        sm = model.observation_model.pseudo_r2(
            y, firing_rate, score_type=score_type, aggregate_sample_scores=jnp.sum
        )
        mn = model.observation_model.pseudo_r2(
            y, firing_rate, score_type=score_type, aggregate_sample_scores=jnp.mean
        )
        assert np.allclose(sm, mn)

    def test_aggregation_score_mcfadden(self, poissonGLM_model_instantiation):
        X, y, model, _, firing_rate = poissonGLM_model_instantiation
        sm = model.observation_model._pseudo_r2_mcfadden(
            y, firing_rate, aggregate_sample_scores=jnp.sum
        )
        mn = model.observation_model._pseudo_r2_mcfadden(
            y, firing_rate, aggregate_sample_scores=jnp.mean
        )
        assert np.allclose(sm, mn)

    def test_aggregation_score_choen(self, poissonGLM_model_instantiation):
        X, y, model, _, firing_rate = poissonGLM_model_instantiation
        sm = model.observation_model._pseudo_r2_cohen(
            y, firing_rate, aggregate_sample_scores=jnp.sum
        )
        mn = model.observation_model._pseudo_r2_cohen(
            y, firing_rate, aggregate_sample_scores=jnp.mean
        )
        assert np.allclose(sm, mn)

    @pytest.mark.parametrize(
        "link_func, link_func_name", [(jnp.exp, "exp"), (jax.nn.softplus, "softplus")]
    )
    def test_repr_out(self, link_func, link_func_name):
        obs = nmo.observation_models.PoissonObservations(
            inverse_link_function=link_func
        )
        assert (
            repr(obs) == f"PoissonObservations(inverse_link_function={link_func_name})"
        )


class TestGammaObservations:
    @pytest.mark.parametrize("link_function", [jnp.exp, lambda x: 1 / x, 1])
    def test_initialization_link_is_callable(self, link_function, gamma_observations):
        """Check that the observation model initializes when a callable is passed."""
        raise_exception = not callable(link_function)
        if raise_exception:
            with pytest.raises(
                TypeError,
                match="The `inverse_link_function` function must be a Callable",
            ):
                gamma_observations(link_function)
        else:
            gamma_observations(link_function)

    @pytest.mark.parametrize(
        "link_function", [jnp.exp, np.exp, lambda x: 1 / x, sm.families.links.Log()]
    )
    def test_initialization_link_is_jax(self, link_function, gamma_observations):
        """Check that the observation model initializes when a callable is passed."""
        raise_exception = isinstance(link_function, np.ufunc) | isinstance(
            link_function, sm.families.links.Link
        )
        if raise_exception:
            with pytest.raises(
                TypeError,
                match="The `inverse_link_function` must return a jax.numpy.ndarray",
            ):
                gamma_observations(link_function)
        else:
            gamma_observations(link_function)

    @pytest.mark.parametrize("link_function", [jnp.exp, jax.nn.softplus, 1])
    def test_initialization_link_is_callable_set_params(
        self, link_function, gamma_observations
    ):
        """Check that the observation model initializes when a callable is passed."""
        observation_model = gamma_observations()
        raise_exception = not callable(link_function)
        if raise_exception:
            with pytest.raises(
                TypeError,
                match="The `inverse_link_function` function must be a Callable",
            ):
                observation_model.set_params(inverse_link_function=link_function)
        else:
            observation_model.set_params(inverse_link_function=link_function)

    @pytest.mark.parametrize(
        "link_function", [jnp.exp, np.exp, lambda x: x, sm.families.links.Log()]
    )
    def test_initialization_link_is_jax_set_params(
        self, link_function, gamma_observations
    ):
        """Check that the observation model initializes when a callable is passed."""
        raise_exception = isinstance(link_function, np.ufunc) | isinstance(
            link_function, sm.families.links.Link
        )
        observation_model = gamma_observations()
        if raise_exception:
            with pytest.raises(
                TypeError,
                match="The `inverse_link_function` must return a jax.numpy.ndarray!",
            ):
                observation_model.set_params(inverse_link_function=link_function)
        else:
            observation_model.set_params(inverse_link_function=link_function)

    @pytest.mark.parametrize(
        "link_function",
        [
            jnp.exp,
            lambda x: jnp.exp(x) if isinstance(x, jnp.ndarray) else "not a number",
        ],
    )
    def test_initialization_link_returns_scalar(
        self, link_function, gamma_observations
    ):
        """Check that the observation model initializes when a callable is passed."""
        raise_exception = not isinstance(link_function(1.0), (jnp.ndarray, float))
        observation_model = gamma_observations()
        if raise_exception:
            with pytest.raises(
                TypeError,
                match="The `inverse_link_function` must handle scalar inputs correctly",
            ):
                observation_model.set_params(inverse_link_function=link_function)
        else:
            observation_model.set_params(inverse_link_function=link_function)

    def test_get_params(self, gamma_observations):
        """Test get_params() returns expected values."""
        observation_model = gamma_observations()

        assert observation_model.get_params() == {
            "inverse_link_function": observation_model.inverse_link_function
        }

    def test_deviance_against_statsmodels(self, gammaGLM_model_instantiation):
        """
        Compare fitted parameters to statsmodels.
        Assesses if the model estimates are close to statsmodels' results.
        """
        _, y, model, _, firing_rate = gammaGLM_model_instantiation
        dev = sm.families.Gamma().deviance(y, firing_rate)
        with model.observation_model._unlinked_rate():
            dev_model = model.observation_model.deviance(y, firing_rate).sum()
        if not np.allclose(dev, dev_model):
            raise ValueError("Deviance doesn't match statsmodels!")

    def test_loglikelihood_against_statsmodels(self, gammaGLM_model_instantiation):
        """
        Compare log-likelihood to scipy.
        Assesses if the model estimates are close to statsmodels' results.
        """
        _, y, model, _, firing_rate = gammaGLM_model_instantiation
        with model.observation_model._unlinked_rate():
            ll_model = model.observation_model.log_likelihood(y, firing_rate)
        ll_sms = sm.families.Gamma().loglike(y, firing_rate) / y.shape[0]
        if not np.allclose(ll_model, ll_sms):
            raise ValueError("Log-likelihood doesn't match statsmodels!")

    @pytest.mark.parametrize("score_type", ["pseudo-r2-Cohen", "pseudo-r2-McFadden"])
    def test_pseudo_r2_range(self, score_type, gammaGLM_model_instantiation):
        """
        Compute the pseudo-r2 and check that is < 1.
        """
        X, y, model, true_params, firing_rate = gammaGLM_model_instantiation
        model.coef_ = true_params[0]
        model.intercept_ = true_params[1]
        model.scale_ = 0.5

        rate = model.predict(X)
        ysim, _ = model.simulate(jax.random.PRNGKey(123), X)
        with model.observation_model._unlinked_rate():
            pseudo_r2 = nmo.observation_models.GammaObservations(
                inverse_link_function=lambda x: 1 / x
            ).pseudo_r2(ysim, rate, score_type=score_type)
        if (pseudo_r2 > 1) or (pseudo_r2 < 0):
            raise ValueError(f"pseudo-r2 of {pseudo_r2} outside the [0,1] range!")

    @pytest.mark.parametrize("score_type", ["pseudo-r2-Cohen", "pseudo-r2-McFadden"])
    def test_pseudo_r2_mean(self, score_type, gammaGLM_model_instantiation):
        """
        Check that the pseudo-r2 of the null model is 0.
        """
        _, y, model, _, _ = gammaGLM_model_instantiation
        with model.observation_model._unlinked_rate():
            pseudo_r2 = model.observation_model.pseudo_r2(
                y, y.mean(), score_type=score_type
            )
        if not np.allclose(pseudo_r2, 0, atol=10**-7, rtol=0.0):
            raise ValueError(
                f"pseudo-r2 of {pseudo_r2} for the null model. Should be equal to 0!"
            )

    def test_emission_probability(self, gammaGLM_model_instantiation):
        """
        Test the gamma emission probability.

        Check that the emission probability is set to jax.random.gamma.
        """
        _, _, model, _, _ = gammaGLM_model_instantiation
        key_array = jax.random.key(123)
        with model.observation_model._unlinked_rate():
            counts = model.observation_model.sample_generator(
                key_array, np.arange(1, 11)
            )
        if not jnp.all(counts == jax.random.gamma(key_array, np.arange(1, 11))):
            raise ValueError(
                "The emission probability should output the results of a call to jax.random.gamma."
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
        self, score_type, expectation, gammaGLM_model_instantiation
    ):
        _, y, model, _, firing_rate = gammaGLM_model_instantiation
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
    def test_scale_setter(self, scale, expectation, gammaGLM_model_instantiation):
        _, _, model, _, firing_rate = gammaGLM_model_instantiation
        with expectation:
            model.observation_model.scale = scale

    def test_scale_getter(self, gammaGLM_model_instantiation):
        _, _, model, _, firing_rate = gammaGLM_model_instantiation
        assert model.observation_model.scale == 1

    def test_non_differentiable_inverse_link(self, gammaGLM_model_instantiation):
        _, _, model, _, _ = gammaGLM_model_instantiation

        # define a jax non-diff function
        non_diff = lambda y: jnp.asarray(njit(lambda x: x)(np.atleast_1d(y)))

        with pytest.raises(
            TypeError,
            match="The `inverse_link_function` function cannot be differentiated",
        ):
            model.observation_model.inverse_link_function = non_diff

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
        with model.observation_model._unlinked_rate():
            pr2_model = model.observation_model.pseudo_r2(
                y, mdl.mu, score_type="pseudo-r2-McFadden", scale=mdl.scale
            )

        if not np.allclose(pr2_model, pr2_sms):
            raise ValueError("Log-likelihood doesn't match statsmodels!")

    def test_aggregation_score_neg_ll(self, gammaGLM_model_instantiation):
        X, y, model, _, firing_rate = gammaGLM_model_instantiation
        sm = model.observation_model._negative_log_likelihood(y, firing_rate, jnp.sum)
        mn = model.observation_model._negative_log_likelihood(y, firing_rate, jnp.mean)
        assert np.allclose(sm, mn * y.shape[0])

    def test_aggregation_score_ll(self, gammaGLM_model_instantiation):
        X, y, model, _, firing_rate = gammaGLM_model_instantiation
        sm = model.observation_model.log_likelihood(
            y, firing_rate, aggregate_sample_scores=jnp.sum
        )
        mn = model.observation_model.log_likelihood(
            y, firing_rate, aggregate_sample_scores=jnp.mean
        )
        assert np.allclose(sm, mn * y.shape[0])

    @pytest.mark.parametrize("score_type", ["pseudo-r2-McFadden", "pseudo-r2-Cohen"])
    def test_aggregation_score_pr2(self, score_type, gammaGLM_model_instantiation):
        X, y, model, _, firing_rate = gammaGLM_model_instantiation
        sm = model.observation_model.pseudo_r2(
            y, firing_rate, score_type=score_type, aggregate_sample_scores=jnp.sum
        )
        mn = model.observation_model.pseudo_r2(
            y, firing_rate, score_type=score_type, aggregate_sample_scores=jnp.mean
        )
        assert np.allclose(sm, mn)

    def test_aggregation_score_mcfadden(self, gammaGLM_model_instantiation):
        X, y, model, _, firing_rate = gammaGLM_model_instantiation
        sm = model.observation_model._pseudo_r2_mcfadden(
            y, firing_rate, aggregate_sample_scores=jnp.sum
        )
        mn = model.observation_model._pseudo_r2_mcfadden(
            y, firing_rate, aggregate_sample_scores=jnp.mean
        )
        assert np.allclose(sm, mn)

    def test_aggregation_score_choen(self, gammaGLM_model_instantiation):
        X, y, model, _, firing_rate = gammaGLM_model_instantiation
        sm = model.observation_model._pseudo_r2_cohen(
            y, firing_rate, aggregate_sample_scores=jnp.sum
        )
        mn = model.observation_model._pseudo_r2_cohen(
            y, firing_rate, aggregate_sample_scores=jnp.mean
        )
        assert np.allclose(sm, mn)

    @pytest.mark.parametrize(
        "link_func, link_func_name",
        [
            (jnp.exp, "exp"),
            (jax.nn.softplus, "softplus"),
            (nmo.utils.one_over_x, "one_over_x"),
        ],
    )
    def test_repr_out(self, link_func, link_func_name):
        obs = nmo.observation_models.GammaObservations(inverse_link_function=link_func)
        assert repr(obs) == f"GammaObservations(inverse_link_function={link_func_name})"


class TestBernoulliObservations:
    @pytest.mark.parametrize(
        "link_function",
        [jax.lax.logistic, jax.scipy.special.expit, jax.scipy.stats.norm.cdf, 1],
    )
    def test_initialization_link_is_callable(
        self, link_function, bernoulli_observations
    ):
        """Check that the observation model initializes when a callable is passed."""
        raise_exception = not callable(link_function)
        if raise_exception:
            with pytest.raises(
                TypeError,
                match="The `inverse_link_function` function must be a Callable",
            ):
                bernoulli_observations(link_function)
        else:
            bernoulli_observations(link_function)

    @pytest.mark.parametrize(
        "link_function",
        [
            jax.scipy.special.expit,
            sp.special.expit,
            jax.scipy.stats.norm.cdf,
            sts.norm.cdf,
            np.exp,
            lambda x: x,
            sm.families.links.Log(),
        ],
    )
    def test_initialization_link_is_jax(self, link_function, bernoulli_observations):
        """Check that the observation model initializes when a callable is passed."""
        raise_exception = (
            isinstance(link_function, np.ufunc)
            | isinstance(link_function, sm.families.links.Link)
            | isinstance(link_function, type(sts.norm.cdf))
        )
        if raise_exception:
            with pytest.raises(
                TypeError,
                match="The `inverse_link_function` must return a jax.numpy.ndarray",
            ):
                bernoulli_observations(link_function)
        else:
            bernoulli_observations(link_function)

    @pytest.mark.parametrize(
        "link_function",
        [jax.lax.logistic, jax.scipy.special.expit, jax.scipy.stats.norm.cdf, 1],
    )
    def test_initialization_link_is_callable_set_params(
        self, link_function, bernoulli_observations
    ):
        """Check that the observation model initializes when a callable is passed."""
        observation_model = bernoulli_observations()
        raise_exception = not callable(link_function)
        if raise_exception:
            with pytest.raises(
                TypeError,
                match="The `inverse_link_function` function must be a Callable",
            ):
                observation_model.set_params(inverse_link_function=link_function)
        else:
            observation_model.set_params(inverse_link_function=link_function)

    @pytest.mark.parametrize(
        "link_function",
        [
            jax.scipy.special.expit,
            sp.special.expit,
            jax.scipy.stats.norm.cdf,
            sts.norm.cdf,
            np.exp,
            lambda x: x,
            sm.families.links.Log(),
        ],
    )
    def test_initialization_link_is_jax_set_params(
        self, link_function, bernoulli_observations
    ):
        """Check that the observation model initializes when a callable is passed."""
        raise_exception = (
            isinstance(link_function, np.ufunc)
            | isinstance(link_function, sm.families.links.Link)
            | isinstance(link_function, type(sts.norm.cdf))
        )
        observation_model = bernoulli_observations()
        if raise_exception:
            with pytest.raises(
                TypeError,
                match="The `inverse_link_function` must return a jax.numpy.ndarray!",
            ):
                observation_model.set_params(inverse_link_function=link_function)
        else:
            observation_model.set_params(inverse_link_function=link_function)

    @pytest.mark.parametrize(
        "link_function",
        [
            jax.lax.logistic,
            lambda x: (
                jax.lax.logistic(x) if isinstance(x, jnp.ndarray) else "not a number"
            ),
        ],
    )
    def test_initialization_link_returns_scalar(
        self, link_function, bernoulli_observations
    ):
        """Check that the observation model initializes when a callable is passed."""
        raise_exception = not isinstance(link_function(1.0), (jnp.ndarray, float))
        observation_model = bernoulli_observations()
        if raise_exception:
            with pytest.raises(
                TypeError,
                match="The `inverse_link_function` must handle scalar inputs correctly",
            ):
                observation_model.set_params(inverse_link_function=link_function)
        else:
            observation_model.set_params(inverse_link_function=link_function)

    def test_get_params(self, bernoulli_observations):
        """Test get_params() returns expected values."""
        observation_model = bernoulli_observations()

        assert observation_model.get_params() == {
            "inverse_link_function": observation_model.inverse_link_function
        }

    def test_deviance_against_statsmodels(self, bernoulliGLM_model_instantiation):
        """
        Compare fitted parameters to statsmodels.
        Assesses if the model estimates are close to statsmodels' results.
        """
        _, y, model, _, firing_rate = bernoulliGLM_model_instantiation
        dev = sm.families.Binomial().deviance(y, firing_rate)
        with model.observation_model._unlinked_rate():
            dev_model = model.observation_model.deviance(y, firing_rate).sum()
        if not np.allclose(dev, dev_model):
            raise ValueError("Deviance doesn't match statsmodels!")

    def test_loglikelihood_against_scipy(self, bernoulliGLM_model_instantiation):
        """
        Compare log-likelihood to scipy.
        Assesses if the model estimates are close to statsmodels' results.
        """
        _, y, model, _, firing_rate = bernoulliGLM_model_instantiation
        with model.observation_model._unlinked_rate():
            ll_model = model.observation_model.log_likelihood(y, firing_rate)
        ll_scipy = sts.bernoulli(firing_rate).logpmf(y).mean()
        if not np.allclose(ll_model, ll_scipy):
            raise ValueError("Log-likelihood doesn't match scipy!")

    @pytest.mark.parametrize("score_type", ["pseudo-r2-Cohen", "pseudo-r2-McFadden"])
    def test_pseudo_r2_range(self, score_type, bernoulliGLM_model_instantiation):
        """
        Compute the pseudo-r2 and check that is < 1.
        """
        _, y, model, _, firing_rate = bernoulliGLM_model_instantiation
        with model.observation_model._unlinked_rate():
            pseudo_r2 = model.observation_model.pseudo_r2(
                y, firing_rate, score_type=score_type
            )
        if (pseudo_r2 > 1) or (pseudo_r2 < 0):
            raise ValueError(f"pseudo-r2 of {pseudo_r2} outside the [0,1] range!")

    @pytest.mark.parametrize("score_type", ["pseudo-r2-Cohen", "pseudo-r2-McFadden"])
    def test_pseudo_r2_mean(self, score_type, bernoulliGLM_model_instantiation):
        """
        Check that the pseudo-r2 of the null model is 0.
        """
        _, y, model, _, _ = bernoulliGLM_model_instantiation
        with model.observation_model._unlinked_rate():
            pseudo_r2 = model.observation_model.pseudo_r2(
                y, y.mean(), score_type=score_type
            )
        # fails with atol=10**-7
        if not np.allclose(pseudo_r2, 0, atol=10**-6, rtol=0.0):
            raise ValueError(
                f"pseudo-r2 of {pseudo_r2} for the null model. Should be equal to 0!"
            )

    def test_emission_probability(selfself, bernoulliGLM_model_instantiation):
        """
        Test the poisson emission probability.

        Check that the emission probability is set to jax.random.poisson.
        """
        _, _, model, _, _ = bernoulliGLM_model_instantiation
        key_array = jax.random.key(123)
        p = np.random.rand(10)
        with model.observation_model._unlinked_rate():
            counts = model.observation_model.sample_generator(key_array, p)
        if not jnp.all(counts == jax.random.bernoulli(key_array, p)):
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
        self, score_type, expectation, bernoulliGLM_model_instantiation
    ):
        _, y, model, _, firing_rate = bernoulliGLM_model_instantiation
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
    def test_scale_setter(self, scale, expectation, bernoulliGLM_model_instantiation):
        _, _, model, _, firing_rate = bernoulliGLM_model_instantiation
        with expectation:
            model.observation_model.scale = scale

    def test_scale_getter(self, bernoulliGLM_model_instantiation):
        _, _, model, _, firing_rate = bernoulliGLM_model_instantiation
        assert model.observation_model.scale == 1

    def test_non_differentiable_inverse_link(self, bernoulliGLM_model_instantiation):
        _, _, model, _, _ = bernoulliGLM_model_instantiation

        # define a jax non-diff function
        non_diff = lambda y: jnp.asarray(njit(lambda x: x)(np.atleast_1d(y)))

        with pytest.raises(
            TypeError,
            match="The `inverse_link_function` function cannot be differentiated",
        ):
            model.observation_model.inverse_link_function = non_diff

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
        with model.observation_model._unlinked_rate():
            pr2_model = model.observation_model.pseudo_r2(
                y, mdl.mu, score_type="pseudo-r2-McFadden"
            )

        if not np.allclose(pr2_model, pr2_sms):
            raise ValueError("Log-likelihood doesn't match statsmodels!")

    def test_aggregation_score_neg_ll(self, bernoulliGLM_model_instantiation):
        X, y, model, _, firing_rate = bernoulliGLM_model_instantiation
        sm = model.observation_model._negative_log_likelihood(y, firing_rate, jnp.sum)
        mn = model.observation_model._negative_log_likelihood(y, firing_rate, jnp.mean)
        assert np.allclose(sm, mn * y.shape[0])

    def test_aggregation_score_ll(self, bernoulliGLM_model_instantiation):
        X, y, model, _, firing_rate = bernoulliGLM_model_instantiation
        sm = model.observation_model.log_likelihood(
            y, firing_rate, aggregate_sample_scores=jnp.sum
        )
        mn = model.observation_model.log_likelihood(
            y, firing_rate, aggregate_sample_scores=jnp.mean
        )
        assert np.allclose(sm, mn * y.shape[0])

    @pytest.mark.parametrize("score_type", ["pseudo-r2-McFadden", "pseudo-r2-Cohen"])
    def test_aggregation_score_pr2(self, score_type, bernoulliGLM_model_instantiation):
        X, y, model, _, firing_rate = bernoulliGLM_model_instantiation
        sm = model.observation_model.pseudo_r2(
            y, firing_rate, score_type=score_type, aggregate_sample_scores=jnp.sum
        )
        mn = model.observation_model.pseudo_r2(
            y, firing_rate, score_type=score_type, aggregate_sample_scores=jnp.mean
        )
        assert np.allclose(sm, mn)

    def test_aggregation_score_mcfadden(self, bernoulliGLM_model_instantiation):
        X, y, model, _, firing_rate = bernoulliGLM_model_instantiation
        sm = model.observation_model._pseudo_r2_mcfadden(
            y, firing_rate, aggregate_sample_scores=jnp.sum
        )
        mn = model.observation_model._pseudo_r2_mcfadden(
            y, firing_rate, aggregate_sample_scores=jnp.mean
        )
        assert np.allclose(sm, mn)

    def test_aggregation_score_choen(self, bernoulliGLM_model_instantiation):
        X, y, model, _, firing_rate = bernoulliGLM_model_instantiation
        sm = model.observation_model._pseudo_r2_cohen(
            y, firing_rate, aggregate_sample_scores=jnp.sum
        )
        mn = model.observation_model._pseudo_r2_cohen(
            y, firing_rate, aggregate_sample_scores=jnp.mean
        )
        assert np.allclose(sm, mn)

    @pytest.mark.parametrize(
        "link_func, link_func_name",
        [
            (jax.lax.logistic, "logistic"),
            (jax.scipy.special.expit, "expit"),
            (jax.scipy.stats.norm.cdf, "norm.cdf"),
        ],
    )
    def test_repr_out(self, link_func, link_func_name):
        obs = nmo.observation_models.BernoulliObservations(
            inverse_link_function=link_func
        )
        assert (
            repr(obs)
            == f"BernoulliObservations(inverse_link_function={link_func_name})"
        )
