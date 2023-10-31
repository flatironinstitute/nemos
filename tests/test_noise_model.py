import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats as sts
import statsmodels.api as sm

import neurostatslib as nsl


class TestPoissonNoiseModel:
    cls = nsl.noise_model.PoissonNoiseModel

    @pytest.mark.parametrize("link_function", [jnp.exp, jax.nn.softplus, 1])
    def test_initialization_link_is_callable(self, link_function):
        """Check that the noise model initializes when a callable is passed."""
        raise_exception = not callable(link_function)
        if raise_exception:
            with pytest.raises(TypeError, match="The `inverse_link_function` function must be a Callable"):
                self.cls(link_function)
        else:
            self.cls(link_function)

    @pytest.mark.parametrize("link_function", [jnp.exp, np.exp, lambda x:x, sm.families.links.log()])
    def test_initialization_link_is_jax(self, link_function):
        """Check that the noise model initializes when a callable is passed."""
        raise_exception = isinstance(link_function, np.ufunc) | isinstance(link_function, sm.families.links.Link)
        if raise_exception:
            with pytest.raises(TypeError, match="The `inverse_link_function` must return a jax.numpy.ndarray"):
                self.cls(link_function)
        else:
            self.cls(link_function)

    @pytest.mark.parametrize("link_function", [jnp.exp, jax.nn.softplus, 1])
    def test_initialization_link_is_callable_set_params(self, link_function):
        """Check that the noise model initializes when a callable is passed."""
        noise_model = self.cls()
        raise_exception = not callable(link_function)
        if raise_exception:
            with pytest.raises(TypeError, match="The `inverse_link_function` function must be a Callable"):
                noise_model.set_params(inverse_link_function=link_function)
        else:
            noise_model.set_params(inverse_link_function=link_function)

    @pytest.mark.parametrize("link_function", [jnp.exp, np.exp, lambda x: x, sm.families.links.log()])
    def test_initialization_link_is_jax_set_params(self, link_function):
        """Check that the noise model initializes when a callable is passed."""
        raise_exception = isinstance(link_function, np.ufunc) | isinstance(link_function, sm.families.links.Link)
        noise_model = self.cls()
        if raise_exception:
            with pytest.raises(TypeError, match="The `inverse_link_function` must return a jax.numpy.ndarray!"):
                noise_model.set_params(inverse_link_function=link_function)
        else:
            noise_model.set_params(inverse_link_function=link_function)

    @pytest.mark.parametrize("link_function", [
        jnp.exp,
        lambda x: jnp.exp(x) if isinstance(x, jnp.ndarray) else "not a number"
    ])
    def test_initialization_link_returns_scalar(self, link_function):
        """Check that the noise model initializes when a callable is passed."""
        raise_exception = not isinstance(link_function(1.), (jnp.ndarray, float))
        noise_model = self.cls()
        if raise_exception:
            with pytest.raises(TypeError, match="The `inverse_link_function` must handle scalar inputs correctly"):
                noise_model.set_params(inverse_link_function=link_function)
        else:
            noise_model.set_params(inverse_link_function=link_function)

    def test_deviance_against_statsmodels(self, poissonGLM_model_instantiation):
        """
        Compare fitted parameters to statsmodels.
        Assesses if the model estimates are close to statsmodels' results.
        """
        _, y, model, _, firing_rate = poissonGLM_model_instantiation
        dev = sm.families.Poisson().deviance(y, firing_rate)
        dev_model = model.noise_model.residual_deviance(firing_rate, y).sum()
        if not np.allclose(dev, dev_model):
            raise ValueError("Deviance doesn't match statsmodels!")

    def test_loglikelihood_against_scipy(self, poissonGLM_model_instantiation):
        """
        Compare log-likelihood to scipy.
        Assesses if the model estimates are close to statsmodels' results.
        """
        _, y, model, _, firing_rate = poissonGLM_model_instantiation
        ll_model = - model.noise_model.negative_log_likelihood(firing_rate, y).sum()\
                   - jax.scipy.special.gammaln(y + 1).mean()
        ll_scipy = sts.poisson(firing_rate).logpmf(y).mean()
        if not np.allclose(ll_model, ll_scipy):
             raise ValueError("Log-likelihood doesn't match scipy!")


    def test_pseudo_r2_range(self, poissonGLM_model_instantiation):
        """
        Compute the pseudo-r2 and check that is < 1.
        """
        _, y, model, _, firing_rate = poissonGLM_model_instantiation
        pseudo_r2 = model.noise_model.pseudo_r2(firing_rate, y)
        if (pseudo_r2 > 1) or (pseudo_r2 < 0):
            raise ValueError(f"pseudo-r2 of {pseudo_r2} outside the [0,1] range!")


    def test_pseudo_r2_mean(self, poissonGLM_model_instantiation):
        """
        Check that the pseudo-r2 of the null model is 0.
        """
        _, y, model, _, _ = poissonGLM_model_instantiation
        pseudo_r2 = model.noise_model.pseudo_r2(y.mean(), y)
        if not np.allclose(pseudo_r2, 0):
            raise ValueError(f"pseudo-r2 of {pseudo_r2} for the null model. Should be equal to 0!")

    def test_emission_probability(selfself, poissonGLM_model_instantiation):
        """
        Test the poisson emission probability.

        Check that the emission probability is set to jax.random.poisson.
        """
        _, _, model, _, _ = poissonGLM_model_instantiation
        key_array = jax.random.PRNGKey(123)
        counts = model.noise_model.sample_generator(key_array, np.arange(1, 11))
        if not jnp.all(counts == jax.random.poisson(key_array, np.arange(1, 11))):
            raise ValueError("The emission probability should output the results of a call to jax.random.poisson.")
