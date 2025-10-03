from functools import partial

import jax
import numpy as np
import pytest

from nemos.fetch import fetch_data
from nemos.glm_hmm.expectation_maximization import forward_backward
from nemos.observation_models import BernoulliObservations


# Below are tests for Bernoulli observation 3 states model glm hmm
@pytest.mark.parametrize(
    "decorator",
    [
        lambda x: x,
        partial(jax.jit, static_argnames=["likelihood_func", "inverse_link_function"]),
    ],
)
def test_forward_backward_regression(decorator):
    jax.config.update("jax_enable_x64", True)

    # Fetch the data
    data_path = fetch_data("em_three_states.npz")
    data = np.load(data_path)

    X, y = data["X"], data["y"]
    new_sess = data["new_sess"]

    # E-step initial parameters
    initial_prob = data["initial_prob"]
    projection_weights = data["projection_weights"]
    transition_prob = data["transition_prob"]

    # E-step output
    xis = data["xis"]
    gammas = data["gammas"]
    ll_orig, ll_norm_orig = data["log_likelihood"], data["ll_norm"]
    alphas, betas = data["alphas"], data["betas"]

    obs = BernoulliObservations()

    likelihood = jax.vmap(
        lambda x, z: obs.likelihood(x, z, aggregate_sample_scores=lambda w: w),
        in_axes=(None, 1),
        out_axes=1,
    )

    decorated_forward_backward = decorator(forward_backward)
    gammas_nemos, xis_nemos, ll_nemos, ll_norm_nemos, alphas_nemos, betas_nemos = (
        decorated_forward_backward(
            X,
            y,
            initial_prob,
            transition_prob,
            projection_weights,
            likelihood_func=likelihood,
            inverse_link_function=obs.default_inverse_link_function,
            is_new_session=new_sess.astype(bool),
        )
    )

    # First testing alphas and betas because they are computed first
    np.testing.assert_almost_equal(alphas_nemos, alphas, decimal=8)
    np.testing.assert_almost_equal(betas_nemos, betas, decimal=8)

    # testing log likelihood and normalized log likelihood
    np.testing.assert_almost_equal(ll_nemos, ll_orig, decimal=8)
    np.testing.assert_almost_equal(ll_norm_nemos, ll_norm_orig, decimal=8)

    # Next testing xis and gammas because they depend on alphas and betas
    # Testing Eq. 13.43 of Bishop
    np.testing.assert_almost_equal(gammas_nemos, gammas, decimal=8)
    # Testing Eq. 13.65 of Bishop
    np.testing.assert_almost_equal(xis_nemos, xis, decimal=8)
