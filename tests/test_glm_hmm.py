import jax
import numpy as np
import pytest

from nemos.fetch import fetch_data
from nemos.glm_hmm import forward_backward
from nemos.observation_models import BernoulliObservations


def test_e_step_regression():
    # Fetch the data
    data_path = fetch_data("e_step_three_states.npz")
    data = np.load(data_path)

    X, y = data["X"], data["y"]
    new_sess = data["new_sess"]

    # E-step initial parameters
    initial_prob = data["initial_prob"]
    latent_weights = data["projection_weights"]
    transition_matrix = data["transition_matrix"]

    # E-step output
    xis = data["xis"]
    gammas = data["gammas"]
    log_likelihood, ll_norm = data["log_likelihood"], data["ll_norm"]
    alphas, betas = data["alphas"], data["betas"]

    obs = BernoulliObservations()

    log_likelihood = lambda x, y: jax.vmap(obs.log_likelihood())
    gammas_nemos, xis_nemos, ll_nemos, ll_norm_nemos, alphas_nemos, betas_nemos = (
        forward_backward(
            X,
            y.flatten(),
            initial_prob,
            transition_matrix,
            latent_weights,
            is_new_session=new_sess.flatten().astype(bool),
        )
    )

    print(log_likelihood, ll_nemos)
    print(f"\n{ll_nemos.shape}\n{log_likelihood.shape}")

    # First testing alphas and betas because they are computed first
    np.testing.assert_almost_equal(alphas.T, alphas_nemos, decimal=4)
    np.testing.assert_almost_equal(betas.T, betas_nemos, decimal=4)
    # Next testing xis and gammas because they depend on alphas and betas
    # Equations 13.43 and 13.65 of Bishop
    np.testing.assert_almost_equal(gammas, gammas_nemos, decimal=4)
    np.testing.assert_almost_equal(xis, xis_nemos, decimal=4)
    # Finally testing log likelihood and normalized log likelihood
    np.testing.assert_almost_equal(log_likelihood, ll_nemos, decimal=4)
    np.testing.assert_almost_equal(ll_norm, ll_norm_nemos, decimal=4)
