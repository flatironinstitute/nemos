import pathlib

import numpy as np
import pytest
from nemos.fetch import fetch_data

from nemos.hmm_glm import run_baum_welch


def test_e_step_regression():
    data_path = fetch_data("e_step_three_states.npz")
    data = np.load(data_path)
    X, y = data["X"], data["y"]
    new_sess = data["new_sess"]

    # E-step initial parameters
    initial_prob = data["initial_prob"]
    transition_matrix = data["transition_matrix"]
    projection_weights = data["projection_weights"]

    # E-step output
    xis = data["xis"]
    gammas = data["gammas"]
    log_likelihood, ll_norm = data["log_likelihood"], data
    alphas, betas = data["alphas"], data["betas"]

    gammas_nemos, xis_nemos, ll_nemos, ll_norm_nemos, alphas_nemos, betas_nemos = (
        run_baum_welch(
            X, y, initial_prob, transition_matrix, projection_weights, new_sess=new_sess
        )
    )
    np.testing.assert_almost_equal(gammas_nemos, gammas)
    np.testing.assert_almost_equal(xis, xis_nemos)
    np.testing.assert_almost_equal(log_likelihood, ll_nemos)
    np.testing.assert_almost_equal(ll_norm, ll_norm_nemos)
    np.testing.assert_almost_equal(alphas, alphas)
    np.testing.assert_almost_equal(betas, betas_nemos)


    # add the alpha, beta and normalized ll
