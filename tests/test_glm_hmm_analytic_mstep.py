import jax.numpy as jnp
import numpy as np
import pytest
from optimistix import LBFGS, minimise

import nemos as nmo
from nemos.glm.params import GLMParams
from nemos.glm_hmm.algorithm_configs import (
    get_analytical_scale_update,
    prepare_ll_estep_likelihood,
    prepare_objective_mstep_numerical_scale,
)
from nemos.glm_hmm.expectation_maximization import forward_backward
from nemos.glm_hmm.utils import compute_rate_per_state


class TestAnalyticMStepScale:

    @pytest.mark.parametrize("inv_link_func", [lambda x: x, jnp.exp])
    @pytest.mark.parametrize(
        "glm_params", [GLMParams(np.random.randn(2, 3), np.random.randn(3))]
    )
    @pytest.mark.requires_x64
    def test_gaussian_obs_mstep(self, inv_link_func, glm_params):
        np.random.seed(111)
        X = np.random.randn(20, 2)
        n_states = glm_params.intercept.shape[-1]
        states = np.random.choice(range(n_states), replace=True, size=X.shape[0])
        rate = np.sum(
            X.T * glm_params.coef[:, states] + glm_params.intercept[states], axis=0
        )
        y = rate + np.random.randn(X.shape[0]) * 0.1
        obs = nmo.observation_models.GaussianObservations()

        is_population_glm = y.ndim > 2
        log_likelihood_fn = prepare_ll_estep_likelihood(is_population_glm, obs)
        expected_negative_log_likelihood_scale = (
            prepare_objective_mstep_numerical_scale(
                is_population_glm=is_population_glm,
                observation_model=obs,
            )
        )
        init_proba = jnp.ones(n_states) / n_states
        transition_probs = (
            jnp.eye(n_states) * 0.94
            + (jnp.ones((n_states, n_states)) - jnp.eye(n_states)) * 0.03
        )

        (
            log_gammas_nemos,
            log_xis_nemos,
            ll_nemos,
            ll_norm_nemos,
            log_alphas_nemos,
            log_betas_nemos,
        ) = forward_backward(
            X,
            y,
            jnp.log(init_proba),
            jnp.log(transition_probs),
            glm_params,
            glm_scale=jnp.ones(n_states).astype(float),
            log_likelihood_func=log_likelihood_fn,
            inverse_link_function=obs.default_inverse_link_function,
            is_new_session=None,
        )

        solver = LBFGS(10**-6, 10**-6)

        def objective(scale, args):
            return expected_negative_log_likelihood_scale(scale, *args)

        rate_per_state = compute_rate_per_state(
            X, glm_params, inverse_link_function=inv_link_func
        )
        solution = minimise(
            objective,
            solver,
            y0=jnp.ones_like(glm_params.intercept),
            args=(y, rate_per_state, jnp.exp(log_gammas_nemos)),
        )
        update = get_analytical_scale_update(obs, is_population_glm=is_population_glm)
        update_solution, _ = update(None, y, rate_per_state, jnp.exp(log_gammas_nemos))
        np.testing.assert_allclose(solution.value, update_solution)
