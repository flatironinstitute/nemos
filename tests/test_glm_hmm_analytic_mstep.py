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
        "glm_params, y_shape", [(GLMParams(np.random.randn(2, 3), np.random.randn(3)), ()),
                                (GLMParams(np.random.randn(2, 4, 3), np.random.randn(4, 3)), (4,))]
    )
    @pytest.mark.requires_x64
    def test_gaussian_obs_mstep(self, inv_link_func, glm_params, y_shape):
        np.random.seed(111)
        X = np.random.randn(20, 2)
        n_states = glm_params.intercept.shape[-1]
        states = np.random.choice(range(n_states), replace=True, size=X.shape[0])

        is_population_glm = len(y_shape) > 0
        if is_population_glm:
            rate = jnp.einsum("ij, jni->in", X, glm_params.coef[..., states]) + glm_params.intercept[...,states].T
        else:
            rate = jnp.einsum("ij, ji->i", X, glm_params.coef[..., states]) + glm_params.intercept[..., states].T

        n_neurons = max(sum(y_shape), 1)
        std = np.random.randn(n_states, n_neurons)
        std = std[states]
        if not is_population_glm:
            std = jnp.squeeze(std)
        y = rate + np.random.randn(X.shape[0], *y_shape) * std
        obs = nmo.observation_models.GaussianObservations()


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
            glm_scale=jnp.ones((*y_shape, n_states)).astype(float),
            log_likelihood_func=log_likelihood_fn,
            inverse_link_function=obs.default_inverse_link_function,
            is_new_session=None,
        )

        solver = LBFGS(10**-12, 10**-12)

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
        np.testing.assert_allclose(solution.value, update_solution, atol=1e-7)
