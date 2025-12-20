from typing import Tuple

import jax.numpy as jnp
import numpy as np
import pytest

import nemos as nmo
import nemos.regularizer
from nemos.glm.params import GLMParams
from nemos.glm_hmm.algorithm_configs import (
    get_analytical_scale_update,
    prepare_ll_estep_likelihood,
    prepare_nll_mstep_numerical_params,
    prepare_objective_mstep_numerical_scale,
)
from nemos.glm_hmm.expectation_maximization import forward_backward, run_m_step
from nemos.glm_hmm.utils import compute_rate_per_state
from nemos.solvers import JaxoptLBFGS


@pytest.fixture
def generate_data_gaussian(request):
    """
    Fixture to generate Gaussian GLM-HMM data.

    Uses indirect parametrization to receive (glm_params, y_shape, inv_link_func) and generates
    data once per parameter combination for all tests in the class.
    """
    glm_params, y_shape, inv_link_func = request.param

    np.random.seed(111)
    X = np.random.randn(20, 2)
    n_states = glm_params.intercept.shape[-1]
    states = np.random.choice(range(n_states), replace=True, size=X.shape[0])

    is_population_glm = len(y_shape) > 0
    if is_population_glm:
        rate = (
            jnp.einsum("ij, jni->in", X, glm_params.coef[..., states])
            + glm_params.intercept[..., states].T
        )
    else:
        rate = (
            jnp.einsum("ij, ji->i", X, glm_params.coef[..., states])
            + glm_params.intercept[..., states].T
        )

    n_neurons = max(sum(y_shape), 1)
    std = np.random.randn(n_states, n_neurons)
    std = std[states]
    if not is_population_glm:
        std = jnp.squeeze(std)
    y = rate + np.random.randn(X.shape[0], *y_shape) * std
    obs = nmo.observation_models.GaussianObservations()

    log_likelihood_fn = prepare_ll_estep_likelihood(is_population_glm, obs)
    expected_negative_log_likelihood_scale = prepare_objective_mstep_numerical_scale(
        is_population_glm=is_population_glm,
        observation_model=obs,
    )
    init_proba = jnp.ones(n_states) / n_states
    transition_probs = (
        jnp.eye(n_states) * 0.94
        + (jnp.ones((n_states, n_states)) - jnp.eye(n_states)) * 0.03
    )

    (
        log_gammas,
        log_xis,
        _,
        _,
        _,
        _,
    ) = forward_backward(
        X,
        y,
        jnp.log(init_proba),
        jnp.log(transition_probs),
        glm_params,
        glm_scale=jnp.ones((*y_shape, n_states)).astype(float),
        log_likelihood_func=log_likelihood_fn,
        inverse_link_function=inv_link_func,
        is_new_session=None,
    )
    return (
        X,
        y,
        obs,
        is_population_glm,
        expected_negative_log_likelihood_scale,
        log_gammas,
        log_xis,
        glm_params,
        inv_link_func,
    )


@pytest.mark.parametrize(
    "generate_data_gaussian",
    [
        (GLMParams(np.random.randn(2, 3), np.random.randn(3)), (), lambda x: x),
        (GLMParams(np.random.randn(2, 3), np.random.randn(3)), (), jnp.exp),
        (GLMParams(np.random.randn(2, 4, 3), np.random.randn(4, 3)), (4,), lambda x: x),
        (GLMParams(np.random.randn(2, 4, 3), np.random.randn(4, 3)), (4,), jnp.exp),
    ],
    indirect=True,
)
@pytest.mark.requires_x64
class TestAnalyticMStepScale:

    def test_gaussian_obs_mstep(self, generate_data_gaussian):
        (
            X,
            y,
            obs,
            is_population_glm,
            expected_negative_log_likelihood_scale,
            log_gammas_nemos,
            _,
            glm_params,
            inv_link_func,
        ) = generate_data_gaussian

        def objective(scale, args):
            return expected_negative_log_likelihood_scale(scale, *args)

        solver = JaxoptLBFGS(
            objective,
            regularizer=nemos.regularizer.UnRegularized(),
            regularizer_strength=None,
            tol=10**-14,
        )
        rate_per_state = compute_rate_per_state(
            X, glm_params, inverse_link_function=inv_link_func
        )
        numerical_update, _ = solver.run(
            jnp.ones_like(glm_params.intercept),
            (y, rate_per_state, jnp.exp(log_gammas_nemos)),
        )
        update = get_analytical_scale_update(obs, is_population_glm=is_population_glm)
        analytical_update, _ = update(
            None, y, rate_per_state, jnp.exp(log_gammas_nemos)
        )
        np.testing.assert_allclose(numerical_update, analytical_update, atol=1e-7)

    def test_gaussian_obs_mstep_via_run_m_step(self, generate_data_gaussian):
        np.random.seed(111)
        (
            X,
            y,
            obs,
            is_population_glm,
            expected_negative_log_likelihood_scale,
            log_gammas,
            log_xis,
            glm_params,
            inv_link_func,
        ) = generate_data_gaussian
        update = get_analytical_scale_update(obs, is_population_glm=is_population_glm)
        nll_fcn = prepare_nll_mstep_numerical_params(
            is_population_glm=is_population_glm,
            observation_model=obs,
            inverse_link_function=inv_link_func,
        )

        solver = JaxoptLBFGS(
            nll_fcn,
            regularizer=nemos.regularizer.UnRegularized(),
            regularizer_strength=None,
            tol=10**-4,
        )

        solver_scale = JaxoptLBFGS(
            expected_negative_log_likelihood_scale,
            regularizer=nemos.regularizer.UnRegularized(),
            regularizer_strength=None,
            tol=10**-12,
        )
        new_sess = np.zeros(y.shape[0], dtype=bool)
        new_sess[[0, 15]] = True
        analytical_update = run_m_step(
            X,
            y,
            log_gammas,
            log_xis,
            glm_params,
            is_new_session=new_sess,
            m_step_fn_glm_params=solver.run,
            glm_scale=jnp.ones_like(glm_params.intercept),
            m_step_fn_glm_scale=update,
            inverse_link_function=inv_link_func,
            dirichlet_prior_alphas_transition=None,
            dirichlet_prior_alphas_init_prob=None,
        )[1]
        numerical_update = run_m_step(
            X,
            y,
            log_gammas,
            log_xis,
            glm_params,
            is_new_session=new_sess,
            m_step_fn_glm_params=solver.run,
            glm_scale=jnp.ones_like(glm_params.intercept),
            m_step_fn_glm_scale=solver_scale.run,
            inverse_link_function=inv_link_func,
            dirichlet_prior_alphas_transition=None,
            dirichlet_prior_alphas_init_prob=None,
        )[1]
        np.testing.assert_allclose(numerical_update, analytical_update, atol=1e-7)
