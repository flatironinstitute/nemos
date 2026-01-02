import jax.numpy as jnp
import numpy as np
import pytest

import nemos as nmo
from nemos.glm.params import GLMParams
from nemos.glm_hmm.algorithm_configs import (
    get_analytical_scale_update,
    prepare_estep_log_likelihood,
    prepare_mstep_nll_objective_param,
    prepare_mstep_nll_objective_scale,
)
from nemos.glm_hmm.expectation_maximization import forward_backward, run_m_step
from nemos.glm_hmm.params import GLMHMMParams, GLMScale, HMMParams
from nemos.glm_hmm.utils import compute_rate_per_state
from nemos.regularizer import UnRegularized
from nemos.solvers import get_solver


def setup_solver(
    objective, init_params, tol=1e-12, reg_strength=0.0, reg=UnRegularized()
):
    lbfgs_class = get_solver("LBFGS").implementation
    solver = lbfgs_class(
        objective,
        init_params=init_params,
        regularizer=reg,
        regularizer_strength=reg_strength,
        has_aux=False,
        tol=tol,
    )
    return solver


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
        rate = inv_link_func(
            jnp.einsum("ij, jni->in", X, glm_params.coef[..., states])
            + glm_params.intercept[..., states].T
        )
    else:
        rate = inv_link_func(
            jnp.einsum("ij, ji->i", X, glm_params.coef[..., states])
            + glm_params.intercept[..., states].T
        )

    n_neurons = max(sum(y_shape), 1)
    std = np.random.randn(n_states, n_neurons) * 0.1
    std = std[states]
    if not is_population_glm:
        std = jnp.squeeze(std)
    y = rate + np.random.randn(X.shape[0], *y_shape) * std
    obs = nmo.observation_models.GaussianObservations()

    log_likelihood_fn = prepare_estep_log_likelihood(is_population_glm, obs)
    expected_negative_log_likelihood_scale = prepare_mstep_nll_objective_scale(
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
        glm_scale=GLMScale(jnp.zeros((*y_shape, n_states)).astype(float)),
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

        init_scale = GLMScale(jnp.zeros_like(glm_params.intercept))
        solver = setup_solver(objective, init_params=init_scale, tol=10**-14)
        rate_per_state = compute_rate_per_state(
            X, glm_params, inverse_link_function=inv_link_func
        )
        numerical_update, _, _ = solver.run(
            init_scale,
            (y, rate_per_state, jnp.exp(log_gammas_nemos)),
        )
        update = get_analytical_scale_update(obs, is_population_glm=is_population_glm)
        analytical_update, _, _ = update(
            None, y, rate_per_state, jnp.exp(log_gammas_nemos)
        )
        np.testing.assert_allclose(
            numerical_update.log_scale, analytical_update.log_scale, atol=1e-7
        )

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
        nll_fcn = prepare_mstep_nll_objective_param(
            is_population_glm=is_population_glm,
            observation_model=obs,
            inverse_link_function=inv_link_func,
        )
        init_scale = GLMScale(jnp.zeros_like(glm_params.intercept))
        solver = setup_solver(nll_fcn, init_params=glm_params, tol=10**-4)

        solver_scale = setup_solver(
            expected_negative_log_likelihood_scale,
            init_params=init_scale,
            tol=10**-12,
        )
        new_sess = np.zeros(y.shape[0], dtype=bool)
        new_sess[[0, 15]] = True

        params = GLMHMMParams(
            hmm_params=HMMParams(None, None),
            glm_params=glm_params,
            glm_scale=GLMScale(jnp.zeros_like(glm_params.intercept)),
        )
        analytical_update, _ = run_m_step(
            params,
            X,
            y,
            log_gammas,
            log_xis,
            is_new_session=new_sess,
            m_step_fn_glm_params=solver.run,
            m_step_fn_glm_scale=update,
            inverse_link_function=inv_link_func,
            dirichlet_prior_alphas_transition=None,
            dirichlet_prior_alphas_init_prob=None,
        )
        numerical_update, _ = run_m_step(
            params,
            X,
            y,
            log_gammas,
            log_xis,
            is_new_session=new_sess,
            m_step_fn_glm_params=solver.run,
            m_step_fn_glm_scale=solver_scale.run,
            inverse_link_function=inv_link_func,
            dirichlet_prior_alphas_transition=None,
            dirichlet_prior_alphas_init_prob=None,
        )
        np.testing.assert_allclose(
            numerical_update.glm_scale.log_scale,
            analytical_update.glm_scale.log_scale,
            atol=1e-7,
        )
