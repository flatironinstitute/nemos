import matplotlib

matplotlib.use('agg')

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp

import neurostatslib as nsl
from neurostatslib.basis import RaisedCosineBasisLog
from neurostatslib.glm import GLM


def test_glm_fit():
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)

    nn, nt, ws = 1, 1000, 100
    simulation_key = jax.random.PRNGKey(123)

    spike_basis = RaisedCosineBasisLog(
        n_basis_funcs=5
    )
    B = spike_basis.evaluate(onp.linspace(0, 1, ws)).T

    simulated_model = GLM(B)
    simulated_model.spike_basis_coeff_ = jnp.array([0, 0, -1, -1, -1])[None, :, None]
    simulated_model.baseline_log_fr_ = jnp.ones(nn) * .1

    init_spikes = jnp.zeros((nn, ws))
    spike_data = simulated_model.simulate(simulation_key, nt, init_spikes)
    sim_pred = simulated_model.predict(spike_data)

    fitted_model = GLM(
        B,
        solver_name="GradientDescent",
        solver_kwargs=dict(maxiter=1000, acceleration=False, verbose=True, stepsize=0.0)

    )
    
    fitted_model.fit(spike_data)
    fit_pred = fitted_model.predict(spike_data)

    fig, ax = plt.subplots(1, 1)
    ax.plot(onp.arange(nt), spike_data[0])
    ax.plot(onp.arange(ws, nt + 1), sim_pred[0])
    ax.plot(onp.arange(ws, nt + 1), fit_pred[0])
    
    
    fig, ax = plt.subplots(1, 1, sharey=True)
    ax.plot(
        B.T @ simulated_model.spike_basis_coeff_[0, :, 0],
        label="true"
    )
    ax.plot(
        B.T @ fitted_model.spike_basis_coeff_[0, :, 0],
        label="est"
    )
    ax.axhline(0, dashes=[2, 2], color='k')
    ax.legend()
    
    plt.close('all')