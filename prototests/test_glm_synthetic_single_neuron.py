import jax
import jax.numpy as jnp
import numpy as onp
from neurostatslib.glm import GLM
from neurostatslib.basis import RaisedCosineBasis
from neurostatslib.utils import convolve_1d_basis
import matplotlib.pyplot as plt
import itertools
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

nn, nt, ws = 1, 5000, 100
simulation_key = jax.random.PRNGKey(123)

spike_basis = RaisedCosineBasis(
    num_basis_funcs=5,
    window_size=ws
)
B = spike_basis.transform()

simulated_model = GLM(
    spike_basis=spike_basis,
    covariate_basis=None
)
simulated_model.spike_basis_coeff_ = jnp.array([0, 0, -1, -1, -1])[None, :, None]
simulated_model.baseline_log_fr_ = jnp.ones(nn) * .1

init_spikes = jnp.zeros((nn, spike_basis.window_size))
spike_data = simulated_model.simulate(simulation_key, nt, init_spikes)
sim_pred = simulated_model.predict(spike_data)

fitted_model = GLM(
    spike_basis=spike_basis,
    covariate_basis=None,
    # solver_name="ScipyMinimize",
    # solver_kwargs=dict(method="newton-cg", maxiter=1000, options=dict(verbose=True)),
    solver_name="GradientDescent",
    solver_kwargs=dict(maxiter=10000, acceleration=False, verbose=True, stepsize=0.0)
    # solver_name="LBFGS",
    # solver_kwargs=dict(maxiter=100, verbose=True, stepsize=0.0)
)
# fitted_model.fit(spike_data, init_params=(
#     jnp.copy(simulated_model.spike_basis_coeff_),
#     jnp.copy(simulated_model.baseline_log_fr_)
# ))
fitted_model.fit(spike_data)
fit_pred = fitted_model.predict(spike_data)

fig, ax = plt.subplots(1, 1)
ax.plot(onp.arange(nt), spike_data[0])
ax.plot(onp.arange(ws, nt + 1), sim_pred[0])
ax.plot(onp.arange(ws, nt + 1), fit_pred[0])
plt.show()

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
plt.show()
