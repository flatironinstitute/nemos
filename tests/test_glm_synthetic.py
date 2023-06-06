import jax
import jax.numpy as jnp
import numpy as onp
from neurostatslib.glm import GLM
from neurostatslib.basis import RaisedCosineBasis
import matplotlib.pyplot as plt
import itertools
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

nn, nt, ws = 2, 5000, 100
simulation_key = jax.random.PRNGKey(123)

spike_basis = RaisedCosineBasis(
    n_basis_funcs=5,
    window_size=ws
)
sim_pts = src.neurostatslib.sample_points.raised_cosine_log(5, ws)
B = spike_basis.gen_basis_funcs(sim_pts)

w0 = onp.array([-.1, -.1, -.2, -.2, -1])
w1 = onp.array([0, .1, .5, .1, 0])

W = onp.empty((2, 5, 2))
for i, j in itertools.product(range(nn), range(nn)):
    W[i, :, j] = w0 if (i == j) else w1

simulated_model = GLM(B)
simulated_model.spike_basis_coeff_ = jnp.array(W)
simulated_model.baseline_log_fr_ = jnp.ones(nn) * .1

init_spikes = jnp.zeros((2, spike_basis.window_size))
spike_data = simulated_model.simulate(simulation_key, nt, init_spikes)
sim_pred = simulated_model.predict(spike_data)

fitted_model = GLM(
    B,
    # solver_name="ScipyMinimize",
    # solver_kwargs=dict(method="newton-cg", maxiter=1000, options=dict(verbose=True)),
    solver_name="GradientDescent",
    solver_kwargs=dict(maxiter=10000, acceleration=False, verbose=True, stepsize=-1)
    # solver_name="LBFGS",
    # solver_kwargs=dict(maxiter=100, verbose=True, stepsize=-1)
)
# fitted_model.fit(spike_data, init_params=(
#     jnp.copy(simulated_model.spike_basis_coeff_),
#     jnp.copy(simulated_model.baseline_log_fr_)
# ))
fitted_model.fit(spike_data)
fit_pred = fitted_model.predict(spike_data)

fig, axes = plt.subplots(2, 1)
axes[0].plot(onp.arange(nt), spike_data[0])
axes[0].plot(onp.arange(ws, nt + 1), sim_pred[0])
axes[0].plot(onp.arange(ws, nt + 1), fit_pred[0])
axes[1].plot(onp.arange(nt), spike_data[1])
axes[1].plot(onp.arange(ws, nt + 1), sim_pred[1])
axes[1].plot(onp.arange(ws, nt + 1), fit_pred[1])
plt.show()

fig, axes = plt.subplots(nn, nn, sharey=True)
for i, j in itertools.product(range(nn), range(nn)):
    axes[i, j].plot(
        B.T @ simulated_model.spike_basis_coeff_[i, :, j],
        label="true"
    )
    axes[i, j].plot(
        B.T @ fitted_model.spike_basis_coeff_[i, :, j],
        label="est"
    )
    axes[i, j].axhline(0, dashes=[2, 2], color='k')
axes[-1, -1].legend()
plt.show()
