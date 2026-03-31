import jax
import jax.numpy as jnp
import numpy as np
import pynapple as nap

from nemos.pp_glm import log_likelihood, utils
from nemos.pp_glm.validation import to_pp_glm_params, to_pp_glm_params_with_key
from nemos.basis import RaisedCosineLogEval

n_neurons = 5
recording_time = nap.IntervalSet(0, 10)
M_samples = 100
inverse_link_function = jnp.exp
n_basis_funcs = 4
history_window = 0.01
scan_size = 1

M_grid = utils.build_mc_sampling_grid(recording_time, M_samples)
basis = RaisedCosineLogEval(n_basis_funcs, bounds=(0,history_window), fill_value=0)
eval_function = lambda pts: basis.evaluate(pts)

np.random.seed(0)
spike_times = np.sort(np.random.uniform(0, 10, 10000))
spike_ids = np.random.choice(np.arange(n_neurons), 10000)

X = jnp.vstack((spike_times, spike_ids))
y = jnp.vstack((X, jnp.arange(spike_times.size)))
max_window = int(utils.compute_max_window_size(jnp.array([-history_window, 0]), X[0], X[0]))
X, y = utils.adjust_indices_and_spike_times(X, history_window, max_window, y)

random_key =  jax.random.PRNGKey(0).astype(jnp.float64)
params = (
    jnp.ones((n_neurons*n_basis_funcs, n_neurons)),
    jnp.zeros(n_neurons),
)
params = to_pp_glm_params(params)
params_with_key = to_pp_glm_params_with_key(params, random_key)

loss = log_likelihood._compute_loss(
    params_with_key,
    X,
    y,
    inverse_link_function=inverse_link_function,
    M_samples=M_samples,
    M_grid=M_grid,
    recording_time=recording_time,
    n_basis_funcs=n_basis_funcs,
    scan_size=scan_size,
    max_window=max_window,
    eval_function=eval_function,
)

print(loss)