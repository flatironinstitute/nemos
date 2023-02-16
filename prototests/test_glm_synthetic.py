import jax
from neurostatslib.glm import GLM
from neurostatslib.basis import OrthExponentials
from neurostatslib.utils import convolve_1d_basis
import matplotlib.pyplot as plt

nn, nt = 2, 1000
key = jax.random.PRNGKey(123)
key, subkey = jax.random.split(key)
spike_data = jax.random.bernoulli(
    subkey, jax.numpy.ones((nn, nt))*.5
).astype("int32")

spike_basis = OrthExponentials(
    decay_rates=jax.numpy.logspace(-1, 0, 5),
    window_size=75
)
B = spike_basis.transform()

w = jax.numpy.array([2, -.5, -1, -1, 2])

fig, axes = plt.subplots(2, 1)
axes[0].plot(B.T)
axes[1].plot(B.T @ w)
plt.show()

# X = convolve_1d_basis(
#     basis.transform(), spikes
# )

# model = GLM(
#     spike_basis=spike_basis,
#     covariate_basis=None
# )

# model.fit(spike_data)
# model.predict(spike_data)
# key, subkey = jax.random.split(key)
# X = model.simulate(subkey, 20, spike_data[:, :100])
