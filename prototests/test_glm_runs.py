import jax
from neurostatslib.glm import GLM
from neurostatslib.basis import MSplineBasis

nn, nt = 10, 1000
key = jax.random.PRNGKey(123)
key, subkey = jax.random.split(key)
spike_data = jax.random.bernoulli(
    subkey, jax.numpy.ones((nn, nt))*.5
).astype("int64")

spike_basis = MSplineBasis(n_basis_funcs=6, window_size=100, order=3)
spike_basis_matrix = spike_basis.gen_basis_funcs(np.arange(100))
model = GLM(spike_basis_matrix)

model.fit(spike_data)
model.predict(spike_data)
key, subkey = jax.random.split(key)
X = model.simulate(subkey, 20, spike_data[:, :100])
