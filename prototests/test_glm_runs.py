import jax
from neurostatslib.glm import GLM
from neurostatslib.basis import MSpline
jax.config.update('jax_platform_name', 'cpu')

nn, nt = 10, 1000
key = jax.random.PRNGKey(123)
key, subkey = jax.random.split(key)
spike_data = jax.random.bernoulli(
    subkey, jax.numpy.ones((nn, nt))*.5
).astype("int32")

model = GLM(
    spike_basis=MSpline(num_basis_funcs=6, window_size=100, order=3),
    covariate_basis=None
)

model.fit(spike_data)
model.predict(spike_data)
key, subkey = jax.random.split(key)
X = model.simulate(subkey, 20, spike_data[:, :100])
