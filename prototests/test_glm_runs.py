import jax
from neurostatslib.glm import GLM
from neurostatslib.basis import MSplineBasis, Cyclic_MSplineBasis
import numpy as np
import matplotlib.pylab as plt

nn, nt = 10, 1000
key = jax.random.PRNGKey(123)
key, subkey = jax.random.split(key)
spike_data = jax.random.bernoulli(
    subkey, jax.numpy.ones((nn, nt))*.5
).astype("int64")

# spike_basis = MSplineBasis(n_basis_funcs=6, window_size=100, order=3)
# spike_basis_matrix = spike_basis.gen_basis_funcs(np.arange(100)/100.)
# spike_basis.generate_knots(np.arange(100)/100.,0.,1.)
# spike_basis_matrix_splev = spike_basis.gen_basis_funcs_splev(np.arange(100)/100., outer_ok=False, der=0)
#
#
# plt.close('all')
# plt.figure()
# plt.plot(spike_basis_matrix_splev.T)

c_basis = Cyclic_MSplineBasis(n_basis_funcs=11, window_size=100, order=4)
spike_basis_matrix_splev = c_basis.gen_basis_funcs_splev(np.arange(100)/100., der=0)

print(c_basis.n_basis_funcs, spike_basis_matrix_splev.shape[0])
plt.figure()
plt.plot(spike_basis_matrix_splev.T)
# model = GLM(spike_basis_matrix)
#
# model.fit(spike_data)
# model.predict(spike_data)
# key, subkey = jax.random.split(key)
# X = model.simulate(subkey, 20, spike_data[:, :100])
