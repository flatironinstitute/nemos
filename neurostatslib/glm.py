import jax.numpy as jnp
from sklearn.linear_model import PoissonRegressor
from scipy.ndimage import convolve1d
from .basis import MSpline
import jax
import jaxopt


# Broadcasted 1d convolution operations.
# [[n x t],[w]] -> [n x (t - w + 1)]
_corr1 = jax.vmap(jax.numpy.correlate, (0, None), 0)
# [[n x t],[p x w]] -> [n x p x (t - w + 1)]
_corr2 = jax.vmap(_corr1, (None, 0), 0)


class GLM:

    def __init__(self, spike_basis, covariate_basis=None, solver_name="LBFGS", solver_kwargs=dict()):
        self.spike_basis = spike_basis
        self.covariate_basis = covariate_basis
        self.solver_name = solver_name
        self.solver_kwargs = solver_kwargs

        # (num_basis_funcs x window_size)
        self._spike_basis_matrix = self.spike_basis.transform()

    def fit(self, spike_data, covariates=None):
        """
        Parameters
        ----------
        spike_data : array (num_neurons x num_timebins)
            Spike counts arranged in a matrix.

        covariates : array (num_covariates x num_timebins)
            Other input variables (e.g. stimulus features)
        """

        if covariates is not None:
            raise NotImplementedError()

        assert spike_data.ndim == 2

        # Number of neurons and timebins
        nn, nt = spike_data.shape
        nbs, nws = self._spike_basis_matrix.shape
        
        # Convolve spikes with basis functions.
        #   Input shapes: (basis_funcs, timebins), (neurons, windowsize)
        #   Output shape: (neurons, basis_funcs, timebins - windowsize + 1)
        X = _corr2(self._spike_basis_matrix, spike_data)

        # Initialize parameters
        init_params = (
            jnp.zeros((nn, nbs, nn)),  # Ws, spike basis coeffs
            jnp.zeros(nn)              # bs, bias terms
        )

        # Poisson negative log-likelihood with an exponential link function.
        #    TODO: other link functions.
        def loss(params, X, y):
            Ws, bs = params
            log_pred = jnp.einsum("nbt,nbj->nt", X, Ws) + bs[:, None]
            # log_pred = vmap(lambda A, B: A @ B)(X, Ws) + bs[:, None]
            return jnp.sum(jnp.exp(log_pred) - y * log_pred)

        # Run optimization
        solver = getattr(jaxopt, self.solver_name)(
            fun=loss, **self.solver_kwargs
        )
        params, state = solver.run(
            init_params, X=X, y=spike_data[:, (nws - 1):]
        )

        # Store parameters
        self._spike_basis_coeff = params[0]
        self._baseline_log_fr = params[1]


    def predict(self, spike_data, covariates=None):
        """
        Parameters
        ----------
        spike_data : array (num_neurons x num_timebins)
            Spike counts arranged in a matrix.

        covariates : array (num_covariates x num_timebins)
            Other input variables (e.g. stimulus features)
        """
        Ws = self._spike_basis_coeff
        bs = self._baseline_log_fr
        X = _corr2(self._spike_basis_matrix, spike_data)
        log_pred = jnp.einsum("nbt,nbj->nt", X, Ws) + bs[:, None]
        return jnp.exp(log_pred) # TODO: different link functions


# Short test
if __name__ == "__main__":

    nn, nt = 10, 1000
    spike_data = jnp.ones((nn, nt))

    model = GLM(
        spike_basis=MSpline(num_basis_funcs=6, window_size=100, order=3),
        covariate_basis=None
    )

    model.fit(spike_data)
    model.predict(spike_data)
