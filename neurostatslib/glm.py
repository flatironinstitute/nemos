import jax
import jax.numpy as jnp
import jaxopt
from .utils import convolve_1d_basis


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
        X = convolve_1d_basis(
            self._spike_basis_matrix,
            spike_data
        )

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
        X = convolve_1d_basis(
            self._spike_basis_matrix,
            spike_data
        )
        log_pred = jnp.einsum("nbt,nbj->nt", X, Ws) + bs[:, None]
        return jnp.exp(log_pred) # TODO: different link functions
    
    def simulate(self, random_key, num_timesteps, init_spikes, covariates=None):
        """
        Simulate GLM as a recurrent network.

        Parameters
        ----------
        num_timesteps : int
            Number of time steps to simulate.

        spike_data : array (num_neurons x window_size)
            Spike counts arranged in a matrix.
        """

        Ws = self._spike_basis_coeff
        bs = self._baseline_log_fr
        B = self._spike_basis_matrix

        subkeys = jax.random.split(random_key, num=num_timesteps)

        def scan_fn(spikes, key):
            X = convolve_1d_basis(B, spikes)
            # X.shape == (num_neurons x num_basis_funcs x 1)
            log_pred = jnp.einsum("nb,nbj->n", jnp.squeeze(X), Ws)
            new_spikes = jax.random.poisson(key, jnp.exp(log_pred))
            concat_spikes = jnp.column_stack(
                (spikes[:, 1:], new_spikes)
            )
            return concat_spikes, new_spikes

        _, simulated_spikes = jax.lax.scan(
            scan_fn, init_spikes, subkeys
        )

        return simulated_spikes.T

