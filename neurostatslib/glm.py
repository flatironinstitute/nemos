import jax
import jax.numpy as jnp
import jaxopt
from .utils import convolve_1d_basis


class GLM:

    def __init__(
            self,
            spike_basis,
            covariate_basis=None,
            solver_name="GradientDescent",
            solver_kwargs=dict(),
            inverse_link_function=jax.nn.softplus
        ):
        self.spike_basis = spike_basis
        self.covariate_basis = covariate_basis
        self.solver_name = solver_name
        self.solver_kwargs = solver_kwargs
        self.inverse_link_function = inverse_link_function

        # (num_basis_funcs x window_size)
        self._spike_basis_matrix = self.spike_basis.transform()

    def fit(self, spike_data, covariates=None, init_params=None):
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
        )[:, :, :-1]

        # Initialize parameters
        if init_params is None:
            init_params = (
                jnp.zeros((nn, nbs, nn)),  # Ws, spike basis coeffs
                jnp.zeros(nn)              # bs, bias terms
            )

        # Poisson negative log-likelihood.
        def loss(params, X, y):
            Ws, bs = params
            pred_fr = self.inverse_link_function(
                jnp.einsum("nbt,nbj->nt", X, Ws) + bs[:, None]
            )
            return jnp.mean(pred_fr - y * jnp.log(pred_fr))

        # Run optimization
        solver = getattr(jaxopt, self.solver_name)(
            fun=loss, **self.solver_kwargs
        )
        params, state = solver.run(
            init_params,
            X=X,
            y=spike_data[:, nws:]
        )

        if jnp.isnan(params[0]).any() or jnp.isnan(params[1]).any():
            raise ValueError("Solver returned at least one NaN parameter, so solution is invalid!"
                             " Try tuning optimization hyperparameters.")
        # Store parameters
        self.spike_basis_coeff_ = params[0]
        self.baseline_log_fr_ = params[1]
        self.solver_state_ = state
        self.solver_ = solver

    def predict(self, spike_data, covariates=None):
        """
        Parameters
        ----------
        spike_data : array (num_neurons x num_timebins)
            Spike counts arranged in a matrix.

        covariates : array (num_covariates x num_timebins)
            Other input variables (e.g. stimulus features)
        """
        Ws = self.spike_basis_coeff_
        bs = self.baseline_log_fr_
        X = convolve_1d_basis(
            self._spike_basis_matrix,
            spike_data
        )
        return self.inverse_link_function(
            jnp.einsum("nbt,nbj->nt", X, Ws) + bs[:, None]
        )

    def score(self, spike_data, covariates=None):
        pred_fr = self.predict(spike_data, covariates=covariates)[:, :-1]
        ws = self.spike_basis.window_size
        return jnp.mean(pred_fr - spike_data[:, ws:] * jnp.log(pred_fr))

    def simulate(self, random_key, num_timesteps, init_spikes, covariates=None):
        """
        Simulate GLM as a recurrent network.

        Parameters
        ----------
        random_key: PRNGKey

        num_timesteps : int
            Number of time steps to simulate.

        init_spikes : array (num_neurons x window_size)
            Spike counts arranged in a matrix. These are used to
            jump start the forward simulation.
        """

        Ws = self.spike_basis_coeff_
        bs = self.baseline_log_fr_

        subkeys = jax.random.split(random_key, num=num_timesteps)

        def scan_fn(spikes, key):
            X = convolve_1d_basis(
                self._spike_basis_matrix,
                spikes
            ) # X.shape == (num_neurons x num_basis_funcs x 1)
            fr = self.inverse_link_function(
                jnp.einsum("nbz,nbj->n", X, Ws) + bs
            )
            new_spikes = jax.random.poisson(key, fr)
            concat_spikes = jnp.column_stack(
                (spikes[:, 1:], new_spikes)
            )
            return concat_spikes, new_spikes

        _, simulated_spikes = jax.lax.scan(
            scan_fn, init_spikes, subkeys
        )

        return simulated_spikes.T

