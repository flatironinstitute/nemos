import jax
import jax.numpy as jnp
import jaxopt
import inspect
from .utils import convolve_1d_basis
from .basis import Basis
from typing import Optional, Callable, Tuple
from numpy.typing import NDArray


class GLM:
    """Generalized Linear Model for neural responses.

    No stimulus / external variables, only connections to other neurons.

    Parameters
    ----------
    spike_basis
        Instantiated Basis object which gives the possible basis functions for
        these neurons
    solver_name
        Name of the solver to use when fitting the GLM. Must be an attribute of
        ``jaxopt``.
    solver_kwargs
        Dictionary of keyword arguments to pass to the solver during its
        initialization.
    inverse_link_function
        Function to transform outputs of convolution with basis to firing rate.
        Must accept any number as input and return all non-negative values.

    Attributes
    ----------
    solver
        jaxopt solver, set during ``fit()``
    solver_state
        state of the solver, set during ``fit()``
    spike_basis_coeff_ : jnp.ndarray, (n_neurons, n_basis_funcs, n_neurons)
        Solutions for the spike basis coefficients, set during ``fit()``
    baseline_log_fr : jnp.ndarray, (n_neurons,)
        Solutions for bias terms, set during ``fit()``

    """

    def __init__(
            self,
            spike_basis: Basis,
            solver_name: str = "GradientDescent",
            solver_kwargs: dict = dict(),
            inverse_link_function: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.softplus,
        ):
        self.spike_basis = spike_basis
        self.solver_name = solver_name
        try:
            solver_args = inspect.getfullargspec(getattr(jaxopt, solver_name)).args
        except AttributeError:
            raise AttributeError(f'module jaxopt has no attribute {solver_name}, pick a different solver!')
        for k in solver_kwargs.keys():
            if k not in solver_args:
                raise NameError(f"kwarg {k} in solver_kwargs is not a kwarg for jaxopt.{solver_name}!")
        self.solver_kwargs = solver_kwargs
        self.inverse_link_function = inverse_link_function

        # (n_basis_funcs, window_size)
        self._spike_basis_matrix = self.spike_basis.transform()

    def fit(self, spike_data: NDArray,
            init_params: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None):
        """Fit GLM to spiking data.

        Following scikit-learn API, the solutions are stored as attributes
        ``spike_basis_coeff_`` and ``baseline_log_fr``.

        Parameters
        ----------
        spike_data : (n_neurons, n_timebins)
            Spike counts arranged in a matrix.
        init_params : ((n_neurons, n_basis_funcs, n_neurons), (n_neurons,))
            Initial values for the spike basis coefficients and bias terms.

        Raises
        ------
        ValueError
            If solver returns at least one NaN parameter, which means it found
            an invalid solution. Try tuning optimization hyperparameters.

        """
        assert spike_data.ndim == 2

        # Number of neurons and timebins
        n_neurons, _ = spike_data.shape
        n_basis_funcs, window_size = self._spike_basis_matrix.shape
        
        # Convolve spikes with basis functions. We drop the last sample, as
        # those are the features that could be used to predict spikes in the
        # next time bin
        X = convolve_1d_basis(self._spike_basis_matrix,
                              spike_data)[:, :, :-1]

        # Initialize parameters
        if init_params is None:
            init_params = (jnp.zeros((n_neurons, n_basis_funcs, n_neurons)),  # Ws, spike basis coeffs
                           jnp.zeros(n_neurons))              # bs, bias terms

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
        params, state = solver.run(init_params, X=X,
                                   y=spike_data[:, window_size:])

        if jnp.isnan(params[0]).any() or jnp.isnan(params[1]).any():
            raise ValueError("Solver returned at least one NaN parameter, so solution is invalid!"
                             " Try tuning optimization hyperparameters.")
        # Store parameters
        self.spike_basis_coeff_ = params[0]
        self.baseline_log_fr_ = params[1]
        self.solver_state = state
        self.solver = solver

    def predict(self, spike_data):
        """
        Parameters
        ----------
        spike_data : array (n_neurons x n_timebins)
            Spike counts arranged in a matrix.
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

    def score(self, spike_data):
        pred_fr = self.predict(spike_data)[:, :-1]
        ws = self.spike_basis.window_size
        return jnp.mean(pred_fr - spike_data[:, ws:] * jnp.log(pred_fr))

    def simulate(self, random_key, n_timesteps, init_spikes):
        """
        Simulate GLM as a recurrent network.

        Parameters
        ----------
        random_key: PRNGKey

        n_timesteps : int
            Number of time steps to simulate.

        init_spikes : array (n_neurons x window_size)
            Spike counts arranged in a matrix. These are used to
            jump start the forward simulation.
        """

        Ws = self.spike_basis_coeff_
        bs = self.baseline_log_fr_

        subkeys = jax.random.split(random_key, num=n_timesteps)

        def scan_fn(spikes, key):
            X = convolve_1d_basis(
                self._spike_basis_matrix,
                spikes
            ) # X.shape == (n_neurons x n_basis_funcs x 1)
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

