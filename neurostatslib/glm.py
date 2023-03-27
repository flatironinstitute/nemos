import jax
import jax.numpy as jnp
import jaxopt
import inspect
from .utils import convolve_1d_basis
from typing import Optional, Callable, Tuple
from numpy.typing import NDArray
from sklearn.exceptions import NotFittedError


class GLM:
    """Generalized Linear Model for neural responses.

    No stimulus / external variables, only connections to other neurons.

    Parameters
    ----------
    spike_basis_matrix : (n_basis_funcs, window_size)
        Matrix of basis functions to use for this GLM. Most likely the output
        of ``Basis.gen_basis_funcs()``
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
            spike_basis_matrix: NDArray,
            solver_name: str = "GradientDescent",
            solver_kwargs: dict = dict(),
            inverse_link_function: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.softplus,
        ):
        # (n_basis_funcs, window_size)
        self.spike_basis_matrix = spike_basis_matrix
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
            Initial values for the spike basis coefficients and bias terms. If
            None, we initialize with zeros.

        Raises
        ------
        ValueError
            If spike_data is not two-dimensional.
        ValueError
            If shapes of init_params are not correct.
        ValueError
            If solver returns at least one NaN parameter, which means it found
            an invalid solution. Try tuning optimization hyperparameters.

        """
        if spike_data.ndim != 2:
            raise ValueError("spike_data must be two-dimensional, with shape (n_neurons, n_timebins)")

        n_neurons, _ = spike_data.shape
        n_basis_funcs, window_size = self.spike_basis_matrix.shape

        # Convolve spikes with basis functions. We drop the last sample, as
        # those are the features that could be used to predict spikes in the
        # next time bin
        X = convolve_1d_basis(self.spike_basis_matrix,
                              spike_data)[:, :, :-1]

        # Initialize parameters
        if init_params is None:
            # Ws, spike basis coeffs
            init_params = (jnp.zeros((n_neurons, n_basis_funcs, n_neurons)),
                           # bs, bias terms
                           jnp.zeros(n_neurons))

        if init_params[0].ndim != 3:
            raise ValueError("spike basis coefficients must be of shape (n_neurons, n_basis_funcs, n_neurons), but"
                             f" init_params[0] has {init_params[0].ndim} dimensions!")
        if init_params[0].shape[0] != init_params[0].shape[-1]:
            raise ValueError("spike basis coefficients must be of shape (n_neurons, n_basis_funcs, n_neurons), but"
                             f" init_params[0] has shape {init_params[0].shape}!")
        if init_params[1].ndim != 1:
            raise ValueError("bias terms must be of shape (n_neurons,) but init_params[0] have"
                             f"{init_params[1].ndim} dimensions!")
        if init_params[0].shape[0] != init_params[1].shape[0]:
            raise ValueError("spike basis coefficients must be of shape (n_neurons, n_basis_funcs, n_neurons), and"
                             "bias terms must be of shape (n_neurons,) but n_neurons doesn't look the same in both!"
                             f"init_params[0]: {init_params[0].shape[0]}, init_params[1]: {init_params[1].shape[0]}")
        if init_params[0].shape[0] != spike_data.shape[0]:
            raise ValueError("spike basis coefficients must be of shape (n_neurons, n_basis_funcs, n_neurons), and"
                             "spike_data must be of shape (n_neurons, n_timebins) but n_neurons doesn't look the same in both!"
                             f"init_params[0]: {init_params[0].shape[0]}, spike_data: {spike_data.shape[0]}")

        def loss(params, X, y):
            predicted_spikes = self._predict(params, X)
            return self._score(predicted_spikes, y)

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

    def _predict(self, params: Tuple[jnp.ndarray, jnp.ndarray],
                 convolved_spike_data: NDArray) -> jnp.ndarray:
        """Helper function for generating predictions.

        This way, can use same functions during and after fitting.

        Note that the ``n_timebins`` here is not necessarily the same as in
        public functions: in particular, this method expects the *convolved*
        spike data, which (since we use the "valid" convolutional output) means
        that it will have fewer timebins than the un-convolved data.

        Parameters
        ----------
        params : ((n_neurons, n_basis_funcs, n_neurons), (n_neurons,))
            Values for the spike basis coefficients and bias terms.
        convolved_spike_data : (n_basis_funcs, n_neurons, n_timebins)
            Spike counts convolved with some set of bases functions.

        Returns
        -------
        predicted_spikes : (n_neurons, n_timebins)
            The predicted spikes.

        """
        Ws, bs = params
        return self.inverse_link_function(
            jnp.einsum("nbt,nbj->nt", convolved_spike_data, Ws) + bs[:, None]
        )

    def _score(self, predicted_spikes: NDArray,
               target_spikes: NDArray) -> jnp.ndarray:
        """Score the predicted against target spikes.

        This computes the Poisson negative log-likehood.

        Parameters
        ----------
        predicted_spikes : (n_neurons, n_timebins)
            The predicted spikes.
        target_spikes : (n_neurons, n_timebins)
            The target spikes to compare against

        Returns
        -------
        score : (1,)
            The Poisson negative log-likehood

        """
        return jnp.mean(predicted_spikes - target_spikes * jnp.log(predicted_spikes))

    def predict(self, spike_data: NDArray) -> jnp.ndarray:
        """Predict spikes based on fit parameters, for checking against existing data.

        Parameters
        ----------
        spike_data : (n_neurons, n_timebins)
            Spike counts arranged in a matrix. n_neurons must be the same as
            during the fitting of this GLM instance.

        Returns
        -------
        predicted_spikes : (n_neurons, n_timebins - window_size + 1)
            The predicted spikes.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called first with this instance.
        ValueError
            If attempting to simulate a different number of neurons than were
            present during fitting (i.e., if ``init_spikes.shape[0] !=
            self.baseline_log_fr_.shape[0]``).

        See Also
        --------
        score
            Score predicted spikes against target.
        simulate
            Simulate GLM as a recurrent network, for extrapolating into the future.


        """
        try:
            Ws = self.spike_basis_coeff_
        except AttributeError:
            raise NotFittedError("This GLM instance is not fitted yet. Call 'fit' with appropriate arguments.")
        bs = self.baseline_log_fr_
        if spike_data.shape[0] != bs.shape[0]:
            raise ValueError("Number of neurons must be the same during prediction and fitting! "
                             f"spike_data n_neurons: {spike_data.shape[0]}, "
                             f"self.baseline_log_fr_ n_neurons: {self.baseline_log_fr_.shape[0]}")
        X = convolve_1d_basis(self.spike_basis_matrix,
                              spike_data)
        return self._predict((Ws, bs), X)

    def score(self, spike_data: NDArray) -> jnp.ndarray:
        """Score the predicted spikes (based on fit) to the target.

        This ignores the last time point of the prediction.

        This computes the Poisson negative log-likehood.

        Parameters
        ----------
        spike_data : (n_neurons, n_timebins)
            Spike counts arranged in a matrix. n_neurons must be the same as
            during the fitting of this GLM instance.

        Returns
        -------
        score : (1,)
            The Poisson negative log-likehood

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called first with this instance.
        ValueError
            If attempting to simulate a different number of neurons than were
            present during fitting (i.e., if ``init_spikes.shape[0] !=
            self.baseline_log_fr_.shape[0]``).

        """
        # ignore the last time point from predict, because that corresponds to
        # the next time step, which we have no observed data for
        predicted_spikes = self.predict(spike_data)[:, :-1]
        window_size = self.spike_basis_matrix.shape[1]
        return self._score(predicted_spikes, spike_data[:, window_size:])

    def simulate(self, random_key: jax.random.PRNGKeyArray,
                 n_timesteps: int, init_spikes: NDArray) -> jnp.ndarray:
        """Simulate GLM as a recurrent network, for extrapolating into the future.

        Parameters
        ----------
        random_key
            jax PRNGKey to seed simulation with.
        n_timesteps
            Number of time steps to simulate.
        init_spikes : (n_neurons, window_size)
            Spike counts arranged in a matrix. These are used to jump start the
            forward simulation. ``n_neurons`` must be the same as during the
            fitting of this GLM instance and ``window_size`` must be the same
            as the bases functions (i.e., ``self.spike_basis_matrix.shape[1]``)

        Returns
        -------
        simulated_spikes : (n_neurons, n_timesteps)
            The simulated spikes.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called first with this instance.
        ValueError
            If attempting to simulate a different number of neurons than were
            present during fitting (i.e., if ``init_spikes.shape[0] !=
            self.baseline_log_fr_.shape[0]``) or if ``init_spikes`` has the
            wrong number of time steps (i.e., if ``init_spikes.shape[1] !=
            self.spike_basis_matrix.shape[1]``)

        See Also
        --------
        predict
            Predict spikes based on fit parameters, for checking against existing data.

        """
        try:
            Ws = self.spike_basis_coeff_
        except AttributeError:
            raise NotFittedError("This GLM instance is not fitted yet. Call 'fit' with appropriate arguments.")
        bs = self.baseline_log_fr_

        if init_spikes.shape[0] != bs.shape[0]:
            raise ValueError("Number of neurons must be the same during simulation and fitting! "
                             f"init_spikes n_neurons: {init_spikes.shape[0]}, "
                             f"self.baseline_log_fr_ n_neurons: {self.baseline_log_fr_.shape[0]}")
        if init_spikes.shape[1] != self.spike_basis_matrix.shape[1]:
            raise ValueError("init_spikes has the wrong number of time steps!"
                             f"init_spikes time steps: {init_spikes.shape[1]}, "
                             f"spike_basis_matrix window size: {self.spike_basis_matrix.shape[1]}")

        subkeys = jax.random.split(random_key, num=n_timesteps)

        def scan_fn(spikes, key):
            # (n_neurons, n_basis_funcs, 1)
            X = convolve_1d_basis(
                self.spike_basis_matrix,
                spikes
            )
            fr = self._predict((Ws, bs), X).squeeze(-1)
            new_spikes = jax.random.poisson(key, fr)
            concat_spikes = jnp.column_stack(
                (spikes[:, 1:], new_spikes)
            )
            return concat_spikes, new_spikes

        _, simulated_spikes = jax.lax.scan(
            scan_fn, init_spikes, subkeys
        )

        return simulated_spikes.T
