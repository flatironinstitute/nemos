"""GLM core module
"""
import inspect
import warnings
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import jaxopt
from numpy.typing import NDArray
from sklearn.exceptions import NotFittedError

from .utils import convolve_1d_trials


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
            raise AttributeError(
                f"module jaxopt has no attribute {solver_name}, pick a different solver!"
            )
        for k in solver_kwargs.keys():
            if k not in solver_args:
                raise NameError(
                    f"kwarg {k} in solver_kwargs is not a kwarg for jaxopt.{solver_name}!"
                )
        self.solver_kwargs = solver_kwargs
        self.inverse_link_function = inverse_link_function

    def fit(
        self,
        spike_data: NDArray,
        init_params: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    ):
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
            raise ValueError(
                "spike_data must be two-dimensional, with shape (n_neurons, n_timebins)"
            )

        n_neurons, _ = spike_data.shape
        n_basis_funcs, window_size = self.spike_basis_matrix.shape

        # Convolve spikes with basis functions. We drop the last sample, as
        # those are the features that could be used to predict spikes in the
        # next time bin
        X = convolve_1d_trials(self.spike_basis_matrix, [spike_data])[0][:, :, :-1]

        # Initialize parameters
        if init_params is None:
            # Ws, spike basis coeffs
            init_params = (
                jnp.zeros((n_neurons, n_basis_funcs, n_neurons)),
                # bs, bias terms
                jnp.zeros(n_neurons),
            )

        if init_params[0].ndim != 3:
            raise ValueError(
                "spike basis coefficients must be of shape (n_neurons, n_basis_funcs, n_neurons), but"
                f" init_params[0] has {init_params[0].ndim} dimensions!"
            )
        if init_params[0].shape[0] != init_params[0].shape[-1]:
            raise ValueError(
                "spike basis coefficients must be of shape (n_neurons, n_basis_funcs, n_neurons), but"
                f" init_params[0] has shape {init_params[0].shape}!"
            )
        if init_params[1].ndim != 1:
            raise ValueError(
                "bias terms must be of shape (n_neurons,) but init_params[0] have"
                f"{init_params[1].ndim} dimensions!"
            )
        if init_params[0].shape[0] != init_params[1].shape[0]:
            raise ValueError(
                "spike basis coefficients must be of shape (n_neurons, n_basis_funcs, n_neurons), and"
                "bias terms must be of shape (n_neurons,) but n_neurons doesn't look the same in both!"
                f"init_params[0]: {init_params[0].shape[0]}, init_params[1]: {init_params[1].shape[0]}"
            )
        if init_params[0].shape[0] != spike_data.shape[0]:
            raise ValueError(
                "spike basis coefficients must be of shape (n_neurons, n_basis_funcs, n_neurons), and"
                "spike_data must be of shape (n_neurons, n_timebins) but n_neurons doesn't look the same in both!"
                f"init_params[0]: {init_params[0].shape[0]}, spike_data: {spike_data.shape[0]}"
            )

        def loss(params, X, y):
            predicted_firing_rates = self._predict(params, X)
            return self._score(predicted_firing_rates, y)

        # Run optimization
        solver = getattr(jaxopt, self.solver_name)(fun=loss, **self.solver_kwargs)
        params, state = solver.run(init_params, X=X, y=spike_data[:, window_size:])

        if jnp.isnan(params[0]).any() or jnp.isnan(params[1]).any():
            raise ValueError(
                "Solver returned at least one NaN parameter, so solution is invalid!"
                " Try tuning optimization hyperparameters."
            )
        # Store parameters
        self.spike_basis_coeff_ = params[0]
        self.baseline_log_fr_ = params[1]
        # note that this will include an error value, which is not the same as
        # the output of loss. I believe it's the output of
        # solver.l2_optimality_error
        self.solver_state = state
        self.solver = solver

    def _predict(
        self, params: Tuple[jnp.ndarray, jnp.ndarray], convolved_spike_data: NDArray
    ) -> jnp.ndarray:
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
        predicted_firing_rates : (n_neurons, n_timebins)
            The predicted firing rates.

        """
        Ws, bs = params
        return self.inverse_link_function(
            jnp.einsum("nbt,nbj->nt", convolved_spike_data, Ws) + bs[:, None]
        )

    def _score(
        self, predicted_firing_rates: NDArray, target_spikes: NDArray
    ) -> jnp.ndarray:
        """Score the predicted firing rates against target spike counts.

                This computes the Poisson negative log-likehood.

                Note that you can end up with infinities in here if there are zeros in
                ``predicted_firing_rates``. We raise a warning in that case.

                Parameters
                ----------
                predicted_firing_rates : (n_neurons, n_timebins)
                    The predicted firing rates.
                target_spikes : (n_neurons, n_timebins)
                    The target spikes to compare against

                Returns
                -------
                score : (1,)
                    The Poisson negative log-likehood

                Notes
                -----
                The Poisson probably mass function is:

                .. math::
                   \frac{\lambda^k \exp(-\lambda)}{k!}

                Thus, the negative log of it is:

                .. math::
        ¨           -\log{\frac{\lambda^k\exp{-\lambda}}{k!}} &= -[\log(\lambda^k)+\log(\exp{-\lambda})-\log(k!)]
                   &= -k\log(\lambda)-\lambda+\log(\Gamma(k+1))

                Because $\Gamma(k+1)=k!$, see
                https://en.wikipedia.org/wiki/Gamma_function.

                And, in our case, ``target_spikes`` is $k$ and
                ``predicted_firing_rates`` is $\lambda$

        """
        x = target_spikes * jnp.log(predicted_firing_rates)
        # this is a jax jit-friendly version of saying "put a 0 wherever
        # there's a NaN". we do this because NaNs result from 0*log(0)
        # (log(0)=-inf and any non-zero multiplied by -inf gives the expected
        # +/- inf)
        x = jnp.where(jnp.isnan(x), jnp.zeros_like(x), x)
        # see above for derivation of this.
        return jnp.mean(
            predicted_firing_rates - x + jax.scipy.special.gammaln(target_spikes + 1)
        )

    def predict(self, spike_data: NDArray) -> jnp.ndarray:
        """Predict firing rates based on fit parameters, for checking against existing data.

        Parameters
        ----------
        spike_data : (n_neurons, n_timebins)
            Spike counts arranged in a matrix. n_neurons must be the same as
            during the fitting of this GLM instance.

        Returns
        -------
        predicted_firing_rates : (n_neurons, n_timebins - window_size + 1)
            The predicted firing rates.

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
            Score predicted firing rates against target spike counts.
        simulate
            Simulate spikes using GLM as a recurrent network, for extrapolating into the future.

        """
        try:
            Ws = self.spike_basis_coeff_
        except AttributeError:
            raise NotFittedError(
                "This GLM instance is not fitted yet. Call 'fit' with appropriate arguments."
            )
        bs = self.baseline_log_fr_
        if spike_data.shape[0] != bs.shape[0]:
            raise ValueError(
                "Number of neurons must be the same during prediction and fitting! "
                f"spike_data n_neurons: {spike_data.shape[0]}, "
                f"self.baseline_log_fr_ n_neurons: {self.baseline_log_fr_.shape[0]}"
            )
        X = convolve_1d_trials(self.spike_basis_matrix, spike_data[None, :, :])[0]
        return self._predict((Ws, bs), X)

    def score(self, spike_data: NDArray) -> jnp.ndarray:
        """Score the predicted firing rates (based on fit) to the target spike counts.

        This ignores the last time point of the prediction.

        This computes the Poisson negative log-likehood, thus the lower the
        number the better, and zero isn't special (you can have a negative
        score if ``spike_data > 0`` and and ``log(predicted_firing_rates) < 0``

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
        UserWarning
            If there are any zeros in ``self.predict(spike_data)``, since this
            will likely lead to infinite log-likelihood values being returned.

        """
        # ignore the last time point from predict, because that corresponds to
        # the next time step, which we have no observed data for
        predicted_firing_rates = self.predict(spike_data)[:, :-1]
        if (predicted_firing_rates == 0).any():
            warnings.warn(
                "predicted_firing_rates array contained zeros, this can "
                "lead to infinite log-likelihood values."
            )
        window_size = self.spike_basis_matrix.shape[1]
        return self._score(predicted_firing_rates, spike_data[:, window_size:])

    def simulate(
        self,
        random_key: jax.random.PRNGKeyArray,
        n_timesteps: int,
        init_spikes: NDArray,
    ) -> jnp.ndarray:
        """Simulate spikes using GLM as a recurrent network, for extrapolating into the future.

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
            Predict firing rates based on fit parameters, for checking against existing data.

        """
        try:
            Ws = self.spike_basis_coeff_
        except AttributeError:
            raise NotFittedError(
                "This GLM instance is not fitted yet. Call 'fit' with appropriate arguments."
            )
        bs = self.baseline_log_fr_

        if init_spikes.shape[0] != bs.shape[0]:
            raise ValueError(
                "Number of neurons must be the same during simulation and fitting! "
                f"init_spikes n_neurons: {init_spikes.shape[0]}, "
                f"self.baseline_log_fr_ n_neurons: {self.baseline_log_fr_.shape[0]}"
            )
        if init_spikes.shape[1] != self.spike_basis_matrix.shape[1]:
            raise ValueError(
                "init_spikes has the wrong number of time steps!"
                f"init_spikes time steps: {init_spikes.shape[1]}, "
                f"spike_basis_matrix window size: {self.spike_basis_matrix.shape[1]}"
            )

        subkeys = jax.random.split(random_key, num=n_timesteps)

        def scan_fn(spikes, key):
            # (n_neurons, n_basis_funcs, 1)
            X = convolve_1d_trials(self.spike_basis_matrix, spikes[None, :, :])[0]
            fr = self._predict((Ws, bs), X).squeeze(-1)
            new_spikes = jax.random.poisson(key, fr)
            concat_spikes = jnp.column_stack((spikes[:, 1:], new_spikes))
            return concat_spikes, new_spikes

        _, simulated_spikes = jax.lax.scan(scan_fn, init_spikes, subkeys)

        return simulated_spikes.T
