"""GLM core module
"""
import inspect
import abc
from typing import Callable, Optional, Tuple, Literal

import jax
import jax.numpy as jnp
import jaxopt
from numpy.typing import NDArray
from sklearn.exceptions import NotFittedError

from .utils import convolve_1d_trials
from .model_base import Model


class GLMBase(Model, abc.ABC):
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
            solver_name: str = "GradientDescent",
            solver_kwargs: dict = dict(),
            inverse_link_function: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.softplus,
            score_type: Literal["log-likelihood", "pseudo-r2"] = "log-likelihood",
            **kwargs
    ):
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

        if score_type not in ['log-likelihood', 'pseudo-r2']:
            raise NotImplementedError("Scoring method not implemented. "
                                      f"score_type must be either 'log-likelihood', or 'pseudo-r2'."
                                      f" {score_type} provided instead.")
        self.score_type = score_type
        self.solver_kwargs = solver_kwargs
        self.inverse_link_function = inverse_link_function
        # set additional kwargs e.g. regularization hyperparameters and so on...
        super().__init__(**kwargs)

    @abc.abstractmethod
    def fit(
            self,
            X: NDArray,
            spike_data: NDArray,
            init_params: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    ) -> None:
        """Fit GLM to spiking data.

        Following scikit-learn API, the solutions are stored as attributes
        ``spike_basis_coeff_`` and ``baseline_log_fr``.

        Parameters
        ----------
        X :
            Predictors, shape (n_time_bins, n_neurons, n_features)
        spike_data :
            Spike counts arranged in a matrix, shape (n_time_bins, n_neurons).
        init_params :
            Initial values for the spike basis coefficients and bias terms. If
            None, we initialize with zeros. shape.  ((n_neurons, n_features), (n_neurons,))

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
        pass

    def _predict(
            self,
            params: Tuple[jnp.ndarray, jnp.ndarray],
            X: NDArray
    ) -> jnp.ndarray:
        """Helper function for generating predictions.

        This way, can use same functions during and after fitting.

        Note that the ``n_timebins`` here is not necessarily the same as in
        public functions: in particular, this method expects the *convolved*
        spike data, which (since we use the "valid" convolutional output) means
        that it will have fewer timebins than the un-convolved data.

        Parameters
        ----------
        params : ((n_neurons, n_features), (n_neurons,))
            Values for the spike basis coefficients and bias terms.
        X : (n_time_bins, n_neurons, n_features)
            The model matrix.

        Returns
        -------
        predicted_firing_rates : (n_time_bins, n_neurons)
            The predicted firing rates.

        """
        Ws, bs = params
        return self.inverse_link_function(
            jnp.einsum("ik,tik->ti", Ws, X) + bs[None, :]
        )

    def _score(
            self,
            X: NDArray,
            target_spikes: NDArray,
            params: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> jnp.ndarray:
        """Score the predicted firing rates against target spike counts.

        This computes the Poisson negative log-likelihood.

        Note that you can end up with infinities in here if there are zeros in
        ``predicted_firing_rates``. We raise a warning in that case.

        Parameters
        ----------
        X : (n_time_bins, n_neurons, n_features)
            The exogenous variables.
        target_spikes : (n_time_bins, n_neurons )
            The target spikes to compare against
        params : ((n_neurons, n_features), (n_neurons,))
            Values for the spike basis coefficients and bias terms.

        Returns
        -------
        score : (1,)
            The Poisson log-likehood

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
        # Avoid the edge-case of 0*log(0), much faster than
        # where on large arrays.
        predicted_firing_rates = jnp.clip(self._predict(params, X), a_min=10 ** -10)
        x = target_spikes * jnp.log(predicted_firing_rates)
        # see above for derivation of this.
        return jnp.mean(
            predicted_firing_rates - x
        )

    def _residual_deviance(self, predicted_rate, y):
        """Compute the residual deviance for a Poisson model.

        Parameters
        ----------
        X:
            The predictors. Shape (n_time_bins, n_neurons, n_features).
        y:
            The spike counts. Shape (n_time_bins, n_neurons).

        Returns
        -------
            The residual deviance of the model.
        """
        # this takes care of 0s in the log
        ratio = jnp.clip(y / predicted_rate, self.FLOAT_EPS, jnp.inf)
        resid_dev = y * jnp.log(ratio) - (y - predicted_rate)
        return resid_dev

    def _pseudo_r2(self, X, y):
        """Pseudo-R2 calculation.

        Parameters
        ----------
        X:
            The predictors. Shape (n_time_bins, n_neurons, n_features).
        y:
            The spike counts. Shape (n_time_bins, n_neurons).

        Returns
        -------
        :
            The pseudo-r2 of the model.
        """
        mu = self.predict(X)
        res_dev_t = self._residual_deviance(mu, y)
        resid_deviance = jnp.sum(res_dev_t ** 2)

        null_mu = jnp.ones(y.shape) * y.sum() / y.size
        null_dev_t = self._residual_deviance(null_mu, y)
        null_deviance = jnp.sum(null_dev_t ** 2)

        return (null_deviance - resid_deviance) / null_deviance


    def check_is_fit(self):
        if not hasattr(self, "spike_basis_coeff_"):
            raise NotFittedError(
                "This GLM instance is not fitted yet. Call 'fit' with appropriate arguments."
            )

    def check_n_neurons(self, spike_data, bs):
        if spike_data.shape[1] != bs.shape[0]:
            raise ValueError(
                "Number of neurons must be the same during prediction and fitting! "
                f"spike_data n_neurons: {spike_data.shape[1]}, "
                f"self.baseline_log_fr_ n_neurons: {self.baseline_log_fr_.shape[0]}"
            )

    def check_n_features(self, spike_data, bs):
        if spike_data.shape[1] != bs.shape[0]:
            raise ValueError(
                "Number of neurons must be the same during prediction and fitting! "
                f"spike_data n_neurons: {spike_data.shape[1]}, "
                f"self.baseline_log_fr_ n_neurons: {self.baseline_log_fr_.shape[0]}"
            )

    def predict(self, X: NDArray) -> jnp.ndarray:
        """Predict firing rates based on fit parameters, for checking against existing data.

        Parameters
        ----------
        X : (n_time_bins, n_neurons, n_features)
            The exogenous variables.

        Returns
        -------
        predicted_firing_rates : (n_neurons, n_time_bins)
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
        self.check_is_fit()
        Ws = self.spike_basis_coeff_
        bs = self.baseline_log_fr_
        self.check_n_neurons(X, bs)
        return self._predict((Ws, bs), X)

    def score(self, X: NDArray, spike_data: NDArray) -> jnp.ndarray:
        r"""Score the predicted firing rates (based on fit) to the target spike counts.

        This ignores the last time point of the prediction.

        This computes the Poisson mean log-likelihood or the pseudo-R2, thus the higher the
        number the better.

        The formula for the mean log-likelihood is the following,

        $$
        \text{LL}(\hat{\lambda} | y) = \frac{1}{T \cdot N} \sum_{n=1}^{N} \sum_{t=1}^{T}
        [y_{tn} \log(\hat{\lambda}_{tn}) - \hat{\lambda}\_{tn} - \log({y\_{tn}!})]
        $$

        The pseudo-R2 can be computed as follows,

        $$
            \frac{\log \text{LL}(\hat{\lambda}| y) - \log \text{LL}(\bar{\lambda}| y)}{\log \text{LL}(y| y)
            - \log \text{LL}(\bar{\lambda}| y)},
        $$

        where $y_{tn}$ and $\hat{\lambda}_{tn}$ are the spike counts and the model predicted rate
         of neuron $n$ at time-point $t$ respectively, and $\bar{\lambda}$ is the mean firing rate.

        Parameters
        ----------
        X : (n_time_bins, n_neurons, n_features)
            The exogenous variables.
        spike_data : (n_time_bins, n_neurons)
            Spike counts arranged in a matrix. n_neurons must be the same as
            during the fitting of this GLM instance.

        Returns
        -------
        score : (1,)
            The Poisson log-likehood

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
        Notes
        -----

        The log-likelihood is not on a standard scale, its value is influenced by many factors,
        among which the number of model parameters. The log-likelihood can assume both positive
        and negative values.

        The pseudo-R2 is a standardized metric and assumes values between 0 and 1.

        """
        # ignore the last time point from predict, because that corresponds to
        # the next time step, which we have no observed data for
        self.check_is_fit()
        Ws = self.spike_basis_coeff_
        bs = self.baseline_log_fr_
        self.check_n_neurons(spike_data, bs)
        if self.score_type == "log-likelihood":
            score = -(self._score(X, spike_data, (Ws, bs)) + jax.scipy.special.gammaln(spike_data + 1).mean())
        elif self.score_type == "pseudo-r2":
            score = self._pseudo_r2(X, spike_data)
        else:
            # this should happen only if one manually set score_type
            raise NotImplementedError(f"Scoring method {self.score_type} not implemented!")
        return score


    def simulate(
            self,
            random_key: jax.random.PRNGKeyArray,
            n_timesteps: int,
            init_spikes: NDArray,
            coupling_basis_matrix: NDArray,
            feedforward_input: NDArray,
            device: str = 'cpu'
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Simulate spike trains using GLM as a recurrent network.

        This method extrapolates spike trains into the future. By default, it runs on the CPU. GPU
        implementations may be slow due to the non-parallelizable nature of computations. Nonetheless,
        device selection is provided to avoid data transfer overheads between devices.

        Parameters
        ----------
        random_key
            PRNGKey for seeding the simulation.
        n_timesteps
            Number of time steps for simulation.
        init_spikes
            Spike counts matrix used to initiate the simulation.
            Expected shape: (window_size, n_neurons).
        coupling_basis_matrix
            Basis matrix for coupling and auto-correlation filters.
            Expected shape: (window_size, n_basis_coupling).
        feedforward_input
            Exogenous matrix representing external inputs like convolved currents, images, etc.
            Expected shape: (n_timesteps, n_neurons, n_basis_input).
        device : optional
            Computational device to use ('cpu' or 'gpu'). Default is 'cpu'.

        Returns
        -------
        simulated_spikes
            Simulated spikes. Shape: (n_neurons, n_timesteps).
        firing_rates
            Simulated firing rates. Shape: (n_neurons, n_timesteps).

        Raises
        ------
        NotFittedError
            Raised if the instance has not been previously fitted.
        ValueError
            Raised for incompatible shapes between `init_spikes` and the fitting data,
            or between `init_spikes` and `coupling_basis_matrix`.

        See Also
        --------
        predict : Method to predict firing rates using fit parameters.

        Notes
        -----
        The sum of n_basis_input and n_basis_coupling should match `self.spike_basis_coeff_.shape[1]`.
        """
        if device == 'cpu':
            target_device = jax.devices('cpu')[0]
        elif device == 'gpu':
            target_device = jax.devices('gpu')[0]
        else:
            raise ValueError(f"Invalid device: {device}. Choose 'cpu' or 'gpu'.")

        # Transfer data to the target device
        init_spikes = jax.device_put(init_spikes, target_device)
        coupling_basis_matrix = jax.device_put(coupling_basis_matrix, target_device)
        feedforward_input = jax.device_put(feedforward_input, target_device)

        self.check_is_fit()

        Ws = self.spike_basis_coeff_
        bs = self.baseline_log_fr_
        self.check_n_neurons(init_spikes, bs)

        if feedforward_input.shape[2] + coupling_basis_matrix.shape[1] * bs.shape[0] != Ws.shape[1]:
            raise ValueError("The number of feed forward input features"
                             "and the number of recurrent features must add up to"
                             "the overall model features."
                             f"The total number of feature of the model is {Ws.shape[1]}. {feedforward_input.shape[1]} "
                             f"feedforward features and {coupling_basis_matrix.shape[1]} recurrent features "
                             f"provided instead.")

        if init_spikes.shape[0] != coupling_basis_matrix.shape[0]:
            raise ValueError(
                "init_spikes has the wrong number of time steps!"
                f"init_spikes time steps: {init_spikes.shape[1]}, "
                f"spike_basis_matrix window size: {coupling_basis_matrix.shape[1]}"
            )

        subkeys = jax.random.split(random_key, num=n_timesteps)

        def scan_fn(data: Tuple[NDArray, int], key: jax.random.PRNGKeyArray)\
                -> Tuple[Tuple[NDArray, int], NDArray]:
            """Function to scan over time steps and simulate spikes and firing rates.

            This function simulates the spikes and firing rates for each time step
            based on the previous spike data, feedforward input, and model coefficients.
            """
            spikes, chunk = data

            # Convolve the spike data with the coupling basis matrix
            conv_spk = convolve_1d_trials(coupling_basis_matrix, spikes[None, :, :])[0]

            # Extract the corresponding slice of the feedforward input for the current time step
            input_slice = jax.lax.dynamic_slice(
                feedforward_input,
                (chunk, 0, 0),
                (1, feedforward_input.shape[1], feedforward_input.shape[2])
            )

            # Reshape the convolved spikes and concatenate with the input slice to form the model input
            conv_spk = jnp.tile(conv_spk.reshape(conv_spk.shape[0], -1),
                                conv_spk.shape[1]
                                ).reshape(conv_spk.shape[0], conv_spk.shape[1], -1)
            X = jnp.concatenate([conv_spk, input_slice], axis=2)

            # Predict the firing rate using the model coefficients
            firing_rate = self._predict((Ws, bs), X)

            # Simulate spikes based on the predicted firing rate
            new_spikes = jax.random.poisson(key, firing_rate)

            # Prepare the spikes for the next iteration (keeping the most recent spikes)
            concat_spikes = jnp.row_stack((spikes[1:], new_spikes)), chunk + 1
            return concat_spikes, (new_spikes, firing_rate)

        _, outputs = jax.lax.scan(scan_fn, (init_spikes, 0), subkeys)
        simulated_spikes, firing_rates = outputs
        return jnp.squeeze(simulated_spikes, axis=1), jnp.squeeze(firing_rates, axis=1)

class GLM(GLMBase):


    def fit(
            self,
            X: NDArray,
            spike_data: NDArray,
            init_params: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    ):
        """Fit GLM to spiking data.

        Following scikit-learn API, the solutions are stored as attributes
        ``spike_basis_coeff_`` and ``baseline_log_fr``.

        Parameters
        ----------
        X :
            Predictors, shape (n_time_bins, n_neurons, n_features)
        spike_data :
            Spike counts arranged in a matrix, shape (n_time_bins, n_neurons).
        init_params :
            Initial values for the spike basis coefficients and bias terms. If
            None, we initialize with zeros. shape.  ((n_neurons, n_features), (n_neurons,))

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

        _, n_neurons = spike_data.shape
        n_features = X.shape[2]

        # Initialize parameters
        if init_params is None:
            # Ws, spike basis coeffs
            init_params = (
                jnp.zeros((n_neurons, n_features)),
                # bs, bias terms
                jnp.log(jnp.mean(spike_data, axis=0))
            )

        if init_params[0].ndim != 2:
            raise ValueError(
                "spike basis coefficients must be of shape (n_neurons, n_features), but"
                f" init_params[0] has {init_params[0].ndim} dimensions!"
            )

        if init_params[1].ndim != 1:
            raise ValueError(
                "bias terms must be of shape (n_neurons,) but init_params[0] have"
                f"{init_params[1].ndim} dimensions!"
            )
        if init_params[0].shape[0] != init_params[1].shape[0]:
            raise ValueError(
                "spike basis coefficients must be of shape (n_neurons, n_features), and"
                "bias terms must be of shape (n_neurons,) but n_neurons doesn't look the same in both!"
                f"init_params[0]: {init_params[0].shape[0]}, init_params[1]: {init_params[1].shape[0]}"
            )
        if init_params[0].shape[0] != spike_data.shape[1]:
            raise ValueError(
                "spike basis coefficients must be of shape (n_neurons, n_features), and "
                "spike_data must be of shape (n_time_bins, n_neurons) but n_neurons doesn't look the same in both! "
                f"init_params[0]: {init_params[0].shape[1]}, spike_data: {spike_data.shape[1]}"
            )

        def loss(params, X, y):
            return -self._score(X, y, params)

        # Run optimization
        solver = getattr(jaxopt, self.solver_name)(fun=loss, **self.solver_kwargs)
        params, state = solver.run(init_params, X=X, y=spike_data)

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