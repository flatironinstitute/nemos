"""GLM core module
"""
import abc
import inspect
from typing import Callable, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import jaxopt
from numpy.typing import NDArray
from sklearn.exceptions import NotFittedError

from .model_base import Model
from .utils import convolve_1d_trials


class PoissonGLMBase(Model, abc.ABC):
    """Abstract base class for Poisson GLMs.

    Provides methods for score computation, simulation, and prediction.
    Must be subclassed with a method for fitting to data.

    Parameters
    ----------
    solver_name
        Name of the solver to use when fitting the GLM. Must be an attribute of
        ``jaxopt``.
    solver_kwargs
        Dictionary of keyword arguments to pass to the solver during its
        initialization.
    inverse_link_function
        Function to transform outputs of convolution with basis to firing rate.
        Must accept any number as input and return all non-negative values.
    kwargs:
        Additional keyword arguments. ``kwargs`` may depend on the concrete
        subclass implementation (e.g. alpha, the regularization hyperparamter, will be present for
        penalized GLMs but not for the un-penalized case).

    """

    def __init__(
        self,
        solver_name: str = "GradientDescent",
        solver_kwargs: dict = dict(),
        inverse_link_function: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.softplus,
        score_type: Literal["log-likelihood", "pseudo-r2"] = "log-likelihood",
        **kwargs,
    ):
        self.solver_name = solver_name
        try:
            solver_args = inspect.getfullargspec(getattr(jaxopt, solver_name)).args
        except AttributeError:
            raise AttributeError(
                f"module jaxopt has no attribute {solver_name}, pick a different solver!"
            )

        undefined_kwargs = set(solver_kwargs.keys()).difference(solver_args)
        if undefined_kwargs:
            raise NameError(f"kwargs {undefined_kwargs} in solver_kwargs not a kwarg for jaxopt.{solver_name}!")

        if score_type not in ["log-likelihood", "pseudo-r2"]:
            raise NotImplementedError(
                "Scoring method not implemented. "
                f"score_type must be either 'log-likelihood', or 'pseudo-r2'."
                f" {score_type} provided instead."
            )
        self.score_type = score_type
        self.solver_kwargs = solver_kwargs
        self.inverse_link_function = inverse_link_function
        # set additional kwargs e.g. regularization hyperparameters and so on...
        super().__init__(**kwargs)
        # initialize parameters to None
        self.baseline_link_fr_ = None
        self.basis_coeff_ = None

    def _predict(
        self, params: Tuple[jnp.ndarray, jnp.ndarray], X: NDArray
    ) -> jnp.ndarray:
        """
        Predict firing rates given predictors and parameters.

        Parameters
        ----------
        params :
            Tuple containing the spike basis coefficients and bias terms.
        X :
            Predictors. Shape (n_time_bins, n_neurons, n_features).

        Returns
        -------
        jnp.ndarray
            The predicted firing rates. Shape (n_time_bins, n_neurons).
        """
        Ws, bs = params
        return self.inverse_link_function(jnp.einsum("ik,tik->ti", Ws, X) + bs[None, :])

    def _score(
        self,
        X: NDArray,
        target_spikes: NDArray,
        params: Tuple[jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        """Score the predicted firing rates against target spike counts.

        This computes the Poisson negative log-likelihood up to a constant.

        Note that you can end up with infinities in here if there are zeros in
        ``predicted_firing_rates``. We raise a warning in that case.

        Parameters
        ----------
        X :
            The exogenous variables. Shape (n_time_bins, n_neurons, n_features).
        target_spikes :
            The target spikes to compare against. Shape (n_time_bins, n_neurons).
        params :
            Values for the spike basis coefficients and bias terms. Shape ((n_neurons, n_features), (n_neurons,)).

        Returns
        -------
        jnp.ndarray
            The Poisson negative log-likehood. Shape (1,).

        Notes
        -----
        The Poisson probability mass function is:

        $$
           \frac{\lambda^k \exp(-\lambda)}{k!}
        $$

        But the $k!$ term is not a function of the parameters and can be disregarded
        when computing the loss-function. Thus, the negative log of it is:

        $$
           -\log{\frac{\lambda^k\exp{-\lambda}}{k!}} &= -[\log(\lambda^k)+\log(\exp{-\lambda})-\log(k!)]
           &= -k\log(\lambda)-\lambda + \text{const}
        $$

        """
        # Avoid the edge-case of 0*log(0), much faster than
        # where on large arrays.
        predicted_firing_rates = jnp.clip(self._predict(params, X), a_min=10**-10)
        x = target_spikes * jnp.log(predicted_firing_rates)
        # see above for derivation of this.
        return jnp.mean(predicted_firing_rates - x)

    def _residual_deviance(self, predicted_rate, y):
        r"""Compute the residual deviance for a Poisson model.

        Parameters
        ----------
        predicted_rate:
            The predicted firing rates.
        y:
            The spike counts.

        Returns
        -------
            The residual deviance of the model.

        Notes
        -----
        Deviance is a measure of the goodness of fit of a statistical model.
        For a Poisson model, the residual deviance is computed as:

        $$
        \begin{aligned}
            D(y, \hat{y}) &= 2 \sum \left[ y \log\left(\frac{y}{\hat{y}}\right) - (y - \hat{y}) \right]\\\
            &= -2 \left( \text{LL}\left(y | \hat{y}\right) - \text{LL}\left(y | y\right)\right)
        \end{aligned}
        $$
        where $ y $ is the observed data, $ \hat{y} $ is the predicted data, and $\text{LL}$ is the model
        log-likelihood. Lower values of deviance indicate a better fit.

        """
        # this takes care of 0s in the log
        ratio = jnp.clip(y / predicted_rate, self.FLOAT_EPS, jnp.inf)
        resid_dev = y * jnp.log(ratio) - (y - predicted_rate)
        return resid_dev

    def _pseudo_r2(self, params, X, y):
        r"""Pseudo-R^2 calculation for a Poisson GLM.

        The Pseudo-R^2 metric gives a sense of how well the model fits the data,
        relative to a null (or baseline) model.

        Parameters
        ----------
        params :
            Tuple containing the spike basis coefficients and bias terms.
        X:
            The predictors.
        y:
            The spike counts.

        Returns
        -------
        :
            The pseudo-$R^2$ of the model. A value closer to 1 indicates a better model fit,
            whereas a value closer to 0 suggests that the model doesn't improve much over the null model.

        """
        mu = self._predict(params, X)
        res_dev_t = self._residual_deviance(mu, y)
        resid_deviance = jnp.sum(res_dev_t**2)

        null_mu = jnp.ones(y.shape) * y.sum() / y.size
        null_dev_t = self._residual_deviance(null_mu, y)
        null_deviance = jnp.sum(null_dev_t**2)

        return (null_deviance - resid_deviance) / null_deviance

    def _check_is_fit(self):
        """Ensure the instance has been fitted."""
        if self.basis_coeff_ is None:
            raise NotFittedError(
                "This GLM instance is not fitted yet. Call 'fit' with appropriate arguments."
            )

    @staticmethod
    def _check_n_neurons(params, *args):
        """
        Validate the number of neurons in model parameters and input arguments.

        This function checks that the number of neurons is consistent across
        the model parameters (`params`) and any additional inputs (`args`).
        Specifically, it ensures that the spike basis coefficients and bias terms
        have the same first dimension and that this dimension matches the second
        dimension of all input matrices in `args`.

        """
        n_neurons = params[0].shape[0]
        if n_neurons != params[1].shape[0]:
            raise ValueError(
                "Model parameters have inconsistent shapes."
                "spike basis coefficients must be of shape (n_neurons, n_features), and"
                "bias terms must be of shape (n_neurons,) but n_neurons doesn't look the same in both!"
                f"coefficients n_neurons: {params[0].shape[0]}, bias n_neurons: {params[1].shape[0]}"
            )
        for arg in args:
            if arg.shape[1] != n_neurons:
                raise ValueError(
                    "The number of neuron in the model parameters and in the inputs"
                    "must match."
                    f"parameters has n_neurons: {n_neurons}, "
                    f"the input provided has n_neurons: {arg.shape[1]}"
                )

    @staticmethod
    def _check_n_features(Ws, X):
        """
        Validate the number of features between model coefficients and input data.

        This function checks that the number of features in the spike basis
        coefficients (`Ws`) matches the number of features in the input data (`X`).

        """
        if Ws.shape[1] != X.shape[2]:
            raise ValueError(
                "Inconsistent number of features. "
                f"spike basis coefficients has {Ws.shape[1]} features, "
                f"X has {X.shape[2]} features instead!"
            )

    def _check_params(
        self, params: Tuple[NDArray, NDArray], X: NDArray, spike_data: NDArray
    ):
        """
        Validate the dimensions and consistency of parameters and data.

        This function checks the consistency of shapes and dimensions for model
        parameters, input predictors (`X`), and spike counts (`spike_data`).
        It ensures that the parameters and data are compatible for the model.

        """
        if len(params) != 2:
            raise ValueError("Params needs to be a JAX pytree of size two of NDArray.")

        if params[0].ndim != 2:
            raise ValueError(
                "Weights must be of shape (n_neurons, n_features), but"
                f"params[0] has {params[0].ndim} dimensions!"
            )
        if params[1].ndim != 1:
            raise ValueError(
                "params[1] term must be of shape (n_neurons,) but params[1] have"
                f"{params[1].ndim} dimensions!"
            )

        # check that the neurons
        self._check_n_neurons(params, X, spike_data)

        if spike_data.ndim != 2:
            raise ValueError(
                "spike_data must be two-dimensional, with shape (n_timebins, n_neurons)"
            )
        if X.ndim != 3:
            raise ValueError(
                "X must be three-dimensional, with shape (n_timebins, n_neurons, n_features)"
            )
        self._check_n_features(params[0], X)

    def predict(self, X: NDArray) -> jnp.ndarray:
        """Predict firing rates based on fit parameters.

        Parameters
        ----------
        X : NDArray
            The exogenous variables. Shape (n_time_bins, n_neurons, n_features).

        Returns
        -------
        predicted_firing_rates : jnp.ndarray
            The predicted firing rates with shape (n_neurons, n_time_bins).

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called first with this instance.
        ValueError
            - If `params` is not a JAX pytree of size two.
            - If weights and bias terms in `params` don't have the expected dimensions.
            - If the number of neurons in the model parameters and in the inputs do not match.
            - If `X` is not three-dimensional.
            - If there's an inconsistent number of features between spike basis coefficients and `X`.

        See Also
        --------
        score
            Score predicted firing rates against target spike counts.
        simulate
            Simulate spikes using GLM as a recurrent network, for extrapolating into the future.
        """
        self._check_is_fit()
        Ws = self.basis_coeff_
        bs = self.baseline_link_fr_
        self._check_n_neurons((Ws, bs), X)
        self._check_n_features(Ws, X)
        return self._predict((Ws, bs), X)

    def score(self,
              X: NDArray,
              spike_data: NDArray,
              score_type: Optional[Literal["log-likelihood", "pseudo-r2"]] = None) -> jnp.ndarray:
        r"""Score the predicted firing rates (based on fit) to the target spike counts.

        This computes the Poisson mean log-likelihood or the pseudo-$R^2$, thus the higher the
        number the better.

        The formula for the mean log-likelihood is the following,

        $$
        \begin{aligned}
        \text{LL}(\hat{\lambda} | y) &= \frac{1}{T \cdot N} \sum_{n=1}^{N} \sum_{t=1}^{T}
        [y\_{tn} \log(\hat{\lambda}\_{tn}) - \hat{\lambda}\_{tn} - \log({y\_{tn}!})] \\\
        &= \frac{1}{T \cdot N} [y\_{tn} \log(\hat{\lambda}\_{tn}) - \hat{\lambda}\_{tn} - \Gamma({y\_{tn}+1})]
        \end{aligned}
        $$

        Because $\Gamma(k+1)=k!$, see
        https://en.wikipedia.org/wiki/Gamma_function.

        The pseudo-$R^2$ can be computed as follows,

        $$
        \begin{aligned}
            R^2_{\text{pseudo}} &= \frac{D_{\text{null}} - D_{\text{model}}}{D_{\text{null}}} \\\
            &= \frac{\log \text{LL}(\hat{\lambda}| y) - \log \text{LL}(\bar{\lambda}| y)}{\log \text{LL}(y| y)
            - \log \text{LL}(\bar{\lambda}| y)},
        \end{aligned}
        $$

        where $D_{\text{null}}$ is the deviance for a null model, $D_{\text{model}}$ is the deviance for
        the current model, $y_{tn}$ and $\hat{\lambda}_{tn}$ are the spike counts and the model predicted rate
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
            The Poisson log-likehood or the pseudo-$R^2$ of the current model.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called first with this instance.
        ValueError
            If attempting to simulate a different number of neurons than were
            present during fitting (i.e., if ``init_spikes.shape[0] !=
            self.baseline_link_fr_.shape[0]``).

        Notes
        -----
        The log-likelihood is not on a standard scale, its value is influenced by many factors,
        among which the number of model parameters. The log-likelihood can assume both positive
        and negative values.

        The Pseudo-$R^2$ is not equivalent to the $R^2$ value in linear regression. While both provide a measure
        of model fit, and assume values in the [0,1] range, the methods and interpretations can differ.
        The Pseudo-$R^2$ is particularly useful for generalized linear models where a traditional $R^2$ doesn't apply.
        """

        # ignore the last time point from predict, because that corresponds to
        # the next time step, which we have no observed data for
        self._check_is_fit()
        Ws = self.basis_coeff_
        bs = self.baseline_link_fr_
        self._check_n_neurons((Ws, bs), X, spike_data)
        self._check_n_features(Ws, X)

        if score_type is None:
            score_type = self.score_type

        if score_type == "log-likelihood":
            score = -(
                self._score(X, spike_data, (Ws, bs))
                + jax.scipy.special.gammaln(spike_data + 1).mean()
            )
        elif score_type == "pseudo-r2":
            score = self._pseudo_r2((Ws,bs), X, spike_data)
        else:
            # this should happen only if one manually set score_type
            raise NotImplementedError(
                f"Scoring method {self.score_type} not implemented!"
            )
        return score

    @abc.abstractmethod
    def fit(
        self,
        X: NDArray,
        spike_data: NDArray,
        init_params: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    ):
        """Fit GLM to spiking data.

        Following scikit-learn API, the solutions are stored as attributes
        ``basis_coeff_`` and ``baseline_link_fr``.

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

    def simulate(
        self,
        random_key: jax.random.PRNGKeyArray,
        n_timesteps: int,
        init_spikes: NDArray,
        coupling_basis_matrix: NDArray,
        feedforward_input: NDArray,
        device: str = "cpu",
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Simulate spike trains using the GLM as a recurrent network.

        This function projects spike trains into the future, employing the fitted
        parameters of the GLM. While the default computation device is the CPU,
        users can opt for GPU; however, it may not provide substantial speed-up due
        to the inherently sequential nature of certain computations.

        Parameters
        ----------
        random_key :
            PRNGKey for seeding the simulation.
        n_timesteps :
            Duration of the simulation in terms of time steps.
        init_spikes :
            Initial spike counts matrix that kickstarts the simulation.
            Expected shape: (window_size, n_neurons).
        coupling_basis_matrix :
            Basis matrix for coupling, representing inter-neuron effects
            and auto-correlations. Expected shape: (window_size, n_basis_coupling).
        feedforward_input :
            External input matrix, representing factors like convolved currents,
            light intensities, etc. Expected shape: (n_timesteps, n_neurons, n_basis_input).
        device :
            Computation device to use ('cpu' or 'gpu'). Default is 'cpu'.

        Returns
        -------
        simulated_spikes :
            Simulated spike counts for each neuron over time.
            Shape: (n_neurons, n_timesteps).
        firing_rates :
            Simulated firing rates for each neuron over time.
            Shape: (n_neurons, n_timesteps).

        Raises
        ------
        NotFittedError
            If the model hasn't been fitted prior to calling this method.
        Raises
        ------
        ValueError
            - If the instance has not been previously fitted.
            - If there's an inconsistency between the number of neurons in model parameters.
            - If the number of neurons in input arguments doesn't match with model parameters.
            - For an invalid computational device selection.


        See Also
        --------
        predict : Method to predict firing rates based on the model's parameters.

        Notes
        -----
        The sum of n_basis_input and n_basis_coupling should equal `self.basis_coeff_.shape[1]` to ensure
        consistency in the model's input feature dimensionality.
        """
        if device == "cpu":
            target_device = jax.devices("cpu")[0]
        elif device == "gpu":
            target_device = jax.devices("gpu")[0]
        else:
            raise ValueError(f"Invalid device: {device}. Choose 'cpu' or 'gpu'.")

        # Transfer data to the target device
        init_spikes = jax.device_put(init_spikes, target_device)
        coupling_basis_matrix = jax.device_put(coupling_basis_matrix, target_device)
        feedforward_input = jax.device_put(feedforward_input, target_device)

        self._check_is_fit()

        Ws = self.basis_coeff_
        bs = self.baseline_link_fr_
        self._check_n_neurons((Ws, bs), feedforward_input, init_spikes)

        if (
            feedforward_input.shape[2] + coupling_basis_matrix.shape[1] * bs.shape[0]
            != Ws.shape[1]
        ):
            raise ValueError(
                "The number of feed forward input features"
                "and the number of recurrent features must add up to"
                "the overall model features."
                f"The total number of feature of the model is {Ws.shape[1]}. {feedforward_input.shape[1]} "
                f"feedforward features and {coupling_basis_matrix.shape[1]} recurrent features "
                f"provided instead."
            )

        if init_spikes.shape[0] != coupling_basis_matrix.shape[0]:
            raise ValueError(
                "init_spikes has the wrong number of time steps!"
                f"init_spikes time steps: {init_spikes.shape[1]}, "
                f"spike_basis_matrix window size: {coupling_basis_matrix.shape[1]}"
            )

        subkeys = jax.random.split(random_key, num=n_timesteps)

        def scan_fn(
            data: Tuple[NDArray, int], key: jax.random.PRNGKeyArray
        ) -> Tuple[Tuple[NDArray, int], NDArray]:
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
                (1, feedforward_input.shape[1], feedforward_input.shape[2]),
            )

            # Reshape the convolved spikes and concatenate with the input slice to form the model input
            conv_spk = jnp.tile(
                conv_spk.reshape(conv_spk.shape[0], -1), conv_spk.shape[1]
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


class PoissonGLM(PoissonGLMBase):
    """Un-regularized Poisson-GLM.

    The class fits the un-penalized maximum likelihood Poisson GLM parameter estimate.

    Parameters
    ----------
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
    basis_coeff_ : jnp.ndarray, (n_neurons, n_basis_funcs, n_neurons)
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
    ):
        super().__init__(
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
            inverse_link_function=inverse_link_function,
            score_type=score_type,
        )

    def fit(
        self,
        X: NDArray,
        spike_data: NDArray,
        init_params: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    ):
        """Fit GLM to spiking data.

        Following scikit-learn API, the solutions are stored as attributes
        ``basis_coeff_`` and ``baseline_log_fr``.

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
            - If `params` is not a JAX pytree of size two.
            - If shapes of init_params are not correct.
            - If the number of neurons in the model parameters and in the inputs do not match.
            - If `X` is not three-dimensional.
            - If spike_data is not two-dimensional.
            - If solver returns at least one NaN parameter, which means it found
              an invalid solution. Try tuning optimization hyperparameters.

        """
        _, n_neurons = spike_data.shape
        n_features = X.shape[2]

        # Initialize parameters
        if init_params is None:
            # Ws, spike basis coeffs
            init_params = (
                jnp.zeros((n_neurons, n_features)),
                # bs, bias terms
                jnp.log(jnp.mean(spike_data, axis=0)),
            )

        self._check_params(init_params, X, spike_data)

        def loss(params, X, y):
            return self._score(X, y, params)

        # Run optimization
        solver = getattr(jaxopt, self.solver_name)(fun=loss, **self.solver_kwargs)
        params, state = solver.run(init_params, X=X, y=spike_data)

        if jnp.isnan(params[0]).any() or jnp.isnan(params[1]).any():
            raise ValueError(
                "Solver returned at least one NaN parameter, so solution is invalid!"
                " Try tuning optimization hyperparameters."
            )
        # Store parameters
        self.basis_coeff_ = params[0]
        self.baseline_link_fr_ = params[1]
        # note that this will include an error value, which is not the same as
        # the output of loss. I believe it's the output of
        # solver.l2_optimality_error
        self.solver_state = state
        self.solver = solver
