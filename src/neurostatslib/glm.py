"""GLM core module
"""
import abc
import inspect
from typing import Callable, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jaxopt
from numpy.typing import ArrayLike, NDArray

from .base_class import _BaseRegressor
from .exceptions import NotFittedError
from .utils import convolve_1d_trials


class _BaseGLM(_BaseRegressor, abc.ABC):
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
            raise NameError(
                f"kwargs {undefined_kwargs} in solver_kwargs not a kwarg for jaxopt.{solver_name}!"
            )

        if score_type not in ["log-likelihood", "pseudo-r2"]:
            raise NotImplementedError(
                f"Scoring method {score_type} not implemented! "
                f"`score_type` must be either 'log-likelihood', or 'pseudo-r2'."
            )
        self.score_type = score_type
        self.solver_kwargs = solver_kwargs

        if not callable(inverse_link_function):
            raise ValueError("inverse_link_function must be a callable!")

        self.inverse_link_function = inverse_link_function
        # set additional kwargs e.g. regularization hyperparameters and so on...
        super().__init__(**kwargs)
        # initialize parameters to None
        self.baseline_link_fr_ = None
        self.basis_coeff_ = None

    def _predict(
        self, params: Tuple[jnp.ndarray, jnp.ndarray], X: jnp.ndarray
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

    @abc.abstractmethod
    def residual_deviance(self, predicted_rate, y):
        r"""Compute the residual deviance for a GLM model.

        Parameters
        ----------
        predicted_rate:
            The predicted rate of the GLM.
        y:
            The observations.

        Returns
        -------
            The residual deviance of the model.

        Notes
        -----
        Deviance is a measure of the goodness of fit of a statistical model.
        For a Poisson model, the residual deviance is computed as:

        $$
        \begin{aligned}
            D(y, \hat{y}) &= -2 \left( \text{LL}\left(y | \hat{y}\right) - \text{LL}\left(y | y\right)\right)
        \end{aligned}
        $$
        where $ y $ is the observed data, $ \hat{y} $ is the predicted data, and $\text{LL}$ is the model
        log-likelihood. Lower values of deviance indicate a better fit.

        """
        pass

    def _pseudo_r2(self, params, X, y):
        r"""Pseudo-R^2 calculation for a GLM.

        The Pseudo-R^2 metric gives a sense of how well the model fits the data,
        relative to a null (or baseline) model.

        Parameters
        ----------
        params :
            Tuple containing the spike basis coefficients and bias terms.
        X:
            The predictors.
        y:
            The neural activity.

        Returns
        -------
        :
            The pseudo-$R^2$ of the model. A value closer to 1 indicates a better model fit,
            whereas a value closer to 0 suggests that the model doesn't improve much over the null model.

        """
        mu = self._predict(params, X)
        res_dev_t = self.residual_deviance(mu, y)
        resid_deviance = jnp.sum(res_dev_t**2)

        null_mu = jnp.ones(y.shape, dtype=jnp.float32) * y.mean()
        null_dev_t = self.residual_deviance(null_mu, y)
        null_deviance = jnp.sum(null_dev_t**2)

        return (null_deviance - resid_deviance) / null_deviance

    def _check_is_fit(self):
        """Ensure the instance has been fitted."""
        if (self.basis_coeff_ is None) or (self.baseline_link_fr_ is None):
            raise NotFittedError(
                "This GLM instance is not fitted yet. Call 'fit' with appropriate arguments."
            )

    def _safe_predict(self, X: Union[NDArray, jnp.ndarray]) -> jnp.ndarray:
        """Predict firing rates based on fit parameters.

        Parameters
        ----------
        X :
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
        """
        # check that the model is fitted
        self._check_is_fit()
        # extract model params
        Ws = self.basis_coeff_
        bs = self.baseline_link_fr_

        (X,) = self._convert_to_jnp_ndarray(X, data_type=jnp.float32)

        # check input dimensionality
        self._check_input_dimensionality(X=X)
        # check consistency between X and params
        self._check_input_and_params_consistency((Ws, bs), X=X)

        return self._predict((Ws, bs), X)

    def _safe_score(
        self,
        X: Union[NDArray, jnp.ndarray],
        y: Union[NDArray, jnp.ndarray],
        score_func: Callable[
            [jnp.ndarray, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]], jnp.ndarray
        ],
        score_type: Optional[Literal["log-likelihood", "pseudo-r2"]] = None,
    ) -> jnp.ndarray:
        r"""Score the predicted firing rates (based on fit) to the target spike counts.

        This computes the GLM mean log-likelihood or the pseudo-$R^2$, thus the higher the
        number the better.

        The pseudo-$R^2$ can be computed as follows,

        $$
        \begin{aligned}
            R^2_{\text{pseudo}} &= \frac{D_{\text{null}} - D_{\text{model}}}{D_{\text{null}}} \\\
            &= \frac{\log \text{LL}(\hat{\lambda}| y) - \log \text{LL}(\bar{\lambda}| y)}{\log \text{LL}(y| y)
            - \log \text{LL}(\bar{\lambda}| y)},
        \end{aligned}
        $$

        where LL is the log-likelihood, $D_{\text{null}}$ is the deviance for a null model, $D_{\text{model}}$ is
        the deviance for the current model, $y_{tn}$ and $\hat{\lambda}_{tn}$ are the observed activity and the model
        predicted rate for neuron $n$ at time-point $t$, and $\bar{\lambda}$ is the mean firing rate. See [1].

        Parameters
        ----------
        X :
            The exogenous variables. Shape (n_time_bins, n_neurons, n_features)
        spike_data :
            Spike counts arranged in a matrix. n_neurons must be the same as
            during the fitting of this GLM instance. Shape (n_time_bins, n_neurons).
        score_type:
            String indicating the type of scoring to return. Options are:
                - `log-likelihood` for the model log-likelihood.
                - `pseudo-r2` for the model pseudo-$R^2$.
            Default is defined at class initialization.
        Returns
        -------
        score : (1,)
            The Poisson log-likelihood or the pseudo-$R^2$ of the current model.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called first with this instance.
        ValueError
            If attempting to simulate a different number of neurons than were
            present during fitting (i.e., if ``init_y.shape[0] !=
            self.baseline_link_fr_.shape[0]``).

        Notes
        -----
        The log-likelihood is not on a standard scale, its value is influenced by many factors,
        among which the number of model parameters. The log-likelihood can assume both positive
        and negative values.

        The Pseudo-$R^2$ is not equivalent to the $R^2$ value in linear regression. While both provide a measure
        of model fit, and assume values in the [0,1] range, the methods and interpretations can differ.
        The Pseudo-$R^2$ is particularly useful for generalized linear models where a traditional $R^2$ doesn't apply.

        Refer to the concrete subclass docstrings `_score` for the specific likelihood equations.

        References
        ----------
        [1] Cohen, Jacob, et al. Applied multiple regression/correlation analysis for the behavioral sciences.
        Routledge, 2013.

        """

        # ignore the last time point from predict, because that corresponds to
        # the next time step, which we have no observed data for
        self._check_is_fit()
        Ws = self.basis_coeff_
        bs = self.baseline_link_fr_

        X, y = self._convert_to_jnp_ndarray(X, y, data_type=jnp.float32)

        self._check_input_dimensionality(X, y)
        self._check_input_n_timepoints(X, y)
        self._check_input_and_params_consistency((Ws, bs), X=X, y=y)

        if score_type is None:
            score_type = self.score_type

        if score_type == "log-likelihood":
            score = -(score_func(X, y, (Ws, bs)))
        elif score_type == "pseudo-r2":
            score = self._pseudo_r2((Ws, bs), X, y)
        else:
            # this should happen only if one manually set score_type
            raise NotImplementedError(
                f"Scoring method {score_type} not implemented! "
                f"`score_type` must be either 'log-likelihood', or 'pseudo-r2'."
            )
        return score

    def _safe_fit(
        self,
        X: Union[NDArray, jnp.ndarray],
        y: Union[NDArray, jnp.ndarray],
        loss: Callable[
            [Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray], jnp.float32
        ],
        init_params: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        device: Literal["cpu", "gpu", "tpu"] = "gpu",
    ):
        """Fit GLM to neuroal activity.

        Following scikit-learn API, the solutions are stored as attributes
        ``basis_coeff_`` and ``baseline_link_fr``.

        Parameters
        ----------
        X :
            Predictors, shape (n_time_bins, n_neurons, n_features)
        y :
            Spike counts arranged in a matrix, shape (n_time_bins, n_neurons).
        loss:
            The loss function to be minimized.
        init_params :
            Initial values for the spike basis coefficients and bias terms. If
            None, we initialize with zeros. shape.  ((n_neurons, n_features), (n_neurons,))
        device:
            Device used for optimizing model parameters.
        Raises
        ------
        ValueError
            - If `init_params` is not of length two.
            - If dimensionality of `init_params` are not correct.
            - If the number of neurons in the model parameters and in the inputs do not match.
            - If `X` is not three-dimensional.
            - If spike_data is not two-dimensional.
            - If solver returns at least one NaN parameter, which means it found
              an invalid solution. Try tuning optimization hyperparameters.
        TypeError
            - If `init_params` are not array-like
            - If `init_params[i]` cannot be converted to jnp.ndarray for all i
        """
        # convert to jnp.ndarray & perform checks
        X, y, init_params = self._preprocess_fit(X, y, init_params)

        # send to device
        target_device = self.select_target_device(device)
        X, y = self.device_put(X, y, device=target_device)
        init_params = self.device_put(*init_params, device=target_device)

        # Run optimization
        solver = getattr(jaxopt, self.solver_name)(fun=loss, **self.solver_kwargs)
        params, state = solver.run(init_params, X=X, y=y)

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

    def _safe_simulate(
        self,
        random_key: jax.random.PRNGKeyArray,
        n_timesteps: int,
        init_y: Union[NDArray, jnp.ndarray],
        coupling_basis_matrix: Union[NDArray, jnp.ndarray],
        random_function: Callable[[jax.random.PRNGKeyArray, ArrayLike], jnp.ndarray],
        feedforward_input: Optional[Union[NDArray, jnp.ndarray]] = None,
        device: Literal["cpu", "gpu", "tpu"] = "cpu",
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Simulate spike trains using the GLM as a recurrent network.

        This function projects neural activity into the future, employing the fitted
        parameters of the GLM. It is capable of simulating activity based on a combination
        of historical spike activity and external feedforward inputs like convolved currents, light
        intensities, etc.

        Parameters
        ----------
        random_key :
            PRNGKey for seeding the simulation.
        n_timesteps :
            Duration of the simulation in terms of time steps.
        init_y :
            Initial observation (spike counts for PoissonGLM) matrix that kickstarts the simulation.
            Expected shape: (window_size, n_neurons).
        coupling_basis_matrix :
            Basis matrix for coupling, representing between-neuron couplings
            and auto-correlations. Expected shape: (window_size, n_basis_coupling).
        random_function :
            A probability emission function, like jax.random.poisson, which takes as input a random.PRNGKeyArray
            and the mean rate, and samples observations, (spike counts for a poisson)..
        feedforward_input :
            External input matrix to the model, representing factors like convolved currents,
            light intensities, etc. When not provided, the simulation is done with coupling-only.
            Expected shape: (n_timesteps, n_neurons, n_basis_input).
        device :
            Computation device to use ('cpu', 'gpu', or 'tpu'). Default is 'cpu'.

        Returns
        -------
        simulated_obs :
            Simulated observations (spike counts for PoissonGLMs) for each neuron over time.
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
        The model coefficients (`self.basis_coeff_`) are structured such that the first set of coefficients
        (of size `n_basis_coupling * n_neurons`) are interpreted as the weights for the recurrent couplings.
        The remaining coefficients correspond to the weights for the feed-forward input.


        The sum of `n_basis_input` and `n_basis_coupling * n_neurons` should equal `self.basis_coeff_.shape[1]`
        to ensure consistency in the model's input feature dimensionality.
        """
        target_device = self.select_target_device(device)
        # check if the model is fit
        self._check_is_fit()

        # convert to jnp.ndarray
        init_y, coupling_basis_matrix, feedforward_input = self._convert_to_jnp_ndarray(
            init_y, coupling_basis_matrix, feedforward_input, data_type=jnp.float32
        )

        # Transfer data to the target device
        init_y, coupling_basis_matrix, feedforward_input = self.device_put(
            init_y, coupling_basis_matrix, feedforward_input, device=target_device
        )

        n_basis_coupling = coupling_basis_matrix.shape[1]
        n_neurons = self.baseline_link_fr_.shape[0]

        # add an empty input (simulate with coupling-only)
        if feedforward_input is None:
            feedforward_input = jnp.zeros(
                (n_timesteps, n_neurons, 0), dtype=jnp.float32
            )

        Ws = self.basis_coeff_
        bs = self.baseline_link_fr_

        self._check_input_dimensionality(feedforward_input, init_y)

        if (
            feedforward_input.shape[2] + coupling_basis_matrix.shape[1] * bs.shape[0]
            != Ws.shape[1]
        ):
            raise ValueError(
                "The number of feed forward input features "
                "and the number of recurrent features must add up to "
                "the overall model features."
                f"The total number of feature of the model is {Ws.shape[1]}. {feedforward_input.shape[1]} "
                f"feedforward features and {coupling_basis_matrix.shape[1]} recurrent features "
                f"provided instead."
            )

        self._check_input_and_params_consistency(
            (Ws[:, n_basis_coupling * n_neurons :], bs),
            X=feedforward_input,
            y=init_y,
        )

        if init_y.shape[0] != coupling_basis_matrix.shape[0]:
            raise ValueError(
                "`init_y` and `coupling_basis_matrix`"
                " should have the same window size! "
                f"`init_y` window size: {init_y.shape[1]}, "
                f"`spike_basis_matrix` window size: {coupling_basis_matrix.shape[1]}"
            )

        if feedforward_input.shape[0] != n_timesteps:
            raise ValueError(
                "`feedforward_input` must be of length `n_timesteps`. "
                f"`feedforward_input` has length {len(feedforward_input)}, "
                f"`n_timesteps` is {n_timesteps} instead!"
            )
        subkeys = jax.random.split(random_key, num=n_timesteps)

        def scan_fn(
            data: Tuple[jnp.ndarray, int], key: jax.random.PRNGKeyArray
        ) -> Tuple[Tuple[jnp.ndarray, int], Tuple[jnp.ndarray, jnp.ndarray]]:
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
            new_spikes = random_function(key, firing_rate)

            # Prepare the spikes for the next iteration (keeping the most recent spikes)
            concat_spikes = jnp.row_stack((spikes[1:], new_spikes)), chunk + 1
            return concat_spikes, (new_spikes, firing_rate)

        _, outputs = jax.lax.scan(scan_fn, (init_y, 0), subkeys)
        simulated_spikes, firing_rates = outputs
        return jnp.squeeze(simulated_spikes, axis=1), jnp.squeeze(firing_rates, axis=1)


class PoissonGLM(_BaseGLM):
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
    baseline_link_fr : jnp.ndarray, (n_neurons,)
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

    def _score(
        self,
        X: jnp.ndarray,
        target_spikes: jnp.ndarray,
        params: Tuple[jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        """Score the predicted firing rates against target spike counts.

        This computes the Poisson negative log-likelihood up to a constant.

        Note that you can end up with infinities in here if there are zeros in
        ``predicted_firing_rates``. We raise a warning in that case.

        The formula for the Poisson mean log-likelihood is the following,

        $$
        \begin{aligned}
        \text{LL}(\hat{\lambda} | y) &= \frac{1}{T \cdot N} \sum_{n=1}^{N} \sum_{t=1}^{T}
        [y\_{tn} \log(\hat{\lambda}\_{tn}) - \hat{\lambda}\_{tn} - \log({y\_{tn}!})] \\\
        &= \frac{1}{T \cdot N} [y\_{tn} \log(\hat{\lambda}\_{tn}) - \hat{\lambda}\_{tn} - \Gamma({y\_{tn}+1})]
        \end{aligned}
        $$

        Because $\Gamma(k+1)=k!$, see
        https://en.wikipedia.org/wiki/Gamma_function.

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
        :
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

    def residual_deviance(
        self, predicted_rate: jnp.ndarray, spike_counts: jnp.ndarray
    ) -> jnp.ndarray:
        r"""Compute the residual deviance for a Poisson model.

        Parameters
        ----------
        predicted_rate:
            The predicted firing rates.
        spike_counts:
            The spike counts.

        Returns
        -------
        :
            The residual deviance of the model.

        Notes
        -----
        Deviance is a measure of the goodness of fit of a statistical model.
        For a Poisson model, the residual deviance is computed as:

        $$
        \begin{aligned}
            D(y\_{tn}, \hat{y}\_{tn}) &= 2 \left[ y\_{tn} \log\left(\frac{y\_{tn}}{\hat{y}\_{tn}}\right)
            - (y\_{tn} - \hat{y}\_{tn}) \right]\\\
            &= -2 \left( \text{LL}\left(y\_{tn} | \hat{y}\_{tn}\right) - \text{LL}\left(y\_{tn} | y\_{tn}\right)\right)
        \end{aligned}
        $$
        where $ y $ is the observed data, $ \hat{y} $ is the predicted data, and $\text{LL}$ is the model
        log-likelihood. Lower values of deviance indicate a better fit.

        """
        # this takes care of 0s in the log
        ratio = jnp.clip(spike_counts / predicted_rate, self.FLOAT_EPS, jnp.inf)
        resid_dev = 2 * (
            spike_counts * jnp.log(ratio) - (spike_counts - predicted_rate)
        )
        return resid_dev

    def predict(self, X: Union[NDArray, jnp.ndarray]):
        """Predict firing rates based on fit parameters.

        Parameters
        ----------
        X :
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
        [score](../glm/#neurostatslib.glm.PoissonGLM.score)
            Score predicted firing rates against target spike counts.
        """
        return self._safe_predict(X)

    def score(
        self,
        X: Union[NDArray, jnp.ndarray],
        y: Union[NDArray, jnp.ndarray],
        score_type: Literal["log-likelihood", "pseudo-r2"] = "log-likelihood",
    ) -> jnp.ndarray:
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
         of neuron $n$ at time-point $t$ respectively, and $\bar{\lambda}$ is the mean firing rate. See [1].

        Parameters
        ----------
        X :
            The exogenous variables. Shape (n_time_bins, n_neurons, n_features)
        y :
            Spike counts arranged in a matrix. n_neurons must be the same as
            during the fitting of this GLM instance. Shape (n_time_bins, n_neurons).
        score_type:
            String indicating the type of scoring to return. Options are:
                - `log-likelihood` for the model log-likelihood.
                - `pseudo-r2` for the model pseudo-$R^2$.
            Default is defined at class initialization.
        Returns
        -------
        score :
            The Poisson log-likelihood or the pseudo-$R^2$ of the current model.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called first with this instance.
        ValueError
            If attempting to simulate a different number of neurons than were
            present during fitting (i.e., if ``init_y.shape[0] !=
            self.baseline_link_fr_.shape[0]``).

        Notes
        -----
        The log-likelihood is not on a standard scale, its value is influenced by many factors,
        among which the number of model parameters. The log-likelihood can assume both positive
        and negative values.

        The Pseudo-$R^2$ is not equivalent to the $R^2$ value in linear regression. While both provide a measure
        of model fit, and assume values in the [0,1] range, the methods and interpretations can differ.
        The Pseudo-$R^2$ is particularly useful for generalized linear models where a traditional $R^2$ doesn't apply.

        References
        ----------
        [1] Cohen, Jacob, et al. Applied multiple regression/correlation analysis for the behavioral sciences.
        Routledge, 2013.

        """
        norm_constant = jax.scipy.special.gammaln(y + 1).mean()
        return (
            super()._safe_score(X=X, y=y, score_type=score_type, score_func=self._score)
            - norm_constant
        )

    def fit(
        self,
        X: Union[NDArray, jnp.ndarray],
        y: Union[NDArray, jnp.ndarray],
        init_params: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        device: Literal["cpu", "gpu", "tpu"] = "gpu",
    ):
        """Fit GLM to spiking data.

        Following scikit-learn API, the solutions are stored as attributes
        ``basis_coeff_`` and ``baseline_link_fr``.

        Parameters
        ----------
        X :
            Predictors, shape (n_time_bins, n_neurons, n_features)
        y :
            Spike counts arranged in a matrix, shape (n_time_bins, n_neurons).
        init_params :
            Initial values for the spike basis coefficients and bias terms. If
            None, we initialize with zeros. shape.  ((n_neurons, n_features), (n_neurons,))
        device:
            Device used for optimizing model parameters.
        Raises
        ------
        ValueError
            - If `init_params` is not of length two.
            - If dimensionality of `init_params` are not correct.
            - If the number of neurons in the model parameters and in the inputs do not match.
            - If `X` is not three-dimensional.
            - If spike_data is not two-dimensional.
            - If solver returns at least one NaN parameter, which means it found
              an invalid solution. Try tuning optimization hyperparameters.
        TypeError
            - If `init_params` are not array-like
            - If `init_params[i]` cannot be converted to jnp.ndarray for all i

        """

        def loss(params, X, y):
            return self._score(X, y, params)

        self._safe_fit(X=X, y=y, loss=loss, init_params=init_params, device=device)

    def simulate(
        self,
        random_key: jax.random.PRNGKeyArray,
        n_timesteps: int,
        init_y: Union[NDArray, jnp.ndarray],
        coupling_basis_matrix: Union[NDArray, jnp.ndarray],
        feedforward_input: Optional[Union[NDArray, jnp.ndarray]] = None,
        device: Literal["cpu", "gpu", "tpu"] = "cpu",
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Simulate spike trains using the  Poisson-GLM as a recurrent network.

        This function projects spike trains into the future, employing the fitted
        parameters of the GLM. It is capable of simulating spike trains based on a combination
        of historical spike activity and external feedforward inputs like convolved currents, light
        intensities, etc.


        Parameters
        ----------
        random_key :
            PRNGKey for seeding the simulation.
        n_timesteps :
            Duration of the simulation in terms of time steps.
        init_y :
            Initial spike counts matrix that kickstarts the simulation.
            Expected shape: (window_size, n_neurons).
        coupling_basis_matrix :
            Basis matrix for coupling, representing between-neuron couplings
            and auto-correlations. Expected shape: (window_size, n_basis_coupling).
        feedforward_input :
            External input matrix to the model, representing factors like convolved currents,
            light intensities, etc. When not provided, the simulation is done with coupling-only.
            Expected shape: (n_timesteps, n_neurons, n_basis_input).
        device :
            Computation device to use ('cpu', 'gpu' or 'tpu'). Default is 'cpu'.

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
        ValueError
            - If the instance has not been previously fitted.
            - If there's an inconsistency between the number of neurons in model parameters.
            - If the number of neurons in input arguments doesn't match with model parameters.
            - For an invalid computational device selection.


        See Also
        --------
        [predict](../glm/#neurostatslib.glm.PoissonGLM.predict) : Method to predict firing rates based on
        the model's parameters.

        Notes
        -----
        The model coefficients (`self.basis_coeff_`) are structured such that the first set of coefficients
        (of size `n_basis_coupling * n_neurons`) are interpreted as the weights for the recurrent couplings.
        The remaining coefficients correspond to the weights for the feed-forward input.


        The sum of `n_basis_input` and `n_basis_coupling * n_neurons` should equal `self.basis_coeff_.shape[1]`
        to ensure consistency in the model's input feature dimensionality.
        """
        simulated_spikes, firing_rates = super()._safe_simulate(
            random_key=random_key,
            n_timesteps=n_timesteps,
            init_y=init_y,
            coupling_basis_matrix=coupling_basis_matrix,
            random_function=jax.random.poisson,
            feedforward_input=feedforward_input,
            device=device,
        )
        return simulated_spikes, firing_rates
