"""GLM core module."""
import inspect
from typing import Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from numpy.typing import NDArray

from . import observation_models as nsm
from . import solver as slv
from .base_class import BaseRegressor
from .exceptions import NotFittedError
from .utils import convert_to_jnp_ndarray, convolve_1d_trials


class GLM(BaseRegressor):
    """
    Generalized Linear Model (GLM) for neural activity data.

    This GLM implementation allows users to model neural activity based on a combination of exogenous inputs
    (like convolved currents or light intensities) and a choice of observation model. It is suitable for scenarios where
    the relationship between predictors and the response variable might be non-linear, and the residuals
    don't follow a normal distribution.

    Parameters
    ----------
    observation_model :
        Observation model to use. The model describes the distribution of the neural activity.
        Default is the Poisson model.
    solver :
        Solver to use for model optimization. Defines the optimization algorithm and related parameters.
        Default is Ridge regression with gradient descent.

    Attributes
    ----------
    baseline_link_fr_ :
        Model baseline link firing rate parameters.
    basis_coeff_ :
        Basis coefficients for the model.
    solver_state :
        State of the solver after fitting. May include details like optimization error.

    Raises
    ------
    TypeError
        If provided `solver` or `observation_model` are not valid or implemented in `neurostatslib.solver` and
        `neurostatslib.observation_models` respectively.
    """

    def __init__(
        self,
        observation_model: nsm.Observations = nsm.PoissonObservations(),
        solver: slv.Solver = slv.RidgeSolver("GradientDescent"),
    ):
        super().__init__()

        if not hasattr(solver, "instantiate_solver"):
            raise TypeError(
                "The provided `solver` does not implements the `instantiate_solver` method."
            )

        # this catches the corner case of users passing classes before instantiation. Example,
        # `sovler = nsl.solver.RidgeSolver` instead of `sovler = nsl.solver.RidgeSolver()`.
        # It also catches solvers that do not respect the api of having a single loss function as input.
        if len(inspect.signature(solver.instantiate_solver).parameters) != 1:
            raise TypeError(
                "The `instantiate_solver` method of `solver` must accept a single parameter, the loss function"
                "Have you instantiate the class?"
            )

        if observation_model.__class__.__name__ not in nsm.__all__:
            raise TypeError(
                "The provided `observation_model` should be one of the implemented models in "
                "`neurostatslib.observation_models`. "
                f"Available options are: {nsm.__all__}."
            )

        self.observation_model = observation_model
        self.solver = solver

        # initialize to None fit output
        self.baseline_link_fr_ = None
        self.basis_coeff_ = None
        self.solver_state = None

    def _check_is_fit(self):
        """Ensure the instance has been fitted."""
        if (self.basis_coeff_ is None) or (self.baseline_link_fr_ is None):
            raise NotFittedError(
                "This GLM instance is not fitted yet. Call 'fit' with appropriate arguments."
            )

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
        :
            The predicted rates. Shape (n_time_bins, n_neurons).
        """
        Ws, bs = params
        return self.observation_model.inverse_link_function(
            jnp.einsum("ik,tik->ti", Ws, X) + bs[None, :]
        )

    def predict(self, X: Union[NDArray, jnp.ndarray]) -> jnp.ndarray:
        """Predict rates based on fit parameters.

        Parameters
        ----------
        X :
            The exogenous variables. Shape (n_time_bins, n_neurons, n_features).

        Returns
        -------
        :
            The predicted rates with shape (n_neurons, n_time_bins).

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
        - [score](./#neurostatslib.glm.GLM.score)
            Score predicted rates against target spike counts.
        - [simulate (feed-forward only)](../glm/#neurostatslib.glm.GLM.simulate)
            Simulate neural activity in response to a feed-forward input .
        - [simulate (feed-forward + coupling)](../glm/#neurostatslib.glm.GLMRecurrent.simulate)
            Simulate neural activity in response to a feed-forward input
            using the GLM as a recurrent network.
        """
        # check that the model is fitted
        self._check_is_fit()
        # extract model params
        Ws = self.basis_coeff_
        bs = self.baseline_link_fr_

        (X,) = convert_to_jnp_ndarray(X)

        # check input dimensionality
        self._check_input_dimensionality(X=X)
        # check consistency between X and params
        self._check_input_and_params_consistency((Ws, bs), X=X)
        return self._predict((Ws, bs), X)

    def _score(
        self,
        params: Tuple[jnp.ndarray, jnp.ndarray],
        X: jnp.ndarray,
        y: jnp.ndarray,
    ) -> jnp.ndarray:
        r"""Score the predicted rates against target neural activity.

        This computes the negative log-likelihood up to a constant term.

        Note that you can end up with infinities in here if there are zeros in
        ``predicted_rates``. We raise a warning in that case.

        Parameters
        ----------
        params :
            Values for the spike basis coefficients and bias terms. Shape ((n_neurons, n_features), (n_neurons,)).
        X :
            The exogenous variables. Shape (n_time_bins, n_neurons, n_features).
        y :
            The target activity to compare against. Shape (n_time_bins, n_neurons).

        Returns
        -------
        :
            The model negative log-likehood. Shape (1,).

        """
        predicted_rate = self._predict(params, X)
        return self.observation_model.negative_log_likelihood(predicted_rate, y)

    def score(
        self,
        X: Union[NDArray, jnp.ndarray],
        y: Union[NDArray, jnp.ndarray],
        score_type: Literal["log-likelihood", "pseudo-r2"] = "pseudo-r2",
    ) -> jnp.ndarray:
        r"""Score the predicted firing rates (based on fit) to the target spike counts.

        This computes the GLM pseudo-$R^2$ or the mean log-likelihood, thus the higher the
        number the better.


        Parameters
        ----------
        X :
            The exogenous variables. Shape (n_time_bins, n_neurons, n_features)
        y :
            Neural activity arranged in a matrix. n_neurons must be the same as
            during the fitting of this GLM instance. Shape (n_time_bins, n_neurons).
        score_type :
            Type of scoring: either log-likelihood or pseudo-r2.

        Returns
        -------
        score :
            The log-likelihood or the pseudo-$R^2$ of the current model.

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
        predicted rate for neuron $n$ at time-point $t$, and $\bar{\lambda}$ is the mean firing rate,
        see references[$^1$](#--references).

        Refer to the `nsl.observation_models.Observations` concrete subclasses for the specific likelihood equations.


        References
        ----------
        1. Cohen, Jacob, et al. Applied multiple regression/correlation analysis for the behavioral sciences.
        Routledge, 2013.

        """
        if score_type not in ["log-likelihood", "pseudo-r2"]:
            raise NotImplementedError(
                f"Scoring method {score_type} not implemented! "
                f"`score_type` must be either 'log-likelihood', or 'pseudo-r2'."
            )
        # ignore the last time point from predict, because that corresponds to
        # the next time step, which we have no observed data for
        self._check_is_fit()
        Ws = self.basis_coeff_
        bs = self.baseline_link_fr_

        X, y = convert_to_jnp_ndarray(X, y)

        self._check_input_dimensionality(X, y)
        self._check_input_n_timepoints(X, y)
        self._check_input_and_params_consistency((Ws, bs), X=X, y=y)
        if score_type == "log-likelihood":
            norm_constant = jax.scipy.special.gammaln(y + 1).mean()
            score = -self._score((Ws, bs), X, y) - norm_constant
        else:
            score = self.observation_model.pseudo_r2(self._predict((Ws, bs), X), y)

        return score

    def fit(
        self,
        X: Union[NDArray, jnp.ndarray],
        y: Union[NDArray, jnp.ndarray],
        init_params: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    ):
        """Fit GLM to neural activity.

        Following scikit-learn API, the solutions are stored as attributes
        ``basis_coeff_`` and ``baseline_link_fr``.

        Parameters
        ----------
        X :
            Predictors, shape (n_time_bins, n_neurons, n_features)
        y :
            Neural activity arranged in a matrix, shape (n_time_bins, n_neurons).
        init_params :
            Initial values for the activity basis coefficients and bias terms. If
            None, we initialize with zeros. shape.  ((n_neurons, n_features), (n_neurons,))

        Raises
        ------
        ValueError
            - If `init_params` is not of length two.
            - If dimensionality of `init_params` are not correct.
            - If the number of neurons in the model parameters and in the inputs do not match.
            - If `X` is not three-dimensional.
            - If `y` is not two-dimensional.
            - If solver returns at least one NaN parameter, which means it found
              an invalid solution. Try tuning optimization hyperparameters.
        TypeError
            - If `init_params` are not array-like
            - If `init_params[i]` cannot be converted to jnp.ndarray for all i
        """
        # convert to jnp.ndarray & perform checks
        X, y, init_params = self._preprocess_fit(X, y, init_params)

        # Run optimization
        runner = self.solver.instantiate_solver(self._score)
        params, state = runner(init_params, X, y)
        # if any observation model other than Poisson are used
        # one should set the scale parameter too.
        # self.observation_model.set_scale(params)

        if jnp.isnan(params[0]).any() or jnp.isnan(params[1]).any():
            raise ValueError(
                "Solver returned at least one NaN parameter, so solution is invalid!"
                " Try tuning optimization hyperparameters."
            )

        # Store parameters
        self.basis_coeff_: jnp.ndarray = params[0]
        self.baseline_link_fr_: jnp.ndarray = params[1]
        # note that this will include an error value, which is not the same as
        # the output of loss. I believe it's the output of
        # solver.l2_optimality_error
        self.solver_state = state

    def simulate(
        self,
        random_key: jax.random.PRNGKeyArray,
        feedforward_input: Union[NDArray, jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Simulate neural activity in response to a feed-forward input.

        Parameters
        ----------
        random_key :
            PRNGKey for seeding the simulation.
        feedforward_input :
            External input matrix to the model, representing factors like convolved currents,
            light intensities, etc. When not provided, the simulation is done with coupling-only.
            Expected shape: (n_timesteps, n_neurons, n_basis_input).

        Returns
        -------
        simulated_activity :
            Simulated activity (spike counts for PoissonGLMs) for each neuron over time.
            Shape: (n_neurons, n_timesteps).
        firing_rates :
            Simulated rates for each neuron over time. Shape, (n_neurons, n_timesteps).

        Raises
        ------
        NotFittedError
            If the model hasn't been fitted prior to calling this method.
        ValueError
            - If the instance has not been previously fitted.
            - If there's an inconsistency between the number of neurons in model parameters.
            - If the number of neurons in input arguments doesn't match with model parameters.


        See Also
        --------
        [predict](./#neurostatslib.glm.GLM.predict) :
        Method to predict rates based on the model's parameters.
        """
        # check if the model is fit
        self._check_is_fit()
        Ws, bs = self.basis_coeff_, self.baseline_link_fr_
        (feedforward_input,) = self._preprocess_simulate(
            feedforward_input, params_feedforward=(Ws, bs)
        )
        predicted_rate = self._predict((Ws, bs), feedforward_input)
        return (
            self.observation_model.sample_generator(
                key=random_key, predicted_rate=predicted_rate
            ),
            predicted_rate,
        )


class GLMRecurrent(GLM):
    """
    A Generalized Linear Model (GLM) with recurrent dynamics.

    This class extends the basic GLM to capture recurrent dynamics between neurons,
    making it more suitable for simulating the activity of interconnected neural populations.
    The recurrent GLM combines both feedforward inputs (like sensory stimuli) and past
    neural activity to simulate or predict future neural activity.

    Parameters
    ----------
    observation_model :
        The observation model to use for the GLM. This defines how neural activity is generated
        based on the underlying firing rate. Common choices include Poisson and Gaussian models.
    solver :
        The optimization solver to use for fitting the GLM parameters.

    See Also
    --------
    [GLM](./#neurostatslib.glm.GLM) : Base class for the generalized linear model.

    Notes
    -----
    - The recurrent GLM assumes that neural activity can be influenced by both feedforward
    inputs and the past activity of the same and other neurons. This makes it particularly
    powerful for capturing the dynamics of neural networks where neurons are interconnected.

    - The attributes of `GLMRecurrent` are inherited from the parent `GLM` class, and might include
    coefficients, fitted status, and other model-related attributes.

    Examples
    --------
    >>> # Initialize the recurrent GLM with default parameters
    >>> model = GLMRecurrent()
    >>> # ... your code for training and simulating using the model ...

    """

    def __init__(
        self,
        observation_model: nsm.Observations = nsm.PoissonObservations(),
        solver: slv.Solver = slv.RidgeSolver(),
    ):
        super().__init__(observation_model=observation_model, solver=solver)

    def simulate_recurrent(
        self,
        random_key: jax.random.PRNGKeyArray,
        feedforward_input: Union[NDArray, jnp.ndarray],
        coupling_basis_matrix: Union[NDArray, jnp.ndarray],
        init_y: Union[NDArray, jnp.ndarray],
    ):
        """
        Simulate neural activity using the GLM as a recurrent network.

        This function projects neural activity into the future, employing the fitted
        parameters of the GLM. It is capable of simulating activity based on a combination
        of historical activity and external feedforward inputs like convolved currents, light
        intensities, etc.

        Parameters
        ----------
        random_key :
            PRNGKey for seeding the simulation.
        feedforward_input :
            External input matrix to the model, representing factors like convolved currents,
            light intensities, etc. When not provided, the simulation is done with coupling-only.
            Expected shape: (n_timesteps, n_neurons, n_basis_input).
        init_y :
            Initial observation (spike counts for PoissonGLM) matrix that kickstarts the simulation.
            Expected shape: (window_size, n_neurons).
        coupling_basis_matrix :
            Basis matrix for coupling, representing between-neuron couplings
            and auto-correlations. Expected shape: (window_size, n_basis_coupling).

        Returns
        -------
        simulated_activity :
            Simulated activity (spike counts for PoissonGLMs) for each neuron over time.
            Shape, (n_neurons, n_timesteps).
        firing_rates :
            Simulated rates for each neuron over time. Shape, (n_neurons, n_timesteps).

        Raises
        ------
        NotFittedError
            If the model hasn't been fitted prior to calling this method.
        ValueError
            - If the instance has not been previously fitted.
            - If there's an inconsistency between the number of neurons in model parameters.
            - If the number of neurons in input arguments doesn't match with model parameters.


        See Also
        --------
        [predict](./#neurostatslib.glm.GLM.predict) :
        Method to predict rates based on the model's parameters.

        Notes
        -----
        The model coefficients (`self.basis_coeff_`) are structured such that the first set of coefficients
        (of size `n_basis_coupling * n_neurons`) are interpreted as the weights for the recurrent couplings.
        The remaining coefficients correspond to the weights for the feed-forward input.


        The sum of `n_basis_input` and `n_basis_coupling * n_neurons` should equal `self.basis_coeff_.shape[1]`
        to ensure consistency in the model's input feature dimensionality.
        """
        # check if the model is fit
        self._check_is_fit()

        # convert to jnp.ndarray
        (coupling_basis_matrix,) = convert_to_jnp_ndarray(coupling_basis_matrix)

        n_basis_coupling = coupling_basis_matrix.shape[1]
        n_neurons = self.baseline_link_fr_.shape[0]

        if init_y is None:
            init_y = jnp.zeros((coupling_basis_matrix.shape[0], n_neurons))

        Wf = self.basis_coeff_[:, n_basis_coupling * n_neurons :]
        Wr = self.basis_coeff_[:, : n_basis_coupling * n_neurons]
        bs = self.baseline_link_fr_

        feedforward_input, init_y = self._preprocess_simulate(
            feedforward_input,
            params_feedforward=(Wf, bs),
            init_y=init_y,
            params_recurrent=(Wr, bs),
        )

        self._check_input_and_params_consistency(
            (Wr, bs),
            y=init_y,
        )

        if init_y.shape[0] != coupling_basis_matrix.shape[0]:
            raise ValueError(
                "`init_y` and `coupling_basis_matrix`"
                " should have the same window size! "
                f"`init_y` window size: {init_y.shape[1]}, "
                f"`coupling_basis_matrix` window size: {coupling_basis_matrix.shape[1]}"
            )

        subkeys = jax.random.split(random_key, num=feedforward_input.shape[0])
        # (n_samples, n_neurons)
        feed_forward_contrib = jnp.einsum("ik,tik->ti", Wf, feedforward_input)

        def scan_fn(
            data: Tuple[jnp.ndarray, int], key: jax.random.PRNGKeyArray
        ) -> Tuple[Tuple[jnp.ndarray, int], Tuple[jnp.ndarray, jnp.ndarray]]:
            """Scan over time steps and simulate activity and rates.

            This function simulates the neural activity and firing rates for each time step
            based on the previous activity, feedforward input, and model coefficients.
            """
            activity, chunk = data

            # Convolve the neural activity with the coupling basis matrix
            conv_act = convolve_1d_trials(coupling_basis_matrix, activity[None])[0]

            # Extract the slice of the feedforward input for the current time step
            input_slice = jax.lax.dynamic_slice(
                feed_forward_contrib,
                (chunk, 0),
                (1, feed_forward_contrib.shape[1]),
            )

            # Reshape the convolved activity and concatenate with the input slice to form the model input
            conv_act = jnp.tile(
                conv_act.reshape(conv_act.shape[0], -1), conv_act.shape[1]
            ).reshape(conv_act.shape[0], conv_act.shape[1], -1)

            # Predict the firing rate using the model coefficients
            # Doesn't use predict because the non-linearity needs
            # to be applied after we add the feed forward input
            firing_rate = self.observation_model.inverse_link_function(
                jnp.einsum("ik,tik->ti", Wr, conv_act) + input_slice + bs[None, :]
            )

            # Simulate activity based on the predicted firing rate
            new_act = self.observation_model.sample_generator(key, firing_rate)

            # Prepare the spikes for the next iteration (keeping the most recent spikes)
            concat_act = jnp.row_stack((activity[1:], new_act)), chunk + 1
            return concat_act, (new_act, firing_rate)

        _, outputs = jax.lax.scan(scan_fn, (init_y, 0), subkeys)
        simulated_activity, firing_rates = outputs
        return jnp.squeeze(simulated_activity, axis=1), jnp.squeeze(
            firing_rates, axis=1
        )
