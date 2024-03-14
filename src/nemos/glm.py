"""GLM core module."""

from typing import Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike, NDArray

from . import observation_models as obs
from . import regularizer as reg
from . import tree_utils, utils
from .base_class import DESIGN_INPUT_TYPE, BaseRegressor
from .exceptions import NotFittedError
from .pytrees import FeaturePytree
from .type_casting import support_pynapple


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
    regularizer :
        Regularization to use for model optimization. Defines the regularization scheme, the optimization algorithm,
        and related parameters.
        Default is Ridge regression with gradient descent.

    Attributes
    ----------
    intercept_ :
        Model baseline linked firing rate parameters, e.g. if the link is the logarithm, the baseline
        firing rate will be `jnp.exp(model.intercept_)`.
    coef_ :
        Basis coefficients for the model.
    solver_state :
        State of the solver after fitting. May include details like optimization error.

    Raises
    ------
    TypeError
        If provided `regularizer` or `observation_model` are not valid.
    """

    def __init__(
        self,
        observation_model: obs.Observations = obs.PoissonObservations(),
        regularizer: reg.Regularizer = reg.Ridge("GradientDescent"),
    ):
        super().__init__()

        self.observation_model = observation_model
        self.regularizer = regularizer

        # initialize to None fit output
        self.intercept_ = None
        self.coef_ = None
        self.solver_state = None

    @property
    def regularizer(self):
        """Getter for the regularizer attribute."""
        return self._regularizer

    @regularizer.setter
    def regularizer(self, regularizer: reg.Regularizer):
        """Setter for the regularizer attribute."""
        if not hasattr(regularizer, "instantiate_solver"):
            raise AttributeError(
                "The provided `solver` doesn't implement the `instantiate_solver` method."
            )
        # test solver instantiation on the GLM loss
        try:
            regularizer.instantiate_solver(self._predict_and_compute_loss)
        except Exception:
            raise TypeError(
                "The provided `solver` cannot be instantiated on "
                "the GLM log-likelihood."
            )
        self._regularizer = regularizer

    @property
    def observation_model(self):
        """Getter for the observation_model attribute."""
        return self._observation_model

    @observation_model.setter
    def observation_model(self, observation: obs.Observations):
        # check that the model has the required attributes
        # and that the attribute can be called
        obs.check_observation_model(observation)
        self._observation_model = observation

    def _check_is_fit(self):
        """Ensure the instance has been fitted."""
        if (self.coef_ is None) or (self.intercept_ is None):
            raise NotFittedError(
                "This GLM instance is not fitted yet. Call 'fit' with appropriate arguments."
            )

    def _predict(
        self, params: Tuple[DESIGN_INPUT_TYPE, jnp.ndarray], X: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Predicts firing rates based on given parameters and design matrix.

        This function computes the predicted firing rates using the provided parameters
        and model design matrix `X`. It is a streamlined version used internally within
        optimization routines, where it serves as the loss function. Unlike the `GLM.predict`
        method, it does not perform any input validation, assuming that the inputs are pre-validated.


        Parameters
        ----------
        params :
            Tuple containing the spike basis coefficients and bias terms.
        X :
            Predictors.

        Returns
        -------
        :
            The predicted rates. Shape (n_time_bins, n_neurons).
        """
        Ws, bs = params
        return self._observation_model.inverse_link_function(
            # First, multiply each feature by its corresponding coefficient,
            # then sum across all features and add the intercept, before
            # passing to the inverse link function
            tree_utils.pytree_map_and_reduce(
                lambda w, x: jnp.einsum("ik,tik->ti", w, x), sum, Ws, X
            )
            + bs[None, :]
        )

    @support_pynapple(conv_type="jax")
    def predict(self, X: DESIGN_INPUT_TYPE) -> jnp.ndarray:
        """Predict rates based on fit parameters.

        Parameters
        ----------
        X :
            Predictors, array of shape (n_time_bins, n_neurons, n_features) or pytree of same.

        Returns
        -------
        :
            The predicted rates with shape (n_time_bins, n_neurons).

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
        - [score](./#nemos.glm.GLM.score)
            Score predicted rates against target spike counts.
        - [simulate (feed-forward only)](../glm/#nemos.glm.GLM.simulate)
            Simulate neural activity in response to a feed-forward input .
        - [simulate_recurrent (feed-forward + coupling)](../glm/#nemos.glm.GLMRecurrent.simulate_recurrent)
            Simulate neural activity in response to a feed-forward input
            using the GLM as a recurrent network.
        """
        # check that the model is fitted
        self._check_is_fit()
        # extract model params
        Ws = self.coef_
        bs = self.intercept_

        X = jax.tree_map(lambda x: jnp.asarray(x, dtype=float), X)

        # check input dimensionality
        self._check_input_dimensionality(X=X)
        # check consistency between X and params
        self._check_input_and_params_consistency((Ws, bs), X=X)
        return self._predict((Ws, bs), X)

    def _predict_and_compute_loss(
        self,
        params: Tuple[DESIGN_INPUT_TYPE, jnp.ndarray],
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
    ) -> jnp.ndarray:
        r"""Predict the rate and compute the negative log-likelihood against neural activity.

        This method computes the negative log-likelihood up to a constant term. Unlike `score`,
        it does not conduct parameter checks prior to evaluation. Passed directly to the solver,
        it serves to establish the optimization objective for learning the model parameters.

        Parameters
        ----------
        params :
            2-tuple containing the spike basis coefficients and bias terms.
        X :
            Predictors.
        y :
            Target neural activity.

        Returns
        -------
        :
            The model negative log-likehood. Shape (1,).

        """
        predicted_rate = self._predict(params, X)
        return self._observation_model.negative_log_likelihood(predicted_rate, y)

    def score(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: ArrayLike,
        score_type: Literal[
            "log-likelihood", "pseudo-r2-McFadden", "pseudo-r2-Cohen"
        ] = "pseudo-r2-McFadden",
    ) -> jnp.ndarray:
        r"""Evaluate the goodness-of-fit of the model to the observed neural data.

        This method computes the goodness-of-fit score, which can either be the mean
        log-likelihood or of two versions of the pseudo-R^2.
        The scoring process includes validation of input compatibility with the model's
        parameters, ensuring that the model has been previously fitted and the input data
        are appropriate for scoring. A higher score indicates a better fit of the model
        to the observed data.


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
            self.intercept_.shape[0]``).

        Notes
        -----
        The log-likelihood is not on a standard scale, its value is influenced by many factors,
        among which the number of model parameters. The log-likelihood can assume both positive
        and negative values.

        The Pseudo-$ R^2 $ is not equivalent to the $ R^2 $ value in linear regression. While both
        provide a measure of model fit, and assume values in the [0,1] range, the methods and
        interpretations can differ. The Pseudo-$ R^2 $ is particularly useful for generalized linear
        models when the interpretation of the $ R^2 $ as explained variance does not apply
        (i.e., when the observations are not Gaussian distributed).

        Why does the traditional $R^2$ is usually a poor measure of performance in GLMs?

        1.  In the context of GLMs the variance and the mean of the observations are related.
        Ignoring the relation between them can result in underestimating the model
        performance; for instance, when we model a Poisson variable with large mean we expect an
        equally large variance. In this scenario, even if our model perfectly captures the mean,
        the high-variance  will result in large residuals and low $R^2$.
        Additionally, when the mean of the observations varies, the variance will vary too. This
        violates the "homoschedasticity" assumption, necessary  for interpreting the $R^2$ as
        variance explained.
        2. The $R^2$ capture the variance explained when the relationship between the observations and
        the predictors is linear. In GLMs, the link function sets a non-linear mapping between the predictors
        and the mean of the observations, compromising the interpretation of the $R^2$.

        Note that it is possible to re-normalized the residuals by a mean-dependent quantity proportional
        to the model standard deviation (i.e. Pearson residuals). This "rescaled" residual distribution however
        deviates substantially from normality for counting data with low mean (common for spike counts).
        Therefore, even the Pearson residuals performs poorly as a measure of fit quality, especially
        for GLM modeling counting data.

        Refer to the `nmo.observation_models.Observations` concrete subclasses for the likelihood and
        pseudo-$R^2$ equations.

        """
        self._check_is_fit()
        Ws = self.coef_
        bs = self.intercept_

        X = jax.tree_map(lambda x: jnp.asarray(x, dtype=float), X)
        y = jnp.asarray(y, dtype=float)

        self._check_input_dimensionality(X, y)
        self._check_input_n_timepoints(X, y)
        self._check_input_and_params_consistency((Ws, bs), X=X, y=y)

        # get valid entries
        is_valid = tree_utils.get_valid_multitree(X, y)

        # filter for valid
        X = jax.tree_map(lambda x: x[is_valid], X)
        y = jax.tree_map(lambda x: x[is_valid], y)

        if score_type == "log-likelihood":
            norm_constant = jax.scipy.special.gammaln(y + 1).mean()
            score = -self._predict_and_compute_loss((Ws, bs), X, y) - norm_constant
        elif score_type.startswith("pseudo-r2"):
            score = self._observation_model.pseudo_r2(
                self._predict((Ws, bs), X), y, score_type=score_type
            )
        else:
            raise NotImplementedError(
                f"Scoring method {score_type} not implemented! "
                "`score_type` must be either 'log-likelihood', 'pseudo-r2-McFadden', "
                "or 'pseudo-r2-Cohen'."
            )
        return score

    def fit(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: ArrayLike,
        init_params: Optional[
            Tuple[Union[DESIGN_INPUT_TYPE, ArrayLike], ArrayLike]
        ] = None,
    ):
        """Fit GLM to neural activity.

        Fit and store the model parameters as attributes
        ``coef_`` and ``coef_``.

        Parameters
        ----------
        X :
            Predictors, array of shape (n_time_bins, n_neurons, n_features) or pytree of same.
        y :
            Target neural activity arranged in a matrix, shape (n_time_bins, n_neurons).
        init_params :
            2-tuple of initial parameter values: (coefficients, intercepts). If
            None, we initialize coefficients with zeros, intercepts with the
            log of the mean neural activity. coefficients is an array of shape
            (n_neurons, n_features) or pytree of same, intercepts is an array
            of shape (n_neurons,)

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
        runner = self.regularizer.instantiate_solver(self._predict_and_compute_loss)
        params, state = runner(init_params, X, y)

        # estimate the GLM scale
        self.observation_model.estimate_scale(self._predict(params, X))

        if (
            tree_utils.pytree_map_and_reduce(
                jnp.any, any, jax.tree_map(jnp.isnan, params[0])
            )
            or jnp.isnan(params[1]).any()
        ):
            raise ValueError(
                "Solver returned at least one NaN parameter, so solution is invalid!"
                " Try tuning optimization hyperparameters."
            )

        # Store parameters
        self.coef_: DESIGN_INPUT_TYPE = params[0]
        self.intercept_: jnp.ndarray = params[1]
        # note that this will include an error value, which is not the same as
        # the output of loss. I believe it's the output of
        # solver.l2_optimality_error
        self.solver_state = state

    def simulate(
        self,
        random_key: jax.Array,
        feedforward_input: DESIGN_INPUT_TYPE,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Simulate neural activity in response to a feed-forward input.

        Parameters
        ----------
        random_key :
            jax.random.key for seeding the simulation.
        feedforward_input :
            External input matrix to the model, representing factors like convolved currents,
            light intensities, etc. When not provided, the simulation is done with coupling-only.
            Array of shape (n_time_bins, n_neurons, n_basis_input) or pytree of same.

        Returns
        -------
        simulated_activity :
            Simulated activity (spike counts for PoissonGLMs) for each neuron over time.
            Shape: (n_time_bins, n_neurons).
        firing_rates :
            Simulated rates for each neuron over time. Shape, (n_neurons, n_time_bins).

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
        [predict](./#nemos.glm.GLM.predict) :
        Method to predict rates based on the model's parameters.
        """
        # check if the model is fit
        self._check_is_fit()
        Ws, bs = self.coef_, self.intercept_
        (feedforward_input,) = self._preprocess_simulate(
            feedforward_input, params_feedforward=(Ws, bs)
        )
        predicted_rate = self._predict((Ws, bs), feedforward_input)
        return (
            self._observation_model.sample_generator(
                key=random_key, predicted_rate=predicted_rate
            ),
            predicted_rate,
        )


class GLMRecurrent(GLM):
    """
    A Generalized Linear Model (GLM) with recurrent dynamics.

    This class extends the basic GLM to capture recurrent dynamics between neurons and
    self-connectivity, making it more suitable for simulating the activity of interconnected
    neural populations. The recurrent GLM combines both feedforward inputs (like sensory
    stimuli) and past neural activity to simulate or predict future neural activity.

    Parameters
    ----------
    observation_model :
        The observation model to use for the GLM. This defines how neural activity is generated
        based on the underlying firing rate. Common choices include Poisson and Gaussian models.
    regularizer :
        The regularization scheme to use for fitting the GLM parameters.

    See Also
    --------
    [GLM](./#nemos.glm.GLM) : Base class for the generalized linear model.

    Notes
    -----
    - The recurrent GLM assumes that neural activity can be influenced by both feedforward
    inputs and the past activity of the same and other neurons. This makes it particularly
    powerful for capturing the dynamics of neural networks where neurons are interconnected.

    - The attributes of `GLMRecurrent` are inherited from the parent `GLM` class, and include
    coefficients, fitted status, and other model-related attributes.
    """

    def __init__(
        self,
        observation_model: obs.Observations = obs.PoissonObservations(),
        regularizer: reg.Regularizer = reg.Ridge(),
    ):
        super().__init__(observation_model=observation_model, regularizer=regularizer)

    def simulate_recurrent(
        self,
        random_key: jax.Array,
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
            jax.random.key for seeding the simulation.
        feedforward_input :
            External input matrix to the model, representing factors like convolved currents,
            light intensities, etc. When not provided, the simulation is done with coupling-only.
            Expected shape: (n_time_bins, n_neurons, n_basis_input).
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
            Shape, (n_time_bins, n_neurons).
        firing_rates :
            Simulated rates for each neuron over time. Shape, (n_time_bins, n_neurons,).

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
        [predict](./#nemos.glm.GLM.predict) :
        Method to predict rates based on the model's parameters.

        Notes
        -----
        The model coefficients (`self.coef_`) are structured such that the first set of coefficients
        (of size `n_basis_coupling * n_neurons`) are interpreted as the weights for the recurrent couplings.
        The remaining coefficients correspond to the weights for the feed-forward input.


        The sum of `n_basis_input` and `n_basis_coupling * n_neurons` should equal `self.coef_.shape[1]`
        to ensure consistency in the model's input feature dimensionality.
        """
        if isinstance(feedforward_input, FeaturePytree):
            raise ValueError(
                "simulate_recurrent works only with arrays. "
                "FeaturePytree provided instead!"
            )
        # check if the model is fit
        self._check_is_fit()

        # convert to jnp.ndarray
        coupling_basis_matrix = jnp.asarray(coupling_basis_matrix, dtype=float)

        n_basis_coupling = coupling_basis_matrix.shape[1]
        n_neurons = self.intercept_.shape[0]

        w_feedforward = self.coef_[:, n_basis_coupling * n_neurons :]
        w_recurrent = self.coef_[:, : n_basis_coupling * n_neurons]
        bs = self.intercept_

        feedforward_input, init_y = self._preprocess_simulate(
            feedforward_input,
            params_feedforward=(w_feedforward, bs),
            init_y=init_y,
            params_recurrent=(w_recurrent, bs),
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
        feed_forward_contrib = jnp.einsum(
            "ik,tik->ti", w_feedforward, feedforward_input
        )

        def scan_fn(
            data: Tuple[jnp.ndarray, int], key: jax.Array
        ) -> Tuple[Tuple[jnp.ndarray, int], Tuple[jnp.ndarray, jnp.ndarray]]:
            """Scan over time steps and simulate activity and rates.

            This function simulates the neural activity and firing rates for each time step
            based on the previous activity, feedforward input, and model coefficients.
            """
            activity, t_sample = data

            # Convolve the neural activity with the coupling basis matrix
            # Output of shape (1, n_neuron, n_basis_coupling)
            # 1. The first dimension is time, and 1 is by construction since we are simulating 1
            #    sample
            # 2. Flatten to shape (n_neuron * n_basis_coupling, )
            conv_act = utils.convolve_1d_trials(coupling_basis_matrix, activity[None])[
                0
            ].flatten()

            # Extract the slice of the feedforward input for the current time step
            input_slice = jax.lax.dynamic_slice(
                feed_forward_contrib,
                (t_sample, 0),
                (1, feed_forward_contrib.shape[1]),
            ).squeeze(axis=0)

            # Predict the firing rate using the model coefficients
            # Doesn't use predict because the non-linearity needs
            # to be applied after we add the feed forward input
            firing_rate = self._observation_model.inverse_link_function(
                w_recurrent.dot(conv_act) + input_slice + bs
            )

            # Simulate activity based on the predicted firing rate
            new_act = self._observation_model.sample_generator(key, firing_rate)

            # Shift of one sample the spike count window
            # for the next iteration (i.e. remove the first counts, and
            # stack the newly generated sample)
            # Increase the t_sample by one
            carry = jnp.vstack((activity[1:], new_act)), t_sample + 1
            return carry, (new_act, firing_rate)

        _, outputs = jax.lax.scan(scan_fn, (init_y, 0), subkeys)
        simulated_activity, firing_rates = outputs
        return simulated_activity, firing_rates
