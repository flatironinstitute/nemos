"""GLM core module."""

# required to get ArrayLike to render correctly
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable, Literal, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike
from sklearn.utils import InputTags, TargetTags

from .. import observation_models as obs
from .. import tree_utils, validation
from .._observation_model_builder import instantiate_observation_model
from ..base_regressor import BaseRegressor, strip_metadata
from ..exceptions import NotFittedError
from ..inverse_link_function_utils import resolve_inverse_link_function
from ..pytrees import FeaturePytree
from ..regularizer import ElasticNet, GroupLasso, Lasso, Regularizer, Ridge
from ..solvers._compute_defaults import glm_compute_optimal_stepsize_configs
from ..type_casting import cast_to_jax, support_pynapple
from ..typing import DESIGN_INPUT_TYPE, RegularizerStrength, SolverState, StepResult
from ..utils import format_repr
from .initialize_parameters import initialize_intercept_matching_mean_rate
from .params import GLMParams, GLMUserParams
from .validation import GLMValidator, PopulationGLMValidator


class GLM(BaseRegressor[GLMParams]):
    r"""Generalized Linear Model (GLM) for neural activity data.

    This GLM implementation allows users to model neural activity based on a combination of exogenous inputs
    (like convolved currents or light intensities) and a choice of observation model. It is suitable for scenarios where
    the relationship between predictors and the response variable might be non-linear, and the residuals
    don't follow a normal distribution.

    Below is a table of the default inverse link function for the availabe observation model.

    +---------------------+---------------------------------+
    | Observation Model   | Default Inverse Link Function   |
    +=====================+=================================+
    | Poisson             | :math:`e^x`                     |
    +---------------------+---------------------------------+
    | Gamma               | :math:`1/x`                     |
    +---------------------+---------------------------------+
    | Bernoulli            | :math:`1 / (1 + e^{-x})`       |
    +---------------------+---------------------------------+
    | NegativeBinomial    | :math:`e^x`                     |
    +---------------------+---------------------------------+

    Below is a table listing the default and available solvers for each regularizer.

    +---------------+------------------+-------------------------------------------------------------+
    | Regularizer   | Default Solver   | Available Solvers                                           |
    +===============+==================+=============================================================+
    | UnRegularized | GradientDescent  | GradientDescent, BFGS, LBFGS, NonlinearCG, ProximalGradient |
    +---------------+------------------+-------------------------------------------------------------+
    | Ridge         | GradientDescent  | GradientDescent, BFGS, LBFGS, NonlinearCG, ProximalGradient |
    +---------------+------------------+-------------------------------------------------------------+
    | Lasso         | ProximalGradient | ProximalGradient                                            |
    +---------------+------------------+-------------------------------------------------------------+
    | GroupLasso    | ProximalGradient | ProximalGradient                                            |
    +---------------+------------------+-------------------------------------------------------------+

    **Fitting Large Models**

    For very large models, you may consider using the Stochastic Variance Reduced Gradient
    :class:`nemos.solvers._svrg.SVRG` or its proximal variant
    :class:`nemos.solvers._svrg.ProxSVRG` solver,
    which take advantage of batched computation. You can change the solver by passing
    ``"SVRG"`` as ``solver_name`` at model initialization.

    The performance of the SVRG solver depends critically on the choice of ``batch_size`` and ``stepsize``
    hyperparameters. These parameters control the size of the mini-batches used for gradient computations
    and the step size for each iteration, respectively. Improper selection of these parameters can lead to slow
    convergence or even divergence of the optimization process.

    To assist with this, for certain GLM configurations, we provide ``batch_size`` and ``stepsize`` default
    values that are theoretically guaranteed to ensure fast convergence.

    Below is a list of the configurations for which we can provide guaranteed default hyperparameters:

    +---------------------------------------+-----------+-------------+
    | GLM / PopulationGLM Configuration     | Stepsize  | Batch Size  |
    +=======================================+===========+=============+
    | Poisson + soft-plus + UnRegularized   | ✅        | ❌          |
    +---------------------------------------+-----------+-------------+
    | Poisson + soft-plus + Ridge           | ✅        | ✅          |
    +---------------------------------------+-----------+-------------+
    | Poisson + soft-plus + Lasso           | ✅        | ❌          |
    +---------------------------------------+-----------+-------------+
    | Poisson + soft-plus + GroupLasso      | ✅        | ❌          |
    +---------------------------------------+-----------+-------------+

    Parameters
    ----------
    observation_model :
        Observation model to use. The model describes the distribution of the neural activity.
        Default is the Poisson model.
    inverse_link_function :
        A function that maps the linear combination of predictors into a firing rate. The default depends
        on the observation model, see the table above.
    regularizer :
        Regularization to use for model optimization. Defines the regularization scheme
        and related parameters.
        Default is UnRegularized regression.
    regularizer_strength :
        Float that is default None. Sets the regularizer strength. If a user does not pass a value, and it is needed for
        regularization, a warning will be raised and the strength will default to 1.0.
    solver_name :
        Solver to use for model optimization. Defines the optimization scheme and related parameters.
        The solver must be an appropriate match for the chosen regularizer.
        Default is ``None``. If no solver specified, one will be chosen based on the regularizer.
        Please see table above for regularizer/optimizer pairings.
    solver_kwargs :
        Optional dictionary for keyword arguments that are passed to the solver when instantiated.
        E.g. stepsize, tol, acceleration, etc.
         For details on each solver's kwargs, see `get_accepted_arguments` and `get_solver_documentation`.

    Attributes
    ----------
    intercept_ :
        Model baseline linked firing rate parameters, e.g. if the link is the logarithm, the baseline
        firing rate will be ``jnp.exp(model.intercept_)``.
    coef_ :
        Basis coefficients for the model.
    solver_state_ :
        State of the solver after fitting. May include details like optimization error.
    scale_:
        Scale parameter for the model. The scale parameter is the constant :math:`\Phi`, for which
        :math:`\text{Var} \left( y \right) = \Phi V(\mu)`. This parameter, together with the estimate
        of the mean :math:`\mu` fully specifies the distribution of the activity :math:`y`.
    dof_resid_:
        Degrees of freedom for the residuals.


    Raises
    ------
    TypeError
        If provided ``regularizer`` or ``observation_model`` are not valid.

    Examples
    --------
    >>> import nemos as nmo
    >>> # define single neuron GLM model
    >>> model = nmo.glm.GLM()
    >>> model
    GLM(
        observation_model=PoissonObservations(),
        inverse_link_function=exp,
        regularizer=UnRegularized(),
        solver_name='GradientDescent'
    )
    >>> print("Regularizer type: ", type(model.regularizer))
    Regularizer type:  <class 'nemos.regularizer.UnRegularized'>
    >>> print("Observation model: ", type(model.observation_model))
    Observation model:  <class 'nemos.observation_models.PoissonObservations'>
    >>> # define a Gamma GLM providing a string
    >>> nmo.glm.GLM(observation_model="Gamma")
    GLM(
        observation_model=GammaObservations(),
        inverse_link_function=one_over_x,
        regularizer=UnRegularized(),
        solver_name='GradientDescent'
    )
    >>> # or equivalently, passing the observation model object
    >>> nmo.glm.GLM(observation_model=nmo.observation_models.GammaObservations())
    GLM(
        observation_model=GammaObservations(),
        inverse_link_function=one_over_x,
        regularizer=UnRegularized(),
        solver_name='GradientDescent'
    )
    >>> # define GLM model of PoissonObservations model with soft-plus NL
    >>> model = nmo.glm.GLM(inverse_link_function=jax.nn.softplus, solver_name="LBFGS")
    >>> print("Regularizer type: ", type(model.regularizer))
    Regularizer type:  <class 'nemos.regularizer.UnRegularized'>
    >>> print("Observation model: ", type(model.observation_model))
    Observation model:  <class 'nemos.observation_models.PoissonObservations'>
    """

    _validator = GLMValidator()

    def __init__(
        self,
        # With python 3.11 Literal[*AVAILABLE_OBSERVATION_MODELS] will be allowed.
        # Replace this manual list after dropping support for 3.10?
        observation_model: (
            obs.Observations
            | Literal["Poisson", "Gamma", "Bernoulli", "NegativeBinomial"]
        ) = "Poisson",
        inverse_link_function: Optional[Callable] = None,
        regularizer: Optional[Union[str, Regularizer]] = None,
        regularizer_strength: Optional[RegularizerStrength] = None,
        solver_name: str = None,
        solver_kwargs: dict = None,
    ):
        super().__init__(
            regularizer=regularizer,
            regularizer_strength=regularizer_strength,
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
        )

        self.observation_model = observation_model
        self.inverse_link_function = inverse_link_function

        # initialize to None fit output
        self.intercept_ = None
        self.coef_ = None
        self.solver_state_ = None
        self.scale_ = None
        self.dof_resid_ = None

    def __sklearn_tags__(self):
        """Return GLM specific estimator tags."""
        tags = super().__sklearn_tags__()
        # Tags for X
        tags.input_tags = InputTags(allow_nan=True, two_d_array=True)
        # Tags for y
        tags.target_tags = TargetTags(
            required=True, one_d_labels=True, two_d_labels=False
        )
        return tags

    @property
    def inverse_link_function(self):
        """Getter for the inverse link function for the model."""
        return self._inverse_link_function

    @inverse_link_function.setter
    def inverse_link_function(self, inverse_link_function: Callable):
        """Setter for the inverse link function for the model."""
        self._inverse_link_function = resolve_inverse_link_function(
            inverse_link_function, self._observation_model
        )

    @property
    def observation_model(self) -> Union[None, obs.Observations]:
        """Getter for the ``observation_model`` attribute."""
        return self._observation_model

    @observation_model.setter
    def observation_model(self, observation: obs.Observations):
        if isinstance(observation, str):
            self._observation_model = instantiate_observation_model(observation)
            return
        # check that the model has the required attributes
        # and that the attribute can be called
        obs.check_observation_model(observation)
        self._observation_model = observation

    def _check_params(
        self,
        params: GLMUserParams,
        data_type: Optional[jnp.dtype] = None,
    ) -> GLMParams:
        """
        Validate the dimensions and consistency of parameters and data.

        This function checks the consistency of shapes and dimensions for model
        parameters.
        It ensures that the parameters and data are compatible for the model.

        """
        return self._validator.validate_and_cast(params)

    def _check_is_fit(self):
        """Ensure the instance has been fitted."""
        if (self.coef_ is None) or (self.intercept_ is None):
            raise NotFittedError(
                "This GLM instance is not fitted yet. Call 'fit' with appropriate arguments."
            )

    def _predict(
        self, params: GLMParams, X: Union[dict[str, jnp.ndarray], jnp.ndarray]
    ) -> jnp.ndarray:
        """
        Predicts firing rates based on given parameters and design matrix.

        This function computes the predicted firing rates using the provided parameters
        and model design matrix ``X``. It is a streamlined version used internally within
        optimization routines, where it serves as the loss function. Unlike the ``GLM.predict``
        method, it does not perform any input validation, assuming that the inputs are pre-validated.


        Parameters
        ----------
        params :
            GLMParams containing the spike basis coefficients and bias terms.
        X :
            Predictors.

        Returns
        -------
        :
            The predicted rates. Shape (n_time_bins, ).
        """
        return self._inverse_link_function(
            # First, multiply each feature by its corresponding coefficient,
            # then sum across all features and add the intercept, before
            # passing to the inverse link function
            tree_utils.pytree_map_and_reduce(
                lambda x, w: jnp.dot(x, w), sum, X, params.coef
            )
            + params.intercept
        )

    @support_pynapple(conv_type="jax")
    def predict(self, X: DESIGN_INPUT_TYPE) -> jnp.ndarray:
        """Predict rates based on fit parameters.

        Parameters
        ----------
        X :
            Predictors, array of shape ``(n_time_bins, n_features)`` or pytree of same.

        Returns
        -------
        :
            The predicted rates with shape ``(n_time_bins, )``.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called first with this instance.
        ValueError
            If ``params`` is not a JAX pytree of size two.
        ValueError
            If weights and bias terms in ``params`` don't have the expected dimensions.
        ValueError
            If ``X`` is not three-dimensional.
        ValueError
            If there's an inconsistent number of features between spike basis coefficients and ``X``.

        Examples
        --------
        >>> # example input
        >>> import numpy as np
        >>> X, y = np.random.normal(size=(10, 2)), np.random.poisson(size=10)
        >>> # define and fit a GLM
        >>> import nemos as nmo
        >>> model = nmo.glm.GLM()
        >>> model = model.fit(X, y)
        >>> # predict new spike data
        >>> Xnew = np.random.normal(size=(20, X.shape[1]))
        >>> predicted_spikes = model.predict(Xnew)

        See Also
        --------
        :meth:`nemos.glm.GLM.score`
            Score predicted rates against target spike counts.

        :meth:`nemos.glm.GLM.simulate`
            Simulate neural activity in response to a feed-forward input (feed-forward only).

        :func:`nemos.simulation.simulate_recurrent`
            Simulate neural activity in response to a feed-forward input
            using the GLM as a recurrent network (feed-forward + coupling).
        """
        # check that the model is fitted
        self._check_is_fit()
        # extract model params
        params = self._get_model_params()

        # filter for non-nans, grab data if needed
        data, _ = self._preprocess_inputs(X, drop_nans=False)

        self._validator.validate_inputs(data)

        # check consistency between X and params
        self._validator.validate_consistency(params, X=data)
        self._validator.feature_mask_consistency(
            getattr(self, "_feature_mask", None), params
        )

        return self._predict(params, data)

    def _compute_loss(
        self,
        params: GLMParams,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        *args,
        **kwargs,
    ) -> jnp.ndarray:
        r"""Predict the rate and compute the negative log-likelihood against neural activity.

        This method computes the negative log-likelihood up to a constant term. Unlike ``score``,
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
        return self._observation_model._negative_log_likelihood(y, predicted_rate)

    def score(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: ArrayLike,
        score_type: Literal[
            "log-likelihood", "pseudo-r2-McFadden", "pseudo-r2-Cohen"
        ] = "log-likelihood",
        aggregate_sample_scores: Callable = jnp.mean,
    ) -> jnp.ndarray:
        r"""Evaluate the goodness-of-fit of the model to the observed neural data.

        This method computes the goodness-of-fit score, which can either be the mean
        log-likelihood or of two versions of the pseudo-:math:`R^2`.
        The scoring process includes validation of input compatibility with the model's
        parameters, ensuring that the model has been previously fitted and the input data
        are appropriate for scoring. A higher score indicates a better fit of the model
        to the observed data.


        Parameters
        ----------
        X :
            The exogenous variables. Shape ``(n_time_bins, n_features)``.
        y :
            Neural activity. Shape ``(n_time_bins, )``.
        score_type :
            Type of scoring: either log-likelihood or pseudo-:math:`R^2`.
        aggregate_sample_scores :
            Function that aggregates the score of all samples.

        Returns
        -------
        score :
            The log-likelihood or the pseudo-:math:`R^2` of the current model.

        Raises
        ------
        NotFittedError

            If ``fit`` has not been called first with this instance.
        ValueError

            If X structure doesn't match the params, and if X and y have different
            number of samples.

        Examples
        --------
        >>> # example input
        >>> import numpy as np
        >>> X, y = np.random.normal(size=(10, 2)), np.random.poisson(size=10)
        >>> import nemos as nmo
        >>> model = nmo.glm.GLM()
        >>> model = model.fit(X, y)
        >>> # get model score
        >>> log_likelihood_score = model.score(X, y)
        >>> # get a pseudo-R2 score
        >>> pseudo_r2_score = model.score(X, y, score_type='pseudo-r2-McFadden')

        Notes
        -----
        The log-likelihood is not on a standard scale, its value is influenced by many factors,
        among which the number of model parameters. The log-likelihood can assume both positive
        and negative values.

        The Pseudo-:math:`R^2` is not equivalent to the :math:`R^2` value in linear regression. While both
        provide a measure of model fit, and assume values in the [0,1] range, the methods and
        interpretations can differ. The Pseudo-:math:`R^2` is particularly useful for generalized linear
        models when the interpretation of the :math:`R^2` as explained variance does not apply
        (i.e., when the observations are not Gaussian distributed).

        Why does the traditional :math:`R^2` is usually a poor measure of performance in GLMs?

        1.  In the context of GLMs the variance and the mean of the observations are related.
            Ignoring the relation between them can result in underestimating the model
            performance; for instance, when we model a Poisson variable with large mean we expect an
            equally large variance. In this scenario, even if our model perfectly captures the mean,
            the high-variance  will result in large residuals and low :math:`R^2`.
            Additionally, when the mean of the observations varies, the variance will vary too. This
            violates the "homoschedasticity" assumption, necessary  for interpreting the :math:`R^2` as
            variance explained.

        2.  The :math:`R^2` capture the variance explained when the relationship between the observations and
            the predictors is linear. In GLMs, the link function sets a non-linear mapping between the predictors
            and the mean of the observations, compromising the interpretation of the :math:`R^2`.

        Note that it is possible to re-normalized the residuals by a mean-dependent quantity proportional
        to the model standard deviation (i.e. Pearson residuals). This "rescaled" residual distribution however
        deviates substantially from normality for counting data with low mean (common for spike counts).
        Therefore, even the Pearson residuals performs poorly as a measure of fit quality, especially
        for GLM modeling counting data.

        Refer to the ``nmo.observation_models.Observations`` concrete subclasses for the likelihood and
        pseudo-:math:`R^2` equations.

        """
        self._check_is_fit()
        params = self._get_model_params()

        self._validator.validate_inputs(X, y)
        X, y = self._preprocess_inputs(X, y, drop_nans=True)
        self._validator.validate_consistency(params, X, y)
        self._validator.feature_mask_consistency(
            getattr(self, "_feature_mask", None), params
        )

        if score_type == "log-likelihood":
            score = self._observation_model.log_likelihood(
                y,
                self._predict(params, X),
                self.scale_,
                aggregate_sample_scores=aggregate_sample_scores,
            )
        elif score_type.startswith("pseudo-r2"):
            score = self._observation_model.pseudo_r2(
                y,
                self._predict(params, X),
                score_type=score_type,
                scale=self.scale_,
                aggregate_sample_scores=aggregate_sample_scores,
            )
        else:
            raise NotImplementedError(
                f"Scoring method {score_type} not implemented! "
                "`score_type` must be either 'log-likelihood', 'pseudo-r2-McFadden', "
                "or 'pseudo-r2-Cohen'."
            )
        return score

    def _model_specific_initialization(
        self,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
    ) -> GLMParams:
        """Initialize the parameters based on the structure and dimensions X and y.

        This method initializes the coefficients (spike basis coefficients) and intercepts (bias terms)
        required for the GLM. The coefficients are initialized to zeros with dimensions based on the input X.
        If X is a :class:`nemos.pytrees.FeaturePytree`, the coefficients retain the pytree structure with
        arrays of zeros shaped according to the features in X.
        If X is a simple ndarray, the coefficients are initialized as a 2D array. The intercepts are initialized
        based on the log mean of the target data y across the first axis, corresponding to the average log activity
        of the neuron.

        Parameters
        ----------
        X :
            The input data which can be a :class:`nemos.pytrees.FeaturePytree` with n_features arrays of shape
            ``(n_timebins, n_features)``, or a simple ndarray of shape ``(n_timebins, n_features)``.
        y :
            The target data array of shape ``(n_timebins, )``, representing
            the neuron firing rates or similar metrics.

        Returns
        -------
        Tuple[Union[FeaturePytree, jnp.ndarray], jnp.ndarray]
            A tuple containing the initialized parameters:
            - The first element is the initialized coefficients
            (either as a FeaturePytree or ndarray, matching the structure of X) with shapes (n_features,).
            - The second element is the initialized intercept (bias terms) as an ndarray of shape (1,).
        """
        if isinstance(X, FeaturePytree):
            data = X.data
        else:
            data = X

        initial_intercept = initialize_intercept_matching_mean_rate(
            self._inverse_link_function, y
        )

        # Initialize parameters
        init_params = GLMParams(
            # coeff, spike basis coeffs.
            # - If X is a FeaturePytree with n_features arrays of shape
            #   (n_timebins, n_features), then this will be a
            #   dict with n_features arrays of shape (n_features,).
            # - If X is an array of shape (n_timebins,
            #   n_features), this will be an array of shape (n_features,).
            jax.tree_util.tree_map(
                lambda x: jnp.zeros((*x[0].shape, *y.shape[1:])), data
            ),
            # intercept, bias terms, keepdims=False needed by PopulationGLM
            initial_intercept,
        )
        return init_params

    @cast_to_jax
    def fit(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: ArrayLike,
        init_params: Optional[GLMUserParams] = None,
    ):
        """Fit GLM to neural activity.

        Fit and store the model parameters as attributes
        ``coef_`` and ``coef_``.

        Parameters
        ----------
        X :
            Predictors, array of shape (n_time_bins, n_features) or pytree of the same
            shape.
        y :
            Target neural activity arranged in a matrix, shape (n_time_bins, ).
        init_params :
            2-tuple of initial parameter values: (coefficients, intercepts). If
            None, we initialize coefficients with zeros, intercepts with the
            log of the mean neural activity. coefficients is an array of shape
            (n_features,) or pytree of same, intercepts is an array
            of shape (1, )

        Raises
        ------
        ValueError
            If ``init_params`` is not of length two.
        ValueError
            If dimensionality of ``init_params`` are not correct.
        ValueError
            If ``X`` is not two-dimensional.
        ValueError
            If ``y`` is not one-dimensional.
        ValueError
            If solver returns at least one NaN parameter, which means it found
              an invalid solution. Try tuning optimization hyperparameters.
        TypeError
            If ``init_params`` are not array-like
        TypeError
            If ``init_params[i]`` cannot be converted to ``jnp.ndarray`` for all ``i``

        Examples
        --------
        >>> # example input
        >>> import numpy as np
        >>> X, y = np.random.normal(size=(10, 2)), np.random.poisson(size=10)
        >>> # fit a ridge regression Poisson GLM
        >>> import nemos as nmo
        >>> model = nmo.glm.GLM(regularizer="Ridge", regularizer_strength=0.1)
        >>> model = model.fit(X, y)
        >>> # get model weights and intercept
        >>> model_weights = model.coef_
        >>> model_intercept = model.intercept_

        """
        self._validator.validate_inputs(X, y)

        # initialize params if no params are provided
        if init_params is None:
            init_params = self._model_specific_initialization(X, y)
        else:
            init_params = self._validator.validate_and_cast(init_params)
            self._validator.validate_consistency(init_params, X=X, y=y)

        self._validator.feature_mask_consistency(
            getattr(self, "_feature_mask", None), init_params
        )

        # filter for non-nans, grab data if needed
        data, y = self._preprocess_inputs(X, y)

        self._initialize_solver_and_state(data, y, init_params)

        params, state = self.solver_run(init_params, data, y)

        if tree_utils.pytree_map_and_reduce(
            lambda x: jnp.any(jnp.isnan(x)), any, params
        ):
            raise ValueError(
                "Solver returned at least one NaN parameter, so solution is invalid!"
                " Try tuning optimization hyperparameters, specifically try decreasing the `stepsize` "
                "and/or setting `acceleration=False`."
            )

        if not self._solver.get_optim_info(state).converged:
            warnings.warn(
                "The fit did not converge. "
                "Consider the following:"
                "\n1) Enable float64 with ``jax.config.update('jax_enable_x64', True)`` "
                "\n2) Increase the max number of iterations or increase tolerance (if reasonable). "
                "These parameters can be specified by providing a ``solver_kwargs`` dictionary. "
                "For the available options see the ``self.solver.__init__`` docstrings.",
                RuntimeWarning,
            )

        self._set_model_params(params)

        self.dof_resid_ = self._estimate_resid_degrees_of_freedom(X)
        self.scale_ = self.observation_model.estimate_scale(
            y, self._predict(params, data), dof_resid=self.dof_resid_
        )

        # note that this will include an error value, which is not the same as
        # the output of loss. I believe it's the output of
        # solver.l2_optimality_error
        self.solver_state_ = state
        return self

    def _get_model_params(self):
        """Pack coef_ and intercept_  into a params pytree.

        This method should be overwritten in case the parameter structure changes,
        or if new regression models will have a different parameter structure.
        """
        # Retrieve parameter tree
        return GLMParams(self.coef_, self.intercept_)

    def _set_model_params(self, params: GLMParams):
        """Unpack and store params pytree to coef_ and intercept_.

        This method should be overwritten in case the parameter structure changes,
        or if new regression models will have a different parameter structure.
        """
        # Store parameters
        self.coef_: DESIGN_INPUT_TYPE = params.coef
        self.intercept_: jnp.ndarray = params.intercept

    @support_pynapple(conv_type="jax")
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
            Array of shape (n_time_bins, n_basis_input) or pytree of same.

        Returns
        -------
        simulated_activity :
            Simulated activity (spike counts for Poisson GLMs) for the neuron over time.
            Shape: ``(n_time_bins, )``.
        firing_rates :
            Simulated rates for the neuron over time. Shape, ``(n_time_bins, )``.

        Raises
        ------
        NotFittedError
            - If the model hasn't been fitted prior to calling this method.
        ValueError
            - If the instance has not been previously fitted.

        Examples
        --------
        >>> # example input
        >>> import numpy as np
        >>> X, y = np.random.normal(size=(10, 2)), np.random.poisson(size=10)
        >>> # define and fit model
        >>> import nemos as nmo
        >>> model = nmo.glm.GLM()
        >>> model = model.fit(X, y)
        >>> # generate spikes and rates
        >>> random_key = jax.random.key(123)
        >>> Xnew = np.random.normal(size=(20, X.shape[1]))
        >>> spikes, rates = model.simulate(random_key, Xnew)

        See Also
        --------
        :meth:`nemos.glm.GLM.predict`
            Method to predict rates based on the model's parameters.
        """
        # check if the model is fit
        self._check_is_fit()

        params = self._get_model_params()

        # if all invalid, raise error
        validation.error_all_invalid(feedforward_input)

        # check input dimensionality
        self._validator.validate_inputs(X=feedforward_input)

        # validate input and params consistency
        self._validator.validate_consistency(params, X=feedforward_input)
        self._validator.feature_mask_consistency(
            getattr(self, "_feature_mask", None), params
        )

        # pre-process
        feedforward_input, _ = self._preprocess_inputs(
            X=feedforward_input, drop_nans=False
        )

        predicted_rate = self._predict(params, feedforward_input)
        return (
            self._observation_model.sample_generator(
                key=random_key, predicted_rate=predicted_rate, scale=self.scale_
            ),
            predicted_rate,
        )

    def _estimate_resid_degrees_of_freedom(
        self, X: DESIGN_INPUT_TYPE, n_samples: Optional[int] = None
    ) -> jnp.ndarray:
        """
        Estimate the degrees of freedom of the residuals.

        Parameters
        ----------
        self :
            A fitted GLM model.
        X :
            The design matrix.
        n_samples :
            The number of samples observed. If not provided, n_samples is set to ``X.shape[0]``. If the fit is
            batched, the n_samples could be larger than ``X.shape[0]``.

        Returns
        -------
        :
            An estimate of the degrees of freedom of the residuals.
        """
        # Convert a pytree to a design-matrix with pytrees
        X = jnp.hstack(jax.tree_util.tree_leaves(X))

        if n_samples is None:
            n_samples = X.shape[0]
        else:
            if not isinstance(n_samples, int):
                raise TypeError(
                    "`n_samples` must either `None` or of type `int`. Type {type(n_sample)} provided "
                    "instead!"
                )

        params = self._get_model_params()
        # if the regularizer is lasso use the non-zero
        # coeff as an estimate of the dof
        # see https://arxiv.org/abs/0712.0881
        if isinstance(self.regularizer, (GroupLasso, Lasso, ElasticNet)):
            resid_dof = tree_utils.pytree_map_and_reduce(
                lambda x: ~jnp.isclose(x, jnp.zeros_like(x)),
                lambda x: sum([jnp.sum(i, axis=0) for i in x]),
                params.coef,
            )
            return n_samples - resid_dof - 1

        elif isinstance(self.regularizer, Ridge):
            # for Ridge, use the tot parameters (X.shape[1] + intercept)
            return (n_samples - X.shape[1] - 1) * jnp.ones_like(params.intercept)
        else:
            # for UnRegularized, use the rank
            rank = jnp.linalg.matrix_rank(X)
            return (n_samples - rank - 1) * jnp.ones_like(params.intercept)

    def _initialize_solver_and_state(
        self,
        X: dict[str, jnp.ndarray] | jnp.ndarray,
        y: jnp.ndarray,
        init_params: GLMParams,
    ) -> SolverState:
        """Initialize the solver by instantiating its init_state, update and, run methods.

        This method also prepares the solver's state by using the initialized model parameters and data.
        This setup is ready to be used for running the solver's optimization routines.

        Parameters
        ----------
        X :
            The predictors used in the model fitting process. This can include feature matrices or other structures
            compatible with the model's design.
        y :
            The response variables or outputs corresponding to the predictors. Used to initialize parameters when
            they are not provided.
        init_params :
            Initial parameters for the model.

        Returns
        -------
        SolverState
            The initialized solver state

        Examples
        --------
        >>> import numpy as np
        >>> import nemos as nmo
        >>> X, y = np.random.normal(size=(10, 2)), np.random.poisson(size=10)
        >>> model = nmo.glm.GLM()
        >>> params = model._initialize_params(X, y)
        >>> opt_state = model._initialize_solver_and_state(X, y, params)
        >>> # Now ready to run optimization or update steps
        """
        opt_solver_kwargs = self._optimize_solver_params(X, y)
        #  set up the solver init/run/update attrs
        self._instantiate_solver(self._compute_loss, solver_kwargs=opt_solver_kwargs)

        opt_state = self.solver_init_state(init_params, X, y)
        return opt_state

    @cast_to_jax
    def update(
        self,
        params: GLMUserParams,
        opt_state: NamedTuple,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        *args,
        n_samples: Optional[int] = None,
        **kwargs,
    ) -> StepResult:
        """
        Update the model parameters and solver state.

        This method performs a single optimization step using the model's current solver.
        It updates the model's coefficients and intercept based on the provided parameters, predictors (X),
        responses (y), and the current optimization state. This method is particularly useful for iterative
        model fitting, especially in scenarios where model parameters need to be updated incrementally,
        such as online learning or when dealing with very large datasets that do not fit into memory at once.

        Parameters
        ----------
        params :
            The current model parameters, typically a tuple of coefficients and intercepts.
        opt_state :
            The current state of the optimizer, encapsulating information necessary for the
            optimization algorithm to continue from the current state. This includes gradients,
            step sizes, and other optimizer-specific metrics.
        X :
            The predictors used in the model fitting process, which may include feature matrices
            or :class:`nemos.pytrees.FeaturePytree` objects.
        y :
            The response variable or output data corresponding to the predictors, used in the model
            fitting process.
        *args
            Additional positional arguments to be passed to the solver's update method.
        n_samples:
            The tot number of samples. Usually larger than the samples of an indivisual batch,
            the ``n_samples`` are used to estimate the scale parameter of the GLM.
        **kwargs
            Additional keyword arguments to be passed to the solver's update method.

        Returns
        -------
        StepResult
            A tuple containing the updated parameters and optimization state. This tuple is
            typically used to continue the optimization process in subsequent steps.

        Raises
        ------
        ValueError
            If the solver has not been instantiated or if the solver returns NaN values
            indicating an invalid update step, typically due to numerical instabilities
            or inappropriate solver configurations.

        Examples
        --------
        >>> import nemos as nmo
        >>> import numpy as np
        >>> X, y = np.random.normal(size=(10, 2)), np.random.uniform(size=10)
        >>> glm_instance = nmo.glm.GLM().fit(X, y)
        >>> params = glm_instance.coef_, glm_instance.intercept_
        >>> opt_state = glm_instance.solver_state_
        >>> new_params, new_opt_state = glm_instance.update(params, opt_state, X, y)

        """
        # find non-nans
        X, y = tree_utils.drop_nans(X, y)

        # grab the data
        data = X.data if isinstance(X, FeaturePytree) else X

        # wrap into GLM params, this assumes params are well structured,
        # if initializaiton is done via `initialize_solver_and_state` it
        # should be fine
        params = self._validator.to_model_params(params)

        # perform a one-step update
        updated_params, updated_state = self.solver_update(
            params, opt_state, data, y, *args, **kwargs
        )

        # store params and state
        self._set_model_params(updated_params)
        self.solver_state_ = updated_state

        # estimate the scale
        self.dof_resid_ = self._estimate_resid_degrees_of_freedom(
            X, n_samples=n_samples
        )
        self.scale_ = self.observation_model.estimate_scale(
            y, self._predict(params, data), dof_resid=self.dof_resid_
        )

        return self._validator.from_model_params(updated_params), updated_state

    def _get_optimal_solver_params_config(self):
        """Return the functions for computing default step and batch size for the solver."""
        return glm_compute_optimal_stepsize_configs(self)

    def __repr__(self):
        """Representation of the GLM class."""
        return format_repr(
            self, multiline=True, use_name_keys=["inverse_link_function"]
        )

    def __sklearn_clone__(self) -> GLM:
        """Clone the GLM."""
        params = self.get_params(deep=False)
        klass = self.__class__(**params)
        return klass

    def save_params(self, filename: Union[str, Path]):
        """
        Save GLM model parameters to a .npz file.

        This method allows to reuse the model parameters. The saved parameters can be loaded back
        into a GLM instance using the `load_params` function.

        Parameters
        ----------
        filename :
            The name of the file where the model parameters will be saved. The file will be saved in `.npz` format.

        Examples
        --------
        >>> import nemos as nmo
        >>> # Create a GLM model with specified parameters
        >>> solver_args = {"stepsize": 0.1, "maxiter": 1000, "tol": 1e-6}
        >>> model = nmo.glm.GLM(
        ...     regularizer="Ridge",
        ...     regularizer_strength=0.1,
        ...     observation_model="Gamma",
        ...     solver_name="BFGS",
        ...     solver_kwargs=solver_args,
        ... )
        >>> for key, value in model.get_params().items():
        ...     print(f"{key}: {value}")
        inverse_link_function: <function one_over_x at ...>
        observation_model: GammaObservations()
        regularizer: Ridge()
        regularizer_strength: 0.1
        solver_kwargs: {'stepsize': 0.1, 'maxiter': 1000, 'tol': 1e-06}
        solver_name: BFGS
        >>> # Save the model parameters to a file
        >>> model.save_params("model_params.npz")
        >>> # Load the model from the saved file
        >>> model = nmo.load_model("model_params.npz")
        >>> # Model has the same parameters before and after load
        >>> for key, value in model.get_params().items():
        ...     print(f"{key}: {value}")
        inverse_link_function: <function one_over_x at ...>
        observation_model: GammaObservations()
        regularizer: Ridge()
        regularizer_strength: 0.1
        solver_kwargs: {'stepsize': 0.1, 'maxiter': 1000, 'tol': 1e-06}
        solver_name: BFGS

        >>> # Saving and loading a custom inverse link function
        >>> model = nmo.glm.GLM(
        ...     observation_model="Poisson",
        ...     inverse_link_function=lambda x: x**2
        ... )
        >>> model.save_params("model_params.npz")
        >>> # Provide a mapping for the custom link function when loading.
        >>> mapping_dict = {
        ...     "inverse_link_function": lambda x: x**2,
        ... }
        >>> loaded_model = nmo.load_model("model_params.npz", mapping_dict=mapping_dict)
        >>> # Now the loaded model will have the updated solver_name and solver_kwargs
        >>> for key, value in loaded_model.get_params().items():
        ...     print(f"{key}: {value}")
        inverse_link_function: <function <lambda> at ...>
        observation_model: PoissonObservations()
        regularizer: UnRegularized()
        regularizer_strength: None
        solver_kwargs: {}
        solver_name: GradientDescent
        """

        # initialize saving dictionary
        fit_attrs = self._get_fit_state()
        fit_attrs.pop("solver_state_")
        string_attrs = ["inverse_link_function"]

        super().save_params(filename, fit_attrs, string_attrs)


class PopulationGLM(GLM):
    """
    Population Generalized Linear Model.

    This class implements a Generalized Linear Model for a neural population.
    This GLM implementation allows users to model the activity of a population of neurons based on a
    combination of exogenous inputs (like convolved currents or light intensities) and a choice of observation model.
    It is suitable for scenarios where the relationship between predictors and the response
    variable might be non-linear, and the residuals  don't follow a normal distribution. The predictors must be
    stored in tabular format, shape (n_timebins, num_features) or as :class:`nemos.pytrees.FeaturePytree`.
    Below is a table listing the default and available solvers for each regularizer.

    +---------------+------------------+-------------------------------------------------------------+
    | Regularizer   | Default Solver   | Available Solvers                                           |
    +===============+==================+=============================================================+
    | UnRegularized | GradientDescent  | GradientDescent, BFGS, LBFGS, NonlinearCG, ProximalGradient |
    +---------------+------------------+-------------------------------------------------------------+
    | Ridge         | GradientDescent  | GradientDescent, BFGS, LBFGS, NonlinearCG, ProximalGradient |
    +---------------+------------------+-------------------------------------------------------------+
    | Lasso         | ProximalGradient | ProximalGradient                                            |
    +---------------+------------------+-------------------------------------------------------------+
    | GroupLasso    | ProximalGradient | ProximalGradient                                            |
    +---------------+------------------+-------------------------------------------------------------+

    **Fitting Large Models**

    For very large models, you may consider using the Stochastic Variance Reduced Gradient
    :class:`nemos.solvers._svrg.SVRG` or its proximal variant
    (:class:`nemos.solvers._svrg.ProxSVRG`) solver,
    which take advantage of batched computation. You can change the solver by passing
    ``"SVRG"`` or ``"ProxSVRG"`` as ``solver_name`` at model initialization.

    The performance of the SVRG solver depends critically on the choice of ``batch_size`` and ``stepsize``
    hyperparameters. These parameters control the size of the mini-batches used for gradient computations
    and the step size for each iteration, respectively. Improper selection of these parameters can lead to slow
    convergence or even divergence of the optimization process.

    To assist with this, for certain GLM configurations, we provide ``batch_size`` and ``stepsize`` default
    values that are theoretically guaranteed to ensure fast convergence.

    Below is a list of the configurations for which we can provide guaranteed hyperparameters:

    +---------------------------------------+-----------+-------------+
    | GLM / PopulationGLM Configuration     | Stepsize  | Batch Size  |
    +=======================================+===========+=============+
    | Poisson + soft-plus + UnRegularized   | ✅         | ❌         |
    +---------------------------------------+-----------+-------------+
    | Poisson + soft-plus + Ridge           | ✅         | ✅         |
    +---------------------------------------+-----------+-------------+
    | Poisson + soft-plus + Lasso           | ✅         | ❌         |
    +---------------------------------------+-----------+-------------+
    | Poisson + soft-plus + GroupLasso      | ✅         | ❌         |
    +---------------------------------------+-----------+-------------+

    Parameters
    ----------
    observation_model :
        Observation model to use. The model describes the distribution of the neural activity.
        Default is the Poisson model.
    inverse_link_function :
        A function that maps the linear combination of predictors into a firing rate. The default depends
        on the observation model, see the table above.
    regularizer :
        Regularization to use for model optimization. Defines the regularization scheme
        and related parameters.
        Default is UnRegularized regression.
    regularizer_strength :
        Float that is default None. Sets the regularizer strength. If a user does not pass a value, and it is needed for
        regularization, a warning will be raised and the strength will default to 1.0.
    solver_name :
        Solver to use for model optimization. Defines the optimization scheme and related parameters.
        The solver must be an appropriate match for the chosen regularizer.
        Default is ``None``. If no solver specified, one will be chosen based on the regularizer.
        Please see table above for regularizer/optimizer pairings.
    solver_kwargs :
        Optional dictionary for keyword arguments that are passed to the solver when instantiated.
        E.g. stepsize, tol, acceleration, etc.
         For details on each solver's kwargs, see `get_accepted_arguments` and `get_solver_documentation`.
    feature_mask :
        Either a matrix of shape (num_features, num_neurons) or a :meth:`nemos.pytrees.FeaturePytree` of 0s and 1s, with
        ``feature_mask[feature_name]`` of shape (num_neurons, ).
        The mask will be used to select which features are used as predictors for which neuron.

    Attributes
    ----------
    intercept_ :
        Model baseline linked firing rate parameters, e.g. if the link is the logarithm, the baseline
        firing rate will be ``jnp.exp(model.intercept_)``.
    coef_ :
        Basis coefficients for the model.
    solver_state_ :
        State of the solver after fitting. May include details like optimization error.

    Raises
    ------
    TypeError
        If provided ``regularizer`` or ``observation_model`` are not valid.
    TypeError
        If provided ``feature_mask`` is not an array-like of dimension two.

    Examples
    --------
    >>> # Example with an array mask
    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>> from nemos.glm import PopulationGLM
    >>> # Define predictors (X), weights, and neural activity (y)
    >>> num_samples, num_features, num_neurons = 100, 3, 2
    >>> X = np.random.normal(size=(num_samples, num_features))
    >>> weights = np.array([[ 0.5,  0. ], [-0.5, -0.5], [ 0. ,  1. ]])
    >>> y = np.random.poisson(np.exp(X.dot(weights)))
    >>> # Define a feature mask, shape (num_features, num_neurons)
    >>> feature_mask = np.array([[1, 0], [1, 1], [0, 1]])
    >>> feature_mask
    array([[1, 0],
           [1, 1],
           [0, 1]])
    >>> # Create and fit the model
    >>> model = PopulationGLM(feature_mask=feature_mask).fit(X, y)
    >>> model
    PopulationGLM(
        observation_model=PoissonObservations(),
        inverse_link_function=exp,
        regularizer=UnRegularized(),
        solver_name='GradientDescent'
    )
    >>> # Check the fitted coefficients
    >>> print(model.coef_.shape)
    (3, 2)
    >>> # Example with a FeaturePytree mask
    >>> from nemos.pytrees import FeaturePytree
    >>> # Define two features
    >>> feature_1 = np.random.normal(size=(num_samples, 2))
    >>> feature_2 = np.random.normal(size=(num_samples, 1))
    >>> # Define the FeaturePytree predictor, and weights
    >>> X = FeaturePytree(feature_1=feature_1, feature_2=feature_2)
    >>> weights = dict(feature_1=jnp.array([[0., 0.5], [0., -0.5]]), feature_2=jnp.array([[1., 0.]]))
    >>> # Compute the firing rate and counts
    >>> rate = np.exp(X["feature_1"].dot(weights["feature_1"]) + X["feature_2"].dot(weights["feature_2"]))
    >>> y = np.random.poisson(rate)
    >>> # Define a feature mask with arrays of shape (num_neurons, )
    >>> feature_mask = FeaturePytree(
    ...     feature_1=jnp.array([0, 1], dtype=jnp.int32),
    ...     feature_2=jnp.array([1, 0], dtype=jnp.int32)
    ... )
    >>> print(feature_mask)
    feature_1: shape (2,), dtype int32
    feature_2: shape (2,), dtype int32
    >>> # Fit a PopulationGLM
    >>> model = PopulationGLM(feature_mask=feature_mask).fit(X, y)
    >>> # Coefficients are stored in a dictionary with keys the feature labels
    >>> print(model.coef_.keys())
    dict_keys(['feature_1', 'feature_2'])
    """

    _validator = PopulationGLMValidator()

    def __init__(
        self,
        observation_model: (
            obs.Observations
            | Literal["Poisson", "Gamma", "Bernoulli", "NegativeBinomial"]
        ) = "Poisson",
        inverse_link_function: Optional[Callable] = None,
        regularizer: Union[str, Regularizer] = "UnRegularized",
        regularizer_strength: Optional[float] = None,
        solver_name: str = None,
        solver_kwargs: dict = None,
        feature_mask: Optional[jnp.ndarray] = None,
        **kwargs,
    ):
        super().__init__(
            observation_model=observation_model,
            inverse_link_function=inverse_link_function,
            regularizer_strength=regularizer_strength,
            regularizer=regularizer,
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
            **kwargs,
        )
        self._metadata = None
        self.feature_mask = feature_mask

    def __sklearn_tags__(self):
        """Return Population GLM specific estimator tags."""
        tags = super().__sklearn_tags__()
        # Tags for y
        tags.target_tags = TargetTags(
            required=True, one_d_labels=False, two_d_labels=True
        )
        return tags

    @property
    def feature_mask(self) -> Union[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Define a feature mask of shape ``(n_features, n_neurons)``."""
        return self._feature_mask

    @feature_mask.setter
    @cast_to_jax
    def feature_mask(self, feature_mask: Union[DESIGN_INPUT_TYPE, dict]):
        # do not allow reassignment after fit
        if (self.coef_ is not None) and (self.intercept_ is not None):
            raise AttributeError(
                "property 'feature_mask' of 'populationGLM' cannot be set after fitting."
            )

        if feature_mask is None:
            self._feature_mask = feature_mask
            return

        elif isinstance(feature_mask, FeaturePytree):
            feature_mask = feature_mask.data

        if isinstance(feature_mask, dict):
            feature_mask = dict((i, jnp.asarray(v)) for i, v in feature_mask.items())

        self._feature_mask = self._validator.validate_and_cast_feature_mask(
            feature_mask
        )

    @strip_metadata(arg_num=1)
    def fit(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: ArrayLike,
        init_params: Optional[Tuple[Union[dict, ArrayLike], ArrayLike]] = None,
    ):
        """Fit GLM to the activity of a population of neurons.

        Fit and store the model parameters as attributes ``coef_`` and ``intercept_``.
        Each neuron can have different predictors. The ``feature_mask`` will determine which
        feature will be used for which neurons. See the note below for more information on
        the ``feature_mask``.

        Parameters
        ----------
        X :
            Predictors, array of shape (n_timebins, n_features) or pytree of the same
            shape.
        y :
            Target neural activity arranged in a matrix, shape (n_timebins, n_neurons).
        init_params :
            2-tuple of initial parameter values: (coefficients, intercepts). If
            None, we initialize coefficients with zeros, intercepts with the
            log of the mean neural activity. coefficients is an array of shape
            (n_features, n_neurons) or pytree of the same shape, intercepts is an array
            of shape (n_neurons, )

        Raises
        ------
        ValueError
            If ``init_params`` is not of length two.
        ValueError
            If dimensionality of ``init_params`` are not correct.
        ValueError
            If ``X`` is not two-dimensional.
        ValueError
            If ``y`` is not two-dimensional.
        ValueError
            If the ``feature_mask`` is not of the right shape.
        ValueError
            If solver returns at least one NaN parameter, which means it found
            an invalid solution. Try tuning optimization hyperparameters.
        TypeError
            If ``init_params`` are not array-like
        TypeError
            If ``init_params[i]`` cannot be converted to jnp.ndarray for all i

        Notes
        -----
        The ``feature_mask`` is used to select features for each neuron, and it is
        an NDArray or a :class:`nemos.pytrees.FeaturePytree` of 0s and 1s. In particular,

        - If the mask is in array format, feature ``i`` is a predictor for neuron ``j`` if
          ``feature_mask[i, j] == 1``.

        - If the mask is a :class:``nemos.pytrees.FeaturePytree``, then
          ``"feature_name"`` is a predictor of neuron ``j`` if ``feature_mask["feature_name"][j] == 1``.

        Examples
        --------
        >>> # Generate sample data
        >>> import jax.numpy as jnp
        >>> import numpy as np
        >>> from nemos.glm import PopulationGLM
        >>> # Define predictors (X), weights, and neural activity (y)
        >>> num_samples, num_features, num_neurons = 100, 3, 2
        >>> X = np.random.normal(size=(num_samples, num_features))
        >>> # Weights is defined by how each feature influences the output, shape (num_features, num_neurons)
        >>> weights = np.array([[ 0.5,  0. ], [-0.5, -0.5], [ 0. ,  1. ]])
        >>> # Output y simulates a Poisson distribution based on a linear model between features X and wegihts
        >>> y = np.random.poisson(np.exp(X.dot(weights)))
        >>> # Define a feature mask, shape (num_features, num_neurons)
        >>> feature_mask = jnp.array([[1, 0], [1, 1], [0, 1]])
        >>> # Create and fit the model
        >>> model = PopulationGLM(feature_mask=feature_mask).fit(X, y)
        >>> print(model.coef_.shape)
        (3, 2)
        """
        return super().fit(X, y, init_params)

    def _initialize_feature_mask(self, X: FeaturePytree, y: jnp.ndarray):
        if self.feature_mask is None:
            # static checker does not realize conversion to ndarray happened in cast_to_jax.
            if isinstance(X, FeaturePytree):
                self._feature_mask = jax.tree_util.tree_map(
                    lambda x: jnp.ones((y.shape[1],)), X.data
                )
            elif isinstance(X, dict):
                self._feature_mask = jax.tree_util.tree_map(
                    lambda x: jnp.ones((y.shape[1],)), X
                )
            else:
                self._feature_mask = jnp.ones((X.shape[1], y.shape[1]))

    def _predict(self, params: GLMParams, X: jnp.ndarray) -> jnp.ndarray:
        """
        Predicts firing rates based on given parameters and design matrix.

        This function computes the predicted firing rates using the provided parameters, the feature
        mask and model design matrix ``X``. It is a streamlined version used internally within
        optimization routines, where it serves as the loss function. Unlike the ``GLM.predict``
        method, it does not perform any input validation, assuming that the inputs are pre-validated.
        The parameters are first element-wise multiplied with the mask, then the canonical
        linear-non-linear GLM map is applied.

        Parameters
        ----------
        params :
            GLMParams containing the spike basis coefficients and bias terms.
        X :
            Predictors.

        Returns
        -------
        :
            The predicted rates. Shape (n_timebins, n_neurons).
        """
        if self._feature_mask is None:
            return super()._predict(params, X)
        return self.inverse_link_function(
            # First, multiply each feature by its corresponding coefficient,
            # then sum across all features and add the intercept, before
            # passing to the inverse link function
            tree_utils.pytree_map_and_reduce(
                lambda x, w, m: jnp.dot(x, w * m),
                sum,
                X,
                params.coef,
                self._feature_mask,
            )
            + params.intercept
        )

    def __sklearn_clone__(self) -> PopulationGLM:
        """Clone the PopulationGLM, dropping feature_mask."""
        params = self.get_params(deep=False)
        params.pop("feature_mask")
        klass = self.__class__(**params)
        # reattach metadata
        klass._metadata = self._metadata
        return klass
