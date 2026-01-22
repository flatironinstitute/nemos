"""GLM core module."""

# required to get ArrayLike to render correctly
from __future__ import annotations

import warnings
from numbers import Number
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray
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
from ..type_casting import cast_to_jax, is_numpy_array_like, support_pynapple
from ..typing import (
    DESIGN_INPUT_TYPE,
    RegularizerStrength,
    SolverState,
    StepResult,
    UserProvidedParamsT,
)
from ..utils import format_repr
from .initialize_parameters import initialize_intercept_matching_mean_rate
from .params import GLMParams, GLMUserParams
from .validation import (
    ClassifierGLMValidator,
    GLMValidator,
    PopulationClassifierGLMValidator,
    PopulationGLMValidator,
)

REGRESSION_GLM_TYPES = Union[
    obs.BernoulliObservations,
    obs.GammaObservations,
    obs.GaussianObservations,
    obs.NegativeBinomialObservations,
    obs.PoissonObservations,
    Literal[
        "Poisson",
        "Gamma",
        "Bernoulli",
        "NegativeBinomial",
        "Gaussian",
    ],
]


class GLM(BaseRegressor[GLMUserParams, GLMParams]):
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
    | Gaussian            | :math:`x`                       |
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
        Default is the Poisson model. Alternatives are "Gamma", "Bernoulli", "NegativeBinomial" and "Gaussian".
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

    _invalid_observation_types = (obs.CategoricalObservations,)
    _validator_class = GLMValidator

    def __init__(
        self,
        observation_model: REGRESSION_GLM_TYPES = "Poisson",
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

        self._validator = self._validator_class(
            extra_params=self._get_validator_extra_params()
        )

        # initialize to None fit output
        self.intercept_ = None
        self.coef_ = None
        self.solver_state_ = None
        self.scale_ = None
        self.dof_resid_ = None
        self.aux_ = None
        self.optim_info_ = None

    @classmethod
    def _validate_observation_class(cls, observation: obs.Observations):
        if observation.__class__ in cls._invalid_observation_types:
            model_name = cls.__name__
            obs_name = observation.__class__.__name__
            error_msg = f"The ``{obs_name}`` observation type is not supported for ``{model_name}`` models."
            is_categorical = isinstance(observation, obs.CategoricalObservations)
            if is_categorical:
                correct_model = (
                    "ClassifierPopulationGLM"
                    if issubclass(cls, PopulationGLM)
                    else "ClassifierGLM"
                )
                error_msg += (
                    f" To use a GLM for classification instantiate a ``{correct_model}`` "
                    f"object."
                )
            else:
                correct_model = (
                    "PopulationGLM" if issubclass(cls, PopulationGLM) else "GLM"
                )
                error_msg += (
                    f" To use a GLM for regression with ``{obs_name}`` instantiate a ``{correct_model}`` "
                    f"object."
                )
            raise TypeError(error_msg)

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
            self._validate_observation_class(self.observation_model)
            return
        # check that the model has the required attributes
        # and that the attribute can be called
        obs.check_observation_model(observation)
        self._observation_model = observation
        self._validate_observation_class(self.observation_model)

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
                lambda x, w: jnp.einsum("tj, j...->t...", x, w), sum, X, params.coef
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

    @cast_to_jax
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

        empty_params = self._validator.get_empty_params(data, y)

        initial_intercept = initialize_intercept_matching_mean_rate(
            self._inverse_link_function, y
        )
        initial_coef = jax.tree_util.tree_map(
            lambda x: jnp.zeros(x.shape), empty_params.coef
        )

        init_params = eqx.tree_at(
            lambda p: (p.coef, p.intercept),
            empty_params,
            (initial_coef, initial_intercept),
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

        # filter for non-nans, grab data if needed
        data, y = self._preprocess_inputs(X, y)
        # initialize params if no params are provided
        if init_params is None:
            init_params = self._model_specific_initialization(X, y)
        else:
            init_params = self._validator.validate_and_cast_params(init_params)
            self._validator.validate_consistency(init_params, X=X, y=y)

        self._validator.feature_mask_consistency(
            getattr(self, "_feature_mask", None), init_params
        )

        self._initialize_solver_and_state(data, y, init_params)

        params, state, aux = self.solver_run(init_params, data, y)

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
        self.optim_info_ = self._solver.get_optim_info(state)

        self._set_model_params(params)

        self.dof_resid_ = self._estimate_resid_degrees_of_freedom(X)
        self.scale_ = self.observation_model.estimate_scale(
            y, self._predict(params, data), dof_resid=self.dof_resid_
        )

        # note that this will include an error value, which is not the same as
        # the output of loss. I believe it's the output of
        # solver.l2_optimality_error
        self.solver_state_ = state
        self.aux_ = aux
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
                    "`n_samples` must be `None` or of type `int`. Type {type(n_sample)} provided "
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
        >>> params = model.initialize_params(X, y)
        >>> opt_state = model.initialize_solver_and_state(X, y, params)
        >>> # Now ready to run optimization or update steps
        """
        opt_solver_kwargs = self._optimize_solver_params(X, y)
        #  set up the solver init/run/update attrs
        self._instantiate_solver(
            self._compute_loss, init_params=init_params, solver_kwargs=opt_solver_kwargs
        )

        opt_state = self.solver_init_state(init_params, X, y)
        return opt_state

    @cast_to_jax
    def update(
        self,
        params: GLMUserParams,
        opt_state: SolverState,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        *args,
        n_samples: Optional[int] = None,
        **kwargs,
    ) -> StepResult:
        """Update the model parameters and solver state.

        This method performs a single optimization step using the model's current solver.
        It updates the model's coefficients and intercept based on the provided parameters, predictors (X),
        responses (y), and the current optimization state. This method is particularly useful for iterative
        model fitting, especially in scenarios where model parameters need to be updated incrementally,
        such as online learning or when dealing with very large datasets that do not fit into memory at once.

        Parameters
        ----------
        params
            The current model parameters, typically a tuple of coefficients and intercepts.
        opt_state
            The current state of the optimizer, encapsulating information necessary for the
            optimization algorithm to continue from the current state. This includes gradients,
            step sizes, and other optimizer-specific metrics.
        X
            The predictors used in the model fitting process, which may include feature matrices
            or :class:`nemos.pytrees.FeaturePytree` objects. Shape ``(n_time_bins, n_features)``.
        y
            The response variable or output data corresponding to the predictors. Shape ``(n_time_bins,)``.
        *args
            Additional positional arguments to be passed to the solver's update method.
        n_samples
            The total number of samples. Usually larger than the samples of an individual batch,
            the ``n_samples`` are used to estimate the scale parameter of the GLM.
        **kwargs
            Additional keyword arguments to be passed to the solver's update method.

        Returns
        -------
        params
            Updated model parameters (coefficients, intercepts).
        state
            Updated optimizer state.

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
        >>> import jax
        >>> X, y = np.random.normal(size=(10, 2)), np.random.poisson(size=10)
        >>> glm_instance = nmo.glm.GLM()
        >>> params = glm_instance.initialize_params(X, y)
        >>> opt_state = glm_instance.initialize_solver_and_state(X, y, params)
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
        updated_params, updated_state, aux = self.solver_update(
            params, opt_state, data, y, *args, **kwargs
        )

        # store params and state
        self._set_model_params(updated_params)
        self.solver_state_ = updated_state
        self.aux_ = aux

        # estimate the scale
        self.dof_resid_ = self._estimate_resid_degrees_of_freedom(
            X, n_samples=n_samples
        )
        self.scale_ = self.observation_model.estimate_scale(
            y, self._predict(updated_params, data), dof_resid=self.dof_resid_
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
        fit_attrs.pop("optim_info_")
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

    _validator_class = PopulationGLMValidator

    def __init__(
        self,
        observation_model: (
            REGRESSION_GLM_TYPES
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
        """
        Mask indicating which features are used for each neuron.

        The feature mask has a tree structure matching the coefficients (``coef_``):

        - **Array input**: Shape ``(n_features, n_neurons)``. Each entry ``[i, j]``
          indicates whether feature ``i`` is used for neuron ``j`` (1 = used, 0 = masked).

        - **Dict/FeaturePytree input**: A dict with keys matching ``coef_``.
          Each leaf array has shape ``(n_neurons,)``, indicating whether that feature
          group is used for each neuron.

        Returns
        -------
        jnp.ndarray or dict[str, jnp.ndarray]
            The feature mask, or None if not set.
        """
        return self._feature_mask

    @feature_mask.setter
    def feature_mask(self, feature_mask: Union[DESIGN_INPUT_TYPE, dict]):
        # do not allow reassignment after fit
        if (self.coef_ is not None) and (self.intercept_ is not None):
            raise AttributeError(
                "property 'feature_mask' of 'populationGLM' cannot be set after fitting."
            )

        self._feature_mask = self._validator.validate_and_cast_feature_mask(
            feature_mask
        )

    @strip_metadata(arg_num=1)
    def fit(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: ArrayLike,
        init_params: Optional[GLMUserParams] = None,
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
        >>> np.random.seed(0)
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
                lambda x, w, m: jnp.einsum("ti, i...->t...", x, w * m),
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


class ClassifierMixin:
    """GLM for classification."""

    # observation model inferred
    _invalid_observation_types = ()

    def _check_classes_is_set(self, method_name, y=None):
        if self._classes_ is None:
            raise ValueError(
                f"Classes are not set. Must call ``set_classes`` before calling ``{method_name}``."
            )

    def set_classes(self, y: ArrayLike) -> ClassifierMixin:
        """
        Infer unique class labels and set the ``classes_`` attribute.

        This method infers class labels from ``y`` and sets up the internal
        encoding/decoding machinery. When labels are the default ``[0, 1, ..., n_classes-1]``,
        encoding is skipped for performance.

        Parameters
        ----------
        y
            An array that must contain all the class labels,
            i.e. ``len(np.unique(y)) == n_classes``.

        Notes
        -----
        :meth:`fit` and :meth:`initialize_solver_and_state` call ``set_classes`` internally,
        making sure that the ``classes_`` attribute matches the provided input.
        If you are fitting in batches by calling :meth:`update`, make sure that the ``classes_``
        are correctly set by calling ``set_classes`` before starting the :meth:`update` loop.

        Examples
        --------
        >>> import nemos as nmo
        >>> import numpy as np
        >>> model = nmo.glm.ClassifierGLM(3)
        >>> # classes_ is None until set
        >>> model.classes_ is None
        True
        >>> # Integer classes
        >>> y = np.array([2, 3, 2, 5, 5])
        >>> model.set_classes(y)
        ClassifierGLM(...)
        >>> model.classes_
        array([2, 3, 5])
        >>> # String classes
        >>> y = np.array(["a", "a", "c", "b", "b"])
        >>> model.set_classes(y)
        ClassifierGLM(...)
        >>> model.classes_
        array(['a', 'b', 'c'], dtype='<U1')

        """
        # note that we must use NumPy, Jax does not allow non-numeric types
        classes = np.unique(y)
        n_unique = len(classes)

        # Validation
        if n_unique > self.n_classes:
            raise ValueError(
                f"Found {n_unique} unique class labels in y, but n_classes={self.n_classes}. "
                f"Increase n_classes or check your data."
            )
        elif n_unique < self.n_classes:
            raise ValueError(
                f"Found only {n_unique} unique class labels in y, but n_classes={self.n_classes}. "
                f"To correctly set the ``classes_`` attribute, provide an array containing all the "
                f"unique class labels.",
            )

        # Always store the actual classes array
        self._classes_ = classes

        # Check if classes are the default [0, 1, ..., n_classes-1]
        # If so, we can skip encoding/decoding for performance
        is_default = np.array_equal(classes, np.arange(self.n_classes))
        self._skip_encoding = is_default

        # Create dict lookup only when needed (non-default classes)
        self._class_to_index_ = (
            None if is_default else {label: i for i, label in enumerate(classes)}
        )
        return self

    def _encode_labels(self, y):
        """Convert user-provided class labels to internal indices [0, n_classes-1]."""
        if self._skip_encoding:
            return y
        # use dict lookup instead of `np.searchsorted`
        # this approach will fail for label mismatches
        try:
            y = np.fromiter(
                (self._class_to_index_[label] for label in y), dtype=int, count=len(y)
            )
        except KeyError as e:
            unq_labels = np.unique(y)
            valid = list(self._class_to_index_.keys())
            invalid = [lab for lab in unq_labels if lab not in valid]
            raise ValueError(
                f"Unrecognized label(s) {invalid}. " f"Valid labels are {valid}."
            ) from e
        return y

    def _decode_labels(self, indices):
        """Convert internal indices [0, n_classes-1] back to user-provided class labels."""
        if self._skip_encoding:
            return indices
        return self._classes_[indices]

    @property
    def classes_(self) -> NDArray | None:
        """Class labels, or None if not set."""
        return self._classes_

    def compute_loss(
        self,
        params,
        X,
        y,
        *args,
        **kwargs,
    ):
        """
        Compute the loss function for the model.

        This method validates inputs, encodes class labels to internal indices,
        and computes the loss (negative log-likelihood).

        Parameters
        ----------
        params
            Parameter tuple of (coefficients, intercept).
        X
            Input data, array of shape ``(n_time_bins, n_features)`` or pytree of same.
        y
            Target class labels in the same format as ``classes_``.
        *args
            Additional positional arguments passed to the model-specific loss function.
        **kwargs
            Additional keyword arguments passed to the model-specific loss function.

        Returns
        -------
        loss
            The loss value (negative log-likelihood).

        Raises
        ------
        ValueError
            If ``classes_`` has not been set, or if inputs/parameters have
            incompatible shapes or invalid values.
        """
        self._check_classes_is_set("compute_loss")
        y = self._encode_labels(y)
        return super().compute_loss(params, X, y, *args, **kwargs)

    @property
    def n_classes(self):
        """Number of classes."""
        return self._n_classes

    @n_classes.setter
    def n_classes(self, value: int):
        # extract item from scalar arrays
        if is_numpy_array_like(value)[1] and value.size == 1:
            value = value.item()

        if not isinstance(value, Number) or value < 2 or not int(value) == value:
            raise ValueError(
                "The number of classes must be an integer greater than or equal to 2."
            )
        self._n_classes = int(value)
        # reset validator.
        self._validator = self._validator_class(
            extra_params=self._get_validator_extra_params()
        )

    def _get_validator_extra_params(self) -> dict:
        """Get validator extra parameters."""
        return {"n_classes": self._n_classes}

    def _preprocess_inputs(
        self,
        X: DESIGN_INPUT_TYPE,
        y: Optional[jnp.ndarray] = None,
        drop_nans: bool = True,
    ) -> Tuple[dict[str, jnp.ndarray] | jnp.ndarray, jnp.ndarray | None]:
        """Preprocess inputs before initializing state."""
        X, y = super()._preprocess_inputs(X, y=y, drop_nans=drop_nans)
        if y is not None:
            y = self._validator.check_and_cast_y_to_integer(y)
            y = jax.nn.one_hot(y, self._n_classes)
        return X, y

    # Note: necessary double decorator. The super().predict is decorated as well,
    # but the pynapple metadata would be dropped if we do not decorate here.
    # This happens because super().predict returns the log-proba which have the same
    # shape of one_hot(y), not matching the original y.shape.
    @support_pynapple(conv_type="jax")
    def predict(self, X: DESIGN_INPUT_TYPE) -> jnp.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X :
            The input samples. Can be an array of shape ``(n_samples, n_features)``
            or a ``FeaturePytree`` with arrays as leaves.

        Returns
        -------
        :
            Predicted class labels for each sample.
            Returns an integer array of shape  ``(n_samples, )`` with values in
            ``[0, n_classes - 1]``.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import nemos as nmo
        >>> X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        >>> y = jnp.array([0, 0, 1, 1])
        >>> model = nmo.glm.ClassifierGLM(n_classes=2).fit(X, y)
        >>> predictions = model.predict(X)
        >>> predictions.shape
        (4,)
        """
        # Below will raise if user set manually coef and intercept
        # and calls predict.
        # One could assume default labels 0,...,n-1
        # but requiring to be explicit is safer
        self._check_classes_is_set("predict")
        log_proba = super().predict(X)
        return self._decode_labels(jnp.argmax(log_proba, axis=-1))

    def predict_proba(
        self,
        X: DESIGN_INPUT_TYPE,
        return_type: Literal["log-proba", "proba"] = "log-proba",
    ) -> jnp.ndarray:
        """
        Predict class probabilities for samples in X.

        Parameters
        ----------
        X :
            The input samples. Can be an array of shape ``(n_samples, n_features)``
            or a ``FeaturePytree`` with arrays as leaves.
        return_type :
            The format of the returned probabilities. If ``"log-proba"``, returns
            log-probabilities. If ``"proba"``, returns probabilities. Defaults to
            ``"log-proba"``.

        Returns
        -------
        :
            Predicted class probabilities. Returns an array of shape ``(n_samples, n_classes)``
            where each row sums to 1 (for probabilities) or to 0 in log-space (for log-probabilities).

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import nemos as nmo
        >>> X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        >>> y = jnp.array([0, 0, 1, 1])
        >>> model = nmo.glm.ClassifierGLM(n_classes=2).fit(X, y)
        >>> proba = model.predict_proba(X, return_type="proba")
        >>> proba.shape
        (4, 2)
        """
        # Below will raise if user set manually coef and intercept
        # and calls predict without setting the class label mapping.
        # One could assume default labels 0,...,n-1
        # but requiring to be explicit makes the mapping between
        # the class labels and the probability index less ambiguous:
        #   `log_proba[:, i]` is the log-proba of class `self.classes_[i]`
        self._check_classes_is_set("predict_proba")
        # log-proba for categorical, proba for Bernoulli
        log_proba = super().predict(X)
        if return_type == "log-proba":
            return log_proba
        elif return_type == "proba":
            exp = support_pynapple(conv_type="jax")(jnp.exp)
            proba = exp(log_proba)
            # renormalize (sum to 1 constraint)
            proba /= proba.sum(axis=-1, keepdims=True)
            return proba
        else:
            raise ValueError(f"Unrecognized return type ``'{return_type}'``")

    def _estimate_resid_degrees_of_freedom(
        self, X: DESIGN_INPUT_TYPE, n_samples: Optional[int] = None
    ) -> jnp.ndarray:
        """
        Estimate the degrees of freedom of the residuals for categorical GLM.

        Parameters
        ----------
        X :
            The design matrix.
        n_samples :
            The number of samples observed. If not provided, n_samples is set to
            ``X.shape[0]``. If the fit is batched, n_samples could be larger than
            ``X.shape[0]``.

        Returns
        -------
        :
            An estimate of the degrees of freedom of the residuals.
        """
        # Convert a pytree to a design-matrix
        x_leaf = jax.tree_util.tree_leaves(X)

        if n_samples is None:
            n_samples = x_leaf[0].shape[0]
        else:
            if not isinstance(n_samples, int):
                raise TypeError(
                    f"`n_samples` must be `None` or of type `int`. "
                    f"Type {type(n_samples)} provided instead!"
                )

        n_features = sum(x.shape[1] for x in x_leaf)
        n_m1_classes = self._n_classes - 1
        params = self._get_model_params()

        # Infer n_neurons from coef shape:
        # ClassifierGLM: coef is (n_features, n_classes-1) -> n_neurons = 1
        # ClassifierPopulationGLM: coef is (n_features, n_neurons, n_classes-1) -> n_neurons = shape[1]
        coef_leaf = jax.tree_util.tree_leaves(params.coef)[0]
        n_neurons = 1 if coef_leaf.ndim == 2 else coef_leaf.shape[1]

        # For Lasso-type regularizers, use the non-zero coefficients as DOF estimate
        # see https://arxiv.org/abs/0712.0881
        if isinstance(self.regularizer, (GroupLasso, Lasso, ElasticNet)):
            # Sum over features (axis 0) and classes (axis -1)
            # This leaves shape (n_neurons,) for ClassifierPopulationGLM
            # or scalar for ClassifierGLM
            resid_dof = tree_utils.pytree_map_and_reduce(
                lambda x: ~jnp.isclose(x, jnp.zeros_like(x)),
                lambda x: sum([jnp.sum(i, axis=(0, -1)) for i in x]),
                params.coef,
            )
            return jnp.atleast_1d(n_samples - resid_dof - n_m1_classes)

        elif isinstance(self.regularizer, Ridge):
            # For Ridge, use total parameters
            return (n_samples - n_m1_classes * n_features - n_m1_classes) * jnp.ones(
                n_neurons
            )

        else:
            # For UnRegularized, use the rank
            rank = jnp.linalg.matrix_rank(jnp.concatenate(x_leaf, axis=1))
            return (n_samples - rank * n_m1_classes - n_m1_classes) * jnp.ones(
                n_neurons
            )

    def simulate(
        self,
        random_key: jax.Array,
        feedforward_input: DESIGN_INPUT_TYPE,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Simulate categorical responses from the model.

        Parameters
        ----------
        random_key :
            A JAX random key used to generate the simulated responses.
        feedforward_input :
            The input samples used to generate the responses. Can be an array of
            shape ``(n_samples, n_features)`` or a ``FeaturePytree`` with arrays
            as leaves.

        Returns
        -------
        :
            A tuple ``(y, log_prob)`` where:
            - ``y`` is an array of shape ``(n_samples,)`` containing the
              simulated class labels (in the same format as ``classes_``).
            - ``log_prob`` is an array of shape ``(n_samples,)`` containing the
              log-probability of the simulated responses under the model.

        Raises
        ------
        ValueError
            If ``classes_`` has not been set. Call :meth:`set_classes` or :meth:`fit`
            before calling this method.

        Examples
        --------
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import nemos as nmo
        >>> X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        >>> y = jnp.array([0, 0, 1, 1])
        >>> model = nmo.glm.ClassifierGLM(n_classes=2).fit(X, y)
        >>> key = jax.random.key(0)
        >>> simulated_y, log_prob = model.simulate(key, X)
        >>> simulated_y.shape
        (4,)
        """
        self._check_classes_is_set("simulate")
        y, log_prob = super().simulate(random_key, feedforward_input)
        argmax = support_pynapple(conv_type="jax")(lambda x: jnp.argmax(x, axis=-1))
        y = self._decode_labels(argmax(y))
        return y, log_prob

    def initialize_solver_and_state(
        self,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        init_params: UserProvidedParamsT,
    ) -> SolverState:
        """Initialize the solver and its state for running fit and update.

        This method must be called before using :meth:`update` for iterative optimization.
        It sets up the solver with the provided initial parameters and data.

        Parameters
        ----------
        X
            Input data, array of shape ``(n_time_bins, n_features)`` or pytree of same.
        y
            Target labels, array of shape ``(n_time_bins,)`` for single neuron/subject models or
            ``(n_time_bins, n_neurons)`` for population models.
        init_params
            Initial parameter tuple of (coefficients, intercept).

        Returns
        -------
        state
            Initial solver state.

        Raises
        ------
        ValueError
            If inputs or parameters have incompatible shapes or invalid values.
        """
        self.set_classes(y)
        y = self._encode_labels(y)
        return super().initialize_solver_and_state(X, y, init_params)

    def initialize_params(
        self,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
    ) -> UserProvidedParamsT:
        """
        Initialize model parameters for categorical GLM.

        Initialize coefficients with zeros and intercept by matching the mean class
        proportions. Class labels are automatically converted to one-hot encoding.

        Parameters
        ----------
        X :
            Input data, array of shape ``(n_time_bins, n_features)`` or pytree of same.
        y :
            Class labels as integers, array of shape ``(n_time_bins,)`` for single neuron
            models or ``(n_time_bins, n_neurons)`` for population models. Values should be
            in the range ``[0, n_classes - 1]``.

        Returns
        -------
        :
            Initial parameter tuple of (coefficients, intercept).

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import nemos as nmo
        >>> X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        >>> y = jnp.array([0, 0, 1, 1])
        >>> model = nmo.glm.ClassifierGLM(n_classes=2)
        >>> coef, intercept = model.initialize_params(X, y)
        >>> coef.shape
        (2, 1)
        """
        self.set_classes(y)
        y = self._encode_labels(y)
        y = self._validator.check_and_cast_y_to_integer(y)
        y = jax.nn.one_hot(y, self.n_classes)
        return super().initialize_params(X, y)

    def update(
        self,
        params: GLMUserParams,
        opt_state: SolverState,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        *args,
        n_samples: Optional[int] = None,
        **kwargs,
    ) -> StepResult:
        """
        Update the model parameters and solver state.

        Performs a single optimization step using the model's solver. Class labels
        are automatically converted to one-hot encoding before the update.

        **Important**: `y` will be converted to integers if floats are provided.
        For max performance provide an array of integers directly.

        Parameters
        ----------
        params :
            The current model parameters, typically a tuple of coefficients and intercepts.
        opt_state :
            The current state of the optimizer, encapsulating information necessary for the
            optimization algorithm to continue from the current state.
        X :
            The predictors used in the model fitting process. Shape ``(n_time_bins, n_features)``
            or a ``FeaturePytree``.
        y :
            Class labels as integers, array of shape ``(n_time_bins,)`` for single neuron
            models or ``(n_time_bins, n_neurons)`` for population models. Values should be
            in the range ``[0, n_classes - 1]``.
        *args :
            Additional positional arguments to be passed to the solver's update method.
        n_samples :
            The total number of samples. Usually larger than the samples of an individual batch,
            used to estimate the scale parameter of the GLM.
        **kwargs :
            Additional keyword arguments to be passed to the solver's update method.

        Returns
        -------
        params :
            Updated model parameters (coefficients, intercepts).
        state :
            Updated optimizer state.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import nemos as nmo
        >>> X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        >>> y = jnp.array([0, 0, 1, 1])
        >>> model = nmo.glm.ClassifierGLM(n_classes=2)
        >>> params = model.initialize_params(X, y)
        >>> opt_state = model.initialize_solver_and_state(X, y, params)
        >>> new_params, new_state = model.update(params, opt_state, X, y)
        """
        self._check_classes_is_set("update")
        y = self._encode_labels(y)
        # note: do not check and cast here. Risky but the performance of
        # the update has priority.
        y = jax.nn.one_hot(jnp.asarray(y, dtype=int), self._n_classes)
        return super().update(
            params, opt_state, X, y, *args, n_samples=n_samples, **kwargs
        )


class ClassifierGLM(ClassifierMixin, GLM):
    """
    Generalized Linear Model for multi-class classification.

    This model predicts discrete class labels from input features by modeling
    the log-odds of each class relative to a reference category.

    The model uses ``n_classes - 1`` sets of coefficients (one per non-reference class),
    resulting in coefficient shape ``(n_features, n_classes - 1)`` and intercept
    shape ``(n_classes - 1,)``.

    Parameters
    ----------
    n_classes
        The number of classes. Must be >= 2.
    inverse_link_function
        The inverse link function.
    regularizer
        The regularization scheme.
    regularizer_strength
        The strength of the regularization.
    solver_name
        The solver to use for optimization.
    solver_kwargs
        Additional keyword arguments for the solver.

    Attributes
    ----------
    coef_
        Fitted coefficients of shape ``(n_features, n_classes - 1)`` after calling :meth:`fit`.
    intercept_
        Fitted intercepts of shape ``(n_classes - 1,)`` after calling :meth:`fit`.

    Notes
    -----
    Class labels ``y`` should contain integer values in ``[0, n_classes - 1]``.
    Float arrays with integer values (e.g., ``[0.0, 1.0, 2.0]``) are accepted and
    converted automatically, but passing integer arrays directly is recommended
    for best performance.

    See Also
    --------
    ClassifierPopulationGLM : Multi-class classification for multiple neurons.
    GLM : Generalized Linear Model for continuous/count responses.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import nemos as nmo
    >>> # Binary classification
    >>> X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    >>> y = jnp.array([0, 0, 1, 1])  # Integer class labels
    >>> model = nmo.glm.ClassifierGLM(n_classes=2)
    >>> model = model.fit(X, y)
    >>> predictions = model.predict(X)  # Returns class labels
    >>> probabilities = model.predict_proba(X, return_type="proba")
    """

    _validator_class = ClassifierGLMValidator

    def __init__(
        self,
        n_classes: Optional[int] = 2,
        inverse_link_function: Optional[Callable] = None,
        regularizer: Optional[Union[str, Regularizer]] = None,
        regularizer_strength: Optional[RegularizerStrength] = None,
        solver_name: str = None,
        solver_kwargs: dict = None,
    ):
        self.n_classes = n_classes
        observation_model = obs.CategoricalObservations()
        super().__init__(
            observation_model=observation_model,
            inverse_link_function=inverse_link_function,
            regularizer=regularizer,
            regularizer_strength=regularizer_strength,
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
        )
        self._classes_ = None
        self._class_to_index_ = None
        self._skip_encoding = True  # default: assume labels are [0, n_classes-1]

    def fit(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: ArrayLike,
        init_params: Optional[GLMUserParams] = None,
    ):
        """
        Fit the model to training data.

        Parameters
        ----------
        X
            Training input samples of shape ``(n_samples, n_features)`` or FeaturePytree.
        y
            Target class labels of shape ``(n_samples,)``. Values should be in
            ``[0, n_classes - 1]``. Float arrays with integer values are
            accepted and converted automatically, but integer arrays are
            recommended for best performance.
        init_params
            Initial parameter values as tuple of ``(coef, intercept)``. If None,
            parameters are initialized automatically.

        Returns
        -------
        :
            The fitted model.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import nemos as nmo
        >>> X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        >>> y = jnp.array([0, 0, 1, 1])
        >>> model = nmo.glm.ClassifierGLM(n_classes=2)
        >>> model = model.fit(X, y)
        >>> model.coef_.shape
        (2, 1)
        """
        self.set_classes(y)
        y = self._encode_labels(y)
        return super().fit(X, y, init_params)

    def score(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: ArrayLike,
        score_type: Literal[
            "log-likelihood", "pseudo-r2-McFadden", "pseudo-r2-Cohen"
        ] = "log-likelihood",
        aggregate_sample_scores: Optional[Callable] = jnp.mean,
    ) -> jnp.ndarray:
        """
        Score the model on test data.

        Parameters
        ----------
        X
            Test input samples of shape ``(n_samples, n_features)`` or FeaturePytree.
        y
            True class labels of shape ``(n_samples,)``. Values should be in
            ``[0, n_classes - 1]``. Float arrays with integer values are
            accepted and converted automatically, but integer arrays are
            recommended for best performance.
        score_type
            The type of score to compute.
        aggregate_sample_scores
            Function to aggregate per-sample scores.

        Returns
        -------
        :
            The computed score.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import nemos as nmo
        >>> X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        >>> y = jnp.array([0, 0, 1, 1])
        >>> model = nmo.glm.ClassifierGLM(n_classes=2).fit(X, y)
        >>> score = model.score(X, y)
        """
        # check if classes are not set, aka user set the coef and intercept
        # manually, raise otherwise there may be ambiguity in interpreting
        # the labels.
        self._check_classes_is_set("score")
        y = self._encode_labels(y)
        return super().score(X, y, score_type, aggregate_sample_scores)


class ClassifierPopulationGLM(ClassifierMixin, PopulationGLM):
    """
    Population Generalized Linear Model for multi-class classification.

    This model predicts discrete class labels from input features by modeling
    the log-odds of each class relative to a reference category, for multiple
    neurons simultaneously.

    The model uses ``n_classes - 1`` sets of coefficients per neuron, resulting in
    coefficient shape ``(n_features, n_neurons, n_classes - 1)`` and intercept
    shape ``(n_neurons, n_classes - 1)``.

    Parameters
    ----------
    n_classes
        The number of classes. Must be >= 2.
    inverse_link_function
        The inverse link function.
    regularizer
        The regularization scheme.
    regularizer_strength
        The strength of the regularization.
    solver_name
        The solver to use for optimization.
    solver_kwargs
        Additional keyword arguments for the solver.
    feature_mask
        Mask indicating which features are used for each neuron.

    Attributes
    ----------
    coef_
        Fitted coefficients of shape ``(n_features, n_neurons, n_classes - 1)``
        after calling :meth:`fit`.
    intercept_
        Fitted intercepts of shape ``(n_neurons, n_classes - 1)`` after calling :meth:`fit`.

    Notes
    -----
    Class labels ``y`` should contain integer values in ``[0, n_classes - 1]``.
    Float arrays with integer values (e.g., ``[0.0, 1.0, 2.0]``) are accepted and
    converted automatically, but passing integer arrays directly is recommended
    for best performance.

    See Also
    --------
    ClassifierGLM : Multi-class classification for a single neuron.
    PopulationGLM : Population GLM for continuous/count responses.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import nemos as nmo
    >>> # Multi-class classification for 2 neurons
    >>> X = jnp.array([[1., 2.], [2., 3.], [3., 4.], [4., 5.], [5., 6.], [6., 7.]])
    >>> y = jnp.array([[0, 0], [0, 1], [1, 0], [1, 2], [2, 1], [2, 2]])
    >>> model = nmo.glm.ClassifierPopulationGLM(n_classes=3)
    >>> model = model.fit(X, y)
    >>> predictions = model.predict(X)  # Returns class labels, shape (n_samples, n_neurons)
    """

    _validator_class = PopulationClassifierGLMValidator

    def __init__(
        self,
        n_classes: Optional[int] = 2,
        inverse_link_function: Optional[Callable] = None,
        regularizer: Optional[Union[str, Regularizer]] = None,
        regularizer_strength: Optional[RegularizerStrength] = None,
        solver_name: str = None,
        solver_kwargs: dict = None,
        feature_mask: Optional[jnp.ndarray] = None,
    ):
        self.n_classes = n_classes
        observation_model = obs.CategoricalObservations()
        super().__init__(
            observation_model=observation_model,
            inverse_link_function=inverse_link_function,
            regularizer=regularizer,
            regularizer_strength=regularizer_strength,
            solver_name=solver_name,
            solver_kwargs=solver_kwargs,
            feature_mask=feature_mask,
        )
        self._classes_ = None
        self._class_to_index_ = None
        self._skip_encoding = True  # default: assume labels are [0, n_classes-1]

    @property
    def feature_mask(self) -> Union[jnp.ndarray, dict[str, jnp.ndarray]]:
        """
        Mask indicating which weights are used, matching the coefficients shape.

        The feature mask has the same structure and shape as the coefficients (``coef_``):

        - **Array input**: Shape ``(n_features, n_neurons, n_classes - 1)``.
          Each entry ``[i, j, k]`` indicates whether the weight for feature ``i``,
          neuron ``j``, and category ``k`` is used (1 = used, 0 = masked).

        - **Dict/FeaturePytree input**: A dict with keys matching ``coef_``.
          Each leaf array has the same shape as the corresponding coefficient leaf
          ``(n_features_per_key, n_neurons, n_classes - 1)``.

        Returns
        -------
        :
            The feature mask, or None if not set.
        """
        return self._feature_mask

    @feature_mask.setter
    def feature_mask(self, feature_mask: Union[DESIGN_INPUT_TYPE, dict]):
        # do not allow reassignment after fit
        if (self.coef_ is not None) and (self.intercept_ is not None):
            raise AttributeError(
                "property 'feature_mask' of 'populationGLM' cannot be set after fitting."
            )

        self._feature_mask = self._validator.validate_and_cast_feature_mask(
            feature_mask
        )

    def fit(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: ArrayLike,
        init_params: Optional[GLMUserParams] = None,
    ):
        """
        Fit the model to training data.

        Parameters
        ----------
        X
            Training input samples of shape ``(n_samples, n_features)`` or FeaturePytree.
        y
            Target class labels of shape ``(n_samples, n_neurons)``. Values should be in
            ``[0, n_classes - 1]``. Float arrays with integer values are
            accepted and converted automatically, but integer arrays are
            recommended for best performance.
        init_params
            Initial parameter values as tuple of ``(coef, intercept)``. If None,
            parameters are initialized automatically.

        Returns
        -------
        :
            The fitted model.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import nemos as nmo
        >>> X = jnp.array([[1., 2.], [2., 3.], [3., 4.], [4., 5.], [5., 6.], [6., 7.]])
        >>> y = jnp.array([[0, 0], [0, 1], [1, 0], [1, 2], [2, 1], [2, 2]])
        >>> model = nmo.glm.ClassifierPopulationGLM(n_classes=3)
        >>> model = model.fit(X, y)
        >>> model.coef_.shape
        (2, 2, 2)
        """
        self.set_classes(y)
        y = self._encode_labels(y)
        return super().fit(X, y, init_params)

    def score(
        self,
        X: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y: ArrayLike,
        score_type: Literal[
            "log-likelihood", "pseudo-r2-McFadden", "pseudo-r2-Cohen"
        ] = "log-likelihood",
        aggregate_sample_scores: Optional[Callable] = jnp.mean,
    ) -> jnp.ndarray:
        """
        Score the model on test data.

        Parameters
        ----------
        X
            Test input samples of shape ``(n_samples, n_features)`` or FeaturePytree.
        y
            True class labels of shape ``(n_samples, n_neurons)``. Values should be in
            ``[0, n_classes - 1]``. Float arrays with integer values are
            accepted and converted automatically, but integer arrays are
            recommended for best performance.
        score_type
            The type of score to compute.
        aggregate_sample_scores
            Function to aggregate per-sample scores.

        Returns
        -------
        :
            The computed score.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> import nemos as nmo
        >>> X = jnp.array([[1., 2.], [2., 3.], [3., 4.], [4., 5.], [5., 6.], [6., 7.]])
        >>> y = jnp.array([[0, 0], [0, 1], [1, 0], [1, 2], [2, 1], [2, 2]])
        >>> model = nmo.glm.ClassifierPopulationGLM(n_classes=3).fit(X, y)
        >>> score = model.score(X, y)
        """
        self._check_classes_is_set("score")
        y = self._encode_labels(y)
        return super().score(X, y, score_type, aggregate_sample_scores)
