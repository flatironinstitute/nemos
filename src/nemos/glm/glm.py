"""GLM core module."""

# required to get ArrayLike to render correctly
from __future__ import annotations

import warnings
from typing import Any, Callable, Literal, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike
from sklearn.utils import TargetTags

from .. import observation_models as obs
from .. import tree_utils
from ..base_regressor import strip_metadata
from ..pytrees import FeaturePytree
from ..regularizer import ElasticNet, GroupLasso, Lasso, Regularizer, Ridge
from ..type_casting import cast_to_jax
from ..typing import DESIGN_INPUT_TYPE, SolverState, StepResult
from .base_glm import BaseGLM
from .initialize_parameters import initialize_intercept_matching_mean_rate
from .params import GLMParams, GLMUserParams
from .validation import (
    GLMValidator,
    PopulationGLMValidator,
)

__all__ = ["GLM", "PopulationGLM"]

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


class GLM(BaseGLM[GLMUserParams, GLMParams]):
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
    | Bernoulli           | :math:`1 / (1 + e^{-x})`        |
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
        Typically a float. Default is None. Sets the regularizer strength.
        If a user does not pass a value, and it is needed for regularization,
        a warning will be raised and the strength will default to 1.0.
        For finer control, the user can pass a pytree that matches the
        parameter structure to regularize parameters differentially.
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
    **Fit a GLM**

    Basic model fitting with default Poisson observation model:

    >>> import numpy as np
    >>> import nemos as nmo
    >>> np.random.seed(123)
    >>> X = np.random.normal(size=(100, 5))
    >>> y = np.random.poisson(size=100)
    >>> model = nmo.glm.GLM().fit(X, y)
    >>> model.coef_.shape
    (5,)

    **Customize the Observation Model**

    Specify the observation model as a string:

    >>> model = nmo.glm.GLM(observation_model="Gamma")
    >>> model.observation_model
    GammaObservations()

    Or pass the observation model object directly:

    >>> model = nmo.glm.GLM(observation_model=nmo.observation_models.GammaObservations())
    >>> model.observation_model
    GammaObservations()

    **Customize the Inverse Link Function**

    Use a soft-plus inverse link function instead of the default exponential:

    >>> model = nmo.glm.GLM(inverse_link_function=jax.nn.softplus)
    >>> model.inverse_link_function.__name__
    'softplus'

    **Use Regularization**

    Fit with Ridge regularization:

    >>> model = nmo.glm.GLM(regularizer="Ridge", regularizer_strength=0.1)
    >>> model = model.fit(X, y)
    >>> model.regularizer
    Ridge()

    Fit with Lasso regularization for sparse coefficients:

    >>> model = nmo.glm.GLM(regularizer="Lasso", regularizer_strength=0.01)
    >>> model = model.fit(X, y)
    >>> model.regularizer
    Lasso()

    **Select a Solver**

    Use LBFGS solver for potentially faster convergence:

    >>> model = nmo.glm.GLM(solver_name="LBFGS").fit(X, y)
    >>> model.solver_name
    'LBFGS'

    **Use a Pytree of arrays as Input**

    Features can be passed as any JAX pytree of 2-D arrays; the fitted
    ``coef_`` will share the same pytree structure:

    >>> X_dict = {"input_1": X[:, :2], "input_2": X[:, 2:]}
    >>> model = nmo.glm.GLM().fit(X_dict, y)
    >>> # The coefficient structure will match the input.
    >>> type(model.coef_)
    <class 'dict'>
    """

    _invalid_observation_types = (obs.CategoricalObservations,)
    _validator_class = GLMValidator

    @classmethod
    def _validate_observation_class(cls, observation: obs.Observations):
        """Raise TypeError with model-specific suggestions if observation type is invalid."""
        if observation.__class__ in cls._invalid_observation_types:
            model_name = cls.__name__
            obs_name = observation.__class__.__name__
            error_msg = (
                f"The ``{obs_name}`` observation type is not supported for "
                f"``{model_name}`` models."
            )
            is_categorical = isinstance(observation, obs.CategoricalObservations)
            if is_categorical:
                correct_model = (
                    "ClassifierPopulationGLM"
                    if issubclass(cls, PopulationGLM)
                    else "ClassifierGLM"
                )
                error_msg += (
                    f" To use a GLM for classification instantiate a "
                    f"``{correct_model}`` object."
                )
            else:
                correct_model = (
                    "PopulationGLM" if issubclass(cls, PopulationGLM) else "GLM"
                )
                error_msg += (
                    f" To use a GLM for regression with ``{obs_name}`` instantiate a "
                    f"``{correct_model}`` object."
                )
            raise TypeError(error_msg)

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

    def _model_specific_initialization(
        self,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
    ) -> GLMParams:
        """Initialize the parameters based on the structure and dimensions X and y.

        This method initializes the coefficients (spike basis coefficients) and intercepts (bias terms)
        required for the GLM. The coefficients are initialized to zeros with dimensions based on the input X.
        If X is a pytree of arrays, the coefficients retain the pytree structure with
        arrays of zeros shaped according to the features in X.
        If X is a simple ndarray, the coefficients are initialized as a 2D array. The intercepts are initialized
        based on the log mean of the target data y across the first axis, corresponding to the average log activity
        of the neuron.

        Parameters
        ----------
        X :
            The input data, either a pytree of arrays with leaves of shape
            ``(n_timebins, n_features)``, or a simple ndarray of shape ``(n_timebins, n_features)``.
        y :
            The target data array of shape ``(n_timebins, )``, representing
            the neuron firing rates or similar metrics.

        Returns
        -------
        Tuple[Union[pytree of arrays, jnp.ndarray], jnp.ndarray]
            A tuple containing the initialized parameters:
            - The first element is the initialized coefficients
            (either as a pytree of arrays or ndarray, matching the structure of X) with shapes (n_features,).
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

        self._validator.feature_mask_consistency(
            getattr(self, "_feature_mask", None), init_params
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

        self._initialize_optimizer_and_state(init_params, data, y)

        params, state, aux = self._optimizer_run(init_params, data, y)

        if tree_utils.pytree_map_and_reduce(
            lambda x: jnp.any(jnp.isnan(x)), any, params
        ):
            raise ValueError(
                "Solver returned at least one NaN parameter, so solution is invalid!"
                " Try tuning optimization hyperparameters, specifically try decreasing the `stepsize` "
                "and/or setting `acceleration=False`."
            )

        if hasattr(state, "stats") and hasattr(state.stats, "converged"):
            converged = state.stats.converged
        elif hasattr(state, "converged"):
            # try if the custom defined solver has a convergence flag directly
            converged = state.converged
        else:
            # custom solver with potentially undefined convergence state
            converged = True
            warnings.warn(
                f"Solver state {state} does not have a ``.converged`` nor a ``.stats.converged`` "
                f"attribute. Convergence state is unknown; assuming converged. "
                f"To assess the optimization manually, "
                f"inspect the ``solver_state_`` attribute of the model.",
                UserWarning,
            )
        if not converged:
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

    def _initialize_optimizer_and_state(
        self,
        init_params: GLMParams,
        X: dict[str, jnp.ndarray] | jnp.ndarray,
        y: jnp.ndarray,
    ) -> SolverState:
        """Initialize the solver by instantiating its init_state, update and, run methods.

        This method also prepares the solver's state by using the initialized model parameters and data.
        This setup is ready to be used for running the solver's optimization routines.

        Parameters
        ----------
        init_params :
            Initial parameters for the model.
        X :
            The predictors used in the model fitting process. This can include feature matrices or other structures
            compatible with the model's design.
        y :
            The response variables or outputs corresponding to the predictors. Used to initialize parameters when
            they are not provided.

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
        >>> opt_state = model.initialize_optimizer_and_state(params, X, y)
        >>> # Now ready to run optimization or update steps
        """
        opt_solver_kwargs = self._optimize_solver_params(X, y)
        #  set up the solver init/run/update attrs
        self._solver = self._instantiate_solver(
            self._compute_loss, init_params=init_params, solver_kwargs=opt_solver_kwargs
        )
        self._optimizer_init_state = self._solver.init_state
        self._optimizer_update = self._solver.update
        self._optimizer_run = self._solver.run
        opt_state = self._optimizer_init_state(init_params, X, y)
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
            or a pytree of arrays. Shape ``(n_time_bins, n_features)``.
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
        >>> opt_state = glm_instance.initialize_optimizer_and_state(params, X, y)
        >>> new_params, new_opt_state = glm_instance.update(params, opt_state, X, y)

        """
        # find non-nans
        X, y = tree_utils.drop_nans(X, y)

        # grab the data
        data = X.data if isinstance(X, FeaturePytree) else X

        # wrap into GLM params, this assumes params are well structured,
        # if initializaiton is done via `initialize_optimizer_and_state` it
        # should be fine
        params = self._validator.to_model_params(params)

        # perform a one-step update
        updated_params, updated_state, aux = self._optimizer_update(
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


class PopulationGLM(GLM):
    """
    Population Generalized Linear Model.

    This class implements a Generalized Linear Model for a neural population.
    This GLM implementation allows users to model the activity of a population of neurons based on a
    combination of exogenous inputs (like convolved currents or light intensities) and a choice of observation model.
    It is suitable for scenarios where the relationship between predictors and the response
    variable might be non-linear, and the residuals  don't follow a normal distribution. The predictors must be
    stored in tabular format, shape (n_timebins, num_features) or as a pytree of arrays of the same shape.
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
        Typically a float. Default is None. Sets the regularizer strength.
        If a user does not pass a value, and it is needed for regularization,
        a warning will be raised and the strength will default to 1.0.
        For finer control, the user can pass a pytree that matches the
        parameter structure to regularize parameters differentially.
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
        Either a matrix of shape (num_features, num_neurons) or a PyTree of 0s and 1s, with
        leaves of shape (num_neurons, ).
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
    **Fit a PopulationGLM**

    Basic model fitting for a population of neurons:

    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>> import nemos as nmo
    >>> np.random.seed(123)
    >>> num_samples, num_features, num_neurons = 100, 3, 2
    >>> X = np.random.normal(size=(num_samples, num_features))
    >>> weights = np.array([[0.5, 0.0], [-0.5, -0.5], [0.0, 1.0]])
    >>> y = np.random.poisson(np.exp(X.dot(weights)))
    >>> model = nmo.glm.PopulationGLM().fit(X, y)
    >>> model.coef_.shape
    (3, 2)

    **Mask Coefficients with an Array**

    Use a feature mask to specify which features predict each neuron.
    The mask has shape ``(num_features, num_neurons)``:

    >>> feature_mask = np.array([[1, 0], [1, 1], [0, 1]])
    >>> model = nmo.glm.PopulationGLM(feature_mask=feature_mask).fit(X, y)
    >>> model.coef_
    Array(...)

    **Use a Dict of Arrays as Input**

    Features can be passed as a dict (or any JAX pytree). The feature mask
    should mirror the same structure, with one 1-D entry per leaf:

    >>> feature_1 = np.random.normal(size=(num_samples, 2))
    >>> feature_2 = np.random.normal(size=(num_samples, 1))
    >>> X_dict = {"feature_1": feature_1, "feature_2": feature_2}
    >>> weights = dict(
    ...     feature_1=jnp.array([[0.0, 0.5], [0.0, -0.5]]),
    ...     feature_2=jnp.array([[1.0, 0.0]])
    ... )
    >>> rate = np.exp(
    ...     X_dict["feature_1"].dot(weights["feature_1"]) +
    ...     X_dict["feature_2"].dot(weights["feature_2"])
    ... )
    >>> y = np.random.poisson(rate)
    >>> feature_mask = {
    ...     "feature_1": jnp.array([0, 1], dtype=jnp.int32),
    ...     "feature_2": jnp.array([1, 0], dtype=jnp.int32)
    ... }
    >>> model = nmo.glm.PopulationGLM(feature_mask=feature_mask).fit(X_dict, y)
    >>> model.coef_
    {...}

    **Customize the Observation Model**

    Use a Gamma observation model for continuous positive data:

    >>> model = nmo.glm.PopulationGLM(observation_model="Gamma")
    >>> model.observation_model
    GammaObservations()

    **Use Regularization**

    Fit with Ridge regularization:

    >>> X = np.random.normal(size=(num_samples, num_features))
    >>> weights = np.array([[0.5, 0.0], [-0.5, -0.5], [0.0, 1.0]])
    >>> y = np.random.poisson(np.exp(X.dot(weights)))
    >>> model = nmo.glm.PopulationGLM(
    ...     regularizer="Ridge",
    ...     regularizer_strength=0.1
    ... ).fit(X, y)
    >>> model.regularizer
    Ridge()
    """

    _validator_class = PopulationGLMValidator

    def __init__(
        self,
        observation_model: (
            REGRESSION_GLM_TYPES
            | Literal["Poisson", "Gamma", "Gaussian", "Bernoulli", "NegativeBinomial"]
        ) = "Poisson",
        inverse_link_function: Optional[Callable] = None,
        regularizer: Union[str, Regularizer] = "UnRegularized",
        regularizer_strength: Any = None,
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

        - **Pytree**: A pytree with structure matching that of ``coef_``.
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

    @strip_metadata(arg_num=1, arg_name="y")
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
        an NDArray or a PyTree of 0s and 1s. In particular,

        - If the mask is in array format, feature ``i`` is a predictor for neuron ``j`` if
          ``feature_mask[i, j] == 1``.

        - If the mask is a PyTree, then
          a leaf is a predictor of neuron ``j`` if the matching leaf in ``feature_mask``
          is equal to 1.

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
        klass = self.__class__(**params)
        # reattach metadata
        klass._metadata = self._metadata
        return klass
