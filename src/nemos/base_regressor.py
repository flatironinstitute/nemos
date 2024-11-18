"""Abstract class for regression models."""

# required to get ArrayLike to render correctly
from __future__ import annotations

import abc
import inspect
import warnings
from abc import abstractmethod
from copy import deepcopy
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jaxopt
from numpy.typing import ArrayLike, NDArray

from . import solvers, utils, validation
from ._regularizer_builder import AVAILABLE_REGULARIZERS, create_regularizer
from .base_class import Base
from .regularizer import Regularizer, UnRegularized
from .typing import DESIGN_INPUT_TYPE, SolverInit, SolverRun, SolverUpdate


class BaseRegressor(Base, abc.ABC):
    """Abstract base class for GLM regression models.

    This class encapsulates the common functionality for Generalized Linear Models (GLM)
    regression models. It provides an abstraction for fitting the model, making predictions,
    scoring the model, simulating responses, and preprocessing data. Concrete classes
    are expected to provide specific implementations of the abstract methods defined here.
    Below is a table listing the default and available solvers for each regularizer.

    | Regularizer   | Default Solver   | Available Solvers                                           |
    | ------------- | ---------------- | ----------------------------------------------------------- |
    | UnRegularized | GradientDescent  | GradientDescent, BFGS, LBFGS, NonlinearCG, ProximalGradient |
    | Ridge         | GradientDescent  | GradientDescent, BFGS, LBFGS, NonlinearCG, ProximalGradient |
    | Lasso         | ProximalGradient | ProximalGradient                                            |
    | GroupLasso    | ProximalGradient | ProximalGradient                                            |

    Parameters
    ----------
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
        Default is `None`. If no solver specified, one will be chosen based on the regularizer.
        Please see table above for regularizer/optimizer pairings.
    solver_kwargs :
        Optional dictionary for keyword arguments that are passed to the solver when instantiated.
        E.g. stepsize, acceleration, value_and_grad, etc.
         See the jaxopt documentation for details on each solver's kwargs: https://jaxopt.github.io/stable/

    See Also
    --------
    Concrete models:

    - [`GLM`](../glm/#nemos.glm.GLM): A feed-forward GLM implementation.
    - [`PopulationGLM`](../glm/#nemos.glm.PopulationGLM): A population GLM implementation.
    """

    def __init__(
        self,
        regularizer: Union[str, Regularizer] = "UnRegularized",
        regularizer_strength: Optional[float] = None,
        solver_name: str = None,
        solver_kwargs: Optional[dict] = None,
    ):
        self.regularizer = regularizer
        self.regularizer_strength = regularizer_strength

        # no solver name provided, use default
        if solver_name is None:
            self.solver_name = self.regularizer.default_solver
        else:
            self.solver_name = solver_name

        if solver_kwargs is None:
            solver_kwargs = dict()
        self.solver_kwargs = solver_kwargs
        self._solver_init_state = None
        self._solver_update = None
        self._solver_run = None

    @property
    def solver_init_state(self) -> Union[None, SolverInit]:
        """
        Provides the initialization function for the solver's state.

        This function is responsible for initializing the solver's state, necessary for the start
        of the optimization process. It sets up initial values for parameters like gradients and step
        sizes based on the model configuration and input data.

        Returns
        -------
        :
            The function to initialize the state of the solver, if available; otherwise, None if
            the solver has not yet been instantiated.
        """
        return self._solver_init_state

    @property
    def solver_update(self) -> Union[None, SolverUpdate]:
        """
        Provides the function for updating the solver's state during the optimization process.

        This function is used to perform a single update step in the optimization process. It updates
        the model's parameters based on the current state, data, and gradients. It is typically used
        in scenarios where fine-grained control over each optimization step is necessary, such as in
        online learning or complex optimization scenarios.

        Returns
        -------
        :
            The function to update the solver's state, if available; otherwise, None if the solver
            has not yet been instantiated.
        """
        return self._solver_update

    @property
    def solver_run(self) -> Union[None, SolverRun]:
        """
        Provides the function to execute the solver's optimization process.

        This function runs the solver using the initialized parameters and state, performing the
        optimization to fit the model to the data. It iteratively updates the model parameters until
        a stopping criterion is met, such as convergence or exceeding a maximum number of iterations.

        Returns
        -------
        :
            The function to run the solver's optimization process, if available; otherwise, None if
            the solver has not yet been instantiated.
        """
        return self._solver_run

    def set_params(self, **params: Any):
        """Manage warnings in case of multiple parameter settings."""
        # if both regularizer and regularizer_strength are set, then only
        # warn in case the strength is not expected for the regularizer type
        if "regularizer" in params and "regularizer_strength" in params:
            reg = params.pop("regularizer")
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message="Caution: regularizer strength.*"
                    "|Unused parameter `regularizer_strength`.*",
                )
                super().set_params(regularizer=reg)

        return super().set_params(**params)

    @property
    def regularizer(self) -> Union[None, Regularizer]:
        """Getter for the regularizer attribute."""
        return self._regularizer

    @regularizer.setter
    def regularizer(self, regularizer: Union[str, Regularizer]):
        """Setter for the regularizer attribute."""
        # instantiate regularizer if str
        if isinstance(regularizer, str):
            self._regularizer = create_regularizer(name=regularizer)
        elif isinstance(regularizer, Regularizer):
            self._regularizer = regularizer
        else:
            raise TypeError(
                f"The regularizer should be either a string from "
                f"{AVAILABLE_REGULARIZERS} or an instance of `nemos.regularizer.Regularizer`"
            )

        # force check of regularizer_strength
        # need to use hasattr to avoid class instantiation issues
        if hasattr(self, "_regularizer_strength"):
            self.regularizer_strength = self._regularizer_strength

    @property
    def regularizer_strength(self) -> float:
        """Regularizer strength getter."""
        return self._regularizer_strength

    @regularizer_strength.setter
    def regularizer_strength(self, strength: Union[float, None]):
        # check regularizer strength
        if strength is None and not isinstance(self._regularizer, UnRegularized):
            warnings.warn(
                UserWarning(
                    "Caution: regularizer strength has not been set. Defaulting to 1.0. Please see "
                    "the documentation for best practices in setting regularization strength."
                )
            )
            strength = 1.0
        elif strength is not None:
            try:
                # force conversion to float to prevent weird GPU issues
                strength = float(strength)
            except ValueError:
                # raise a more detailed ValueError
                raise ValueError(
                    f"Could not convert the regularizer strength: {strength} to a float."
                )
            if isinstance(self._regularizer, UnRegularized):
                warnings.warn(
                    UserWarning(
                        "Unused parameter `regularizer_strength` for UnRegularized GLM. "
                        "The regularizer strength parameter is not required and won't be used when the regularizer "
                        "is set to UnRegularized."
                    )
                )

        self._regularizer_strength = strength

    @property
    def solver_name(self) -> str:
        """Getter for the solver_name attribute."""
        return self._solver_name

    @solver_name.setter
    def solver_name(self, solver_name: str):
        """Setter for the solver_name attribute."""
        # check if solver str passed is valid for regularizer
        if solver_name not in self._regularizer.allowed_solvers:
            raise ValueError(
                f"The solver: {solver_name} is not allowed for "
                f"{self._regularizer.__class__.__name__} regularization. Allowed solvers are "
                f"{self._regularizer.allowed_solvers}."
            )
        self._solver_name = solver_name

    @property
    def solver_kwargs(self):
        """Getter for the solver_kwargs attribute."""
        return self._solver_kwargs

    @solver_kwargs.setter
    def solver_kwargs(self, solver_kwargs: dict):
        """Setter for the solver_kwargs attribute."""
        self._check_solver_kwargs(
            self._get_solver_class(self.solver_name), solver_kwargs
        )
        self._solver_kwargs = solver_kwargs

    @staticmethod
    def _check_solver_kwargs(solver_class, solver_kwargs):
        """
        Check if provided solver keyword arguments are valid.

        Parameters
        ----------
        solver_class :
            Class of the solver.
        solver_kwargs :
            Additional keyword arguments for the solver.

        Raises
        ------
        NameError
            If any of the solver keyword arguments are not valid.
        """
        solver_args = inspect.getfullargspec(solver_class).args
        undefined_kwargs = set(solver_kwargs.keys()).difference(solver_args)
        if undefined_kwargs:
            raise NameError(
                f"kwargs {undefined_kwargs} in solver_kwargs not a kwarg for {solver_class.__name__}!"
            )

    def instantiate_solver(
        self, *args, solver_kwargs: Optional[dict] = None
    ) -> BaseRegressor:
        """
        Instantiate the solver with the provided loss function.

        Instantiate the solver with the provided loss function, and store callable functions
        that initialize the solver state, update the model parameters, and run the optimization
        as attributes.

        This method creates a solver instance from nemos.solvers or the jaxopt library, tailored to
        the specific loss function and regularization approach defined by the Regularizer instance.
        It also handles the proximal operator if required for the optimization method. The returned
        functions are directly usable in optimization loops, simplifying the syntax by pre-setting
        common arguments like regularization strength and other hyperparameters.

        Parameters
        ----------
        *args:
            Positional arguments for the jaxopt `solver.run` method, e.g. the regularizing
            strength for proximal gradient methods.
        solver_kwargs:
            Optional dictionary with the solver kwargs.
            If nothing is provided, it defaults to self.solver_kwargs.

        Returns
        -------
        :
            The instance itself for method chaining.
        """
        # final check that solver is valid for chosen regularizer
        if self.solver_name not in self.regularizer.allowed_solvers:
            raise ValueError(
                f"The solver: {self.solver_name} is not allowed for "
                f"{self._regularizer.__class__.__name__} regularization. Allowed solvers are "
                f"{self._regularizer.allowed_solvers}."
            )

        # only use penalized loss if not using proximal gradient descent
        # In proximal method you must use the unpenalized loss independently
        # of what regularizer you are using.
        if self.solver_name not in ("ProximalGradient", "ProxSVRG"):
            loss = self.regularizer.penalized_loss(
                self._predict_and_compute_loss, self.regularizer_strength
            )
        else:
            loss = self._predict_and_compute_loss

        if solver_kwargs is None:
            # copy dictionary of kwargs to avoid modifying user settings
            solver_kwargs = deepcopy(self.solver_kwargs)

        # check that the loss is Callable
        utils.assert_is_callable(loss, "loss")

        # some parsing to make sure solver gets instantiated properly
        if self.solver_name in ("ProximalGradient", "ProxSVRG"):
            if "prox" in self.solver_kwargs:
                raise ValueError(
                    "Proximal operator specification is not permitted. "
                    "The proximal operator is automatically determined based on the selected regularizer. "
                    "Please remove the 'prox' argument from the `solver_kwargs` "
                )

            solver_kwargs.update(prox=self.regularizer.get_proximal_operator())
            # add self.regularizer_strength to args
            args += (self.regularizer_strength,)

        (
            solver_run_kwargs,
            solver_init_state_kwargs,
            solver_update_kwargs,
            solver_init_kwargs,
        ) = self._inspect_solver_kwargs(solver_kwargs)

        # instantiate the solver
        solver = self._get_solver_class(self.solver_name)(
            fun=loss, **solver_init_kwargs
        )

        self._solver_loss_fun_ = loss

        def solver_run(
            init_params: Tuple[DESIGN_INPUT_TYPE, jnp.ndarray], *run_args: jnp.ndarray
        ) -> jaxopt.OptStep:
            return solver.run(init_params, *args, *run_args, **solver_run_kwargs)

        def solver_update(params, state, *run_args, **run_kwargs) -> jaxopt.OptStep:
            return solver.update(
                params, state, *args, *run_args, **solver_update_kwargs, **run_kwargs
            )

        def solver_init_state(params, *run_args, **run_kwargs) -> NamedTuple:
            return solver.init_state(
                params,
                *run_args,
                **run_kwargs,
                **solver_init_state_kwargs,
            )

        self._solver_init_state = solver_init_state
        self._solver_update = solver_update
        self._solver_run = solver_run
        return self

    def _inspect_solver_kwargs(
        self, solver_kwargs: dict
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Inspect and categorize the solver keyword arguments.

        This method inspects the provided `solver_kwargs` dictionary and categorizes
        the keyword arguments based on which solver functions they apply to:
        `run`, `init_state`, `update`, and `__init__`. This ensures that the
        appropriate arguments are passed to each function when the solver is used.

        Parameters
        ----------
        solver_kwargs :
            Dictionary containing keyword arguments for the solver.

        Returns
        -------
        :
            A tuple containing four dictionaries:
            - solver_run_kwargs: Arguments for the solver's `run` method.
            - solver_init_state_kwargs: Arguments for the solver's `init_state` method.
            - solver_update_kwargs: Arguments for the solver's `update` method.
            - solver_init_kwargs: Arguments for the solver's `__init__` constructor.
        """
        solver_run_kwargs = dict()
        solver_init_state_kwargs = dict()
        solver_update_kwargs = dict()
        solver_init_kwargs = dict()

        if solver_kwargs:
            # instantiate a solver to then inspect the params of its various functions
            solver = self._get_solver_class(self.solver_name)

            for key, value in solver_kwargs.items():
                if key in inspect.getfullargspec(solver.run).args:
                    solver_run_kwargs[key] = value
                if key in inspect.getfullargspec(solver.init_state).args:
                    solver_init_state_kwargs[key] = value
                if key in inspect.getfullargspec(solver.update).args:
                    solver_update_kwargs[key] = value
                if key in inspect.getfullargspec(solver.__init__).args:
                    solver_init_kwargs[key] = value

        return (
            solver_run_kwargs,
            solver_init_state_kwargs,
            solver_update_kwargs,
            solver_init_kwargs,
        )

    @abc.abstractmethod
    def fit(self, X: DESIGN_INPUT_TYPE, y: Union[NDArray, jnp.ndarray]):
        """Fit the model to neural activity."""
        pass

    @abc.abstractmethod
    def predict(self, X: DESIGN_INPUT_TYPE) -> jnp.ndarray:
        """Predict rates based on fit parameters."""
        pass

    @abc.abstractmethod
    def score(
        self,
        X: DESIGN_INPUT_TYPE,
        y: Union[NDArray, jnp.ndarray],
        # may include score_type or other additional model dependent kwargs
        **kwargs,
    ) -> jnp.ndarray:
        """Score the predicted firing rates (based on fit) to the target neural activity."""
        pass

    @abc.abstractmethod
    def simulate(
        self,
        random_key: jax.Array,
        feed_forward_input: DESIGN_INPUT_TYPE,
    ):
        """Simulate neural activity in response to a feed-forward input and recurrent activity."""
        pass

    @staticmethod
    @abc.abstractmethod
    def _check_params(
        params: Tuple[Union[DESIGN_INPUT_TYPE, ArrayLike], ArrayLike],
        data_type: Optional[jnp.dtype] = None,
    ) -> Tuple[DESIGN_INPUT_TYPE, jnp.ndarray]:
        """
        Validate the dimensions and consistency of parameters.

        This function checks the consistency of shapes and dimensions for model
        parameters.
        It ensures that the parameters and data are compatible for the model.

        """
        pass

    @staticmethod
    @abc.abstractmethod
    def _check_input_dimensionality(
        X: Optional[Union[DESIGN_INPUT_TYPE, jnp.ndarray]] = None,
        y: Optional[jnp.ndarray] = None,
    ):
        pass

    @abc.abstractmethod
    def _get_coef_and_intercept(self) -> Tuple[Any, Any]:
        """Pack coef_ and intercept_  into a params pytree."""
        pass

    @abc.abstractmethod
    def _set_coef_and_intercept(self, params: Any):
        """Unpack and store params pytree to coef_ and intercept_."""
        pass

    @staticmethod
    @abc.abstractmethod
    def _check_input_and_params_consistency(
        params: Tuple[Union[DESIGN_INPUT_TYPE, jnp.ndarray], jnp.ndarray],
        X: Optional[Union[DESIGN_INPUT_TYPE, jnp.ndarray]] = None,
        y: Optional[jnp.ndarray] = None,
    ):
        """Validate the number of features in model parameters and input arguments.

        Raises
        ------
        ValueError
            - if the number of features is inconsistent between params[1] and X
              (when provided).

        """
        pass

    @staticmethod
    def _check_input_n_timepoints(
        X: Union[DESIGN_INPUT_TYPE, jnp.ndarray], y: jnp.ndarray
    ):
        if y.shape[0] != X.shape[0]:
            raise ValueError(
                "The number of time-points in X and y must agree. "
                f"X has {X.shape[0]} time-points, "
                f"y has {y.shape[0]} instead!"
            )

    @abc.abstractmethod
    def _predict_and_compute_loss(self, params, X, y):
        """Loss function for a given model to be optimized over."""
        pass

    def _validate(
        self,
        X: Union[DESIGN_INPUT_TYPE, jnp.ndarray],
        y: Union[NDArray, jnp.ndarray],
        init_params: Tuple[DESIGN_INPUT_TYPE, jnp.ndarray],
    ):
        # check input dimensionality
        self._check_input_dimensionality(X, y)
        self._check_input_n_timepoints(X, y)

        # error if all samples are invalid
        validation.error_all_invalid(X, y)

        # validate input and params consistency
        init_params = self._check_params(init_params)

        # validate input and params consistency
        self._check_input_and_params_consistency(init_params, X=X, y=y)

    @abc.abstractmethod
    def update(
        self,
        params: Tuple[jnp.ndarray, jnp.ndarray],
        opt_state: NamedTuple,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        *args,
        **kwargs,
    ) -> jaxopt.OptStep:
        """Run a single update step of the jaxopt solver."""
        pass

    @abc.abstractmethod
    def initialize_params(
        self,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        params: Optional = None,
    ) -> Union[Any, NamedTuple]:
        """Initialize the solver's state and optionally sets initial model parameters for the optimization."""
        pass

    @abc.abstractmethod
    def initialize_state(
        self,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        init_params,
    ) -> Union[Any, NamedTuple]:
        """Initialize the state of the solver for running fit and update."""
        pass

    @staticmethod
    def _get_solver_class(solver_name: str):
        """
        Find a solver class first looking in nemos.solvers, then in jaxopt.

        Parameters
        ----------
        solver_name : str
            Name of the solver class to load.

        Returns
        -------
        solver_class :
            Solver class ready to be instantiated.

        Raises
        ------
        AttributeError
            If a solver class with that name is not found.
        """
        try:
            solver_class = getattr(solvers, solver_name)
        except AttributeError:
            try:
                solver_class = getattr(jaxopt, solver_name)
            except AttributeError:
                raise AttributeError(
                    f"Could not find {solver_name} in nemos.solvers or jaxopt"
                )

        return solver_class

    def _optimize_solver_params(self, X: DESIGN_INPUT_TYPE, y: jnp.ndarray) -> dict:
        """
        Compute and update solver parameters with optimal defaults if available.

        This method checks the current solver configuration and, if an optimal
        configuration is known for the given model parameters, computes the optimal
        batch size, step size, and other hyperparameters to ensure faster convergence.

        Parameters
        ----------
        X :
            Input data used to compute smoothness and strong convexity constants.
        y :
            Target values used in conjunction with X for the same purpose.

        Returns
        -------
        :
            A dictionary containing the solver parameters, updated with optimal defaults
            where applicable.

        """
        # Start with a copy of the existing solver parameters
        new_solver_kwargs = self.solver_kwargs.copy()

        # get the model specific configs
        compute_defaults, compute_l_smooth, strong_convexity = (
            self._get_optimal_solver_params_config()
        )
        if compute_defaults and compute_l_smooth:
            # Check if the user has provided batch size or stepsize, or else use None
            batch_size = new_solver_kwargs.get("batch_size", None)
            stepsize = new_solver_kwargs.get("stepsize", None)

            # Compute the optimal batch size and stepsize based on smoothness, strong convexity, etc.
            new_params = compute_defaults(
                compute_l_smooth,
                X,
                y,
                batch_size=batch_size,
                stepsize=stepsize,
                strong_convexity=strong_convexity,
            )

            # Update the solver parameters with the computed optimal values
            new_solver_kwargs.update(new_params)

        return new_solver_kwargs

    @abstractmethod
    def _get_optimal_solver_params_config(self):
        """Return the functions for computing default step and batch size for the solver."""
        pass
