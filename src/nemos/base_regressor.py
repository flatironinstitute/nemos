"""Abstract class for regression models."""

# required to get ArrayLike to render correctly
from __future__ import annotations

import abc
from abc import abstractmethod
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import Any, NamedTuple, Optional, Tuple, Type, Union

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import solvers, utils, validation
from ._regularizer_builder import AVAILABLE_REGULARIZERS, instantiate_regularizer
from .base_class import Base
from .regularizer import Regularizer
from .solvers._abstract_solver import SolverState, StepResult
from .typing import (
    DESIGN_INPUT_TYPE,
    RegularizerStrength,
    SolverInit,
    SolverRun,
    SolverUpdate,
)
from .utils import _flatten_dict, _get_name, _unpack_params, get_env_metadata


def strip_metadata(arg_num: Optional[int] = None, kwarg_key: Optional[str] = None):
    """Strip metadata from arg."""
    if arg_num is None and kwarg_key is None:
        raise ValueError("Must specify either arg_num or kwarg_key.")

    def decorator(func):
        """Strip metadata if available."""

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            inp = args[arg_num] if arg_num is not None else kwargs[kwarg_key]
            self._metadata = {
                "metadata": inp._metadata if hasattr(inp, "_metadata") else None,
                "columns": inp.columns if hasattr(inp, "columns") else None,
            }
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


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
        E.g. stepsize, tol, acceleration, etc.
         For details on each solver's kwargs, see `get_accepted_arguments` and `get_solver_documentation`.

    See Also
    --------
    Concrete models:

    - [`GLM`](../glm/#nemos.glm.GLM): A feed-forward GLM implementation.
    - [`PopulationGLM`](../glm/#nemos.glm.PopulationGLM): A population GLM implementation.
    """

    def __init__(
        self,
        regularizer: Union[str, Regularizer] = "UnRegularized",
        regularizer_strength: Optional[RegularizerStrength] = None,
        solver_name: Optional[str] = None,
        solver_kwargs: Optional[dict] = None,
    ):
        self.regularizer = "UnRegularized" if regularizer is None else regularizer
        self.regularizer_strength = regularizer_strength

        # no solver name provided, use default
        if solver_name is None:
            self._solver_name = self.regularizer.default_solver
        else:
            self.solver_name = solver_name

        if solver_kwargs is None:
            solver_kwargs = dict()

        solver_class = solvers.solver_registry[self.solver_name]
        self._check_solver_kwargs(solver_class, solver_kwargs)

        self.solver_kwargs = solver_kwargs
        self._solver_init_state = None
        self._solver_update = None
        self._solver_run = None

    def __sklearn_tags__(self):
        """Return regression model specific estimator tags."""
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        tags.non_deterministic = True
        tags.requires_fit = True
        # conversion happens internally
        tags.array_api_support = True
        return tags

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
        if "regularizer" in params:
            # override _regularizer_strength to None to avoid conficts between regularizers
            self._regularizer_strength = None

            if "regularizer_strength" in params:
                # if both regularizer and regularizer_strength are set, then only
                # warn in case the strength is not expected for the regularizer type
                reg = params.pop("regularizer")
                super().set_params(regularizer=reg)

            elif self.regularizer_strength is not None:
                reg = params.pop("regularizer")
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
            self._regularizer = instantiate_regularizer(name=regularizer)
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
    def regularizer_strength(self) -> RegularizerStrength:
        """Regularizer strength getter."""
        return self._regularizer_strength

    @regularizer_strength.setter
    def regularizer_strength(self, strength: Union[None, RegularizerStrength]):
        # check regularizer strength
        strength = self.regularizer._validate_regularizer_strength(strength)
        self._regularizer_strength = strength

    @property
    def solver_name(self) -> str:
        """Getter for the solver_name attribute."""
        return self._solver_name

    @solver_name.setter
    def solver_name(self, solver_name: str):
        """Setter for the solver_name attribute."""
        # check if solver str passed is valid for regularizer
        self._regularizer.check_solver(solver_name)
        self._solver_name = solver_name

    @property
    def solver_kwargs(self):
        """Getter for the solver_kwargs attribute."""
        return self._solver_kwargs

    @solver_kwargs.setter
    def solver_kwargs(self, solver_kwargs: dict):
        """Setter for the solver_kwargs attribute."""
        if solver_kwargs:
            solver_cls = solvers.solver_registry[self.solver_name]
            self._check_solver_kwargs(solver_cls, solver_kwargs)
        self._solver_kwargs = solver_kwargs

    @staticmethod
    def _check_solver_kwargs(solver_class: Type, solver_kwargs: dict[str, Any]) -> None:
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
        accepted_args = solver_class.get_accepted_arguments()

        undefined_kwargs = set(solver_kwargs.keys()) - set(accepted_args)

        if undefined_kwargs:
            raise NameError(
                f"kwargs {undefined_kwargs} in solver_kwargs not a kwarg for {solver_class.__name__}!"
            )

    def instantiate_solver(self, solver_kwargs: Optional[dict] = None) -> BaseRegressor:
        """
        Instantiate the solver with the provided loss function.

        Instantiate the solver with the provided loss function, and store callable functions
        that initialize the solver state, update the model parameters, and run the optimization
        as attributes.

        This method creates a solver instance from the solver registry, tailored to
        the specific loss function and regularization approach defined by the Regularizer instance.
        It also handles the proximal operator if required for the optimization method. The returned
        functions are directly usable in optimization loops, simplifying the syntax by pre-setting
        common arguments like regularization strength and other hyperparameters.

        Solvers are expected to adhere to the `AbstractSolver` API.

        Parameters
        ----------
        solver_kwargs:
            Optional dictionary with the solver kwargs.
            If nothing is provided, it defaults to self.solver_kwargs.

        Returns
        -------
        :
            The instance itself for method chaining.
        """
        # final check that solver is valid for chosen regularizer
        self._regularizer.check_solver(self.solver_name)

        if solver_kwargs is None:
            # copy dictionary of kwargs to avoid modifying user settings
            solver_kwargs = deepcopy(self.solver_kwargs)

        # instantiate the solver
        solver_cls = solvers.solver_registry[self.solver_name]

        self._check_solver_kwargs(solver_cls, solver_kwargs)

        solver = solver_cls(
            self._predict_and_compute_loss,
            self.regularizer,
            self.regularizer_strength,
            **solver_kwargs,
        )
        self._solver = solver

        # nemos's solvers store a .fun attribute, but it's not necessary for a solver to work.
        # A test relies on having _solver_loss_fun saved, so still check and save it if possible.
        # But it's not a problem if .fun doesn't exist in user-defined solvers.
        if hasattr(solver, "fun"):
            # check that the loss is Callable
            utils.assert_is_callable(solver.fun, "solver's loss")
            self._solver_loss_fun = solver.fun

        self._solver_init_state = solver.init_state
        self._solver_update = solver.update
        self._solver_run = solver.run

        return self

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
        feedforward_input: DESIGN_INPUT_TYPE,
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
    ) -> StepResult:
        """Run a single update step of the underlying solver."""
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
        cast_to_jax_and_drop_nans: bool = True,
    ) -> SolverState:
        """Initialize the state of the solver for running fit and update."""
        pass

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
        (
            compute_defaults,
            compute_l_smooth,
            strong_convexity,
        ) = self._get_optimal_solver_params_config()
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

    @abstractmethod
    def save_params(
        self,
        filename: Union[str, Path],
        fit_attrs: dict,
        string_attrs: list = None,
    ):
        """
        Save model parameters and specified attributes to a .npz file.

        This is a private method intended to be used by subclasses to implement.
        Adds metadata about the jax and nemos versions used to save the model.

        Parameters
        ----------
        filename :
            The output filename.
        fit_attrs :
            Dictionary containing the fitting parameters specific to the subclass model.
        string_attrs :
            List of attributes to be saved as strings.
        """

        # extract model parameters
        model_params = self.get_params(deep=False)
        model_params = _unpack_params(model_params, string_attrs)

        # append the fit attributes to the model parameters
        model_params.update(fit_attrs)

        # set solver_kwargs to None so tha it can be saved in the npz
        if model_params["solver_kwargs"] == {}:
            model_params["solver_kwargs"] = None

        # save jax and nemos versions
        model_params["save_metadata"] = get_env_metadata()

        # save the model class name
        model_params["model_class"] = _get_name(self.__class__)

        # flatten the parameters dictionary to ensure it can be saved
        model_params = _flatten_dict(model_params)
        np.savez(filename, **model_params)

    def _get_fit_state(self) -> dict:
        """
        Collect all attributes that follow the fitted attribute convention.

        Collect all attributes ending with an underscore.

        Returns
        -------
        :
            A dictionary of attribute names and their values.
        """
        return {
            name: getattr(self, name)
            for name in dir(self)
            # sklearn has "_repr_html_" and "_repr_mimebundle_" methods
            # filter callables
            if name.endswith("_")
            and not name.endswith("__")
            and (not callable(getattr(self, name)))
        }
