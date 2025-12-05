"""Abstract class for regression models."""

# required to get ArrayLike to render correctly
from __future__ import annotations

import abc
import warnings
from abc import abstractmethod
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import solvers, tree_utils, utils, validation
from ._regularizer_builder import AVAILABLE_REGULARIZERS, instantiate_regularizer
from .base_class import Base
from .regularizer import GroupLasso, Regularizer
from .tree_utils import pytree_map_and_reduce
from .type_casting import cast_to_jax
from .typing import (
    DESIGN_INPUT_TYPE,
    FeaturePytree,
    RegularizerStrength,
    SolverInit,
    SolverRun,
    SolverState,
    SolverUpdate,
    StepResult,
)
from .utils import _flatten_dict, _get_name, _unpack_params, get_env_metadata

_SOLVER_ARGS_CACHE = {}

ParamsT = TypeVar("ParamsT")
# User provided init_params (e.g. for GLMs Tuple[array, array])
UserProvidedParamsT = TypeVar("UserProvidedParamsT")
# Model internal representation (e.g. for GLMs nemos.glm.glm.GLMParams)
ModelParamsT = TypeVar("ModelParamsT")


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


class BaseRegressor(Base, abc.ABC, Generic[ParamsT]):
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

    def instantiate_solver(
        self, loss, solver_kwargs: Optional[dict] = None
    ) -> BaseRegressor:
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
        loss:
            The un-regularized loss function.
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
            loss,
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
    def _get_model_params(self) -> ParamsT:
        """Pack coef_ and intercept_  into a params pytree."""
        pass

    @abc.abstractmethod
    def _set_model_params(self, params: ParamsT):
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
    def compute_loss(self, params, X, y, *args, **kwargs):
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

    @cast_to_jax
    def initialize_params(
        self,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        init_params: Optional = None,
    ) -> ParamsT:
        """Initialize the solver's state and optionally sets initial model parameters for the optimization."""
        if init_params is None:
            init_params = self._initialize_parameters(X, y)  # initialize
        else:
            err_message = "Initial parameters must be array-like objects (or pytrees of array-like objects) "
            "with numeric data-type!"
            init_params = validation.convert_tree_leaves_to_jax_array(
                init_params, err_message=err_message, data_type=float
            )

        # validate input
        self._validate(X, y, init_params)

        return init_params

    @abc.abstractmethod
    def _initialize_parameters(
        self,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
    ) -> ParamsT:
        """Model specific initialization logic."""
        pass

    def _preprocess_inputs(
        self,
        X: DESIGN_INPUT_TYPE,
        y: jnp.ndarray,
        cast_to_jax_and_drop_nans: bool = True,
    ) -> Tuple[dict | jnp.ndarray, jnp.ndarray]:
        """Preprocess inputs before initializing state."""
        if cast_to_jax_and_drop_nans:
            X, y = cast_to_jax(tree_utils.drop_nans)(X, y)
            data = X.data if isinstance(X, FeaturePytree) else X
        else:
            data = X

        if isinstance(self.regularizer, GroupLasso):
            if self.regularizer.mask is None:
                warnings.warn(
                    "Mask has not been set. Defaulting to a single group for all parameters. "
                    "Please see the documentation on GroupLasso regularization for defining a mask."
                )
                self.regularizer.mask = jnp.ones((1, data.shape[1]))

        return X, y

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


class ParameterValidator[UserProvidedParamsT, ModelParamsT](eqx.Module):
    """
    Base class for validating and converting user-provided parameters to model parameters.

    This class provides a configurable validation pipeline that transforms user-provided
    parameters (typically simple structures like tuples of arrays) into validated model
    parameter objects with proper structure and type checking.

    The validation sequence consists of five steps:
    1. check_user_params_structure: Validate the overall structure of user input
    2. convert_to_jax_arrays: Convert array-like objects to JAX arrays
    3. check_array_dimensions: Verify array dimensionality matches expectations
    4. cast_to_model_params: Transform validated input into model parameter structure
    5. additional_validation_model_params: Perform custom validation on the final parameter object

    Subclasses should:
    - Set `expected_array_dims` to specify required dimensionality for each parameter array
    - Set `model_param_structure` to define the target pytree structure
    - Set `model_class` to reference the associated model class
    - Override `check_user_params_structure` to validate user-provided parameter structure
    - Override `additional_validation_model_params` to implement custom validation logic

    Attributes
    ----------
    expected_array_dims :
        Expected dimensionality for each array in the user-provided parameters.
        Should match the structure of user input (e.g., (2, 1) for GLM coef and intercept).
    to_model_params :
        Function to transform validated user parameters into model parameter structure.
    model_class :
        The model class these parameters belong to (used for error messages).
    validation_sequence :
        Names of validation methods to call in order.
    validation_sequence_kwargs :
        Keyword arguments for each validation method (None = no kwargs).
    """

    expected_array_dims: Tuple[int] = eqx.field(static=True, default=None)
    model_class: type = eqx.field(static=True, default=None)
    to_model_params: Callable[[UserProvidedParamsT], ModelParamsT] = eqx.field(
        static=True, default=None
    )
    validation_sequence: Tuple[str, ...] = eqx.field(
        static=True,
        default=(
            "check_user_params_structure",
            "convert_to_jax_arrays",
            "check_array_dimensions",
            "cast_to_model_params",
            "additional_validation_model_params",
        ),
    )
    validation_sequence_kwargs: Tuple[Optional[dict], ...] = eqx.field(
        static=True, default=(None, None, None, None, None)
    )

    @abc.abstractmethod
    def check_user_params_structure(
        self, params: UserProvidedParamsT, **kwargs
    ) -> UserProvidedParamsT:
        """
        Validate the structure of user-provided parameters.

        This method should verify that the user input has an acceptable structure
        (e.g., is an iterable of length 2 for GLM parameters). This is called first
        in the validation pipeline to provide early, clear error messages.

        This method should NOT check:
        - Array element types (handled by convert_to_jax_arrays)
        - Array dimensionality (handled by check_array_dimensions)
        - Parameter value constraints (handled by additional_validation_model_params)

        Parameters
        ----------
        params : UserProvidedParamsT
            User-provided parameters in their original format.

        Returns
        -------
        UserProvidedParamsT
            The same parameters, validated for structure.

        Raises
        ------
        ValueError
            If the parameter structure is invalid.
        """
        return params

    def check_array_dimensions(
        self,
        params: UserProvidedParamsT,
        err_msg: Optional[str] = None,
        **kwargs,
    ) -> UserProvidedParamsT:
        """
        Verify that all arrays have the expected dimensionality.

        Checks that each array in the parameter pytree has the dimensionality
        specified in `expected_array_dims`.

        Parameters
        ----------
        params : UserProvidedParamsT
            User-provided parameters (must contain JAX arrays at leaves).
        err_msg : str, optional
            Custom error message. If None, a default message is generated.

        Returns
        -------
        UserProvidedParamsT
            The same parameters, validated for dimensionality.

        Raises
        ------
        ValueError
            If any array has unexpected dimensionality.
        """
        if not pytree_map_and_reduce(
            lambda arr, expected_dim: arr.ndim == expected_dim,
            all,
            params,
            self.expected_array_dims,
        ):
            if err_msg is None:
                provided_dims = jax.tree_util.tree_map(lambda x: x.ndim, params)
                provided_dims_flat = tuple(jax.tree_util.tree_leaves(provided_dims))
                err_msg = (
                    f"Unexpected array dimensionality for {self.model_class.__name__} parameters. "
                    f"Expected dimensions: {self.expected_array_dims}. "
                    f"Provided dimensions: {provided_dims_flat}"
                )
            raise ValueError(err_msg)
        return params

    @classmethod
    def convert_to_jax_arrays(
        cls,
        params: UserProvidedParamsT,
        data_type: Optional[jax.dtypes] = None,
        err_msg: Optional[str] = None,
        **kwargs,
    ) -> UserProvidedParamsT:
        """
        Convert all array-like objects in parameters to JAX arrays.

        Parameters
        ----------
        params : UserProvidedParamsT
            User-provided parameters with array-like objects at leaves.
        data_type : jax.dtype, optional
            Target JAX dtype for arrays. If None, infers from input.
        err_msg : str, optional
            Custom error message for conversion failures.

        Returns
        -------
        UserProvidedParamsT
            Parameters with JAX arrays at all leaves.

        Raises
        ------
        TypeError
            If any leaf cannot be converted to a JAX array.
        """
        if err_msg is None:
            err_msg = (
                "Failed to convert parameters to JAX arrays. "
                "Parameters must be array-like objects (or pytrees of array-like objects) "
                "with numeric data types."
            )
        return validation.convert_tree_leaves_to_jax_array(
            params, err_message=err_msg, data_type=data_type
        )

    def cast_to_model_params(
        self,
        params: UserProvidedParamsT,
        model_param_structure: jax.tree_util.PyTreeDef,
        **kwargs,
    ) -> ModelParamsT:
        """
        Transform validated user parameters into model parameter structure.

        Uses `model_param_structure` to unflatten the validated parameter arrays
        into the target model parameter object (e.g., GLMParams).

        This method assumes parameters have already been validated for structure
        and dimensionality, so it should not fail.

        Parameters
        ----------
        params :
            Validated user parameters as JAX arrays.
        model_param_structure:
            Target model parameter structure.

        Returns
        -------
        ModelParamsT
            Parameters in model parameter structure.
        """
        return self.to_model_params(params)

    def validate_and_cast(
        self, params: UserProvidedParamsT, **validation_kwargs
    ) -> ModelParamsT:
        """
        Run the complete validation pipeline on user-provided parameters.

        Executes all validation steps in sequence, transforming user input
        into a validated model parameter object.

        Parameters
        ----------
        params : UserProvidedParamsT
            User-provided parameters to validate.
        **validation_kwargs
            Additional keyword arguments passed to validation methods.

        Returns
        -------
        ModelParamsT
            Validated and structured model parameters.

        Raises
        ------
        ValueError, TypeError
            If any validation step fails.
        """
        validated_params = params
        kwargs_sequence = [
            kwargs if kwargs is not None else {}
            for kwargs in self.validation_sequence_kwargs
        ]

        for method_name, method_kwargs in zip(
            self.validation_sequence, kwargs_sequence
        ):
            # Merge default kwargs with any user-provided kwargs
            merged_kwargs = {**method_kwargs, **validation_kwargs}
            validated_params = getattr(self, method_name)(
                validated_params, **merged_kwargs
            )

        return validated_params

    @abc.abstractmethod
    def additional_validation_model_params(
        self, params: ModelParamsT, **kwargs
    ) -> ModelParamsT:
        """
        Perform custom validation on model parameters.

        This method is called after parameters have been cast to the model
        parameter structure. It should implement model-specific validation
        logic (e.g., checking shape consistency between coefficients and intercepts).

        Since parameters are already in model structure, you can use attribute
        access for readable validation logic.

        Parameters
        ----------
        params : ModelParamsT
            Parameters in model structure (e.g., GLMParams with .coef and .intercept).

        Returns
        -------
        ModelParamsT
            The same parameters, validated for model-specific constraints.

        Raises
        ------
        ValueError
            If any model-specific validation check fails.

        Examples
        --------
        >>> def additional_validation_model_params(self, params: GLMParams) -> GLMParams:
        ...     n_features = params.coef.shape[0]
        ...     n_neurons = params.intercept.shape[0]
        ...     if params.coef.shape != (n_features, n_neurons):
        ...         raise ValueError("Coefficient shape mismatch")
        ...     return params
        """
        return params
