"""Abstract class for regression models."""

# required to get ArrayLike to render correctly
from __future__ import annotations

import abc
import inspect
import warnings
from typing import Any, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jaxopt
from numpy.typing import ArrayLike, NDArray

from . import utils, validation
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
    | UnRegularized | GradientDescent  | GradientDescent, BFGS, LBFGS, NonlinearCG, ProximalGradient, LBFGSB |
    | Ridge         | GradientDescent  | GradientDescent, BFGS, LBFGS, NonlinearCG, ProximalGradient, LBFGSB |
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
    - [`GLMRecurrent`](../glm/#nemos.glm.GLMRecurrent): A recurrent GLM implementation.
    """

    def __init__(
        self,
        regularizer: Union[str, Regularizer] = "UnRegularized",
        regularizer_strength: Optional[float] = None,
        solver_name: str = None,
        solver_kwargs: Optional[dict] = None,
    ):
        self.regularizer = regularizer

        # check regularizer strength
        if regularizer_strength is None:
            warnings.warn(
                UserWarning(
                    "Caution: regularizer strength has not been set. Defaulting to 1.0. Please see "
                    "the documentation for best practices in setting regularization strength."
                )
            )
            self._regularizer_strength = 1.0
        else:
            self.regularizer_strength = regularizer_strength

        # no solver name provided, use default
        if solver_name is None:
            self.solver_name = self.regularizer.default_solver
        else:
            self.solver_name = solver_name

        if solver_kwargs is None:
            solver_kwargs = dict()
        self.solver_kwargs = solver_kwargs

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

        if isinstance(self._regularizer, UnRegularized):
            self._regularizer_strength = None

    @property
    def regularizer_strength(self) -> float:
        return self._regularizer_strength

    @regularizer_strength.setter
    def regularizer_strength(self, strength: float):
        try:
            # force conversion to float to prevent weird GPU issues
            strength = float(strength)
        except ValueError:
            # raise a more detailed ValueError
            raise ValueError(
                f"Could not convert the regularizer strength: {strength} to a float."
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
                f"{self._regularizer.__class__.__name__} regularizaration. Allowed solvers are "
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
        self._check_solver_kwargs(self.solver_name, solver_kwargs)
        self._solver_kwargs = solver_kwargs

    @staticmethod
    def _check_solver_kwargs(solver_name, solver_kwargs):
        """
        Check if provided solver keyword arguments are valid.

        Parameters
        ----------
        solver_name :
            Name of the solver.
        solver_kwargs :
            Additional keyword arguments for the solver.

        Raises
        ------
        NameError
            If any of the solver keyword arguments are not valid.
        """
        solver_args = inspect.getfullargspec(getattr(jaxopt, solver_name)).args
        undefined_kwargs = set(solver_kwargs.keys()).difference(solver_args)
        if undefined_kwargs:
            raise NameError(
                f"kwargs {undefined_kwargs} in solver_kwargs not a kwarg for jaxopt.{solver_name}!"
            )

    def instantiate_solver(
        self, *args, **kwargs
    ) -> Tuple[SolverInit, SolverUpdate, SolverRun]:
        """
        Instantiate the solver with the provided loss function.

        Instantiate the solver with the provided loss function, and return callable functions
        that initialize the solver state, update the model parameters, and run the optimization.

        This method creates a solver instance from jaxopt library, tailored to the specific loss
        function and regularization approach defined by the Regularizer instance. It also handles
        the proximal operator if required for the optimization method. The returned functions are
        directly usable in optimization loops, simplifying the syntax by pre-setting
        common arguments like regularization strength and other hyperparameters.

        Parameters
        ----------
        loss :
            The loss function to be optimized.

        *args:
            Positional arguments for the jaxopt `solver.run` method, e.g. the regularizing
            strength for proximal gradient methods.

        prox:
            Optional, the proximal projection operator.

        *kwargs:
            Keyword arguments for the jaxopt `solver.run` method.

        Returns
        -------
        :
            A tuple containing three callable functions:
            - solver_init_state: Function to initialize the solver's state, necessary before starting the optimization.
            - solver_update: Function to perform a single update step in the optimization process,
            returning new parameters and state.
            - solver_run: Function to execute the optimization process, applying multiple updates until a
            stopping criterion is met.
        """
        # final check that solver is valid for chosen regularizer
        if self.solver_name not in self.regularizer.allowed_solvers:
            raise ValueError(
                f"The solver: {self.solver_name} is not allowed for "
                f"{self._regularizer.__class__.__name__} regularizaration. Allowed solvers are "
                f"{self._regularizer.allowed_solvers}."
            )

        # only use penalized loss if not using proximal gradient descent
        if self.solver_name != "ProximalGradient":
            loss = self.regularizer.penalized_loss(
                self._predict_and_compute_loss, self.regularizer_strength
            )
        else:
            loss = self._predict_and_compute_loss

        # check that the loss is Callable
        utils.assert_is_callable(loss, "loss")

        # some parsing to make sure solver gets instantiated properly
        if self.solver_name == "ProximalGradient":
            self.solver_kwargs.update(prox=self.regularizer.get_proximal_operator())
            # add self.regularizer_strength to args
            args += (self.regularizer_strength,)

        # instantiate the solver
        solver = getattr(jaxopt, self._solver_name)(fun=loss, **self.solver_kwargs)

        def solver_run(
            init_params: Tuple[DESIGN_INPUT_TYPE, jnp.ndarray], *run_args: jnp.ndarray
        ) -> jaxopt.OptStep:
            return solver.run(init_params, *args, *run_args, **kwargs)

        def solver_update(params, state, *run_args, **run_kwargs) -> jaxopt.OptStep:
            return solver.update(
                params, state, *args, *run_args, **kwargs, **run_kwargs
            )

        def solver_init_state(params, state, *run_args, **run_kwargs) -> NamedTuple:
            return solver.init_state(
                params, state, *args, *run_args, **kwargs, **run_kwargs
            )

        return solver_init_state, solver_update, solver_run

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
        Validate the dimensions and consistency of parameters and data.

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
        """The loss function for a given model to be optimized over."""
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
