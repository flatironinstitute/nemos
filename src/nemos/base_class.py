"""Abstract class for estimators."""

# required to get ArrayLike to render correctly
from __future__ import annotations

import abc
import inspect
import warnings
from collections import defaultdict
from typing import Any, NamedTuple, Optional, Tuple, Union, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jaxopt import OptStep
from numpy.typing import ArrayLike, NDArray

from . import validation
from .pytrees import FeaturePytree
from ._regularizer_builder import create_regularizer

DESIGN_INPUT_TYPE = Union[jnp.ndarray, FeaturePytree]

if TYPE_CHECKING:
    from regularizer import Regularizer


class Base:
    """Base class for NeMoS estimators.

    A base class for estimators with utilities for getting and setting parameters,
    and for interacting with specific devices like CPU, GPU, and TPU.

    This class provides utilities for:
    - Getting and setting parameters using introspection.
    - Sending arrays to target devices (CPU, GPU, TPU).

    Notes
    -----
    The class provides helper methods mimicking scikit-learn's get_params and set_params.
    Additionally, it has methods for selecting target devices and sending arrays to them.
    """

    def get_params(self, deep=True):
        """
        From scikit-learn, get parameters by inspecting init.

        Parameters
        ----------
        deep

        Returns
        -------
            out:
                A dictionary containing the parameters. Key is the parameter
                name, value is the parameter value.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params: Any):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)
        nested_params: defaultdict = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            # TODO(1.4): remove specific handling of "base_estimator".
            # The "base_estimator" key is special. It was deprecated and
            # renamed to "estimator" for several estimators. This means we
            # need to translate it here and set sub-parameters on "estimator",
            # but only if the user did not explicitly set a value for
            # "base_estimator".
            if (
                    key == "base_estimator"
                    and valid_params[key] == "deprecated"
                    and self.__module__.startswith("sklearn.")
            ):
                warnings.warn(
                    (
                        f"Parameter 'base_estimator' of {self.__class__.__name__} is"
                        " deprecated in favor of 'estimator'. See"
                        f" {self.__class__.__name__}'s docstring for more details."
                    ),
                    FutureWarning,
                    stacklevel=2,
                )
                key = "estimator"
            valid_params[key].set_params(**sub_params)

        return self

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator."""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "GLM estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )

        # Consider the constructor parameters excluding 'self'
        parameters = [
            p.name for p in init_signature.parameters.values() if p.name != "self"
        ]

        # remove kwargs
        if "kwargs" in parameters:
            parameters.remove("kwargs")
        # Extract and sort argument names excluding 'self'
        return sorted(parameters)


class BaseRegressor(Base, abc.ABC):
    """Abstract base class for GLM regression models.

    This class encapsulates the common functionality for Generalized Linear Models (GLM)
    regression models. It provides an abstraction for fitting the model, making predictions,
    scoring the model, simulating responses, and preprocessing data. Concrete classes
    are expected to provide specific implementations of the abstract methods defined here.

    Parameters
    ----------
    regularizer :
        Regularization to use for model optimization. Defines the regularization scheme
        and related parameters.
        Default is UnRegularized regression.
    solver :
        Solver to use for model optimization. Defines the optimization scheme and related parameters.
        The solver must be an appropriate match for the chosen regularizer. Please see table below for
        regularizer/optimizer pairings.
        Default is `None`. If no solver specified, one will be chosen based on the regularizer.

    # TODO: insert nice table that shows the available optimizers for each regularizer

    See Also
    --------
    Concrete models:

    - [`GLM`](../glm/#nemos.glm.GLM): A feed-forward GLM implementation.
    - [`GLMRecurrent`](../glm/#nemos.glm.GLMRecurrent): A recurrent GLM implementation.
    """

    def __init__(
            self,
            regularizer: str | Regularizer = "unregularized",
            solver: str = None
    ):
        self._parse_regularizer_optimizer_params(regularizer=regularizer, solver=solver)

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

    def _parse_regularizer_optimizer_params(self, regularizer: str | Regularizer, solver: str):
        """Parse the regularizer and solver parameters."""
        # check if solver is in the regularizers allowed solvers
        if not isinstance(regularizer, str):
            # if no solver passed, use default for given regularizer
            if solver is None:
                self._solver = regularizer.default_solver
            # check if solver is in allowed solvers list
            if solver not in regularizer.allowed_solvers:
                raise ValueError(f"The solver: {solver} is not allowed for "
                                 f"{self._regularizer.__class__} regularizaration. Allowed solvers are "
                                 f"{self._regularizer.allowed_solvers}.")
            # store regularizer
            self._regularizer = regularizer
            # store solver
            self._solver = solver
        else:
            # try to instantiate solver
            self._regularizer = create_regularizer(name=regularizer)

            # if no solver passed, use default for given regularizer
            if solver is None:
                solver = self._regularizer.default_solver
            # check if solver str passed is valid for regularizer
            if solver not in self._regularizer.allowed_solvers:
                raise ValueError(f"The solver: {solver} is not allowed for "
                                 f"{self._regularizer.__class__} regularizaration. Allowed solvers are "
                                 f"{self._regularizer.allowed_solvers}.")
            # store solver
            self._solver = solver

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
    ) -> OptStep:
        """Run a single update step of the jaxopt solver."""
        pass

    @abc.abstractmethod
    def initialize_solver(
            self,
            X: DESIGN_INPUT_TYPE,
            y: jnp.ndarray,
            *args,
            params: Optional = None,
            **kwargs,
    ) -> Tuple[Any, NamedTuple]:
        """Initialize the solver's state and optionally sets initial model parameters for the optimization."""
        pass
