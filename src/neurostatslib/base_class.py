"""## Abstract class for estimators."""

import abc
import inspect
import warnings
from collections import defaultdict
from typing import Any, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax._src.lib import xla_client
from numpy.typing import ArrayLike, NDArray

from .utils import has_local_device, is_sequence


class _Base:
    """Base class for neurostatslib estimators.

    A base class for estimators with utilities for getting and setting parameters,
    and for interacting with specific devices like CPU, GPU, and TPU.

    This class provides utilities for:
    - Getting and setting parameters using introspection.
    - Sending arrays to target devices (CPU, GPU, TPU).

    Parameters
    ----------
    **kwargs : dict
        Arbitrary keyword arguments.

    Attributes
    ----------
    _kwargs_keys : list
        List of keyword arguments provided during the initialization.

    Notes
    -----
    The class provides helper methods mimicking scikit-learn's get_params and set_params.
    Additionally, it has methods for selecting target devices and sending arrays to them.
    """

    def __init__(self, **kwargs):
        self._kwargs_keys = list(kwargs.keys())
        for key in kwargs:
            setattr(self, key, kwargs[key])

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
        # add kwargs
        for key in self._kwargs_keys:
            out[key] = getattr(self, key)
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

    @staticmethod
    def select_target_device(device: Literal["cpu", "tpu", "gpu"]) -> xla_client.Device:
        """Select a device.

        Parameters
        ----------
        device
            A device between "cpu", "gpu" or "tpu". Rolls back to "cpu" if device is not found.

        Returns
        -------
            The selected device.

        Raises
        ------
            ValueError
                If the an invalid device name is provided.
        """
        if device in ["cpu", "gpu", "tpu"]:
            if has_local_device(device):
                target_device = jax.devices(device)[0]
            else:
                raise RuntimeError(
                    f"Unknown backend: '{device}' requested, but no "
                    f"platforms that are instances of {device} are present."
                )

        else:
            raise ValueError(
                f"Invalid device specification: {device}. Choose `cpu`, `gpu` or `tpu`."
            )
        return target_device

    def device_put(
        self, *args: jnp.ndarray, device: Literal["cpu", "tpu", "gpu"]
    ) -> Union[Any, jnp.ndarray]:
        """Send arrays to device.

        This function sends the arrays to the target device, if the arrays are
        not already there.

        Parameters
        ----------
        *args:
            NDArray
        device:
            A target device between "cpu", "tpu", "gpu".

        Returns
        -------
        :
            The arrays on the desired device.
        """
        device_obj = self.select_target_device(device)
        return tuple(
            jax.device_put(arg, device_obj)
            if arg.device_buffer.device() != device_obj
            else arg
            for arg in args
        )

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


class BaseRegressor(_Base, abc.ABC):
    """Abstract base class for GLM regression models.

    This class encapsulates the common functionality for Generalized Linear Models (GLM)
    regression models. It provides an abstraction for fitting the model, making predictions,
    scoring the model, simulating responses, and preprocessing data. Concrete classes
    are expected to provide specific implementations of the abstract methods defined here.

    Attributes
    ----------
    FLOAT_EPS : float
        A small float representing machine epsilon for float32, used to handle numerical
        stability issues.

    See Also
    --------
    Concrete models:

    - [`GLM`](../glm/#neurostatslib.glm.GLM): A feed-forward GLM implementation.
    - [`GLMRecurrent`](../glm/#neurostatslib.glm.GLMRecurrent): A recurrent GLM implementation.
    """

    FLOAT_EPS = jnp.finfo(jnp.float32).eps

    @abc.abstractmethod
    def fit(self, X: Union[NDArray, jnp.ndarray], y: Union[NDArray, jnp.ndarray]):
        """Fit the model to neural activity."""
        pass

    @abc.abstractmethod
    def predict(self, X: Union[NDArray, jnp.ndarray]) -> jnp.ndarray:
        """Predict rates based on fit parameters."""
        pass

    @abc.abstractmethod
    def score(
        self,
        X: Union[NDArray, jnp.ndarray],
        y: Union[NDArray, jnp.ndarray],
        # may include score_type or other additional model dependent kwargs
        **kwargs,
    ) -> jnp.ndarray:
        """Score the predicted firing rates (based on fit) to the target neural activity."""
        pass

    @abc.abstractmethod
    def simulate(
        self,
        random_key: jax.random.PRNGKeyArray,
        feed_forward_input: Union[NDArray, jnp.ndarray],
    ):
        """Simulate neural activity in response to a feed-forward input and recurrent activity."""
        pass

    @staticmethod
    def _convert_to_jnp_ndarray(
        *args: Union[NDArray, jnp.ndarray], data_type: Optional[jnp.dtype] = None
    ) -> Tuple[jnp.ndarray, ...]:
        """Convert provided arrays to jnp.ndarray of specified type.

        Parameters
        ----------
        *args :
            Input arrays to convert.
        data_type :
            Data type to convert to. Default is None, which means that the data-type
            is inferred from the input.

        Returns
        -------
        :
            Converted arrays.
        """
        return tuple(jnp.asarray(arg, dtype=data_type) for arg in args)

    @staticmethod
    def _has_invalid_entry(array: jnp.ndarray) -> bool:
        """Check if the array has nans or infs.

        Parameters
        ----------
        array:
            The array to be checked.

        Returns
        -------
            True if a nan or an inf is present, False otherwise

        """
        return (jnp.isinf(array) | jnp.isnan(array)).any()

    @staticmethod
    def _check_and_convert_params(
        params: Tuple[ArrayLike, ArrayLike], data_type: Optional[jnp.dtype] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Validate the dimensions and consistency of parameters and data.

        This function checks the consistency of shapes and dimensions for model
        parameters.
        It ensures that the parameters and data are compatible for the model.

        """
        if not is_sequence(params):
            raise TypeError("Initial parameters must be array-like!")

        if len(params) != 2:
            raise ValueError("Params needs to be array-like of length two.")

        try:
            params = jnp.asarray(params[0], dtype=data_type), jnp.asarray(
                params[1], dtype=data_type
            )
        except (ValueError, TypeError):
            raise TypeError(
                "Initial parameters must be array-like of array-like objects"
                "with numeric data-type!"
            )

        if params[0].ndim != 2:
            raise ValueError(
                "params[0] must be of shape (n_neurons, n_features), but"
                f"params[0] has {params[0].ndim} dimensions!"
            )
        if params[1].ndim != 1:
            raise ValueError(
                "params[1] must be of shape (n_neurons,) but "
                f"params[1] has {params[1].ndim} dimensions!"
            )
        return params

    @staticmethod
    def _check_input_dimensionality(
        X: Optional[jnp.ndarray] = None, y: Optional[jnp.ndarray] = None
    ):
        if not (y is None):
            if y.ndim != 2:
                raise ValueError(
                    "y must be two-dimensional, with shape (n_timebins, n_neurons)"
                )
        if not (X is None):
            if X.ndim != 3:
                raise ValueError(
                    "X must be three-dimensional, with shape (n_timebins, n_neurons, n_features)"
                )

    @staticmethod
    def _check_input_and_params_consistency(
        params: Tuple[jnp.ndarray, jnp.ndarray],
        X: Optional[jnp.ndarray] = None,
        y: Optional[jnp.ndarray] = None,
    ):
        """
        Validate the number of neurons in model parameters and input arguments.

        Raises
        ------
            ValueError
                - if the number of neurons is consistent across the model parameters (`params`) and
                any additional inputs (`X` or `y` when provided).
                - if the number of features is inconsistent between params[1] and X (when provided).

        """
        n_neurons = params[0].shape[0]
        if n_neurons != params[1].shape[0]:
            raise ValueError(
                "Model parameters have inconsistent shapes. "
                "Spike basis coefficients must be of shape (n_neurons, n_features), and "
                "bias terms must be of shape (n_neurons,) but n_neurons doesn't look the same in both! "
                f"Coefficients n_neurons: {params[0].shape[0]}, bias n_neurons: {params[1].shape[0]}"
            )

        if y is not None:
            if y.shape[1] != n_neurons:
                raise ValueError(
                    "The number of neurons in the model parameters and in the inputs"
                    "must match."
                    f"parameters has n_neurons: {n_neurons}, "
                    f"the input provided has n_neurons: {y.shape[1]}"
                )

        if X is not None:
            if X.shape[1] != n_neurons:
                raise ValueError(
                    "The number of neurons in the model parameters and in the inputs"
                    "must match."
                    f"parameters has n_neurons: {n_neurons}, "
                    f"the input provided has n_neurons: {X.shape[1]}"
                )
            if params[0].shape[1] != X.shape[2]:
                raise ValueError(
                    "Inconsistent number of features. "
                    f"spike basis coefficients has {params[0].shape[1]} features, "
                    f"X has {X.shape[2]} features instead!"
                )

    @staticmethod
    def _check_input_n_timepoints(X: jnp.ndarray, y: jnp.ndarray):
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "The number of time-points in X and y must agree. "
                f"X has {X.shape[0]} time-points, "
                f"y has {y.shape[0]} instead!"
            )

    def preprocess_fit(
        self,
        X: Union[NDArray, jnp.ndarray],
        y: Union[NDArray, jnp.ndarray],
        init_params: Optional[Tuple[ArrayLike, ArrayLike]] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Preprocess input data and initial parameters for the fit method.

        This method carries out the following preprocessing steps:

        - Convert to jax.numpy.ndarray

        - Check the dimensionality of the inputs.

        - Check for any NaNs or Infs in the inputs.

        - If `init_params` is not provided, initialize it with default values.

        - Validate the consistency of input dimensions with the initial parameters.

        Parameters
        ----------
        X :
            Input data, expected to be of shape (n_timebins, n_neurons, n_features).
        y :
            Target values, expected to be of shape (n_timebins, n_neurons).
        init_params :
            Initial parameters for the model. If None, they are initialized with default values.

        Returns
        -------
        X :
            Preprocessed input data `X` converted to jnp.ndarray.
        y :
            Target values `y` converted to jnp.ndarray.
        init_param :
            Initialized parameters converted to jnp.ndarray.

        Raises
        ------
        ValueError
            If there are inconsistencies in the input shapes or if NaNs or Infs are detected.
        """
        X, y = self._convert_to_jnp_ndarray(X, y)

        # check input dimensionality
        self._check_input_dimensionality(X, y)
        self._check_input_n_timepoints(X, y)

        if self._has_invalid_entry(X):
            raise ValueError("Input X contains a NaNs or Infs!")
        if self._has_invalid_entry(y):
            raise ValueError("Input y contains a NaNs or Infs!")

        _, n_neurons = y.shape
        n_features = X.shape[2]

        # Initialize parameters
        if init_params is None:
            # Ws, spike basis coeffs
            init_params = (
                jnp.zeros((n_neurons, n_features)),
                # bs, bias terms
                jnp.log(jnp.mean(y, axis=0)),
            )
        else:
            # check parameter length, shape and dimensionality, convert to jnp.ndarray.
            init_params = self._check_and_convert_params(init_params)

        # check that the inputs and the parameters has consistent sizes
        self._check_input_and_params_consistency(init_params, X=X, y=y)

        return X, y, init_params

    def preprocess_simulate(
        self,
        feedforward_input: Union[NDArray, jnp.ndarray],
        params_f: Tuple[jnp.ndarray, jnp.ndarray],
        init_y: Optional[Union[NDArray, jnp.ndarray]] = None,
        params_r: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    ) -> Tuple[jnp.ndarray, ...]:
        """
        Preprocess the input data and parameters for simulation.

        This method handles the conversion of the input data to `jnp.ndarray`, checks the
        input's dimensionality, and ensures the input's consistency with the provided parameters.
        It also verifies that the feedforward input does not have any invalid entries (NaNs or Infs).

        Parameters
        ----------
        feedforward_input :
            Input data for the feedforward process. Expected shape: (n_timesteps, n_neurons, n_basis_input).
        params_f :
            Parameters corresponding to the feedforward input. Expected shape: (n_neurons, n_basis_input).
        init_y :
            Initial values for the feedback process. If provided, its dimensionality and consistency
            with params_r will be checked. Expected shape if provided: (window_size, n_neurons).
        params_r :
            Parameters corresponding to the feedback input (init_y). Required if init_y is provided.
            Expected shape if provided: (window_size, n_basis_coupling)

        Returns
        -------
        :
            Preprocessed input data, optionally with the initial values for feedback if provided.

        Raises
        ------
        ValueError
            If the feedforward_input contains NaNs or Infs.
            If the dimensionality or consistency checks fail for the provided data and parameters.
        """
        (feedforward_input,) = self._convert_to_jnp_ndarray(feedforward_input)
        self._check_input_dimensionality(X=feedforward_input)
        self._check_input_and_params_consistency(params_f, X=feedforward_input)

        if self._has_invalid_entry(feedforward_input):
            raise ValueError("feedforward_input contains a NaNs or Infs!")

        # Ensure that both or neither of `init_y` and `params_r` are provided
        if (init_y is None) != (params_r is None):
            raise ValueError(
                "Both `init_y` and `params_r` should be provided, or neither should be provided."
            )
        # If both are provided, perform checks and conversions
        elif init_y is not None and params_r is not None:
            init_y = self._convert_to_jnp_ndarray(init_y)[
                0
            ]  # Assume this method returns a tuple
            self._check_input_dimensionality(y=init_y)
            self._check_input_and_params_consistency(params_r, y=init_y)
            return feedforward_input, init_y

        return (feedforward_input,)
