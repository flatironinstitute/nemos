"""Base validator class for regressor models."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Generic, List, Optional, Tuple

import jax
import jax.numpy as jnp
from numpy.typing import DTypeLike

if TYPE_CHECKING:
    from pynapple import Tsd, TsdFrame

from . import utils
from .base_class import Base
from .tree_utils import pytree_map_and_reduce
from .type_casting import all_same_time_info, is_pynapple_tsd
from .typing import DESIGN_INPUT_TYPE, ModelParamsT, UserProvidedParamsT
from .validation import (
    check_tree_leaves_dimensionality,
    convert_tree_leaves_to_jax_array,
    error_all_invalid,
)


@dataclass(frozen=True)
class RegressorValidator(abc.ABC, Base, Generic[UserProvidedParamsT, ModelParamsT]):
    """
    Base class for validating regressor models' parameters, inputs, and consistency.

    This class provides a comprehensive validation framework for regression models,
    handling three types of validation:

    1. **Parameter Validation**: Transforms user-provided parameters (typically tuples
       of arrays) into validated model parameter objects with proper structure and type
       checking.

    2. **Input Validation**: Checks that input data (X, y) have the expected
       dimensionality and compatible sample sizes.

    3. **Consistency Validation**: Ensures model parameters are compatible with input
       data (e.g., matching feature counts, neuron counts).

    The parameter validation pipeline consists of five steps:
    1. check_user_params_structure: Validate the overall structure of user input
    2. convert_to_jax_arrays: Convert array-like objects to JAX arrays
    3. check_array_dimensions: Verify array dimensionality matches expectations
    4. cast_to_model_params: Transform validated input into model parameter structure
    5. additional_validation_model_params: Perform custom validation on the final parameter object

    Subclasses should:
    - Set `expected_array_dims` to specify required dimensionality for each parameter array
    - Set `X_dimensionality` and `y_dimensionality` for input validation
    - Set `to_model_params` to define the transformation function to model parameter structure
    - Set `model_class` to reference the associated model class (string for error messages)
    - Override `check_user_params_structure` to validate user-provided parameter structure
    - Override `additional_validation_model_params` to implement custom validation logic
    - Implement `validate_consistency` to check parameter/input compatibility

    Notes
    -----
    When subclassing, you MUST use type annotations on class attributes to override the
    default field values. Without type annotations, attributes become class attributes
    rather than instance fields, and the defaults from the parent class will be used.

    Example::

        # Correct - with type annotations:
        class MyValidator(RegressorValidator[UserParams, ModelParams]):
            expected_array_dims: Tuple[int, int] = (1, 1)  # Instance field
            X_dimensionality: int = 2  # Instance field
            y_dimensionality: int = 1  # Instance field
            model_class: str = "MyModel"  # Instance field

        # Incorrect - without type annotations:
        class MyValidator(RegressorValidator[UserParams, ModelParams]):
            expected_array_dims = (1, 1)  # Class attribute, won't override!
            model_class = "MyModel"  # Class attribute, won't override!

    Attributes
    ----------
    expected_array_dims :
        Expected dimensionality for each array in the user-provided parameters.
        Should match the structure of user input (e.g., (1, 1) for GLM coef and intercept).
    X_dimensionality :
        Expected number of dimensions for input X (e.g., 2 for shape (n_samples, n_features)).
    y_dimensionality :
        Expected number of dimensions for output y (e.g., 1 for shape (n_samples,)).
    to_model_params :
        Function to transform validated user parameters into model parameter structure.
    model_class :
        The model class name (string, used for error messages).
    params_validation_sequence :
        Names of parameter validation methods to call in order.
    """

    expected_param_dims: Tuple[int] = None
    model_class: str = None
    to_model_params: Callable[[UserProvidedParamsT], ModelParamsT] = None
    from_model_params: Callable[[ModelParamsT], UserProvidedParamsT] = None
    X_dimensionality: int = None
    y_dimensionality: int = None
    extra_params: Optional[dict] = None

    # tuples [(meth, kwargs), (meth,), ]
    params_validation_sequence: Tuple[
        Tuple[str, None] | Tuple[str, dict[str, Any]], ...
    ] = (
        ("check_user_params_structure", None),
        ("convert_to_jax_arrays", None),
        ("check_array_dimensions", None),
        ("to_model_params", None),
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
        **kwargs
            Additional keyword arguments (unused in base implementation).

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

    @staticmethod
    def wrap_user_params(params: UserProvidedParamsT) -> List:
        """Wrap user provided parameters into a list.

        Notes
        -----
        The output list should be of the same length as ``self.expected_array_dims``, one element
        per parameter. The user parameter structure is verified by ``self.check_user_params_structure``,
        if the structure is valid, the output list should have the expected length. May need re-implementation
        for complex parameter logic.

        """
        if hasattr(params, "ndim"):
            return [params]
        return list(params)

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
        **kwargs
            Additional keyword arguments (unused in base implementation).

        Returns
        -------
        UserProvidedParamsT
            The same parameters, validated for dimensionality.

        Raises
        ------
        ValueError
            If any array has unexpected dimensionality.
        """
        for par, exp_dim in zip(
            self.wrap_user_params(params), self.expected_param_dims
        ):
            dim_match = pytree_map_and_reduce(lambda x: x.ndim == exp_dim, all, par)
            is_empty = pytree_map_and_reduce(lambda x: x.size == 0, all, par)
            if not dim_match or is_empty:
                if err_msg is None:
                    provided_dims = jax.tree_util.tree_map(lambda x: x.ndim, params)
                    provided_dims_flat = tuple(jax.tree_util.tree_leaves(provided_dims))
                    err_msg = (
                        f"Unexpected array dimensionality for ``{self.model_class}`` parameters. "
                        f"Expected dimensions: {self.expected_param_dims}. "
                        f"Provided dimensions: {provided_dims_flat}"
                    )
                raise ValueError(err_msg)
        return params

    def convert_to_jax_arrays(
        self,
        params: UserProvidedParamsT,
        data_type: Optional[DTypeLike] = None,
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
        **kwargs
            Additional keyword arguments (unused in base implementation).

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
        if data_type is None:
            # default to float for model params.
            data_type = float
        return convert_tree_leaves_to_jax_array(
            params, err_message=err_msg, data_type=data_type
        )

    def validate_and_cast_params(
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

        for method_name, method_kwargs in self.params_validation_sequence:
            method_kwargs = {} if method_kwargs is None else method_kwargs
            # Merge default kwargs with any user-provided kwargs
            merged_kwargs = {**method_kwargs, **validation_kwargs}
            validated_params = getattr(self, method_name)(
                validated_params, **merged_kwargs
            )

        return validated_params

    def validate_inputs(
        self,
        X: Optional[DESIGN_INPUT_TYPE] = None,
        y: Optional[jnp.ndarray | Tsd | TsdFrame] = None,
    ):
        """
        Validate input data dimensions and sample consistency.

        Checks that X and y have the expected dimensionality (as specified by
        X_dimensionality and y_dimensionality) and that they have the same
        number of samples along axis 0.

        Parameters
        ----------
        X : DESIGN_INPUT_TYPE, optional
            Input features. Should have dimensionality matching X_dimensionality.
        y : jnp.ndarray, optional
            Output/target data. Should have dimensionality matching y_dimensionality.

        Raises
        ------
        ValueError
            If X or y don't have the expected dimensionality.
        ValueError
            If X and y have different number of samples along axis 0.
        ValueError
            If all samples are invalid (contain only NaN/Inf values).
        """
        # check same support
        if not all_same_time_info(X, y):
            raise ValueError(
                "Time axis mismatch. X and y pynapple objects have mismatching time axis."
            )

        check_vals = []
        if X is not None:
            if is_pynapple_tsd(X):
                X = X.values
            check_tree_leaves_dimensionality(
                X,
                expected_dim=self.X_dimensionality,
                err_message=f"X must be {self.X_dimensionality}-dimensional.",
            )
            check_vals.append(X)

        if y is not None:
            if is_pynapple_tsd(y):
                y = y.values
            check_tree_leaves_dimensionality(
                y,
                expected_dim=self.y_dimensionality,
                err_message=f"y must be {self.y_dimensionality}-dimensional.",
            )
            check_vals.append(y)

        if X is not None and y is not None:
            if y.shape[0] != X.shape[0]:
                raise ValueError(
                    "X and y must have the same number of samples (same length along axis 0). "
                    f"X has {X.shape[0]} samples, "
                    f"y has {y.shape[0]} samples instead!"
                )
        # error if all samples are invalid
        error_all_invalid(*check_vals)

    @abc.abstractmethod
    def validate_consistency(
        self,
        params: ModelParamsT,
        X: Optional[DESIGN_INPUT_TYPE] = None,
        y: Optional[jnp.ndarray] = None,
    ):
        """
        Validate consistency between model parameters and input data.

        This abstract method should be implemented by subclasses to check that
        model parameters are compatible with the provided input data. For example,
        checking that the number of features in parameters matches the number of
        features in X, or that the number of neurons in parameters matches the
        neuron dimension in y.

        Parameters
        ----------
        params : ModelParamsT
            Model parameters in their validated structure (e.g., GLMParams).
        X : DESIGN_INPUT_TYPE, optional
            Input features to validate against parameters.
        y : jnp.ndarray, optional
            Output/target data to validate against parameters.

        Raises
        ------
        ValueError
            If parameters and inputs are inconsistent (e.g., mismatched dimensions,
            incompatible structures).
        """
        pass

    def __repr__(self):
        """Small repr for the validator class."""
        return utils.format_repr(
            self, multiline=True, use_name_keys=["to_model_params", "from_model_params"]
        )

    @abc.abstractmethod
    def get_empty_params(self, X, y) -> ModelParamsT:
        """Return the param shape given the input data."""
        pass
