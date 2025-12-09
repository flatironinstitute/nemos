"""Collection of methods utilities."""

import abc
import difflib
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Generic, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from numpy.typing import DTypeLike, NDArray

from . import utils
from .base_class import Base
from .pytrees import FeaturePytree
from .tree_utils import get_valid_multitree, pytree_map_and_reduce
from .typing import DESIGN_INPUT_TYPE, ModelParamsT, UserProvidedParamsT


def error_invalid_entry(*pytree: Any):
    """
    Raise an error if any entry in the provided pytrees contains NaN or Infinite (Inf) values.

    Parameters
    ----------
    *pytree : Any
        Variable number of pytrees to be checked for invalid entries. A pytree is defined as a nested structure
        of lists, tuples, dictionaries, or other containers, with leaves that are arrays.

    Raises
    ------
    ValueError
        If any NaN or Inf values are found in the provided pytrees.
    """
    any_infs = pytree_map_and_reduce(
        jnp.any, any, jax.tree_util.tree_map(jnp.isinf, pytree)
    )
    any_nans = pytree_map_and_reduce(
        jnp.any, any, jax.tree_util.tree_map(jnp.isnan, pytree)
    )
    if any_infs and any_nans:
        raise ValueError("The provided trees contain Infs and Nans!")
    elif any_infs:
        raise ValueError("The provided trees contain Infs!")
    elif any_nans:
        raise ValueError("The provided trees contain Nans!")


def error_all_invalid(*pytree: Any):
    """
    Raise an error if all sample points across multiple pytrees are invalid.

    This function checks multiple pytrees with NDArrays as leaves to determine if all sample points are invalid.
    A sample point is considered invalid if it contains NaN or infinite values in at least one of the pytrees.
    The sample axis is the first dimension of the NDArray.

    Parameters
    ----------
    pytree :
        Variable number of pytrees to be evaluated. Each pytree is expected to have NDArrays as leaves with a
        consistent size for the first dimension (sample dimension).
        The function checks for the validity of sample points across these pytrees.

    Raises
    ------
    ValueError
        If all sample points across the provided pytrees are invalid (i.e., contain NaN or infinite values).
    """
    if all(~get_valid_multitree(*pytree)):
        raise ValueError("At least a NaN or an Inf at all sample points!")


def check_length(x: Any, expected_len: int, err_message: str):
    """
    Check if the provided object has a length of two.

    Parameters
    ----------
    x :
        Object to check the length of.
    expected_len :
        The expected length of the object.
    err_message :
        Error message to raise if the length is not two.

    Raises
    ------
    ValueError
        If the object does not have the specified length.
    """
    try:
        assert len(x) == expected_len
    except Exception as e:
        raise ValueError(err_message) from e


def convert_tree_leaves_to_jax_array(
    pytree: Any, err_message: str, data_type: Optional[DTypeLike] = None
):
    """
    Convert the leaves of a given pytree to JAX arrays with the specified data type.

    Parameters
    ----------
    pytree :
        Pytree with leaves that are array-like objects.
    err_message:
        The error message to raise if the leaves do not have the specified data type.
    data_type :
        Data type to convert the leaves to.

    Raises
    ------
    TypeError
        If conversion to JAX arrays fails due to incompatible types.

    Returns
    -------
    :
        A tree of the same structure as the original, with leaves converted
        to JAX arrays.
    """
    try:
        pytree = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x, dtype=data_type), pytree
        )
    except (ValueError, TypeError) as e:
        raise TypeError(err_message) from e
    return pytree


def check_tree_leaves_dimensionality(pytree: Any, expected_dim: int, err_message: str):
    """
    Check if the leaves of the pytree have the specified dimensionality.

    Parameters
    ----------
    pytree :
        Pytree to check the dimensionality of its leaves.
    expected_dim :
        Expected dimensionality of the leaves.
    err_message :
        Error message to raise if the dimensionality does not match.

    Raises
    ------
    ValueError
        If any leaf does not match the expected dimensionality.
    """
    if pytree_map_and_reduce(lambda x: x.ndim != expected_dim, any, pytree):
        raise ValueError(err_message)


def check_same_shape_on_axis(*arrays: NDArray, axis: int = 0, err_message: str):
    """
    Check if the arrays have the same shape along a specified axis.

    Parameters
    ----------
    *arrays :
        Arrays to check shape consistency of.
    axis :
        Axis along which to check the shape consistency.
    err_message :
        Error message to raise if the shapes are inconsistent.

    Raises
    ------
    ValueError
        If the arrays do not have the same shape along the specified axis.
    """
    if len(arrays) > 1:
        if any(arr.shape[axis] != arrays[0].shape[axis] for arr in arrays[1:]):
            raise ValueError(err_message)


def check_array_shape_match_tree(
    pytree: Any, array_or_shape: NDArray | int, axis: int, err_message: str
):
    """
    Check if the shape of an array matches the shape of arrays in a pytree along a specified axis.

    Parameters
    ----------
    pytree :
        Pytree with arrays as leaves.
    array_or_shape :
        Array to compare the shape with or expected shape.
    axis :
        Axis along which to compare the shape.
    err_message :
        Error message to raise if the shapes do not match.

    Raises
    ------
    ValueError
        If the array's shape does not match the pytree leaves' shapes along the specified axis.
    """
    if isinstance(array_or_shape, int):
        _raise = pytree_map_and_reduce(
            lambda arr: arr.shape[axis] != array_or_shape, any, pytree
        )
    else:
        _raise = pytree_map_and_reduce(
            lambda arr: arr.shape[axis] != array_or_shape.shape[axis], any, pytree
        )
    if _raise:
        raise ValueError(err_message)


def array_axis_consistency(
    array_1: Union[FeaturePytree, jnp.ndarray, NDArray],
    array_2: Union[FeaturePytree, jnp.ndarray, NDArray],
    axis_1: int,
    axis_2: int,
):
    """
    Check if two arrays are consistent along specified axes.

    Parameters
    ----------
    array_1 :
        First array to check.
    array_2 :
        Second array to check.
    axis_1 :
        Axis to check in the first array.
    axis_2 :
        Axis to check in the second array.

    Returns
    -------
    bool
        True if inconsistent, otherwise False.
    """
    if array_1.shape[axis_1] != array_2.shape[axis_2]:
        return True
    else:
        return False


def check_tree_axis_consistency(
    pytree_1: Any,
    pytree_2: Any,
    axis_1: int,
    axis_2: int,
    err_message: str,
):
    """
    Check if two pytrees are consistent along specified axes for their respective leaves.

    Parameters
    ----------
    pytree_1 :
        First pytree to check.
    pytree_2 :
        Second pytree to check.
    axis_1 :
        Axis to check in the first pytree.
    axis_2 :
        Axis to check in the second pytree.
    err_message :
        Error message to raise if the pytrees' leaves are inconsistent along the given axes.

    Raises
    ------
    ValueError
        If the pytrees' leaves are inconsistent along the specified axes.
    """
    if pytree_map_and_reduce(
        lambda x, y: array_axis_consistency(x, y, axis_1, axis_2),
        any,
        pytree_1,
        pytree_2,
    ):
        raise ValueError(err_message)


def check_tree_structure(pytree_1: Any, pytree_2: Any, err_message: str):
    """Check if two pytrees have the same structure.

    Parameters
    ----------
    pytree_1 :
        First pytree to compare.
    pytree_2 :
        Second pytree to compare.
    err_message :
        Error message to raise if the structures of the pytrees do not match.

    Raises
    ------
    TypeError
        If the structures of the pytrees do not match.
    """
    if jax.tree_util.tree_structure(pytree_1) != jax.tree_util.tree_structure(pytree_2):
        raise TypeError(err_message)


def check_fraction_valid_samples(*pytree: Any, err_msg: str, warn_msg: str) -> None:
    """
    Check the fraction of entries that are not infinite or NaN.

    Parameters
    ----------
    *pytree :
        Trees containing arrays with the same sample axis.
    err_msg :
        The exception message.
    warn_msg :
        The warning message.

    Raises
    ------
    ValueError
        If all the samples contain invalid entries (either NaN or Inf).

    Warns
    -----
    UserWarning
        If more than 90% of the sample points contain NaNs or Infs.
    """
    valid = get_valid_multitree(pytree)
    if all(~valid):
        raise ValueError(err_msg)
    elif valid.mean() <= 0.1:
        warnings.warn(warn_msg, UserWarning)


def _warn_if_not_float64(feature_matrix: Any, message: str):
    """Warn if the feature matrix uses float32 precision."""
    all_float64 = pytree_map_and_reduce(
        lambda x: jnp.issubdtype(x.dtype, jnp.float64), all, feature_matrix
    )
    if not all_float64:
        warnings.warn(
            message,
            UserWarning,
        )


def _check_basis_matrix_shape(basis_matrix):
    basis_matrix = jnp.asarray(basis_matrix)
    if not utils.check_dimensionality(basis_matrix, 2):
        raise ValueError(
            "basis_matrix must be a 2 dimensional array! "
            f"{basis_matrix.ndim} dimensions provided instead."
        )
    if basis_matrix.shape[0] == 1:
        raise ValueError("`basis_matrix.shape[0]` should be at least 2!")
    return basis_matrix


def _check_non_empty_inputs(time_series, basis_matrix):
    utils.check_non_empty(basis_matrix, "basis_matrix")
    utils.check_non_empty(time_series, "time_series")


def _check_time_series_ndim(time_series, axis):
    if not utils.pytree_map_and_reduce(lambda x: x.ndim > axis, all, time_series):
        raise ValueError(
            "`time_series` should contain arrays of at least one-dimension. "
            "At least one 0-dimensional array provided."
        )


def _check_shift_causality_consistency(shift, predictor_causality):
    """Check shift causality consistency."""
    if shift and predictor_causality == "acausal":
        raise ValueError(
            "Cannot shift `predictor` when `predictor_causality` is `acausal`!"
        )


def _check_batch_size(batch_size, var_name):
    """Check if ``batch_size`` is a positive integer."""
    if batch_size is None:
        return
    elif not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(
            f"When provided ``{var_name}`` must be a strictly positive integer! "
            f"``{batch_size}`` provided instead."
        )


def _check_trials_longer_than_time_window(
    time_series: Any, window_size: int | Any, axis: int = 0
):
    """
    Check if the duration of each trial in the time series is at least as long as the window size.

    Parameters
    ----------
    time_series :
        A pytree of trial data.
    window_size :
        The size of the window to be used in convolution. Either an int or a pytree with the same
        struct as time_series.
    axis :
        The axis in the arrays representing the time dimension.

    Raises
    ------
    ValueError
        If any trial in the time series is shorter than the window size.
    """
    has_same_struct = jax.tree_util.tree_structure(
        time_series
    ) == jax.tree_util.tree_structure(window_size)
    if has_same_struct:
        insufficient_window_size = pytree_map_and_reduce(
            lambda x, w: x.shape[axis] < w, any, time_series, window_size
        )
    else:
        insufficient_window_size = pytree_map_and_reduce(
            lambda x: x.shape[axis] < window_size, any, time_series
        )
    # Check window size
    all_empty_or_valid = pytree_map_and_reduce(
        lambda x: x.shape[axis] == 0 or x.shape[axis] >= window_size, all, time_series
    )
    if insufficient_window_size and not all_empty_or_valid:
        warnings.warn(
            f"One or more trials are shorter than the convolution window size "
            f"({window_size} samples). These trials will produce NaN values in the output.",
            category=UserWarning,
            stacklevel=2,
        )


def _check_batch_size_larger_than_convolution_window(
    batch_size: int | Any, window_size: int | Any
):
    """Check if the batch_size is larger than the window size."""
    has_same_struct = jax.tree_util.tree_structure(
        batch_size
    ) == jax.tree_util.tree_structure(window_size)
    if has_same_struct:
        insufficient_window_size = pytree_map_and_reduce(
            lambda x, w: x < w, any, batch_size, window_size
        )
    else:
        insufficient_window_size = pytree_map_and_reduce(
            lambda x: x < window_size, any, batch_size
        )
    if insufficient_window_size:
        bs = jax.tree_util.tree_leaves(batch_size)[0]
        ws = jax.tree_util.tree_leaves(window_size)[0]
        raise ValueError(
            "Batch size too small. Batch size must be larger than the convolution window size. "
            f"The provided batch size is ``{bs}``, while the window size for the convolution is ``{ws}``. "
            "Please increase the batch size."
        )


def _suggest_keys(
    unmatched_keys: List[str], valid_keys: List[str], cutoff: float = 0.6
):
    """
    Suggest the closest matching valid key for each unmatched key using fuzzy string matching.

    This function compares each unmatched key to a list of valid keys and returns a suggestion
    if a close match is found based on the similarity score.

    Parameters
    ----------
    unmatched_keys :
        Keys that were provided by the user but not found in the expected set.
    valid_keys :
        The list of valid/expected keys to compare against.
    cutoff :
        The minimum similarity ratio (between 0 and 1) required to consider a match.
        A higher value means stricter matching. Defaults to 0.6.

    Returns
    -------
    :
        A list of (provided_key, suggested_key) pairs. If no match is found,
        `suggested_key` will be `None`.

    Examples
    --------
    >>> _suggest_keys(["observaton_model"], ["observation_model", "regularization"])
    [('observaton_model', 'observation_model')]
    """
    key_paris = []  # format, (user_provided, similar key)
    for unmatched_key in unmatched_keys:
        suggestions = difflib.get_close_matches(
            unmatched_key, valid_keys, n=1, cutoff=cutoff
        )
        key_paris.append((unmatched_key, suggestions[0] if suggestions else None))
    return key_paris


@dataclass(frozen=True)
class RegressorValidator(Base, Generic[UserProvidedParamsT, ModelParamsT]):
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

    # tuples [(meth, kwargs), (meth,), ]
    params_validation_sequence: Tuple[
        Tuple[str, None] | Tuple[str, dict[str, Any]], ...
    ] = (
        ("check_user_params_structure", None),
        ("convert_to_jax_arrays", None),
        ("check_array_dimensions", None),
        ("cast_to_model_params", None),
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
            if not dim_match:
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

    def cast_to_model_params(
        self,
        params: UserProvidedParamsT,
        **kwargs,
    ) -> ModelParamsT:
        """
        Transform validated user parameters into model parameter structure.

        Uses the `to_model_params` function to convert validated parameter arrays
        into the target model parameter object (e.g., GLMParams).

        This method assumes parameters have already been validated for structure
        and dimensionality, so it should not fail.

        Parameters
        ----------
        params : UserProvidedParamsT
            Validated user parameters as JAX arrays.
        **kwargs
            Additional keyword arguments (unused in base implementation).

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

        for method_name, method_kwargs in self.params_validation_sequence:
            method_kwargs = {} if method_kwargs is None else method_kwargs
            # Merge default kwargs with any user-provided kwargs
            merged_kwargs = {**method_kwargs, **validation_kwargs}
            validated_params = getattr(self, method_name)(
                validated_params, **merged_kwargs
            )

        return validated_params

    def validate_inputs(
        self, X: Optional[DESIGN_INPUT_TYPE] = None, y: Optional[jnp.ndarray] = None
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
        check_vals = []
        if X is not None:
            check_tree_leaves_dimensionality(
                X,
                expected_dim=self.X_dimensionality,
                err_message=f"X must be {self.X_dimensionality}-dimensional.",
            )
            check_vals.append(X)

        if y is not None:
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
            self, multiline=True, use_name_keys=["to_model_params"]
        )
