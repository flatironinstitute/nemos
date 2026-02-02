"""Validation classes for GLM and PopulationGLM models."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from numpy.typing import ArrayLike

from .. import validation
from ..tree_utils import pytree_map_and_reduce
from ..typing import DESIGN_INPUT_TYPE, FeaturePytree
from ..validation import RegressorValidator
from .params import GLMParams, GLMUserParams


def to_glm_params(user_params: GLMUserParams) -> GLMParams:
    """Map from GLMUserParams to GLMParams."""
    return GLMParams(*user_params)


def from_glm_params(params: GLMParams) -> GLMUserParams:
    """Map from GLMParams to GLMUserParams."""
    return params.coef, params.intercept


@dataclass(frozen=True, repr=False)
class GLMValidator(RegressorValidator[GLMUserParams, GLMParams]):
    """
    Validator for single-neuron GLM models.

    Validates and transforms user-provided parameters, inputs, and checks consistency
    between parameters and data for single-neuron GLMs. Single-neuron GLMs have:
    - 1D coefficients: shape (n_features,) or dict of (n_features,) arrays
    - 1D intercept: shape (1,)
    - 2D input X: shape (n_samples, n_features) or pytree of same
    - 1D output y: shape (n_samples,)

    This validator extends RegressorValidator with GLM-specific validation logic,
    including custom error messages and consistency checks between model parameters
    and input data.
    """

    extra_params: dict = None
    expected_param_dims: Tuple[int] = (
        1,
        1,
    )  # this should be (coef.ndim, intercept.ndim)
    to_model_params: Callable[[GLMUserParams], GLMParams] = to_glm_params
    from_model_params: Callable[[GLMParams], GLMUserParams] = from_glm_params
    model_class: str = "GLM"
    X_dimensionality: int = 2
    y_dimensionality: int = 1
    params_validation_sequence: Tuple[Tuple[str, None] | Tuple[str, dict[str, Any]]] = (
        *RegressorValidator.params_validation_sequence[:2],
        (
            "check_array_dimensions",
            dict(
                err_message_format="Invalid parameter dimensionality. coef must be an array "
                "or nemos.pytree.FeaturePytree with array leafs of shape "
                "(n_features, ). intercept must be of shape (1,)."
                "\nThe provided coef and intercept have shape ``{}`` and ``{}`` "
                "instead."
            ),
        ),
        *RegressorValidator.params_validation_sequence[3:],
        ("validate_intercept_shape", None),
    )

    def validate_intercept_shape(self, params: GLMParams, **kwargs):
        """
        Perform GLM-specific parameter validation.

        Validates that the intercept has the correct shape for a single-neuron GLM.

        Parameters
        ----------
        params : GLMParams
            GLM parameters with coef and intercept attributes.
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        GLMParams
            The validated parameters.

        Raises
        ------
        ValueError
            If intercept does not have shape (1,).
        """
        # check intercept shape
        if params.intercept.shape != (1,):
            raise ValueError(
                "Intercept term should be a one-dimensional array with shape ``(1,)``."
            )
        return params

    def check_array_dimensions(
        self,
        params: GLMUserParams,
        err_msg: Optional[str] = None,
        err_message_format: str = None,
    ) -> GLMUserParams:
        """
        Check array dimensions with custom error formatting for GLM parameters.

        Overrides the base implementation to provide GLM-specific error messages
        that include the actual shapes of the provided coefficient and intercept arrays.

        Parameters
        ----------
        params : GLMUserParams
            User-provided parameters as a tuple (coef, intercept).
        err_msg : str, optional
            Custom error message (unused, overridden by err_message_format).
        err_message_format : str, optional
            Format string for error message that takes two shape arguments.

        Returns
        -------
        GLMUserParams
            The validated parameters.

        Raises
        ------
        ValueError
            If arrays have incorrect dimensionality.
        """
        wrapped = self.wrap_user_params(params)
        shapes = tuple(jax.tree_util.tree_map(lambda x: x.shape, p) for p in wrapped)
        err_msg = err_message_format.format(*shapes)
        return super().check_array_dimensions(params, err_msg=err_msg)

    def check_user_params_structure(
        self, params: GLMUserParams, **kwargs
    ) -> GLMUserParams:
        """
        Validate that user parameters are a two-element structure.

        Parameters
        ----------
        params : GLMUserParams
            User-provided parameters (should be a tuple/list of length 2).
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        GLMUserParams
            The validated parameters.

        Raises
        ------
        ValueError
            If parameters do not have length two.
        """
        validation.check_length(params, 2, "Params must have length two.")
        if not isinstance(params, (tuple, list)):
            raise TypeError("GLM params must be a tuple/list of length two.")
        return params

    def validate_consistency(
        self,
        params: GLMParams,
        X: Optional[DESIGN_INPUT_TYPE] = None,
        y: Optional[jnp.ndarray] = None,
    ):
        """
        Validate consistency between parameters and inputs for single-neuron GLM.

        For single-neuron GLM, only validates feature consistency with X.
        Does not validate y since it's 1D (single neuron, no neuron axis to check).
        """
        if X is not None:
            # check that X and params.coef have the same structure
            msg = "X and coef have mismatched structure."
            if isinstance(X, FeaturePytree):
                data = X.data
                msg += (
                    " X was provided as a FeaturePytree, and coef should be a dictionary with matching keys. "
                    f"X keys are ``{X.keys()}``, the provided coef is {params.coef} instead."
                )
            else:
                data = X
                msg += (
                    f" X was provided as an array, and coef should be an array too. "
                    f"The provided coef is of type ``{type(params.coef)}`` instead."
                )

            validation.check_tree_structure(
                data,
                params.coef,
                err_message=msg,
            )
            # check the consistency of the feature axis
            validation.check_tree_axis_consistency(
                params.coef,
                data,
                axis_1=0,
                axis_2=1,
                err_message="Inconsistent number of features. "
                f"Model coefficients have {jax.tree_util.tree_map(lambda p: p.shape[0], params.coef)} features, "
                f"X has {jax.tree_util.tree_map(lambda x: x.shape[1], X)} features instead!",
            )

    @staticmethod
    def validate_and_cast_feature_mask(
        feature_mask: Union[DESIGN_INPUT_TYPE, dict[str, ArrayLike], ArrayLike, None],
        data_type: Optional[DTypeLike] = None,
    ) -> Union[dict[str, jnp.ndarray], jnp.ndarray, None]:
        """
        Validate and cast a feature mask to JAX arrays.

        Validates that the feature mask contains only 0s and 1s, then converts
        it to JAX arrays with the specified data type. Handles FeaturePytree
        inputs by extracting the underlying data. Subclasses can extend
        this to add parameter-specific validation (e.g., checking that mask
        shape matches parameter dimensions).

        Parameters
        ----------
        feature_mask :
            Feature mask indicating which features are used. Must contain only 0s and 1s.
            If a FeaturePytree, the underlying data dict is extracted.
        data_type :
            Target data type for the mask arrays. Defaults to float.

        Returns
        -------
        :
            The validated and cast feature mask, or None if input was None.

        Raises
        ------
        ValueError
            If feature_mask contains values other than 0 or 1.
        """
        if feature_mask is None:
            return None

        # Extract data from FeaturePytree
        if isinstance(feature_mask, FeaturePytree):
            feature_mask = feature_mask.data

        # Convert values to jnp.asarray
        feature_mask = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x, dtype=data_type), feature_mask
        )

        if pytree_map_and_reduce(
            lambda x: jnp.any(jnp.logical_and(x != 0, x != 1)), any, feature_mask
        ):
            raise ValueError("'feature_mask' must contain only 0s and 1s!")

        return feature_mask

    def feature_mask_consistency(
        self,
        feature_mask: Union[dict[str, jnp.ndarray], jnp.ndarray] | None,
        params: GLMParams,
    ):
        """Check consistency of feature_mask and params."""
        if feature_mask is None:
            return
        validation.check_tree_structure(
            params.coef,
            feature_mask,
            err_message=f"feature_mask and X must have the same structure, but feature_mask has structure  "
            f"{jax.tree_util.tree_structure(feature_mask)}, coef is of "
            f"{jax.tree_util.tree_structure(params.coef)} structure instead!",
        )

        if isinstance(params.coef, dict):
            # Note: in this case, the tree structure matching already takes care of the feature matching.
            # aka, same dict keys implies same feature masked. All we need to check is the match of
            # n_neurons.
            neural_axis = 0
            n_neurons = (
                1
                if self.y_dimensionality == 1
                else next(iter(params.coef.values())).shape[1]
            )
            shape_match = pytree_map_and_reduce(
                lambda fm: fm.shape == (n_neurons,), all, feature_mask
            )
            if not shape_match:
                raise ValueError(
                    "Inconsistent number of neurons. "
                    f"feature_mask has {jax.tree_util.tree_map(lambda m: m.shape[neural_axis], feature_mask)} neurons, "
                    f"model coefficients have {jax.tree_util.tree_map(lambda x: x.shape[1], params.coef)}  instead!",
                )
        else:
            shape_match = feature_mask.shape == params.coef.shape
            if not shape_match:
                raise ValueError(
                    "The shape of the ``feature_mask`` array must match that of the ``coef``. "
                    f"The shape of the ``coef`` is ``{params.coef.shape}``, "
                    f"that of the ``feature_mask`` is ``{feature_mask.shape}`` instead!"
                )
        return

    def get_empty_params(self, X, y) -> GLMParams:
        """Return the param shape given the input data."""
        empty_coef = jax.tree_util.tree_map(lambda x: jnp.empty((x.shape[1],)), X)
        empty_intercept = jnp.empty((1,))
        return to_glm_params((empty_coef, empty_intercept))


@dataclass(frozen=True, repr=False)
class PopulationGLMValidator(GLMValidator):
    """
    Validator for population (multi-neuron) GLM models.

    Validates and transforms user-provided parameters, inputs, and checks consistency
    between parameters and data for population GLMs. Population GLMs have:
    - 2D coefficients: shape (n_features, n_neurons) or dict of (n_features, n_neurons) arrays
    - 1D intercept: shape (n_neurons,)
    - 2D input X: shape (n_samples, n_features) or pytree of same
    - 2D output y: shape (n_samples, n_neurons)

    This validator extends GLMValidator with additional validation for the neuron dimension,
    checking that the number of neurons is consistent between model parameters and output y.
    """

    y_dimensionality: int = 2
    expected_param_dims: Tuple[int] = (
        2,
        1,
    )  # this should be (coef.ndim, intercept.ndim)
    model_class: str = "PopulationGLM"
    params_validation_sequence: Tuple[Tuple[str, None] | Tuple[str, dict[str, Any]]] = (
        *RegressorValidator.params_validation_sequence[:2],
        (
            "check_array_dimensions",
            dict(
                err_message_format="Invalid parameter dimensionality. "
                "coef must be an array or nemos.pytree.FeaturePytree "
                "with array leafs of shape (n_features, n_neurons). "
                "intercept must be of shape (n_neurons,)."
                "\nThe provided coef and intercept have shape ``{}`` and ``{}`` instead."
            ),
        ),
        *RegressorValidator.params_validation_sequence[3:],
    )

    def validate_consistency(
        self,
        params: GLMParams,
        X: Optional[DESIGN_INPUT_TYPE] = None,
        y: Optional[jnp.ndarray] = None,
    ):
        """
        Validate consistency between parameters and inputs for population GLM.

        For population GLM, validates both feature consistency with X and
        neuron count consistency with y (since y is 2D with shape (n_timebins, n_neurons)).
        """
        # First validate X consistency (features) using parent implementation
        super().validate_consistency(params, X=X, y=None)

        # Then validate y consistency (neurons) - specific to population GLM
        if y is not None:
            validation.check_array_shape_match_tree(
                params.coef,
                y,
                axis=1,
                err_message="Inconsistent number of neurons. "
                f"Model coefficients assume "
                f"{jax.tree_util.tree_map(lambda p: p.shape[1], params.coef)} neurons, "
                f"y has {jax.tree_util.tree_map(lambda x: x.shape[1], y)} neurons instead!",
            )

    def get_empty_params(self, X, y) -> GLMParams:
        """Return the param shape given the input data."""
        n_neurons = y.shape[1]
        empty_coef = jax.tree_util.tree_map(
            lambda x: jnp.empty((x.shape[1], n_neurons)), X
        )
        empty_intercept = jnp.empty((n_neurons,))
        return to_glm_params((empty_coef, empty_intercept))


@dataclass(frozen=True, repr=False)
class ClassifierGLMValidator(GLMValidator):
    """
    Validator for classifier GLM models.

    Validates and transforms user-provided parameters, inputs, and checks consistency
    between parameters and data for classifier GLMs. Classifier GLMs have:
    - 2D coefficients: shape (n_features, n_classes) or dict of (n_features, n_classes) arrays
    - 1D intercept: shape (n_classes,)
    - 2D input X: shape (n_samples, n_features) or pytree of same
    - 1D output y: shape (n_samples,) containing integer class labels
    """

    extra_params: Dict[Literal["n_classes"], int] = field(kw_only=True)
    expected_param_dims: Tuple[int] = (2, 1)
    model_class: str = "ClassifierGLM"
    params_validation_sequence: Tuple[Tuple[str, None] | Tuple[str, dict[str, Any]]] = (
        *RegressorValidator.params_validation_sequence[:2],
        (
            "check_array_dimensions",
            dict(
                err_message_format="Invalid parameter dimensionality. "
                "coef must be an array or nemos.pytree.FeaturePytree "
                "with array leafs of shape (n_features, n_classes). "
                "intercept must be of shape (n_classes,)."
                "\nThe provided coef and intercept have shape ``{}`` and ``{}`` instead."
            ),
        ),
        (
            "validate_n_classes_shape",
            dict(
                intercept_err_format="intercept must have shape ({},) for n_classes={}. "
                "Got intercept with shape {}."
            ),
        ),
        *RegressorValidator.params_validation_sequence[3:],
    )

    def validate_n_classes_shape(
        self,
        params: GLMUserParams,
        intercept_err_format: str = None,
        **kwargs,
    ) -> GLMUserParams:
        """
        Validate that coef and intercept last dimensions match n_classes.

        Parameters
        ----------
        params : GLMUserParams
            User-provided parameters as a tuple (coef, intercept).
        intercept_err_format : str
            Format string for intercept error message. Should have 3 placeholders:
            expected_class_dim, n_classes, actual_shape.
        """

        coef, intercept = params
        n_classes = self.extra_params["n_classes"]
        expected_class_dim = n_classes

        # Check coef last dimension
        coef_class_mismatch = pytree_map_and_reduce(
            lambda c: c.shape[-1] != expected_class_dim, any, coef
        )
        if coef_class_mismatch:
            coef_shapes = jax.tree_util.tree_map(lambda c: c.shape, coef)
            raise ValueError(
                f"coef last dimension must be n_classes = {expected_class_dim}. "
                f"Got coef with shape(s) {coef_shapes}."
            )

        # Check intercept last dimension
        if intercept.shape[-1] != expected_class_dim:
            raise ValueError(
                intercept_err_format.format(
                    expected_class_dim, n_classes, intercept.shape
                )
            )

        return params

    def validate_consistency(
        self,
        params: GLMParams,
        X: Optional[DESIGN_INPUT_TYPE] = None,
        y: Optional[jnp.ndarray] = None,
    ):
        """
        Validate consistency between parameters and inputs for classifier GLM.

        For classifier GLM, validates feature consistency with X.
        Does not validate y since it's 1D (single output, no neuron axis to check).
        """
        if X is not None:
            # check that X and params.coef have the same structure
            msg = "X and coef have mismatched structure. "
            if isinstance(X, FeaturePytree):
                data = X.data
                msg += (
                    " X was provided as a FeaturePytree, and coef should be a dictionary with matching keys. "
                    f"X keys are ``{X.keys()}``, the provided coef is {params.coef} instead."
                )
            else:
                data = X
                msg += (
                    f" X was provided as an array, and coef should be an array too. "
                    f"The coef are of type ``{type(params.coef)}`` instead."
                )

            validation.check_tree_structure(
                data,
                params.coef,
                err_message=msg,
            )
            # check the consistency of the feature axis
            # For classifier GLM: coef is (n_features, n_classes), X is (n_samples, n_features)
            validation.check_tree_axis_consistency(
                params.coef,
                data,
                axis_1=0,
                axis_2=1,
                err_message="Inconsistent number of features. "
                f"Model coefficients have {jax.tree_util.tree_map(lambda p: p.shape[0], params.coef)} features, "
                f"X has {jax.tree_util.tree_map(lambda x: x.shape[1], X)} features instead!",
            )

    @staticmethod
    def check_and_cast_y_to_integer(
        y: ArrayLike,
    ) -> jnp.ndarray:
        """Check that y is an array of integers.

        This method checks that the entries of y are all integers.
        If so, it converts the array to integer type.
        """
        y = jnp.asarray(y)
        if not jnp.issubdtype(y.dtype, jnp.integer):
            y_int = y.astype(int)
            if not jnp.all(y == y_int):
                raise ValueError("y must be an array of integers.")
        else:
            y_int = y
        return y_int

    def feature_mask_consistency(
        self,
        feature_mask: Union[dict[str, jnp.ndarray], jnp.ndarray] | None,
        params: GLMParams,
    ):
        """Check consistency of feature_mask and params for classifier GLM."""
        if feature_mask is None:
            return
        validation.check_tree_structure(
            params.coef,
            feature_mask,
            err_message=f"feature_mask and coef must have the same structure, but feature_mask has structure "
            f"{jax.tree_util.tree_structure(feature_mask)}, coef is of "
            f"{jax.tree_util.tree_structure(params.coef)} structure instead!",
        )

        shape_match = pytree_map_and_reduce(
            lambda fm, coef: fm.shape == coef.shape,
            all,
            feature_mask,
            params.coef,
        )

        if not shape_match:
            if isinstance(params.coef, jnp.ndarray):
                raise ValueError(
                    "The shape of the ``feature_mask`` array must match coef shape. "
                    f"Expected shape ``{params.coef.shape}``, "
                    f"got ``{feature_mask.shape}`` instead!"
                )
            else:
                raise ValueError(
                    "Inconsistent feature mask shape. "
                    f"feature_mask has shapes {jax.tree_util.tree_map(lambda m: m.shape, feature_mask)}, "
                    f"expected shapes {jax.tree_util.tree_map(lambda c: c.shape, params.coef)}!"
                )

    def get_empty_params(self, X, y) -> GLMParams:
        """Return the param shape given the input data."""
        n_classes = self.extra_params["n_classes"]
        empty_coef = jax.tree_util.tree_map(
            lambda x: jnp.empty((x.shape[1], n_classes)), X
        )
        empty_intercept = jnp.empty((n_classes,))
        return to_glm_params((empty_coef, empty_intercept))


@dataclass(frozen=True, repr=False)
class PopulationClassifierGLMValidator(ClassifierGLMValidator):
    """
    Validator for population (multi-neuron) classifier GLM models.

    Validates and transforms user-provided parameters, inputs, and checks consistency
    between parameters and data for population classifier GLMs. Population classifier GLMs have:
    - 3D coefficients: shape (n_features, n_neurons, n_classes) or dict of same
    - 2D intercept: shape (n_neurons, n_classes)
    - 2D input X: shape (n_samples, n_features) or pytree of same
    - 2D output y: shape (n_samples, n_neurons) containing integer class labels per neuron
    """

    y_dimensionality: int = 2
    expected_param_dims: Tuple[int] = (3, 2)  # coef is 3D, intercept is 2D
    model_class: str = "PopulationClassifierGLM"
    params_validation_sequence: Tuple[Tuple[str, None] | Tuple[str, dict[str, Any]]] = (
        *RegressorValidator.params_validation_sequence[:2],
        (
            "check_array_dimensions",
            dict(
                err_message_format="Invalid parameter dimensionality. "
                "coef must be an array or nemos.pytree.FeaturePytree "
                "with array leafs of shape (n_features, n_neurons, n_classes). "
                "intercept must be of shape (n_neurons, n_classes)."
                "\nThe provided coef and intercept have shape ``{}`` and ``{}`` instead."
            ),
        ),
        (
            "validate_n_classes_shape",
            dict(
                intercept_err_format="intercept last dimension must be n_classes = {} "
                "for n_classes={}. Got intercept with shape {}."
            ),
        ),
        *RegressorValidator.params_validation_sequence[3:],
    )

    def validate_consistency(
        self,
        params: GLMParams,
        X: Optional[DESIGN_INPUT_TYPE] = None,
        y: Optional[jnp.ndarray] = None,
    ):
        """
        Validate consistency between parameters and inputs for population classifier GLM.

        For population classifier GLM, validates both feature consistency with X and
        neuron count consistency with y.
        """
        # First validate X consistency (features) using parent implementation
        super().validate_consistency(params, X=X, y=None)

        # Then validate y consistency (neurons) - specific to population classifier GLM
        if y is not None:
            # coef shape is (n_features, n_neurons, n_classes), y shape is (n_samples, n_neurons)
            validation.check_array_shape_match_tree(
                params.coef,
                y,
                axis=1,
                err_message="Inconsistent number of neurons. "
                f"Model coefficients assume "
                f"{jax.tree_util.tree_map(lambda p: p.shape[1], params.coef)} neurons, "
                f"y has {jax.tree_util.tree_map(lambda x: x.shape[1], y)} neurons instead!",
            )

    def get_empty_params(self, X, y) -> GLMParams:
        """Return the param shape given the input data."""
        n_neurons = y.shape[1]
        n_classes = self.extra_params["n_classes"]
        empty_coef = jax.tree_util.tree_map(
            lambda x: jnp.empty((x.shape[1], n_neurons, n_classes)), X
        )
        empty_intercept = jnp.empty((n_neurons, n_classes))

        return to_glm_params((empty_coef, empty_intercept))
