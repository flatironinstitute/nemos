"""Validation classes for GLM and PopulationGLM models."""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

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
class GLMValidator(validation.RegressorValidator[GLMUserParams, GLMParams]):
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
        feature_mask: Union[dict[str, jnp.ndarray], jnp.ndarray],
        data_type: Optional[DTypeLike] = None,
    ) -> Union[dict[str, jnp.ndarray], jnp.ndarray]:
        """
        Validate and cast a feature mask to JAX arrays.

        Validates that the feature mask contains only 0s and 1s, then converts
        it to JAX arrays with the specified data type. Subclasses can extend
        this to add parameter-specific validation (e.g., checking that mask
        shape matches parameter dimensions).

        Parameters
        ----------
        feature_mask : dict[str, jnp.ndarray] or jnp.ndarray
            Feature mask indicating which features are used. Must contain only 0s and 1s.
        data_type : jnp.dtype, optional
            Target data type for the mask arrays. Defaults to float.

        Returns
        -------
        dict[str, jnp.ndarray] or jnp.ndarray
            The validated and cast feature mask.

        Raises
        ------
        ValueError
            If feature_mask contains values other than 0 or 1.
        """
        if pytree_map_and_reduce(
            lambda x: jnp.any(jnp.logical_and(x != 0, x != 1)), any, feature_mask
        ):
            raise ValueError("'feature_mask' must contain only 0s and 1s!")

        # cast to jax - default to float if not specified
        if data_type is None:
            data_type = float
        feature_mask = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x, dtype=data_type), feature_mask
        )

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
