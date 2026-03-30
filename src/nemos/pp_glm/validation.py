"""Validation classes for PPGLM and PopulationPPGLM models."""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from .. import validation
from ..pytrees import FeaturePytree
from ..tree_utils import pytree_map_and_reduce
from ..base_validator import RegressorValidator
from ..glm.validation import GLMValidator
from ..typing import DESIGN_INPUT_TYPE
from .params import PPGLMParams, PPGLMUserParams, PPGLMParamsWithKey

def to_pp_glm_params_with_key(params: PPGLMParams, random_key: jnp.array):
    """Map from PPGLMParams to PPGLMParamsWithKey by appending a jax random key.
    The key is converted from uint32 to float to avoid solver initialization error"""
    return PPGLMParamsWithKey(params, random_key.astype(float))

def to_pp_glm_params(user_params: PPGLMUserParams):
    """Map from PPGLMUserParams to PPGLMParams"""
    return PPGLMParams(*user_params)

def from_pp_glm_params(params: PPGLMParams) -> PPGLMUserParams:
    """Map from PPGLMParams to PPGLMUserParams."""
    return params.coef, params.intercept

@dataclass(frozen=True, repr=False)
class PPGLMValidator(RegressorValidator[PPGLMUserParams, PPGLMParams]):
    """
    Validator for a single-neuron PP-GLM models.

    Validates and transforms user-provided parameters, inputs, and checks consistency
    between parameters and data for single-neuron PP-GLMs. Single-neuron PP-GLMs have:
    - 1D coefficients: shape (n_features,) or dict of (n_features,) arrays
    - 1D intercept: shape (1, )
    - 2D input X: shape (2, n_events) or pytree of same
    - 2D output y: shape (3, n_events)

    """

    n_basis_funcs: int = field(kw_only=True)
    random_key: jnp.array = field(kw_only=True)
    expected_param_dims: Tuple[int] = (
        1,
        1,
    )  # this should be (coef.ndim, intercept.ndim)
    to_model_params: Callable[[PPGLMUserParams], PPGLMParams] = to_pp_glm_params
    from_model_params: Callable[[PPGLMParams], PPGLMUserParams] = from_pp_glm_params
    model_class: str = "PPGLM"
    X_dimensionality: int = 2
    y_dimensionality: int = 2
    _glm_validator: GLMValidator = GLMValidator()
    params_validation_sequence: Tuple[Tuple[str, None] | Tuple[str, dict[str, Any]]] = (
        *RegressorValidator.params_validation_sequence[:2],
        (
            "check_array_dimensions",
            dict(
                err_message_format="Invalid parameter dimensionality. coef must be an array "
                "or nemos.pytree.FeaturePytree with array leafs of shape "
                "(n_features, ). intercept must be of shape (1,)."
                "\nThe provided coef, intercept and random_key have shapes ``{}`` and ``{}`` "
                "instead."
            ),
        ),
        *RegressorValidator.params_validation_sequence[3:],
        ("validate_intercept_shape", None),
        ("validate_random_key", None),
    )

    def check_array_dimensions(
        self,
        params: PPGLMUserParams,
        err_msg: Optional[str] = None,
        err_message_format: str = None,
    ) -> PPGLMUserParams:
        """
        Check array dimensions with custom error formatting for GLM parameters.

        Overrides the base implementation to provide GLM-specific error messages
        that include the actual shapes of the provided coefficient and intercept arrays.

        Parameters
        ----------
        params : PPGLMUserParams
            User-provided parameters as a tuple (coef, intercept).
        err_msg : str, optional
            Custom error message (unused, overridden by err_message_format).
        err_message_format : str, optional
            Format string for error message that takes two shape arguments.

        Returns
        -------
        PPGLMUserParams
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

    def validate_random_key(self):
        """Validate that the provided random key is valid"""
        if jax.random.key_data(self.random_key.astype(jnp.uint32)).shape != 2:
            raise ValueError(
                "random key should be a one-dimensional array with shape ``(2,)``"
            )

    def validate_intercept_shape(self, params: PPGLMParams, **kwargs):
        """Validate that intercept has the correct shape"""
        # check intercept shape
        if params.intercept.shape != (1,):
            raise ValueError(
                "Intercept term should be a one-dimensional array with shape ``(1,)``."
            )
        return params

    def check_user_params_structure(
            self, params: PPGLMUserParams, **kwargs
    ) -> PPGLMUserParams:
        """
        Validate that user parameters are a three-element structure.

        Parameters
        ----------
        params : PPGLMUserParams
            User-provided parameters (should be a tuple/list of length 2).
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        PPGLMUserParams
            The validated parameters.

        Raises
        ------
        ValueError
            If parameters do not have length two.
        """
        validation.check_length(params, 3, "Params must have length three.")
        if not isinstance(params, (tuple, list)):
            raise TypeError("GLM params must be a tuple/list of length three.")
        return params

    def validate_consistency(
        self,
        params: PPGLMParams,
        X: Optional[DESIGN_INPUT_TYPE] = None,
        y: Optional[jnp.ndarray] = None,
    ):
        """
        Validate consistency between parameters and inputs for PP-GLM.
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
                    f" X has pytree structure ``{jax.tree_util.tree_structure(X)}``, "
                    f"but coef has structure ``{jax.tree_util.tree_structure(params.coef)}``."
                )

            validation.check_tree_structure(
                data,
                params.coef,
                err_message=msg,
            )

            n_features = jax.tree_util.tree_map(lambda p: p.shape[0], params.coef)
            n_groups_coef = int(n_features/self.n_basis_funcs)
            X_groups = jnp.unique(data[1])

            if n_groups_coef != X_groups.size:
                raise ValueError(
                    "Inconsistent number of features. "
                    f"Model coefficients assume {n_groups_coef} groups and {self.n_basis_funcs} basis functions, "
                    f"X has {X_groups.size} groups instead!"
                )

            # there might be a better way to do this
            if not jnp.allclose(jnp.arange(X_groups.size), X_groups):
                raise ValueError(
                    "Inconsistent feature labels. "
                    "Input feature labels must map to coefficient indices."
                )

    def validate_and_cast_feature_mask(
        self,
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
        return self._glm_validator.validate_and_cast_feature_mask(
            feature_mask, data_type=data_type
        )

    def feature_mask_consistency(
        self,
        feature_mask: Union[dict[str, jnp.ndarray], jnp.ndarray] | None,
        params: PPGLMParams,
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
            n_neurons = 1
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
class PopulationPPGLMValidator(PPGLMValidator):
    """
    Validator for population (multi-neuron) PP-GLM models.

    Validates and transforms user-provided parameters, inputs, and checks consistency
    between parameters and data for population PP-GLMs. Population PP-GLMs have:
    - 2D coefficients: shape (n_features, n_neurons) or dict of (n_features, n_neurons) arrays
    - 1D intercept: shape (n_neurons,)
    - 2D input X: shape (2, n_events) or pytree of same
    - 2D output y: shape (3, n_events)

    """
    expected_param_dims: Tuple[int] = (
        2,
        1,
    )  # this should be (coef.ndim, intercept.ndim)
    model_class: str = "PopulationPPGLM"
    params_validation_sequence: Tuple[Tuple[str, None] | Tuple[str, dict[str, Any]]] = (
        *RegressorValidator.params_validation_sequence[:2],
        (
            "check_array_dimensions",
            dict(
                err_message_format="Invalid parameter dimensionality. "
                "coef must be an array or pytree "
                "with array leaves of shape (n_features, n_neurons). "
                "intercept must be of shape (n_neurons,)."
                "\nThe provided coef, intercept and random_key have shapes ``{}`` and ``{}`` "
                "instead."
            ),
        ),
        *RegressorValidator.params_validation_sequence[3:],
        ("validate_random_key", None),
    )
    def validate_consistency(
        self,
        params: PPGLMParams,
        X: Optional[DESIGN_INPUT_TYPE] = None,
        y: Optional[jnp.ndarray] = None,
    ):

        # First validate X consistency (features) using parent implementation
        super().validate_consistency(params, X=X, y=None)

        # Then validate y consistency (neurons) - specific to population GLM
        if y is not None:
            n_neurons_coef = jax.tree_util.tree_map(lambda p: p.shape[1], params.coef)
            y_neurons = jnp.unique(y[1])
            if n_neurons_coef != y_neurons.size:
                raise ValueError(
                    "Inconsistent number of neurons. "
                    f"Model coefficients assume "
                    f"{n_neurons_coef} neurons, "
                    f"y has {y_neurons.size} neurons instead!"
                )

            # there might be a better way to do this
            if not jnp.allclose(jnp.arange(y_neurons.size), y_neurons):
                raise ValueError(
                    "Inconsistent neurons labels. "
                    "Neuron labels must map to coefficient indices."
                )

