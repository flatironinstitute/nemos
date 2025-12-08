from dataclasses import dataclass
from .. import validation
from .params import GLMParams, GLMUserParams
from typing import Callable, Optional, Tuple, Union

from ..tree_utils import pytree_map_and_reduce
from ..typing import DESIGN_INPUT_TYPE, FeaturePytree
import jax.numpy as jnp
import jax
from .. import tree_utils

def check_feature_mask(feature_mask, params: Optional[GLMParams]=None):
    # check if the mask is of 0s and 1s
    if tree_utils.pytree_map_and_reduce(
            lambda x: jnp.any(jnp.logical_and(x != 0, x != 1)), any, feature_mask
    ):
        raise ValueError("'feature_mask' must contain only 0s and 1s!")



@dataclass(frozen=True, repr=False)
class GLMValidator(validation.RegressorValidator[GLMUserParams, GLMParams]):
    """Parameter validator for GLM models."""

    expected_array_dims: Tuple[int] = (
        1,
        1,
    )  # this should be (coef.ndim, intercept.ndim)
    to_model_params: Callable[[GLMUserParams], GLMParams] = lambda p: GLMParams(
        *p
    )  # casting from tuple of array to GLMParams
    model_class: str = "GLM"
    X_dimensionality: int = 2
    y_dimensionality: int = 1
    validation_sequence_kwargs: Tuple[Optional[dict], ...] = (
        None,
        None,
        dict(
            err_message_format="Invalid parameter dimensionality. coef must be an array or nemos.pytree.FeaturePytree "
            "with array leafs of shape (n_features, ). intercept must be of shape (1,). "
            "\nThe provided coef and intercept have shape ``{}`` and ``{}`` instead."
        ),
        None,
        None,
    )


    def additional_validation_model_params(self, params: GLMParams, **kwargs):
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
        err_msg = err_message_format.format(params[0].shape, params[1].shape)
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
        return params

    def validate_consistency(self, params: GLMParams, X: Optional[DESIGN_INPUT_TYPE] = None, y: Optional[jnp.ndarray]=None):
        if X is not None:
            # check that X and params[0] have the same structure
            msg = "X and coef have mismatched same structure."
            if isinstance(X, FeaturePytree):
                data = X.data
                msg += (" X was provided as a FeaturePytree, and coef should be a dictionary with matching keys."
                       f"X keys are ``{X.keys()}``, the provided coef is {params.coef} instead.")
            else:
                data = X
                msg += f" X was provided as an array, and coef should be a array too. The provided coef is of type ``{type(params.coef)}``  instead."

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
                            f"spike basis coefficients has {jax.tree_util.tree_map(lambda p: p.shape[0], params.coef)} features, "
                            f"X has {jax.tree_util.tree_map(lambda x: x.shape[1], X)} features instead!",
            )
        if y is not None:
            validation.check_array_shape_match_tree(
                params.coef,
                y,
                axis=1,
                err_message="Inconsistent number of neurons. "
                f"spike basis coefficients assumes {jax.tree_util.tree_map(lambda p: p.shape[1], params.coef)} neurons, "
                f"y has {jax.tree_util.tree_map(lambda x: x.shape[1], y)} neurons instead!",
            )

    def validate_and_cast_feature_mask(self, feature_mask: Union[dict[str, jnp.ndarray], jnp.ndarray], params: Optional[GLMParams]=None, data_type: jnp.dtype=float) -> Union[dict[str, jnp.ndarray], jnp.ndarray]:
        feature_mask = super().validate_and_cast_feature_mask(feature_mask, params)
        if params is None:
            return feature_mask

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
            n_neurons = 1 if self.y_dimensionality == 1 else next(iter(params.coef.values())).shape[1]
            shape_match = pytree_map_and_reduce(lambda fm: fm.shape == (n_neurons,), all, feature_mask)
            if not shape_match:
                raise ValueError(
                    "Inconsistent number of neurons. "
                    f"feature_mask has {jax.tree_util.tree_map(lambda m: m.shape[neural_axis], feature_mask)} neurons, "
                    f"model coefficients have {jax.tree_util.tree_map(lambda x: x.shape[1], params.coef)}  instead!",
                )
        else:
            shape_match = feature_mask.shape == params.coef.shape
            if not shape_match:
                raise ValueError("The shape of the ``feature_mask`` array must match that of the ``coef``. "
                                 f"The shape of the ``coef`` is ``{params.coef.shape}``, "
                                 f"that of the ``feature_mask`` is ``{feature_mask.shape}`` instead!")
        return feature_mask


@dataclass(frozen=True, repr=False)
class PopulationGLMValidator(GLMValidator):
    y_dimensionality: int = 2
    expected_array_dims: Tuple[int] = (
        2,
        1,
    )  # this should be (coef.ndim, intercept.ndim)
    model_class: str = "PopulationGLM"
    validation_sequence_kwargs: Tuple[Optional[dict], ...] = (
        None,
        None,
        dict(
            err_message_format="Invalid parameter dimensionality. coef must be an array or nemos.pytree.FeaturePytree "
                               "with array leafs of shape (n_features, n_neurons). intercept must be of shape (n_neurons,). "
                               "\nThe provided coef and intercept have shape ``{}`` and ``{}`` instead."
        ),
        None,
        None,
    )