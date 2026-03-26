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
from ..glm.validation import GLMValidator, to_glm_params
from ..glm.params import GLMParams, GLMUserParams

class PointProcessGLMValidator(GLMValidator):
    """
    Continuous-time PPGLM
    """
    extra_params: Dict[Literal["random_key"], ArrayLike] = field(kw_only=True)
    expected_param_dims: Tuple[int] = (1, 1)
    model_class: str = "PointProcessGLM"
    X_dimensionality: int = 2
    y_dimensionality: int = 2
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

    def validate_consistency(
        self,
        params: GLMParams,
        X: Optional[DESIGN_INPUT_TYPE] = None,
        y: Optional[jnp.ndarray] = None,
    ):
        """
        Validate consistency between parameters and inputs for PP-GLM.
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


            #TODO: consistency between unique X[1].size and n neurons in params

    def get_empty_params(self, X, y) -> GLMParams:
        """Return the param shape given the input data."""
        empty_coef = jax.tree_util.tree_map(lambda x: jnp.empty((x.shape[1],)), X)
        empty_intercept = jnp.empty((1,))
        return to_glm_params((empty_coef, empty_intercept))


class PopulationPointProcessGLMValidator(PointProcessGLMValidator):
    """
    Population Continuous-time PPGLM
    """
    extra_params: Dict[Literal["random_key"], ArrayLike] = field(kw_only=True)
    expected_param_dims: Tuple[int] = (2, 1)
    model_class: str = "PopulationPointProcessGLM"
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
    def validate_consistency(
            self,
            params: GLMParams,
            X: Optional[DESIGN_INPUT_TYPE] = None,
            y: Optional[jnp.ndarray] = None,
    ):

        # First validate X consistency (features) using parent implementation
        super().validate_consistency(params, X=X, y=None)

        # Then validate y consistency (neurons)
        if y is not None:
            # coef shape is (n_features, n_neurons)
            y_neurons = jnp.unique(y[1]).size

            if params.coef.shape[1] != y_neurons:
                raise ValueError("Inconsistent number of neurons. "
                f"Model coefficients assume "
                f"{jax.tree_util.tree_map(lambda p: p.shape[1], params.coef)} neurons, "
                f"y has {y_neurons} neurons instead!"
                )

    def get_empty_params(self, X, y) -> GLMParams:
        """Return the param shape given the input data."""
        n_neurons = jnp.unique(y[1]).size

        empty_coef = jax.tree_util.tree_map(
            lambda x: jnp.empty((x.shape[1], n_neurons,)), X
        )
        empty_intercept = jnp.empty((n_neurons,))

        return to_glm_params((empty_coef, empty_intercept))
