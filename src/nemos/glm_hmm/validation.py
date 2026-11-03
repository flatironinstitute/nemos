"""Validation classes for GLMHMM and PopulationGLMHMM models."""

from typing import Any, Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from .._read_only_dict import ReadOnlyDict
from ..base_validator import RegressorValidator
from ..glm.validation import GLMValidator
from ..hmm.validation import HMMValidator
from ..typing import DESIGN_INPUT_TYPE
from .params import GLMHMMParams, GLMHMMUserParams


class GLMHMMValidator(HMMValidator[GLMHMMUserParams, GLMHMMParams]):
    """Validate GLM-HMM parameters and inputs."""

    expected_param_dims: Tuple[int] = ReadOnlyDict(
        {
            "coef": 2,
            "intercept": 1,
            "initial_prob": 1,
            "transition_prob": 2,
            "dispersion": 1,
            "scale": 1,
        }
    )
    model_param_names: frozenset[str] = frozenset(
        {
            "coef",
            "intercept",
            "scale",
            "dispersion",
            *HMMValidator.model_param_names,
        }
    )
    model_class: str = "GLMHMM"
    X_dimensionality: int = 2
    y_dimensionality: int = 1
    _glm_validator: GLMValidator = GLMValidator()
    params_validation_sequence: Tuple[Tuple[str, None] | Tuple[str, dict[str, Any]]] = (
        *RegressorValidator.params_validation_sequence[:2],
        (
            "check_array_dimensions",
            dict(
                err_message_format="Invalid parameter dimensionality.\n- coef must be an array "
                "or any JAX pytree with array leaves of shape "
                "``(n_features, n_states)``.\n- intercept must be of shape ``(n_states,)``.\n"
                "- scale must be of shape ``(n_states,)``.\n"
                "- initial_prob must be of shape ``(n_states,)``.\n"
                "- transition_prob must be of shape ``(n_states, n_states)``.\n"
                "\nThe provided coef, intercept, scale, initial_prob and transition_prob "
                "have shape ``{}``, ``{}``, ``{}``, ``{}`` and ``{}`` "
                "instead."
            ),
        ),
        ("check_model_params_shape", None),
        *HMMValidator.params_validation_sequence,
        *RegressorValidator.params_validation_sequence[3:],
    )
    to_model_params: Callable[[GLMHMMUserParams], GLMHMMParams] = None
    from_model_params: Callable[[GLMHMMParams], GLMHMMUserParams] = None

    def check_model_params_shape(self, params: GLMHMMUserParams) -> GLMHMMUserParams:
        """Check the length of the glm parameters state axis."""
        coef, intercept = params.get("coef"), params.get("intercept")
        flat_coef = jax.tree_util.tree_leaves(coef)
        invalid_shapes = jax.tree_util.tree_map(
            lambda x: x.shape[-1] != self.n_states, flat_coef
        )
        if any(invalid_shapes):
            raise ValueError(
                "GLM coef must be of shape ``(n_features, n_states)`` or a dict of arrays "
                "with shape ``(n_features, n_states)``. "
                f"n_states is {self.n_states} but coef has shape(s) ``{invalid_shapes}``."
            )
        if (intercept is not None) and intercept.shape[-1] != self.n_states:
            raise ValueError(
                "GLM intercept must be of shape ``(n_states,)``. "
                f"n_states is {self.n_states} but coef has shape ``{intercept.shape}``."
            )
        return params

    def validate_consistency(
        self,
        params: GLMHMMParams,
        X: Optional[DESIGN_INPUT_TYPE] = None,
        y: Optional[jnp.ndarray] = None,
    ):
        """
        Validate consistency between coef and inputs for single-neuron GLMHMM.

        For single-neuron GLMHMM, only validates feature consistency with X.
        Does not validate y since it's 1D (single neuron, no neuron axis to check).
        """
        self._glm_validator.validate_consistency(params.model_params, X, y)
        has_extras = params.model_params.extra_params is not None
        invalid = has_extras and any(
            v.shape != params.model_params.intercept.shape
            for v in params.model_params.extra_params.values()
        )
        if invalid:
            shapes = jax.tree_util.tree_map(
                lambda x: x.shape, params.model_params.extra_params
            )
            raise ValueError(
                "The distribution extra parameters and the intercept must be of shape ``(n_neurons,)``."
                f"\nThe extra parameters are of shape ``{shapes}`` and the intercept "
                f"is of shape ``{params.model_params.intercept.shape}`` instead."
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
        params: GLMHMMParams,
    ):
        """Check consistency of feature_mask and params."""
        self._glm_validator.feature_mask_consistency(feature_mask, params.model_params)
        return
