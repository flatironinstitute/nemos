"""Validation classes for GLMHMM and PopulationGLMHMM models."""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike


from ..base_validator import RegressorValidator
from ..glm.validation import GLMValidator
from ..hmm.validation import HMMValidator, from_hmm_params, to_hmm_params

from ..typing import DESIGN_INPUT_TYPE
from .params import GLMHMMModelParams, GLMHMMParams, GLMHMMUserParams


def to_glm_hmm_params(user_params: GLMHMMUserParams) -> GLMHMMParams:
    """Map from GLMHMMUserParams to GLMHMMParams.

    Converts user-provided parameters (scale and probabilities in regular space)
    to internal model parameters (log_scale and log probabilities).
    """
    return GLMHMMParams(
        model_params=GLMHMMModelParams(user_params[:3]),
        hmm_params=to_hmm_params(user_params[3:]),
    )


def from_glm_hmm_params(params: GLMHMMParams) -> GLMHMMUserParams:
    """Map from GLMHMMParams to GLMHMMUserParams.

    Converts internal model parameters (log_scale and log probabilities)
    to user-facing parameters (scale and probabilities in regular space).
    """
    initial_prob, transition_prob = from_hmm_params(params.hmm_params)
    return (
        params.model_params.coef,
        params.model_params.intercept,
        jnp.exp(params.model_params.log_scale),
        initial_prob,
        transition_prob,
    )


@dataclass(frozen=True, repr=False)
class GLMHMMValidator(HMMValidator[GLMHMMUserParams, GLMHMMParams]):
    """Validate GLM-HMM parameters and inputs."""

    expected_param_dims: Tuple[int] = (
        2,
        1,
        1,
        *HMMValidator.expected_param_dims,
    )  # (coef.ndim, intercept.ndim, scale.ndim, init_prob.ndim, transition_prob.ndim)
    initial_prob_ind: int = 3
    transition_prob_ind: int = 4
    model_param_names: Tuple[str] = (
        "coef",
        "intercept",
        "scale",
        *HMMValidator.model_param_names,
    )
    to_model_params: Callable[[GLMHMMUserParams], GLMHMMParams] = to_glm_hmm_params
    from_model_params: Callable[[GLMHMMParams], GLMHMMUserParams] = from_glm_hmm_params
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

    def check_array_dimensions(
        self,
        params: GLMHMMUserParams,
        err_msg: Optional[str] = None,
        err_message_format: str = None,
    ) -> GLMHMMUserParams:
        """
        Check array dimensions with custom error formatting for HMM-based model parameters.

        Overrides the base implementation to provide model-specific error messages
        that include the actual shapes of the provided parameters. The expected shapes of
        additional model parameters and error message should be set in the child class (e.g
        see GLMHMMValidator for an example).

        Parameters
        ----------
        params :
            User-provided parameters as a tuple.
        err_msg :
            Custom error message (unused, overridden by err_message_format).
        err_message_format :
            Format string for error message that takes two shape arguments.

        Returns
        -------
        :
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

    def check_model_params_shape(self, params: GLMHMMUserParams) -> GLMHMMUserParams:
        """Check the length of the glm parameters state axis."""
        wrapped = self.wrap_user_params(params)
        coef, intercept = wrapped[:2]
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
        if intercept.shape[-1] != self.n_states:
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
        if params.model_params.log_scale.shape != params.model_params.intercept.shape:
            raise ValueError(
                "The scale parameter and the intercept must be of shape ``(n_neurons,)``."
                f"\nThe scale is of shape ``{params.model_params.log_scale.shape}`` and the intercept "
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

    def get_empty_params(self, X, y) -> GLMHMMParams:
        """Return the param shape given the input data."""
        pass
