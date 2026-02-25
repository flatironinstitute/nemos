"""Validation classes for GLMHMM and PopulationGLMHMM models."""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from .. import validation
from ..base_validator import RegressorValidator
from ..glm.params import GLMParams, GLMUserParams
from ..glm.validation import GLMValidator
from ..typing import DESIGN_INPUT_TYPE
from .params import GLMHMMParams, GLMHMMUserParams, GLMScale, HMMParams


def to_glm_hmm_params(user_params: GLMHMMUserParams) -> GLMHMMParams:
    """Map from GLMHMMUserParams to GLMHMMParams.

    Converts user-provided parameters (scale and probabilities in regular space)
    to internal model parameters (log_scale and log probabilities).
    """
    return GLMHMMParams(
        glm_params=GLMParams(*user_params[:2]),
        glm_scale=GLMScale(jnp.log(user_params[2])),
        hmm_params=HMMParams(*(jnp.log(p) for p in user_params[3:])),
    )


def from_glm_hmm_params(params: GLMHMMParams) -> GLMHMMUserParams:
    """Map from GLMHMMParams to GLMHMMUserParams.

    Converts internal model parameters (log_scale and log probabilities)
    to user-facing parameters (scale and probabilities in regular space).
    """
    # exponentiate and re-normalize
    initial_prob = jnp.exp(params.hmm_params.log_initial_prob)
    initial_prob /= initial_prob.sum()
    transition_prob = jnp.exp(params.hmm_params.log_transition_prob)
    transition_prob /= transition_prob.sum(axis=1, keepdims=True)
    return (
        params.glm_params.coef,
        params.glm_params.intercept,
        jnp.exp(params.glm_scale.log_scale),
        initial_prob,
        transition_prob,
    )


@dataclass(frozen=True, repr=False)
class GLMHMMValidator(RegressorValidator[GLMUserParams, GLMParams]):
    """Validate GLM-HMM parameters and inputs."""

    n_states: int = field(kw_only=True)  # keyword only and required.
    expected_param_dims: Tuple[int] = (
        2,
        1,
        1,
        1,
        2,
    )  # (coef.ndim, intercept.ndim, scale.ndim, init_prob.ndim, transition_prob.ndim)
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
                "or nemos.pytree.FeaturePytree with array leafs of shape "
                "``(n_features, n_states)``.\n- intercept must be of shape ``(n_states,)``.\n"
                "- scale must be of shape ``(n_states,)``.\n"
                "- initial_prob must be of shape ``(n_states,)``.\n"
                "- transition_prob must be of shape ``(n_states, n_states)``.\n"
                "\nThe provided coef, intercept, scale, initial_prob and transition_prob "
                "have shape ``{}``, ``{}``, ``{}``, ``{}`` and ``{}`` "
                "instead."
            ),
        ),
        ("check_init_and_transition_prob_shape", None),
        ("check_init_and_transition_prob_sum_to_1", None),
        ("check_glm_params_shape", None),
        *RegressorValidator.params_validation_sequence[3:],
    )

    def check_array_dimensions(
        self,
        params: GLMHMMUserParams,
        err_msg: Optional[str] = None,
        err_message_format: str = None,
    ) -> GLMHMMUserParams:
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

    def check_init_and_transition_prob_shape(
        self, params: GLMHMMUserParams
    ) -> GLMHMMUserParams:
        """Check initial and transition probabilities shape."""
        wrapped = self.wrap_user_params(params)
        initial_prob, transition_prob = wrapped[-2:]
        if initial_prob.shape != (self.n_states,):
            raise ValueError(
                f"initial_prob must be a 1-dimensional array of shape ``({self.n_states},)``. "
                f"Provided initial_prob is of shape ``{initial_prob.shape}`` instead."
            )
        if transition_prob.shape != (self.n_states, self.n_states):
            raise ValueError(
                f"transition_prob must be a 2-dimensional array of shape ``({self.n_states}, {self.n_states})``."
                f"Provided transition_prob is of shape ``{transition_prob.shape}`` instead."
            )
        return params

    def check_glm_params_shape(self, params: GLMHMMUserParams) -> GLMHMMUserParams:
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

    def check_init_and_transition_prob_sum_to_1(
        self, params: GLMHMMUserParams
    ) -> GLMHMMUserParams:
        """Check that initial and transition probability sum to 1."""
        wrapped = self.wrap_user_params(params)
        initial_prob, transition_prob = wrapped[-2:]
        if not jnp.allclose(initial_prob.sum(), 1):
            raise ValueError(
                f"initial_prob must sum to 1, but got sum = {initial_prob.sum()}. "
            )
        if not jnp.allclose(jnp.sum(transition_prob, axis=1), 1):
            row_sums = jnp.sum(transition_prob, axis=1)
            raise ValueError(
                f"transition_prob matrix rows must sum to 1 over columns, but got sum = {row_sums}. "
                f"Each row i represents the probability distribution of transitioning from state i"
                f"and must sum to 1. "
            )
        return params

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
        validation.check_length(
            params,
            5,
            "Params must have length 5: "
            "(coef, intercept, scale, initial_prob, transition_prob).",
        )
        if not isinstance(params, (tuple, list)):
            raise TypeError(
                "GLMHMM params must be a tuple/list of length 5, "
                "(coef, intercept, scale, initial_prob, transition_prob)."
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
        self._glm_validator.validate_consistency(params.glm_params, X, y)
        if params.glm_scale.log_scale.shape != params.glm_params.intercept.shape:
            raise ValueError(
                "The scale parameter and the intercept must be of shape ``(n_neurons,)``."
                f"\nThe scale is of shape ``{params.glm_scale.log_scale.shape}`` and the intercept "
                f"is of shape ``{params.glm_params.intercept.shape}`` instead."
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
        self._glm_validator.feature_mask_consistency(feature_mask, params.glm_params)
        return
