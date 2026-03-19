"""Validation classes for GLMHMM and PopulationGLMHMM models."""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple, Union

import jax.numpy as jnp

from .. import validation
from ..base_validator import RegressorValidator
from ..typing import DESIGN_INPUT_TYPE
from .params import HMMParams, HMMUserParams


def to_hmm_params(user_params: HMMUserParams) -> HMMParams:
    """Map from GLMHMMUserParams to GLMHMMParams.

    Converts user-provided parameters (scale and probabilities in regular space)
    to internal model parameters (log_scale and log probabilities).
    """
    return HMMParams(
        hmm_params=HMMParams(*(jnp.log(p) for p in user_params)),
    )


def from_hmm_params(params: HMMParams) -> HMMUserParams:
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
        initial_prob,
        transition_prob,
    )


@dataclass(frozen=True, repr=False)
class HMMValidator(RegressorValidator[HMMUserParams, HMMParams]):
    """Validate GLM-HMM parameters and inputs."""

    n_states: int = field(kw_only=True)  # keyword only and required.
    expected_param_dims: Tuple[int] = (
        1,
        2,
    )  # (coef.ndim, intercept.ndim, scale.ndim, init_prob.ndim, transition_prob.ndim)
    to_model_params: Callable[[HMMUserParams], HMMParams] = to_hmm_params
    from_model_params: Callable[[HMMParams], HMMUserParams] = from_hmm_params
    model_class: str = "HMM"
    params_validation_sequence: Tuple[Tuple[str, None] | Tuple[str, dict[str, Any]]] = (
        *RegressorValidator.params_validation_sequence[:2],
        (
            "check_array_dimensions",
            dict(
                err_message_format="Invalid parameter dimensionality.\n"
                "- initial_prob must be of shape ``(n_states,)``.\n"
                "- transition_prob must be of shape ``(n_states, n_states)``.\n"
                "\nThe provided initial_prob and transition_prob "
                "have shape ``{}``, ``{}``, ``{}``, ``{}`` and ``{}`` "
                "instead."
            ),
        ),
        ("check_init_and_transition_prob_shape", None),
        ("check_init_and_transition_prob_sum_to_1", None),
        *RegressorValidator.params_validation_sequence[3:],
    )

    def check_init_and_transition_prob_shape(
        self, params: HMMUserParams
    ) -> HMMUserParams:
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

    def check_init_and_transition_prob_sum_to_1(
        self, params: HMMUserParams
    ) -> HMMUserParams:
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
        self, params: HMMUserParams, **kwargs
    ) -> HMMUserParams:
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
            2,
            "Params must have length 2: " "(initial_prob, transition_prob).",
        )
        if not isinstance(params, (tuple, list)):
            raise TypeError(
                "HMM params must be a tuple/list of length 5, "
                "(initial_prob, transition_prob)."
            )
        return params

    def validate_consistency(
        self,
        params: GLMHMMParams,
        X: Optional[DESIGN_INPUT_TYPE] = None,
        y: Optional[jnp.ndarray] = None,
    ):
        """
        Validate consistency between coef and inputs for single-neuron HMM.

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

    # def validate_and_cast_feature_mask(
    #     self,
    #     feature_mask: Union[dict[str, jnp.ndarray], jnp.ndarray],
    #     data_type: Optional[DTypeLike] = None,
    # ) -> Union[dict[str, jnp.ndarray], jnp.ndarray]:
    #     """
    #     Validate and cast a feature mask to JAX arrays.

    #     Validates that the feature mask contains only 0s and 1s, then converts
    #     it to JAX arrays with the specified data type. Subclasses can extend
    #     this to add parameter-specific validation (e.g., checking that mask
    #     shape matches parameter dimensions).

    #     Parameters
    #     ----------
    #     feature_mask : dict[str, jnp.ndarray] or jnp.ndarray
    #         Feature mask indicating which features are used. Must contain only 0s and 1s.
    #     data_type : jnp.dtype, optional
    #         Target data type for the mask arrays. Defaults to float.

    #     Returns
    #     -------
    #     dict[str, jnp.ndarray] or jnp.ndarray
    #         The validated and cast feature mask.

    #     Raises
    #     ------
    #     ValueError
    #         If feature_mask contains values other than 0 or 1.
    #     """
    #     return self._glm_validator.validate_and_cast_feature_mask(
    #         feature_mask, data_type=data_type
    #     )

    # def feature_mask_consistency(
    #     self,
    #     feature_mask: Union[dict[str, jnp.ndarray], jnp.ndarray] | None,
    #     params: HMMParams,
    # ):
    #     """Check consistency of feature_mask and params."""
    #     self._glm_validator.feature_mask_consistency(feature_mask, params.glm_params)
    #     return

    def get_empty_params(self, X, y) -> HMMParams:
        """Return the param shape given the input data."""
        pass
