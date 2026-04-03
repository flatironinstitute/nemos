"""Validation mixin class for HMM-based models."""

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, TypeVar

import jax
import jax.numpy as jnp

from .. import validation
from ..base_validator import RegressorValidator
from .params import HMMParams, HMMUserParams

# from .typing import ModelParamsT, UserProvidedParamsT

# HMM-type User provided init_params (e.g. for GLM-HMM Tuple[array, array, array, array, array]])
HMMUserProvidedParamsT = TypeVar("HMMUserProvidedParamsT")
# HMM-type Model internal representation (e.g. for GLM-s nemos.glm_hmm.glm_hmm.GLMHMMParams)
HMMModelParamsT = TypeVar("HMMModelParamsT")


def to_hmm_params(user_params: HMMUserParams) -> HMMParams:
    """Map from HMMUserParams to HMMParams.

    Converts user-provided parameters (scale and probabilities in regular space)
    to internal model parameters (log_scale and log probabilities).
    """
    return HMMParams(*(jnp.log(p) for p in user_params))


def from_hmm_params(params: HMMParams) -> HMMUserParams:
    """Map from HMMParams to HMMUserParams.

    Converts internal model parameters (log_scale and log probabilities)
    to user-facing parameters (scale and probabilities in regular space).
    """
    # exponentiate and re-normalize
    initial_prob = jnp.exp(params.log_initial_prob)
    initial_prob /= initial_prob.sum()
    transition_prob = jnp.exp(params.log_transition_prob)
    transition_prob /= transition_prob.sum(axis=1, keepdims=True)
    return (
        initial_prob,
        transition_prob,
    )


@dataclass(frozen=True, repr=False)
class HMMValidator(RegressorValidator[HMMUserProvidedParamsT, HMMModelParamsT]):
    """Validate HMM parameters. Meant to be used as a mixin class for models that use HMMs."""

    n_states: int = field(kw_only=True)  # keyword only and required.
    expected_param_dims: Tuple[int] = (
        1,
        2,
    )  # (init_prob.ndim, transition_prob.ndim)
    initial_prob_ind: int = None  # index of initial probability in user params tuple
    transition_prob_ind: int = (
        None  # index of transition probability in user params tuple
    )
    model_param_names: Tuple[str] = ("initial_prob", "transition_prob")
    model_class: str = "HMM"
    params_validation_sequence: Tuple[Tuple[str, None] | Tuple[str, dict[str, Any]]] = (
        ("check_init_and_transition_prob_shape", None),
        ("check_init_and_transition_prob_sum_to_1", None),
    )

    def check_array_dimensions(
        self,
        params: HMMUserProvidedParamsT,
        err_msg: Optional[str] = None,
        err_message_format: str = None,
    ) -> HMMUserProvidedParamsT:
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

    def check_user_params_structure(
        self, params: HMMUserProvidedParamsT, **kwargs
    ) -> HMMUserProvidedParamsT:
        """
        Validate that user parameters are a two-element structure.

        Parameters
        ----------
        params :
            User-provided parameters (should be a tuple/list of length 2).
        **kwargs
            Additional keyword arguments (unused).

        Returns
        -------
        :
            The validated parameters.

        Raises
        ------
        ValueError
            If parameters do not have length two.
        """
        validation.check_length(
            params,
            len(self.expected_param_dims),
            f"Params must have length {len(self.expected_param_dims)}: "
            f"({', '.join(self.model_param_names)}).",
        )
        if not isinstance(params, (tuple, list)):
            raise TypeError(
                f"{self.model_class} params must be a tuple/list of length {len(self.expected_param_dims)}, "
                f"({', '.join(self.model_param_names)})."
            )
        return params

    def check_init_and_transition_prob_shape(
        self, params: HMMUserProvidedParamsT
    ) -> HMMUserProvidedParamsT:
        """Check initial and transition probabilities shape."""
        wrapped = self.wrap_user_params(params)
        initial_prob, transition_prob = (
            wrapped[self.initial_prob_ind],
            wrapped[self.transition_prob_ind],
        )
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
        self, params: HMMUserProvidedParamsT
    ) -> HMMUserProvidedParamsT:
        """Check that initial and transition probability sum to 1."""
        wrapped = self.wrap_user_params(params)
        initial_prob, transition_prob = (
            wrapped[self.initial_prob_ind],
            wrapped[self.transition_prob_ind],
        )
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
