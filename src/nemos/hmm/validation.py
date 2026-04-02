"""Validation mixin class for HMM-based models."""

from dataclasses import dataclass, field
from typing import Any, Tuple

import jax.numpy as jnp
from .params import HMMParams, HMMUserParams


def to_hmm_params(user_params: HMMUserParams) -> HMMParams:
    """Map from HMMUserParams to HMMParams.

    Converts user-provided parameters (scale and probabilities in regular space)
    to internal model parameters (log_scale and log probabilities).
    """
    return HMMParams(
        hmm_params=HMMParams(*(jnp.log(p) for p in user_params)),
    )


def from_hmm_params(params: HMMParams) -> HMMUserParams:
    """Map from HMMParams to HMMUserParams.

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
class HMMValidatorMixin:
    """Validate HMM parameters. Meant to be used as a mixin class for models that use HMMs."""

    n_states: int = field(kw_only=True)  # keyword only and required.
    expected_param_dims: Tuple[int] = (
        1,
        2,
    )  # (init_prob.ndim, transition_prob.ndim)
    params_validation_sequence: Tuple[Tuple[str, None] | Tuple[str, dict[str, Any]]] = (
        ("check_init_and_transition_prob_shape", None),
        ("check_init_and_transition_prob_sum_to_1", None),
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
