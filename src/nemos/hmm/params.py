"""HMM parameter definitions and type aliases."""

from typing import TypeVar

import jax.numpy as jnp

from ..params import ModelParams
from ..typing import LeafT

# HMM-type User provided init_params (e.g. for GLM-HMM Tuple[array, array, array, array, array]])
HMMUserProvidedParamsT = TypeVar("HMMUserProvidedParamsT")
# HMM-type Model internal representation (e.g. for GLM-s nemos.glm_hmm.glm_hmm.GLMHMMParams)
HMMModelParamsT = TypeVar("HMMModelParamsT")


class HMMParams(ModelParams[LeafT]):
    """Parameter container for HMM models."""

    log_initial_prob: jnp.ndarray
    log_transition_prob: jnp.ndarray


# Dict[Literal["init_proba", "transition_proba"], Array]
HMMUserParams = dict[str, LeafT]


def to_hmm_params(user_params: HMMUserParams) -> HMMParams:
    """Map from HMMUserParams to HMMParams.

    Converts user-provided parameters (scale and probabilities in regular space)
    to internal model parameters (log_scale and log probabilities).
    """
    return HMMParams(
        **{
            "log_" + k: jnp.log(p) if p is not None else None
            for k, p in user_params.items()
        }
    )


def re_normalize(value):
    """Enforce sum to 1."""
    value = jnp.exp(value)
    value /= value.sum(axis=-1, keepdims=True)
    return value


def from_hmm_params(params: HMMParams) -> HMMUserParams:
    """Map from HMMParams to HMMUserParams.

    Converts internal model parameters (log_scale and log probabilities)
    to user-facing parameters (scale and probabilities in regular space).
    """
    return {
        k[4:]: re_normalize(getattr(params, k))
        for k in ["log_initial_prob", "log_transition_prob"]
        if getattr(params, k) is not None
    }
