"""HMM parameter definitions and type aliases."""

from typing import Tuple, TypeVar

import jax.numpy as jnp
from numpy.typing import ArrayLike

from ..params import ModelParams


# HMM-type User provided init_params (e.g. for GLM-HMM Tuple[array, array, array, array, array]])
HMMUserProvidedParamsT = TypeVar("HMMUserProvidedParamsT")
# HMM-type Model internal representation (e.g. for GLM-s nemos.glm_hmm.glm_hmm.GLMHMMParams)
HMMModelParamsT = TypeVar("HMMModelParamsT")


class HMMParams(ModelParams):
    """Parameter container for HMM models."""

    log_initial_prob: jnp.ndarray
    log_transition_prob: jnp.ndarray


# Tuple[init_proba, transition_proba]
HMMUserParams = Tuple[ArrayLike, ArrayLike]
