"""GLM-HMM parameter definitions and type aliases."""

from typing import Tuple

import jax.numpy as jnp
from numpy.typing import ArrayLike

from ..params import ModelParams


class HMMParams(ModelParams):
    """Parameter container for HMM models."""

    log_initial_prob: jnp.ndarray
    log_transition_prob: jnp.ndarray


# Tuple[init_proba, transition_proba]
HMMUserParams = Tuple[ArrayLike, ArrayLike]
