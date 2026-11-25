"""GLM-HMM parameter definitions and type aliases."""

from typing import Callable, NotRequired, TypedDict, Union

import equinox as eqx
import jax.numpy as jnp
from numpy.typing import ArrayLike

from ..glm.params import GLMParams, GLMUserParams
from ..hmm.params import HMMParams, from_hmm_params, to_hmm_params
from ..params import ModelParams
from ..typing import DESIGN_INPUT_TYPE


class GLMHMMParams(ModelParams):
    """Parameter container for GLM-HMM models."""

    model_params: GLMParams
    hmm_params: HMMParams

    @staticmethod
    def regularizable_subtrees() -> list[
        Callable[["GLMHMMParams"], jnp.ndarray | dict]
    ]:
        """Filter regularizable subtrees."""
        return [lambda p: p.model_params.coef]

    @staticmethod
    def solver_param_subtree() -> Callable[["GLMHMMParams"], GLMParams]:
        """Accessor for the sub-pytree the numerical solver and regularizer operate on.

        GLM-HMM is a composite model: its EM M-step optimizes only the GLM-level
        params (coef, intercept, scale). The regularizer and its GroupLasso mask are
        interpreted at this level. Flat models have no such method — consumers must
        treat its absence as the identity accessor.
        """
        return lambda p: p.model_params


# define the dict schema for input types.
class GLMHMMUserParams(TypedDict):
    """TypeDict for the union of the user parameters."""

    coef: NotRequired[Union[DESIGN_INPUT_TYPE, ArrayLike]]
    intercept: NotRequired[ArrayLike]
    scale: NotRequired[ArrayLike]
    dispersion: NotRequired[ArrayLike]
    initial_prob: NotRequired[ArrayLike]
    transition_prob: NotRequired[ArrayLike]


class ToGLMHMMParams(eqx.Module):
    """Convert GLMHMMUserParams to GLMHMMParams."""

    to_glm_params: Callable[[GLMUserParams], GLMParams]

    def __call__(self, user_params: GLMHMMUserParams) -> GLMHMMParams:
        """Convert user param dictionary to GLMHMMParams."""
        user_params = {} if user_params is None else user_params.copy()
        hmm_user_params = {
            k: user_params.pop(k, None) for k in ["initial_prob", "transition_prob"]
        }
        return GLMHMMParams(
            hmm_params=to_hmm_params(hmm_user_params),
            model_params=self.to_glm_params(user_params),
        )


class FromGLMHMMParams(eqx.Module):
    """Convert GLMHMMParams to GLMHMMUserParams."""

    from_glm_params: Callable[[GLMParams], GLMUserParams]

    def __call__(self, params: GLMHMMParams) -> GLMHMMUserParams:
        """Convert GLMHMMParams to user param dictionary."""
        hmm_params = from_hmm_params(params.hmm_params)
        glm_params = self.from_glm_params(params.model_params)
        return {**glm_params, **hmm_params}
