"""GLM parameter definitions and type aliases."""

from typing import Callable, Dict, Optional

import equinox as eqx
from jaxtyping import PyTree

from ..inverse_link_function_utils import identity
from ..params import ModelParams
from ..typing import LeafT

# def map_extra_params(extra_params: dict, direction: MapDirection) -> dict:
#     """Map extra params.
#
#     Map extra parameters from user provided to internal representation.
#     An example of that can be the dispersion which undergoes unconstrained
#     optimization, and therefore is mapped to log-space.
#     """
#     return {
#         k: _MAP_LIKELIHOOD_PARS.get(k, {direction: identity})[direction](v)
#         for k, v in extra_params.items()
#     }


class GLMParams[LeafT](ModelParams[LeafT]):
    """Parameter container for GLM models."""

    coef: PyTree[LeafT]
    intercept: LeafT
    extra_params: Optional[dict[str, LeafT]] = None

    @staticmethod
    def regularizable_subtrees() -> list[Callable[["GLMParams[LeafT]"], PyTree[LeafT]]]:
        """Filter regularizable subtrees."""
        return [lambda p: p.coef]


type GLMUserParams[LeafT] = dict[str, LeafT]


class ToGLMParams(eqx.Module):
    """Callable mapping from user dict to GLMParams."""

    map_to: Dict[str, Callable]

    def __call__(self, user_params: GLMUserParams[LeafT]) -> GLMParams[LeafT]:
        """Convert user params to GLM params.

        Parameters
        ----------
        user_params:
            The user parameters as a dictionary.

        Returns
        -------
            The GLMParams.

        Notes
        -----
        This function transform the parameters according to the dictionary of
        callables ``map_to``. If a user parameter key is in map_to, the corresponding
        parameter value will be transformed via map_to[key](value) before being stored
        in GLMParams.extra_params.
        """
        user_params = {} if user_params is None else user_params.copy()
        coef = user_params.pop("coef", None)
        intercept = user_params.pop("intercept", None)
        if user_params != {}:
            extra_params = {
                k: self.map_to.get(p, identity)(p) for k, p in user_params.items()
            }
        else:
            extra_params = None
        return GLMParams(coef, intercept, extra_params)


class FromGLMParams(eqx.Module):
    """Callable mapping from GLMParams to user dict."""

    map_from: Dict[str, Callable]

    def __call__(self, glm_params: GLMParams[LeafT]) -> GLMUserParams[LeafT]:
        """Convert user parameter dictionary to GLMParams.

        Parameters
        ----------
        glm_params:
            The model parameters as a GLMParams.

        Returns
        -------
            The dictionary of user parameters.

        Notes
        -----
        This function transform the parameters according to the dictionary of
        callables ``map_from``.
        If a parameter key is in map_from, the corresponding
        parameter value will be transformed via map_to[key](value) before being stored
        in user_params.
        """
        user_params = {"coef": glm_params.coef, "intercept": glm_params.intercept}
        extra_params = (
            {} if glm_params.extra_params is None else glm_params.extra_params
        )
        user_params.update(
            {k: self.map_from.get(k, identity)(v) for k, v in extra_params.items()}
        )
        return user_params
