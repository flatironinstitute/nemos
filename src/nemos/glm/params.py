"""GLM parameter definitions and type aliases."""

from typing import Callable

from jaxtyping import PyTree

from ..params import ModelParams


class GLMParams[LeafT](ModelParams[LeafT]):
    """Parameter container for GLM models."""

    coef: PyTree[LeafT]
    intercept: LeafT

    @staticmethod
    def regularizable_subtrees() -> list[Callable[["GLMParams[LeafT]"], PyTree[LeafT]]]:
        """Filter regularizable subtrees."""
        return [lambda p: p.coef]


type GLMUserParams[LeafT] = tuple[PyTree[LeafT], LeafT]
