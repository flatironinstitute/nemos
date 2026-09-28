from abc import ABC, abstractmethod

import equinox as eqx
import lineax as lx
import optimistix as optx

from ... import tree_utils
from .._fista import FISTA
from ._curvature import AbstractCurvature, S, Y


class AbstractDirection(eqx.Module, ABC):
    @abstractmethod
    def direction(
        self, params: Y, grad: Y, state: S, curvature: AbstractCurvature
    ) -> Y: ...


class ProxQuadraticDirection(AbstractDirection):
    _inner_solver: FISTA
    _inner_iter: int

    def direction(
        self, params: Y, grad: Y, state: S, curvature: AbstractCurvature
    ) -> Y:
        def quadratic(z, _):
            step = tree_utils.tree_sub(z, params)
            hvp = curvature.hvp(state, params, step)
            return lx.internal.tree_dot(grad, step) + 0.5 * lx.internal.tree_dot(
                step, hvp
            )

        new_params = optx.minimise(
            quadratic,
            self._inner_solver,
            y0=params,
            max_steps=self._inner_iter,
            throw=False,
        ).value
        return tree_utils.tree_sub(new_params, params)
