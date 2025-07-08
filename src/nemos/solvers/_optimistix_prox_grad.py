from collections.abc import Callable
from typing import Generic, TypeAlias

import equinox as eqx
from equinox.internal import ω
from jaxtyping import PyTree, Scalar

from optimistix._custom_types import Aux, Y
from optimistix._misc import (
    max_norm,
)
from optimistix._search import (
    AbstractDescent,
    AbstractSearch,
    FunctionInfo,
)
from optimistix._solution import RESULTS
from ._optimistix_helpers import ProxBacktrackingArmijo
from optimistix._solver.gradient_methods import AbstractGradientDescent


class _ProxDescentState(eqx.Module, Generic[Y]):
    current_point: Y
    grad: Y


_FnInfo: TypeAlias = (
    FunctionInfo.EvalGrad
    | FunctionInfo.EvalGradHessian
    | FunctionInfo.EvalGradHessianInv
    | FunctionInfo.ResidualJac
)


class ProxDescent(AbstractDescent[Y, _FnInfo, _ProxDescentState]):
    """Descent direction given by prox(x - grad)"""

    regularizer_strength: float
    prox: Callable
    norm: Callable[[PyTree], Scalar] | None = None

    def init(self, y: Y, f_info_struct: _FnInfo) -> _ProxDescentState:
        del f_info_struct
        # Dummy; unused
        return _ProxDescentState(y, y)

    def query(
        self, y: Y, f_info: _FnInfo, state: _ProxDescentState
    ) -> _ProxDescentState:
        if isinstance(
            f_info,
            (
                FunctionInfo.EvalGrad,
                FunctionInfo.EvalGradHessian,
                FunctionInfo.EvalGradHessianInv,
            ),
        ):
            grad = f_info.grad
        elif isinstance(f_info, FunctionInfo.ResidualJac):
            grad = f_info.compute_grad()
        else:
            raise ValueError(
                "Cannot use `SteepestDescent` with this solver. This is because "
                "`SteepestDescent` requires gradients of the target function, but "
                "this solver does not evaluate such gradients."
            )
        if self.norm is not None:
            grad = (grad**ω / self.norm(grad)).ω

        # this is called if the search accepted the stepsize,
        # so y is the newly accepted point
        # and we can set the current point to it
        return _ProxDescentState(y, grad)

    def step(self, step_size: Scalar, state: _ProxDescentState) -> tuple[Y, RESULTS]:
        next_point = (state.current_point**ω - step_size * state.grad**ω).ω
        next_point = self.prox(
            next_point,
            self.regularizer_strength,
            step_size,
        )
        return (next_point**ω - state.current_point**ω).ω, RESULTS.successful


class _ProximalGradient(AbstractGradientDescent[Y, Aux]):
    """
    Proximal gradient.
    """

    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    descent: ProxDescent[Y]
    search: AbstractSearch

    def __init__(
        self,
        prox: Callable,
        regularizer_strength: float,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar],
        search: AbstractSearch = ProxBacktrackingArmijo(),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.descent = ProxDescent(regularizer_strength, prox)
        self.search = search
