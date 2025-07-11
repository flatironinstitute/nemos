from typing import cast, ClassVar, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from equinox.internal import ω
from jaxtyping import Array, Bool, Int, Scalar, ScalarLike

from optimistix._custom_types import Y
from optimistix._misc import two_norm
from optimistix._search import AbstractSearch, FunctionInfo
from optimistix._solution import RESULTS


IntScalar = Int[Scalar, ""]


class _ProxBacktrackingState(eqx.Module):
    step_size: Scalar
    ls_iter: IntScalar


_FnInfo: TypeAlias = (
    FunctionInfo.EvalGrad
    | FunctionInfo.EvalGradHessian
    | FunctionInfo.EvalGradHessianInv
    | FunctionInfo.ResidualJac
)
_FnEvalInfo: TypeAlias = FunctionInfo


class ProxBacktrackingArmijo(
    AbstractSearch[Y, _FnInfo, _FnEvalInfo, _ProxBacktrackingState]
):
    """Perform a backtracking Armijo line search."""

    decrease_factor: ScalarLike = 0.5
    slope: ScalarLike = 0.1
    step_init: ScalarLike = 1.0
    _needs_grad_at_y_eval: ClassVar[bool] = False
    max_ls: int = 15

    def __post_init__(self):
        self.decrease_factor = eqx.error_if(
            self.decrease_factor,
            (self.decrease_factor <= 0)  # pyright: ignore
            | (self.decrease_factor >= 1),  # pyright: ignore
            "`BacktrackingArmoji(decrease_factor=...)` must be between 0 and 1.",
        )
        self.slope = eqx.error_if(
            self.slope,
            (self.slope <= 0) | (self.slope >= 1),  # pyright: ignore
            "`BacktrackingArmoji(slope=...)` must be between 0 and 1.",
        )
        self.step_init = eqx.error_if(
            self.step_init,
            self.step_init <= 0,  # pyright: ignore
            "`BacktrackingArmoji(step_init=...)` must be strictly greater than 0.",
        )

    def init(self, y: Y, f_info_struct: _FnInfo) -> _ProxBacktrackingState:
        del y, f_info_struct
        return _ProxBacktrackingState(
            step_size=jnp.array(self.step_init), ls_iter=jnp.array(0)
        )

    def step(
        self,
        first_step: Bool[Array, ""],
        y: Y,
        y_eval: Y,
        f_info: _FnInfo,
        f_eval_info: _FnEvalInfo,
        state: _ProxBacktrackingState,
    ) -> tuple[Scalar, Bool[Array, ""], RESULTS, _ProxBacktrackingState]:
        if not isinstance(
            f_info,
            (
                FunctionInfo.EvalGrad,
                FunctionInfo.EvalGradHessian,
                FunctionInfo.EvalGradHessianInv,
                FunctionInfo.ResidualJac,
            ),
        ):
            raise ValueError(
                "Cannot use `BacktrackingArmijo` with this solver. This is because "
                "`BacktrackingArmijo` requires gradients of the target function, but "
                "this solver does not evaluate such gradients."
            )

        y_diff = (y_eval**ω - y**ω).ω
        predicted_reduction = f_info.compute_grad_dot(y_diff)
        # Terminate when the Armijo condition is satisfied. That is, `fn(y_eval)`
        # must do better than its linear approximation:
        # `fn(y_eval) < fn(y) + grad•y_diff`
        f_min = f_info.as_min()
        f_min_eval = f_eval_info.as_min()
        f_min_diff = f_min_eval - f_min  # This number is probably negative
        # satisfies_armijo = f_min_diff <= self.slope * predicted_reduction
        satisfies_armijo = (
            f_min_diff
            <= predicted_reduction + 0.5 / state.step_size * two_norm(y_diff) ** 2
        )
        # satisfies_armijo = f_min_diff <= (
        #    self.slope * predicted_reduction
        #    + 0.5 / state.step_size * two_norm(y_diff) ** 2
        # )
        has_reduction = predicted_reduction <= 0

        reached_max_ls = state.ls_iter + 1 == self.max_ls
        reached_min_stepsize = state.step_size <= 1e-6
        accept = first_step | (satisfies_armijo & has_reduction) | reached_max_ls
        step_size = jnp.where(
            accept,
            state.step_size / self.decrease_factor,  # increase if accepted
            state.step_size * self.decrease_factor,  # decrease if not
        )
        step_size = jnp.where(
            accept & reached_min_stepsize,
            self.step_init,  # reset if too small
            step_size,  # keep the increased value
        )
        step_size = cast(Scalar, step_size)
        next_ls_iter = jnp.where(reached_max_ls, jnp.array(0), state.ls_iter + 1)
        return (
            step_size,
            accept,
            RESULTS.successful,
            _ProxBacktrackingState(step_size=step_size, ls_iter=next_ls_iter),
        )
