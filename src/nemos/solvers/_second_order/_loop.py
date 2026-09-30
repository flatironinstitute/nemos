"""Optimization loop for second order methods."""

from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Scalar
from optimistix._misc import cauchy_termination

from ... import tree_utils
from ...typing import StepResult
from .._abstract_solver import OptimizationInfo
from ._curvature import AbstractCurvature
from ._direction import AbstractDirection
from ._linesearches import AbstractLineSearch

# parameters
Y = TypeVar("Y")
# the state a solver carries between iterations
S = TypeVar("S")
# whatever the objective returns alongside its value
Aux = TypeVar("Aux")


class Loop(eqx.Module, Generic[Y, S]):
    """Sequence one iteration: curvature, direction, line search, then the stopping test.

    ``fval_diff_fn`` is what differs between solvers built on it: it supplies the
    function-value arm of the Cauchy test, and returns zero for a solver that suppresses
    that arm.
    """

    maxiter: int
    atol: float
    rtol: float
    fval_diff_fn: Callable[[Scalar, S], Scalar]
    # ``(params, *args) -> ((fval, aux), grad)``
    fval_and_grad_fn: Callable[..., tuple[tuple[Scalar, Aux], Y]]

    def update(
        self,
        params: Y,
        state: S,
        curvature: AbstractCurvature,
        direction: AbstractDirection,
        line_search: AbstractLineSearch,
        *args: Any,
    ) -> StepResult:
        (fval, aux), grad = self.fval_and_grad_fn(params, *args)

        gnorm = jnp.sqrt(lx.internal.tree_dot(grad, grad))
        converged = self.converged(params, state, grad, fval)

        def step(_):
            hvp_fn, hessian_tensor, new_hessian_state = curvature.update(
                state.hessian_update_state,
                params,
                state.y_diff,
                grad,
                *args,
            )
            step, dir_state = direction.update(
                params,
                grad,
                hvp_fn,
                hessian_tensor,
                state.direction_state,
            )

            new_params, new_ls_state, no_step_found = self._apply_or_reject(
                params,
                step,
                grad,
                state,
                fval,
                line_search,
                *args,
            )

            return (
                new_params,
                self._update_state(state, new_ls_state, new_hessian_state, dir_state),
                no_step_found,
            )

        def no_step(_):
            return params, state, jnp.array(False)

        new_params, new_state, no_step_found = jax.lax.cond(
            converged,
            no_step,
            step,
            None,
        )

        new_iter = jnp.where(
            converged,
            state.stats.num_steps,
            state.stats.num_steps + 1,
        )
        new_state = eqx.tree_at(
            lambda s: (s.grad_norm, s.stats, s.y_diff, s.no_step_found),
            new_state,
            (
                gnorm,
                OptimizationInfo(
                    function_val=fval,
                    num_steps=new_iter,
                    converged=converged,
                    reached_max_steps=new_iter >= self.maxiter,
                ),
                tree_utils.tree_sub(new_params, params),
                (~converged) & no_step_found,
            ),
        )
        return new_params, new_state, aux

    def _apply_or_reject(
        self,
        params: Y,
        step: Y,
        grad: Y,
        state: S,
        fval: Scalar,
        line_search: AbstractLineSearch,
        *args: Any,
    ) -> tuple[Y, Any, Bool[Array, ""]]:
        """Accept or reject step based on descent condition and line search.

        Returns ``no_step_found`` for this iteration: true
        when no step was taken and the iterate is not stationary.
        """
        updates, new_ls_state = line_search.update(
            params, step, grad, fval, state.ls_state, *args
        )
        value = new_ls_state.loss_value
        descent = new_ls_state.descent
        step_taken = new_ls_state.step_taken
        # A failed search at a slope this small means the iterate is stationary rather
        # than broken, so the zero step is left for the convergence test to read.
        eps = jnp.finfo(jnp.asarray(value).dtype).eps
        stationary = jnp.abs(descent) <= eps * jnp.abs(value)
        return updates, new_ls_state, ~step_taken & ~stationary

    def converged(self, params: Y, state: S, grad: Y, fval: Scalar) -> Bool[Array, ""]:
        """Check convergence via a Cauchy criterion on the accepted step size.

        We rely solely on the step-norm arm of :func:`~optimistix.cauchy_termination`
        and suppress its function-value arm by passing ``f_diff=0``.  The f-diff arm
        would check ``|f(x_new) - f(x_old)| < atol``, which is an absolute threshold
        that fails under catastrophic cancellation when the objective is large.  The
        step-norm criterion ``‖Δx‖ < atol + rtol * ‖x‖`` is scale-invariant provided
        ``rtol > 0``, so callers should prefer setting ``rtol`` over ``tol`` alone.
        """
        return cauchy_termination(
            self.rtol,
            self.atol,
            lx.internal.two_norm,
            params,
            state.y_diff,
            fval,
            self.fval_diff_fn(fval, state),
        )

    def run(
        self,
        init_params: Y,
        state: S,
        curvature: AbstractCurvature,
        direction: AbstractDirection,
        line_search: AbstractLineSearch,
        jit: bool,
        *args: Any,
    ) -> tuple[Y, S]:
        def cond(carry: tuple[Y, S]) -> Bool[Array, ""]:
            _, s = carry
            return (
                (~s.stats.converged)
                & (~s.no_step_found)
                & (s.stats.num_steps < self.maxiter)
            )

        def body(carry: tuple[Y, S]) -> tuple[Y, S]:
            p, s = carry
            # Discard aux; convergence only needs params and state
            return self.update(p, s, curvature, direction, line_search, *args)[:2]

        if jit:
            final_params, final_state = eqx.internal.while_loop(
                cond,
                body,
                (init_params, state),
                kind="lax",
            )
        else:
            carry = (init_params, state)
            while cond(carry):
                carry = body(carry)
            final_params, final_state = carry
        return final_params, final_state
