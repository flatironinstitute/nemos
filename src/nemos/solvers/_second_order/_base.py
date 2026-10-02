"""What every second-order solver in this module shares.

A solver built here is an assembly rather than a hierarchy: it holds a curvature model,
a direction and a line search, and the iteration sequences those three. Neither the
sequencing nor the wiring around it -- building the objective, seeding the state,
choosing between the compiled and the Python loop -- varies between solvers, so both
live here.

Subclasses supply their ``__init__``, their state through :meth:`init_state`, the
keyword arguments they accept, and, where they differ from the defaults here, the two
seams :meth:`_initial_y_diff` and :meth:`_fval_diff`.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, ClassVar, Generic

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Scalar
from optimistix._misc import cauchy_termination

from ... import tree_utils
from ...typing import StepResult
from .._abstract_solver import OptimizationInfo
from ._typing import S, Y

if TYPE_CHECKING:
    from ._curvature import AbstractCurvature
    from ._direction import AbstractDirection
    from ._linesearches import AbstractLineSearch


class SecondOrderState(eqx.Module, Generic[Y]):
    """What every second-order solver carries between iterations.

    There is one class rather than one per solver because the curvature models own
    whatever else they need: L-BFGS keeps its pair history, including the previous
    gradient, inside ``hessian_update_state``, and Newton leaves that field ``None``.
    """

    grad_norm: Scalar
    stats: OptimizationInfo
    # Last accepted step, read by the Cauchy convergence test and, for a model that
    # builds itself from pairs, as the ``s`` of the curvature pair.
    y_diff: Y
    # Set when an iteration produced no usable step: the direction was not a descent
    # direction, or it was not finite. It ends the run, and it is not convergence.
    no_step_found: Bool[Array, ""]
    # optax's line-search state, whose type is private to the chosen transformation.
    ls_state: Any = None
    # the identity-shift ladder for ``Newton``, ``None`` for the proximal solvers
    direction_state: Any = None
    # whatever the curvature model carries; ``None`` for a model that assembles the
    # Hessian afresh at every iterate
    hessian_update_state: Any = None


class AbstractSecondOrderSolver(abc.ABC, Generic[Y, S]):
    """Wiring shared by every second-order solver: objective, state seed, run loop."""

    # Whether the penalty is modelled by the quadratic. False for solvers that reach the
    # penalty through a proximal operator instead.
    _proximal: ClassVar[bool] = False

    # Set by every subclass' ``__init__``.
    has_aux: bool
    jit: bool
    fun: Any
    fun_with_aux: Any
    maxiter: int
    tol: float
    rtol: float
    curvature: AbstractCurvature
    direction: AbstractDirection
    _line_search: AbstractLineSearch
    _fval_and_grad: Any

    def _set_objective(
        self, loss_fn: Any, has_aux: bool, maxiter: int, tol: float, rtol: float
    ) -> None:
        """Split the objective into its scalar and ``(scalar, aux)`` forms, and derive.

        ``_fval_and_grad`` is built here because every solver differentiates whatever
        smooth objective it was handed, and the stopping tolerances ride along because
        nothing else consumes them.
        """
        if has_aux:
            self.fun_with_aux = loss_fn
            self.fun = lambda p, *a: loss_fn(p, *a)[0]
        else:
            self.fun = loss_fn
            self.fun_with_aux = lambda p, *a: (loss_fn(p, *a), None)
        self._fval_and_grad = jax.value_and_grad(self.fun_with_aux, has_aux=True)
        self.maxiter = maxiter
        self.tol = tol
        self.rtol = rtol

    def _fval_diff(self, fval: Scalar, state: S) -> Scalar:
        """The function-value arm of the Cauchy test, suppressed by returning zero.

        Suppressed by default: ``|f(x_new) - f(x_old)| < atol`` is an absolute threshold
        that fails under catastrophic cancellation when the objective is large. A solver
        whose step-norm arm cannot fire on its own overrides this.
        """
        del fval, state
        return jnp.zeros(())

    def _scalar_dtype(self, init_params: Y, *args: Any) -> jnp.dtype:
        """The objective's dtype, which the state's scalars must already carry.

        The ``while_loop`` carry fails to typecheck otherwise.
        """
        return jax.eval_shape(self.fun, init_params, *args).dtype

    def _initial_y_diff(self, init_params: Y) -> Y:
        """The step the first convergence test reads, before any step has been taken.

        Infinite, so a Cauchy criterion on the step alone cannot fire on iteration one.
        A solver whose curvature model reads ``y_diff`` as a curvature pair overrides
        this: see :meth:`~nemos.solvers._second_order._lbfgs.ProximalLBFGS._initial_y_diff`.
        """
        return jax.tree.map(lambda x: jnp.full_like(x, jnp.inf), init_params)

    def _common_state_fields(self, init_params: Y, *args: Any) -> dict[str, Any]:
        """The state fields every second-order solver carries, ready to splat."""
        scalar_dtype = self._scalar_dtype(init_params, *args)
        return dict(
            grad_norm=jnp.asarray(jnp.inf, dtype=scalar_dtype),
            stats=OptimizationInfo(
                function_val=jnp.asarray(jnp.nan, dtype=scalar_dtype),
                num_steps=jnp.array(0),
                converged=jnp.array(False),
                reached_max_steps=jnp.array(False),
            ),
            ls_state=self._line_search.init(init_params),
            y_diff=self._initial_y_diff(init_params),
            no_step_found=jnp.array(False),
        )

    @abc.abstractmethod
    def init_state(self, init_params: Y, *args: Any) -> S: ...

    def update(
        self,
        params: Y,
        state: S,
        *args: Any,
    ) -> StepResult:
        (fval, aux), grad = self._fval_and_grad(params, *args)

        gnorm = jnp.sqrt(lx.internal.tree_dot(grad, grad))
        converged = self.converged(params, state, grad, fval)

        def step(_):
            hvp_fn, hessian_tensor, new_hessian_state = self.curvature.update(
                state.hessian_update_state,
                params,
                state.y_diff,
                grad,
                *args,
            )
            step, dir_state = self.direction.update(
                params,
                grad,
                hvp_fn,
                hessian_tensor,
                state.direction_state,
            )

            new_params, new_ls_state, no_step_found = self._apply_or_reject(
                params, step, grad, state, fval, *args
            )

            # ``is_leaf`` so a field still holding its ``None`` default is written
            # rather than treated as an empty subtree.
            new_state = eqx.tree_at(
                lambda s: (s.ls_state, s.hessian_update_state, s.direction_state),
                state,
                (new_ls_state, new_hessian_state, dir_state),
                is_leaf=lambda x: x is None,
            )
            return new_params, new_state, no_step_found

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
        *args: Any,
    ) -> tuple[Y, Any, Bool[Array, ""]]:
        """Accept or reject step based on descent condition and line search.

        Returns ``no_step_found`` for this iteration: true
        when no step was taken and the iterate is not stationary.
        """
        updates, new_ls_state = self._line_search.update(
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
            self.tol,
            lx.internal.two_norm,
            params,
            state.y_diff,
            fval,
            self._fval_diff(fval, state),
        )

    def _iterate(self, init_params: Y, state: S, *args: Any) -> tuple[Y, S]:
        """Step until the stopping rule fires, the run stalls, or ``maxiter`` passes."""

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
            return self.update(p, s, *args)[:2]

        if self.jit:
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

    def run(self, init_params: Y, *args: Any) -> StepResult:
        """Iterate to convergence, to a stall, or to ``maxiter``.

        ``jit`` picks which of the two loops in :meth:`_run` executes, so the compiled
        path has to be reached through a separate method: decorating this one would trace
        the Python loop and fail on its data-dependent condition.
        """
        if self.jit:
            return self._run_jit(init_params, *args)
        return self._run(init_params, *args)

    @eqx.filter_jit
    def _run_jit(self, init_params: Y, *args: Any) -> StepResult:
        return self._run(init_params, *args)

    def _run(self, init_params: Y, *args: Any) -> StepResult:
        state = self.init_state(init_params, *args)
        final_params, final_state = self._iterate(init_params, state, *args)
        # ``fun_with_aux`` is the objective again: only pay for it when there is an aux
        # to collect, since without one it returns a ``None`` the caller already knows.
        aux = (
            self.fun_with_aux(final_params, *args)[1]
            if self._line_search.has_aux
            else None
        )
        return final_params, final_state, aux

    @classmethod
    def get_accepted_arguments(cls) -> set[str]:
        return {"maxiter", "tol", "rtol", "jit"}

    def _get_optim_info(self, state: S, **kwargs: Any) -> OptimizationInfo:
        return state.stats
