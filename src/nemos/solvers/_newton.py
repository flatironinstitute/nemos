"""Newton-based optimization solvers.

This module provides second-order optimization routines based on Newton's method.
"""

from typing import Any, Callable, Generic, Optional, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optax

from ..tree_utils import ravel_pytree_nest
from ..typing import Params
from ._abstract_solver import OptimizationInfo, SolverProtocol, SolverState
from ._aux_helpers import wrap_aux
from ._hess import BlockDiagonal, Full, General, HessianTag, PositiveDefinite


@runtime_checkable
class NewtonSolverProtocol(SolverProtocol[SolverState], Protocol, Generic[SolverState]):
    def setup_hessian(
        self,
        hess_fn: Callable | None,
        hess_tag: HessianTag | None,
    ) -> None: ...


class NewtonState(eqx.Module):
    """State of the Newton optimization process."""

    iter_num: int
    converged: bool
    grad_norm: float
    stats: OptimizationInfo
    ls_state: Optional[Any] = None


class _Newton:
    """Minimal step-based Newton optimizer."""

    def __init__(
        self,
        func: Callable,
        maxiter: int = 30,
        tol: float = 1e-6,
        force_autodiff_hessian: bool = False,
        jit: bool = True,
    ):
        self.func = func
        self.maxiter = maxiter
        self.tol = tol
        self.force_autodiff_hessian = force_autodiff_hessian
        self.jit = jit

        self._val_and_grad = jax.value_and_grad(func)

        self._hess_tag: HessianTag | None = None
        self._hess_fn: Callable | None = None

        self._line_search = optax.scale_by_backtracking_linesearch(
            max_backtracking_steps=30
        )

        self._linear_solver: lx.AbstractLinearSolver = lx.LU()

    def init_state(self, init_params, *args):
        params_flat, _ = ravel_pytree_nest(init_params)

        ls_state = (
            self._line_search.init(params_flat)
            if self._line_search is not None
            else None
        )
        if self._hess_tag is None:
            self._hess_tag = HessianTag(structure=Full, property=General)

        return NewtonState(
            iter_num=jnp.array(0),
            converged=jnp.array(False),
            grad_norm=jnp.array(jnp.inf),
            stats=OptimizationInfo(
                function_val=jnp.array(jnp.nan),
                num_steps=jnp.array(0),
                converged=jnp.array(False),
                reached_max_steps=jnp.array(False),
            ),
            ls_state=ls_state,
        )

    def newton_step(
        self,
        H,
        g_flat,
        tag: HessianTag,
    ):
        if tag.structure is BlockDiagonal and not self.force_autodiff_hessian:

            def _solve_subproblem(H_sub, g_flat_sub):
                return self.newton_step(
                    H_sub,
                    g_flat_sub,
                    HessianTag(Full, tag.property),
                )

            return jax.vmap(_solve_subproblem)(H, g_flat.reshape(H.shape[0], -1))

        if tag.property is PositiveDefinite:
            step_flat = lx.linear_solve(
                lx.MatrixLinearOperator(
                    H,
                    lx.positive_semidefinite_tag,
                ),
                -g_flat,
                lx.Cholesky(),
            ).value

        else:
            step_flat = lx.linear_solve(
                lx.MatrixLinearOperator(H),
                -g_flat,
                lx.LU(),
            ).value

        return step_flat

    def update(self, params, state: NewtonState, *args):
        params_flat, unravel = ravel_pytree_nest(params)

        def value_fn_flat(x):
            return self.func(unravel(x), *args)

        f = value_fn_flat(params_flat)

        g_flat = jax.grad(value_fn_flat)(params_flat)

        gnorm = jnp.linalg.norm(g_flat)
        converged = gnorm <= self.tol

        def do_step(_):
            H = (
                self._hess_fn(params, *args)
                if (not self.force_autodiff_hessian and self._hess_fn is not None)
                else jax.hessian(value_fn_flat)(params_flat)
            )

            step_flat = self.newton_step(H, g_flat, self._hess_tag)

            descent = jnp.vdot(g_flat, step_flat) < 0.0

            step_tree = unravel(step_flat)

            def accept_step(_):
                if self._line_search is not None:
                    updates, new_ls_state = self._line_search.update(
                        step_tree,
                        state.ls_state,
                        params,
                        value=f,
                        grad=step_tree,  # now consistent (tree form)
                        value_fn=lambda p: self.func(p, *args),
                    )
                else:
                    updates = step_tree
                    new_ls_state = state.ls_state

                new_params = jax.tree_util.tree_map(
                    lambda p, u: p + u,
                    params,
                    updates,
                )

                return new_params, new_ls_state

            def reject_step(_):
                return params, state.ls_state

            return jax.lax.cond(descent, accept_step, reject_step, operand=None)

        def no_step(_):
            return params, state.ls_state

        new_params, new_ls_state = jax.lax.cond(
            converged,
            no_step,
            do_step,
            operand=None,
        )

        new_iter = jnp.where(converged, state.iter_num, state.iter_num + 1)

        new_stats = OptimizationInfo(
            function_val=f,
            num_steps=new_iter,
            converged=converged,
            reached_max_steps=new_iter >= self.maxiter,
        )

        new_state = NewtonState(
            iter_num=new_iter,
            converged=converged,
            grad_norm=gnorm,
            stats=new_stats,
            ls_state=new_ls_state,
        )

        return new_params, new_state, None

    def run(self, init_params, *args):
        state = self.init_state(init_params, *args)
        params = init_params

        if self.jit:

            def cond(carry):
                params, state = carry

                return (~state.converged) & (state.iter_num < self.maxiter)

            def body(carry):
                params, state = carry

                params, state, _ = self.update(
                    params,
                    state,
                    *args,
                )

                return params, state

            params, state = eqx.internal.while_loop(
                cond,
                body,
                (params, state),
                kind="lax",
            )

        else:
            for _ in range(self.maxiter):
                params, state, _ = self.update(
                    params,
                    state,
                    *args,
                )
                if state.converged:
                    break

        return params, state, None


class Newton(NewtonSolverProtocol[NewtonState]):
    def __init__(
        self,
        unregularized_loss: Callable,
        regularizer,
        regularizer_strength: float | None,
        has_aux: bool,
        init_params: Params | None = None,
        **solver_kwargs,
    ):
        loss_fn = regularizer.penalized_loss(
            unregularized_loss,
            params=init_params,
            strength=regularizer_strength,
        )

        self.regularizer_strength = regularizer_strength

        self.fun = loss_fn
        self.fun_with_aux = wrap_aux(self.fun)

        self._solver = _Newton(
            self.fun,
            **solver_kwargs,
        )

    def setup_hessian(
        self,
        hess_fn: Optional[Callable] = None,
        hess_tag: HessianTag | None = None,
    ):
        self._solver._hess_fn = hess_fn
        self._solver._hess_tag = hess_tag

    def init_state(self, init_params: Params, *args):
        return self._solver.init_state(init_params, *args)

    def update(self, params: Params, state: NewtonState, *args):
        return self._solver.update(params, state, *args)

    def run(self, init_params: Params, *args):
        return self._solver.run(init_params, *args)

    @classmethod
    def get_accepted_arguments(cls) -> set[str]:
        return {
            "maxiter",
            "tol",
            "force_autodiff_hessian",
            "jit",
        }

    def _get_optim_info(
        self,
        state: NewtonState,
        **kwargs,
    ) -> OptimizationInfo:
        return OptimizationInfo(
            function_val=state.stats.function_val,
            num_steps=state.stats.num_steps,
            converged=state.stats.converged,
            reached_max_steps=state.stats.reached_max_steps,
        )
