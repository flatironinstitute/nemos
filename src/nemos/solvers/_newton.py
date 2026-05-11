"""Newton-based optimization solvers.

This module provides second-order optimization routines based on Newton's method.
"""

from typing import Any, Callable, Generic, Optional, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optax
from jax.flatten_util import ravel_pytree

from ..typing import Params
from ._abstract_solver import OptimizationInfo, SolverProtocol, SolverState
from ._aux_helpers import wrap_aux
from ._hess import BlockDiagonal, Full, HessianTag, PositiveDefinite


def pack_grad(g):
    # g.coef: (D, K)
    # g.intercept: (K,)
    blocks = [
        jnp.concatenate([g.coef[:, i], g.intercept[i : i + 1]])
        for i in range(g.coef.shape[1])
    ]
    return jnp.stack(blocks, axis=0)


def unpack_grad(g_flat):
    # g_flat: (K, D+1)
    coef = g_flat[:, :-1]  # (K, D)
    intercept = g_flat[:, -1]  # (K,)

    return jnp.concatenate(
        [coef.T.reshape(-1), intercept],
        axis=0,
    )


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
        params_flat, _ = ravel_pytree(init_params)

        ls_state = (
            self._line_search.init(params_flat)
            if self._line_search is not None
            else None
        )

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

    def newton_solve(self, H, g_flat, g_tree, tag: HessianTag):
        """Solve Hx = g."""

        if tag.structure is BlockDiagonal:
            g = pack_grad(g_tree)

            return unpack_grad(
                jax.vmap(
                    lambda h, gi: self.newton_solve(
                        h,
                        gi,
                        None,
                        HessianTag(Full, tag.property),
                    )
                )(H, g)
            )

        if tag.property is PositiveDefinite:
            return lx.linear_solve(
                lx.MatrixLinearOperator(H, lx.positive_semidefinite_tag),
                g_flat,
                lx.Cholesky(),
            ).value

        return lx.linear_solve(
            lx.MatrixLinearOperator(H),
            g_flat,
            lx.LU(),
        ).value

    def update(self, params, state: NewtonState, *args):
        params_flat, unravel = ravel_pytree(params)

        def value_fn_flat(p):
            return self.func(unravel(p), *args)

        f, g_tree = self._val_and_grad(params, *args)
        g_flat, _ = ravel_pytree(g_tree)

        gnorm = jnp.linalg.norm(g_flat)
        converged = gnorm <= self.tol

        def do_step(_):
            H = (
                self._hess_fn(params, *args)
                if (not self.force_autodiff_hessian and self._hess_fn is not None)
                else jax.hessian(value_fn_flat)(params_flat)
            )

            step = self.newton_solve(
                H,
                -g_flat,
                jax.tree_util.tree_map(lambda x: -x, g_tree),
                self._hess_tag,
            )

            # Optional safeguard against ascent directions
            descent = jnp.dot(g_flat, step) < 0.0

            def accept_step(_):
                if self._line_search is not None:
                    updates, new_ls_state = self._line_search.update(
                        step,
                        state.ls_state,
                        params_flat,
                        value=f,
                        grad=g_flat,
                        value_fn=value_fn_flat,
                    )
                else:
                    updates = step
                    new_ls_state = state.ls_state

                new_params_flat = params_flat + updates

                return new_params_flat, new_ls_state

            def reject_step(_):
                # fallback to no update
                return params_flat, state.ls_state

            return jax.lax.cond(
                descent,
                accept_step,
                reject_step,
                operand=None,
            )

        def no_step(_):
            return params_flat, state.ls_state

        new_params_flat, new_ls_state = jax.lax.cond(
            converged,
            no_step,
            do_step,
            operand=None,
        )

        new_iter = jnp.where(
            converged,
            state.iter_num,
            state.iter_num + 1,
        )

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

        return unravel(new_params_flat), new_state, None

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
