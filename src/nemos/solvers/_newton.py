"""Newton-based optimization solvers."""

from typing import Any, Callable, Generic, Optional, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optax

from ..tree_utils import ravel_pytree_nest
from ..typing import Params
from ._abstract_solver import OptimizationInfo, SolverProtocol, SolverState
from ._hess import (
    BlockDiagonal,
    Full,
    General,
    HessianTag,
    PositiveDefinite,
)


@runtime_checkable
class NewtonSolverProtocol(SolverProtocol[SolverState], Protocol, Generic[SolverState]):
    def setup_hessian(
        self,
        hess_fn: Callable | None,
        hess_tag: HessianTag | None,
    ) -> None: ...


class NewtonState(eqx.Module):
    grad_norm: jax.Array
    stats: OptimizationInfo
    ls_state: Optional[Any] = None


NewtonStepResult = tuple[Params, NewtonState]


class _Newton:
    def __init__(
        self,
        fun: Callable,
        tol: float = 1e-6,
    ):
        self.fun = fun
        self.tol = tol

        self._hess_tag: HessianTag | None = None
        self._hess_fn: Callable | None = None

        self._line_search = optax.scale_by_backtracking_linesearch(
            max_backtracking_steps=30
        )

        self._linear_solver = lx.LU()

    def init_state(self, init_params, *args):
        params_flat, _ = ravel_pytree_nest(init_params)

        ls_state = self._line_search.init(params_flat)

        if self._hess_tag is None:
            self._hess_tag = HessianTag(structure=Full, property=General)

        return NewtonState(
            grad_norm=jnp.inf,
            stats=OptimizationInfo(
                function_val=jnp.nan,
                num_steps=jnp.array(0),
                converged=jnp.array(False),
                reached_max_steps=jnp.array(False),
            ),
            ls_state=ls_state,
        )

    def _flat(self, params, args):
        params_flat, unravel = ravel_pytree_nest(params)

        def f(x):
            return self.fun(unravel(x), *args)

        return params_flat, unravel, f

    def newton_step(self, H, g_flat, tag: HessianTag):
        if tag.structure is BlockDiagonal and H.ndim == 3:

            def solve(Hb, gb):
                return self.newton_step(
                    Hb,
                    gb,
                    HessianTag(Full, tag.property),
                )

            return jax.vmap(solve)(H, g_flat.reshape(H.shape[0], -1))

        if tag.property is PositiveDefinite:
            return lx.linear_solve(
                lx.MatrixLinearOperator(H, lx.positive_semidefinite_tag),
                -g_flat,
                lx.Cholesky(),
            ).value

        return lx.linear_solve(
            lx.MatrixLinearOperator(H),
            -g_flat,
            lx.LU(),
        ).value

    def update(
        self,
        params,
        state: NewtonState,
        *args,
        maxiter: int,
        force_autodiff_hessian: bool,
    ) -> NewtonStepResult:

        params_flat, unravel, f = self._flat(params, args)

        fval = f(params_flat)
        g_flat = jax.grad(f)(params_flat)

        gnorm = jnp.linalg.norm(g_flat)
        converged = gnorm <= self.tol

        def step(_):
            H = (
                self._hess_fn(params, *args)
                if (self._hess_fn is not None and not force_autodiff_hessian)
                else jax.hessian(f)(params_flat)
            )

            step_flat = self.newton_step(H, g_flat, self._hess_tag)

            descent = jnp.vdot(g_flat, step_flat) < 0.0

            step_tree = unravel(step_flat)
            grad_tree = unravel(g_flat)

            def accept(_):
                updates, new_ls_state = self._line_search.update(
                    step_tree,
                    state.ls_state,
                    params,
                    value=fval,
                    grad=grad_tree,
                    value_fn=lambda p: self.fun(p, *args),
                )

                new_params = jax.tree_util.tree_map(
                    lambda p, u: p + u,
                    params,
                    updates,
                )

                return new_params, new_ls_state

            def reject(_):
                return params, state.ls_state

            return jax.lax.cond(descent, accept, reject, None)

        def no_step(_):
            return params, state.ls_state

        new_params, new_ls_state = jax.lax.cond(
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

        new_state = NewtonState(
            grad_norm=gnorm,
            stats=OptimizationInfo(
                function_val=fval,
                num_steps=new_iter,
                converged=converged,
                reached_max_steps=new_iter >= maxiter,
            ),
            ls_state=new_ls_state,
        )

        return new_params, new_state

    def run(
        self,
        init_params,
        *args,
        jit: bool = True,
        force_autodiff_hessian: bool = False,
        maxiter: int = 100,
    ):
        state = self.init_state(init_params, *args)
        params = init_params

        if jit:

            def cond(carry):
                _, s = carry
                return (~s.stats.converged) & (s.stats.num_steps < maxiter)

            def body(carry):
                p, s = carry
                return self.update(
                    p,
                    s,
                    *args,
                    maxiter=maxiter,
                    force_autodiff_hessian=force_autodiff_hessian,
                )

            params, state = eqx.internal.while_loop(
                cond,
                body,
                (params, state),
                kind="lax",
            )

        else:
            for _ in range(maxiter):
                params, state = self.update(
                    params,
                    state,
                    *args,
                    maxiter=maxiter,
                    force_autodiff_hessian=force_autodiff_hessian,
                )
                if state.stats.converged:
                    break

        return params, state


class Newton(NewtonSolverProtocol[NewtonState]):
    def __init__(
        self,
        unregularized_loss: Callable,
        regularizer,
        regularizer_strength: float | None,
        has_aux: bool,
        init_params: Params | None = None,
        jit: bool = True,
        force_autodiff_hessian: bool = False,
        maxiter: int = 100,
        **solver_kwargs,
    ):
        self.jit = jit
        self.force_autodiff_hessian = force_autodiff_hessian
        self.maxiter = maxiter
        self.has_aux = has_aux

        loss_fn = regularizer.penalized_loss(
            unregularized_loss,
            params=init_params,
            strength=regularizer_strength,
        )

        # split scalar vs aux
        if has_aux:
            self.fun_with_aux = loss_fn
            self.fun = lambda p, *a: loss_fn(p, *a)[0]
        else:
            self.fun = loss_fn
            self.fun_with_aux = lambda p, *a: (loss_fn(p, *a), None)

        self._solver = _Newton(self.fun, **solver_kwargs)

    def setup_hessian(
        self,
        hess_fn: Optional[Callable] = None,
        hess_tag: HessianTag | None = None,
        regularizer: Optional[Any] = None,
    ):
        self._solver._hess_fn = hess_fn
        self._solver._hess_tag = hess_tag

    def init_state(self, init_params: Params, *args):
        return self._solver.init_state(init_params, *args)

    def update(self, params, state, *args):
        _, aux = self.fun_with_aux(params, *args)
        return (
            *self._solver.update(
                params,
                state,
                *args,
                maxiter=self.maxiter,
                force_autodiff_hessian=self.force_autodiff_hessian,
            ),
            aux,
        )

    def run(self, init_params: Params, *args):
        params, state = self._solver.run(
            init_params,
            *args,
            jit=self.jit,
            maxiter=self.maxiter,
            force_autodiff_hessian=self.force_autodiff_hessian,
        )

        _, aux = self.fun_with_aux(params, *args)
        return params, state, aux

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
        return state.stats
