"""Newton-based optimization solvers.

This module provides second-order optimization routines based on Newton's method.
"""

from numpy.ma import ravel

from typing import Any, Callable, Optional, Generic, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from ._aux_helpers import (
    drop_aux,
    pack_args,
    wrap_aux,
)


import equinox as eqx
import lineax as lx
import optax
from ._abstract_solver import (
    AbstractSolver,
    OptimizationInfo,
    SolverProtocol,
    SolverState,
)
from ..typing import Params, StepResult


@runtime_checkable
class NewtonSolverProtocol(SolverProtocol[SolverState], Protocol, Generic[SolverState]):
    def setup_hessian(
        self,
        hess_fn: Callable | None,
        hess_tag: lx.AbstractLinearOperator | None,
    ) -> None: ...


class NewtonState(eqx.Module):
    """State of the Newton optimization process.

    Attributes
    ----------
    iter_num :
        Number of Newton iterations performed.
    converged :
        Whether the convergence criterion was satisfied.
    grad_norm :
        L2 norm of the gradient at the final iterate.
    params :
        Current parameter values (flattened array, for incremental updates).
    ls_state :
        Optax line search state (for incremental updates).
    """

    iter_num: int
    converged: bool
    grad_norm: float
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

        self._hess_fn: Callable | None = None
        self._line_search = optax.scale_by_backtracking_linesearch(
            max_backtracking_steps=30
        )

    def init_state(self, init_params, *args):
        params_flat, _ = ravel_pytree(init_params)

        ls_state = (
            self._line_search.init(params_flat)
            if self._line_search is not None
            else None
        )

        return NewtonState(
            iter_num=0,
            converged=False,
            grad_norm=jnp.inf,
            ls_state=ls_state,
        )

    def update(self, params, state: NewtonState, *args):
        params_flat, unravel = ravel_pytree(params)

        def value_fn_flat(p):
            return self.func(unravel(p), *args)

        f, g_tree = self._val_and_grad(params, *args)
        g_flat, _ = ravel_pytree(g_tree)

        gnorm = jnp.linalg.norm(g_flat)
        converged = gnorm < self.tol

        def do_step(_):
            H = (
                self._hess_fn(params, *args)
                if not self.force_autodiff_hessian and self._hess_fn is not None
                else jax.hessian(value_fn_flat)(params_flat)
            )
            step = jnp.linalg.solve(H, -g_flat)

            if self._line_search is not None:
                step, new_ls_state = self._line_search.update(
                    step,
                    state.ls_state,
                    params_flat,
                    value=f,
                    grad=g_flat,
                    value_fn=value_fn_flat,
                )
            else:
                new_ls_state = state.ls_state

            new_params_flat = params_flat + step
            return new_params_flat, new_ls_state

        def no_step(_):
            return params_flat, state.ls_state

        new_params_flat, new_ls_state = jax.lax.cond(
            converged,
            no_step,
            do_step,
            operand=None,
        )

        new_state = NewtonState(
            iter_num=state.iter_num + 1,
            converged=converged,
            grad_norm=gnorm,
            ls_state=new_ls_state,
        )

        return unravel(new_params_flat), new_state, None

    def run(self, init_params, *args):
        state = self.init_state(init_params, *args)
        params = init_params

        if self.jit:

            def cond(carry):
                state, _ = carry
                return (~state.converged) & (state.iter_num < self.maxiter)

            def body(carry):
                state, params = carry
                params, state, _ = self.update(params, state, *args)
                return state, params

            state, params = eqx.internal.while_loop(
                cond,
                body,
                (state, params),
                kind="lax",
            )
        else:
            for _ in range(self.maxiter):
                params, state, _ = self.update(params, state, *args)
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

        self._solver = _Newton(self.fun, **solver_kwargs)

    def setup_hessian(
        self,
        hess_fn: Optional[Callable] = None,
        hess_tag: lx.AbstractLinearOperator | None = None,
    ):
        self._solver._hess_fn = hess_fn

    def init_state(self, init_params: Params, *args):
        return self._solver.init_state(init_params, *args)

    def update(self, params: Params, state: NewtonState, *args):
        return self._solver.update(params, state, *args)

    def run(self, init_params: Params, *args):
        return self._solver.run(init_params, *args)

    @classmethod
    def get_accepted_arguments(cls) -> set[str]:
        return {"maxiter", "tol", "force_autodiff_hessian", "jit"}
