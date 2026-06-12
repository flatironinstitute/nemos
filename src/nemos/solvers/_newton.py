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
    combine_hessian_tags,
)


@runtime_checkable
class NewtonSolverProtocol(SolverProtocol[SolverState], Protocol, Generic[SolverState]):
    autodiff: bool

    def setup_hessian(
        self,
        hess_fn: Callable | None,
        hess_tag: HessianTag | None,
        reg_tag: HessianTag | None = None,
        property_override: type | None = None,
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
        fun_with_aux: Callable,
        tol: float = 1e-6,
    ):
        self.fun = fun
        self.fun_with_aux = fun_with_aux
        self.tol = tol

        self._hess_tag: HessianTag | None = None
        self._hess_fn: Callable | None = None

        self._line_search = optax.scale_by_backtracking_linesearch(
            max_backtracking_steps=30
        )

        # Cached closures and transforms, populated on first init_state
        self._unravel: Optional[Callable] = None
        self._fun_flat: Optional[Callable] = None
        self._fun_flat_with_aux: Optional[Callable] = None
        self._vag_flat_with_aux: Optional[Callable] = None
        self._hessian_flat: Optional[Callable] = None

        # Linear solver + operator tags, resolved once from the Hessian tag in init_state
        self._linear_solver = lx.AutoLinearSolver(well_posed=False)
        self._operator_tags = ()

    def _build_cache(self, init_params):
        """Build and cache flattened functions and autodiff transforms."""
        _, self._unravel = ravel_pytree_nest(init_params)
        self._fun_flat = lambda x, *a: self.fun(self._unravel(x), *a)
        self._fun_flat_with_aux = lambda x, *a: self.fun_with_aux(self._unravel(x), *a)
        self._vag_flat_with_aux = jax.value_and_grad(
            self._fun_flat_with_aux, has_aux=True
        )
        self._hessian_flat = jax.hessian(self._fun_flat)

    def init_state(self, init_params, *args):
        if self._unravel is None:
            self._build_cache(init_params)

        params_flat, _ = ravel_pytree_nest(init_params)

        ls_state = self._line_search.init(params_flat)

        if self._hess_tag is None:
            self._hess_tag = HessianTag(structure=Full, property=General)

        # Resolve the linear solver once: Cholesky for positive-definite Hessians,
        # otherwise a robust least-squares solve that tolerates rank deficiency.
        if self._hess_tag.property is PositiveDefinite:
            self._linear_solver = lx.Cholesky()
            self._operator_tags = lx.positive_semidefinite_tag
        else:
            self._linear_solver = lx.AutoLinearSolver(well_posed=False)
            self._operator_tags = ()

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

    def _newton_direction(self, g_flat, H, tag: HessianTag):
        """Compute Newton step direction from gradient and Hessian."""
        if tag.structure is BlockDiagonal and H.ndim == 3:

            def solve(Hb, gb):
                return self._newton_direction(
                    gb,
                    Hb,
                    HessianTag(Full, tag.property),
                )

            return jax.vmap(solve)(H, g_flat.reshape(H.shape[0], -1))

        # Solver and operator tags were resolved once from the Hessian tag in
        # init_state; reuse them rather than re-inferring per iteration.
        operator = lx.MatrixLinearOperator(H, self._operator_tags)
        return lx.linear_solve(operator, -g_flat, self._linear_solver).value

    def _apply_or_reject(
        self,
        params,
        step_tree,
        grad_tree,
        state: NewtonState,
        fval,
        *args,
    ):
        """Accept or reject step based on descent condition and line search."""
        params_flat, _ = ravel_pytree_nest(params)
        grad_flat, _ = ravel_pytree_nest(grad_tree)
        step_flat, _ = ravel_pytree_nest(step_tree)
        descent = jnp.vdot(grad_flat, step_flat) < 0.0

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

        new_params, new_ls_state = jax.lax.cond(descent, accept, reject, None)
        return new_params, new_ls_state

    def update(
        self,
        params,
        state: NewtonState,
        *args,
        maxiter: int,
    ) -> NewtonStepResult:

        params_flat, _ = ravel_pytree_nest(params)

        (fval, aux), g_flat = self._vag_flat_with_aux(params_flat, *args)

        gnorm = jnp.linalg.norm(g_flat)
        converged = gnorm <= self.tol

        def step(_):
            H = (
                self._hess_fn(params, *args)
                if self._hess_fn is not None
                else self._hessian_flat(params_flat, *args)
            )

            step_flat = self._newton_direction(g_flat, H, self._hess_tag)
            step_tree = self._unravel(step_flat)
            grad_tree = self._unravel(g_flat)

            new_params, new_ls_state = self._apply_or_reject(
                params,
                step_tree,
                grad_tree,
                state,
                fval,
                *args,
            )

            return new_params, new_ls_state

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

        return new_params, new_state, aux

    def run(
        self,
        init_params,
        *args,
        jit: bool = True,
        maxiter: int = 100,
    ):
        state = self.init_state(init_params, *args)
        params = init_params

        def cond(carry):
            p, s = carry
            return (~s.stats.converged) & (s.stats.num_steps < maxiter)

        def body(carry):
            p, s = carry
            return self.update(
                p,
                s,
                *args,
                maxiter=maxiter,
            )[
                :2
            ]  # Discard aux; convergence only needs params and state

        if jit:
            final_params, final_state = eqx.internal.while_loop(
                cond,
                body,
                (params, state),
                kind="lax",
            )
        else:
            carry = (params, state)
            while cond(carry):
                carry = body(carry)
            final_params, final_state = carry

        _, aux = self.fun_with_aux(final_params, *args)
        return final_params, final_state, aux


class Newton(NewtonSolverProtocol[NewtonState]):
    def __init__(
        self,
        unregularized_loss: Callable,
        regularizer,
        regularizer_strength: float | None,
        has_aux: bool,
        init_params: Params | None = None,
        **solver_init_kwargs,
    ):
        if init_params is None:
            raise ValueError(
                "init_params is required for Newton solver. "
                "It is needed to determine the parameter structure for regularization."
            )

        self.jit = solver_init_kwargs.get("jit", False)
        self.autodiff = solver_init_kwargs.get("autodiff", False)
        self.maxiter = solver_init_kwargs.get("maxiter", False)
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

        self._solver = _Newton(self.fun, self.fun_with_aux, **solver_init_kwargs)

    def setup_hessian(
        self,
        hess_fn: Optional[Callable] = None,
        hess_tag: HessianTag | None = None,
        reg_tag: HessianTag | None = None,
        property_override: Optional[type] = None,
    ):
        # ``reg_tag`` is the regularizer's coverage-resolved tag; combine it with the
        # model's loss tag, then let the model override the definiteness when it can
        # certify more than coverage alone (e.g. GLM + Ridge is positive definite).
        tag = hess_tag if reg_tag is None else combine_hessian_tags(hess_tag, reg_tag)
        if property_override is not None and tag is not None:
            tag = HessianTag(tag.structure, property_override)
        self._solver._hess_fn = hess_fn
        self._solver._hess_tag = tag

    def init_state(self, init_params: Params, *args):
        return self._solver.init_state(init_params, *args)

    def update(self, params, state, *args):
        return self._solver.update(
            params,
            state,
            *args,
            maxiter=self.maxiter,
        )

    def run(self, init_params: Params, *args):
        return self._solver.run(
            init_params,
            *args,
            jit=self.jit,
            maxiter=self.maxiter,
        )

    @classmethod
    def get_accepted_arguments(cls) -> set[str]:
        return {
            "maxiter",
            "tol",
            "autodiff",
            "jit",
        }

    def _get_optim_info(
        self,
        state: NewtonState,
        **kwargs,
    ) -> OptimizationInfo:
        return state.stats
