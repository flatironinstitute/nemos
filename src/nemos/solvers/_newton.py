"""Newton-based optimization solvers."""

from typing import Any, Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optax

from ..typing import Params
from ._abstract_solver import OptimizationInfo
from ._hess import (
    BlockDiagonal,
    Full,
    General,
    HessianTag,
    PositiveDefinite,
    combine_hessian_tags,
)

DEFAULT_ATOL = 1e-4
DEFAULT_MAX_STEPS = 100


class NewtonState(eqx.Module):
    grad_norm: jax.Array
    stats: OptimizationInfo
    ls_state: Optional[Any] = None


NewtonStepResult = tuple[Params, NewtonState]


class Newton:
    def __init__(
        self,
        unregularized_loss: Callable,
        regularizer,
        regularizer_strength: float | None,
        has_aux: bool,
        init_params: Params | None = None,
        jit: bool = True,
        maxiter: int = DEFAULT_MAX_STEPS,
        tol: float = DEFAULT_ATOL,
    ):
        if init_params is None:
            raise ValueError(
                "init_params is required for Newton solver. "
                "It is needed to determine the parameter structure for regularization."
            )

        self.has_aux = has_aux
        self.jit = jit
        self.maxiter = maxiter
        self.tol = tol

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

        self._hess_tag: HessianTag | None = None

        self._line_search = optax.scale_by_backtracking_linesearch(
            max_backtracking_steps=30
        )

        # Cache
        self._gradient: Callable | None = None
        self._hessian: Callable | None = None

        # Linear solver + operator tags, resolved once from the Hessian tag in init_state
        self._linear_solver = lx.AutoLinearSolver(well_posed=False)
        self._operator_tags = ()

    def setup_hessian(
        self,
        hess_fn: Callable | None = None,
        hess_tag: HessianTag | None = None,
        reg_tag: HessianTag | None = None,
        property_override: Optional[type] = None,
    ):
        tag = hess_tag if reg_tag is None else combine_hessian_tags(hess_tag, reg_tag)
        if property_override is not None and tag is not None:
            tag = HessianTag(
                tag.structure, property_override, batch_axes=tag.batch_axes
            )
        self._hess_tag = tag
        self._hessian = hess_fn

    def _build_cache(self):
        if self._gradient is None:
            self._gradient = jax.value_and_grad(
                self.fun_with_aux,
                has_aux=True,
            )

        if self._hessian is None:
            self._hessian = jax.hessian(self.fun)

    def init_state(self, init_params, *args):
        if self._hess_tag is None:
            self._hess_tag = HessianTag(structure=Full, property=General)

        self._build_cache()
        ls_state = self._line_search.init(init_params)

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

    def _solve(self, grad, H):
        operator = lx.PyTreeLinearOperator(
            H,
            jax.eval_shape(lambda: grad),
            tags=self._operator_tags,
        )

        return lx.linear_solve(
            operator,
            jax.tree.map(lambda x: -x, grad),
            self._linear_solver,
        ).value

    def _newton_direction(self, grad, H):
        if self._hess_tag.structure is BlockDiagonal:
            return jax.vmap(
                self._solve,
                in_axes=(0, self._hess_tag.batch_axes),
                out_axes=self._hess_tag.batch_axes,
            )(grad, H)
        else:
            return self._solve(grad, H)

    def _apply_or_reject(
        self,
        params,
        step,
        grad,
        state: NewtonState,
        fval,
        *args,
    ):
        """Accept or reject step based on descent condition and line search."""
        descent = lx.internal.tree_dot(grad, step)

        def accept(_):
            updates, new_ls_state = self._line_search.update(
                step,
                state.ls_state,
                params,
                value=fval,
                grad=grad,
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
    ) -> NewtonStepResult:

        (fval, aux), grad = self._gradient(params, *args)
        gnorm = jnp.sqrt(lx.internal.tree_dot(grad, grad))
        converged = gnorm <= self.tol

        def step(_):
            H = self._hessian(params, *args)
            step = self._newton_direction(grad, H)

            new_params, new_ls_state = self._apply_or_reject(
                params,
                step,
                grad,
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
                reached_max_steps=new_iter >= self.maxiter,
            ),
            ls_state=new_ls_state,
        )

        return new_params, new_state, aux

    def run(
        self,
        init_params,
        *args,
    ):
        state = self.init_state(init_params, *args)
        params = init_params

        def cond(carry):
            p, s = carry
            return (~s.stats.converged) & (s.stats.num_steps < self.maxiter)

        def body(carry):
            p, s = carry
            return self.update(
                p,
                s,
                *args,
            )[:2]  # Discard aux; convergence only needs params and state

        if self.jit:
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

    @classmethod
    def get_accepted_arguments(cls) -> set[str]:
        return {"maxiter", "tol", "autodiff", "jit"}

    def _get_optim_info(
        self,
        state: NewtonState,
        **kwargs,
    ) -> OptimizationInfo:
        return state.stats
