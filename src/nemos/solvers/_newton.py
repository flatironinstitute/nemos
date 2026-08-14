"""Newton-based optimization solvers."""

import math
from typing import Any, Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optax

from .. import tree_utils
from ..typing import Params
from ._abstract_solver import OptimizationInfo
from ._hess import (
    BlockDiagonal,
    Full,
    General,
    HessianTag,
    PositiveDefinite,
    PositiveSemiDefinite,
    combine_hessian_tags,
)

DEFAULT_ATOL = 1e-4
DEFAULT_MAX_STEPS = 100


class NewtonState(eqx.Module):
    grad_norm: jax.Array
    newton_decrement: jax.Array
    diverged: jax.numpy.ndarray
    stats: OptimizationInfo
    ls_state: Optional[Any] = None


NewtonStepResult = tuple[Params, NewtonState]


def _compute_diagonal_shift(H, shift_const):
    leaves_with_paths = jax.tree_util.tree_leaves_with_path(H)
    diag_maxes = [
        jnp.max(jnp.abs(jnp.diagonal(leaf, axis1=-2, axis2=-1)))
        for path, leaf in leaves_with_paths
        if (n := len(path)) % 2 == 0 and path[: n // 2] == path[n // 2 :]
    ]
    max_diag = jax.tree_util.tree_reduce(jnp.maximum, diag_maxes)
    eps = max(jnp.finfo(leaf.dtype).eps for leaf in jax.tree.leaves(H))
    return shift_const * jnp.where(max_diag > 0, max_diag, 1.0) * eps**0.5


def _add_diagonal_shift(H, tau):
    def damp(path, h):
        n = len(path)
        if n % 2 != 0 or path[: n // 2] != path[n // 2 :]:
            return h
        size = math.prod(h.shape[: h.ndim // 2]) if h.ndim > 0 else 1
        return (
            h + tau
            if h.ndim == 0
            else h + tau * jnp.eye(size, dtype=h.dtype).reshape(h.shape)
        )

    return jax.tree.map_with_path(damp, H)


class NewtonCholesky:
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
        shift_const: float = 1.0,
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

        # kept so setup_hessian can ask the regularizer for its penalty Hessian
        self._regularizer = regularizer
        self._regularizer_strength = regularizer_strength
        self._init_params = init_params

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

        self._linear_solver = lx.Cholesky()
        self._operator_tags = lx.positive_semidefinite_tag
        self._shift_fn: Callable | None = None
        self._shift_const: float = shift_const

    def setup_hessian(
        self,
        hess_fn: Callable | None = None,
        hess_tag: HessianTag | None = None,
        reg_tag: HessianTag | None = None,
        property_override: Optional[type] = None,
    ):
        tag = hess_tag if reg_tag is None else combine_hessian_tags(hess_tag, reg_tag)
        if tag is not None and tag.property not in (
            PositiveDefinite,
            PositiveSemiDefinite,
        ):
            raise ValueError(
                "NewtonCholesky requires a positive (semi)definite Hessian; use the Newton solver for the general case."
            )
        if property_override is not None and tag is not None:
            tag = HessianTag(
                tag.structure, property_override, batch_axes=tag.batch_axes
            )
        self._hess_tag = tag
        self._hessian = self._penalize_hessian(hess_fn, hess_tag)

    def _penalize_hessian(self, hess_fn, model_tag):
        """Add the regularizer's penalty Hessian to the model's likelihood Hessian.

        Models supply the second derivative of the likelihood alone. Adding the penalty's
        is valid because ``Regularizer.penalized_loss`` returns ``loss + penalty``, and the
        second derivative of a sum is the sum of the second derivatives.

        ``None`` passes through: without a model-supplied Hessian, ``_build_cache``
        autodiffs ``self.fun``, which is the penalized loss and already carries the penalty.

        The batching comes from ``model_tag`` rather than the combined tag, because whether
        the Hessian is assembled one block per neuron is a property of the model.
        """
        if hess_fn is None:
            return None

        batch_axes = (
            model_tag.batch_axes
            if model_tag is not None and model_tag.structure is BlockDiagonal
            else None
        )
        penalty_hess_fn = self._regularizer._get_hess_fn(
            self._init_params, self._regularizer_strength, batch_axes=batch_axes
        )
        if penalty_hess_fn is None:
            # the regularizer declares no curvature, so the likelihood term is the whole
            return hess_fn

        def penalized_hessian(params, *args):
            return tree_utils.tree_add(hess_fn(params, *args), penalty_hess_fn(params))

        return penalized_hessian

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

        # Resolve the shift and convergence policies
        if self._hess_tag.property is PositiveDefinite:
            self._shift_fn = lambda H: jnp.zeros((), jax.tree.leaves(H)[0].dtype)
            self._converged_fn = lambda lambda_sq, gnorm: 0.5 * lambda_sq <= self.tol
        else:
            self._shift_fn = lambda H: _compute_diagonal_shift(H, self._shift_const)
            self._converged_fn = lambda lambda_sq, gnorm: gnorm <= self.tol

        return NewtonState(
            grad_norm=jnp.inf,
            newton_decrement=jnp.array(0.0),
            diverged=jnp.array(False),
            stats=OptimizationInfo(
                function_val=jnp.nan,
                num_steps=jnp.array(0),
                converged=jnp.array(False),
                reached_max_steps=jnp.array(False),
            ),
            ls_state=ls_state,
        )

    def _solve(self, grad, H):
        tau = self._shift_fn(H)
        H_mod = _add_diagonal_shift(H, tau)

        operator = lx.PyTreeLinearOperator(
            H_mod,
            jax.eval_shape(lambda: grad),
            tags=self._operator_tags,
        )
        step = lx.linear_solve(
            operator,
            jax.tree.map(lambda x: -x, grad),
            self._linear_solver,
        ).value
        lambda_sq = -sum(
            jnp.vdot(g, s) for g, s in zip(jax.tree.leaves(grad), jax.tree.leaves(step))
        )
        return step, lambda_sq

    def _newton_direction(self, grad, H, tag):
        if tag.structure is BlockDiagonal:
            step, lambda_sq_per_block = jax.vmap(
                self._solve,
                in_axes=(tag.batch_axes, 0),
                out_axes=(tag.batch_axes, 0),
            )(grad, H)
            return step, jnp.sum(lambda_sq_per_block)
        else:
            return self._solve(grad, H)

    def _apply(
        self,
        params,
        step,
        grad,
        state: NewtonState,
        fval,
        *args,
    ):
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

    def update(
        self,
        params,
        state: NewtonState,
        *args,
    ) -> NewtonStepResult:
        (fval, aux), grad = self._gradient(params, *args)
        H = self._hessian(params, *args)
        step, lambda_sq = self._newton_direction(grad, H, self._hess_tag)

        gnorm = jnp.sqrt(lx.internal.tree_dot(grad, grad))
        converged = self._converged_fn(lambda_sq, gnorm)

        def do_step(_):
            return self._apply(
                params,
                step,
                grad,
                state,
                fval,
                *args,
            )

        def no_step(_):
            return params, state.ls_state

        new_params, new_ls_state = jax.lax.cond(
            converged,
            no_step,
            do_step,
            None,
        )

        new_params_flat = jax.tree.leaves(new_params)
        diverged = ~jnp.all(
            jnp.isfinite(jnp.concatenate([x.ravel() for x in new_params_flat]))
        )

        new_iter = jnp.where(
            converged,
            state.stats.num_steps,
            state.stats.num_steps + 1,
        )
        new_state = NewtonState(
            grad_norm=gnorm,
            newton_decrement=jnp.sqrt(lambda_sq),
            diverged=diverged,
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
            return (
                (~s.stats.converged)
                & (~s.diverged)
                & (s.stats.num_steps < self.maxiter)
            )

        def body(carry):
            p, s = carry
            return self.update(
                p,
                s,
                *args,
            )[
                :2
            ]  # Discard aux; convergence only needs params and state

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
        return {"maxiter", "tol", "autodiff", "jit", "shift_const"}

    def _get_optim_info(
        self,
        state: NewtonState,
        **kwargs,
    ) -> OptimizationInfo:
        return state.stats
