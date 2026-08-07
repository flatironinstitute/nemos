"""Newton-based optimization solvers."""

from typing import Any, Callable, ClassVar, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optax
import optimistix as optx
from optimistix._misc import cauchy_termination

from .. import tree_utils
from ..typing import Params
from ._abstract_solver import OptimizationInfo
from ._fista import FISTA
from ._hess import (
    BlockDiagonal,
    Full,
    General,
    HessianTag,
    PositiveDefinite,
    combine_hessian_tags,
)

DEFAULT_ATOL = 1e-4
DEFAULT_RTOL = 0.0
DEFAULT_MAX_STEPS = 100


class NewtonState(eqx.Module):
    grad_norm: jax.Array
    stats: OptimizationInfo
    ls_state: Optional[Any] = None
    # Previous accepted step, for solvers whose convergence test is Cauchy rather than
    # gradient based. Initialized to inf so the first iteration never counts as converged.
    y_diff: Optional[Any] = None


NewtonStepResult = tuple[Params, NewtonState]


class Newton:
    # When True, the solver minimizes a composite objective: it receives the
    # unregularized loss and reaches the penalty through its proximal operator.
    # Same convention as ``OptimistixAdapter._proximal``.
    _proximal: ClassVar[bool] = False

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
        rtol: float = DEFAULT_RTOL,
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
        self.rtol = rtol

        # kept so setup_hessian can ask the regularizer for its penalty Hessian
        self._regularizer = regularizer
        self._regularizer_strength = regularizer_strength
        self._init_params = init_params

        # A proximal solver differentiates the smooth part only and carries the penalty
        # in its proximal operator, so it must not be handed the penalized loss.
        if self._proximal:
            loss_fn = unregularized_loss
            self.prox = regularizer.get_proximal_operator(
                params=init_params, strength=regularizer_strength
            )
        else:
            loss_fn = regularizer.penalized_loss(
                unregularized_loss,
                params=init_params,
                strength=regularizer_strength,
            )
            self.prox = None

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
        # A proximal solver models the smooth part only, so the penalty contributes
        # neither curvature nor structure to the Hessian it works with.
        if self._proximal:
            reg_tag = None
        tag = hess_tag if reg_tag is None else combine_hessian_tags(hess_tag, reg_tag)
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

        # For a proximal solver the penalty is applied by the prox, not modelled by the
        # quadratic, so adding its curvature here would double-count it.
        if self._proximal:
            return hess_fn

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
            # inf so a Cauchy criterion cannot fire before the first step is taken
            y_diff=jax.tree.map(lambda x: jnp.full_like(x, jnp.inf), init_params),
        )

    def _solve(self, grad, H, params):
        # ``params`` is unused for the smooth step, which depends only on the local
        # quadratic. Proximal subclasses need it: their penalty is evaluated at
        # ``params + d``, not at ``d``.
        del params
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

    def _block_apply(self, fn, grad, H, other):
        """Apply ``fn(grad, H, other)`` once per Hessian block.

        Split out so that anything needing the block structure -- the Newton solve, or a
        subclass' Hessian-vector product -- shares one place that reads ``_hess_tag``.
        """
        if self._hess_tag.structure is BlockDiagonal:
            axes = self._hess_tag.batch_axes
            return jax.vmap(fn, in_axes=(axes, 0, axes), out_axes=axes)(grad, H, other)
        return fn(grad, H, other)

    def _newton_direction(self, grad, H, params):
        return self._block_apply(self._solve, grad, H, params)

    def _converged(self, params, state, grad, fval):
        """Convergence test. ``||grad|| <= tol`` for a smooth objective."""
        del params, state, fval
        return jnp.sqrt(lx.internal.tree_dot(grad, grad)) <= self.tol

    def _line_search_inputs(self, params, step, grad, fval, *args):
        """Value, slope and objective handed to ``self._line_search``.

        ``optax``'s backtracking search forms the slope as ``vdot(updates, grad)``; the
        vector is an argument it never differentiates, so a composite subclass can
        supply a slope accounting for its nonsmooth term without a bespoke search.
        """
        del params, step
        return fval, grad, lambda p: self.fun(p, *args)

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
        value, slope, value_fn = self._line_search_inputs(
            params, step, grad, fval, *args
        )
        descent = lx.internal.tree_dot(slope, step)

        def accept(_):
            updates, new_ls_state = self._line_search.update(
                step,
                state.ls_state,
                params,
                value=value,
                grad=slope,
                value_fn=value_fn,
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
        converged = self._converged(params, state, grad, fval)

        def step(_):
            H = self._hessian(params, *args)
            step = self._newton_direction(grad, H, params)

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
            y_diff=tree_utils.tree_sub(new_params, params),
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
        return {"maxiter", "tol", "rtol", "autodiff", "jit"}

    def _get_optim_info(
        self,
        state: NewtonState,
        **kwargs,
    ) -> OptimizationInfo:
        return state.stats


class ProximalNewton(Newton):
    r"""Proximal Newton solver for composite objectives.

    Minimizes :math:`f(\beta) + P(\beta)` with :math:`f` the smooth loss and :math:`P`
    a penalty reached through its proximal operator. Each iteration builds the quadratic
    model of :math:`f` and solves

    .. math::
        \min_d \; \nabla f^\top d + \tfrac{1}{2} d^\top H d + P(\beta + d)

    with :class:`~nemos.solvers._fista.FISTA`, then backtracks on the composite
    objective. This is the scheme ``glmnet`` [1]_ uses, with FISTA in place of
    coordinate descent for the inner problem; see [2]_ for the general method.

    Unlike :class:`Newton` the Hessian is never inverted, only multiplied, so a singular
    smooth Hessian is not by itself a problem: any quadratic part of the penalty makes
    the inner subproblem strongly convex. This suits penalties such as
    :class:`~nemos.regularizer.ElasticNet`, whose nonsmooth term puts first-order
    methods at the mercy of the design's conditioning.

    Parameters
    ----------
    inner_iter :
        Maximum FISTA steps on the subproblem. The subproblem uses the assembled
        Hessian block and touches no data, so these steps are cheap.
    inner_atol, inner_rtol :
        Tolerances for the subproblem, acting as the forcing sequence of the inexact
        proximal Newton method.

    References
    ----------
    .. [1] Friedman, J., Hastie, T., & Tibshirani, R. (2010).
        "Regularization Paths for Generalized Linear Models via Coordinate Descent."
        *Journal of Statistical Software*, 33(1), 1-22.
        https://doi.org/10.18637/jss.v033.i01
    .. [2] Lee, J. D., Sun, Y., & Saunders, M. A. (2014).
        "Proximal Newton-type methods for minimizing composite functions."
        *SIAM Journal on Optimization*, 24(3), 1420-1443.
        https://doi.org/10.1137/130921428
    """

    _proximal: ClassVar[bool] = True

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
        inner_iter: int = 100,
        inner_atol: float = 1e-8,
        inner_rtol: float = 1e-8,
    ):
        super().__init__(
            unregularized_loss,
            regularizer,
            regularizer_strength,
            has_aux,
            init_params=init_params,
            jit=jit,
            maxiter=maxiter,
            tol=tol,
        )
        # the penalty alone, for the composite line search. self.fun is the smooth
        # loss here, so the composite objective is self.fun + self._penalty.
        filter_kwargs = regularizer._get_filter_kwargs(
            strength=regularizer_strength, params=init_params
        )
        self._penalty = lambda p: regularizer._penalization(
            p, filter_kwargs=filter_kwargs
        )

        self.inner_iter = inner_iter
        self.inner_atol = inner_atol
        self.inner_rtol = inner_rtol

    def _hvp_block(self, grad, H, d):
        """Hessian-vector product for a single block."""
        del grad
        return lx.PyTreeLinearOperator(
            H, jax.eval_shape(lambda: d), tags=self._operator_tags
        ).mv(d)

    def _newton_direction(self, grad, H, params):
        r"""Minimize :math:`\nabla f^\top d + \frac12 d^\top H d + P(\beta + d)`.

        The proximal operator carries metadata defined on the whole parameter tree --
        ``GroupLasso``'s mask, or a per-feature strength -- so the subproblem is solved
        on the full tree and only the Hessian-vector product is split per block. That
        keeps every regularizer usable without slicing each one's penalty metadata.
        """

        def hvp(d):
            return self._block_apply(self._hvp_block, grad, H, d)

        def quadratic(d, _):
            return lx.internal.tree_dot(grad, d) + 0.5 * lx.internal.tree_dot(d, hvp(d))

        def prox(d, hyperparams, scaling=1.0):
            # the penalty applies to params + d, so shift the prox into d-space
            shifted = self.prox(tree_utils.tree_add(params, d), hyperparams, scaling)
            return tree_utils.tree_sub(shifted, params)

        return optx.minimise(
            quadratic,
            FISTA(
                atol=self.inner_atol,
                rtol=self.inner_rtol,
                norm=optx.two_norm,
                prox=prox,
                while_loop_kind="lax",
            ),
            y0=tree_utils.tree_zeros_like(grad),
            max_steps=self.inner_iter,
            throw=False,
        ).value

    def _converged(self, params, state, grad, fval):
        """Cauchy criterion on the accepted step, as :class:`~nemos.solvers._fista.FISTA` uses.

        A gradient-based test is unusable here: this solver differentiates the smooth
        part only, so its gradient does not vanish at the optimum of a composite
        objective, and any residual built from it inherits the curvature scale -- on
        badly conditioned data it never falls below ``tol`` even once the iterate has
        stopped moving.
        """
        return cauchy_termination(
            self.rtol,
            self.tol,
            optx.two_norm,
            params,
            state.y_diff,
            fval,
            fval - state.stats.function_val,
        )

    def _line_search_inputs(self, params, step, grad, fval, *args):
        r"""Feed the composite objective and its slope to the inherited line search.

        Tseng & Yun (2009) require the sufficient-decrease slope of a composite
        objective to be

        .. math::
            \Delta = \nabla f^\top d + P(\beta + d) - P(\beta),

        the :math:`P` difference being what makes :math:`\Delta < 0` a descent
        certificate when :math:`F` is nonsmooth. Since the search only ever forms
        ``vdot(step, slope)``, adding the penalty difference along ``step`` reproduces
        :math:`\Delta` exactly, and the stock Armijo search then applies unchanged.
        """
        penalty = self._penalty(params)
        penalty_diff = self._penalty(tree_utils.tree_add(params, step)) - penalty
        sq_norm = lx.internal.tree_dot(step, step)
        slope = tree_utils.tree_add_scalar_mul(
            grad, jnp.where(sq_norm > 0.0, penalty_diff / sq_norm, 0.0), step
        )
        return (
            fval + penalty,
            slope,
            lambda p: self.fun(p, *args) + self._penalty(p),
        )

    @classmethod
    def get_accepted_arguments(cls) -> set[str]:
        return super().get_accepted_arguments() | {
            "inner_iter",
            "inner_atol",
            "inner_rtol",
        }
