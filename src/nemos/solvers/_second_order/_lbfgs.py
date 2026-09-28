"""Proximal L-BFGS solver for composite objectives.

The limited-memory curvature model lives in :class:`~nemos.solvers._second_order._curvature.LBFGSCurvature`,
derived from ``optimistix`` (Apache-2.0, ``optimistix/_solver/limited_memory_bfgs.py``):
only the direct-Hessian branch is kept, the update returns the ring-buffer state rather
than a ``FunctionInfo``, and it receives the curvature pair already differenced by the
caller. ``_LBFGSHessianUpdateState`` is copied unchanged.

``optimistix.LBFGS`` is not reused whole because its loop is flat. One
``AbstractQuasiNewton.step`` is a single trial evaluation, and ``update_hessian`` sits in
the accepted branch of a ``filter_cond`` on the search's verdict, so backtracking is
unrolled into solver iterations. That pins the line search to ``AbstractSearch.step``, a
one-trial-per-call transition function; a bracketing search such as ``optax``'s zoom runs
its own inner loop and would have to be re-expressed as a state machine to fit it. The loop
here is nested instead, and the search is whichever ``optax`` transformation
``self._line_search`` holds. The curvature model is the only part that does not depend on
the loop shape, so it is the only part taken.
"""

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Optional,
    TypeVar,
)

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optax
import optimistix as optx
from jaxtyping import Array, Bool, Scalar
from optimistix._misc import cauchy_termination

from ... import tree_utils
from ...typing import Params, StepResult
from .._abstract_solver import OptimizationInfo
from .._fista import FISTA
from ._curvature import LBFGSCurvature, _LBFGSHessianUpdateState

if TYPE_CHECKING:
    from ...regularizer import Regularizer

# The parameter pytree. Both the state and the solver follow it, so a ``GLMParams`` fit
# and a ``PopulationGLM`` fit are distinct instantiations rather than ``Any``.
Y = TypeVar("Y")


DEFAULT_ATOL = 1e-4
DEFAULT_RTOL = 0.0
DEFAULT_MAX_STEPS = 100


class LBFGSState(eqx.Module, Generic[Y]):
    """State carried between outer iterations of :class:`ProximalLBFGS`."""

    grad_norm: Scalar
    stats: OptimizationInfo
    # Last accepted step, read twice per iteration: by the Cauchy criterion, and as the
    # ``s`` of the curvature pair. Initialized to zero, which the ``s^T y`` guard reads as
    # "no pair yet"; the first-iteration Cauchy hit is blocked by ``function_val`` being
    # NaN, since ``cauchy_termination`` requires both the y and the f test.
    y_diff: Y
    grad_prev: Y
    # Curvature model built from the history up to and including ``y_diff``, so it is the
    # one this iteration's subproblem uses.
    hessian_update_state: _LBFGSHessianUpdateState[Y]
    # Set when an iteration produced no usable step: the direction was not a descent
    # direction, or it was not finite. It ends the run, and it is not convergence.
    no_step_found: Bool[Array, ""]
    # optax's line-search state, whose type is private to the chosen transformation.
    ls_state: Optional[Any] = None


class ProximalLBFGS(Generic[Y]):
    r"""Proximal L-BFGS solver for composite objectives.

    Minimizes :math:`f(\beta) + P(\beta)` with :math:`f` the smooth loss and :math:`P` a
    penalty reached through its proximal operator, as
    :class:`~nemos.solvers._newton.ProximalNewton` does, but with the assembled Hessian
    replaced by the limited-memory approximation :math:`B_k` of Byrd et al. [1]_,
    Algorithm 3.2. Each iteration solves

    .. math::
        \min_d \; \nabla f^\top d + \tfrac{1}{2} d^\top B_k d + P(\beta + d)

    with :class:`~nemos.solvers._fista.FISTA`, then backtracks on the composite objective;
    see [2]_ for the method and [3]_ for the composite Armijo slope.

    :math:`B_k` is built from the last ``history_length`` pairs
    :math:`(s, y) = (\beta_k - \beta_{k-1},\, \nabla f_k - \nabla f_{k-1})`, keeping only
    those with :math:`s^\top y > \varepsilon` so that :math:`B_k \succ 0` and the
    subproblem stays convex. A rejected step leaves :math:`s = 0` and so leaves the
    history untouched. With an empty history :math:`B_0 = I`, making the first iteration a
    unit-stepsize proximal-gradient step whose scale the line search has to supply.

    The subproblem consumes :math:`B_k v` only and never factorizes :math:`B_k`, so this
    solver needs no Hessian from the model and carries no
    :class:`~nemos._hess.HessianTag`. A ``PopulationGLM`` fit therefore builds one
    curvature model over all neurons rather than one per block.

    Parameters
    ----------
    history_length :
        Number of curvature pairs retained. Larger values track curvature better and cost
        :math:`O(mpK)` memory and :math:`O(mpK)` work per operator application.
    tol, rtol :
        Absolute and relative tolerances of the outer Cauchy criterion on the accepted
        step; both are read, see :meth:`_converged`.
    inner_iter :
        Maximum FISTA steps on the subproblem. The subproblem applies the curvature model
        and touches no data, so these steps are cheap.
    inner_atol, inner_rtol :
        Tolerances for the subproblem, acting as the forcing sequence of the inexact
        proximal quasi-Newton method.

    References
    ----------
    .. [1] Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994).
        "Representations of quasi-Newton matrices and their use in limited memory
        methods." *Mathematical Programming*, 63(1), 129-156.
        https://doi.org/10.1007/BF01582063
    .. [2] Lee, J. D., Sun, Y., & Saunders, M. A. (2014).
        "Proximal Newton-type methods for minimizing composite functions."
        *SIAM Journal on Optimization*, 24(3), 1420-1443.
        https://doi.org/10.1137/130921428
    .. [3] Tseng, P., & Yun, S. (2009).
        "A coordinate gradient descent method for nonsmooth separable minimization."
        *Mathematical Programming*, 117(1-2), 387-423.
        https://doi.org/10.1007/s10107-007-0170-0
    """

    _proximal: ClassVar[bool] = True

    def __init__(
        self,
        unregularized_loss: Callable,
        regularizer: "Regularizer",
        regularizer_strength: float | None,
        has_aux: bool,
        init_params: Params | None = None,
        history_length: int = 10,
        jit: bool = True,
        maxiter: int = DEFAULT_MAX_STEPS,
        tol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        inner_iter: int = 100,
        inner_atol: float = 1e-8,
        inner_rtol: float = 1e-8,
    ):

        if init_params is None:
            raise ValueError(
                "init_params is required for ProximalLBFGS solver. "
                "It is needed to determine the parameter structure for regularization."
            )

        self.has_aux = has_aux
        self.jit = jit
        self.maxiter = maxiter
        self.tol = tol
        self.rtol = rtol
        self._curvature = LBFGSCurvature[Y](history_length=history_length)

        # the penalty alone, for the composite line search. self.fun is the smooth
        # loss here, so the composite objective is self.fun + self._penalty, which is
        # exactly what ``regularizer.penalized_loss`` builds from the same accessor.
        self._penalty = regularizer.penalty_fn(
            params=init_params, strength=regularizer_strength
        )

        self.inner_iter = inner_iter
        self.inner_atol = inner_atol
        self.inner_rtol = inner_rtol

        self.prox = regularizer.get_proximal_operator(
            params=init_params, strength=regularizer_strength
        )

        # The subproblem is solved for the new parameters, so the prox is the
        # regularizer's own and the solver does not depend on the current iterate:
        # build it once rather than per outer iteration.
        self._inner_solver = FISTA(
            atol=inner_atol,
            rtol=inner_rtol,
            norm=lx.internal.two_norm,
            prox=self.prox,
            while_loop_kind="lax",
        )
        # split scalar vs aux
        if has_aux:
            self.fun_with_aux = unregularized_loss
            self.fun = lambda p, *a: unregularized_loss(p, *a)[0]
        else:
            self.fun = unregularized_loss
            self.fun_with_aux = lambda p, *a: (unregularized_loss(p, *a), None)

        self._line_search = optax.scale_by_backtracking_linesearch(
            max_backtracking_steps=30
        )

        # Cache
        self._gradient: Callable | None = None

    def _build_cache(self) -> None:
        if self._gradient is None:
            self._gradient = jax.value_and_grad(
                self.fun_with_aux,
                has_aux=True,
            )

    def init_state(self, init_params: Y, *args: Any) -> LBFGSState[Y]:
        self._build_cache()
        hessian_state = self._curvature.init(init_params)
        state = LBFGSState(
            ls_state=self._line_search.init(init_params),
            grad_norm=jnp.array(jnp.inf),
            stats=OptimizationInfo(
                function_val=jnp.array(jnp.nan),
                num_steps=jnp.array(0),
                converged=jnp.array(False),
                reached_max_steps=jnp.array(False),
            ),
            y_diff=jax.tree.map(jnp.zeros_like, init_params),
            grad_prev=jax.tree.map(jnp.zeros_like, init_params),
            hessian_update_state=hessian_state,
            no_step_found=jnp.array(False),
        )
        return state

    def _lbfgs_direction(
        self, params: Y, grad: Y, state: _LBFGSHessianUpdateState[Y]
    ) -> Y:
        r"""Minimize :math:`\nabla f^\top (z - \beta) + \frac12 (z - \beta)^\top H (z - \beta) + P(z)`.

        Solving for the new parameters :math:`z` rather than the step keeps the penalty
        where it is defined, so ``self.prox`` applies unchanged and the inner solver does
        not depend on the current iterate.

        The proximal operator carries metadata defined on the whole parameter tree --
        ``GroupLasso``'s mask, or a per-feature strength -- so the subproblem is solved
        on the full tree and only the Hessian-vector product is split per block. That
        keeps every regularizer usable without slicing each one's penalty metadata.
        """

        def quadratic(z, _):
            step = tree_utils.tree_sub(z, params)
            hvp = self._curvature.hvp(state, params, step)
            return lx.internal.tree_dot(grad, step) + 0.5 * lx.internal.tree_dot(
                step, hvp
            )

        new_params = optx.minimise(
            quadratic,
            self._inner_solver,
            y0=params,
            max_steps=self.inner_iter,
            throw=False,
        ).value
        # ``_apply_or_reject`` scales and adds the result, so return the step
        return tree_utils.tree_sub(new_params, params)

    def _apply_or_reject(
        self,
        params: Y,
        step: Y,
        grad: Y,
        state: LBFGSState[Y],
        fval: Scalar,
        *args: Any,
    ) -> tuple[Y, Any, Bool[Array, ""]]:
        """Accept or reject step based on descent condition and line search.

        Returns whether the step was taken, so the caller can tell a rejection from a
        null step; see :attr:`LBFGSState.no_step_found`.
        """
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

        # A zero slope means the iterate is stationary, a positive one that the direction
        # is unusable, and a NaN one that the subproblem diverged. Report
        # ``no_step_found``.
        take_step = descent < 0
        new_params, new_ls_state = jax.lax.cond(take_step, accept, reject, None)
        return new_params, new_ls_state, take_step

    def _converged(
        self, params: Y, state: LBFGSState[Y], grad: Y, fval: Scalar
    ) -> Bool[Array, ""]:
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
            lx.internal.two_norm,
            params,
            state.y_diff,
            fval,
            fval - state.stats.function_val,
        )

    def _line_search_inputs(
        self, params: Y, step: Y, grad: Y, fval: Scalar, *args: Any
    ) -> tuple[Scalar, Y, Callable[[Y], Scalar]]:
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
        return {
            "maxiter",
            "tol",
            "rtol",
            "jit",
            "history_length",
            "inner_iter",
            "inner_atol",
            "inner_rtol",
        }

    def _get_optim_info(self, state: LBFGSState[Y], **kwargs) -> OptimizationInfo:
        return state.stats

    def update(
        self,
        params: Y,
        state: LBFGSState[Y],
        *args: Any,
    ) -> StepResult:

        (fval, aux), grad = self._gradient(params, *args)

        gnorm = jnp.sqrt(lx.internal.tree_dot(grad, grad))
        converged = self._converged(params, state, grad, fval)

        def step(_):
            new_hessian_state = self._curvature.update(
                state.hessian_update_state,
                params,
                state.y_diff,
                tree_utils.tree_sub(grad, state.grad_prev),
                *args,
            )
            step = self._lbfgs_direction(params, grad, new_hessian_state)

            new_params, new_ls_state, took_step = self._apply_or_reject(
                params,
                step,
                grad,
                state,
                fval,
                *args,
            )

            # ``grad_prev`` is written here rather than in the loop: this is the point at
            # which the gradient has been folded into the history, so it is what the next
            # iteration must difference against.
            return (
                new_params,
                eqx.tree_at(
                    lambda x: (x.ls_state, x.hessian_update_state, x.grad_prev),
                    state,
                    (new_ls_state, new_hessian_state, grad),
                ),
                took_step,
            )

        def no_step(_):
            return params, state, jnp.array(False)

        new_params, new_state, took_step = jax.lax.cond(
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
                (~converged) & (~took_step),
            ),
        )
        return new_params, new_state, aux

    @eqx.filter_jit
    def run(
        self,
        init_params: Y,
        *args: Any,
    ) -> StepResult:
        state = self.init_state(init_params, *args)

        def cond(carry):
            _, s = carry
            return (
                (~s.stats.converged)
                & (~s.no_step_found)
                & (s.stats.num_steps < self.maxiter)
            )

        def body(carry):
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

        _, aux = self.fun_with_aux(final_params, *args)
        return final_params, final_state, aux
