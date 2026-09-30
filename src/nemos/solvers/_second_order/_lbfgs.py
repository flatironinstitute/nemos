"""Proximal L-BFGS solver for composite objectives.

The limited-memory curvature model lives in :class:`~nemos.solvers._second_order._curvature.LBFGSCurvature`,
derived from ``optimistix`` (Apache-2.0, ``optimistix/_solver/limited_memory_bfgs.py``):
only the direct-Hessian branch is kept, the update returns the ring-buffer state rather
than a ``FunctionInfo``, and it receives the curvature pair already differenced by the
caller. ``_LBFGSHessianUpdateState`` is copied with one field added, ``initial_scale``,
which carries the scale of :math:`B_0` while the history is empty.

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
from jaxtyping import Array, Bool, Scalar
from optimistix._misc import cauchy_termination

from ... import tree_utils
from ...typing import Params, StepResult
from .._abstract_solver import OptimizationInfo
from .._fista import FISTA
from ._curvature import LBFGSCurvature, _LBFGSHessianUpdateState
from ._direction import ProxQuadraticDirection
from ._linesearches import TsengYunBacktracking
from ._loop import Loop

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
    # direction state (shift_index for Newton, None for the ProxNewton & ProxLBFGS)
    direction_state: Optional[Any] = None


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
    history untouched. With an empty history :math:`B_0 = \|\nabla f\| I`, making the first
    iteration a unit-length proximal-gradient step; see
    :meth:`~nemos.solvers._second_order._curvature.LBFGSCurvature.hvp` for why it is not
    left at :math:`I`.

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
        self.curvature = LBFGSCurvature[Y](history_length=history_length)

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

        # the penalty alone, for the composite line search. self.fun is the smooth
        # loss here, so the composite objective is self.fun + self._penalty, which is
        # exactly what ``regularizer.penalized_loss`` builds from the same accessor.
        penalty_fn = regularizer.penalty_fn(
            params=init_params, strength=regularizer_strength
        )
        penalized_loss = regularizer.penalized_loss(
            self.fun, init_params, strength=regularizer_strength
        )
        backtrack = optax.scale_by_backtracking_linesearch(30)
        self._line_search = TsengYunBacktracking(backtrack, penalized_loss, penalty_fn)

        # Cache
        self._gradient: Callable | None = None
        self.direction = ProxQuadraticDirection(self._inner_solver, inner_iter, None)
        fval_and_grad = jax.value_and_grad(
            self.fun_with_aux,
            has_aux=True,
        )
        self.loop = Loop(
            atol=self.tol,
            rtol=self.rtol,
            maxiter=self.maxiter,
            fval_diff_fn=lambda fx, s: fx - s.stats.function_val,
            fval_and_grad_fn=fval_and_grad,
            grad_diff_fn=lambda g, s: tree_utils.tree_sub(g, s.grad_prev),
        )

    def _build_cache(self) -> None:
        if self._gradient is None:
            self._gradient = jax.value_and_grad(
                self.fun_with_aux,
                has_aux=True,
            )

    def _scalar_dtype(self, init_params: Y, *args: Any):
        """The objective's dtype, which the state's scalars must already carry.

        The ``while_loop`` carry fails to typecheck otherwise.
        """
        return jax.eval_shape(self.fun, init_params, *args).dtype

    def init_state(self, init_params: Y, *args: Any) -> LBFGSState[Y]:
        self._build_cache()
        hessian_state = self.curvature.init(init_params)
        scalar_dtype = self._scalar_dtype(init_params, *args)
        state = LBFGSState(
            ls_state=self._line_search.init(init_params),
            grad_norm=jnp.asarray(jnp.inf, dtype=scalar_dtype),
            stats=OptimizationInfo(
                function_val=jnp.asarray(jnp.nan, dtype=scalar_dtype),
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

        Returns the value of :attr:`LBFGSState.no_step_found` for this iteration: true
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
        return self.loop.update(
            params, state, self.curvature, self.direction, self._line_search, *args
        )

    def run(
        self,
        init_params: Y,
        *args: Any,
    ) -> StepResult:
        """Iterate to convergence, to a stall, or to ``maxiter``.

        ``jit`` picks which of the two loops in :meth:`_run` executes, so the compiled
        path has to be reached through a separate method: decorating this one would trace
        the Python loop and fail on its data-dependent condition.
        """
        if self.jit:
            return self._run_jit(init_params, *args)
        return self._run(init_params, *args)

    @eqx.filter_jit
    def _run_jit(
        self,
        init_params: Y,
        *args: Any,
    ) -> StepResult:
        return self._run(init_params, *args)

    def _run(
        self,
        init_params: Y,
        *args: Any,
    ) -> StepResult:
        state = self.init_state(init_params, *args)
        final_params, final_state = self.loop.run(
            init_params,
            state,
            self.curvature,
            self.direction,
            self._line_search,
            self.jit,
            *args,
        )
        _, aux = self.fun_with_aux(final_params, *args)
        return final_params, final_state, aux
