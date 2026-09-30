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

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    TypeVar,
)

import jax
import jax.numpy as jnp
import lineax as lx
import optax

from ...typing import Params
from .._fista import FISTA
from ._base import AbstractSecondOrderSolver, SecondOrderState
from ._curvature import LBFGSCurvature
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


class ProximalLBFGS(AbstractSecondOrderSolver[Y, SecondOrderState[Y]], Generic[Y]):
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
        step; both are read, see
        :meth:`~nemos.solvers._second_order._loop.Loop.converged`.
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
        regularizer: Regularizer,
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
    ) -> None:

        if init_params is None:
            raise ValueError(
                "init_params is required for ProximalLBFGS solver. "
                "It is needed to determine the parameter structure for regularization."
            )

        self.has_aux = has_aux
        self.jit = jit
        self.curvature = LBFGSCurvature[Y](history_length=history_length)

        prox = regularizer.get_proximal_operator(
            params=init_params, strength=regularizer_strength
        )

        self._set_objective(unregularized_loss, has_aux)

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

        # The subproblem is solved for the new parameters, so the prox is the
        # regularizer's own and the solver does not depend on the current iterate:
        # build it once rather than per outer iteration. The curvature model is never
        # assembled, so the direction carries no tag.
        self.direction = ProxQuadraticDirection(
            FISTA(
                atol=inner_atol,
                rtol=inner_rtol,
                norm=lx.internal.two_norm,
                prox=prox,
                while_loop_kind="lax",
            ),
            inner_iter,
            None,
        )
        self.loop = Loop(
            atol=tol,
            rtol=rtol,
            maxiter=maxiter,
            fval_diff_fn=lambda fx, s: fx - s.stats.function_val,
            fval_and_grad_fn=jax.value_and_grad(self.fun_with_aux, has_aux=True),
        )

    def _initial_y_diff(self, init_params: Y) -> Y:
        """Zero rather than infinite: this is also the ``s`` of the first curvature pair.

        The ``s^T y`` guard reads a zero pair as "no pair yet" and leaves the history
        untouched. The first-iteration Cauchy hit that a zero step would otherwise cause
        is blocked by ``function_val`` being NaN, since the criterion needs both arms.
        """
        return jax.tree.map(jnp.zeros_like, init_params)

    def init_state(self, init_params: Y, *args: Any) -> SecondOrderState[Y]:
        return SecondOrderState(
            **self._common_state_fields(init_params, *args),
            hessian_update_state=self.curvature.init(init_params),
        )

    @classmethod
    def get_accepted_arguments(cls) -> set[str]:
        return super().get_accepted_arguments() | {
            "history_length",
            "inner_iter",
            "inner_atol",
            "inner_rtol",
        }
