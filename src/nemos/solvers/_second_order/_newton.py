"""Newton-based optimization solvers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic

import lineax as lx
import optax

from ..._hess import HessianTag
from ...typing import Params
from .._fista import FISTA
from ._base import AbstractSecondOrderSolver, SecondOrderState
from ._direction import (
    LinearSolveDirection,
    LinearSolverTag,
    ProxQuadraticDirection,
    validate_linear_solver,
)
from ._hessian_mixins import HessianMixin
from ._linesearches import ArmijoBacktracking, TsengYunBacktracking
from ._typing import Y

if TYPE_CHECKING:
    from ...regularizer import Regularizer
    from ._direction import AbstractDirection

DEFAULT_ATOL = 1e-4
DEFAULT_RTOL = 0.0
DEFAULT_MAX_STEPS = 100


class BaseNewtonSolver(
    AbstractSecondOrderSolver[Y, SecondOrderState[Y]], HessianMixin, Generic[Y]
):
    def __init__(
        self,
        unregularized_loss: Callable,
        regularizer: Regularizer,
        line_search: ArmijoBacktracking | TsengYunBacktracking,
        direction_factory: Callable[[HessianTag, Callable | None], AbstractDirection],
        regularizer_strength: float | None,
        has_aux: bool,
        init_params: Params | None = None,
        jit: bool = True,
        maxiter: int = DEFAULT_MAX_STEPS,
        tol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        hess_fn: Callable | None = None,
        hessian_tag: HessianTag | None = None,
        reg_tag: HessianTag | None = None,
        property_override: type | None = None,
    ) -> None:
        if init_params is None:
            raise ValueError(
                "init_params is required for Newton solver. "
                "It is needed to determine the parameter structure for regularization."
            )

        self.has_aux = has_aux
        self.jit = jit

        # A proximal solver differentiates the smooth part only and carries the penalty
        # in its proximal operator, so it must not be handed the penalized loss.
        if self._proximal:
            loss_fn = unregularized_loss
            prox = regularizer.get_proximal_operator(
                params=init_params, strength=regularizer_strength
            )
        else:
            loss_fn = regularizer.penalized_loss(
                unregularized_loss,
                params=init_params,
                strength=regularizer_strength,
            )
            prox = None

        self._set_objective(loss_fn, has_aux, maxiter, tol, rtol)
        self._line_search = line_search
        # Neither the tag nor the prox is stored: both go straight to the direction,
        # which is the object that uses them, and the properties below read them back.
        self.direction = direction_factory(
            self._init_hessian(
                regularizer,
                regularizer_strength,
                init_params,
                hess_fn=hess_fn,
                hessian_tag=hessian_tag,
                reg_tag=reg_tag,
                property_override=property_override,
            ),
            prox,
        )

    def init_state(self, init_params: Y, *args: Any) -> SecondOrderState[Y]:
        return SecondOrderState(**self._common_state_fields(init_params, *args))


class Newton(BaseNewtonSolver[Y], Generic[Y]):
    r"""
    Newton solver with backtracking and Hessian-aware linear solves.

    At each iteration, the solver computes a Newton direction from

    .. math::
        H d = -g,

    modifying the Hessian when necessary to obtain a finite descent direction.

    ``linear_solver`` controls how this system is solved:

    ``"auto"``

        Selects a strategy from the resolved Hessian properties. Positive-definite
        Hessians use Cholesky directly, positive-semidefinite Hessians use Cholesky
        with a small numerical shift, and Hessians without a positivity guarantee
        use spectral eigenvalue modification.

    ``"cholesky"``

        Solves the system using a Cholesky factorization. A positive-definite
        Hessian is used directly. For a positive-semidefinite Hessian, a small
        dtype-dependent diagonal shift is added to avoid numerical singularity.
        This is the cheapest strategy when the Hessian is known to be positive.

    ``"eigh"``

        Computes ``H = Q diag(lambda) Q.T`` and replaces each eigenvalue by
        .. math::
            \\widetilde{\\lambda}_i
            = \\max(|\\lambda_i|, \\delta).
        The direction is then
        .. math::
            d = -Q\\widetilde{\\Lambda}^{-1}Q^T g.
        This handles positive-definite, singular, and indefinite symmetric
        Hessians in one eigendecomposition. It follows the spectral modification
        in Nocedal and Wright, 2nd ed., section 3.4, equation (3.49), p. 50.

    ``"identity_shift"``

        Attempts Cholesky factorizations of ``H + tau I`` until one succeeds.
        The initial shift accounts for the smallest diagonal entry and is
        warm-started one ladder step below the shift accepted at the preceding
        iteration. Failed attempts increase the shift by a factor of ten, up to
        ``identity_shift_max_steps`` retries.
        This is an adaptation of Nocedal and Wright, Algorithm 3.3, pp. 51--52.

    Block-diagonal Hessians apply the selected strategy independently to each
    block. Accepted directions are passed through a backtracking line search.

    References
    ----------
    Jorge Nocedal and Stephen J. Wright, *Numerical Optimization*, 2nd ed.,
    Springer, 2006, section 3.4, equation (3.49) and Algorithm 3.3.
    """

    def __init__(
        self,
        unregularized_loss: Callable,
        regularizer: Regularizer,
        regularizer_strength: float | None,
        has_aux: bool,
        init_params: Params | None = None,
        jit: bool = True,
        maxiter: int = DEFAULT_MAX_STEPS,
        tol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        identity_shift_beta: float = 1e-3,
        identity_shift_max_steps: int = 20,
        linear_solver: LinearSolverTag = "auto",
        hess_fn: Callable | None = None,
        hessian_tag: HessianTag | None = None,
        reg_tag: HessianTag | None = None,
        property_override: type | None = None,
    ) -> None:
        # Before ``super().__init__``, which builds the loss, the proximal operator, the
        # line search and the Hessian wiring: a rejected argument should cost none of it.
        validate_linear_solver(linear_solver)
        self.linear_solver: LinearSolverTag = linear_solver
        if identity_shift_beta < 0:
            raise ValueError(
                "identity_shift_beta must be nonnegative; "
                f"received {identity_shift_beta}."
            )
        if identity_shift_max_steps <= 0:
            raise ValueError(
                "identity_shift_max_steps must be positive; "
                f"received {identity_shift_max_steps}."
            )
        penalized_loss = regularizer.penalized_loss(
            unregularized_loss, init_params, strength=regularizer_strength
        )
        super().__init__(
            unregularized_loss,
            regularizer,
            line_search=ArmijoBacktracking(
                optax.scale_by_backtracking_linesearch(30), penalized_loss
            ),
            direction_factory=lambda tag, _: LinearSolveDirection.from_tag(
                init_params,
                tag,
                linear_solver,
                identity_shift_beta,
                identity_shift_max_steps,
            ),
            regularizer_strength=regularizer_strength,
            has_aux=has_aux,
            init_params=init_params,
            jit=jit,
            maxiter=maxiter,
            tol=tol,
            rtol=rtol,
            hess_fn=hess_fn,
            hessian_tag=hessian_tag,
            reg_tag=reg_tag,
            property_override=property_override,
        )

    def init_state(self, init_params: Y, *args: Any) -> SecondOrderState[Y]:
        return SecondOrderState(
            **self._common_state_fields(init_params, *args),
            direction_state=self.direction.init(init_params),
        )

    @classmethod
    def get_accepted_arguments(cls) -> set[str]:
        return (
            super()
            .get_accepted_arguments()
            .union(
                {
                    "linear_solver",
                    "identity_shift_beta",
                    "identity_shift_max_steps",
                }
            )
        )


class ProximalNewton(BaseNewtonSolver[Y], Generic[Y]):
    r"""Proximal Newton solver for composite objectives.

    Minimizes :math:`f(\beta) + P(\beta)` with :math:`f` the smooth loss and :math:`P`
    a penalty reached through its proximal operator. Each iteration builds the quadratic
    model of :math:`f` and solves

    .. math::
        \min_d \; \nabla f^\top d + \tfrac{1}{2} d^\top H d + P(\beta + d)

    with :class:`~nemos.solvers._fista.FISTA`, then backtracks on the composite
    objective. This is the scheme ``glmnet`` [1]_ uses, with FISTA in place of
    coordinate descent for the inner problem; see [2]_ for the general method.

    Well-posedness of the subproblem needs two conditions:

    - :math:`H \succeq 0`, making it convex. Definiteness is only needed to invert
      :math:`H`, and here :math:`H` is only multiplied (:meth:`_hvp_block`).
    - :math:`\nabla f` restricted to :math:`\ker H` dominated by the growth of :math:`P`,
      making it bounded below. This constrains the quadratic model at the current iterate,
      not :math:`f`: a loss bounded below still has an unbounded model wherever :math:`H`
      is singular and the gradient has a component in :math:`\ker H`.

    A singular :math:`H` is therefore not by itself a problem, and no :math:`\ell_2` term
    is needed to supply the missing curvature. An indefinite :math:`H` is unsupported: the
    subproblem is unbounded below, so no solver has a minimum to find.

    Parameters
    ----------
    tol, rtol :
        Absolute and relative tolerances of the outer Cauchy criterion on the accepted
        step. Unlike :class:`Newton`, which tests ``||grad|| <= tol``, both are read
        here: see :meth:`~nemos.solvers._second_order._loop.Loop.converged`.
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
        regularizer: Regularizer,
        regularizer_strength: float | None,
        has_aux: bool,
        init_params: Params | None = None,
        jit: bool = True,
        maxiter: int = DEFAULT_MAX_STEPS,
        tol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        inner_iter: int = 100,
        inner_atol: float = 1e-8,
        inner_rtol: float = 1e-8,
        hess_fn: Callable | None = None,
        hessian_tag: HessianTag | None = None,
        reg_tag: HessianTag | None = None,
        property_override: type | None = None,
    ) -> None:
        # the penalty alone, for the composite line search. self.fun is the smooth
        # loss here, so the composite objective is fun + penalty_fn, which is
        # exactly what ``regularizer.penalized_loss`` builds from the same accessor.
        penalty_fn = regularizer.penalty_fn(
            params=init_params, strength=regularizer_strength
        )
        penalized_loss = regularizer.penalized_loss(
            unregularized_loss, init_params, strength=regularizer_strength
        )
        super().__init__(
            unregularized_loss,
            regularizer,
            line_search=TsengYunBacktracking(
                optax.scale_by_backtracking_linesearch(30), penalized_loss, penalty_fn
            ),
            # The subproblem is solved for the new parameters, so the prox is the
            # regularizer's own and the solver does not depend on the current iterate:
            # build it once rather than per outer iteration.
            direction_factory=lambda tag, prox: ProxQuadraticDirection(
                FISTA(
                    atol=inner_atol,
                    rtol=inner_rtol,
                    norm=lx.internal.two_norm,
                    prox=prox,
                    while_loop_kind="lax",
                ),
                inner_iter,
                tag,
            ),
            regularizer_strength=regularizer_strength,
            has_aux=has_aux,
            init_params=init_params,
            jit=jit,
            maxiter=maxiter,
            tol=tol,
            rtol=rtol,
            hess_fn=hess_fn,
            hessian_tag=hessian_tag,
            reg_tag=reg_tag,
            property_override=property_override,
        )

    @classmethod
    def get_accepted_arguments(cls) -> set[str]:
        return (
            super()
            .get_accepted_arguments()
            .union(
                {
                    "inner_iter",
                    "inner_atol",
                    "inner_rtol",
                }
            )
        )
