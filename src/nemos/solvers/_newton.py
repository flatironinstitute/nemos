"""Newton-based optimization solvers."""

from typing import Any, Callable

import jax
import lineax as lx

from ..typing import Params
from ._composite_mixins import CompositeQuadraticMixin
from ._hessian_mixins import HessianMixin
from ._line_search_mixins import (
    DEFAULT_ATOL,
    DEFAULT_MAX_STEPS,
    DEFAULT_RTOL,
    LineSearchLoopMixin,
    LineSearchState,
    Y,
)


class Newton(HessianMixin, LineSearchLoopMixin[Y]):
    """Newton's method with a backtracking line search on the penalized loss."""

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

        # Before ``_init_loop``: it resolves the smooth objective, and which objective
        # that is depends on ``_proximal``, which this also sets up the machinery for.
        self._init_hessian(regularizer, regularizer_strength, init_params)

        self._init_loop(
            unregularized_loss,
            regularizer,
            regularizer_strength,
            init_params,
            has_aux=has_aux,
            jit=jit,
            maxiter=maxiter,
            tol=tol,
            rtol=rtol,
        )

    def _build_cache(self) -> None:
        super()._build_cache()
        if self._hessian is None:
            self._hessian = jax.hessian(self.fun)

    def _init_curvature(self, init_params: Y) -> None:
        """Nothing is carried: the Hessian is rebuilt from the data every iteration.

        The linear solver, by contrast, is resolved once, as soon as the tag is known.
        """
        self._resolve_linear_solver(init_params)
        return None

    def _curvature(
        self, params: Y, state: LineSearchState[Y], grad: Y, *args: Any
    ) -> tuple[Any, None]:
        del state, grad
        return self._hessian(params, *args), None

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

    def _direction(self, grad: Y, H: Any, params: Y) -> Y:
        return self._block_apply(self._solve, grad, H, params)


class ProximalNewton(CompositeQuadraticMixin[Y], Newton[Y]):
    r"""Proximal Newton solver for composite objectives.

    Minimizes :math:`f(\beta) + P(\beta)` with :math:`f` the smooth loss and :math:`P`
    a penalty reached through its proximal operator. The scheme and the well-posedness
    conditions on the subproblem are documented on
    :class:`~nemos.solvers._composite_mixins.CompositeQuadraticMixin`; here the curvature
    model is the assembled Hessian, which makes this the scheme ``glmnet`` [1]_ uses,
    with FISTA in place of coordinate descent for the inner problem.

    Parameters
    ----------
    tol, rtol :
        Absolute and relative tolerances of the outer Cauchy criterion on the accepted
        step. Unlike :class:`Newton`, which tests ``||grad|| <= tol``, both are read
        here: see :meth:`~nemos.solvers._composite_mixins.CompositeQuadraticMixin._converged`.
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
    """

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
            rtol=rtol,
        )
        self._init_composite(inner_iter, inner_atol, inner_rtol)
