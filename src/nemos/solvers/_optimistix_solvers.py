from typing import Any, Callable

import lineax as lx
import optimistix as optx
from jaxtyping import Array, PyTree, Scalar

from ._optimistix_adapter import OptimistixAdapter


def _make_rate_scaler(
    stepsize: float | None,
    linesearch_kwargs: dict[str, Any] | None,
) -> optx.AbstractSearch:
    """
    Create fixed stepsize or linesearch based on available settings.

    If `stepsize` is not None and larger than 0, use it as a constant learning rate.
    Otherwise `BacktrackingArmijo`.
    """
    if stepsize is None or stepsize <= 0.0:
        if linesearch_kwargs is None:
            linesearch_kwargs = {}

        return optx.BacktrackingArmijo(**linesearch_kwargs)
    else:
        if linesearch_kwargs:
            raise ValueError("Only provide stepsize or linesearch_kwargs.")

        return optx.LearningRate(stepsize)


class BFGS(optx.BFGS):
    # as opposed to BacktrackingArmijo in optx.BFGS
    search: optx.AbstractSearch

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = optx.max_norm,
        use_inverse: bool = True,
        verbose: frozenset[str] = frozenset(),
        stepsize: float | None = None,
        linesearch_kwargs: dict[str, Any] | None = None,
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.use_inverse = use_inverse
        self.descent = optx.NewtonDescent(linear_solver=lx.Cholesky())
        self.search = _make_rate_scaler(stepsize, linesearch_kwargs)
        self.verbose = verbose


class NonlinearCG(optx.NonlinearCG):
    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree[Array]], Scalar] = optx.max_norm,
        method: Callable = optx._solver.nonlinear_cg.polak_ribiere,
        stepsize: float | None = None,
        linesearch_kwargs: dict[str, Any] | None = None,
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.descent = optx.NonlinearCGDescent(method=method)
        self.search = _make_rate_scaler(stepsize, linesearch_kwargs)


class OptimistixBFGS(OptimistixAdapter):
    """Adapter for optimistix.BFGS."""

    _solver_cls = BFGS


class OptimistixNonlinearCG(OptimistixAdapter):
    """Adapter for optimistix.NonlinearCG."""

    _solver_cls = NonlinearCG
