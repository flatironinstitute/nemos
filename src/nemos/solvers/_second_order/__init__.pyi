"""Second-order solvers module stubs."""

from ._curvature import NewtonCurvature
from ._direction import LinearSolveDirection, ProxQuadraticDirection
from ._linesearches import ArmijoBacktracking, TsengYunBacktracking
from ._newton import Newton, ProximalNewton

__all__ = [
    "NewtonCurvature",
    "LinearSolveDirection",
    "ProxQuadraticDirection",
    "ArmijoBacktracking",
    "TsengYunBacktracking",
    "Newton",
    "ProximalNewton",
]
