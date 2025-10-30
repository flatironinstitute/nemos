"""Registry for mapping from solver name to concrete implementation."""

from typing import Type

from ._jaxopt_solvers import (
    JaxoptBFGS,
    JaxoptGradientDescent,
    JaxoptLBFGS,
    JaxoptNonlinearCG,
    JaxoptProximalGradient,
)
from ._svrg import WrappedProxSVRG, WrappedSVRG

solver_registry: dict[str, Type] = {
    "GradientDescent": JaxoptGradientDescent,
    #
    "ProximalGradient": JaxoptProximalGradient,
    #
    "LBFGS": JaxoptLBFGS,
    #
    "BFGS": JaxoptBFGS,
    #
    "SVRG": WrappedSVRG,
    "ProxSVRG": WrappedProxSVRG,
    #
    "NonlinearCG": JaxoptNonlinearCG,
}


def list_available_solvers():
    """
    List the available solvers that can be used for fitting models.

    To access an extended documentation about a specific solver,
    see `get_solver_documentation`.

    Example
    -------
    >>> import nemos as nmo
    >>> nmo.solvers.list_available_solvers()
    ['GradientDescent', 'ProximalGradient', 'LBFGS', 'BFGS', 'SVRG', 'ProxSVRG', 'NonlinearCG']
    >>> print(nmo.solvers.get_solver_documentation("SVRG"))
    Showing docstring of nemos.solvers._svrg.WrappedSVRG.
    For potentially more info, use `show_help=True`.
    <BLANKLINE>
    Adapter for NeMoS's implementation of SVRG following the AbstractSolver interface.
    <BLANKLINE>
    ...
    """
    return list(solver_registry.keys())
