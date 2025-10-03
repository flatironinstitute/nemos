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


def available_solvers():
    """
    List the available solvers that can be used for fitting models.

    Example
    -------
    >>> import nemos as nmo
    >>> nmo.solvers.available_solvers()
    ['GradientDescent', 'ProximalGradient', 'LBFGS', 'BFGS', 'SVRG', 'ProxSVRG', 'NonlinearCG']
    """
    return list(solver_registry.keys())
