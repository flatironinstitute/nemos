"""Registry for mapping from solver name to concrete implementation."""

from ._jaxopt_solvers import (
    JaxoptBFGS,
    JaxoptGradientDescent,
    JaxoptLBFGS,
    JaxoptNonlinearCG,
    JaxoptProximalGradient,
)
from ._optax_optimistix_solvers import (
    OptaxOptimistixGradientDescent,
    OptaxOptimistixLBFGS,
    OptaxOptimistixProximalGradient,
)
from ._optimistix_solvers import (  # OptimistixLBFGS,
    OptimistixBFGS,
    OptimistixNonlinearCG,
    OptimistixProximalGradient,
)
from ._svrg import WrappedProxSVRG, WrappedSVRG

solver_registry = {
    # "GradientDescent": JaxoptGradientDescent,
    "GradientDescent": OptaxOptimistixGradientDescent,
    #
    # "ProximalGradient": JaxoptProximalGradient,
    # "ProximalGradient": OptaxOptimistixProximalGradient,
    "ProximalGradient": OptimistixProximalGradient,
    #
    "LBFGS": OptaxOptimistixLBFGS,
    # "LBFGS": OptimistixLBFGS,
    #
    # "BFGS": JaxoptBFGS,
    "BFGS": OptimistixBFGS,
    #
    "SVRG": WrappedSVRG,
    "ProxSVRG": WrappedProxSVRG,
    #
    # "NonlinearCG": JaxoptNonlinearCG,
    "NonlinearCG": OptimistixNonlinearCG,
}
