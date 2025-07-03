"""Registry for mapping from solver name to concrete implementation."""

from ._svrg import (
    WrappedProxSVRG,
    WrappedSVRG,
)
from .jaxopt_solvers import (
    JaxoptBFGS,
    JaxoptGradientDescent,
    JaxoptLBFGS,
    JaxoptNonlinearCG,
    JaxoptProximalGradient,
)
from .optax_optimistix_solvers import (
    OptaxOptimistixGradientDescent,
    OptaxOptimistixProximalGradient,
)
from .optimistix_solvers import (  # OptimistixLBFGS,
    OptimistixBFGS,
    OptimistixNonlinearCG,
)

solver_registry = {
    # "GradientDescent": JaxoptGradientDescent,
    "GradientDescent": OptaxOptimistixGradientDescent,
    #
    # "ProximalGradient": JaxoptProximalGradient,
    "ProximalGradient": OptaxOptimistixProximalGradient,
    #
    "LBFGS": JaxoptLBFGS,
    # "LBFGS": OptimistixLBFGS,
    #
    "BFGS": OptimistixBFGS,
    #
    "SVRG": WrappedSVRG,
    "ProxSVRG": WrappedProxSVRG,
    #
    "NonlinearCG": JaxoptNonlinearCG,
    # "NonlinearCG": OptimistixNonlinearCG,
}
