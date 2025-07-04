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
from .optimistix_solvers import (
    OptimistixBFGS,
    # OptimistixLBFGS,
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
    # "BFGS": JaxoptBFGS,
    "BFGS": OptimistixBFGS,
    #
    "SVRG": WrappedSVRG,
    "ProxSVRG": WrappedProxSVRG,
    #
    # "NonlinearCG": JaxoptNonlinearCG,
    "NonlinearCG": OptimistixNonlinearCG,
}
