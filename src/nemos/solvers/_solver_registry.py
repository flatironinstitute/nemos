"""Registry for mapping from solver name to concrete implementation."""

from .jaxopt_solvers import (
    JaxoptGradientDescent,
    JaxoptProximalGradient,
    JaxoptBFGS,
    JaxoptLBFGS,
    JaxoptNonlinearCG,
)

from ._svrg import (
    WrappedProxSVRG,
    WrappedSVRG,
)

from .optimistix_solvers import (
    OptimistixBFGS,
    # OptimistixLBFGS,
    OptimistixNonlinearCG,
)

from .optax_optimistix_solvers import (
    OptaxOptimistixProximalGradient,
    OptaxOptimistixGradientDescent,
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
