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

from ._prox_grad import (
    OptaxOptimistixProximalGradient,
)

solver_registry = {
    "GradientDescent": JaxoptGradientDescent,
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
