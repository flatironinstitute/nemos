"""Registry for mapping from solver name to concrete implementation."""

from typing import Type

from ._jaxopt_solvers import (
    JaxoptBFGS,
    JaxoptGradientDescent,
    JaxoptLBFGS,
    JaxoptNonlinearCG,
    JaxoptProximalGradient,
)

# from ._optimistix_solvers import OptimistixBFGS, OptimistixNonlinearCG
from ._svrg import WrappedProxSVRG, WrappedSVRG

# from ._optax_optimistix_solvers import (
#    OptimistixOptaxGradientDescent,
#    OptimistixOptaxLBFGS,
#    OptimistixOptaxProximalGradient,
# )


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
