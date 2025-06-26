"""Custom solvers module."""

from ._svrg import SVRG, ProxSVRG, WrappedSVRG, WrappedProxSVRG
from ._svrg_defaults import (
    glm_softplus_poisson_l_max_and_l,
    svrg_optimal_batch_and_stepsize,
)

from .jaxopt_solvers import (
    JaxoptGradientDescent,
    JaxoptProximalGradient,
    JaxoptBFGS,
    JaxoptLBFGS,
)

from .optimistix_solvers import (
    OptimistixBFGS,
    # OptimistixLBFGS,
)

from ._solver_registry import solver_registry

from ._prox_grad import OptaxOptimistixProximalGradient
