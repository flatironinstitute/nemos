"""Custom solvers module."""

from ._solver_registry import solver_registry
from ._svrg import SVRG, ProxSVRG, WrappedProxSVRG, WrappedSVRG
from ._svrg_defaults import (
    glm_softplus_poisson_l_max_and_l,
    svrg_optimal_batch_and_stepsize,
)
from ._jaxopt_solvers import (
    JaxoptBFGS,
    JaxoptGradientDescent,
    JaxoptLBFGS,
    JaxoptNonlinearCG,
    JaxoptProximalGradient,
)
from ._optax_optimistix_solvers import (
    OptaxOptimistixGradientDescent,
    OptaxOptimistixProximalGradient,
)
from ._optimistix_solvers import (  # OptimistixLBFGS,
    OptimistixBFGS,
    OptimistixNonlinearCG,
)
