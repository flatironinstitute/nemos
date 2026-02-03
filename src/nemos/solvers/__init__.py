"""Custom solvers module."""

from ._fista import OptimistixFISTA, OptimistixNAG
from ._jaxopt_solvers import JAXOPT_AVAILABLE
from ._optax_optimistix_solvers import (
    OptimistixOptaxGradientDescent,
    OptimistixOptaxLBFGS,
)
from ._optimistix_solvers import OptimistixBFGS, OptimistixNonlinearCG
from ._solver_doc_helper import get_solver_documentation
from ._solver_registry import list_available_solvers, solver_registry
from ._svrg import SVRG, ProxSVRG, WrappedProxSVRG, WrappedSVRG
from ._svrg_defaults import (
    glm_softplus_poisson_l_max_and_l,
    svrg_optimal_batch_and_stepsize,
)

if JAXOPT_AVAILABLE:
    from ._jaxopt_solvers import (
        JaxoptBFGS,
        JaxoptGradientDescent,
        JaxoptLBFGS,
        JaxoptNonlinearCG,
        JaxoptProximalGradient,
    )
