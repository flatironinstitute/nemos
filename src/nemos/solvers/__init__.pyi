"""Custom solvers module stubs."""

from ._abstract_solver import (
    AbstractSolver,
    OptimizationInfo,
    Params,
    SolverProtocol,
    SolverState,
    StepResult,
)
from ._fista import OptimistixFISTA, OptimistixNAG
from ._jaxopt_solvers import (
    JAXOPT_AVAILABLE,
    JaxoptBFGS,
    JaxoptGradientDescent,
    JaxoptLBFGS,
    JaxoptNonlinearCG,
    JaxoptProximalGradient,
)
from ._optax_optimistix_solvers import (
    OptimistixOptaxGradientDescent,
    OptimistixOptaxLBFGS,
)
from ._optimistix_solvers import OptimistixBFGS, OptimistixNonlinearCG
from ._solver_doc_helper import get_solver_documentation
from ._solver_registry import (
    SolverSpec,
    get_solver,
    list_algo_backends,
    list_available_algorithms,
    list_available_solvers,
    register,
    set_default_backend,
)
from ._svrg import SVRG, ProxSVRG, WrappedProxSVRG, WrappedSVRG
from ._svrg_defaults import (
    glm_softplus_poisson_l_max_and_l,
    svrg_optimal_batch_and_stepsize,
)
from ._validation import validate_solver_class

__all__ = [
    "AbstractSolver",
    "OptimizationInfo",
    "Params",
    "SolverProtocol",
    "SolverState",
    "StepResult",
    "OptimistixFISTA",
    "OptimistixNAG",
    "JAXOPT_AVAILABLE",
    "JaxoptBFGS",
    "JaxoptGradientDescent",
    "JaxoptLBFGS",
    "JaxoptNonlinearCG",
    "JaxoptProximalGradient",
    "OptimistixOptaxGradientDescent",
    "OptimistixOptaxLBFGS",
    "OptimistixBFGS",
    "OptimistixNonlinearCG",
    "get_solver_documentation",
    "SolverSpec",
    "get_solver",
    "list_algo_backends",
    "list_available_algorithms",
    "list_available_solvers",
    "register",
    "set_default_backend",
    "SVRG",
    "ProxSVRG",
    "WrappedProxSVRG",
    "WrappedSVRG",
    "glm_softplus_poisson_l_max_and_l",
    "svrg_optimal_batch_and_stepsize",
    "validate_solver_class",
]
