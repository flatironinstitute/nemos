"""Registry of optimization algorithms and their implementations."""

from dataclasses import dataclass
from typing import Type

from ._abstract_solver import SolverProtocol
from ._fista import OptimistixFISTA, OptimistixNAG
from ._jaxopt_solvers import JAXOPT_AVAILABLE
from ._optax_optimistix_solvers import OptimistixOptaxLBFGS
from ._optimistix_solvers import OptimistixBFGS, OptimistixNonlinearCG
from ._svrg import WrappedProxSVRG, WrappedSVRG
from ._validation import validate_solver_class


@dataclass
class SolverSpec:
    """
    Solver specification representing an entry in the solver registry.

    A solver is specified by:
    - the name of the algorithm it implements
    - its backend (optimization library or custom)
    - the class implementing the optimization method
      (ideally compatible with the AbstractSolver and SolverProtocol interface)

    Examples
    --------
    >>> import nemos as nmo
    >>> spec = nmo.solvers.SolverSpec("BFGS", "optimistix", nmo.solvers._optimistix_solvers.OptimistixBFGS)
    >>> spec.algo_name
    'BFGS'
    >>> spec.backend
    'optimistix'
    >>> spec.implementation
    <class 'nemos.solvers._optimistix_solvers.OptimistixBFGS'>
    """

    algo_name: str
    backend: str
    implementation: Type[SolverProtocol]

    @property
    def full_name(self) -> str:
        return f"{self.algo_name}[{self.backend}]"

    def __repr__(self) -> str:
        return (
            f"{self.full_name!r} - "
            f"{self.__class__.__name__}("
            f"algo_name={self.algo_name!r}, "
            f"backend={self.backend!r}, "
            f"implementation={f'{self.implementation.__module__}.{self.implementation.__qualname__}'!r})"
        )


# mapping is {algo_name : {backend : implementation}}
_registry: dict[str, dict[str, SolverSpec]] = {}
# mapping is {algo_name : backend}
_defaults: dict[str, str] = {}


def _parse_name(name: str) -> tuple[str, str | None]:
    """Parse an algo_name[backend] string."""
    algo_name = name
    backend = None
    if "[" in name:
        if name.count("[") > 1:
            raise ValueError(
                f"Found multiple opening '[' in solver name of {name}. "
                "Only use '[' for specifying the backend "
                "using the algo_name[backend_name] syntax. "
            )
        if name.count("]") > 1:
            raise ValueError(
                f"Found multiple closing ']' in solver name of {name}. "
                "Only use ']' for specifying the backend "
                "using the algo_name[backend_name] syntax. "
            )
        if "]" not in name:
            raise ValueError(
                "Found opening '[' in solver name but it does not end with closing ']'. "
                "Solver name can only use '[' for specifying the backend "
                "using the algo_name[backend_name] syntax. "
                f"Got {name}"
            )
        if name.index("]") != len(name) - 1:
            raise ValueError(
                "Found closing ']' in the middle of the solver name. "
                "Brackets are reserved for specifying the backend "
                "using the algo_name[backend_name] syntax. "
                f"Got {name}"
            )

        algo_name = name[: name.index("[")]
        backend = name[name.index("[") + 1 : -1]
    elif "]" in name:
        raise ValueError(
            "Found closing ']' in solver name without opening '['. "
            "Solver name can only use '[' for specifying the backend "
            "using the algo_name[backend_name] syntax. "
            f"Got {name}"
        )

    if algo_name == "":
        raise ValueError("Algorithm name cannot be an empty string.")
    if backend == "":
        raise ValueError("Backend name cannot be an empty string.")

    return algo_name, backend


def _raise_if_not_in_registry(algo_name: str):
    """Raise an error if an algorithm is not in the registry."""
    if algo_name not in _registry:
        raise ValueError(f"No solver registered for algorithm {algo_name}.")


def _resolve_backend(name: str, raise_if_given: bool) -> str:
    """
    Return the backend that will be used for the algorithm if not specified.

    Parameters
    ----------
    name:
        Name of the algorithm.
    raise_if_given:
        Raise an error if a backend is given, i.e. algo_name[backend_name]
        format is used.

    Returns
    -------
    Backend name extracted from the registry.
    """
    algo_name, backend = _parse_name(name)

    if backend is not None:
        if not raise_if_given:
            return backend

        raise ValueError(
            f"Provide an algorithm name only. Got {algo_name} with backend {backend}."
        )

    _raise_if_not_in_registry(algo_name)
    algo_versions = _registry[algo_name]

    backend = _defaults.get(algo_name, None)
    if backend is None:
        if len(algo_versions) == 1:
            backend = next(iter(algo_versions.keys()))
        else:
            _spec = " " if raise_if_given else " specify or "
            raise ValueError(
                f"Multiple backends and no default found for {algo_name}. "
                f"Please{_spec}set a default backend."
            )

    return backend


def get_solver(name: str) -> SolverSpec:
    """
    Fetch the solver spec. from the registry for a given solver.

    Parameters
    ----------
    name :
        Name of the solver with or without backend specified.

    Returns
    -------
    spec :
        Specification for the solver, listing algorithm name, backend, implementation class.
    """
    algo_name, _ = _parse_name(name)
    backend = _resolve_backend(name, False)

    # make sure we have the algorithm
    _raise_if_not_in_registry(algo_name)
    algo_versions = _registry[algo_name]

    if backend not in algo_versions:
        raise ValueError(
            f"{backend} backend not available for {algo_name}. "
            f"Available backends: {list_algo_backends(algo_name)}"
        )

    return algo_versions[backend]


def register(
    algo_name: str,
    implementation: Type[SolverProtocol],
    backend: str = "custom",
    replace: bool = False,
    default: bool = False,
    validate: bool = True,
    test_ridge_without_aux: bool = False,
    test_ridge_with_aux: bool = False,
) -> None:
    """
    Register a solver implementation in the registry.

    Parameters
    ----------
    algo_name :
        Name of the optimization algorithm.
    implementation :
        Class implementing the solver.
        Has to adhere to the AbstractSolver interface.
    backend :
        Backend name. Defaults to "custom".
        When wrapping and registering an existing solver from an external
        package, this would be the package name.
    replace :
        If an implementation for the given algorithm and backend names
        is already present in the registry, overwrite it.
    default :
        Set this implementation as the default for the algorithm.
        Can also be done with `set_default`.
    validate :
        Validate all required methods exist and have correct signatures.
    test_ridge_without_aux :
        Validate solver signatures and functionality by running a small ridge
        regression, objective function without aux.
    test_ridge_with_aux :
        Validate solver signatures and functionality by running a small ridgeregression,
        testing that objective functions with auxiliary variables are handled.

    Examples
    --------
    >>> import nemos as nmo
    >>> nmo.solvers.register("FISTA", nmo.solvers._fista.OptimistixFISTA, backend="optimistix")
    """
    if not replace and backend in _registry.get(algo_name, {}):
        raise ValueError(
            f"{algo_name}[{backend}] already registered. Use replace=True to overwrite."
        )

    if not issubclass(implementation, SolverProtocol):
        raise TypeError(f"{implementation.__name__} doesn't implement SolverProtocol.")

    if validate:
        validate_solver_class(implementation, False, False)

    if test_ridge_without_aux:
        validate_solver_class(implementation, True, False)

    if test_ridge_with_aux:
        validate_solver_class(implementation, True, True)

    if algo_name not in _registry:
        _registry[algo_name] = {}

    _registry[algo_name][backend] = SolverSpec(algo_name, backend, implementation)

    if default:
        set_default(algo_name, backend)


def set_default(algo_name: str, backend: str) -> None:
    """
    Set the default backend for a given algorithm.

    Parameters
    ----------
    algo_name :
        Name of the optimization algorithm whose default
        backend to set.
    backend :
        Name of the backend to set as default.

    Examples
    --------
    >>> import nemos as nmo
    >>> nmo.solvers.set_default("LBFGS", "optax+optimistix")
    >>> nmo.solvers.get_solver("LBFGS").backend
    'optax+optimistix'
    """
    _raise_if_not_in_registry(algo_name)

    if backend not in _registry[algo_name]:
        raise ValueError(
            f"{backend} backend not available for {algo_name}."
            f"Available backends: {list_algo_backends(algo_name)}"
        )
    _defaults[algo_name] = backend


def list_algo_backends(algo_name: str) -> list[str]:
    """
    List the available backends for an algorithm.

    Parameters
    ----------
    algo_name :
        Name of the optimization algorithm.
    """
    return list(_registry[algo_name].keys())


def list_available_solvers() -> list[SolverSpec]:
    """List all available solvers."""
    return [
        spec for algo_versions in _registry.values() for spec in algo_versions.values()
    ]


def list_available_algorithms() -> list[str]:
    """
    List the available algorithms that can be used for fitting models.

    To list the available backends for a given algorithm,
    see `list_algo_backends`.

    To access an extended documentation about a specific solver,
    see `nemos.solvers.get_solver_documentation`.

    Example
    -------
    >>> import nemos as nmo
    >>> nmo.solvers.list_available_algorithms()
    ['GradientDescent', 'ProximalGradient', 'LBFGS', 'BFGS', 'NonlinearCG', 'SVRG', 'ProxSVRG']
    """
    return list(_registry.keys())


register("GradientDescent", OptimistixNAG, "optimistix", default=True)
register("ProximalGradient", OptimistixFISTA, "optimistix", default=True)
register("LBFGS", OptimistixOptaxLBFGS, "optax+optimistix", default=True)
register("BFGS", OptimistixBFGS, "optimistix", default=True)
register("NonlinearCG", OptimistixNonlinearCG, "optimistix", default=True)
register("SVRG", WrappedSVRG, "nemos", default=True)
register("ProxSVRG", WrappedProxSVRG, "nemos", default=True)

if JAXOPT_AVAILABLE:
    from ._jaxopt_solvers import (
        JaxoptBFGS,
        JaxoptGradientDescent,
        JaxoptLBFGS,
        JaxoptNonlinearCG,
        JaxoptProximalGradient,
    )

    register("GradientDescent", JaxoptGradientDescent, "jaxopt", default=False)
    register("ProximalGradient", JaxoptProximalGradient, "jaxopt", default=False)
    register("LBFGS", JaxoptLBFGS, "jaxopt", default=False)
    register("BFGS", JaxoptBFGS, "jaxopt", default=False)
    register("NonlinearCG", JaxoptNonlinearCG, "jaxopt", default=False)
