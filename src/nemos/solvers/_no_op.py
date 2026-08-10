"""Solver stand-in for models whose every parameter is fixed."""

from typing import Any, NamedTuple

from ..typing import Params, StepResult


class NoOpState(NamedTuple):
    """State of a solver with nothing to optimize.

    ``converged`` is what the fit entry points inspect to decide whether to warn.
    """

    converged: bool = True


class NoOpSolver:
    """Identity solver for an empty active parameter tree.

    When every parameter is fixed there is nothing left to optimize: the pinned values
    already are the solution. The real solvers cannot be initialized on an empty pytree
    (optax's line search indexes into its leaves), so this takes their place and lets
    the model entry points keep their shape — partition, run, recombine the frozen
    values. It is internal: not registered, so it cannot be selected by name.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def init_state(self, init_params: Params, *args: Any) -> NoOpState:
        """Return the state of a solver that will not step."""
        return NoOpState()

    def update(self, params: Params, state: NoOpState, *args: Any) -> StepResult:
        """Return ``params`` unchanged."""
        return params, state, None

    def run(self, init_params: Params, *args: Any) -> StepResult:
        """Return ``init_params`` unchanged."""
        return init_params, NoOpState(), None

    @classmethod
    def get_accepted_arguments(cls) -> set[str]:
        """No configuration is accepted."""
        return set()
