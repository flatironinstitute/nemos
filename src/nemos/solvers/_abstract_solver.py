"""Base class defining the interface for solvers that can be used by `BaseRegressor`."""

import abc
from typing import Callable, Generic, TypeAlias, TypeVar

from jaxtyping import PyTree

from ..regularizer import Regularizer

Params: TypeAlias = PyTree
SolverState = TypeVar("SolverState")
StepResult = TypeVar("StepResult")


class AbstractSolver(abc.ABC, Generic[SolverState, StepResult]):
    """Base class defining the interface for solvers that can be used by `BaseRegressor`."""

    @abc.abstractmethod
    def __init__(
        self,
        unregularized_loss: Callable,
        regularizer: Regularizer,
        regularizer_strength: float | None,
        **solver_init_kwargs,
    ):
        pass

    @abc.abstractmethod
    def init_state(self, init_params: Params, *args) -> SolverState:
        """
        Initialize the solver state.

        Used by `BaseRegressor.initialize_state`
        """
        pass

    @abc.abstractmethod
    def update(self, params: Params, state: SolverState, *args) -> StepResult:
        """
        Perform a single step/update of the optimization process.

        Used by `BaseRegressor.update`.
        """
        pass

    @abc.abstractmethod
    def run(self, init_params: Params, *args) -> StepResult:
        """
        Run a full optimization process until a stopping criterion is reached.

        Used by `BaseRegressor.fit`.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def get_accepted_arguments(cls) -> set[str]:
        """
        Set the argument names accepted by the solver.

        Used by `BaseRegressor` to determine what arguments
        can be passed to the solver's __init__.
        """
        pass
