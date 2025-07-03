import abc
from typing import Generic, TypeVar

SolverState = TypeVar("SolverState")
StepResult = TypeVar("StepResult")


class AbstractSolver(abc.ABC, Generic[SolverState, StepResult]):
    @abc.abstractmethod
    def init_state(self, init_params, *args) -> SolverState:
        pass

    @abc.abstractmethod
    def update(self, params, state, *args) -> StepResult:
        pass

    @abc.abstractmethod
    def run(self, init_params, *args) -> StepResult:
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
