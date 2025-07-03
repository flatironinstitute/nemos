from .abstract_solver import AbstractSolver, SolverState, StepResult
from typing import Generic, TypeVar, Type, Any, ClassVar
import inspect


class SolverAdapter(AbstractSolver[SolverState, StepResult]):
    _solver_cls: ClassVar[Type]
    _solver: Any

    def __getattr__(self, name: str):
        # without this guard deepcopy leads to a RecursionError
        try:
            solver = object.__getattribute__(self, "_solver")
        except AttributeError:
            raise AttributeError(name)

        return getattr(solver, name)

    @classmethod
    def get_accepted_arguments(cls) -> set[str]:
        """Set of accepted argument names, extended with the wrapped solver's arguments."""
        own_arguments = set(inspect.getfullargspec(cls.__init__).args)
        solver_arguments = set(inspect.getfullargspec(cls._solver_cls.__init__).args)

        all_arguments = own_arguments | solver_arguments

        # discard arguments that are passed by BaseRegressor
        all_arguments.discard("self")
        all_arguments.discard("unregularized_loss")
        all_arguments.discard("regularizer")
        all_arguments.discard("regularizer_strength")

        return all_arguments
