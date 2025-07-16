"""Base class defining the interface for solvers that can be used by `BaseRegressor`."""

import abc
from typing import Callable, Generic, TypeAlias, TypeVar, Iterator

from jaxtyping import PyTree

from ..regularizer import Regularizer

Params: TypeAlias = PyTree
SolverState = TypeVar("SolverState")
StepResult = TypeVar("StepResult")


class AbstractSolver(abc.ABC, Generic[SolverState, StepResult]):
    """
    Base class defining the interface for solvers that can be used by `BaseRegressor`.

    All solver parameters (e.g. tolerance, number of steps) are passed to `__init__`,
    the other methods only take parameters, solver state, and the positional arguments of
    the objective function.
    """

    @abc.abstractmethod
    def __init__(
        self,
        unregularized_loss: Callable,
        regularizer: Regularizer,
        regularizer_strength: float | None,
        **solver_init_kwargs,
    ):
        """
        Create the solver.

        Arguments
        ---------
        unregularized_loss:
            Unregularized loss function.
            Currently `BaseRegressor._predict_and_compute_loss`.
        regularizer:
            Regularizer object used to create the penalized loss
            or get the proximal operator from.
        regularizer_strength:
            Regularizer strength.
        **solver_init_kwargs:
            Keyword arguments modifying the solver's behavior.
        """
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

    def run_iterator(self, iterator: Iterator) -> StepResult:
        raise NotImplementedError(
            "If the solver is stochastic, i.e. works with mini-batches of data, this method should be implemented."
            "If the solver is not stochastic, this method should not be called."
        )

    @classmethod
    @abc.abstractmethod
    def get_accepted_arguments(cls) -> set[str]:
        """
        Set of argument names accepted by the solver.

        Used by `BaseRegressor` to determine what arguments
        can be passed to the solver's __init__.
        """
        pass
