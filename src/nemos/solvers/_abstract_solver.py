"""Base class defining the interface for solvers that can be used by `BaseRegressor`."""

import abc
from typing import Callable, Generic, TypeAlias, TypeVar, Iterator, Protocol

import itertools

from jaxtyping import PyTree

from ..regularizer import Regularizer

Params: TypeAlias = PyTree
SolverState = TypeVar("SolverState")
StepResult = TypeVar("StepResult")

# TODO If we want to accept solvers that implement this interface,
# but are not implemented as a subclass of AbstractSolver,
# we could just create a protocol listing the methods.


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

    def run_iterator(self, init_params: Params, iterator: Iterator) -> StepResult:
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


class StochasticSolver(Protocol):
    def run_iterator(self, init_params: Params, iterator: Iterator) -> StepResult: ...


class StochasticMixin(AbstractSolver[SolverState, tuple[Params, SolverState]]):
    """
    Mixin providing a `run_iterator` method for stochastic optimization.

    Requires the solver to implement the `AbstractSolver` interface
    plus provide a `maxiter` attribute or property.
    """

    @property
    @abc.abstractmethod
    def maxiter(self) -> int:
        """If run_iterator is implemented with this, the child class needs maxiter."""
        pass

    def run_iterator(
        self,
        init_params: Params,
        iterator: Iterator,
    ) -> tuple[Params, SolverState]:
        """
        Iterate through `iterator` and take a step using the data provided until the maximum number of steps is reached.

        Implementation copied from `jaxopt.base.StochasticSolver`.
        """
        data = next(iterator)
        state = self.init_state(init_params, *data)
        params = init_params

        for data in itertools.islice(iterator, 0, self.maxiter):
            params, state = self.update(params, state, *data)

        return params, state
