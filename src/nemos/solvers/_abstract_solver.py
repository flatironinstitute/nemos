"""Base class defining the interface for solvers that can be used by `BaseRegressor`."""

import abc
from typing import Any, Callable, Generic, NamedTuple

from ..regularizer import Regularizer
from ..typing import Params, SolverState, StepResult


class OptimizationInfo(NamedTuple):
    """Basic diagnostic information about finished optimization runs."""

    # Not all JAXopt solvers store the function value.
    # None means missing value, while NaN usually indicates a diverged optimization
    function_val: float | None
    num_steps: int
    converged: bool
    reached_max_steps: bool


class AbstractSolver(abc.ABC, Generic[SolverState]):
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
    def init_state(self, init_params: Params, *args: Any) -> SolverState:
        """
        Initialize the solver state.

        Used by `BaseRegressor.initialize_state`
        """
        pass

    @abc.abstractmethod
    def update(self, params: Params, state: SolverState, *args: Any) -> StepResult:
        """
        Perform a single step/update of the optimization process.

        Used by `BaseRegressor.update`.
        """
        pass

    @abc.abstractmethod
    def run(self, init_params: Params, *args: Any) -> StepResult:
        """
        Run a full optimization process until a stopping criterion is reached.

        Used by `BaseRegressor.fit`.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def get_accepted_arguments(cls) -> set[str]:
        """
        Set of argument names accepted by the solver.

        Used by `BaseRegressor` to determine what arguments
        can be passed to the solver's __init__.
        """
        pass

    @abc.abstractmethod
    def get_optim_info(self, state: SolverState) -> OptimizationInfo:
        """Extract some commong info about the optimization process.

        Currently, the following info is extracted:
        - final function value (where available)
        - number of steps
        - whether the optimization converged
        - whether the max number of steps were reached
        """
        pass
