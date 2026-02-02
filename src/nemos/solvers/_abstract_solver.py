"""Base class defining the interface for solvers that can be used by `BaseRegressor`."""

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, NamedTuple

from ..regularizer import Regularizer
from ..typing import Params, SolverState, StepResult

if TYPE_CHECKING:
    from ..batching import DataLoader


@dataclass
class OptimizationInfo:
    """Basic diagnostic information about finished optimization runs."""

    # Not all JAXopt solvers store the function value.
    # None means missing value, while NaN usually indicates a diverged optimization
    function_val: float | None  #: Function value. Optional as not all solvers store it.
    num_steps: int  #: Number of optimization steps taken.
    converged: bool  #: Whether the optimization converged.
    reached_max_steps: bool  #: Reached the maximum number of allowed steps.


class AbstractSolver(abc.ABC, Generic[SolverState]):
    """
    Base class defining the interface for solvers that can be used by `BaseRegressor`.

    All solver parameters (e.g. tolerance, number of steps) are passed to `__init__`,
    the other methods only take parameters, solver state, and the positional arguments of
    the objective function.
    """

    # Set to True in subclasses that support stochastic optimization
    _supports_stochastic: ClassVar[bool] = False

    @abc.abstractmethod
    def __init__(
        self,
        unregularized_loss: Callable,
        regularizer: Regularizer,
        regularizer_strength: float | None,
        has_aux: bool,
        init_params: Params | None = None,
        **solver_init_kwargs,
    ):
        """
        Create the solver.

        Arguments
        ---------
        unregularized_loss:
            Unregularized loss function.
            Currently `BaseRegressor.compute_loss`.
        regularizer:
            Regularizer object used to create the penalized loss
            or get the proximal operator from.
        regularizer_strength:
            Regularizer strength.
        has_aux:
            Whether `unregularized_loss` returns auxiliary variables.
            If False, the loss function is expected to return a single scalar.
            If True, the loss is expected to return a tuple of length 2 with a scalar and auxiliary variables.
        init_params:
            Initial model parameters. Passed to the regularizer's `get_proximal_operator` or `penalized_loss`.
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

    def stochastic_run(
        self,
        init_params: Params,
        data_loader: "DataLoader",
        num_epochs: int = 1,
    ) -> StepResult:
        """Run optimization over mini-batches from a data loader.

        Parameters
        ----------
        init_params : Params
            Initial parameter values.
        data_loader : DataLoader
            Data loader providing batches and metadata.
            Must be re-iterable (each __iter__ call returns fresh iterator).
        num_epochs : int
            Number of passes over the data. Must be >= 1.

        Returns
        -------
        StepResult
            Final (params, state, aux) tuple.

        Raises
        ------
        NotImplementedError
            If the solver does not support stochastic optimization.
        ValueError
            If num_epochs < 1.
        """
        if not self._supports_stochastic:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support stochastic optimization. "
                f"Use 'GradientDescent', 'ProximalGradient', 'SVRG', or 'ProxSVRG'."
            )
        if num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")
        return self._stochastic_run_impl(init_params, data_loader, num_epochs)

    def _stochastic_run_impl(
        self,
        init_params: Params,
        data_loader: "DataLoader",
        num_epochs: int,
    ) -> StepResult:
        """Override in stochastic-capable solvers.

        Parameters
        ----------
        init_params : Params
            Initial parameter values.
        data_loader : DataLoader
            Data loader providing batches and metadata.
        num_epochs : int
            Number of passes over the data.

        Returns
        -------
        StepResult
            Final (params, state, aux) tuple.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _stochastic_run_impl"
        )


@runtime_checkable
class SolverProtocol(Protocol, Generic[SolverState]):
    """
    Protocol mirroring the interface of AbstractSolver.

    Implementations can be checked at runtime via isinstance(solver_object, SolverProtocol)
    and issubclass(solver_class, SolverProtocol).
    """

    def __init__(
        self,
        unregularized_loss: Callable,
        regularizer: Regularizer,
        regularizer_strength: float | None,
        has_aux: bool,
        init_params: Params | None = None,
        **solver_init_kwargs: Any,
    ) -> None: ...

    def init_state(self, init_params: Params, *args: Any) -> SolverState: ...

    def update(self, params: Params, state: SolverState, *args: Any) -> StepResult: ...

    def run(self, init_params: Params, *args: Any) -> StepResult: ...

    @classmethod
    def get_accepted_arguments(cls) -> set[str]: ...

    def get_optim_info(self, state: SolverState) -> OptimizationInfo: ...
