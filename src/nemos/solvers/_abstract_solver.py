"""Base class defining the interface for solvers that can be used by `BaseRegressor`."""

from __future__ import annotations

import abc
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Protocol,
    runtime_checkable,
)

from ..callbacks import Callback, TrainingContext
from ..typing import Params, SolverState, StepResult

if TYPE_CHECKING:
    from ..batching import DataLoader
    from ..regularizer import Regularizer
import equinox as eqx
import jax


class SolverAdapterState(eqx.Module, Generic[SolverState]):
    solver_state: SolverState  # backend-specific internal state
    stats: OptimizationInfo  # num_steps, converged, etc. — computed during run, valid JAX pytree


class OptimizationInfo(eqx.Module):
    """Basic diagnostic information about finished optimization runs."""

    # None means missing value, while NaN usually indicates a diverged optimization
    function_val: (
        jax.numpy.ndarray | None
    )  #: Function value. Optional as not all solvers store it.
    num_steps: jax.numpy.ndarray  #: Number of optimization steps taken, array of int.
    converged: jax.numpy.ndarray  #: Whether the optimization converged, array of bool.
    reached_max_steps: (
        jax.numpy.ndarray
    )  #: Reached the maximum number of allowed steps.


class AbstractSolver(abc.ABC, Generic[SolverState]):
    """
    Base class defining the interface for solvers that can be used by `BaseRegressor`.

    All solver parameters (e.g. tolerance, number of steps) are passed to ``__init__``,
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

        Used by ``BaseRegressor.initialize_state``.
        """
        pass

    @abc.abstractmethod
    def update(self, params: Params, state: SolverState, *args: Any) -> StepResult:
        """
        Perform a single step/update of the optimization process.

        Used by ``BaseRegressor.update``.
        """
        pass

    @abc.abstractmethod
    def run(self, init_params: Params, *args: Any) -> StepResult:
        """
        Run a full optimization process until a stopping criterion is reached.

        Used by ``BaseRegressor.fit``.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def get_accepted_arguments(cls) -> set[str]:
        """
        Return the set of argument names accepted by the solver.

        Used by ``BaseRegressor`` to determine what arguments
        can be passed to the solver's ``__init__``.
        """
        pass

    @abc.abstractmethod
    def _get_optim_info(self, state: SolverState, **kwargs) -> OptimizationInfo:
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
        data_loader: DataLoader,
        num_epochs: int,
        callback: Callback | None = None,
        ctx: TrainingContext | None = None,
    ) -> StepResult:
        """
        Run optimization over mini-batches from a data loader.

        Parameters
        ----------
        init_params
            Initial parameter values.
        data_loader
            Data loader providing batches and metadata.
            Must be re-iterable (each ``__iter__`` call returns fresh iterator).
        num_epochs
            Number of passes over the data. Must be >= 1.
        callback :
            Training callback. A single ``Callback`` instance (use
            ``CallbackList`` for multiple callbacks).
            If None, use the no-op base callback.
        ctx :
            Training context for callback communication.
            If None, a default context is created.

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
            )
        if num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")
        if callback is None:
            callback = Callback()
        if ctx is None:
            ctx = TrainingContext()

        ctx.solver = self
        ctx.params = init_params
        ctx.num_epochs = num_epochs

        return self._stochastic_run_impl(
            init_params,
            data_loader,
            num_epochs,
            callback=callback,
            ctx=ctx,
        )

    def _stochastic_run_impl(
        self,
        init_params: Params,
        data_loader: DataLoader,
        num_epochs: int,
        callback: Callback,
        ctx: TrainingContext | None = None,
    ) -> StepResult:
        """
        Override in stochastic-capable solvers.

        For details see ``stochastic_run``
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _stochastic_run_impl"
        )

    def stochastic_convergence_criterion(
        self,
        params: Params,
        prev_params: Params,
        state: SolverState,
        prev_state: SolverState,
        aux: Any,
        epoch: int,
    ) -> bool:
        """
        Evaluate convergence criterion for stochastic optimization.

        Called once per epoch. Subclasses that support stochastic optimization
        should override this to provide a meaningful default.

        Parameters
        ----------
        params :
            Parameter values at end of current epoch.
        prev_params :
            Parameter values at end of previous epoch.
        state :
            Solver state at end of current epoch.
        prev_state :
            Solver state at end of previous epoch.
        aux :
            Auxiliary output from the last batch of the current epoch.
        epoch :
            Current epoch index (0-based).

        Returns
        -------
        bool
            ``True`` if optimization should stop.
        """
        if not self._supports_stochastic:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support stochastic optimization."
            )

        raise NotImplementedError(
            f"For convergence monitoring during stochastic optimization "
            f"{self.__class__.__name__} must implement stochastic_convergence_criterion. "
            "Please implement it or disable convergence monitoring."
        )


@runtime_checkable
class SolverProtocol(Protocol, Generic[SolverState]):
    """
    Protocol mirroring the interface of AbstractSolver.

    Implementations can be checked at runtime via isinstance(solver_object, SolverProtocol)
    and issubclass(solver_class, SolverProtocol).

    Solvers that support stochastic (mini-batch) optimization can additionally
    implement ``stochastic_run`` and ``stochastic_convergence_criterion``, and
    set the class variable ``_supports_stochastic = True``.
    See ``AbstractSolver`` and ``StochasticSolverMixin`` for details.
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
