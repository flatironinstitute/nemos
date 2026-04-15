"""Callback system for stochastic training loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union


@dataclass
class TrainingContext:
    """
    Mutable context object passed to callbacks during training.

    One instance is created per training run. Fields are updated in-place
    by the training loop before each callback invocation.

    Parameters
    ----------
    model :
        The model being trained (e.g. GLM instance). Set by ``stochastic_fit``.
    solver :
        The solver instance running the optimization.
    params :
        Current model parameters.
    state :
        Current solver state.
    aux :
        Auxiliary output from the last batch.
    epoch_idx :
        Current epoch index (0-based).
    batch_idx :
        Current batch index within the epoch.
    num_epochs :
        Total number of epochs requested.
    """

    model: Any = None
    solver: Any = None
    params: Any = None
    state: Any = None
    aux: Any = None
    epoch_idx: int | None = None
    batch_idx: int | None = None
    num_epochs: int = 0

    # signals for stopping optimization,
    # controlled through methods, not included in constructor
    _stop_requested: bool = field(default=False, init=False, repr=False)
    _stop_reason: str = field(default="", init=False, repr=False)

    def request_stop(self, reason: str = "") -> None:
        """
        Request early stopping of the training loop.

        Parameters
        ----------
        reason :
            Human-readable reason for stopping.
        """
        self._stop_requested = True
        self._stop_reason = reason

    @property
    def should_stop(self) -> bool:
        """Whether a callback has requested early stopping."""
        return self._stop_requested

    @property
    def stop_reason(self) -> str:
        """Reason for the stop request, if any."""
        return self._stop_reason


class Callback:
    """
    Base class for training callbacks.

    All hooks are no-ops by default. Subclass and override the hooks you need.
    """

    def on_train_begin(self, ctx: TrainingContext) -> None:
        """Run once at the start of training."""
        pass

    def on_train_end(self, ctx: TrainingContext) -> None:
        """Run once at the end of training."""
        pass

    def on_epoch_begin(self, ctx: TrainingContext) -> None:
        """
        Run at the start of an epoch.

        This hook is called after the training loop advances ``ctx.epoch`` and
        before the first batch of that epoch is processed. It marks the start
        of epoch-level work from the callback perspective.

        Solver-specific epoch preparation may still occur after this hook and
        before the first batch update. Callbacks should therefore treat this
        hook as notification that a new epoch is starting, not as a guarantee
        that all solver-internal epoch setup has already completed.
        """
        pass

    def on_epoch_end(self, ctx: TrainingContext) -> None:
        """Run at the end of each epoch."""
        pass

    def on_batch_begin(self, ctx: TrainingContext) -> None:
        """Run before each batch update."""
        pass

    def on_batch_end(self, ctx: TrainingContext) -> None:
        """Run after each batch update."""
        pass


class CallbackList(Callback):
    """
    Composite callback that dispatches each hook to all registered callbacks.

    Parameters
    ----------
    callbacks :
        List of ``Callback`` instances.
    """

    def __init__(self, callbacks: list[Callback] | None = None):
        self._callbacks: list[Callback] = list(callbacks) if callbacks else []

    def on_train_begin(self, ctx: TrainingContext) -> None:
        """Dispatch train-begin hook to all callbacks."""
        for cb in self._callbacks:
            cb.on_train_begin(ctx)

    def on_train_end(self, ctx: TrainingContext) -> None:
        """Dispatch train-end hook to all callbacks."""
        for cb in self._callbacks:
            cb.on_train_end(ctx)

    def on_epoch_begin(self, ctx: TrainingContext) -> None:
        """Dispatch epoch-begin hook to all callbacks."""
        for cb in self._callbacks:
            cb.on_epoch_begin(ctx)

    def on_epoch_end(self, ctx: TrainingContext) -> None:
        """Dispatch epoch-end hook to all callbacks."""
        for cb in self._callbacks:
            cb.on_epoch_end(ctx)

    def on_batch_begin(self, ctx: TrainingContext) -> None:
        """Dispatch batch-begin hook to all callbacks."""
        for cb in self._callbacks:
            cb.on_batch_begin(ctx)

    def on_batch_end(self, ctx: TrainingContext) -> None:
        """Dispatch batch-end hook to all callbacks."""
        for cb in self._callbacks:
            cb.on_batch_end(ctx)


class SolverConvergenceCallback(Callback):
    """
    Delegate convergence checking to the solver's built-in criterion.

    Calls ``ctx.solver.stochastic_convergence_criterion(...)`` at the end of
    each epoch and requests a stop if it returns ``True``.

    Tracks previous params and state internally so the context doesn't have to.
    """

    def __init__(self):
        self._prev_params = None
        self._prev_state = None

    def on_epoch_begin(self, ctx: TrainingContext) -> None:
        """Save current params and state before the epoch runs."""
        self._prev_params = ctx.params
        self._prev_state = ctx.state

    def on_epoch_end(self, ctx: TrainingContext) -> None:
        """Check solver convergence criterion and request stop if met."""
        converged = ctx.solver.stochastic_convergence_criterion(
            ctx.params,
            self._prev_params,
            ctx.state,
            self._prev_state,
            ctx.aux,
            ctx.epoch,
        )
        if converged:
            ctx.request_stop("Satisfied the solver's convergence criterion.")


def _normalize_callbacks(
    callbacks: Union[Callback, list[Callback], None],
) -> Callback:
    """
    Normalize callback argument into a ``Callback``.

    Parameters
    ----------
    callbacks :
        A single callback, a list, or ``None``.
        ``None`` means no callback and returns a no-op ``Callback``.

    Returns
    -------
    Callback
    """
    if callbacks is None:
        return Callback()
    if isinstance(callbacks, Callback):
        return callbacks
    if isinstance(callbacks, list):
        return CallbackList(callbacks)
    raise TypeError(
        f"callbacks must be a Callback, list[Callback], or None; "
        f"got {type(callbacks).__name__}"
    )
