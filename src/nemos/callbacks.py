"""Callback system for stochastic training loops."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import partial
from typing import Any, Iterable, Union

import equinox as eqx
from numpy.typing import ArrayLike

from .typing import DESIGN_INPUT_TYPE


@dataclass(frozen=True)
class StochasticFitSummary:
    """
    Post-fit summary of a stochastic training run.

    Stored on fitted models as ``stochastic_fit_summary_`` after
    ``stochastic_fit`` completes. This summary is runtime state and is not
    serialized by ``save_params()``.

    In machine-learning terminology, a pass over the data is what is usually
    called an *epoch*; NeMoS says "pass" to avoid clashing with pynapple's
    recording epochs.

    Parameters
    ----------
    pass_idx :
        Final pass index reached by the training loop.
    batch_idx :
        Final batch index reached within the last pass.
    n_passes :
        Total number of passes requested for the run.
    should_stop :
        Whether a callback requested early stopping.
    stop_reason :
        Human-readable stop reason, if any.
    """

    pass_idx: int | None = None
    batch_idx: int | None = None
    n_passes: int = 0
    should_stop: bool = False
    stop_reason: str = ""


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
        Current model parameters. Exposed as a read/write property: the training loop
        assigns the actively optimized subtree (``ctx.params = ...``), and reading
        ``ctx.params`` recombines the frozen subtree (see ``frozen``) so callbacks
        always see the complete parameters.
    state :
        Current solver state.
    aux :
        Auxiliary output from the last batch.
    pass_idx :
        Current pass index (0-based).
    batch_idx :
        Current batch index within the pass.
    n_passes :
        Total number of passes requested.
    frozen :
        Parameter subtree held fixed during optimization (e.g. a zero intercept when
        ``fit_intercept=False``). The solver optimizes only the active subtree; this is
        recombined with it so callbacks always see the complete parameters. ``None``
        when nothing is frozen.
    """

    def __init__(
        self,
        model: Any = None,
        solver: Any = None,
        params: Any = None,
        state: Any = None,
        aux: Any = None,
        pass_idx: int | None = None,
        batch_idx: int | None = None,
        n_passes: int = 0,
        frozen: Any = None,
    ):
        self.model = model
        self.solver = solver
        self.state = state
        self.aux = aux
        self.pass_idx = pass_idx
        self.batch_idx = batch_idx
        self.n_passes = n_passes
        self.frozen = frozen
        self._stop_requested = False
        self._stop_reason = ""
        self.params = params

    @property
    def params(self) -> Any:
        """Current parameters, with the frozen subtree recombined into the active one."""
        if self.frozen is None:
            return self._params
        return eqx.combine(self._params, self.frozen)

    @params.setter
    def params(self, value: Any) -> None:
        self._params = value

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

    def to_summary(self) -> StochasticFitSummary:
        """Create a post-fit summary from the current training context."""
        return StochasticFitSummary(
            pass_idx=self.pass_idx,
            batch_idx=self.batch_idx,
            n_passes=self.n_passes,
            should_stop=self.should_stop,
            stop_reason=self.stop_reason,
        )

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """Represent this context as a string.

        Simple string representation, similar to that of a dataclass.
        """
        cls = self.__class__.__name__
        parts = []
        for name in inspect.signature(self.__init__).parameters:
            value = getattr(self, name)
            if value is None:
                continue
            parts.append(f"{name}={value}")
        if not parts:
            return f"{cls}()"
        single_line = f"{cls}({', '.join(parts)})"
        if len(single_line) <= N_CHAR_MAX:
            return single_line
        body = "".join(f"    {part},\n" for part in parts)
        return f"{cls}(\n{body})"


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

    def on_pass_begin(self, ctx: TrainingContext) -> None:
        """
        Run at the start of a pass.

        This hook is called after the training loop advances ``ctx.pass_idx`` and
        before the first batch of that pass is processed. It marks the start
        of pass-level work from the callback perspective.

        Solver-specific pass preparation may still occur after this hook and
        before the first batch update. Callbacks should therefore treat this
        hook as notification that a new pass is starting, not as a guarantee
        that all solver-internal pass setup has already completed.
        """
        pass

    def on_pass_end(self, ctx: TrainingContext) -> None:
        """Run at the end of each pass."""
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

    def on_pass_begin(self, ctx: TrainingContext) -> None:
        """Dispatch pass-begin hook to all callbacks."""
        for cb in self._callbacks:
            cb.on_pass_begin(ctx)

    def on_pass_end(self, ctx: TrainingContext) -> None:
        """Dispatch pass-end hook to all callbacks."""
        for cb in self._callbacks:
            cb.on_pass_end(ctx)

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
    each pass and requests a stop if it returns ``True``.

    Tracks previous params and state internally so the context doesn't have to.
    """

    def __init__(self):
        self._prev_params = None
        self._prev_state = None

    def on_pass_begin(self, ctx: TrainingContext) -> None:
        """Save current params and state before the pass runs."""
        self._prev_params = ctx.params
        self._prev_state = ctx.state

    def on_pass_end(self, ctx: TrainingContext) -> None:
        """Check solver convergence criterion and request stop if met."""
        converged = ctx.solver.stochastic_convergence_criterion(
            ctx.params,
            self._prev_params,
            ctx.state,
            self._prev_state,
            ctx.aux,
            ctx.pass_idx,
        )
        if converged:
            ctx.request_stop("Satisfied the solver's convergence criterion.")


class TestLossLogger(Callback):
    """
    Log the loss evaluated on a fixed test set at the requested events.

    Appends ``model.compute_loss(params, X_test, y_test)`` to ``loss_history``
    each time one of the requested ``events`` fires.

    Parameters
    ----------
    X_test :
        Test input (design matrix).
    y_test :
        Test target (e.g. spike counts).
    events :
        Event name or names at which to log the test score. Each must be one
        of ``{"train_begin", "train_end", "pass_begin", "pass_end",
        "batch_begin", "batch_end"}``.

    Attributes
    ----------
    loss_history :
        List of ``(event, pass_idx, batch_idx, test_score)`` tuples, one per
        logged event, recording which event fired and the training position at
        the time.
    """

    # Tell pytest not to collect this as a test class; "Test" refers to the
    # held-out test set, not a unit test.
    __test__ = False

    _VALID_EVENTS = frozenset(
        {
            "train_begin",
            "train_end",
            "pass_begin",
            "pass_end",
            "batch_begin",
            "batch_end",
        }
    )

    def __init__(
        self,
        X_test: Union[DESIGN_INPUT_TYPE, ArrayLike],
        y_test: ArrayLike,
        events: str | Iterable[str],
    ):
        self.loss_history = []
        self.X_test = X_test
        self.y_test = y_test

        events = {events} if isinstance(events, str) else set(events)
        invalid = events - self._VALID_EVENTS
        if invalid:
            raise ValueError(
                f"Unknown events {sorted(invalid)}; "
                f"valid events are {sorted(self._VALID_EVENTS)}."
            )
        self.events = events

        # Wire only the requested hooks; the rest stay no-ops from the base class.
        for event in events:
            setattr(self, f"on_{event}", partial(self._log_test_score, event))

    def _log_test_score(self, event: str, ctx: TrainingContext) -> None:
        test_score = ctx.model.compute_loss(
            (ctx.params.coef, ctx.params.intercept),
            self.X_test,
            self.y_test,
        )
        self.loss_history.append((event, ctx.pass_idx, ctx.batch_idx, test_score))


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
