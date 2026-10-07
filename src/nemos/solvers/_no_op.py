"""Solver stand-in for models whose every parameter is fixed."""

from typing import Any, NamedTuple

import jax.numpy as jnp

from ..typing import Params, StepResult
from ._abstract_solver import AbstractSolver, OptimizationInfo
from ._stochastic_mixins import StochasticSolverMixin


class NoOpState(NamedTuple):
    """State of a solver with nothing to optimize.

    ``converged`` is what the fit entry points inspect to decide whether to warn.
    """

    converged: bool = True


class NoOpSolver(StochasticSolverMixin, AbstractSolver[NoOpState]):
    """Identity solver for an empty active parameter tree.

    When every parameter is fixed there is nothing left to optimize: the pinned values
    already are the solution. The real solvers cannot be initialized on an empty pytree
    (optax's line search indexes into its leaves), so this stands in for them and lets
    the model entry points keep their shape — partition, run, recombine the frozen
    values.

    It implements the full solver interface, including the stochastic one, so that it can
    replace any solver the model was configured with. It is internal: not registered, so
    it cannot be selected by name.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Accept and ignore whatever the real solver would have been given."""
        pass

    @property
    def acceleration_turned_on(self) -> bool:
        """Nothing to accelerate."""
        return False

    @property
    def linesearch_turned_on(self) -> bool:
        """Nothing to search along."""
        return False

    def init_state(self, init_params: Params, *args: Any) -> NoOpState:
        """Return the state of a solver that will not step."""
        return NoOpState()

    def update(self, params: Params, state: NoOpState, *args: Any) -> StepResult:
        """Return ``params`` unchanged."""
        return params, state, None

    def run(self, init_params: Params, *args: Any) -> StepResult:
        """Return ``init_params`` unchanged."""
        return init_params, NoOpState(), None

    def _stochastic_run_impl(
        self,
        init_params: Params,
        data_loader,
        n_passes: int,
        callback,
        ctx,
    ) -> StepResult:
        """Bracket the run with the train callbacks, without touching the data.

        Overrides the inherited loop rather than stepping through it: no batch can move a
        fully fixed parameter tree, and iterating a data loader built for out-of-memory
        data would read it in full to change nothing. The per-pass and per-batch hooks
        are not called because no pass or batch is run.
        """
        state = self.init_state(init_params)
        ctx.params, ctx.state = init_params, state
        callback.on_train_begin(ctx)
        callback.on_train_end(ctx)
        return init_params, state, None

    def _get_optim_info(self, state: NoOpState, **kwargs) -> OptimizationInfo:
        """Report a converged run of zero steps."""
        return OptimizationInfo(
            function_val=None,
            num_steps=jnp.asarray(0),
            converged=jnp.asarray(True),
            reached_max_steps=jnp.asarray(False),
        )

    @classmethod
    def get_accepted_arguments(cls) -> set[str]:
        """No configuration is accepted."""
        return set()
