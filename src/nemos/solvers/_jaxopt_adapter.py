"""Base class for adapters wrapping JAXopt-style solvers."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    NamedTuple,
    Tuple,
    Type,
    TypeAlias,
)

import lazy_loader as lazy

from ..typing import Aux, Params

if TYPE_CHECKING:
    from ..regularizer import Regularizer

from ._abstract_solver import AbstractSolverState, OptimizationInfo
from ._solver_adapter import SolverAdapter

jax = lazy.load("jax")

JaxoptSolverState: TypeAlias = NamedTuple


class JaxoptAdapterState(AbstractSolverState[JaxoptSolverState]):
    """Solver state for JAXopt-based adapters."""


JaxoptStepResult: TypeAlias = Tuple[Params, JaxoptAdapterState, Aux]


class JaxoptAdapter(SolverAdapter[JaxoptAdapterState]):
    """
    Base class for adapters wrapping JAXopt-style solvers.

    Besides `_solver_cls`, for proximal solvers the `_proximal` class variable
    needs to be set to `True`.
    """

    _solver_cls: ClassVar[Type]
    _proximal: ClassVar[bool] = False

    def __init__(
        self,
        unregularized_loss: Callable,
        regularizer: Regularizer,
        regularizer_strength: float | None,
        has_aux: bool,
        init_params: Params | None = None,
        **solver_init_kwargs,
    ):
        if self._proximal:
            self.fun = unregularized_loss
            solver_init_kwargs["prox"] = regularizer.get_proximal_operator(
                params=init_params, strength=regularizer_strength
            )
        else:
            self.fun = regularizer.penalized_loss(
                unregularized_loss, params=init_params, strength=regularizer_strength
            )

        self.regularizer_strength = regularizer_strength

        # Prepend the regularizer strength to args for proximal solvers.
        # Methods of `jaxopt.ProximalGradient` expect `hyperparams_prox` before
        # the objective function's arguments, while others do not need this.
        self.hyperparams_prox = (self.regularizer_strength,) if self._proximal else ()

        self._solver = self._solver_cls(
            fun=self.fun,
            has_aux=has_aux,
            **solver_init_kwargs,
        )

    def init_state(self, init_params: Params, *args: Any) -> JaxoptAdapterState:
        return JaxoptAdapterState(
            solver_state=self._solver.init_state(
                init_params, *self.hyperparams_prox, *args
            ),
            stats=OptimizationInfo(
                function_val=jax.numpy.nan,  # pyright: ignore
                num_steps=jax.numpy.array(0),
                converged=jax.numpy.array(False),  # pyright: ignore
                reached_max_steps=jax.numpy.array(False),
            ),
        )

    def update(
        self, params: Params, state: JaxoptAdapterState, *args: Any
    ) -> JaxoptStepResult:
        params, solver_state = self._solver.update(
            params, state.solver_state, *self.hyperparams_prox, *args
        )
        aux = self._extract_aux(solver_state, fallback_name="aux_batch")
        stats = self._get_optim_info(solver_state)
        state = JaxoptAdapterState(solver_state=solver_state, stats=stats)
        return (params, state, aux)

    def run(self, init_params: Params, *args: Any) -> JaxoptStepResult:
        params, solver_state = self._solver.run(
            init_params, *self.hyperparams_prox, *args
        )
        aux = self._extract_aux(solver_state, fallback_name="aux_full")
        stats = self._get_optim_info(solver_state)
        state = JaxoptAdapterState(solver_state=solver_state, stats=stats)
        return (params, state, aux)

    @classmethod
    def get_accepted_arguments(cls) -> set[str]:
        arguments = super().get_accepted_arguments()
        # prox is read from the regularizer, not provided as a solver argument
        if cls._proximal:
            arguments.remove("prox")
        return arguments

    def _get_optim_info(self, state: JaxoptSolverState, **kwargs) -> OptimizationInfo:
        num_steps = state.iter_num.item()  # pyright: ignore
        function_val = (
            state.value if hasattr(state, "value") else None
        )  # pyright: ignore

        return OptimizationInfo(
            function_val=function_val,  # pyright: ignore
            num_steps=num_steps,
            converged=state.error.item() <= self.tol,  # pyright: ignore
            reached_max_steps=(num_steps >= self.maxiter),
        )

    @property
    def maxiter(self):
        return self._solver.maxiter

    def _extract_aux(self, state: JaxoptAdapterState, fallback_name: str):
        """
        Return auxiliary output from a solver state.

        Prefers `state.aux` when present; otherwise falls back to the provided field name
        (e.g., `aux_batch` for SVRG updates or `aux_full` for SVRG run).
        """
        # solvers imported from jaxopt have state.aux
        if hasattr(state, "aux"):
            return state.aux

        # for SVRG get state.aux_batch or state.aux_full
        return getattr(state, fallback_name)
