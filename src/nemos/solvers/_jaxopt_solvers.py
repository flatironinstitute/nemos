"""Base class for adapters wrapping JAXopt solvers."""

from typing import Any, Callable, ClassVar, NamedTuple, Type, TypeAlias

from nemos.third_party.jaxopt import jaxopt

from ..regularizer import Regularizer
from ._abstract_solver import OptimizationInfo, Params
from ._solver_adapter import SolverAdapter

JaxoptSolverState: TypeAlias = NamedTuple
JaxoptStepResult: TypeAlias = jaxopt.OptStep  # this is just a namedtuple(params, state)


class JaxoptAdapter(SolverAdapter[JaxoptSolverState]):
    """
    Base class for adapters wrapping JAXopt solvers.

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
        **solver_init_kwargs,
    ):
        if self._proximal:
            self.fun = unregularized_loss
            solver_init_kwargs["prox"] = regularizer.get_proximal_operator()
        else:
            self.fun = regularizer.penalized_loss(
                unregularized_loss, regularizer_strength
            )

        self.regularizer_strength = regularizer_strength

        # Prepend the regularizer strength to args for proximal solvers.
        # Methods of `jaxopt.ProximalGradient` expect `hyperparams_prox` before
        # the objective function's arguments, while others do not need this.
        self.hyperparams_prox = (self.regularizer_strength,) if self._proximal else ()

        self._solver = self._solver_cls(
            fun=self.fun,
            **solver_init_kwargs,
        )

    def init_state(self, init_params: Params, *args: Any) -> JaxoptSolverState:
        return self._solver.init_state(init_params, *self.hyperparams_prox, *args)

    def update(
        self, params: Params, state: JaxoptSolverState, *args: Any
    ) -> JaxoptStepResult:
        return self._solver.update(params, state, *self.hyperparams_prox, *args)

    def run(self, init_params: Params, *args: Any) -> JaxoptStepResult:
        return self._solver.run(init_params, *self.hyperparams_prox, *args)

    @classmethod
    def get_accepted_arguments(cls) -> set[str]:
        arguments = super().get_accepted_arguments()
        # prox is read from the regularizer, not provided as a solver argument
        if cls._proximal:
            arguments.remove("prox")
        return arguments

    def get_optim_info(self, state: JaxoptSolverState) -> OptimizationInfo:
        num_steps = state.iter_num.item()  # pyright: ignore
        function_val = (
            state.value if hasattr(state, "value") else None
        )  # pyright: ignore

        return OptimizationInfo(
            function_val=function_val,  # pyright: ignore
            num_steps=num_steps,
            converged=state.error.item() <= self.tol,  # pyright: ignore
            reached_max_steps=(num_steps == self.maxiter),
        )

    @property
    def maxiter(self):
        return self._solver.maxiter


class JaxoptProximalGradient(JaxoptAdapter):
    """
    Adapter for `jaxopt.ProximalGradient`.

    The `prox` argument passed to `jaxopt.ProximalGradient`
    is read from the regularizer.
    """

    _solver_cls = jaxopt.ProximalGradient
    _proximal = True


class JaxoptGradientDescent(JaxoptAdapter):
    """Adapter for jaxopt.GradientDescent."""

    _solver_cls = jaxopt.GradientDescent


class JaxoptBFGS(JaxoptAdapter):
    """Adapter for jaxopt.BFGS."""

    _solver_cls = jaxopt.BFGS


class JaxoptLBFGS(JaxoptAdapter):
    """Adapter for jaxopt.LBFGS."""

    _solver_cls = jaxopt.LBFGS


class JaxoptNonlinearCG(JaxoptAdapter):
    """Adapter for jaxopt.NonlinearCG."""

    _solver_cls = jaxopt.NonlinearCG
