"""Base class for adapters wrapping JAXopt solvers."""

from typing import Callable, ClassVar, NamedTuple, Type, TypeAlias

from nemos.third_party.jaxopt import jaxopt

from ..regularizer import Regularizer
from ._abstract_solver import Params
from ._solver_adapter import SolverAdapter

JaxoptSolverState: TypeAlias = NamedTuple
# JaxoptStepResult ~ jaxopt.OptStep

# TODO do we want the JAXopt solvers to use the same tolerance and max_steps as the Optimistix solvers?


class JaxoptWrapper(SolverAdapter[JaxoptSolverState, jaxopt.OptStep]):
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

        self._solver = self._solver_cls(
            fun=self.fun,
            **solver_init_kwargs,
        )

    def _extend_args(self, args):
        """
        Prepend the regularizer strength to args for proximal solvers.

        Methods of `jaxopt.ProximalGradient` expect `hyperparams_prox` before
        the objective function's arguments, while others do not need this.
        """
        if self._proximal:
            return (self.regularizer_strength, *args)
        else:
            return args

    def init_state(self, init_params: Params, *args) -> JaxoptSolverState:
        return self._solver.init_state(init_params, *self._extend_args(args))

    def update(self, params: Params, state: JaxoptSolverState, *args) -> jaxopt.OptStep:
        return self._solver.update(params, state, *self._extend_args(args))

    def run(self, init_params: Params, *args) -> jaxopt.OptStep:
        return self._solver.run(init_params, *self._extend_args(args))


class JaxoptProximalGradient(JaxoptWrapper):
    """Adapter for jaxopt.ProximalGradient."""

    _solver_cls = jaxopt.ProximalGradient
    _proximal = True


class JaxoptGradientDescent(JaxoptWrapper):
    """Adapter for jaxopt.GradientDescent."""

    _solver_cls = jaxopt.GradientDescent


class JaxoptBFGS(JaxoptWrapper):
    """Adapter for jaxopt.BFGS."""

    _solver_cls = jaxopt.BFGS


class JaxoptLBFGS(JaxoptWrapper):
    """Adapter for jaxopt.LBFGS."""

    _solver_cls = jaxopt.LBFGS


class JaxoptNonlinearCG(JaxoptWrapper):
    """Adapter for jaxopt.NonlinearCG."""

    _solver_cls = jaxopt.NonlinearCG
