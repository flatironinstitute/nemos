"""Base class for adapters wrapping JAXopt solvers."""

from .solver_adapter import SolverAdapter

import jaxopt

from typing import ClassVar, Type, NamedTuple, TypeAlias

JaxoptSolverState: TypeAlias = NamedTuple
# JaxoptStepResult ~ jaxopt.OptStep


class JaxoptWrapper(SolverAdapter[JaxoptSolverState, jaxopt.OptStep]):
    """
    Base class for adapters wrapping JAXopt solvers.

    Besides `_solver_cls`, for proximal solvers the `_proximal` class variable
    needs to be set to `True`
    """

    _solver_cls: ClassVar[Type]
    _proximal: ClassVar[bool] = False

    def __init__(
        self,
        unregularized_loss,
        regularizer,
        regularizer_strength,
        **solver_init_kwargs,
    ):
        self.fun = self._make_fun(unregularized_loss, regularizer, regularizer_strength)

        if self._proximal:
            solver_init_kwargs["prox"] = regularizer.get_proximal_operator()

        self.regularizer_strength = regularizer_strength

        self._solver = self._solver_cls(
            fun=self.fun,
            **solver_init_kwargs,
        )

    def _make_fun(self, unregularized_loss, regularizer, regularizer_strength):
        if self._proximal:
            return unregularized_loss
        else:
            return regularizer.penalized_loss(unregularized_loss, regularizer_strength)

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

    def init_state(self, init_params, *args) -> JaxoptSolverState:
        return self._solver.init_state(init_params, *self._extend_args(args))

    def update(self, params, state, *args) -> jaxopt.OptStep:
        return self._solver.update(params, state, *self._extend_args(args))

    def run(self, init_params, *args) -> jaxopt.OptStep:
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
