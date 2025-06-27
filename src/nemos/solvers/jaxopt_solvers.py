from .abstract_solver import AbstractSolver

import jaxopt

from typing import Generic, TypeVar, ClassVar, Type, NamedTuple, TypeAlias
import inspect

JaxoptSolverState: TypeAlias = NamedTuple
# JaxoptStepResult ~ jaxopt.OptStep


class JaxoptWrapper(AbstractSolver[JaxoptSolverState, jaxopt.OptStep]):
    _solver_cls: Type
    _proximal: bool = False

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

    def __getattr__(self, name: str):
        # without this guard deepcopy leads to a RecursionError
        try:
            solver = object.__getattribute__(self, "_solver")
        except AttributeError:
            raise AttributeError(name)

        return getattr(solver, name)

    @classmethod
    def get_accepted_arguments(cls) -> list[str]:
        own_arguments = set(inspect.getfullargspec(cls).args)
        solver_arguments = set(inspect.getfullargspec(cls._solver_cls).args)

        all_arguments = own_arguments | solver_arguments

        # discard arguments that are passed by BaseRegressor
        all_arguments.discard("self")
        all_arguments.discard("unregularized_loss")
        all_arguments.discard("regularizer")
        all_arguments.discard("regularizer_strength")

        return list(all_arguments)


class JaxoptProximalGradient(JaxoptWrapper):
    _solver_cls = jaxopt.ProximalGradient
    _proximal = True


class JaxoptGradientDescent(JaxoptWrapper):
    _solver_cls = jaxopt.GradientDescent


class JaxoptBFGS(JaxoptWrapper):
    _solver_cls = jaxopt.BFGS


class JaxoptLBFGS(JaxoptWrapper):
    _solver_cls = jaxopt.LBFGS


class JaxoptNonlinearCG(JaxoptWrapper):
    _solver_cls = jaxopt.NonlinearCG
