from .abstract_solver import AbstractSolver

from typing import Type, Callable, Any, TypeAlias
from jaxtyping import PyTree, ArrayLike
from dataclasses import dataclass, fields, field

import jax
import jax.numpy as jnp
import optimistix as optx
import equinox as eqx

DEFAULT_ATOL = 1e-8
DEFAULT_RTOL = 0.0
DEFAULT_MAX_STEPS = 100_000

_float_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

ModelParams: TypeAlias = PyTree
OptimistixSolverState: TypeAlias = eqx.Module
OptimistixStepResult: TypeAlias = tuple[ModelParams, OptimistixSolverState]


@dataclass
class OptimistixConfig:
    """Defaults for common arguments required by optimistix solvers."""

    # max number of steps
    max_steps: int = DEFAULT_MAX_STEPS
    # options dict passed around within optimistix. e.g. ProximalGradient uses it to pass regularizer_strength
    options: dict[str, Any] = field(default_factory=dict)
    # "The shape+dtype of the output of `fn`"
    f_struct: PyTree[jax.ShapeDtypeStruct] = jax.ShapeDtypeStruct((), _float_dtype)
    # this would be the output shape + dtype of the aux variables fn returns
    aux_struct: PyTree[jax.ShapeDtypeStruct] = None
    # "Any Lineax tags describing the structure of the Jacobian matrix d(fn)/dy."
    tags: frozenset = frozenset()
    # increase optimistix's default of 256
    max_steps: int = DEFAULT_MAX_STEPS
    # sets if the minimisation throws an error if an iterative solver runs out of steps
    throw: bool = False
    # norm used in the Cauchy convergence criterion
    norm: Callable = optx.two_norm
    adjoint: optx.AbstractAdjoint = optx.ImplicitAdjoint()
    has_aux: bool = False


class OptimistixAdapter(AbstractSolver[OptimistixSolverState, OptimistixStepResult]):
    _solver_cls: Type

    # NOTE currently no proximal solvers are in Optimistix

    def __init__(
        self,
        unregularized_loss,
        regularizer,
        regularizer_strength,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        **solver_init_kwargs,
    ):
        loss_fn = regularizer.penalized_loss(unregularized_loss, regularizer_strength)
        self.fun = lambda params, args: loss_fn(params, *args)
        self.fun_with_aux = lambda params, args: (loss_fn(params, *args), None)

        solver_init_kwargs = self._replace_tol(solver_init_kwargs)
        atol = solver_init_kwargs.pop("atol", atol)
        rtol = solver_init_kwargs.pop("rtol", rtol)

        solver_init_kwargs = self._replace_maxiter(solver_init_kwargs)

        if "stepsize" in solver_init_kwargs:
            solver_init_kwargs["search"] = optx.LearningRate(
                solver_init_kwargs.pop("stepsize")
            )

        # take out the arguments that go into minimise, init, terminate and so on
        # and only pass the actually needed things to __init__

        # user_args = {}
        # for f in fields(OptimistixConfig):
        #    kw = f.name
        #    if kw in solver_init_kwargs:
        #        user_args[kw] = solver_init_kwargs.pop(kw)

        # another solution for the same
        user_args = {
            f.name: solver_init_kwargs.pop(f.name)
            for f in fields(OptimistixConfig)
            if f.name in solver_init_kwargs
        }

        self.config = OptimistixConfig(**user_args)

        self._solver = self._solver_cls(atol=atol, rtol=rtol, **solver_init_kwargs)

        self.stats = {}

    def _replace_maxiter(self, solver_init_kwargs):
        if "maxiter" in solver_init_kwargs:
            solver_init_kwargs["max_steps"] = solver_init_kwargs.pop("maxiter")

        return solver_init_kwargs

    def _replace_tol(self, solver_init_kwargs):
        if "tol" in solver_init_kwargs:
            if "atol" in solver_init_kwargs:
                raise ValueError("tol and atol can't both be given.")
            if "rtol" in solver_init_kwargs:
                raise ValueError("tol and rtol can't both be given.")

            solver_init_kwargs["atol"] = solver_init_kwargs.pop("tol")
            solver_init_kwargs["rtol"] = 0.0

        return solver_init_kwargs

    def init_state(self, init_params, *args) -> OptimistixSolverState:
        return self._solver.init(
            self.fun,
            init_params,
            args,
            self.config.options,
            self.config.f_struct,
            self.config.aux_struct,
            self.config.tags,
        )

    def update(self, params, state, *args) -> OptimistixStepResult:
        new_params, state, aux = self._solver.step(
            fn=self.fun_with_aux,
            y=params,
            args=args,
            state=state,
            options=self.config.options,
            tags=self.config.tags,
        )

        return new_params, state

    def run(
        self,
        init_params,
        *args,
    ) -> OptimistixStepResult:
        solution = optx.minimise(
            fn=self.fun,
            solver=self._solver,
            y0=init_params,
            args=args,
            options=self.config.options,
            has_aux=self.config.has_aux,
            max_steps=self.config.max_steps,
            adjoint=self.config.adjoint,
            throw=self.config.throw,
            tags=self.config.tags,
        )

        self.stats.update(solution.stats)

        return solution.value, solution.state


class OptimistixBFGS(OptimistixAdapter):
    _solver_cls = optx.BFGS


class OptimistixOptaxSolver(OptimistixAdapter):
    _solver_cls = optx.OptaxMinimiser


# class OptimistixLBFGS(OptimistixAdapter):
#    _solver_cls = optx.LBFGS
