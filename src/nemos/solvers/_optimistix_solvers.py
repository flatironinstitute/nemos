import dataclasses
import inspect
from typing import Any, Callable, ClassVar, Type, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import PyTree

from ..regularizer import Regularizer
from ._abstract_solver import Params
from ._solver_adapter import SolverAdapter

DEFAULT_ATOL = 1e-8
DEFAULT_RTOL = 0.0
DEFAULT_MAX_STEPS = 100_000

_float_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

OptimistixSolverState: TypeAlias = eqx.Module
OptimistixStepResult: TypeAlias = tuple[Params, OptimistixSolverState]


@dataclasses.dataclass
class OptimistixConfig:
    """Defaults for common arguments required by optimistix solvers."""

    # max number of steps
    max_steps: int = DEFAULT_MAX_STEPS
    # options dict passed around within optimistix
    options: dict[str, Any] = dataclasses.field(default_factory=dict)
    # "The shape+dtype of the output of `fn`"
    f_struct: PyTree[jax.ShapeDtypeStruct] = jax.ShapeDtypeStruct((), _float_dtype)
    # this would be the output shape + dtype of the aux variables fn returns
    aux_struct: PyTree[jax.ShapeDtypeStruct] = None
    # "Any Lineax tags describing the structure of the Jacobian matrix d(fn)/dy."
    tags: frozenset = frozenset()
    # sets if the minimisation throws an error if an iterative solver runs out of steps
    throw: bool = False
    # norm used in the Cauchy convergence criterion. Required by all Optimistix solvers.
    norm: Callable = optx.two_norm
    # way of autodifferentiation: https://docs.kidger.site/optimistix/api/adjoints/
    adjoint: optx.AbstractAdjoint = optx.ImplicitAdjoint()
    # whether the objective function returns any auxiliary results.
    # We assume False throughout NeMoS.
    has_aux: bool = False


class OptimistixAdapter(SolverAdapter[OptimistixSolverState, OptimistixStepResult]):
    """Base class for adapters wrapping Optimistix minimizers."""

    _solver_cls: ClassVar[Type]

    # NOTE currently no proximal solvers are in Optimistix

    def __init__(
        self,
        unregularized_loss: Callable,
        regularizer: Regularizer,
        regularizer_strength: float | None,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        **solver_init_kwargs,
    ):
        loss_fn = regularizer.penalized_loss(unregularized_loss, regularizer_strength)
        self.fun = lambda params, args: loss_fn(params, *args)
        self.fun_with_aux = lambda params, args: (loss_fn(params, *args), None)

        atol = solver_init_kwargs.pop("atol", atol)
        rtol = solver_init_kwargs.pop("rtol", rtol)

        if "stepsize" in solver_init_kwargs:
            solver_init_kwargs["search"] = optx.LearningRate(
                solver_init_kwargs.pop("stepsize")
            )

        # take out the arguments that go into minimise, init, terminate and so on
        # and only pass the actually needed things to __init__
        user_args = {}
        for f in dataclasses.fields(OptimistixConfig):
            kw = f.name
            if kw in solver_init_kwargs:
                user_args[kw] = solver_init_kwargs.pop(kw)
        self.config = OptimistixConfig(**user_args)

        self._solver = self._solver_cls(
            atol=atol,
            rtol=rtol,
            norm=self.config.norm,
            **solver_init_kwargs,
        )

        self.stats = {}

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

    @classmethod
    def get_accepted_arguments(cls) -> set[str]:
        own_arguments = set(inspect.getfullargspec(cls.__init__).args)
        solver_arguments = set(inspect.getfullargspec(cls._solver_cls.__init__).args)
        common_optx_arguments = set(
            [f.name for f in dataclasses.fields(OptimistixConfig)]
        )

        all_arguments = own_arguments | solver_arguments | common_optx_arguments

        # discard arguments that are passed by BaseRegressor
        all_arguments.discard("self")
        all_arguments.discard("unregularized_loss")
        all_arguments.discard("regularizer")
        all_arguments.discard("regularizer_strength")

        return all_arguments


class OptimistixBFGS(OptimistixAdapter):
    """Adapter for optimistix.BFGS."""

    _solver_cls = optx.BFGS


class OptimistixOptaxSolver(OptimistixAdapter):
    """Adapter for optimistix.OptaxMinimiser which is an adapter for Optax solvers."""

    _solver_cls = optx.OptaxMinimiser


# class OptimistixLBFGS(OptimistixAdapter):
#    """Adapter for optimistix.LBFGS"""
#
#    _solver_cls = optx.LBFGS


class OptimistixNonlinearCG(OptimistixAdapter):
    """Adapter for optimistix.NonlinearCG."""

    _solver_cls = optx.NonlinearCG
