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

OptimistixSolverState: TypeAlias = eqx.Module
OptimistixStepResult: TypeAlias = tuple[Params, OptimistixSolverState]


def _current_float_dtype() -> jnp.dtype:
    """Return the floating point dtype matching the current JAX precision."""
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


@dataclasses.dataclass
class OptimistixConfig:
    """Defaults for common arguments required by optimistix solvers."""

    # max number of steps
    max_steps: int = DEFAULT_MAX_STEPS
    # options dict passed around within optimistix
    options: dict[str, Any] = dataclasses.field(default_factory=dict)
    # "The shape+dtype of the output of `fn`"
    f_struct: PyTree[jax.ShapeDtypeStruct] = dataclasses.field(
        default_factory=lambda: jax.ShapeDtypeStruct((), _current_float_dtype())
    )
    # this would be the output shape + dtype of the aux variables fn returns
    aux_struct: PyTree[jax.ShapeDtypeStruct] = None
    # "Any Lineax tags describing the structure of the Jacobian matrix d(fn)/dy."
    tags: frozenset = frozenset()
    # sets if the minimisation throws an error if an iterative solver runs out of steps
    throw: bool = False
    # norm used in the Cauchy convergence criterion. Required by all Optimistix solvers.
    norm: Callable = optx.max_norm
    # way of autodifferentiation: https://docs.kidger.site/optimistix/api/adjoints/
    adjoint: optx.AbstractAdjoint = optx.ImplicitAdjoint()
    # whether the objective function returns any auxiliary results.
    # We assume False throughout NeMoS.
    has_aux: bool = False


class OptimistixAdapter(SolverAdapter[OptimistixSolverState, OptimistixStepResult]):
    """
    Base class for adapters wrapping Optimistix minimizers.

    Subclasses must define the `_solver_cls` class attribute.
    The `_solver` and `stats` attributes are assumed to exist after construction,
    so if a subclass is overwriting ``__init__`, these must be created.
    """

    _solver_cls: ClassVar[Type]
    _solver: optx.AbstractMinimiser

    # used for storing info after an optimization run
    # updated with the dict from an optimistix._solution.Solution.stats
    stats: dict[str, Any]

    _proximal: ClassVar[bool] = False

    def __init__(
        self,
        unregularized_loss: Callable,
        regularizer: Regularizer,
        regularizer_strength: float | None,
        atol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        **solver_init_kwargs,
    ):
        if self._proximal:
            loss_fn = unregularized_loss
            solver_init_kwargs["prox"] = regularizer.get_proximal_operator()
            solver_init_kwargs["regularizer_strength"] = regularizer_strength
        else:
            loss_fn = regularizer.penalized_loss(
                unregularized_loss, regularizer_strength
            )
        self.fun = lambda params, args: loss_fn(params, *args)
        self.fun_with_aux = lambda params, args: (loss_fn(params, *args), None)

        # by default Optimistix doesn't expose the search attribute of concrete solvers
        # but in our custom implementations we might want to flexibly switch between
        # linesearches and constant learning rates depending on whether `stepsize` is passed
        # if "stepsize" in solver_init_kwargs:
        #    solver_init_kwargs["search"] = optx.LearningRate(
        #        solver_init_kwargs.pop("stepsize")
        #    )

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

    def init_state(self, init_params: Params, *args) -> OptimistixSolverState:
        return self._solver.init(
            self.fun_with_aux,
            init_params,
            args,
            self.config.options,
            self.config.f_struct,
            self.config.aux_struct,
            self.config.tags,
        )

    def update(
        self,
        params: Params,
        state: OptimistixSolverState,
        *args,
    ) -> OptimistixStepResult:
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
        init_params: Params,
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
        own_and_solver_args = super().get_accepted_arguments()
        common_optx_arguments = set(
            [f.name for f in dataclasses.fields(OptimistixConfig)]
        )
        all_arguments = own_and_solver_args | common_optx_arguments

        # in case we decide to create a LearningRate search from stepsize
        # all_arguments.add("stepsize")

        return all_arguments

    @property
    def maxiter(self) -> int:
        return self.config.max_steps


class OptimistixBFGS(OptimistixAdapter):
    """Adapter for optimistix.BFGS."""

    _solver_cls = optx.BFGS


class OptimistixOptaxSolver(OptimistixAdapter):
    """Adapter for optimistix.OptaxMinimiser which is an adapter for Optax solvers."""

    _solver_cls = optx.OptaxMinimiser


class OptimistixNonlinearCG(OptimistixAdapter):
    """Adapter for optimistix.NonlinearCG."""

    _solver_cls = optx.NonlinearCG
