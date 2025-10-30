import dataclasses
from typing import Any, Callable, ClassVar, Type, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx

from ..regularizer import Regularizer
from ..typing import Pytree
from ._abstract_solver import OptimizationInfo, Params
from ._solver_adapter import SolverAdapter

DEFAULT_ATOL = 1e-8
DEFAULT_RTOL = 0.0
DEFAULT_MAX_STEPS = 100_000

OptimistixSolverState: TypeAlias = eqx.Module
OptimistixStepResult: TypeAlias = tuple[Params, OptimistixSolverState]


def _f_struct_factory():
    """Create the output shape and dtype of the objective function using the currently set JAX precision."""
    current_float_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    return jax.ShapeDtypeStruct((), current_float_dtype)


@dataclasses.dataclass
class OptimistixConfig:
    """
    Collection of arguments required by and cached for methods of Optimistix solvers.

    They rarely need to be overwritten, and the defaults here should suffice.
    The user has the ability to overwrite them with `solver_kwargs`, and on the solver's construction
    they are saved in `OptimistixAdapter.config` for later use: passing them to `optimistix.optimise`, `init`, `step`.
    """

    # max number of steps
    maxiter: int
    # options dict passed around within optimistix
    options: dict[str, Any] = dataclasses.field(default_factory=dict)
    # "The shape+dtype of the output of `fn`"
    # dtype is dynamically determined by calling _f_struct_factory to read the currently
    # set JAX floating point precision from the config
    # f_struct: PyTree[jax.ShapeDtypeStruct] = dataclasses.field(
    f_struct: Pytree = dataclasses.field(default_factory=_f_struct_factory)
    # this would be the output shape + dtype of the aux variables fn returns
    # aux_struct: PyTree[jax.ShapeDtypeStruct] = None
    aux_struct: Pytree = None
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


class OptimistixAdapter(SolverAdapter[OptimistixSolverState]):
    """
    Base class for adapters wrapping Optimistix minimizers.

    Subclasses must define the `_solver_cls` class attribute.
    The `_solver` and `stats` attributes are assumed to exist after construction,
    so if a subclass is overwriting `__init__`, these must be created.

    Note that for backward compatibility the `atol` parameter used in Optimistix
    is referred to as `tol` in NeMoS.
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
        tol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        maxiter: int = DEFAULT_MAX_STEPS,
        **solver_init_kwargs,
    ):
        if "atol" in solver_init_kwargs:
            raise TypeError("Please use tol instead of atol.")

        if "max_steps" in solver_init_kwargs:
            raise TypeError("Please use maxiter instead of max_steps.")

        if self._proximal:
            loss_fn = unregularized_loss
            self.prox = regularizer.get_proximal_operator()
            self.regularizer_strength = regularizer_strength
        else:
            loss_fn = regularizer.penalized_loss(
                unregularized_loss, regularizer_strength
            )
        self.fun = lambda params, args: loss_fn(params, *args)
        self.fun_with_aux = lambda params, args: (loss_fn(params, *args), None)

        # take out the arguments that go into minimise, init, terminate and so on
        # and only pass the actually needed things to __init__
        user_args = {}
        for f in dataclasses.fields(OptimistixConfig):
            kw = f.name
            if kw in solver_init_kwargs:
                user_args[kw] = solver_init_kwargs.pop(kw)
        self.config = OptimistixConfig(maxiter=maxiter, **user_args)

        self._solver = self._solver_cls(
            atol=tol,
            rtol=rtol,
            norm=self.config.norm,
            **solver_init_kwargs,
        )

        self.stats = {}

    def init_state(self, init_params: Params, *args: Any) -> OptimistixSolverState:
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
        *args: Any,
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
        *args: Any,
    ) -> OptimistixStepResult:
        solution = optx.minimise(
            fn=self.fun,
            solver=self._solver,
            y0=init_params,
            args=args,
            options=self.config.options,
            has_aux=self.config.has_aux,
            max_steps=self.config.maxiter,
            adjoint=self.config.adjoint,
            throw=self.config.throw,
            tags=self.config.tags,
        )

        self.stats.update(solution.stats)

        return solution.value, solution.state

    @classmethod
    def get_accepted_arguments(cls) -> set[str]:
        own_and_solver_args = super().get_accepted_arguments()

        # atol is added from wrapped optimistix solvers
        # but currently throughout nemos tol is used
        own_and_solver_args.remove("atol")

        common_optx_arguments = set(
            [f.name for f in dataclasses.fields(OptimistixConfig)]
        )
        all_arguments = own_and_solver_args | common_optx_arguments

        return all_arguments

    @classmethod
    def _note_about_accepted_arguments(cls) -> str:
        return """
        Note that for backward compatibility the `atol` parameter used in Optimistix
        is referred to as `tol` in NeMoS.
        """

    @property
    def maxiter(self) -> int:
        return self.config.maxiter

    def get_optim_info(self, state: OptimistixSolverState) -> OptimizationInfo:
        num_steps = self.stats["num_steps"].item()

        function_val = (
            state.f.item() if hasattr(state, "f") else state.f_info.f.item()
        )  # pyright: ignore

        return OptimizationInfo(
            function_val=function_val,
            num_steps=num_steps,
            converged=state.terminate.item(),  # pyright: ignore
            reached_max_steps=(num_steps == self.maxiter),
        )


class OptimistixBFGS(OptimistixAdapter):
    """Adapter for optimistix.BFGS."""

    _solver_cls = optx.BFGS


class OptimistixNonlinearCG(OptimistixAdapter):
    """Adapter for optimistix.NonlinearCG."""

    _solver_cls = optx.NonlinearCG
