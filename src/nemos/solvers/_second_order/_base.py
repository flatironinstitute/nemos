"""What every second-order solver in this module shares.

A solver built here is an assembly rather than a hierarchy: it holds a curvature model,
a direction, a line search and a :class:`~nemos.solvers._second_order._loop.Loop`, and
the iteration itself lives in the loop. What is left on the solver is the wiring --
building the objective, seeding the state, and choosing between the compiled and the
Python loop -- and that part does not vary between solvers, so it lives here.

Subclasses supply their ``__init__``, their state class through :meth:`init_state`, and
the keyword arguments they accept.
"""

from __future__ import annotations

import abc
from typing import Any, ClassVar, Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Scalar

from ...typing import StepResult
from .._abstract_solver import OptimizationInfo
from ._curvature import AbstractCurvature
from ._direction import AbstractDirection
from ._linesearches import AbstractLineSearch
from ._loop import Loop

# parameters
Y = TypeVar("Y")
# the state the solver carries between iterations
S = TypeVar("S")


class SecondOrderState(eqx.Module, Generic[Y]):
    """What every second-order solver carries between iterations.

    There is one class rather than one per solver because the curvature models own
    whatever else they need: L-BFGS keeps its pair history, including the previous
    gradient, inside ``hessian_update_state``, and Newton leaves that field ``None``.
    """

    grad_norm: Scalar
    stats: OptimizationInfo
    # Last accepted step, read by the Cauchy convergence test and, for a model that
    # builds itself from pairs, as the ``s`` of the curvature pair.
    y_diff: Y
    # Set when an iteration produced no usable step: the direction was not a descent
    # direction, or it was not finite. It ends the run, and it is not convergence.
    no_step_found: Bool[Array, ""]
    # optax's line-search state, whose type is private to the chosen transformation.
    ls_state: Any = None
    # the identity-shift ladder for ``Newton``, ``None`` for the proximal solvers
    direction_state: Any = None
    # whatever the curvature model carries; ``None`` for a model that assembles the
    # Hessian afresh at every iterate
    hessian_update_state: Any = None


class AbstractSecondOrderSolver(abc.ABC, Generic[Y, S]):
    """Wiring shared by every second-order solver: objective, state seed, run loop."""

    # Whether the penalty is modelled by the quadratic. False for solvers that reach the
    # penalty through a proximal operator instead.
    _proximal: ClassVar[bool] = False

    # Set by every subclass' ``__init__``.
    has_aux: bool
    jit: bool
    fun: Any
    fun_with_aux: Any
    loop: Loop
    curvature: AbstractCurvature
    direction: AbstractDirection
    _line_search: AbstractLineSearch

    @property
    def maxiter(self) -> int:
        return self.loop.maxiter

    @property
    def tol(self) -> float:
        return self.loop.atol

    @property
    def rtol(self) -> float:
        return self.loop.rtol

    def _set_objective(self, loss_fn: Any, has_aux: bool) -> None:
        """Split the objective into its scalar and its ``(scalar, aux)`` forms."""
        if has_aux:
            self.fun_with_aux = loss_fn
            self.fun = lambda p, *a: loss_fn(p, *a)[0]
        else:
            self.fun = loss_fn
            self.fun_with_aux = lambda p, *a: (loss_fn(p, *a), None)

    def _scalar_dtype(self, init_params: Y, *args: Any) -> jnp.dtype:
        """The objective's dtype, which the state's scalars must already carry.

        The ``while_loop`` carry fails to typecheck otherwise.
        """
        return jax.eval_shape(self.fun, init_params, *args).dtype

    def _initial_y_diff(self, init_params: Y) -> Y:
        """The step the first convergence test reads, before any step has been taken.

        Infinite, so a Cauchy criterion on the step alone cannot fire on iteration one.
        A solver whose curvature model reads ``y_diff`` as a curvature pair overrides
        this: see :meth:`~nemos.solvers._second_order._lbfgs.ProximalLBFGS._initial_y_diff`.
        """
        return jax.tree.map(lambda x: jnp.full_like(x, jnp.inf), init_params)

    def _common_state_fields(self, init_params: Y, *args: Any) -> dict[str, Any]:
        """The state fields every second-order solver carries, ready to splat."""
        scalar_dtype = self._scalar_dtype(init_params, *args)
        return dict(
            grad_norm=jnp.asarray(jnp.inf, dtype=scalar_dtype),
            stats=OptimizationInfo(
                function_val=jnp.asarray(jnp.nan, dtype=scalar_dtype),
                num_steps=jnp.array(0),
                converged=jnp.array(False),
                reached_max_steps=jnp.array(False),
            ),
            ls_state=self._line_search.init(init_params),
            y_diff=self._initial_y_diff(init_params),
            no_step_found=jnp.array(False),
        )

    @abc.abstractmethod
    def init_state(self, init_params: Y, *args: Any) -> S: ...

    def update(self, params: Y, state: S, *args: Any) -> StepResult:
        return self.loop.update(
            params, state, self.curvature, self.direction, self._line_search, *args
        )

    def run(self, init_params: Y, *args: Any) -> StepResult:
        """Iterate to convergence, to a stall, or to ``maxiter``.

        ``jit`` picks which of the two loops in :meth:`_run` executes, so the compiled
        path has to be reached through a separate method: decorating this one would trace
        the Python loop and fail on its data-dependent condition.
        """
        if self.jit:
            return self._run_jit(init_params, *args)
        return self._run(init_params, *args)

    @eqx.filter_jit
    def _run_jit(self, init_params: Y, *args: Any) -> StepResult:
        return self._run(init_params, *args)

    def _run(self, init_params: Y, *args: Any) -> StepResult:
        state = self.init_state(init_params, *args)
        final_params, final_state = self.loop.run(
            init_params,
            state,
            self.curvature,
            self.direction,
            self._line_search,
            self.jit,
            *args,
        )
        # ``fun_with_aux`` is the objective again: only pay for it when there is an aux
        # to collect, since without one it returns a ``None`` the caller already knows.
        aux = self.fun_with_aux(final_params, *args)[1] if self.has_aux else None
        return final_params, final_state, aux

    @classmethod
    def get_accepted_arguments(cls) -> set[str]:
        return {"maxiter", "tol", "rtol", "jit"}

    def _get_optim_info(self, state: S, **kwargs: Any) -> OptimizationInfo:
        return state.stats
