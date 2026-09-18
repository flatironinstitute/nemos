"""Loop, state and step acceptance shared by NeMoS' native line-search solvers.

``Newton`` and ``ProximalLBFGS`` run the same outer loop: evaluate the gradient, test
convergence, build a direction from a curvature model, scale it with a backtracking line
search. They differ in three places, which are the seams this mixin leaves open:

- ``_curvature`` / ``_init_curvature``: where the curvature model comes from, and what of
  it survives into the next iteration. ``Newton`` rebuilds its Hessian from the data each
  time and carries nothing; a quasi-Newton method carries a history.
- ``_direction``: how a step is read off the curvature model.
- ``_converged``: the default is the gradient-norm test, valid for a smooth objective.
"""

from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optax
from jaxtyping import Array, Bool, Scalar
from optimistix._misc import filter_cond

from .. import tree_utils
from ..typing import Params, StepResult
from ._abstract_solver import OptimizationInfo

if TYPE_CHECKING:
    from ..regularizer import Regularizer

DEFAULT_ATOL = 1e-4
DEFAULT_RTOL = 0.0
DEFAULT_MAX_STEPS = 100

# The parameter pytree. The state follows it, so a ``GLMParams`` fit and a
# ``PopulationGLM`` fit are distinct instantiations rather than ``Any``.
Y = TypeVar("Y")

# ``optax`` searches usable as the step-acceptance rule. Both take the same
# ``(updates, state, params, value=, grad=, value_fn=)`` call with a scalar ``value_fn``;
# zoom autodiffs it internally to test the curvature condition, which is what makes it
# only provisionally applicable to a composite objective.
LINE_SEARCHES: dict[str, Callable] = {
    "backtracking": lambda **kw: optax.scale_by_backtracking_linesearch(
        **{"max_backtracking_steps": 30, **kw}
    ),
    "zoom": lambda **kw: optax.scale_by_zoom_linesearch(
        **{"max_linesearch_steps": 30, **kw}
    ),
}


class LineSearchState(eqx.Module, Generic[Y]):
    """State carried between outer iterations of a :class:`LineSearchLoopMixin` solver."""

    grad_norm: Scalar
    stats: OptimizationInfo
    # Last accepted step. Read by a Cauchy convergence test, and by a quasi-Newton
    # ``_curvature`` as the ``s`` of its curvature pair. Zero at init: a Cauchy test
    # cannot fire on the first iteration regardless, because ``cauchy_termination`` also
    # requires the f-test and ``stats.function_val`` starts NaN.
    y_diff: Y
    # optax's line-search state, whose type is private to the chosen transformation.
    ls_state: Optional[Any] = None
    # Whatever ``_curvature`` carries forward, opaque to the loop. ``None`` for a solver
    # that rebuilds its curvature model from scratch every iteration.
    curvature: Optional[Any] = None


class LineSearchLoopMixin(Generic[Y]):
    """Outer loop, step acceptance and state for NeMoS' native second-order solvers.

    A host calls ``_init_loop`` from its ``__init__`` and fills in ``_direction`` and
    ``_curvature``; everything else has a working default here. The loop is a mixin
    rather than the interface itself for the same reason ``StochasticSolverMixin`` is: it
    is one implementation of ``AbstractSolver``, not the contract, and the solvers using
    it satisfy ``SolverProtocol`` structurally.
    """

    def _init_loop(
        self,
        unregularized_loss: Callable,
        regularizer: "Regularizer",
        regularizer_strength: float | None,
        init_params: Params,
        has_aux: bool,
        jit: bool,
        maxiter: int,
        tol: float,
        rtol: float,
        line_search: str = "backtracking",
    ) -> None:
        """Store the loop settings, resolve the smooth objective and split off its aux."""
        self.has_aux = has_aux
        self.jit = jit
        self.maxiter = maxiter
        self.tol = tol
        self.rtol = rtol
        self.line_search = line_search

        loss_fn = self._resolve_loss(
            unregularized_loss, regularizer, regularizer_strength, init_params
        )
        if has_aux:
            self.fun_with_aux = loss_fn
            self.fun = lambda p, *a: loss_fn(p, *a)[0]
        else:
            self.fun = loss_fn
            self.fun_with_aux = lambda p, *a: (loss_fn(p, *a), None)

        if line_search not in LINE_SEARCHES:
            raise ValueError(
                f"Unknown line_search={line_search!r}. "
                f"Available: {sorted(LINE_SEARCHES)}."
            )
        self._line_search = LINE_SEARCHES[line_search]()

        # Cache
        self._gradient: Callable | None = None

    def _resolve_loss(
        self,
        unregularized_loss: Callable,
        regularizer: "Regularizer",
        regularizer_strength: float | None,
        init_params: Params,
    ) -> Callable:
        """The smooth objective this solver differentiates: the penalized loss.

        A composite solver overrides this, taking the unregularized loss and reaching the
        penalty through a proximal operator instead.
        """
        self.prox = None
        return regularizer.penalized_loss(
            unregularized_loss, params=init_params, strength=regularizer_strength
        )

    def _build_cache(self) -> None:
        if self._gradient is None:
            self._gradient = jax.value_and_grad(self.fun_with_aux, has_aux=True)

    def _init_curvature(self, init_params: Y) -> Any:
        """What ``state.curvature`` starts as. ``None`` unless the host carries a model."""
        del init_params
        return None

    def _curvature(
        self, params: Y, state: LineSearchState[Y], grad: Y, *args: Any
    ) -> tuple[Any, Any]:
        """Curvature model for this step, and the part of it to carry forward.

        The carried part goes into ``state.curvature`` and is never read by the loop.
        """
        raise NotImplementedError

    def _direction(self, grad: Y, H: Any, params: Y) -> Y:
        """The step, before the line search scales it."""
        raise NotImplementedError

    def _converged(
        self, params: Y, state: LineSearchState[Y], grad: Y, fval: Scalar
    ) -> Bool[Array, ""]:
        """Convergence test. ``||grad|| <= tol`` for a smooth objective."""
        del params, state, fval
        return jnp.sqrt(lx.internal.tree_dot(grad, grad)) <= self.tol

    def _line_search_inputs(
        self, params: Y, step: Y, grad: Y, fval: Scalar, *args: Any
    ) -> tuple[Scalar, Y, Callable[[Y], Scalar]]:
        """Value, slope and objective handed to ``self._line_search``.

        ``optax``'s backtracking search forms the slope as ``vdot(updates, grad)``; the
        vector is an argument it never differentiates, so a composite subclass can
        supply a slope accounting for its nonsmooth term without a bespoke search.
        """
        del params, step
        return fval, grad, lambda p: self.fun(p, *args)

    def _apply_or_reject(
        self,
        params: Y,
        step: Y,
        grad: Y,
        state: LineSearchState[Y],
        fval: Scalar,
        *args: Any,
    ) -> tuple[Y, Any]:
        """Accept or reject step based on descent condition and line search."""
        value, slope, value_fn = self._line_search_inputs(
            params, step, grad, fval, *args
        )
        descent = lx.internal.tree_dot(slope, step)

        def accept(_):
            updates, new_ls_state = self._line_search.update(
                step,
                state.ls_state,
                params,
                value=value,
                grad=slope,
                value_fn=value_fn,
            )

            new_params = jax.tree_util.tree_map(
                lambda p, u: p + u,
                params,
                updates,
            )

            return new_params, new_ls_state

        def reject(_):
            return params, state.ls_state

        # A NaN slope is not a stationary point: rejecting it would leave a zero step
        # behind for the Cauchy criterion to report as convergence, so it is stepped on
        # and the iterate goes non-finite instead.
        take_step = jnp.isnan(descent) | (descent < 0)
        new_params, new_ls_state = jax.lax.cond(take_step, accept, reject, None)
        return new_params, new_ls_state

    def init_state(self, init_params: Y, *args: Any) -> LineSearchState[Y]:
        self._build_cache()
        return LineSearchState(
            grad_norm=jnp.array(jnp.inf),
            stats=OptimizationInfo(
                function_val=jnp.array(jnp.nan),
                num_steps=jnp.array(0),
                converged=jnp.array(False),
                reached_max_steps=jnp.array(False),
            ),
            y_diff=jax.tree.map(jnp.zeros_like, init_params),
            ls_state=self._line_search.init(init_params),
            curvature=self._init_curvature(init_params),
        )

    def update(
        self,
        params: Y,
        state: LineSearchState[Y],
        *args: Any,
    ) -> StepResult:
        (fval, aux), grad = self._gradient(params, *args)
        gnorm = jnp.sqrt(lx.internal.tree_dot(grad, grad))
        converged = self._converged(params, state, grad, fval)

        # A carried curvature model may hold a jaxpr, which cannot cross the ``cond``.
        # Only its arrays do; the rest is closed over and restored on the way out.
        static = eqx.filter(state.curvature, eqx.is_array, inverse=True)

        def step(_):
            H, curvature = self._curvature(params, state, grad, *args)
            direction = self._direction(grad, H, params)

            new_params, new_ls_state = self._apply_or_reject(
                params,
                direction,
                grad,
                state,
                fval,
                *args,
            )

            return new_params, new_ls_state, eqx.filter(curvature, eqx.is_array)

        def no_step(_):
            return params, state.ls_state, eqx.filter(state.curvature, eqx.is_array)

        new_params, new_ls_state, new_curvature = filter_cond(
            converged,
            no_step,
            step,
            None,
        )

        new_iter = jnp.where(
            converged,
            state.stats.num_steps,
            state.stats.num_steps + 1,
        )

        new_state = LineSearchState(
            grad_norm=gnorm,
            stats=OptimizationInfo(
                function_val=fval,
                num_steps=new_iter,
                converged=converged,
                reached_max_steps=new_iter >= self.maxiter,
            ),
            y_diff=tree_utils.tree_sub(new_params, params),
            ls_state=new_ls_state,
            curvature=eqx.combine(new_curvature, static),
        )

        return new_params, new_state, aux

    def run(
        self,
        init_params: Y,
        *args: Any,
    ) -> StepResult:
        state = self.init_state(init_params, *args)
        dynamic, static = eqx.partition(state, eqx.is_array)

        def cond(carry):
            _, s = carry
            return (~s.stats.converged) & (s.stats.num_steps < self.maxiter)

        def body(carry):
            p, s = carry
            pnew, snew = self.update(
                p,
                eqx.combine(s, static),
                *args,
            )[:2]  # Discard aux; convergence only needs params and state
            return pnew, eqx.filter(snew, eqx.is_array)

        if self.jit:
            final_params, final_state = eqx.internal.while_loop(
                cond,
                body,
                (init_params, dynamic),
                kind="lax",
            )
        else:
            carry = (init_params, dynamic)
            while cond(carry):
                carry = body(carry)
            final_params, final_state = carry

        _, aux = self.fun_with_aux(final_params, *args)
        return final_params, eqx.combine(final_state, static), aux

    @classmethod
    def get_accepted_arguments(cls) -> set[str]:
        return {"maxiter", "tol", "rtol", "jit"}

    def _get_optim_info(
        self,
        state: LineSearchState[Y],
        **kwargs,
    ) -> OptimizationInfo:
        return state.stats
