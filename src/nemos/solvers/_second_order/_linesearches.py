"""Line searches for the second-order solvers.

A line search owns what happens to a direction once it has been computed: the test that
it points downhill, the backtracking that scales it, and the report of what came of that.
Only the value being decreased and the slope certifying descent differ between a smooth
and a composite objective, so that pair is what a subclass supplies.
"""

import abc
from typing import Any, Callable, Generic, Tuple, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Bool, Float
from optax import (
    GradientTransformationExtraArgs,
    ScaleByBacktrackingLinesearchState,
)

from ... import tree_utils

# parameters
Y = TypeVar("Y")


class LineSearchState(eqx.Module):
    """What the search carries between iterations, and what it reports back.

    ``loss_value`` and ``descent`` are the objective and the slope the search actually
    used, so a caller reading them need not know whether a penalty was folded in.
    ``step_taken`` is false both when the direction was refused and when the search ran
    out of budget without reaching sufficient decrease.
    """

    linesearch_state: ScaleByBacktrackingLinesearchState
    loss_value: Float[jax.Array, ""]
    descent: Float[jax.Array, ""]
    step_taken: Bool[jax.Array, ""]


class AbstractLineSearch(abc.ABC, Generic[Y]):
    """Scale a direction and report whether a step was taken."""

    @abc.abstractmethod
    def init(self, params: Y) -> LineSearchState: ...

    @abc.abstractmethod
    def update(
        self,
        params: Y,
        step: Y,
        grad: Y,
        fval: Float[jax.Array, ""],
        state: LineSearchState,
        *args: Any,
    ) -> Tuple[Y, LineSearchState]: ...


class ArmijoBacktracking(AbstractLineSearch[Y], eqx.Module, Generic[Y]):
    """Backtracking Armijo search on a smooth objective.

    ``fun`` is the objective the search evaluates at each trial point, called as
    ``fun(params, *args)``. It has to be the objective ``fval`` comes from, or the
    sufficient-decrease test compares two different functions.
    """

    _line_search: GradientTransformationExtraArgs = eqx.field(static=True)
    fun: Callable[..., Float[jax.Array, ""]] = eqx.field(static=True)

    def init(self, params: Y) -> LineSearchState:
        dtype = jnp.result_type(*jax.tree_util.tree_leaves(params))
        state = self._line_search.init(params)
        return LineSearchState(
            state,
            jnp.array(jnp.inf, dtype=dtype),
            jnp.array(-jnp.inf, dtype=dtype),
            jnp.array(False),
        )

    def _accept(
        self,
        step: Y,
        state: ScaleByBacktrackingLinesearchState,
        params: Y,
        fval: Float[jax.Array, ""],
        slope: Y,
        *args: Any,
    ) -> Tuple[Y, ScaleByBacktrackingLinesearchState, Bool[jax.Array, ""]]:
        def _fun(p: Y) -> Float[jax.Array, ""]:
            return self.fun(p, *args)

        updates, ls_state = self._line_search.update(
            step,
            state,
            params,
            value=fval,
            grad=slope,
            value_fn=_fun,
        )
        # optax returns its last trial even when the sufficient-decrease test was never
        # met within ``max_backtracking_steps``, and that trial can be worse than where
        # it started. A positive ``decrease_error`` is its report.
        found = ls_state.info.decrease_error <= 0
        new_params = jax.tree_util.tree_map(
            lambda p, u: jnp.where(found, p + u, p),
            params,
            updates,
        )
        return new_params, ls_state, found

    def update(
        self,
        params: Y,
        step: Y,
        grad: Y,
        fval: Float[jax.Array, ""],
        state: LineSearchState,
        *args: Any,
    ) -> Tuple[Y, LineSearchState]:
        slope, descent, value = self._slope_descent_value(params, step, grad, fval)

        # Only a negative slope certifies descent. Zero means the iterate is stationary,
        # positive that the direction goes uphill, and NaN that the subproblem diverged;
        # every comparison with NaN is false.
        updates, backtracking_state, step_taken = jax.lax.cond(
            descent < 0,
            self._accept,
            lambda *a, **kw: (params, state.linesearch_state, jnp.array(False)),
            step,
            state.linesearch_state,
            params,
            value,
            slope,
            *args,
        )

        return updates, LineSearchState(backtracking_state, value, descent, step_taken)

    def _slope_descent_value(
        self,
        params: Y,
        step: Y,
        grad: Y,
        fval: Float[jax.Array, ""],
    ) -> Tuple[Y, Float[jax.Array, ""], Float[jax.Array, ""]]:
        """The slope, its contraction with the step, and the value to decrease."""
        return grad, lx.internal.tree_dot(grad, step), fval


class TsengYunBacktracking(ArmijoBacktracking[Y], Generic[Y]):
    """Backtracking search on a composite objective ``f + P``.

    ``penalty`` is the penalty alone, called as ``penalty(params)``. ``fun`` has to be
    the penalized objective, since that is what the search evaluates at each trial point.
    """

    penalty: Callable[[Y], Float[jax.Array, ""]] = eqx.field(static=True)

    def _slope_descent_value(
        self,
        params: Y,
        step: Y,
        grad: Y,
        fval: Float[jax.Array, ""],
    ) -> Tuple[Y, Float[jax.Array, ""], Float[jax.Array, ""]]:
        r"""Rewrite the slope so an Armijo search applies to a nonsmooth objective.

        Tseng & Yun (2009) require the sufficient-decrease slope of a composite
        objective to be

        .. math::
            \Delta = \nabla f^\top d + P(\beta + d) - P(\beta),

        the :math:`P` difference being what makes :math:`\Delta < 0` a descent
        certificate when the objective is nonsmooth. ``optax`` only ever forms
        ``vdot(step, slope)``, so adding the penalty difference along ``step``
        reproduces :math:`\Delta` exactly and the search applies unchanged.
        """
        penalty = self.penalty(params)
        penalty_diff = self.penalty(tree_utils.tree_add(params, step)) - penalty
        sq_norm = lx.internal.tree_dot(step, step)
        slope = tree_utils.tree_add_scalar_mul(
            grad, jnp.where(sq_norm > 0.0, penalty_diff / sq_norm, 0.0), step
        )
        descent = lx.internal.tree_dot(slope, step)
        value = fval + penalty
        return slope, descent, value
