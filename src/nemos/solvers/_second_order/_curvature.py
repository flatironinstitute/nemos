from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from jaxtyping import Array, Float, PyTree, Scalar
from optimistix._solver.limited_memory_bfgs import _lbfgs_hessian_operator_fn

from ..._hess import HessianTag
from ._utils import map_blocks

# what a curvature model carries between iterations
S = TypeVar("S")
# how B_k is represented at this iterate, which is what ``hvp`` multiplies from: the
# pair history in the compact representation of Byrd et al. [1]_, or the assembled
# Hessian. Neither is the curvature; both determine it.
R = TypeVar("R")
# parameters
Y = TypeVar("Y")


class _LBFGSHessianUpdateState(eqx.Module, Generic[Y]):
    r"""
    State variables for LBFGS.

    State variables for Algorithm 3.2 of:

        Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994).
        "Representations of quasi-Newton matrices and their use in limited memory
        methods." *Mathematical Programming*, 63(1), 129–156.

    This holds a ring buffer of the history of differences in the optimisation variable
    `y` and the gradients `grad`, at the last `n` accepted steps, where `n` is the
    history length.

    **Arguments:**

    - `index_start`: Index of the most recent update in the circular buffer.
    - `y_diff_history`: Circular buffer containing the history of differences in the
        optimisation variable `y` between consecutive accepted steps. In most textbooks
        and in the paper above, the difference in the optimisation variable at iteration
        `k` is denoted as `s_k` and refers to $y_{k+1} - y_k$`. We use the term `y_diff`
        here to maintain consistency with variable names in Optimistix.
        The oldest element is at index `index_start % history_len`.
    - `grad_diff_history`: Circular buffer containing the history of differences in the
        gradient values. Similarly to `y_diff_history`, we follow the Optimistix
        nomenclature rather than the textbook one, where the difference in the gradients
        is usually denoted as `y_k`.
        Indexation is handled as for `y_diff_history`.
    - `y_diff_grad_diff_cross_inner`: Lower triangular matrix with the inner products
        between parameters and gradient difference histories. This parameter corresponds
        to `L_k` in the paper, see def. (2.18), with

            L[ij] = y_diff[i-1]^T grad_diff[j-1] if i > j, 0 otherwise,

        for each iteration `k` that is part of the history (k omitted for clarity).
    - `y_diff_grad_diff_inner`: Array containing the inner products of the differences
        in `y` and the gradients. This parameter corresponds to `diag(D_k)` from the
        paper, see def. (2.7). `[D_k]_{ii} = s_{i-1}^T \cdot y_{i-1}`.
    - `y_diff_cross_inner`: outer product of the parameter difference history. In the
        paper notation, this is equal to `S_{k-1}^T \cdot S_{k-1}`, which is a matrix of
        shape `(history_length, history_length)`.
    - `initial_scale`: NeMoS addition, not part of the paper's state. The `B_0 = c I`
        used while the history is empty; see :meth:`LBFGSCurvature.hvp`.
    """

    index_start: Scalar
    history_length: int = eqx.field(static=True)
    y_diff_history: PyTree[Y]
    grad_diff_history: PyTree[Y]
    y_diff_grad_diff_cross_inner: Float[Array, " history_length history_length"]
    y_diff_grad_diff_inner: Float[Array, " history_length"]
    y_diff_cross_inner: Float[Array, " history_length history_length"]
    initial_scale: Scalar


class AbstractCurvature(eqx.Module, ABC, Generic[Y, S, R]):
    """A model of the loss curvature, in whichever form the model has it.

    ``update`` hands back three things: a Hessian-vector product, usable by any
    direction; the assembled Hessian, or ``None`` when the model never assembles one;
    and the state to carry into the next iteration.

    ``S`` and ``R`` are separate because they coincide for only one of the two models:
    L-BFGS represents B_k by the same pair history it carries forward, while Newton
    carries nothing and represents it by the Hessian it has just assembled.
    """

    @abstractmethod
    def init(self, params: Y, *args: Any) -> S: ...

    @abstractmethod
    def update(
        self,
        state: S,
        params: Y,
        y_diff: Y,
        grad_diff: Y,
        *args: Any,
    ) -> tuple[Callable[[Y, HessianTag | None], Y], PyTree[Array] | None, S]: ...

    @abstractmethod
    def hvp(
        self, hessian_repr: R, params: Y, v: Y, hessian_tag: HessianTag | None
    ) -> Y: ...


def _batched_tree_zeros_like(y: Y, batch_dimension: int) -> PyTree[Array]:
    return jtu.tree_map(lambda y: jnp.zeros((batch_dimension, *y.shape)), y)


v_tree_dot = jax.vmap(lx.internal.tree_dot, in_axes=(0, None), out_axes=0)


class LBFGSCurvature(
    AbstractCurvature[Y, _LBFGSHessianUpdateState, _LBFGSHessianUpdateState], Generic[Y]
):
    history_length: int = eqx.field(static=True)

    def init(self, params: Y, *args: Any) -> _LBFGSHessianUpdateState[Y]:
        """Build the identity curvature model and the state that will accumulate it."""
        hess_state = _LBFGSHessianUpdateState(
            index_start=jnp.array(0),
            history_length=self.history_length,
            y_diff_history=_batched_tree_zeros_like(params, self.history_length),
            grad_diff_history=_batched_tree_zeros_like(params, self.history_length),
            y_diff_grad_diff_cross_inner=jnp.zeros(
                (self.history_length, self.history_length)
            ),
            y_diff_grad_diff_inner=jnp.ones(self.history_length),
            y_diff_cross_inner=jnp.eye(self.history_length),
            # ``update`` recomputes this from the gradient, so it has to carry the
            # parameter dtype from the start or the two ``lax.cond`` branches disagree.
            initial_scale=jnp.array(
                1.0, dtype=jnp.result_type(*jtu.tree_leaves(params))
            ),
        )

        return hess_state  # pyright: ignore

    def update(
        self,
        state: _LBFGSHessianUpdateState[Y],
        params: Y,
        y_diff: Y,
        grad_diff: Y,
        *args: Any,
    ) -> tuple[Callable[[Y, HessianTag | None], Y], None, _LBFGSHessianUpdateState[Y]]:

        history_length = state.history_length

        # Update only if the inner product is positive, to maintain positive definiteness
        # of the Hessian approximation.
        inner = lx.internal.tree_dot(y_diff, grad_diff)
        positive_curvature = inner > jnp.finfo(inner.dtype).eps

        def no_update(_):
            return state

        def update(_):
            updated_y_diff_history = jtu.tree_map(
                lambda x, z: x.at[state.index_start % history_length].set(z),
                state.y_diff_history,
                y_diff,
            )
            updated_grad_diff_history = jtu.tree_map(
                lambda x, z: x.at[state.index_start % history_length].set(z),
                state.grad_diff_history,
                grad_diff,
            )

            # Here we gradually fill in the lower-triangular matrix of inner
            # products between the history of gradient differences and the current
            # difference in the optimisation variable `y`. This matrix has a zero
            # diagonal, it corresponds to `L_k` in the paper, where it is defined by
            # (2.18). At the start of the optimisation, parts of the lower triangle
            # will still be zero. We catch this before computing the Cholesky
            # factorisation by setting the diagonal to 1.0 in the affected rows, and
            # mapping the solution for these elements to a zero vector. (This
            # happens in the operator function.)
            y_diff_grad_diff_cross_inner = state.y_diff_grad_diff_cross_inner.at[
                state.index_start % history_length
            ].set(v_tree_dot(state.grad_diff_history, y_diff))
            y_diff_grad_diff_cross_inner = y_diff_grad_diff_cross_inner.at[
                :, state.index_start % history_length
            ].set(0)
            assert y_diff_grad_diff_cross_inner.shape == (
                history_length,
                history_length,
            )

            # Here we update the history of inner products in the circular buffer.
            y_diff_grad_diff_inner = state.y_diff_grad_diff_inner.at[
                state.index_start % history_length
            ].set(
                lx.internal.tree_dot(
                    jtu.tree_map(
                        lambda x: x[state.index_start % history_length],
                        updated_grad_diff_history,
                    ),
                    y_diff,
                )
            )
            assert y_diff_grad_diff_inner.shape == (history_length,)

            # Update the matrix of inner products of `y_diff` by updating one row
            # and one column per iteration. This matrix has nonzero elements for
            # rows up to k and columns up to k, where k is the current iteration
            # index and may be smaller than the history length. Cholesky
            # factorisation works because this matrix is initialised as an identity,
            # and elements > k are mapped to zero by setting the right-hand-side
            # appropriately in the operator function.
            cross_inner = v_tree_dot(
                updated_y_diff_history,
                jtu.tree_map(
                    lambda x: x[state.index_start % history_length],
                    updated_y_diff_history,
                ),
            )
            assert cross_inner.shape == (history_length,)
            y_diff_cross_inner = state.y_diff_cross_inner.at[
                state.index_start % history_length
            ].set(cross_inner)
            y_diff_cross_inner = y_diff_cross_inner.at[
                :, state.index_start % history_length
            ].set(cross_inner)
            assert y_diff_cross_inner.shape == (
                history_length,
                history_length,
            )

            updated_state = _LBFGSHessianUpdateState(
                index_start=state.index_start + 1,
                history_length=history_length,
                y_diff_history=updated_y_diff_history,
                grad_diff_history=updated_grad_diff_history,
                y_diff_grad_diff_cross_inner=y_diff_grad_diff_cross_inner,
                y_diff_grad_diff_inner=y_diff_grad_diff_inner,
                y_diff_cross_inner=y_diff_cross_inner,
                initial_scale=state.initial_scale,
            )

            return updated_state

        # Both branches return a ``_LBFGSHessianUpdateState`` whose only non-array field
        # is static, so the state is an ordinary pytree of arrays and ``lax.cond`` applies.
        new_state = jax.lax.cond(positive_curvature, update, no_update, None)

        # Set the scale ``hvp`` uses while there is no history to scale it; this is the
        # only place holding a gradient to set it from.
        empty_history = state.index_start == 0
        # on the first call ``grad_diff`` is the gradient itself, ``grad_prev`` being zero
        grad_norm = jnp.sqrt(lx.internal.tree_dot(grad_diff, grad_diff))
        # SciPy's ``stpmx`` cap on ``1 / ||d||``, but read off the dtype rather than its
        # f64 constant ``1e10``, as ``HessianSolverMixin`` reads its ``_delta``
        floor = jnp.sqrt(jnp.finfo(grad_norm.dtype).eps)

        def _hvp(v: Y, tag: HessianTag | None) -> Y:
            return self.hvp(new_state, params, v, tag)

        new_state = eqx.tree_at(
            lambda s: s.initial_scale,
            new_state,
            jnp.where(
                empty_history, jnp.maximum(grad_norm, floor), state.initial_scale
            ),
        )

        return _hvp, None, new_state

    def hvp(
        self,
        hessian_repr: _LBFGSHessianUpdateState[Y],
        params: Y,
        v: Y,
        hessian_tag: HessianTag | None,
    ) -> Y:
        r"""Apply :math:`B_k`, scaling it by hand while there is no history to scale it.

        With a pair stored, :func:`_lbfgs_hessian_operator_fn` scales itself by
        :math:`\gamma_k = y^\top y / s^\top y`. With none it falls back to
        :math:`\gamma_k = 1` over zero histories, so it returns ``v`` unchanged and
        :math:`B_0 = I`. The first step is then :math:`-\nabla f`, whose norm is the
        gradient's, and a backtracking line search with a bounded number of halvings
        cannot always shorten it enough: on a badly scaled problem the iterate overflows
        on iteration one.

        ``initial_scale`` is :math:`\|\nabla f\|`, making that step a unit-length one.
        This matches SciPy, whose L-BFGS-B [1]_ takes ``stp = 1.0`` at every iteration
        except the first, where it takes ``min(1 / ||d||, stpmx)`` with ``stpmx = 1e10``
        unconstrained (``lnsrlb``, ``scipy/optimize/src/lbfgsb.c``). That cap is an f64
        constant; ``initial_scale`` is floored at ``sqrt(eps)`` of the parameter dtype
        instead, so the bound it puts on the first step follows the working precision.

        References
        ----------
        .. [1] Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995).
            "A Limited Memory Algorithm for Bound Constrained Optimization."
            *SIAM Journal on Scientific Computing*, 16(5), 1190-1208.
            https://doi.org/10.1137/0916069
        """
        del params, hessian_tag
        history = hessian_repr
        scaled = _lbfgs_hessian_operator_fn(v, history)
        return jax.tree.map(
            lambda s, u: jnp.where(
                history.index_start == 0, history.initial_scale * u, s
            ),
            scaled,
            v,
        )


class NewtonCurvature(AbstractCurvature[Y, None, PyTree[Array]], Generic[Y]):
    """Newton curvature model.

    The Hessian is assembled afresh at every iteration, so nothing is carried between
    them and the state is ``None``. What the model multiplies by is the assembled
    tensor, one block per leading axis of ``hessian_tag.batch_axes`` when the tag
    reports block structure.
    """

    hessian_fn: Callable[..., PyTree[Array]]

    def init(self, params: Y, *args: Any) -> None:
        del params, args
        return None

    def update(
        self,
        state: None,
        params: Y,
        y_diff: Y,
        grad_diff: Y,
        *args: Any,
    ) -> tuple[Callable[[Y, HessianTag | None], Y], PyTree[Array], None]:
        del state, grad_diff, y_diff
        hessian_tensor = self.hessian_fn(params, *args)

        def _hvp(v: Y, tag: HessianTag | None) -> Y:
            return self.hvp(hessian_tensor, params, v, tag)

        return _hvp, hessian_tensor, None

    def hvp(
        self,
        hessian_repr: PyTree[Array],
        params: Y,
        v: Y,
        hessian_tag: HessianTag | None,
    ) -> Y:
        del params

        def apply_operator(H_b, v_b):
            return lx.PyTreeLinearOperator(H_b, jax.eval_shape(lambda: v_b)).mv(v_b)

        return map_blocks(apply_operator, hessian_repr, (v,), hessian_tag=hessian_tag)
