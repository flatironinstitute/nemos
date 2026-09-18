"""Proximal L-BFGS solver for composite objectives.

The outer loop and step acceptance come from
:class:`~nemos.solvers._line_search_mixins.LineSearchLoopMixin`, the composite subproblem
from :class:`~nemos.solvers._composite_mixins.CompositeQuadraticMixin`. What is local to
this module is the curvature model.

That model is derived from ``optimistix`` (Apache-2.0,
``optimistix/_solver/limited_memory_bfgs.py``) and modified here: only the direct-Hessian
branch is kept, ``update_hessian`` returns the ``lineax`` operator rather than a
``FunctionInfo`` and receives the curvature pair already differenced by the caller.
``_LBFGSHessianUpdateState`` is copied unchanged.

``optimistix.LBFGS`` is not reused whole because its loop is flat. One
``AbstractQuasiNewton.step`` is a single trial evaluation, and ``update_hessian`` sits in
the accepted branch of a ``filter_cond`` on the search's verdict, so backtracking is
unrolled into solver iterations. That pins the line search to ``AbstractSearch.step``, a
one-trial-per-call transition function; a bracketing search such as ``optax``'s zoom runs
its own inner loop and would have to be re-expressed as a state machine to fit it. The loop
here is nested instead, and the search is whichever ``optax`` transformation
``self._line_search`` holds. The curvature model is the only part that does not depend on
the loop shape, so it is the only part taken.
"""

from typing import TYPE_CHECKING, Any, Callable, Generic

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
from jaxtyping import Array, Float, PyTree, Scalar
from optimistix._misc import filter_cond
from optimistix._solver.limited_memory_bfgs import _lbfgs_hessian_operator_fn

from .. import tree_utils
from ..typing import Params
from ._composite_mixins import CompositeQuadraticMixin
from ._line_search_mixins import (
    DEFAULT_ATOL,
    DEFAULT_MAX_STEPS,
    DEFAULT_RTOL,
    LineSearchLoopMixin,
    LineSearchState,
    Y,
)

if TYPE_CHECKING:
    from ..regularizer import Regularizer


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
    """

    index_start: Scalar
    history_length: int
    y_diff_history: PyTree[Y]
    grad_diff_history: PyTree[Y]
    y_diff_grad_diff_cross_inner: Float[Array, " history_length history_length"]
    y_diff_grad_diff_inner: Float[Array, " history_length"]
    y_diff_cross_inner: Float[Array, " history_length history_length"]


class _LBFGSCurvature(eqx.Module, Generic[Y]):
    """What :class:`ProximalLBFGS` carries in ``LineSearchState.curvature``.

    ``hessian`` is the compact-form operator built from the history in ``update_state``,
    so it is the model this iteration's subproblem uses. ``grad_prev`` is the gradient at
    the previous accepted iterate, which pairs with ``state.y_diff`` to form ``(s, y)``.
    """

    hessian: lx.FunctionLinearOperator
    update_state: _LBFGSHessianUpdateState[Y]
    grad_prev: Y


def _batched_tree_zeros_like(y, batch_dimension):
    return jtu.tree_map(lambda y: jnp.zeros((batch_dimension, *y.shape)), y)


v_tree_dot = jax.vmap(lx.internal.tree_dot, in_axes=(0, None), out_axes=0)


class ProximalLBFGS(CompositeQuadraticMixin[Y], LineSearchLoopMixin[Y]):
    r"""Proximal L-BFGS solver for composite objectives.

    Minimizes :math:`f(\beta) + P(\beta)` with :math:`f` the smooth loss and :math:`P` a
    penalty reached through its proximal operator, as
    :class:`~nemos.solvers._newton.ProximalNewton` does, but with the assembled Hessian
    replaced by the limited-memory approximation :math:`B_k` of Byrd et al. [1]_,
    Algorithm 3.2. The subproblem and the composite line search are documented on
    :class:`~nemos.solvers._composite_mixins.CompositeQuadraticMixin`.

    :math:`B_k` is built from the last ``history_length`` pairs
    :math:`(s, y) = (\beta_k - \beta_{k-1},\, \nabla f_k - \nabla f_{k-1})`, keeping only
    those with :math:`s^\top y > \varepsilon` so that :math:`B_k \succ 0` and the
    subproblem stays convex. A rejected step leaves :math:`s = 0` and so leaves the
    history untouched. With an empty history :math:`B_0 = I`, making the first iteration a
    unit-stepsize proximal-gradient step whose scale the line search has to supply.

    The subproblem consumes :math:`B_k v` only and never factorizes :math:`B_k`, so this
    solver needs no Hessian from the model and carries no
    :class:`~nemos._hess.HessianTag`. A ``PopulationGLM`` fit therefore builds one
    curvature model over all neurons rather than one per block.

    Parameters
    ----------
    history_length :
        Number of curvature pairs retained. Larger values track curvature better and cost
        :math:`O(mpK)` memory and :math:`O(mpK)` work per operator application.
    line_search :
        Step-acceptance rule, ``"backtracking"`` (Armijo, the default) or ``"zoom"``.
        Armijo is the one the composite slope of Tseng & Yun [3]_ is derived for. Zoom
        additionally enforces the curvature condition, and does so by autodiffing the
        composite objective at each trial point, which compares a pointwise derivative
        against the Tseng & Yun surrogate: no result covers that pairing, so it is
        offered for experiment rather than recommended.
    tol, rtol :
        Absolute and relative tolerances of the outer Cauchy criterion on the accepted
        step; both are read, see
        :meth:`~nemos.solvers._composite_mixins.CompositeQuadraticMixin._converged`.
    inner_iter :
        Maximum FISTA steps on the subproblem. The subproblem applies the curvature model
        and touches no data, so these steps are cheap.
    inner_atol, inner_rtol :
        Tolerances for the subproblem, acting as the forcing sequence of the inexact
        proximal quasi-Newton method.

    References
    ----------
    .. [1] Byrd, R. H., Nocedal, J., & Schnabel, R. B. (1994).
        "Representations of quasi-Newton matrices and their use in limited memory
        methods." *Mathematical Programming*, 63(1), 129-156.
        https://doi.org/10.1007/BF01582063
    .. [2] Lee, J. D., Sun, Y., & Saunders, M. A. (2014).
        "Proximal Newton-type methods for minimizing composite functions."
        *SIAM Journal on Optimization*, 24(3), 1420-1443.
        https://doi.org/10.1137/130921428
    .. [3] Tseng, P., & Yun, S. (2009).
        "A coordinate gradient descent method for nonsmooth separable minimization."
        *Mathematical Programming*, 117(1-2), 387-423.
        https://doi.org/10.1007/s10107-007-0170-0
    """

    def __init__(
        self,
        unregularized_loss: Callable,
        regularizer: "Regularizer",
        regularizer_strength: float | None,
        has_aux: bool,
        init_params: Params | None = None,
        history_length: int = 10,
        jit: bool = True,
        maxiter: int = DEFAULT_MAX_STEPS,
        tol: float = DEFAULT_ATOL,
        rtol: float = DEFAULT_RTOL,
        line_search: str = "backtracking",
        inner_iter: int = 100,
        inner_atol: float = 1e-8,
        inner_rtol: float = 1e-8,
    ):
        if init_params is None:
            raise ValueError(
                "init_params is required for ProximalLBFGS solver. "
                "It is needed to determine the parameter structure for regularization."
            )

        # Read by ``_init_curvature`` when the state is built.
        self.history_length = history_length

        self._init_loop(
            unregularized_loss,
            regularizer,
            regularizer_strength,
            init_params,
            has_aux=has_aux,
            jit=jit,
            maxiter=maxiter,
            tol=tol,
            rtol=rtol,
            line_search=line_search,
        )
        self._init_composite(inner_iter, inner_atol, inner_rtol)

    def _init_curvature(self, init_params: Y) -> _LBFGSCurvature[Y]:
        """An empty history, whose compact form is the identity."""
        update_state = _LBFGSHessianUpdateState(
            index_start=jnp.array(0),
            history_length=self.history_length,
            y_diff_history=_batched_tree_zeros_like(init_params, self.history_length),
            grad_diff_history=_batched_tree_zeros_like(
                init_params, self.history_length
            ),
            y_diff_grad_diff_cross_inner=jnp.zeros(
                (self.history_length, self.history_length)
            ),
            y_diff_grad_diff_inner=jnp.ones(self.history_length),
            y_diff_cross_inner=jnp.eye(self.history_length),
        )
        hessian = lx.FunctionLinearOperator(
            lambda y: _lbfgs_hessian_operator_fn(y, update_state),
            jax.eval_shape(lambda: init_params),
            tags=lx.positive_semidefinite_tag,
        )
        return _LBFGSCurvature(
            hessian=hessian,
            update_state=update_state,
            grad_prev=jax.tree.map(jnp.zeros_like, init_params),
        )

    def _curvature(
        self, params: Y, state: LineSearchState[Y], grad: Y, *args: Any
    ) -> tuple[lx.FunctionLinearOperator, _LBFGSCurvature[Y]]:
        """Fold the move that produced ``params`` into the history, and use the result.

        ``state.y_diff`` is that move, so it pairs with the gradient difference over the
        same move. On the first iteration it is zero, the ``s^T y`` guard skips, and the
        identity model built by :meth:`_init_curvature` is used unchanged.
        """
        del params, args
        curvature = state.curvature
        hessian, update_state = self.update_hessian(
            state.y_diff,
            tree_utils.tree_sub(grad, curvature.grad_prev),
            curvature.hessian,
            curvature.update_state,
        )
        return hessian, _LBFGSCurvature(
            hessian=hessian, update_state=update_state, grad_prev=grad
        )

    def _hvp(self, grad: Y, H: lx.FunctionLinearOperator, d: Y) -> Y:
        """The compact form already knows how to multiply; there are no blocks to split."""
        del grad
        return H.mv(d)

    def update_hessian(
        self,
        y_diff: Y,
        grad_diff: Y,
        hessian: lx.FunctionLinearOperator,
        hessian_update_state: _LBFGSHessianUpdateState[Y],
    ) -> tuple[lx.FunctionLinearOperator, _LBFGSHessianUpdateState[Y]]:
        """Fold one curvature pair into the compact-form Hessian approximation.

        Takes the pair already differenced, unlike optimistix's version, which takes the
        two iterates and the two ``FunctionInfo``s: the caller holds ``y_diff`` anyway,
        for the Cauchy criterion.
        """
        # Update only if the inner product is positive, to maintain positive definiteness
        # of the Hessian approximation.
        inner = lx.internal.tree_dot(y_diff, grad_diff)
        positive_curvature = inner > jnp.finfo(inner.dtype).eps

        def no_update(args):
            *_, hessian, _ = args
            return eqx.filter(hessian, eqx.is_array), hessian_update_state

        def update(args):
            inner, grad_diff, y_diff, hessian, state = args
            updated_y_diff_history = jtu.tree_map(
                lambda x, z: x.at[state.index_start % self.history_length].set(z),
                state.y_diff_history,
                y_diff,
            )
            updated_grad_diff_history = jtu.tree_map(
                lambda x, z: x.at[state.index_start % self.history_length].set(z),
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
                state.index_start % self.history_length
            ].set(v_tree_dot(state.grad_diff_history, y_diff))
            y_diff_grad_diff_cross_inner = y_diff_grad_diff_cross_inner.at[
                :, state.index_start % self.history_length
            ].set(0)
            assert y_diff_grad_diff_cross_inner.shape == (
                self.history_length,
                self.history_length,
            )

            # Here we update the history of inner products in the circular buffer.
            y_diff_grad_diff_inner = state.y_diff_grad_diff_inner.at[
                state.index_start % self.history_length
            ].set(
                lx.internal.tree_dot(
                    jtu.tree_map(
                        lambda x: x[state.index_start % self.history_length],
                        updated_grad_diff_history,
                    ),
                    y_diff,
                )
            )
            assert y_diff_grad_diff_inner.shape == (self.history_length,)

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
                    lambda x: x[state.index_start % self.history_length],
                    updated_y_diff_history,
                ),
            )
            assert cross_inner.shape == (self.history_length,)
            y_diff_cross_inner = state.y_diff_cross_inner.at[
                state.index_start % self.history_length
            ].set(cross_inner)
            y_diff_cross_inner = y_diff_cross_inner.at[
                :, state.index_start % self.history_length
            ].set(cross_inner)
            assert y_diff_cross_inner.shape == (
                self.history_length,
                self.history_length,
            )

            updated_state = _LBFGSHessianUpdateState(
                index_start=state.index_start + 1,
                history_length=self.history_length,
                y_diff_history=updated_y_diff_history,
                grad_diff_history=updated_grad_diff_history,
                y_diff_grad_diff_cross_inner=y_diff_grad_diff_cross_inner,
                y_diff_grad_diff_inner=y_diff_grad_diff_inner,
                y_diff_cross_inner=y_diff_cross_inner,
            )

            hessian = lx.FunctionLinearOperator(
                lambda y: _lbfgs_hessian_operator_fn(y, updated_state),
                jax.eval_shape(lambda: y_diff),
                tags=lx.positive_semidefinite_tag,
            )
            # Only return the dynamic part of the operator, keep jaxpr across
            # iterations
            return eqx.filter(hessian, eqx.is_array), updated_state

        # We have a jaxpr in the FunctionLinearOperator, which needs to be filtered to
        # enable downstream checks for equality.
        static_hessian = eqx.filter(hessian, eqx.is_array, inverse=True)
        args = (inner, grad_diff, y_diff, hessian, hessian_update_state)
        new_dynamic_hessian, new_state = filter_cond(
            positive_curvature,
            update,
            no_update,
            args,
        )
        new_hessian = eqx.combine(static_hessian, new_dynamic_hessian)

        return new_hessian, new_state  # pyright: ignore

    @classmethod
    def get_accepted_arguments(cls) -> set[str]:
        return super().get_accepted_arguments() | {"history_length", "line_search"}
