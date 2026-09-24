"""Proximal L-BFGS solver for composite objectives.

The limited-memory curvature model is derived from ``optimistix`` (Apache-2.0,
``optimistix/_solver/limited_memory_bfgs.py``) and modified here: only the direct-Hessian
branch is kept, ``init_hessian`` and ``update_hessian`` return the ``lineax`` operator
rather than a ``FunctionInfo``, and ``update_hessian`` receives the curvature pair already
differenced by the caller. ``_LBFGSHessianUpdateState`` is copied unchanged.

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

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Optional,
    Tuple,
    TypeVar,
)

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
import optax
import optimistix as optx
from jaxtyping import Array, Bool, Float, PyTree, Scalar
from optimistix._misc import cauchy_termination, filter_cond
from optimistix._solver.limited_memory_bfgs import _lbfgs_hessian_operator_fn

from .. import tree_utils
from ..typing import Params, StepResult
from ._abstract_solver import OptimizationInfo
from ._fista import FISTA
from ._newton import DEFAULT_ATOL, DEFAULT_MAX_STEPS, DEFAULT_RTOL

if TYPE_CHECKING:
    from ..regularizer import Regularizer

# The parameter pytree. Both the state and the solver follow it, so a ``GLMParams`` fit
# and a ``PopulationGLM`` fit are distinct instantiations rather than ``Any``.
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
    """

    index_start: Scalar
    history_length: int
    y_diff_history: PyTree[Y]
    grad_diff_history: PyTree[Y]
    y_diff_grad_diff_cross_inner: Float[Array, " history_length history_length"]
    y_diff_grad_diff_inner: Float[Array, " history_length"]
    y_diff_cross_inner: Float[Array, " history_length history_length"]


class LBFGSState(eqx.Module, Generic[Y]):
    """State carried between outer iterations of :class:`ProximalLBFGS`."""

    grad_norm: Scalar
    stats: OptimizationInfo
    # Last accepted step, read twice per iteration: by the Cauchy criterion, and as the
    # ``s`` of the curvature pair. Initialized to zero, which the ``s^T y`` guard reads as
    # "no pair yet"; the first-iteration Cauchy hit is blocked by ``function_val`` being
    # NaN, since ``cauchy_termination`` requires both the y and the f test.
    y_diff: Y
    grad_prev: Y
    # Curvature model built from the history up to and including ``y_diff``, so it is the
    # one this iteration's subproblem uses.
    hessian: lx.FunctionLinearOperator
    hessian_update_state: _LBFGSHessianUpdateState[Y]
    # optax's line-search state, whose type is private to the chosen transformation.
    ls_state: Optional[Any] = None


def _batched_tree_zeros_like(y, batch_dimension):
    return jtu.tree_map(lambda y: jnp.zeros((batch_dimension, *y.shape)), y)


v_tree_dot = jax.vmap(lx.internal.tree_dot, in_axes=(0, None), out_axes=0)


class ProximalLBFGS(Generic[Y]):
    r"""Proximal L-BFGS solver for composite objectives.

    Minimizes :math:`f(\beta) + P(\beta)` with :math:`f` the smooth loss and :math:`P` a
    penalty reached through its proximal operator, as
    :class:`~nemos.solvers._newton.ProximalNewton` does, but with the assembled Hessian
    replaced by the limited-memory approximation :math:`B_k` of Byrd et al. [1]_,
    Algorithm 3.2. Each iteration solves

    .. math::
        \min_d \; \nabla f^\top d + \tfrac{1}{2} d^\top B_k d + P(\beta + d)

    with :class:`~nemos.solvers._fista.FISTA`, then backtracks on the composite objective;
    see [2]_ for the method and [3]_ for the composite Armijo slope.

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
    tol, rtol :
        Absolute and relative tolerances of the outer Cauchy criterion on the accepted
        step; both are read, see :meth:`_converged`.
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

    _proximal: ClassVar[bool] = True

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
        inner_iter: int = 100,
        inner_atol: float = 1e-8,
        inner_rtol: float = 1e-8,
    ):

        if init_params is None:
            raise ValueError(
                "init_params is required for ProximalLBFGS solver. "
                "It is needed to determine the parameter structure for regularization."
            )

        self.has_aux = has_aux
        self.jit = jit
        self.maxiter = maxiter
        self.tol = tol
        self.rtol = rtol
        self.history_length = history_length

        # the penalty alone, for the composite line search. self.fun is the smooth
        # loss here, so the composite objective is self.fun + self._penalty, which is
        # exactly what ``regularizer.penalized_loss`` builds from the same accessor.
        self._penalty = regularizer.penalty_fn(
            params=init_params, strength=regularizer_strength
        )

        self.inner_iter = inner_iter
        self.inner_atol = inner_atol
        self.inner_rtol = inner_rtol

        self.prox = regularizer.get_proximal_operator(
            params=init_params, strength=regularizer_strength
        )

        # The subproblem is solved for the new parameters, so the prox is the
        # regularizer's own and the solver does not depend on the current iterate:
        # build it once rather than per outer iteration.
        self._inner_solver = FISTA(
            atol=inner_atol,
            rtol=inner_rtol,
            norm=lx.internal.two_norm,
            prox=self.prox,
            while_loop_kind="lax",
        )
        # split scalar vs aux
        if has_aux:
            self.fun_with_aux = unregularized_loss
            self.fun = lambda p, *a: unregularized_loss(p, *a)[0]
        else:
            self.fun = unregularized_loss
            self.fun_with_aux = lambda p, *a: (unregularized_loss(p, *a), None)

        self._line_search = optax.scale_by_backtracking_linesearch(
            max_backtracking_steps=30
        )

        # Cache
        self._gradient: Callable | None = None

    def init_hessian(self, y: Y) -> LBFGSState[Y]:
        """Build the identity curvature model and the state that will accumulate it."""
        hess_state = _LBFGSHessianUpdateState(
            index_start=jnp.array(0),
            history_length=self.history_length,
            y_diff_history=_batched_tree_zeros_like(y, self.history_length),
            grad_diff_history=_batched_tree_zeros_like(y, self.history_length),
            y_diff_grad_diff_cross_inner=jnp.zeros(
                (self.history_length, self.history_length)
            ),
            y_diff_grad_diff_inner=jnp.ones(self.history_length),
            y_diff_cross_inner=jnp.eye(self.history_length),
        )
        hessian = lx.FunctionLinearOperator(
            lambda y: _lbfgs_hessian_operator_fn(y, hess_state),
            jax.eval_shape(lambda: y),
            tags=lx.positive_semidefinite_tag,
        )
        state = LBFGSState(
            ls_state=self._line_search.init(y),
            grad_norm=jnp.array(jnp.inf),
            stats=OptimizationInfo(
                function_val=jnp.array(jnp.nan),
                num_steps=jnp.array(0),
                converged=jnp.array(False),
                reached_max_steps=jnp.array(False),
            ),
            y_diff=jax.tree.map(jnp.zeros_like, y),
            grad_prev=jax.tree.map(jnp.zeros_like, y),
            hessian=hessian,
            hessian_update_state=hess_state,
        )
        return state  # pyright: ignore

    def update_hessian(
        self,
        grad: Y,
        params: Y,
        state: LBFGSState[Y],
        *args: Any,
    ) -> tuple[lx.FunctionLinearOperator, LBFGSState[Y]]:
        """Fold one curvature pair into the compact-form Hessian approximation.

        The pair is read off the state rather than passed in, unlike optimistix's
        version, which takes the two iterates and the two ``FunctionInfo``s:
        ``state.y_diff`` is the move that produced ``params``, so it is already the
        ``s`` of the pair, and ``state.grad_prev`` differences against it.
        """
        del params, args
        y_diff = state.y_diff
        grad_diff = tree_utils.tree_sub(grad, state.grad_prev)
        hessian = state.hessian
        hessian_update_state = state.hessian_update_state

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
        new_dynamic_hessian, new_update_state = filter_cond(
            positive_curvature,
            update,
            no_update,
            args,
        )
        new_hessian = eqx.combine(static_hessian, new_dynamic_hessian)

        # ``grad_prev`` is written here rather than in the loop: this is the point at
        # which the gradient has been folded into the history, so it is what the next
        # iteration must difference against.
        return new_hessian, eqx.tree_at(
            lambda s: (s.hessian, s.hessian_update_state, s.grad_prev),
            state,
            (new_hessian, new_update_state, grad),
        )

    def _build_cache(self) -> None:
        if self._gradient is None:
            self._gradient = jax.value_and_grad(
                self.fun_with_aux,
                has_aux=True,
            )

    def init_state(self, init_params: Y, *args: Any) -> LBFGSState[Y]:
        self._build_cache()
        return self.init_hessian(init_params)

    def _lbfgs_direction(
        self, grad: Y, H: lx.FunctionLinearOperator, params: Y, state: LBFGSState
    ) -> Tuple[Y, LBFGSState]:
        r"""Minimize :math:`\nabla f^\top (z - \beta) + \frac12 (z - \beta)^\top H (z - \beta) + P(z)`.

        Solving for the new parameters :math:`z` rather than the step keeps the penalty
        where it is defined, so ``self.prox`` applies unchanged and the inner solver does
        not depend on the current iterate.

        The proximal operator carries metadata defined on the whole parameter tree --
        ``GroupLasso``'s mask, or a per-feature strength -- so the subproblem is solved
        on the full tree and only the Hessian-vector product is split per block. That
        keeps every regularizer usable without slicing each one's penalty metadata.
        """

        def quadratic(z, _):
            step = tree_utils.tree_sub(z, params)
            hvp = H.mv(step)
            return lx.internal.tree_dot(grad, step) + 0.5 * lx.internal.tree_dot(
                step, hvp
            )

        new_params = optx.minimise(
            quadratic,
            self._inner_solver,
            y0=params,
            max_steps=self.inner_iter,
            throw=False,
        ).value
        # ``_apply_or_reject`` scales and adds the result, so return the step
        return tree_utils.tree_sub(new_params, params), state

    def _apply_or_reject(
        self,
        params: Y,
        step: Y,
        grad: Y,
        state: LBFGSState[Y],
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

    def _converged(
        self, params: Y, state: LBFGSState[Y], grad: Y, fval: Scalar
    ) -> Bool[Array, ""]:
        """Cauchy criterion on the accepted step, as :class:`~nemos.solvers._fista.FISTA` uses.

        A gradient-based test is unusable here: this solver differentiates the smooth
        part only, so its gradient does not vanish at the optimum of a composite
        objective, and any residual built from it inherits the curvature scale -- on
        badly conditioned data it never falls below ``tol`` even once the iterate has
        stopped moving.
        """
        return cauchy_termination(
            self.rtol,
            self.tol,
            lx.internal.two_norm,
            params,
            state.y_diff,
            fval,
            fval - state.stats.function_val,
        )

    def _line_search_inputs(
        self, params: Y, step: Y, grad: Y, fval: Scalar, *args: Any
    ) -> tuple[Scalar, Y, Callable[[Y], Scalar]]:
        r"""Feed the composite objective and its slope to the inherited line search.

        Tseng & Yun (2009) require the sufficient-decrease slope of a composite
        objective to be

        .. math::
            \Delta = \nabla f^\top d + P(\beta + d) - P(\beta),

        the :math:`P` difference being what makes :math:`\Delta < 0` a descent
        certificate when :math:`F` is nonsmooth. Since the search only ever forms
        ``vdot(step, slope)``, adding the penalty difference along ``step`` reproduces
        :math:`\Delta` exactly, and the stock Armijo search then applies unchanged.
        """
        penalty = self._penalty(params)
        penalty_diff = self._penalty(tree_utils.tree_add(params, step)) - penalty
        sq_norm = lx.internal.tree_dot(step, step)
        slope = tree_utils.tree_add_scalar_mul(
            grad, jnp.where(sq_norm > 0.0, penalty_diff / sq_norm, 0.0), step
        )
        return (
            fval + penalty,
            slope,
            lambda p: self.fun(p, *args) + self._penalty(p),
        )

    @classmethod
    def get_accepted_arguments(cls) -> set[str]:
        return {
            "maxiter",
            "tol",
            "rtol",
            "jit",
            "history_length",
            "inner_iter",
            "inner_atol",
            "inner_rtol",
        }

    def _get_optim_info(self, state: LBFGSState[Y], **kwargs) -> OptimizationInfo:
        return state.stats

    def update(
        self,
        params: Y,
        state: LBFGSState[Y],
        *args: Any,
    ) -> StepResult:

        (fval, aux), grad = self._gradient(params, *args)

        gnorm = jnp.sqrt(lx.internal.tree_dot(grad, grad))
        converged = self._converged(params, state, grad, fval)
        static = eqx.filter(state, eqx.is_array, inverse=True)

        def step(_):
            new_hessian, new_state = self.update_hessian(
                grad,
                params,
                state,
                *args,
            )
            step, new_state = self._lbfgs_direction(
                grad, new_hessian, params, new_state
            )

            new_params, new_ls_state = self._apply_or_reject(
                params,
                step,
                grad,
                state,
                fval,
                *args,
            )

            return (
                new_params,
                eqx.filter(
                    eqx.tree_at(lambda x: x.ls_state, new_state, new_ls_state),
                    eqx.is_array,
                ),
            )

        def no_step(_):
            return (
                params,
                eqx.filter(state, eqx.is_array),
            )

        new_params, new_state = filter_cond(
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
        new_state = eqx.tree_at(
            lambda s: (s.grad_norm, s.stats, s.y_diff),
            new_state,
            (
                gnorm,
                OptimizationInfo(
                    function_val=fval,
                    num_steps=new_iter,
                    converged=converged,
                    reached_max_steps=new_iter >= self.maxiter,
                ),
                tree_utils.tree_sub(new_params, params),
            ),
        )
        new_state = eqx.combine(new_state, static)
        return new_params, new_state, aux

    def run(
        self,
        init_params: Y,
        *args: Any,
    ) -> StepResult:
        state = self.init_state(init_params, *args)
        # The curvature model holds a jaxpr, which cannot cross a loop carry. Only the
        # arrays are carried; the jaxpr is closed over and restored on the way out.
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
