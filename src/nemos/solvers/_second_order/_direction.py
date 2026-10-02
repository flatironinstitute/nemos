from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Generic, Literal, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree
from lineax import AbstractLinearSolver

from ... import tree_utils
from ..._hess import HessianTag, MatrixProperty, MatrixStructure
from ...typing import Params
from .._fista import FISTA
from ._typing import D, Y

if TYPE_CHECKING:
    from ._typing import HvpFn
from ._utils import map_blocks


class AbstractDirection(eqx.Module, ABC, Generic[Y, D]):
    """Turn a curvature model and a gradient into a step.

    A direction reads the curvature in whichever of the two forms it can use -- the
    Hessian-vector product, or the assembled tensor -- and touches no data, so it takes
    no ``*args``.
    """

    @abstractmethod
    def update(
        self,
        params: Y,
        grad: Y,
        hvp_fn: HvpFn,
        hessian_tensor: PyTree[Array] | None,
        direction_state: D,
    ) -> Tuple[Y, D]: ...

    @abstractmethod
    def init(self, params: Y) -> D: ...

    @property
    def prox(self) -> Callable | None:
        """The proximal operator this direction applies, or ``None`` if it applies none.

        Declared here rather than discovered, so a caller can ask any direction without
        knowing which one it holds.
        """
        return None


def _solve_shifted_system(
    operator: lx.AbstractLinearOperator,
    grad: Y,
    shift: Array | float | None,
    *,
    solver: AbstractLinearSolver,
    tags: object,
    throw: bool = True,
) -> lx.Solution:
    r"""Solve a shifted Newton system.

    Solves

    .. math::
        (H + \tau I)d = -g,

    while preserving the PyTree structure of the Hessian operator.

    Parameters
    ----------
    operator :
        Linear operator representing the Hessian.
    grad :
        Gradient on the right-hand side of the Newton system.
    shift :
        Scalar identity shift.
    solver :
        Lineax linear solver.
    tags :
        Lineax tags describing the shifted operator.
    throw :
        Whether Lineax should raise an exception when the solve fails.

    Returns
    -------
    :
        The result returned by :func:`lineax.linear_solve`.
    """
    if shift is not None:
        identity = lx.IdentityLinearOperator(operator.in_structure())
        operator = operator + shift * identity

    operator = lx.TaggedLinearOperator(
        operator,
        tags=tags,
    )

    solution = lx.linear_solve(
        operator,
        jax.tree.map(jnp.negative, grad),
        solver=solver,
        throw=False,
    )

    if throw:
        failed = (
            solution.result != lx.RESULTS.successful
        ) | ~tree_utils.tree_all_finite(solution.value)

        checked_value = eqx.error_if(
            solution.value,
            failed,
            "Cholesky solve failed; the Hessian may not be positive definite. "
            "Try using the 'eigh' or 'identity_shift' solver instead.",
        )
        solution = eqx.tree_at(
            lambda result: result.value,
            solution,
            checked_value,
        )

    return solution


class ProxQuadraticDirection(AbstractDirection[Y, None], Generic[Y]):
    _inner_solver: FISTA
    _inner_iter: int
    hessian_tag: HessianTag | None = eqx.field(static=True)

    def update(
        self,
        params: Y,
        grad: Y,
        hvp_fn: HvpFn,
        hessian_tensor: PyTree[Array] | None,
        direction_state: None,
    ) -> Tuple[Y, None]:
        del hessian_tensor

        def quadratic(z, _):
            step = tree_utils.tree_sub(z, params)
            hvp = hvp_fn(step, self.hessian_tag)
            return lx.internal.tree_dot(grad, step) + 0.5 * lx.internal.tree_dot(
                step, hvp
            )

        new_params = optx.minimise(
            quadratic,
            self._inner_solver,
            y0=params,
            max_steps=self._inner_iter,
            throw=False,
        ).value
        return tree_utils.tree_sub(new_params, params), direction_state

    def init(self, params: Y) -> None:
        del params
        return None

    @property
    def prox(self) -> Callable:
        """The subproblem solver's proximal operator, which is the only copy of it."""
        return self._inner_solver.prox


LinearSolverTag = Literal["auto", "cholesky", "eigh", "identity_shift"]
ResolvedLinearSolverTag = Literal["cholesky", "eigh", "identity_shift"]

VALID_SOLVERS = {"auto", "cholesky", "eigh", "identity_shift"}

POSITIVE_PROPERTIES = {
    MatrixProperty.POSITIVE_DEFINITE,
    MatrixProperty.POSITIVE_SEMI_DEFINITE,
}
SYMMETRIC_PROPERTIES = {
    MatrixProperty.SYMMETRIC,
    MatrixProperty.NEGATIVE_DEFINITE,
    MatrixProperty.NEGATIVE_SEMI_DEFINITE,
}


def validate_linear_solver(linear_solver: str) -> None:
    """Reject an unknown strategy before a caller pays for anything else."""
    if linear_solver not in VALID_SOLVERS:
        raise ValueError(
            f"Unknown linear solver {linear_solver!r}. "
            f"Expected one of {sorted(VALID_SOLVERS)}."
        )


class LinearSolveDirection(AbstractDirection[Y, Array], Generic[Y]):
    """Solve :math:`Hd = -g`, one block at a time when the tag reports block structure.

    ``direction_state`` is the identity shift the previous iteration accepted, which
    seeds the ladder; it is a scalar, or one scalar per block.
    """

    # ``None`` for the strategies that do not reach Lineax: "eigh" decomposes by hand,
    # "identity_shift" builds its own Cholesky per attempt.
    linear_solver: AbstractLinearSolver | None
    # the eigenvalue floor, an array whose dtype follows the parameters for "eigh"
    delta: float | Array
    resolved_linear_solver: Literal["cholesky", "eigh", "identity_shift"] | None
    shift_fn: Callable[[lx.AbstractLinearOperator], Array | float | None] | None
    identity_shift_beta: float
    identity_shift_max_steps: int
    hessian_tag: HessianTag | None = eqx.field(static=True)

    @classmethod
    def from_tag(  # noqa: C901
        cls,
        init_params: Params,
        hessian_tag: HessianTag,
        linear_solver: LinearSolverTag,
        identity_shift_beta: float,
        identity_shift_max_steps: int,
    ) -> "LinearSolveDirection":
        """Resolve the solution strategy from the tag and the user's request.

        Every field below follows from the pair: what the tag claims about the matrix,
        and which strategy the caller asked for. ``"auto"`` reads the claim, anything
        else is taken as given and warned about when the claim disagrees.
        """
        matrix_property = hessian_tag.property
        if matrix_property not in POSITIVE_PROPERTIES | SYMMETRIC_PROPERTIES:
            raise ValueError(
                f"Hessian has unsupported matrix property: {matrix_property}"
            )
        validate_linear_solver(linear_solver)

        if linear_solver == "auto":
            resolved: ResolvedLinearSolverTag = (
                "cholesky" if matrix_property in POSITIVE_PROPERTIES else "eigh"
            )
        else:
            resolved = linear_solver

        if resolved == "cholesky" and matrix_property not in POSITIVE_PROPERTIES:
            warnings.warn(
                "linear_solver='cholesky' was requested, but the Hessian tag "
                f"reports {matrix_property}. Cholesky generally requires a "
                "positive-definite or positive-semidefinite Hessian. Proceeding "
                "with Cholesky as requested; the solve may fail.",
                RuntimeWarning,
                stacklevel=2,
            )

        delta: float | Array = 0.0
        if resolved == "cholesky":
            solver = lx.Cholesky()
            # Continue using the tag to distinguish the PSD and PD branches
            if matrix_property is MatrixProperty.POSITIVE_SEMI_DEFINITE:

                def shift_fn(operator):
                    diagonal = lx.diagonal(operator)
                    return (
                        diagonal.size * jnp.finfo(diagonal.dtype).eps * diagonal.max()
                    )

            else:

                def shift_fn(_):
                    return None

        elif resolved == "eigh":
            solver = None

            def shift_fn(_):
                return 0.0

            dtype = jnp.result_type(*jax.tree_util.tree_leaves(init_params))
            delta = jnp.sqrt(jnp.finfo(dtype).eps)
        else:
            # Nocedal and Wright Algorithm 3.3: add tau * I until Cholesky succeeds.
            # The retry loop itself runs in ``_solve``.
            solver = None

            def shift_fn(_):
                return 0.0

        return cls(
            linear_solver=solver,
            delta=delta,
            resolved_linear_solver=resolved,
            shift_fn=shift_fn,
            identity_shift_beta=identity_shift_beta,
            identity_shift_max_steps=identity_shift_max_steps,
            hessian_tag=hessian_tag,
        )

    def _solve(
        self, H: PyTree[Array], grad: Y, previous_shift: Array
    ) -> Tuple[Y, Array]:

        operator = lx.PyTreeLinearOperator(H, jax.eval_shape(lambda: grad))

        # Modify the Hessian eigenvalues, then solve the resulting positive-definite system
        # This handles symmetric indefinite Hessians in one decomposition
        # Nocedal and Wright Algorithm equation (3.49)
        if self.resolved_linear_solver == "eigh":
            g_flat, unravel = ravel_pytree(grad)
            H_dense = operator.as_matrix()
            eigvals, Q = jnp.linalg.eigh(H_dense)
            lam_mod = jnp.maximum(jnp.abs(eigvals), self.delta)
            direction = Q @ ((Q.T @ (-g_flat)) / lam_mod)
            return unravel(direction), previous_shift

        # Add tau * I and increase tau until Cholesky succeeds
        # Nocedal and Wright Algorithm 3.3
        elif self.resolved_linear_solver == "identity_shift":
            diag = lx.diagonal(operator)
            dtype = diag.dtype

            # Beta scales with the matrix so the shift ladder is scale-equivariant;
            # N&W give 1e-3 as the typical magnitude, here relative to the diagonal
            eps = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
            beta = jnp.maximum(
                jnp.asarray(self.identity_shift_beta, dtype=dtype)
                * jnp.max(jnp.abs(diag)),
                eps,
            )

            min_diag = jnp.min(diag)
            # N&W seed the ladder from the shift the last iteration accepted.
            # Decaying it by the factor the ladder climbs by costs at most one extra
            # factorization when the required shift is stable, and lets tau fall back
            # to zero once the iterates reach a region where the unshifted Cholesky succeeds.
            warm_start = jnp.where(
                previous_shift > beta,
                jnp.asarray(0.1, dtype=dtype) * previous_shift,  # scale it down of 1/10
                jnp.zeros((), dtype=dtype),
            )
            tau0 = jnp.maximum(
                jnp.where(min_diag > 0, jnp.zeros((), dtype=dtype), -min_diag + beta),
                warm_start,
            )

            cholesky = lx.Cholesky()

            def _solve(tau):
                result = _solve_shifted_system(
                    operator,
                    grad,
                    tau,
                    solver=cholesky,
                    tags=lx.positive_semidefinite_tag,
                    throw=False,
                )
                failed = (
                    result.result != lx.RESULTS.successful
                ) | ~tree_utils.tree_all_finite(result.value)
                return result.value, failed

            direction0, failed0 = _solve(tau0)

            def cond(carry):
                iteration, _, _, failed = carry
                return failed & (iteration < self.identity_shift_max_steps)

            def body(carry):
                iteration, tau, _, _ = carry
                new_tau = jnp.maximum(jnp.asarray(10.0, dtype=dtype) * tau, beta)
                direction, failed = _solve(new_tau)
                return iteration + 1, new_tau, direction, failed

            _, tau, direction, failed = eqx.internal.while_loop(
                cond,
                body,
                (
                    jnp.asarray(0),
                    tau0,
                    direction0,
                    failed0,
                ),
                kind="lax",
            )

            accepted_shift = jnp.where(failed, previous_shift, tau)
            return direction, accepted_shift

        # Catch use before init_state has resolved the requested strategy.
        elif self.resolved_linear_solver != "cholesky":
            raise RuntimeError(
                "The solver has not been resolved. Call init_state before update."
            )

        # Solve a positive Hessian with Lineax Cholesky.
        # For a PSD tag, _shift_fn adds the small numerical shift selected by the tag
        direction = _solve_shifted_system(
            operator,
            grad,
            self.shift_fn(operator),
            solver=self.linear_solver,
            tags=lx.positive_semidefinite_tag,  # it has to be otherwise it'll raise
        ).value

        return direction, previous_shift

    def update(
        self,
        params: Y,
        grad: Y,
        hvp_fn: HvpFn,
        hessian_tensor: PyTree[Array],
        direction_state: Array,
    ) -> Tuple[Y, Array]:
        del hvp_fn
        direction, accepted_shift = map_blocks(
            self._solve,
            hessian_tensor,
            (grad,),
            hessian_tag=self.hessian_tag,
            block_state=direction_state,
        )
        return direction, accepted_shift

    def init(self, params: Y) -> Array:
        dtype = jnp.result_type(*jax.tree_util.tree_leaves(params))
        zero = jnp.zeros((), dtype=dtype)
        tag = self.hessian_tag
        if tag is not None and tag.structure is MatrixStructure.BLOCK_DIAGONAL:
            return jax.vmap(
                lambda _: zero,
                in_axes=(tag.batch_axes,),
                out_axes=0,
            )(params)
        return zero
