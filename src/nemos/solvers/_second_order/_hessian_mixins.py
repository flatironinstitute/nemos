"""Mixin providing the curvature machinery for second-order solvers."""

import warnings
from typing import Any, Callable, ClassVar, Literal, Optional

import jax
import jax.numpy as jnp
import lineax as lx

from ... import tree_utils
from ..._hess import (
    HessianTag,
    MatrixProperty,
    MatrixStructure,
    combine_hessian_tags,
    mask_claim_none,
)
from . import NewtonCurvature
from ._direction import LinearSolveDirection

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


class HessianMixin:
    """
    Receive and exploit the model's analytic Hessian.

    ``BaseRegressor`` routes the Hessian to any solver carrying this mixin, so
    curvature stays out of ``AbstractSolver``: a first-order solver has no use for a
    Hessian, a Hessian tag, or a linear solver, and should not inherit them.
    This mirrors ``StochasticSolverMixin``, which holds the stochastic machinery for
    the solvers that support it.

    Subclasses are expected to call ``_init_hessian`` from their ``__init__``, which
    resolves the tag and the Hessian in one pass; everything downstream of it -- the
    directions, which hold the tag -- can then be built in ``__init__`` as well.
    """

    # Declares the capability to ``BaseRegressor``, mirroring ``_supports_stochastic``.
    _uses_hessian: ClassVar[bool] = True

    # Whether the penalty is modelled by the quadratic. False for solvers that reach
    # the penalty through a proximal operator instead, which must not also add its
    # curvature here -- ``prox_elastic_net`` already applies the L2 rescale.
    _proximal: ClassVar[bool] = False

    def _init_hessian(
        self,
        regularizer,
        regularizer_strength,
        init_params,
        hess_fn: Callable | None = None,
        hessian_tag: HessianTag | None = None,
        reg_tag: HessianTag | None = None,
        property_override: Optional[type] = None,
    ) -> HessianTag:
        """Build the curvature model and return the tag describing its matrix.

        The invariant, whichever branch runs: ``curvature.hessian_fn`` is the Hessian of
        the smooth objective the solver differentiates, and the returned tag describes
        that same matrix. The caller hands the tag to the direction, which is where it
        comes to rest.
        """
        # A model with no analytic Hessian sends neither a Hessian nor a tag, so the
        # solver falls back to differentiating its own objective, which the unstructured
        # symmetric tag describes.
        default_tag = HessianTag(
            structure=MatrixStructure.FULL,
            property=MatrixProperty.SYMMETRIC,
            flat_on=mask_claim_none(init_params),
            definite_on=mask_claim_none(init_params),
        )
        autodiff_hess_fn: Callable = jax.hessian(self.fun)

        if self._proximal:
            # NeMoS splits the *whole* penalty into the proximal operator -- so much so
            # that ``prox_elastic_net`` rescales for its own L2 term -- leaving the smooth
            # objective equal to the unregularized loss. Its curvature is therefore the
            # model's alone: the penalty contributes no curvature, no structure, and no
            # ``property_override``, the last because that override describes the
            # *penalized* Hessian, which a proximal solver does not hold. Promoting the
            # tag here would claim definiteness for a matrix that is merely positive
            # semidefinite.
            reg_tag = property_override = None
        else:
            hess_fn = self._penalize_hessian(
                hess_fn,
                hessian_tag,
                regularizer,
                regularizer_strength,
                init_params,
            )

        tag = (
            hessian_tag
            if reg_tag is None
            else combine_hessian_tags(hessian_tag, reg_tag)
        )
        if property_override is not None and tag is not None:
            tag = HessianTag(
                tag.structure,
                property_override,
                flat_on=tag.flat_on,
                definite_on=tag.definite_on,
                batch_axes=tag.batch_axes,
            )

        # ``hess_fn`` is None when the model has no analytic Hessian to offer, and the
        # autodiff one already stands in for it. ``combine_hessian_tags`` likewise
        # returns None as soon as one of its arguments is None, so keep the unstructured
        # symmetric default rather than dropping back to None.
        self.curvature = NewtonCurvature(
            autodiff_hess_fn if hess_fn is None else hess_fn
        )
        return default_tag if tag is None else tag

    def _penalize_hessian(
        self, hess_fn, model_tag, regularizer, regularizer_strength, init_params
    ):
        """Add the regularizer's penalty Hessian to the model's likelihood Hessian.

        Models supply the second derivative of the likelihood alone. Adding the penalty's
        is valid because ``Regularizer.penalized_loss`` returns ``loss + penalty``, and the
        second derivative of a sum is the sum of the second derivatives.

        ``None`` passes through: without a model-supplied Hessian the solver autodiffs
        the penalized loss, which already carries the penalty.

        The batching comes from ``model_tag`` rather than the combined tag, because whether
        the Hessian is assembled one block per neuron is a property of the model.

        Only reached for a non-proximal solver: ``setup_hessian`` decides whether there is
        a penalty to add, so this method always adds one.
        """
        if hess_fn is None:
            return None

        batch_axes = (
            model_tag.batch_axes
            if model_tag is not None
            and model_tag.structure is MatrixStructure.BLOCK_DIAGONAL
            else None
        )
        penalty_hess_fn = regularizer._get_hess_fn(
            init_params, regularizer_strength, batch_axes=batch_axes
        )
        if penalty_hess_fn is None:
            # the regularizer declares no curvature, so the likelihood term is the whole
            return hess_fn

        def penalized_hessian(params, *args):
            return tree_utils.tree_add(hess_fn(params, *args), penalty_hess_fn(params))

        return penalized_hessian

    def _block_apply(self, fn, grad, H, other, block_state=None) -> Any:
        """Apply ``fn(grad, H, other)`` once per Hessian block.

        The one place that reads ``_hess_tag`` for block structure, shared by the Newton
        solve and by any subclass' Hessian-vector product.
        """
        if self.direction.hessian_tag.structure is MatrixStructure.BLOCK_DIAGONAL:
            axes = self.direction.hessian_tag.batch_axes
            if block_state is not None:
                return jax.vmap(
                    fn,
                    in_axes=(axes, 0, axes, 0),
                    out_axes=(axes, 0),
                )(grad, H, other, block_state)
            return jax.vmap(
                fn,
                in_axes=(axes, 0, axes),
                out_axes=axes,
            )(grad, H, other)
        if block_state is not None:
            return fn(grad, H, other, block_state)
        return fn(grad, H, other)


class HessianSolverMixin:
    """Resolve and hold the strategy for solving :math:`Hd = -g`.

    Reads ``_hess_tag`` off :class:`HessianMixin`, which a host must carry as well, and
    exposes nothing back to it: a solver that only multiplies by its curvature model
    inherits :class:`HessianMixin` alone and gets none of the state below.
    """

    def _init_block_state(self, params, value: jax.Array) -> jax.Array:
        """Broadcast a scalar to one value per Hessian block."""
        if self.direction.hessian_tag.structure is MatrixStructure.BLOCK_DIAGONAL:
            return jax.vmap(
                lambda _: value,
                in_axes=(self.direction.hessian_tag.batch_axes,),
                out_axes=0,
            )(params)
        return value

    def _init_solver(
        self,
        linear_solver: LinearSolverTag = "auto",
    ):
        if linear_solver not in VALID_SOLVERS:
            raise ValueError(
                f"Unknown linear solver {linear_solver!r}. "
                f"Expected one of {sorted(VALID_SOLVERS)}."
            )
        self.linear_solver: LinearSolverTag = linear_solver

    def _build_linear_solve_direction(  # noqa: C901
        self,
        init_params,
        hessian_tag: HessianTag,
        identity_shift_beta: float,
        identity_shift_max_steps: int,
    ) -> LinearSolveDirection:
        """Resolve the Hessian solution strategy from the tag and user request."""
        matrix_property = hessian_tag.property

        if matrix_property not in POSITIVE_PROPERTIES | SYMMETRIC_PROPERTIES:
            raise ValueError(
                f"Hessian has unsupported matrix property: {matrix_property}"
            )

        requested = self.linear_solver

        # Revalidate in case a caller changed the public field after construction
        if requested not in VALID_SOLVERS:
            raise ValueError(
                f"Unknown linear solver {requested!r}. "
                f"Expected one of {sorted(VALID_SOLVERS)}."
            )

        if requested == "auto":
            resolved: ResolvedLinearSolverTag = (
                "cholesky" if matrix_property in POSITIVE_PROPERTIES else "eigh"
            )
        else:
            resolved = requested

        if resolved == "cholesky" and matrix_property not in POSITIVE_PROPERTIES:
            warnings.warn(
                "linear_solver='cholesky' was requested, but the Hessian tag "
                f"reports {matrix_property}. Cholesky generally requires a "
                "positive-definite or positive-semidefinite Hessian. Proceeding "
                "with Cholesky as requested; the solve may fail.",
                RuntimeWarning,
                stacklevel=2,
            )
        _delta = 0.0
        if resolved == "cholesky":
            _linear_solver = lx.Cholesky()
            # Continue using the tag to distinguish the PSD and PD branches
            if matrix_property is MatrixProperty.POSITIVE_SEMI_DEFINITE:

                def _compute_shift(operator):
                    diagonal = lx.diagonal(operator)
                    return (
                        diagonal.size * jnp.finfo(diagonal.dtype).eps * diagonal.max()
                    )

                _shift_fn = _compute_shift
            else:

                def _shift_fn(_):
                    return None

        elif resolved == "eigh":
            _linear_solver = None

            def _shift_fn(_):
                return 0.0

            dtype = jnp.result_type(*jax.tree_util.tree_leaves(init_params))
            _delta = jnp.sqrt(jnp.finfo(dtype).eps)
        else:
            # Nocedal and Wright Algorithm 3.3: add tau * I until Cholesky
            # succeeds. The actual retry loop runs in Newton._solve.
            _linear_solver = None

            def _shift_fn(_):
                return 0.0

        return LinearSolveDirection(
            linear_solver=_linear_solver,
            delta=_delta,
            resolved_linear_solver=resolved,
            shift_fn=_shift_fn,
            identity_shift_beta=identity_shift_beta,
            identity_shift_max_steps=identity_shift_max_steps,
            hessian_tag=hessian_tag,
        )
