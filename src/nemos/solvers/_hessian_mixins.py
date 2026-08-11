"""Mixin providing the curvature machinery for second-order solvers."""

from typing import Any, Callable, ClassVar, Optional

import jax
import lineax as lx

from .. import tree_utils
from .._hess import (
    BlockDiagonal,
    Full,
    General,
    HessianTag,
    PositiveDefinite,
    combine_hessian_tags,
)


class HessianMixin:
    """
    Receive and exploit the model's analytic Hessian.

    ``BaseRegressor`` offers the Hessian to any solver carrying this mixin, so
    curvature stays out of ``AbstractSolver``: a first-order solver has no use for
    ``setup_hessian``, a Hessian tag, or a linear solver, and should not inherit them.
    This mirrors ``StochasticSolverMixin``, which holds the stochastic machinery for
    the solvers that support it.

    Subclasses are expected to call ``_init_hessian`` from their ``__init__`` and
    ``_resolve_linear_solver`` from their ``init_state``.
    """

    # Declares the capability to ``BaseRegressor``, mirroring ``_supports_stochastic``.
    _uses_hessian: ClassVar[bool] = True

    # Whether the penalty is modelled by the quadratic. False for solvers that reach
    # the penalty through a proximal operator instead, which must not also add its
    # curvature here -- ``prox_elastic_net`` already applies the L2 rescale.
    _proximal: ClassVar[bool] = False

    def _init_hessian(self, regularizer, regularizer_strength, init_params) -> None:
        """Store what ``setup_hessian`` needs and default the resolved solver state."""
        self._regularizer = regularizer
        self._regularizer_strength = regularizer_strength
        self._init_params = init_params

        self._hess_tag: HessianTag | None = None
        self._hessian: Callable | None = None

        # Overwritten in _resolve_linear_solver once the tag is known.
        self._linear_solver = lx.AutoLinearSolver(well_posed=False)
        self._operator_tags = ()

    def setup_hessian(
        self,
        hess_fn: Callable | None = None,
        hess_tag: HessianTag | None = None,
        reg_tag: HessianTag | None = None,
        property_override: Optional[type] = None,
    ) -> None:
        """Accept the model's analytic Hessian and resolve the tag describing it.

        The invariant, whichever branch runs: ``self._hessian`` is the Hessian of the
        smooth objective the solver differentiates, and ``self._hess_tag`` describes that
        same matrix.
        """
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
            hess_fn = self._penalize_hessian(hess_fn, hess_tag)

        tag = hess_tag if reg_tag is None else combine_hessian_tags(hess_tag, reg_tag)
        if property_override is not None and tag is not None:
            tag = HessianTag(
                tag.structure, property_override, batch_axes=tag.batch_axes
            )
        self._hess_tag = tag
        self._hessian = hess_fn

    def _penalize_hessian(self, hess_fn, model_tag):
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
            if model_tag is not None and model_tag.structure is BlockDiagonal
            else None
        )
        penalty_hess_fn = self._regularizer._get_hess_fn(
            self._init_params, self._regularizer_strength, batch_axes=batch_axes
        )
        if penalty_hess_fn is None:
            # the regularizer declares no curvature, so the likelihood term is the whole
            return hess_fn

        def penalized_hessian(params, *args):
            return tree_utils.tree_add(hess_fn(params, *args), penalty_hess_fn(params))

        return penalized_hessian

    def _resolve_linear_solver(self) -> None:
        """Pick the linear solver once, from the Hessian tag.

        Cholesky for positive-definite Hessians, otherwise a robust least-squares solve
        that tolerates rank deficiency.
        """
        if self._hess_tag is None:
            self._hess_tag = HessianTag(structure=Full, property=General)

        if self._hess_tag.property is PositiveDefinite:
            self._linear_solver = lx.Cholesky()
            self._operator_tags = lx.positive_semidefinite_tag
        else:
            self._linear_solver = lx.AutoLinearSolver(well_posed=False)
            self._operator_tags = ()

    def _block_apply(self, fn, grad, H, other) -> Any:
        """Apply ``fn(grad, H, other)`` once per Hessian block.

        The one place that reads ``_hess_tag`` for block structure, shared by the Newton
        solve and by any subclass' Hessian-vector product.
        """
        if self._hess_tag.structure is BlockDiagonal:
            axes = self._hess_tag.batch_axes
            return jax.vmap(fn, in_axes=(axes, 0, axes), out_axes=axes)(grad, H, other)
        return fn(grad, H, other)
