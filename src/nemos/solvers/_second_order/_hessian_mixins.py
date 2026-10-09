"""The mixin through which a solver receives the model's analytic Hessian."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, ClassVar, Optional

import jax
from jaxtyping import Array, PyTree

from ... import tree_utils
from ..._hess import (
    HessianTag,
    MatrixProperty,
    MatrixStructure,
    combine_hessian_tags,
    mask_claim_none,
)
from ...typing import Params
from ._curvature import NewtonCurvature

if TYPE_CHECKING:
    from ...regularizer import Regularizer


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

    # ``_proximal`` is read below and declared by the host, on
    # ``AbstractSecondOrderSolver``: whether the penalty is modelled by the quadratic.
    # It is False for solvers that reach the penalty through a proximal operator
    # instead, which must not also add its curvature here -- ``prox_elastic_net``
    # already applies the L2 rescale.

    def _init_hessian(
        self,
        regularizer: Regularizer,
        regularizer_strength: float | None,
        init_params: Params,
        hess_fn: Callable[..., PyTree[Array]] | None = None,
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
        self,
        hess_fn: Callable[..., PyTree[Array]] | None,
        model_tag: HessianTag | None,
        regularizer: Regularizer,
        regularizer_strength: float | None,
        init_params: Params,
    ) -> Callable[..., PyTree[Array]] | None:
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
