from __future__ import annotations

import operator
from dataclasses import dataclass
from enum import Enum, IntEnum, auto

import jax
from jaxtyping import PyTree

from .inverse_link_function_utils import identity
from .tree_utils import pytree_map_and_reduce

# --- Properties ---


class MatrixProperty(Enum):
    """The sign a term claims for its Hessian, at every parameter value.

    Only these five are needed, because they are what the decisions turn on: whether a
    Newton step points downhill at all, and whether the matrix can be Cholesky-factorized.
    ``SYMMETRIC`` is the floor, claiming no sign whatsoever.
    """

    SYMMETRIC = auto()
    POSITIVE_SEMI_DEFINITE = auto()
    POSITIVE_DEFINITE = auto()
    NEGATIVE_SEMI_DEFINITE = auto()
    NEGATIVE_DEFINITE = auto()


# --- Structures ---


class MatrixStructure(IntEnum):
    """The sparsity a term claims for its Hessian.

    The value is how general the structure is, so the structure of a sum is the larger of
    the two: a diagonal matrix plus a block diagonal one is block diagonal, and anything
    plus a full one is full. ``FULL`` is the floor, claiming no sparsity to exploit.
    """

    DIAGONAL = 0
    BLOCK_DIAGONAL = 1
    FULL = 2


def combine_structure(s1: MatrixStructure, s2: MatrixStructure) -> MatrixStructure:
    """Combine two structures: ``H1 + H2`` is as general as the more general of the two.

    Only correct because the block diagonal case in nemos always comes from vmapping over
    a batch axis, so the two terms are block diagonal with respect to the *same* partition.
    Two block diagonal matrices with different partitions sum to something with no block
    structure at all; see ``developers_notes/08-hessian_tagging.md``.

    Parameters
    ----------
    s1, s2 :
        The structures of the two terms.

    Returns
    -------
    :
        The structure of their sum.
    """
    return max(s1, s2)


# --- Leaf claims ---


class LeafClaim(Enum):
    """What a term certifies about one leaf's own block of the Hessian.

    A leaf carries exactly one of these, so it can never be declared both flat and definite
    — a pair of claims no matrix satisfies, since a zero block is singular.

    - ``UNCLAIMED``: nothing is certified about this leaf's block.
    - ``FLAT``: the block is zero, i.e. the term has no curvature there at all.
    - ``DEFINITE``: the block is definite, with the sign carried by the tag's ``property``.
    """

    UNCLAIMED = auto()
    FLAT = auto()
    DEFINITE = auto()


# --- Leaf sets ---
#
# A leaf set is a tree with the structure of the parameters and a boolean at every leaf:
# ``True`` where the claim holds, ``False`` where it does not. Two leaf sets can be
# combined leaf by leaf, and a claim about the whole matrix is a leaf set that is ``True``
# everywhere. Parameters held fixed are ``None`` in the parameter tree and ``None`` in the
# leaf set as well, so a claim about them drops out with them.


def claim_nothing(params: PyTree) -> PyTree[LeafClaim]:
    """Build a claim tree that certifies nothing about any leaf.

    Parameters
    ----------
    params :
        The parameters the claims are about. Only its structure is read.

    Returns
    -------
    :
        A tree shaped like ``params``, carrying ``LeafClaim.UNCLAIMED`` at every leaf. Leaves
        that are ``None`` in ``params`` — parameters held fixed — stay ``None``, so the
        tree keeps talking about exactly the parameters that are being fitted.
    """
    return jax.tree_util.tree_map(lambda _: LeafClaim.UNCLAIMED, params)


def mask_of_claim(claims: PyTree[LeafClaim], claim: LeafClaim) -> PyTree[bool]:
    """Turn a claim tree into the leaf set carrying one particular claim.

    Parameters
    ----------
    claims :
        A tree with a :class:`LeafClaim` member at every leaf.
    claim :
        The claim to look for, e.g. ``LeafClaim.DEFINITE``.

    Returns
    -------
    :
        A leaf set: a tree shaped like ``claims`` holding ``True`` at the leaves carrying
        ``claim`` and ``False`` elsewhere. ``None`` leaves stay ``None``.
    """
    return jax.tree_util.tree_map(lambda leaf_claim: leaf_claim is claim, claims)


def mask_union(m1: PyTree[bool], m2: PyTree[bool]) -> PyTree[bool]:
    """Leaves claimed by either one."""
    return jax.tree_util.tree_map(operator.or_, m1, m2)


def mask_intersection(m1: PyTree[bool], m2: PyTree[bool]) -> PyTree[bool]:
    """Leaves claimed by both."""
    return jax.tree_util.tree_map(operator.and_, m1, m2)


def mask_claim_none(like: PyTree) -> PyTree[bool]:
    """Build a leaf set over the same tree that claims no leaf."""
    return jax.tree_util.tree_map(lambda _: False, like)


def mask_claim_all(like: PyTree) -> PyTree[bool]:
    """Build a leaf set over the same tree that claims every leaf."""
    return jax.tree_util.tree_map(lambda _: True, like)


def mask_claims_all(mask: PyTree[bool]) -> bool:
    """Whether every leaf is claimed."""
    return pytree_map_and_reduce(identity, all, mask)


def mask_n_claimed(mask: PyTree[bool]) -> int:
    """How many leaves are claimed."""
    return pytree_map_and_reduce(identity, sum, mask)


# --- Combined tag ---


@dataclass(frozen=True)
class HessianTag:
    """Structure and definiteness of a Hessian, with detail per parameter.

    ``flat_on`` and ``definite_on`` are leaf sets: trees shaped like the parameters, with a
    boolean at every leaf. They talk about single parts of the parameter tree rather than
    about the whole matrix, and ``combine_hessian_tags`` reads them to tell whether the sum
    of two Hessians is definite even when neither one is. They say different things:

    - ``flat_on``: the term has no curvature at all on these parameters. A ridge penalty
      that skips the intercept says this, because the penalty does not depend on the
      intercept, so its second derivative there is zero.
    - ``definite_on``: the term curves on these parameters, with the sign of ``property``,
      and nothing is claimed about the rest. A GLM loss says this about its intercept,
      where the curvature is the sum of the per-sample weights and so is positive.

    Both are claims about the parameters the solver is optimizing, so they are built when
    the solver is set up rather than declared ahead of time: which parameters a penalty
    reaches depends on its strength, and which parameters exist at all depends on what is
    held fixed. A term with nothing to say about single parameters passes leaf sets that
    are ``False`` everywhere.
    """

    structure: MatrixStructure
    property: MatrixProperty
    flat_on: PyTree[bool]
    definite_on: PyTree[bool]
    batch_axes: PyTree[int | None] | None = None


@dataclass(frozen=True)
class NormalizedHessianTag:
    """Normalized version of ``HessianTag``."""

    structure: MatrixStructure
    property: MatrixProperty
    flat_on: PyTree[bool]
    definite_on: PyTree[bool]
    batch_axes: PyTree[int | None] | None


def combine_definite_on(
    t1: NormalizedHessianTag, t2: NormalizedHessianTag
) -> PyTree[bool]:
    """Find the largest set of leaves on which ``H1 + H2`` is guaranteed definite.

    A set of leaves is guaranteed definite when the two tags restricted to it are linked in
    the sense of Theorem 2, which is a pair of conditions of the form ``S <= ...``, so
    each side of the disjunction has a greatest element: the term's own ``definite_on``,
    grown by the leaves it is flat on and the other term curves on.

    The two candidates are both maximal and their union is *not* sound — two terms can
    each be definite on their own leaves and share a null direction that neither one
    mentions — so one of them has to be chosen and the other discarded. Ties go to the
    first argument. A model is flat on no leaf, so it is the penalty side that grows.

    Opposite signs cancel, and an unsigned term cannot be bounded below on the leaves
    the other one curves on, so a pair that is not signed the same way keeps nothing.
    """
    if not (
        (is_positive_signed(t1) and is_positive_signed(t2))
        or (is_negative_signed(t1) and is_negative_signed(t2))
    ):
        return mask_claim_none(t1.definite_on)
    m1 = mask_union(t1.definite_on, mask_intersection(t1.flat_on, t2.definite_on))
    m2 = mask_union(t2.definite_on, mask_intersection(t2.flat_on, t1.definite_on))
    return max(m1, m2, key=mask_n_claimed)


def combine_property(
    t1: NormalizedHessianTag, t2: NormalizedHessianTag, definite_on: PyTree[bool]
) -> MatrixProperty:
    """Resolve the definiteness of the sum ``H1 + H2`` from the summands' properties.

    Used for additive objectives (e.g. loss + regularizer), where the Hessian of a sum
    is the sum of the Hessians. Definiteness combines as:

    - a combined ``definite_on`` that reaches every leaf is a definite block on the whole
      matrix, hence a definite matrix. This is where the sum can be definite although
      neither summand is, and it subsumes the case of a summand that is definite already,
      which ``normalize`` states as a ``definite_on`` claiming every leaf.
    - like-signed semidefinite terms stay semidefinite.
    - anything else degrades to ``MatrixProperty.SYMMETRIC`` (no definiteness guarantee, e.g. mixing
      positive and negative curvature).
    """
    # Theorem 2, developers_notes/08-hessian_tagging.md: the linked condition holds
    # exactly when the combined definite set covers the tree.
    if mask_claims_all(definite_on):
        return (
            MatrixProperty.POSITIVE_DEFINITE
            if is_positive_signed(t1)
            else MatrixProperty.NEGATIVE_DEFINITE
        )
    elif is_positive_signed(t1) and is_positive_signed(t2):
        return MatrixProperty.POSITIVE_SEMI_DEFINITE
    elif is_negative_signed(t1) and is_negative_signed(t2):
        return MatrixProperty.NEGATIVE_SEMI_DEFINITE
    return MatrixProperty.SYMMETRIC


def normalize(tag: HessianTag | None) -> NormalizedHessianTag | None:
    """Restate a tag as the strongest equivalent one, using only what it already implies.

    Two tags with the same satisfying set are interchangeable, and the rewrites below
    each replace an under-claiming declaration with the strongest one describing the same
    matrices, so the combination rule can be written against a single form.
    """
    if tag is None:
        return None

    structure = tag.structure
    sign = tag.property
    flat_on = tag.flat_on
    definite_on = tag.definite_on

    if mask_claims_all(flat_on):
        # every block vanishes, so the matrix is zero
        sign = (
            MatrixProperty.NEGATIVE_SEMI_DEFINITE
            if is_negative_signed(tag)
            else MatrixProperty.POSITIVE_SEMI_DEFINITE
        )
        structure = MatrixStructure.DIAGONAL
        definite_on = mask_claim_none(definite_on)
    elif not (is_positive_signed(tag) or is_negative_signed(tag)):
        # with no sign for the whole matrix, definite_on carries no sign either
        definite_on = mask_claim_none(definite_on)
    elif sign in (MatrixProperty.POSITIVE_DEFINITE, MatrixProperty.NEGATIVE_DEFINITE):
        # a definite matrix is definite on every block and flat on none
        flat_on, definite_on = mask_claim_none(flat_on), mask_claim_all(definite_on)
    elif mask_claims_all(definite_on):
        # the block on every leaf is the whole matrix
        sign = (
            MatrixProperty.POSITIVE_DEFINITE
            if is_positive_signed(tag)
            else MatrixProperty.NEGATIVE_DEFINITE
        )
        flat_on = mask_claim_none(flat_on)

    return NormalizedHessianTag(
        structure=structure,
        property=sign,
        batch_axes=tag.batch_axes,
        flat_on=flat_on,
        definite_on=definite_on,
    )


def is_negative_signed(tag: HessianTag | NormalizedHessianTag) -> bool:
    return tag.property in (
        MatrixProperty.NEGATIVE_DEFINITE,
        MatrixProperty.NEGATIVE_SEMI_DEFINITE,
    )


def is_positive_signed(tag: HessianTag | NormalizedHessianTag) -> bool:
    return tag.property in (
        MatrixProperty.POSITIVE_DEFINITE,
        MatrixProperty.POSITIVE_SEMI_DEFINITE,
    )


def is_covering(tag: NormalizedHessianTag) -> bool:
    """Whether the tag speaks about every leaf, each one either flat or definite."""
    return mask_claims_all(mask_union(tag.flat_on, tag.definite_on))


def combine_hessian_tags(
    t1: HessianTag | None, t2: HessianTag | None
) -> HessianTag | None:
    """Structure and definiteness of the sum of two Hessians.

    Valid when the total objective is a sum of two functions (e.g. loss + regularizer),
    since the Hessian of a sum is the sum of the Hessians.

    Two positive semidefinite terms normally add up to a positive semidefinite one, but
    the sum is positive definite when one term is flat only where the other curves; see
    ``combine_definite_on``. What the two tags say about single parameters carries over to
    the sum: the sum vanishes on a leaf only when both terms do, which makes ``flat_on``
    an intersection, and it curves on the leaves ``combine_definite_on`` keeps.

    The batch_axes are taken from t1, which will typically be the model tag.
    """
    # fast out: undefined properties
    if t1 is None or t2 is None:
        return None

    t1 = normalize(t1)
    t2 = normalize(t2)
    definite_on = combine_definite_on(t1, t2)

    return HessianTag(
        structure=combine_structure(t1.structure, t2.structure),
        property=combine_property(t1, t2, definite_on),
        batch_axes=t1.batch_axes,
        flat_on=mask_intersection(t1.flat_on, t2.flat_on),
        definite_on=definite_on,
    )
