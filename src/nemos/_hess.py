from dataclasses import dataclass
from typing import Any

# --- Properties ---


class MatrixProperty:
    pass


class Symmetric(MatrixProperty):
    pass


class PositiveSemiDefinite(MatrixProperty):
    pass


class PositiveDefinite(MatrixProperty):
    pass


class NegativeSemiDefinite(MatrixProperty):
    pass


class NegativeDefinite(MatrixProperty):
    pass


# --- Structures ---


class MatrixStructure:
    pass


class Full(MatrixStructure):
    pass


class BlockDiagonal(MatrixStructure):
    pass


class Diagonal(MatrixStructure):
    pass


_STRUCTURE_GENERALITY: dict[type, int] = {
    Diagonal: 0,
    BlockDiagonal: 1,
    Full: 2,
}


def combine_structure(s1, s2) -> type:
    c1 = s1 if isinstance(s1, type) else type(s1)
    c2 = s2 if isinstance(s2, type) else type(s2)
    return max(c1, c2, key=lambda t: _STRUCTURE_GENERALITY[t])


# --- Combined tag ---


@dataclass(frozen=True)
class HessianTag:
    """Structure and definiteness of a Hessian, with optional detail per parameter.

    ``flat_on`` and ``definite_on`` describe single parts of the parameter tree rather
    than the whole matrix. Both are sets of leaf ids, taken with ``id()`` on the leaves
    of the parameters. ``combine_hessian_tags`` reads them to tell whether the sum of two
    Hessians is positive definite even when neither one is. They say different things:

    - ``flat_on``: the term has no curvature at all on these parameters, and is positive
      definite on all the others. A ridge penalty that skips the intercept says this,
      because the penalty does not depend on the intercept, so its second derivative
      there is zero.
    - ``definite_on``: the term curves on these parameters, with the sign of ``property``,
      and nothing is claimed about the rest. A GLM loss says this about its intercept,
      where the curvature is the sum of the per-sample weights and so is positive.
    - ``leaves``: every leaf id the tag speaks about. ``flat_on`` and ``definite_on`` are
      subsets of it, and comparing against it is how the algebra recognizes a claim that
      reaches the whole matrix.

    ``None`` means the tag says nothing about single parameters.
    """

    structure: type
    property: type
    leaves: frozenset
    batch_axes: Any = None
    flat_on: frozenset | None = None
    definite_on: frozenset | None = None


@dataclass(frozen=True)
class NormalizedHessianTag:
    """Normalized version of ``HessianTag``."""

    structure: type
    property: type
    leaves: frozenset
    batch_axes: Any
    flat_on: frozenset
    definite_on: frozenset


def combine_definite_on(
    t1: NormalizedHessianTag, t2: NormalizedHessianTag
) -> frozenset:
    """Largest leaf set on which ``H1 + H2`` is guaranteed definite.

    A leaf set is guaranteed definite when the two tags restricted to it are linked in
    the sense of Theorem 2, which is a pair of conditions of the form ``S <= ...``, so
    each side of the disjunction has a greatest element: the term's own ``definite_on``,
    grown by the leaves it is flat on and the other term curves on.

    The two candidates are both maximal and their union is *not* sound — two terms can
    each be definite on their own leaves and share a null direction that neither set
    names — so one of them has to be chosen and the other discarded. A model has an
    empty ``flat_on``, so it is the penalty side that grows.

    Opposite signs cancel, and an unsigned term cannot be bounded below on the leaves
    the other one curves on, so a pair that is not signed the same way keeps nothing.
    """
    if not (
        (is_positive_signed(t1) and is_positive_signed(t2))
        or (is_negative_signed(t1) and is_negative_signed(t2))
    ):
        return frozenset()
    m1 = t1.definite_on.union(t1.flat_on.intersection(t2.definite_on))
    m2 = t2.definite_on.union(t2.flat_on.intersection(t1.definite_on))
    return max(m1, m2, key=len)


def combine_property(
    t1: NormalizedHessianTag, t2: NormalizedHessianTag, definite_on: frozenset
) -> type:
    """Resolve the definiteness of the sum ``H1 + H2`` from the summands' properties.

    Used for additive objectives (e.g. loss + regularizer), where the Hessian of a sum
    is the sum of the Hessians. Definiteness combines as:

    - a combined ``definite_on`` that reaches every leaf is a definite block on the whole
      matrix, hence a definite matrix. This is where the sum can be definite although
      neither summand is, and it subsumes the case of a summand that is definite already,
      which ``normalize`` states as ``definite_on == leaves``.
    - like-signed semidefinite terms stay semidefinite.
    - anything else degrades to ``Symmetric`` (no definiteness guarantee, e.g. mixing
      positive and negative curvature).
    """
    # Theorem 2, developers_notes/08-hessian_tagging.md: the linked condition holds
    # exactly when the combined definite set covers the tree.
    if definite_on == t1.leaves:
        return PositiveDefinite if is_positive_signed(t1) else NegativeDefinite
    elif is_positive_signed(t1) and is_positive_signed(t2):
        return PositiveSemiDefinite
    elif is_negative_signed(t1) and is_negative_signed(t2):
        return NegativeSemiDefinite
    return Symmetric


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
    flat_on = frozenset() if tag.flat_on is None else tag.flat_on
    definite_on = frozenset() if tag.definite_on is None else tag.definite_on

    if flat_on == tag.leaves:
        # every block vanishes, so the matrix is zero
        structure, sign = Diagonal, PositiveSemiDefinite
        definite_on = frozenset()
    elif not (is_positive_signed(tag) or is_negative_signed(tag)):
        # with no sign for the whole matrix, definite_on carries no sign either
        definite_on = frozenset()
    elif sign in (PositiveDefinite, NegativeDefinite):
        # a definite matrix is definite on every block and flat on none
        flat_on, definite_on = frozenset(), tag.leaves
    elif definite_on == tag.leaves:
        # the block on every leaf is the whole matrix
        sign = PositiveDefinite if is_positive_signed(tag) else NegativeDefinite
        flat_on = frozenset()

    return NormalizedHessianTag(
        structure=structure,
        property=sign,
        leaves=tag.leaves,
        batch_axes=tag.batch_axes,
        flat_on=flat_on,
        definite_on=definite_on,
    )


def is_negative_signed(tag: HessianTag | NormalizedHessianTag) -> bool:
    return tag.property in (NegativeDefinite, NegativeSemiDefinite)


def is_positive_signed(tag: HessianTag | NormalizedHessianTag) -> bool:
    return tag.property in (PositiveDefinite, PositiveSemiDefinite)


def is_covering(tag: NormalizedHessianTag) -> bool:
    """Whether the tag speaks about every leaf, each one either flat or definite."""
    return tag.flat_on.union(tag.definite_on) == tag.leaves


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
        leaves=t1.leaves,
        batch_axes=t1.batch_axes,
        flat_on=t1.flat_on.intersection(t2.flat_on),
        definite_on=definite_on,
    )
