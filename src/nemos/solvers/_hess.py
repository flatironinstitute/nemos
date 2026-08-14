from dataclasses import dataclass
from typing import Any

# --- Properties ---


class MatrixProperty:
    pass


class General(MatrixProperty):
    pass


class Symmetric(MatrixProperty):
    pass


class PositiveSemiDefinite(MatrixProperty):
    pass


class PositiveDefinite(MatrixProperty):
    pass


class NegativeDefinite(MatrixProperty):
    pass


PROPERTY_IMPLIES: dict[type, set[type]] = {
    PositiveDefinite: {PositiveSemiDefinite, Symmetric, General},
    PositiveSemiDefinite: {Symmetric, General},
    NegativeDefinite: {Symmetric, General},
    Symmetric: {General},
    General: set(),
}


def _expand_property(p) -> set[type]:
    cls = p if isinstance(p, type) else type(p)
    return {cls} | PROPERTY_IMPLIES.get(cls, set())


def combine_property(p1, p2) -> type:
    """Resolve the definiteness of the sum ``H1 + H2`` from the summands' properties.

    Used for additive objectives (e.g. loss + regularizer), where the Hessian of a sum
    is the sum of the Hessians. Definiteness combines as:

    - ``General`` on either side yields ``General`` (no structure survives a general term).
    - a strictly definite term absorbs a same-signed semidefinite one, e.g.
      ``PositiveDefinite + PositiveSemiDefinite -> PositiveDefinite``; like-signed definite
      terms stay definite.
    - any other combination of symmetric terms degrades to ``Symmetric`` (no definiteness
      guarantee, e.g. mixing positive and negative curvature).
    """
    c1 = p1 if isinstance(p1, type) else type(p1)
    c2 = p2 if isinstance(p2, type) else type(p2)
    pair = {c1, c2}

    if General in pair:
        return General

    positive = {PositiveDefinite, PositiveSemiDefinite}
    if pair <= positive:
        return PositiveDefinite if PositiveDefinite in pair else PositiveSemiDefinite

    if pair == {NegativeDefinite}:
        return NegativeDefinite

    # both summands are symmetric (PD/PSD/NegDef/Symmetric) but the combination
    # carries no definiteness guarantee.
    return Symmetric


_WEAKENED_PROPERTY: dict[type, type] = {
    PositiveDefinite: PositiveSemiDefinite,
    NegativeDefinite: Symmetric,
}


def weaken_property(p) -> type:
    """Docstring with jargon.

    A term can curve on the parameters it acts on and be zero on the rest, like a ridge
    penalty that skips the intercept. Padding a matrix with zeros, or multiplying it by
    zero, only adds directions with no curvature at all; it never turns curvature
    negative. So the sign stays and the strictness goes:

    - positive definite becomes positive semidefinite,
    - negative definite becomes symmetric, there being no negative semidefinite property
      to fall back on,
    - a claim that was not strict in the first place has nothing to lose and comes back
      unchanged.
    """
    cls = p if isinstance(p, type) else type(p)
    return _WEAKENED_PROPERTY.get(cls, cls)


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
    - ``definite_on``: the term curves on these parameters, and nothing is claimed about
      the rest. A GLM loss says this about its intercept, where the curvature is the sum
      of the per-sample weights and so is positive.

    ``None`` means the tag says nothing about single parameters.
    """

    structure: type
    property: type
    batch_axes: Any = None
    flat_on: frozenset | None = None
    definite_on: frozenset | None = None


def _certifies_definite(flat_side: HessianTag, other: HessianTag) -> bool:
    """Whether the sum of two positive semidefinite Hessians is positive definite.

    In words: one term is flat only where the other one curves.

    ``flat_side`` has no curvature on ``flat_on`` and curves everywhere else, so those
    parameters are the only directions it is flat in. If ``other`` curves on those same
    parameters, then no direction is flat for both terms. A sum of two positive
    semidefinite matrices is singular only along directions that are flat in both, so the
    sum is positive definite.

    Both halves are needed. Two terms that each curve on one block, but are exactly zero
    nowhere, can still add up to something singular: ``A = B = [[1, 1], [1, 1]]`` are
    both positive semidefinite, the first curves on the second coordinate and the second
    on the first, and the sum ``[[2, 2], [2, 2]]`` is singular.
    """
    if flat_side.flat_on is None:
        return False
    curved_by_other = other.definite_on or frozenset()
    return flat_side.flat_on.issubset(curved_by_other)


def _union(s1: frozenset | None, s2: frozenset | None) -> frozenset | None:
    if s1 is None and s2 is None:
        return None
    return (s1 or frozenset()).union(s2 or frozenset())


def combine_hessian_tags(
    t1: HessianTag | None, t2: HessianTag | None
) -> HessianTag | None:
    """Structure and definiteness of the sum of two Hessians.

    Valid when the total objective is a sum of two functions (e.g. loss + regularizer),
    since the Hessian of a sum is the sum of the Hessians.

    Two positive semidefinite terms normally add up to a positive semidefinite one, but
    the sum is positive definite when one term is flat only where the other curves; see
    ``_certifies_definite``. What the two tags say about single parameters carries over to
    the sum: there is no curvature only where neither term has any, and a parameter one
    term curves on keeps curving once the other is added to it.

    The batch_axes are taken from t1, which will typically be the model tag.
    """
    if t1 is None or t2 is None:
        return None
    prop = combine_property(t1.property, t2.property)
    if prop is PositiveSemiDefinite and (
        _certifies_definite(t1, t2) or _certifies_definite(t2, t1)
    ):
        prop = PositiveDefinite
    flat_on = (
        None
        if t1.flat_on is None or t2.flat_on is None
        else t1.flat_on.intersection(t2.flat_on)
    )
    return HessianTag(
        structure=combine_structure(t1.structure, t2.structure),
        property=prop,
        batch_axes=t1.batch_axes,
        flat_on=flat_on,
        definite_on=_union(t1.definite_on, t2.definite_on),
    )
