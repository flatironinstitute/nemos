from dataclasses import dataclass

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
    structure: type
    property: type


def combine_hessian_tags(
    t1: HessianTag | None, t2: HessianTag | None
) -> HessianTag | None:
    """Combine tags assuming additivity.

    Valid when the total objective is a sum of two functions (e.g. loss + regularizer),
    since the Hessian of a sum is the sum of the Hessians.
    """
    if t1 is None or t2 is None:
        return None
    prop = combine_property(t1.property, t2.property)
    return HessianTag(
        structure=combine_structure(t1.structure, t2.structure),
        property=prop,
    )
