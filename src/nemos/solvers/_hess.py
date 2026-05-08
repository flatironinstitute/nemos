from dataclasses import dataclass
from functools import wraps
from typing import Callable

import jax
import jax.numpy as jnp

# --- Properties ---


class MatrixProperty:
    pass


class General(MatrixProperty):
    pass


class Symmetric(MatrixProperty):
    pass


class PositiveDefinite(MatrixProperty):
    pass


class NegativeDefinite(MatrixProperty):
    pass


PROPERTY_IMPLIES: dict[type, set[type]] = {
    PositiveDefinite: {Symmetric, General},
    NegativeDefinite: {Symmetric, General},
    Symmetric: {General},
    General: set(),
}


def _expand_property(p) -> set[type]:
    cls = p if isinstance(p, type) else type(p)
    return {cls} | PROPERTY_IMPLIES.get(cls, set())


def combine_property(p1, p2) -> type | None:
    common = _expand_property(p1) & _expand_property(p2)
    if not common:
        return None
    most_specific = [
        p
        for p in common
        if not any(p in PROPERTY_IMPLIES.get(q, set()) for q in common)
    ]
    return most_specific[0] if len(most_specific) == 1 else None


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
    if prop is None:
        return None
    return HessianTag(
        structure=combine_structure(t1.structure, t2.structure),
        property=prop,
    )


def _elementwise_derivative(f: Callable) -> Callable:
    """Construct the element-wise derivative of a function using forward-mode AD.

    Parameters
    ----------
    f :
        A function acting element-wise on an array.

    Returns
    -------
    Callable
        A function that computes the derivative of ``f`` evaluated element-wise.
    """

    @wraps(f)
    def df(x):
        _, grad = jax.jvp(f, (x,), (jnp.ones_like(x),))
        return grad

    return df
