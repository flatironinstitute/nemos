import lineax as lx
from functools import wraps
import jax
import jax.numpy as jnp
from typing import Callable

IMPLIES = {
    lx.unit_diagonal_tag: {
        lx.diagonal_tag,
        lx.symmetric_tag,
        lx.lower_triangular_tag,
        lx.upper_triangular_tag,
    },
    lx.diagonal_tag: {
        lx.symmetric_tag,
        lx.lower_triangular_tag,
        lx.upper_triangular_tag,
    },
    lx.positive_semidefinite_tag: {lx.symmetric_tag},
    lx.negative_semidefinite_tag: {lx.symmetric_tag},
    lx.symmetric_tag: set(),
    lx.lower_triangular_tag: set(),
    lx.upper_triangular_tag: set(),
}
NOT_ADDITIVE = {lx.unit_diagonal_tag}


def _expand(tag):
    return ({tag} | IMPLIES.get(tag, set())) - NOT_ADDITIVE


def _combine_hess_tags(tag1, tag2):
    if tag1 is None or tag2 is None:
        return None
    combined = _expand(tag1) & _expand(tag2)
    # drop tags implied by something more specific already in the set
    return frozenset(
        t for t in combined if not any(t in IMPLIES.get(s, set()) for s in combined)
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
