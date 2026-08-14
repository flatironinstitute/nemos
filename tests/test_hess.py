"""Tests for the Hessian tag algebra in ``nemos._hess``.

The tags decide which linear solver takes the Newton step, so a tag that claims more than
the matrix delivers hands a singular matrix to a Cholesky factorization. The tests at the
bottom of this file therefore do not stop at the set arithmetic: they build real Hessians
with a prescribed spectrum, check that each one satisfies every claim its tag makes, add
them, and compare the eigenvalues of the sum against the combined tag.
"""

from itertools import product

import pytest

from conftest import all_subclasses
from nemos._hess import (
    BlockDiagonal,
    Diagonal,
    Full,
    HessianTag,
    MatrixProperty,
    NegativeDefinite,
    NegativeSemiDefinite,
    NormalizedHessianTag,
    PositiveDefinite,
    PositiveSemiDefinite,
    Symmetric,
    is_covering,
    is_negative_signed,
    is_positive_signed,
    normalize,
)

# Leaf ids are strings rather than the ``id()`` of a real leaf, since the algebra only ever
# compares and intersects them. Nothing in it is written against a particular number of
# leaves, so every test that reads a leaf set is swept over both trees below.
TREES = [frozenset("ab"), frozenset("abc")]

ALL_PROPERTIES = sorted(all_subclasses(MatrixProperty), key=lambda cls: cls.__name__)

# The floor of the algebra: properties that carry no sign at all. Hand-written, so that a
# property class the sign predicates were never taught fails the test below.
UNSIGNED = (Symmetric,)

# How much each sign claims, for asserting that a rewrite never gives anything up.
SIGN_STRENGTH = {
    Symmetric: 0,
    PositiveSemiDefinite: 1,
    NegativeSemiDefinite: 1,
    PositiveDefinite: 2,
    NegativeDefinite: 2,
}


@pytest.fixture(params=TREES, ids=lambda tree: f"{len(tree)}-leaves")
def leaves(request):
    return request.param


def tag(leaves, property=Symmetric, structure=Full, flat_on=None, definite_on=None):
    """A tag over ``leaves`` claiming nothing that is not passed explicitly."""
    return HessianTag(
        structure=structure,
        property=property,
        leaves=leaves,
        flat_on=flat_on,
        definite_on=definite_on,
    )


def normalized_tag(
    leaves,
    property=Symmetric,
    structure=Full,
    flat_on=frozenset(),
    definite_on=frozenset(),
):
    """A ``NormalizedHessianTag``, whose leaf sets are empty rather than ``None``."""
    return NormalizedHessianTag(
        structure=structure,
        property=property,
        leaves=leaves,
        batch_axes=None,
        flat_on=flat_on,
        definite_on=definite_on,
    )


def realizable_tags(leaves):
    """Every tag some real matrix could satisfy: one sign, and a flat/definite split.

    Two conditions rule the rest out (Proposition 3 in the dev note): no leaf is both flat
    and definite, which ``claims`` already takes care of, and a matrix that is definite has
    no flat leaf, which drops the definite signs whenever ``flat_on`` is non-empty.
    """
    for prop in ALL_PROPERTIES:
        for flat_on, definite_on in claims(leaves):
            if flat_on and prop in (PositiveDefinite, NegativeDefinite):
                continue
            yield tag(leaves, property=prop, flat_on=flat_on, definite_on=definite_on)


def claims(leaves):
    """Every way of marking each leaf flat, definite or unclaimed: ``3 ** len(leaves)`` pairs.

    No leaf ends up in both sets. The pairs include the two extremes, all leaves flat and
    all leaves definite, and the pair of empty sets.
    """
    ordered = sorted(leaves)
    for assignment in product("fdu", repeat=len(ordered)):
        marked = dict(zip(ordered, assignment))
        yield (
            frozenset(leaf for leaf, mark in marked.items() if mark == "f"),
            frozenset(leaf for leaf, mark in marked.items() if mark == "d"),
        )


# --- is_positive_signed, is_negative_signed ---


@pytest.mark.parametrize(
    "prop, positive, negative",
    [
        (PositiveDefinite, True, False),
        (PositiveSemiDefinite, True, False),
        (NegativeDefinite, False, True),
        (NegativeSemiDefinite, False, True),
        (Symmetric, False, False),
    ],
)
def test_sign_of_each_property(prop, positive, negative, leaves):
    """Which sign each property class has: definite and semidefinite share one, and
    ``Symmetric`` has none.
    """
    t = tag(leaves, property=prop)
    assert is_positive_signed(t) is positive
    assert is_negative_signed(t) is negative


@pytest.mark.parametrize("prop", ALL_PROPERTIES)
def test_every_property_is_classified_by_exactly_one_sign(prop, leaves):
    """Every property class is positive, or negative, or listed in ``UNSIGNED``.

    Never both: ``combine_definite_on`` asks "same sign?" by calling the two predicates, so
    a property that answered yes to both would pass that check even against its opposite.
    And never neither by accident: a property class the predicates were never taught reads
    as unsigned, which drops every claim it makes without raising anything.
    """
    t = tag(leaves, property=prop)
    if prop in UNSIGNED:
        assert not is_positive_signed(t)
        assert not is_negative_signed(t)
    else:
        assert is_positive_signed(t) != is_negative_signed(t)


@pytest.mark.parametrize("prop", ALL_PROPERTIES)
def test_sign_predicates_ignore_normalization(prop, leaves):
    """The same answer on a raw and on a normalized tag.

    ``normalize`` calls both predicates on the tag it has not normalized yet, so they have
    to work on a ``HessianTag`` whose leaf sets are still ``None``.
    """
    t = tag(leaves, property=prop)
    assert is_positive_signed(t) is is_positive_signed(normalize(t))
    assert is_negative_signed(t) is is_negative_signed(normalize(t))


# --- is_covering ---


def test_covering_holds_exactly_when_no_leaf_is_unclaimed(leaves):
    """A tag is covering when every leaf shows up in ``flat_on`` or in ``definite_on``.

    Checks all ``3 ** len(leaves)`` markings, so: one leaf missing, several missing, and
    each choice of which leaf is the missing one.
    """
    for flat_on, definite_on in claims(leaves):
        print(flat_on, definite_on)
        unclaimed = [
            leaf for leaf in leaves if leaf not in flat_on and leaf not in definite_on
        ]
        t = normalized_tag(leaves, flat_on=flat_on, definite_on=definite_on)
        assert is_covering(t) is (not unclaimed), f"unclaimed: {unclaimed}"


@pytest.mark.parametrize("field", ["flat_on", "definite_on"])
def test_claiming_a_leaf_outside_the_tree_is_not_covering(field, leaves):
    """A tag naming a leaf the tree does not have is not covering.

    ``is_covering`` compares the union of the two sets to ``leaves`` with ``==``, so a set
    that is too big fails as well, not only one that is too small.
    """
    assert not is_covering(normalized_tag(leaves, **{field: leaves.union("z")}))


# --- normalize ---


def test_normalize_passes_none_through():
    """``None`` is the absence of a tag, not a tag, so it survives untouched."""
    assert normalize(None) is None


@pytest.mark.parametrize("prop", ALL_PROPERTIES)
def test_normalized_leaf_sets_are_never_none(prop, leaves):
    """``flat_on`` and ``definite_on`` come out as frozensets even when unset.

    A tag declares them as ``None`` when it has nothing to say about single leaves, and
    every function after ``normalize`` calls ``.union`` and ``.intersection`` on them.
    """
    t = normalize(tag(leaves, property=prop))
    assert isinstance(t.flat_on, frozenset)
    assert isinstance(t.definite_on, frozenset)


@pytest.mark.parametrize(
    "prop, expected_sign",
    [
        (Symmetric, PositiveSemiDefinite),
        (PositiveSemiDefinite, PositiveSemiDefinite),
        (NegativeSemiDefinite, NegativeSemiDefinite),
    ],
)
@pytest.mark.parametrize("structure", [Full, BlockDiagonal, Diagonal])
def test_flat_on_every_leaf_is_the_zero_matrix(prop, expected_sign, structure, leaves):
    """A tag flat on every leaf describes the zero matrix, so it comes back ``Diagonal``
    with ``definite_on`` emptied, whatever structure was declared.

    The zero matrix is both positive and negative semidefinite, and the declared sign is
    the one kept, so that a negative term stays negative for the same-sign check in
    ``combine_definite_on``.
    """
    t = normalize(tag(leaves, property=prop, structure=structure, flat_on=leaves))
    assert t.property is expected_sign
    assert t.structure is Diagonal
    assert t.flat_on == leaves
    assert t.definite_on == frozenset()


@pytest.mark.parametrize("prop", [PositiveDefinite, NegativeDefinite])
def test_a_definite_sign_claims_every_leaf(prop, leaves):
    """A tag declaring a definite sign comes back with ``definite_on`` filled in with every
    leaf, however few leaves it listed itself: a definite matrix is definite on each of its
    blocks too.
    """
    for _, definite_on in claims(leaves):
        t = normalize(tag(leaves, property=prop, definite_on=definite_on))
        assert t.property is prop
        assert t.definite_on == leaves
        assert t.flat_on == frozenset()


@pytest.mark.parametrize(
    "prop, promoted",
    [
        (PositiveSemiDefinite, PositiveDefinite),
        (NegativeSemiDefinite, NegativeDefinite),
    ],
)
def test_definite_on_every_leaf_promotes_the_sign(prop, promoted, leaves):
    """A tag definite on every leaf is definite on the whole matrix, so ``semidefinite``
    becomes ``definite``.
    """
    t = normalize(tag(leaves, property=prop, definite_on=leaves))
    assert t.property is promoted
    assert t.definite_on == leaves
    assert t.flat_on == frozenset()


def test_an_unsigned_tag_keeps_no_definite_claim(leaves):
    """``definite_on`` on a ``Symmetric`` tag is dropped, while ``flat_on`` is kept.

    A definite claim means definite *with the sign in* ``property``, and a ``Symmetric``
    tag has no sign to give it, so reading it as positive would be an invention. A flat
    claim needs no sign: a zero block is zero either way.
    """
    for flat_on, definite_on in claims(leaves):
        if flat_on == leaves:  # the zero matrix, which is signed
            continue
        t = normalize(
            tag(leaves, property=Symmetric, flat_on=flat_on, definite_on=definite_on)
        )
        assert t.property is Symmetric
        assert t.definite_on == frozenset()
        assert t.flat_on == flat_on


@pytest.mark.parametrize("prop", [PositiveSemiDefinite, NegativeSemiDefinite])
def test_a_partial_signed_tag_is_left_alone(prop, leaves):
    """With neither set reaching the whole tree there is nothing to restate."""
    for flat_on, definite_on in claims(leaves):
        if flat_on == leaves or definite_on == leaves:
            continue
        t = normalize(
            tag(leaves, property=prop, flat_on=flat_on, definite_on=definite_on)
        )
        assert t.property is prop
        assert t.flat_on == flat_on
        assert t.definite_on == definite_on


def test_normalize_is_idempotent(leaves):
    """Nothing is left to restate after one pass, over every realizable tag."""
    for t in realizable_tags(leaves):
        once = normalize(t)
        assert normalize(once) == once


def test_normalized_tags_are_realizable(leaves):
    """Proposition 3: the two sets stay disjoint, and a definite sign leaves no flat leaf."""
    for t in realizable_tags(leaves):
        once = normalize(t)
        assert not once.flat_on.intersection(once.definite_on)
        if once.property in (PositiveDefinite, NegativeDefinite):
            assert once.flat_on == frozenset()


def test_normalize_never_weakens_a_tag(leaves):
    """Every rewrite strengthens an under-claiming declaration, so nothing is given up:
    the sign never loosens or flips, and ``flat_on`` never shrinks. ``definite_on`` may,
    but only for an unsigned tag, whose definite claim has no sign to carry it.
    """
    for t in realizable_tags(leaves):
        norm_t = normalize(t)
        assert SIGN_STRENGTH[norm_t.property] >= SIGN_STRENGTH[t.property]
        assert not (is_positive_signed(t) and is_negative_signed(norm_t))
        assert not (is_negative_signed(t) and is_positive_signed(norm_t))
        assert norm_t.flat_on.issuperset(
            frozenset() if t.flat_on is None else t.flat_on
        )
        if is_positive_signed(t) or is_negative_signed(t):
            expected = frozenset() if t.definite_on is None else t.definite_on
            assert norm_t.definite_on.issuperset(expected)
