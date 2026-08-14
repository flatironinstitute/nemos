"""Tests for the Hessian tag algebra in ``nemos._hess``.

The tags decide which linear solver takes the Newton step, so a tag that claims more than
the matrix delivers hands a singular matrix to a Cholesky factorization. The tests at the
bottom of this file therefore do not stop at the set arithmetic: they build real Hessians
with a prescribed spectrum, check that each one satisfies every claim its tag makes, add
them, and compare the eigenvalues of the sum against the combined tag.
"""

from itertools import combinations, product

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
    combine_definite_on,
    combine_property,
    is_covering,
    is_negative_signed,
    is_positive_signed,
    normalize,
)

# Leaf ids are strings here, not the ``id()`` of a real pytree leaf: the functions under
# test only compare and intersect them. None of them depends on how many leaves there are,
# so tests that use a leaf set run against both trees.
TREES = [frozenset("ab"), frozenset("abc")]

ALL_PROPERTIES = sorted(all_subclasses(MatrixProperty), key=lambda cls: cls.__name__)

# Properties with no sign at all. Written out by hand, so that a new property class the
# sign predicates do not know about fails the test below instead of passing as unsigned.
UNSIGNED = (Symmetric,)

# How much each sign claims, from nothing up to definite. Used to check that a rewrite
# never gives anything up.
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


def tag(leaves, prop: type = Symmetric, structure=Full, flat_on=None, definite_on=None):
    """A tag over ``leaves`` claiming nothing that is not passed explicitly."""
    return HessianTag(
        structure=structure,
        property=prop,
        leaves=leaves,
        flat_on=flat_on,
        definite_on=definite_on,
    )


def normalized_tag(
    leaves,
    prop=Symmetric,
    structure=Full,
    flat_on=frozenset(),
    definite_on=frozenset(),
):
    """A ``NormalizedHessianTag``, whose leaf sets are empty rather than ``None``."""
    return NormalizedHessianTag(
        structure=structure,
        property=prop,
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
            yield tag(leaves, prop=prop, flat_on=flat_on, definite_on=definite_on)


def subsets(leaves):
    """Every subset of ``leaves``, smallest first."""
    ordered = sorted(leaves)
    for size in range(len(ordered) + 1):
        for chosen in combinations(ordered, size):
            yield frozenset(chosen)


def linked_on(t1, t2, subset):
    """Whether ``t1`` and ``t2`` are linked on ``subset``, i.e. whether the sum of two
    matrices satisfying them has to be definite on those leaves.

    The two conditions are Theorem 2's, read on ``subset`` alone: ``t1`` accounts for every
    leaf of it, each one either flat or definite, and every leaf of it that ``t1`` is flat
    on is one that ``t2`` is definite on. Then on those leaves the sum is ``t1``'s
    curvature where ``t1`` has some, and ``t2``'s where it has none.
    """
    accounted = subset.issubset(t1.flat_on.union(t1.definite_on))
    filled_in = subset.intersection(t1.flat_on).issubset(t2.definite_on)
    return accounted and filled_in


def definite_sets(t1, t2):
    """Every leaf set the sum of the two is guaranteed definite on, found by trying them.

    Empty unless the two tags carry the same sign: opposite curvatures can cancel, and an
    unsigned tag gives no bound at all on the leaves the other one curves on.
    """
    same_sign = (is_positive_signed(t1) and is_positive_signed(t2)) or (
        is_negative_signed(t1) and is_negative_signed(t2)
    )
    if not same_sign:
        return []
    return [
        subset
        for subset in subsets(t1.leaves)
        if linked_on(t1, t2, subset) or linked_on(t2, t1, subset)
    ]


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
    t = tag(leaves, prop=prop)
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
    t = tag(leaves, prop=prop)
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
    t = tag(leaves, prop=prop)
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
    t = normalize(tag(leaves, prop=prop))
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
    t = normalize(tag(leaves, prop=prop, structure=structure, flat_on=leaves))
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
        t = normalize(tag(leaves, prop=prop, definite_on=definite_on))
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
    t = normalize(tag(leaves, prop=prop, definite_on=leaves))
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
            tag(leaves, prop=Symmetric, flat_on=flat_on, definite_on=definite_on)
        )
        assert t.property is Symmetric
        assert t.definite_on == frozenset()
        assert t.flat_on == flat_on


@pytest.mark.parametrize("prop", [PositiveSemiDefinite, NegativeSemiDefinite])
def test_a_partial_signed_tag_is_left_alone(prop, leaves):
    """A signed tag whose two sets both stop short of the whole tree comes back unchanged:
    none of the rewrites apply, so the sign and both sets are the ones declared.
    """
    for flat_on, definite_on in claims(leaves):
        if flat_on == leaves or definite_on == leaves:
            continue
        t = normalize(tag(leaves, prop=prop, flat_on=flat_on, definite_on=definite_on))
        assert t.property is prop
        assert t.flat_on == flat_on
        assert t.definite_on == definite_on


def test_normalize_is_idempotent(leaves):
    """Normalizing an already normalized tag changes nothing, for every realizable tag."""
    for t in realizable_tags(leaves):
        once = normalize(t)
        assert normalize(once) == once


def test_normalized_tags_are_realizable(leaves):
    """A normalized tag still describes some real matrix: no leaf is both flat and definite,
    and a tag with a definite sign has no flat leaf (Proposition 3 in the dev note).
    """
    for t in realizable_tags(leaves):
        once = normalize(t)
        assert not once.flat_on.intersection(once.definite_on)
        if once.property in (PositiveDefinite, NegativeDefinite):
            assert once.flat_on == frozenset()


def test_normalize_never_weakens_a_tag(leaves):
    """Normalizing only ever adds to what a tag says, it never takes away.

    So the sign comes back at least as strong as it went in, a positive tag never comes
    back negative or the other way round, and ``flat_on`` keeps every leaf it had.
    ``definite_on`` can lose leaves, but only on a ``Symmetric`` tag, where the claim has
    no sign to carry it.
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


# --- combine_definite_on ---


def test_ridge_penalized_glm_is_definite_on_both_leaves():
    """The case the whole mechanism exists for: a GLM loss that curves on the intercept,
    plus a ridge penalty that curves on the coefficients and not on the intercept. Neither
    matrix is definite by itself, and the sum is definite on both leaves.
    """
    glm_leaves = frozenset({"coef", "intercept"})
    loss = normalize(
        tag(glm_leaves, prop=PositiveSemiDefinite, definite_on=frozenset({"intercept"}))
    )
    penalty = normalize(
        tag(
            glm_leaves,
            prop=PositiveSemiDefinite,
            flat_on=frozenset({"intercept"}),  # as in ridge currently
            definite_on=frozenset({"coef"}),
        )
    )
    # this make sure that the combined tag is definite (it has a strict sign)
    assert combine_definite_on(loss, penalty) == glm_leaves
    assert combine_definite_on(penalty, loss) == glm_leaves


def test_two_definite_claims_are_not_unioned():
    """Two terms definite on a leaf each keep one of the two sets, not both, and which one
    is settled by the argument order: the first tag's set wins.

    Their union would be wrong. Take ``H = I - vv.T / 2`` with ``v = e_a - e_b``: ``H`` is
    semidefinite, it is definite on leaf ``a`` and on leaf ``b``, and ``H + H`` is singular.
    Both tags hold for that one matrix, and neither says anything about the direction ``v``
    the two terms share, so the sum cannot be claimed definite on both leaves.

    The two sets are the same size and neither contains the other, so nothing about the
    matrices picks a winner. The order does, and it is pinned here because which leaves come
    back changes which linear solver the Newton step ends up using.
    """
    pair_leaves = frozenset("ab")
    t1 = normalize(
        tag(pair_leaves, prop=PositiveSemiDefinite, definite_on=frozenset("a"))
    )
    t2 = normalize(
        tag(pair_leaves, prop=PositiveSemiDefinite, definite_on=frozenset("b"))
    )
    assert combine_definite_on(t1, t2) == frozenset("a")
    assert combine_definite_on(t2, t1) == frozenset("b")


def test_definite_claims_need_the_same_sign(leaves):
    """A positive term and a negative one keep no definite leaf, and neither does a pair
    with an unsigned term in it: the two curvatures can cancel exactly.
    """
    for t1, t2 in product(realizable_tags(leaves), repeat=2):
        n1, n2 = normalize(t1), normalize(t2)
        same_sign = (is_positive_signed(n1) and is_positive_signed(n2)) or (
            is_negative_signed(n1) and is_negative_signed(n2)
        )
        if not same_sign:
            assert combine_definite_on(n1, n2) == frozenset()


def test_combined_definite_set_is_the_largest_guaranteed_one(leaves):
    """Over every pair of tags: the leaves kept are ones the sum must be definite on, and
    no set that is guaranteed can be added to them.

    ``definite_sets`` works the guarantee out set by set, so this checks both that the
    claim holds and that nothing claimable was left behind.
    """
    for t1, t2 in product(realizable_tags(leaves), repeat=2):
        n1, n2 = normalize(t1), normalize(t2)
        combined = combine_definite_on(n1, n2)
        guaranteed = definite_sets(n1, n2)
        assert combined in guaranteed or (not guaranteed and combined == frozenset())
        bigger = [s for s in guaranteed if s > combined]
        assert not bigger, f"{combined} could have been {bigger}"


def test_swapping_the_two_tags_keeps_as_many_leaves(leaves):
    """Argument order never changes how many leaves come back, and changes which ones only
    when there are two largest guaranteed sets to choose between.

    When one guaranteed set contains all the others, both orders have to return it. When two
    are tied, the first argument's is the one kept, so the two orders differ.
    """
    for t1, t2 in product(realizable_tags(leaves), repeat=2):
        n1, n2 = normalize(t1), normalize(t2)
        forward = combine_definite_on(n1, n2)
        backward = combine_definite_on(n2, n1)
        assert len(forward) == len(backward)
        guaranteed = definite_sets(n1, n2)
        largest = [s for s in guaranteed if not any(s < other for other in guaranteed)]
        if len(largest) == 1:
            assert forward == backward == largest[0]


# --- combine_property ---


def expected_sign(t1, t2):
    """The strongest sign the sum of two matrices satisfying the tags is bound to have.

    Two same-signed terms add up to a term with that sign, and the sum is definite when the
    leaves it is guaranteed definite on are all of them. A pair that is not same-signed
    gives nothing, since the two curvatures can cancel.
    """
    everywhere = t1.leaves in definite_sets(t1, t2)
    if is_positive_signed(t1) and is_positive_signed(t2):
        return PositiveDefinite if everywhere else PositiveSemiDefinite
    if is_negative_signed(t1) and is_negative_signed(t2):
        return NegativeDefinite if everywhere else NegativeSemiDefinite
    return Symmetric


# Short names, so that each row of the table below fits on one line.
PD, PSD, ND, NSD = (
    PositiveDefinite,
    PositiveSemiDefinite,
    NegativeDefinite,
    NegativeSemiDefinite,
)


@pytest.mark.parametrize(
    "spec1, spec2, expected",
    [
        # -- Neither tag is rewritten, so the verdict rests on the sets as declared.
        # Flat where the other one curves, curving where the other one is flat, i.e. the
        # ridge-penalized GLM: the sum is definite though neither term is.
        ((PSD, "", "a"), (PSD, "a", "b"), PD),
        ((NSD, "", "a"), (NSD, "a", "b"), ND),
        # Curving on one leaf each and silent about the other: the sum can still be
        # singular, so the verdict stops at semidefinite.
        ((PSD, "", "a"), (PSD, "", "b"), PSD),
        ((NSD, "", "a"), (NSD, "", "b"), NSD),
        # Neither one says anything per leaf, and they still add up with their common sign.
        ((PSD, "", ""), (PSD, "", ""), PSD),
        # -- A semidefinite tag definite on every leaf, which ``normalize`` turns definite.
        # The promotion carries the sum with it, whatever the other tag says per leaf.
        ((PSD, "", "ab"), (PSD, "", ""), PD),
        ((NSD, "", "ab"), (NSD, "", ""), ND),
        # ...but not across a missing common sign.
        ((PSD, "", "ab"), (Symmetric, "", ""), Symmetric),
        # -- A tag declaring a definite sign and no leaves, which ``normalize`` fills in.
        ((PD, "", ""), (PSD, "", ""), PD),
        ((ND, "", ""), (NSD, "a", "b"), ND),
        ((PD, "", ""), (PD, "", ""), PD),
        # -- A tag flat on every leaf, which is the zero matrix. Adding it changes nothing,
        # so the verdict is the one the other tag supports on its own.
        ((PSD, "ab", ""), (PSD, "", "a"), PSD),
        ((PSD, "ab", ""), (PD, "", ""), PD),
        ((PSD, "ab", ""), (PSD, "ab", ""), PSD),
        # The zero matrix is negative semidefinite as well, and declaring it that way has
        # to keep the pair same-signed, or a negative term would lose its sign to it.
        ((NSD, "ab", ""), (NSD, "", "a"), NSD),
        # Declared with no sign it is still the zero matrix, hence semidefinite.
        ((Symmetric, "ab", ""), (PSD, "", "a"), PSD),
        # -- No common sign, so nothing survives.
        ((PSD, "", "a"), (NSD, "a", "b"), Symmetric),
        ((Symmetric, "", ""), (PSD, "a", "b"), Symmetric),
        ((Symmetric, "", "a"), (Symmetric, "a", "b"), Symmetric),
        # An unsigned tag's definite claim is dropped by ``normalize`` instead of being read
        # as positive, so covering every leaf with it buys no definite verdict.
        ((Symmetric, "", "ab"), (PSD, "", ""), Symmetric),
    ],
)
def test_sign_of_the_sum(spec1, spec2, expected):
    """The verdict on the sum, for each shape of pair on a two-leaf tree.

    The rows are grouped by which rewrite ``normalize`` applies to each tag on the way in,
    since a tag can reach the verdict with its sign strengthened or its definite claim
    dropped, and those paths are as much part of the result as the sets themselves.
    """
    pair_leaves = frozenset("ab")
    t1, t2 = (
        normalize(
            tag(
                pair_leaves,
                prop=prop,
                flat_on=frozenset(flat),
                definite_on=frozenset(definite),
            )
        )
        for prop, flat, definite in (spec1, spec2)
    )
    assert combine_property(t1, t2, combine_definite_on(t1, t2)) is expected


def test_sign_of_the_sum_over_every_pair(leaves):
    """Over every pair of tags, the verdict is the strongest sign the sum must have, and it
    does not depend on which tag is passed first.
    """
    for t1, t2 in product(realizable_tags(leaves), repeat=2):
        n1, n2 = normalize(t1), normalize(t2)
        forward = combine_property(n1, n2, combine_definite_on(n1, n2))
        backward = combine_property(n2, n1, combine_definite_on(n2, n1))
        assert forward is expected_sign(n1, n2)
        assert forward is backward


def test_the_sum_is_definite_exactly_when_the_pair_is_linked(leaves):
    """Theorem 2 read on the whole tree: the sum of two same-signed terms is definite when
    one of them accounts for every leaf and the other curves on the ones it is flat on.
    """
    for t1, t2 in product(realizable_tags(leaves), repeat=2):
        n1, n2 = normalize(t1), normalize(t2)
        same_sign = (is_positive_signed(n1) and is_positive_signed(n2)) or (
            is_negative_signed(n1) and is_negative_signed(n2)
        )
        linked = same_sign and (linked_on(n1, n2, leaves) or linked_on(n2, n1, leaves))
        verdict = combine_property(n1, n2, combine_definite_on(n1, n2))
        assert (verdict in (PositiveDefinite, NegativeDefinite)) is linked
