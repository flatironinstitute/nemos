"""Tests for the Hessian tag algebra in ``nemos._hess``.

The tags decide which linear solver takes the Newton step, so a tag that claims more than
the matrix delivers hands a singular matrix to a Cholesky factorization. The tests at the
bottom of this file therefore do not stop at the set arithmetic: they build real Hessians
with a prescribed spectrum, check that each one satisfies every claim its tag makes, add
them, and compare the eigenvalues of the sum against the combined tag.
"""

from itertools import combinations, product

import pytest

from nemos._hess import (
    HessianTag,
    MatrixProperty,
    MatrixStructure,
    NormalizedHessianTag,
    combine_definite_on,
    combine_property,
    is_covering,
    is_negative_signed,
    is_positive_signed,
    normalize,
)

# The trees tags are declared against, named by their leaves. Plain dicts stand in for a
# parameter container: the functions under test only map over the structure and read the
# booleans, so nothing here depends on the leaves being arrays or on the tree being a
# ``ModelParams``. Nothing depends on how many leaves there are either, so every test that
# builds a leaf set runs against both.
TREES = [("a", "b"), ("a", "b", "c")]

# A two-leaf tree for the tests that spell their cases out by hand.
PAIR = ("a", "b")

ALL_PROPERTIES = list(MatrixProperty)

# Properties with no sign at all. Written out by hand, so that a new property the sign
# predicates do not know about fails the test below instead of passing as unsigned.
UNSIGNED = (MatrixProperty.SYMMETRIC,)

# How much each sign claims, from nothing up to definite. Used to check that a rewrite
# never gives anything up.
SIGN_STRENGTH = {
    MatrixProperty.SYMMETRIC: 0,
    MatrixProperty.POSITIVE_SEMI_DEFINITE: 1,
    MatrixProperty.NEGATIVE_SEMI_DEFINITE: 1,
    MatrixProperty.POSITIVE_DEFINITE: 2,
    MatrixProperty.NEGATIVE_DEFINITE: 2,
}

# Short names, so that each row of the tables below fits on one line.
PD, PSD, ND, NSD, SYM = (
    MatrixProperty.POSITIVE_DEFINITE,
    MatrixProperty.POSITIVE_SEMI_DEFINITE,
    MatrixProperty.NEGATIVE_DEFINITE,
    MatrixProperty.NEGATIVE_SEMI_DEFINITE,
    MatrixProperty.SYMMETRIC,
)


@pytest.fixture(params=TREES, ids=lambda tree: f"{len(tree)}-leaves")
def tree(request):
    """Sweep every test that reads a leaf set over a two- and a three-leaf tree."""
    return request.param


def mask(tree, claimed=()):
    """Build a leaf set over ``tree``, claiming the leaves named in ``claimed``.

    ``claimed`` is any container of leaf names, so ``mask(tree)`` claims nothing and
    ``mask(tree, tree)`` claims everything.
    """
    return {leaf: leaf in claimed for leaf in tree}


def claimed(leaf_set):
    """Read a leaf set back as the plain set of names it claims.

    Expectations are stated in set language, which keeps them independent of the tree
    arithmetic under test and lets subset and superset comparisons be Python's own.
    """
    return {leaf for leaf, is_claimed in leaf_set.items() if is_claimed}


def universe(t):
    """Read the leaves a tag talks about off the structure of its leaf sets."""
    return set(t.flat_on)


def tag(tree, prop=SYM, structure=MatrixStructure.FULL, flat_on=None, definite_on=None):
    """Build a tag over ``tree``, claiming nothing that is not passed explicitly."""
    return HessianTag(
        structure=structure,
        property=prop,
        flat_on=mask(tree) if flat_on is None else flat_on,
        definite_on=mask(tree) if definite_on is None else definite_on,
    )


def normalized_tag(
    tree, prop=SYM, structure=MatrixStructure.FULL, flat_on=None, definite_on=None
):
    """Build a ``NormalizedHessianTag`` over ``tree``."""
    return NormalizedHessianTag(
        structure=structure,
        property=prop,
        batch_axes=None,
        flat_on=mask(tree) if flat_on is None else flat_on,
        definite_on=mask(tree) if definite_on is None else definite_on,
    )


def claims(tree):
    """Mark each leaf flat, definite or unclaimed, every way there is: ``3 ** n`` pairs.

    No leaf ends up in both sets. The pairs include the two extremes, all leaves flat and
    all leaves definite, and the pair that claims nothing.
    """
    for assignment in product("fdu", repeat=len(tree)):
        marked = dict(zip(tree, assignment))
        yield (
            mask(tree, [leaf for leaf, m in marked.items() if m == "f"]),
            mask(tree, [leaf for leaf, m in marked.items() if m == "d"]),
        )


def realizable_tags(tree):
    """Yield every tag some real matrix could satisfy: a sign, and a flat/definite split.

    Two conditions rule the rest out (Proposition 3 in the dev note): no leaf is both flat
    and definite, which ``claims`` already takes care of, and a matrix that is definite has
    no flat leaf, which drops the definite signs whenever ``flat_on`` claims anything.
    """
    for prop in ALL_PROPERTIES:
        for flat_on, definite_on in claims(tree):
            if claimed(flat_on) and prop in (PD, ND):
                continue
            yield tag(tree, prop=prop, flat_on=flat_on, definite_on=definite_on)


def subsets(names):
    """Yield every subset of ``names``, smallest first."""
    ordered = sorted(names)
    for size in range(len(ordered) + 1):
        for chosen in combinations(ordered, size):
            yield frozenset(chosen)


def linked_on(t1, t2, subset):
    """Say whether ``t1`` and ``t2`` are linked on ``subset``.

    That is whether the sum of two matrices satisfying them has to be definite on those
    leaves. The two conditions are Theorem 2's, read on ``subset`` alone: ``t1`` accounts
    for every leaf of it, each one either flat or definite, and every leaf of it that ``t1``
    is flat on is one that ``t2`` is definite on. On those leaves the sum is then ``t1``'s
    curvature where ``t1`` has some, and ``t2``'s where it has none.
    """
    accounted = subset <= claimed(t1.flat_on) | claimed(t1.definite_on)
    filled_in = subset & claimed(t1.flat_on) <= claimed(t2.definite_on)
    return accounted and filled_in


def definite_sets(t1, t2):
    """Find every set of leaves the sum of the two is guaranteed definite on, by trying them.

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
        for subset in subsets(universe(t1))
        if linked_on(t1, t2, subset) or linked_on(t2, t1, subset)
    ]


# --- is_positive_signed, is_negative_signed ---


@pytest.mark.parametrize(
    "prop, positive, negative",
    [
        (PD, True, False),
        (PSD, True, False),
        (ND, False, True),
        (NSD, False, True),
        (SYM, False, False),
    ],
)
def test_sign_of_each_property(prop, positive, negative, tree):
    """Check which sign each property has.

    Definite and semidefinite share one, and ``SYMMETRIC`` has none.
    """
    t = tag(tree, prop=prop)
    assert is_positive_signed(t) is positive
    assert is_negative_signed(t) is negative


@pytest.mark.parametrize("prop", ALL_PROPERTIES)
def test_every_property_is_classified_by_exactly_one_sign(prop, tree):
    """Check every property is positive, or negative, or listed in ``UNSIGNED``.

    Never both: ``combine_definite_on`` asks "same sign?" by calling the two predicates, so
    a property that answered yes to both would pass that check even against its opposite.
    And never neither by accident: a property the predicates were never taught reads as
    unsigned, which drops every claim it makes without raising anything.
    """
    t = tag(tree, prop=prop)
    if prop in UNSIGNED:
        assert not is_positive_signed(t)
        assert not is_negative_signed(t)
    else:
        assert is_positive_signed(t) != is_negative_signed(t)


@pytest.mark.parametrize("prop", ALL_PROPERTIES)
def test_sign_predicates_ignore_normalization(prop, tree):
    """Check the answer is the same on a raw and on a normalized tag.

    ``normalize`` calls both predicates on the tag it has not normalized yet, so they have
    to work either side of the rewrites.
    """
    t = tag(tree, prop=prop)
    assert is_positive_signed(t) is is_positive_signed(normalize(t))
    assert is_negative_signed(t) is is_negative_signed(normalize(t))


# --- is_covering ---


def test_covering_holds_exactly_when_no_leaf_is_unclaimed(tree):
    """Check a tag is covering when every leaf is in ``flat_on`` or in ``definite_on``.

    Runs all ``3 ** n`` markings, so: one leaf missing, several missing, and each choice of
    which leaf is the missing one.
    """
    for flat_on, definite_on in claims(tree):
        unclaimed = set(tree) - claimed(flat_on) - claimed(definite_on)
        t = normalized_tag(tree, flat_on=flat_on, definite_on=definite_on)
        assert is_covering(t) is (not unclaimed), f"unclaimed: {unclaimed}"


def test_leaf_sets_over_different_trees_do_not_combine(tree):
    """Check a tag whose two leaf sets are over different trees raises rather than answers.

    The leaf sets are combined leaf by leaf, so a mismatch is a structural error that
    ``jax.tree_util.tree_map`` reports. Nothing silently intersects two different trees and
    returns a claim about neither of them.
    """
    wider = normalized_tag(tree, flat_on=mask(tuple(tree) + ("z",)))
    with pytest.raises(ValueError, match="key mismatch"):
        is_covering(wider)


# --- normalize ---


def test_normalize_passes_none_through():
    """Check ``None`` survives untouched: it is the absence of a tag, not a tag."""
    assert normalize(None) is None


@pytest.mark.parametrize(
    "prop, expected_sign",
    [(SYM, PSD), (PSD, PSD), (NSD, NSD)],
)
@pytest.mark.parametrize(
    "structure",
    [MatrixStructure.FULL, MatrixStructure.BLOCK_DIAGONAL, MatrixStructure.DIAGONAL],
)
def test_flat_on_every_leaf_is_the_zero_matrix(prop, expected_sign, structure, tree):
    """Check a tag flat on every leaf comes back diagonal with ``definite_on`` emptied.

    Such a tag describes the zero matrix, whatever structure was declared. The zero matrix
    is both positive and negative semidefinite, and the declared sign is the one kept, so
    that a negative term stays negative for the same-sign check in ``combine_definite_on``.
    """
    t = normalize(tag(tree, prop=prop, structure=structure, flat_on=mask(tree, tree)))
    assert t.property is expected_sign
    assert t.structure is MatrixStructure.DIAGONAL
    assert t.flat_on == mask(tree, tree)
    assert t.definite_on == mask(tree)


@pytest.mark.parametrize("prop", [PD, ND])
def test_a_definite_sign_claims_every_leaf(prop, tree):
    """Check a tag declaring a definite sign comes back definite on every leaf.

    However few leaves it listed itself: a definite matrix is definite on each of its blocks
    too.
    """
    for _, definite_on in claims(tree):
        t = normalize(tag(tree, prop=prop, definite_on=definite_on))
        assert t.property is prop
        assert t.definite_on == mask(tree, tree)
        assert t.flat_on == mask(tree)


@pytest.mark.parametrize("prop, promoted", [(PSD, PD), (NSD, ND)])
def test_definite_on_every_leaf_promotes_the_sign(prop, promoted, tree):
    """Check a tag definite on every leaf comes back with a definite sign.

    Being definite on every leaf is being definite on the whole matrix, so semidefinite
    becomes definite.
    """
    t = normalize(tag(tree, prop=prop, definite_on=mask(tree, tree)))
    assert t.property is promoted
    assert t.definite_on == mask(tree, tree)
    assert t.flat_on == mask(tree)


def test_an_unsigned_tag_keeps_no_definite_claim(tree):
    """Check a ``SYMMETRIC`` tag loses ``definite_on`` and keeps ``flat_on``.

    A definite claim means definite *with the sign in* ``property``, and a ``SYMMETRIC`` tag
    has no sign to give it, so reading it as positive would be an invention. A flat claim
    needs no sign: a zero block is zero either way.
    """
    for flat_on, definite_on in claims(tree):
        if claimed(flat_on) == set(tree):  # the zero matrix, which is signed
            continue
        t = normalize(tag(tree, prop=SYM, flat_on=flat_on, definite_on=definite_on))
        assert t.property is SYM
        assert t.definite_on == mask(tree)
        assert t.flat_on == flat_on


@pytest.mark.parametrize("prop", [PSD, NSD])
def test_a_partial_signed_tag_is_left_alone(prop, tree):
    """Check a signed tag whose two sets both stop short of the whole tree is unchanged.

    None of the rewrites apply, so the sign and both sets are the ones declared.
    """
    for flat_on, definite_on in claims(tree):
        if set(tree) in (claimed(flat_on), claimed(definite_on)):
            continue
        t = normalize(tag(tree, prop=prop, flat_on=flat_on, definite_on=definite_on))
        assert t.property is prop
        assert t.flat_on == flat_on
        assert t.definite_on == definite_on


def test_normalize_is_idempotent(tree):
    """Check normalizing an already normalized tag changes nothing, for every tag."""
    for t in realizable_tags(tree):
        once = normalize(t)
        assert normalize(once) == once


def test_normalized_tags_are_realizable(tree):
    """Check a normalized tag still describes some real matrix.

    No leaf is both flat and definite, and a tag with a definite sign has no flat leaf
    (Proposition 3 in the dev note).
    """
    for t in realizable_tags(tree):
        once = normalize(t)
        assert not claimed(once.flat_on) & claimed(once.definite_on)
        if once.property in (PD, ND):
            assert claimed(once.flat_on) == set()


def test_normalize_never_weakens_a_tag(tree):
    """Check normalizing only ever adds to what a tag says, never takes away.

    So the sign comes back at least as strong as it went in, a positive tag never comes back
    negative or the other way round, and ``flat_on`` keeps every leaf it had.
    ``definite_on`` can lose leaves, but only on a ``SYMMETRIC`` tag, where the claim has no
    sign to carry it.
    """
    for t in realizable_tags(tree):
        norm_t = normalize(t)
        assert SIGN_STRENGTH[norm_t.property] >= SIGN_STRENGTH[t.property]
        assert not (is_positive_signed(t) and is_negative_signed(norm_t))
        assert not (is_negative_signed(t) and is_positive_signed(norm_t))
        assert claimed(norm_t.flat_on) >= claimed(t.flat_on)
        if is_positive_signed(t) or is_negative_signed(t):
            assert claimed(norm_t.definite_on) >= claimed(t.definite_on)


# --- combine_definite_on ---


def test_ridge_penalized_glm_is_definite_on_both_leaves():
    """Check the case the whole mechanism exists for.

    A GLM loss that curves on the intercept, plus a ridge penalty that curves on the
    coefficients and not on the intercept. Neither matrix is definite by itself, and the sum
    is definite on both leaves.
    """
    glm_tree = ("coef", "intercept")
    loss = normalize(tag(glm_tree, prop=PSD, definite_on=mask(glm_tree, ["intercept"])))
    penalty = normalize(
        tag(
            glm_tree,
            prop=PSD,
            flat_on=mask(glm_tree, ["intercept"]),
            definite_on=mask(glm_tree, ["coef"]),
        )
    )
    assert combine_definite_on(loss, penalty) == mask(glm_tree, glm_tree)
    assert combine_definite_on(penalty, loss) == mask(glm_tree, glm_tree)


def test_two_definite_claims_are_not_unioned():
    """Check two terms definite on a leaf each keep one of the two sets, not both.

    Which one is settled by the argument order: the first tag's set wins. Their union would
    be wrong. Take ``H = I - vv.T / 2`` with ``v = e_a - e_b``: ``H`` is semidefinite, it is
    definite on leaf ``a`` and on leaf ``b``, and ``H + H`` is singular. Both tags hold for
    that one matrix, and neither says anything about the direction ``v`` the two terms
    share, so the sum cannot be claimed definite on both leaves.

    The two sets are the same size and neither contains the other, so nothing about the
    matrices picks a winner. The order does, and it is pinned here because which leaves come
    back changes which linear solver the Newton step ends up using.
    """
    t1 = normalize(tag(PAIR, prop=PSD, definite_on=mask(PAIR, "a")))
    t2 = normalize(tag(PAIR, prop=PSD, definite_on=mask(PAIR, "b")))
    assert combine_definite_on(t1, t2) == mask(PAIR, "a")
    assert combine_definite_on(t2, t1) == mask(PAIR, "b")


def test_definite_claims_need_the_same_sign(tree):
    """Check a pair that is not same-signed keeps no definite leaf.

    A positive term and a negative one can cancel exactly, and so can an unsigned term with
    anything.
    """
    for t1, t2 in product(realizable_tags(tree), repeat=2):
        n1, n2 = normalize(t1), normalize(t2)
        same_sign = (is_positive_signed(n1) and is_positive_signed(n2)) or (
            is_negative_signed(n1) and is_negative_signed(n2)
        )
        if not same_sign:
            assert combine_definite_on(n1, n2) == mask(tree)


def test_combined_definite_set_is_the_largest_guaranteed_one(tree):
    """Check the leaves kept are guaranteed, and that no guaranteed set is larger.

    ``definite_sets`` works the guarantee out set by set from Theorem 2, so this checks both
    that the claim holds and that nothing claimable was left behind.
    """
    for t1, t2 in product(realizable_tags(tree), repeat=2):
        n1, n2 = normalize(t1), normalize(t2)
        combined = claimed(combine_definite_on(n1, n2))
        guaranteed = definite_sets(n1, n2)
        assert combined in guaranteed or (not guaranteed and not combined)
        bigger = [s for s in guaranteed if s > combined]
        assert not bigger, f"{combined} could have been {bigger}"


def test_swapping_the_two_tags_keeps_as_many_leaves(tree):
    """Check argument order never changes how many leaves come back.

    It changes which ones only when there are two largest guaranteed sets to choose between.
    When one guaranteed set contains all the others, both orders have to return it; when two
    are tied, the first argument's is the one kept, so the two orders differ.
    """
    for t1, t2 in product(realizable_tags(tree), repeat=2):
        n1, n2 = normalize(t1), normalize(t2)
        forward = claimed(combine_definite_on(n1, n2))
        backward = claimed(combine_definite_on(n2, n1))
        assert len(forward) == len(backward)
        guaranteed = definite_sets(n1, n2)
        largest = [s for s in guaranteed if not any(s < other for other in guaranteed)]
        if len(largest) == 1:
            assert forward == backward == largest[0]


# --- combine_property ---


def expected_sign(t1, t2):
    """Work out the strongest sign the sum of two matrices satisfying the tags must have.

    Two same-signed terms add up to a term with that sign, and the sum is definite when the
    leaves it is guaranteed definite on are all of them. A pair that is not same-signed
    gives nothing, since the two curvatures can cancel.
    """
    everywhere = universe(t1) in definite_sets(t1, t2)
    if is_positive_signed(t1) and is_positive_signed(t2):
        return PD if everywhere else PSD
    if is_negative_signed(t1) and is_negative_signed(t2):
        return ND if everywhere else NSD
    return SYM


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
        ((PSD, "", "ab"), (SYM, "", ""), SYM),
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
        ((SYM, "ab", ""), (PSD, "", "a"), PSD),
        # -- No common sign, so nothing survives.
        ((PSD, "", "a"), (NSD, "a", "b"), SYM),
        ((SYM, "", ""), (PSD, "a", "b"), SYM),
        ((SYM, "", "a"), (SYM, "a", "b"), SYM),
        # An unsigned tag's definite claim is dropped by ``normalize`` instead of being read
        # as positive, so covering every leaf with it buys no definite verdict.
        ((SYM, "", "ab"), (PSD, "", ""), SYM),
    ],
)
def test_sign_of_the_sum(spec1, spec2, expected):
    """Check the verdict on the sum, for each shape of pair on a two-leaf tree.

    The rows are grouped by which rewrite ``normalize`` applies to each tag on the way in,
    since a tag can reach the verdict with its sign strengthened or its definite claim
    dropped, and those paths are as much part of the result as the sets themselves.
    """
    t1, t2 = (
        normalize(
            tag(
                PAIR,
                prop=prop,
                flat_on=mask(PAIR, flat),
                definite_on=mask(PAIR, definite),
            )
        )
        for prop, flat, definite in (spec1, spec2)
    )
    assert combine_property(t1, t2, combine_definite_on(t1, t2)) is expected


def test_sign_of_the_sum_over_every_pair(tree):
    """Check the verdict is the strongest sign the sum must have, over every pair of tags.

    And that it does not depend on which tag is passed first.
    """
    for t1, t2 in product(realizable_tags(tree), repeat=2):
        n1, n2 = normalize(t1), normalize(t2)
        forward = combine_property(n1, n2, combine_definite_on(n1, n2))
        backward = combine_property(n2, n1, combine_definite_on(n2, n1))
        assert forward is expected_sign(n1, n2)
        assert forward is backward


def test_the_sum_is_definite_exactly_when_the_pair_is_linked(tree):
    """Check Theorem 2 read on the whole tree.

    The sum of two same-signed terms is definite when one of them accounts for every leaf
    and the other curves on the ones it is flat on.
    """
    for t1, t2 in product(realizable_tags(tree), repeat=2):
        n1, n2 = normalize(t1), normalize(t2)
        same_sign = (is_positive_signed(n1) and is_positive_signed(n2)) or (
            is_negative_signed(n1) and is_negative_signed(n2)
        )
        whole = set(tree)
        linked = same_sign and (linked_on(n1, n2, whole) or linked_on(n2, n1, whole))
        verdict = combine_property(n1, n2, combine_definite_on(n1, n2))
        assert (verdict in (PD, ND)) is linked
