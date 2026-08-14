"""Tests for the Hessian tag algebra in ``nemos.solvers._hess``.

The tags decide which linear solver takes the Newton step, so a tag that claims more than
the matrix delivers hands a singular matrix to a Cholesky factorization. The tests at the
bottom of this file therefore do not stop at the set arithmetic: they build real Hessians
with a prescribed spectrum, check that each one satisfies every claim its tag makes, add
them, and compare the eigenvalues of the sum against the combined tag.
"""

from itertools import product

import numpy as np
import pytest

from nemos.solvers._hess import (
    Diagonal,
    Full,
    General,
    HessianTag,
    NegativeDefinite,
    PositiveDefinite,
    PositiveSemiDefinite,
    Symmetric,
    _expand_property,
    combine_hessian_tags,
    weaken_property,
)

_ALL_PROPERTIES = [
    PositiveDefinite,
    PositiveSemiDefinite,
    NegativeDefinite,
    Symmetric,
    General,
]

# Leaf ids standing in for a parameter tree, since that is what the tags carry. Two leaves
# are the plain GLM, one coefficient array and one intercept. The third is the shape a
# pytree design gives, where ``coef`` holds one leaf per group of features, and it is the
# only shape in which one tag's ``flat_on`` can be a strict subset of another's
# ``definite_on`` rather than equal to it.
_INTERCEPT_LEAF, _COEF_LEAF, _SECOND_COEF_LEAF = 1, 2, 3
_INTERCEPT = frozenset({_INTERCEPT_LEAF})
_COEF = frozenset({_COEF_LEAF})
_SECOND_COEF = frozenset({_SECOND_COEF_LEAF})

# Which rows and columns of a Hessian belong to which parameter, keyed by leaf id. A tag
# names parameters by leaf id while a Hessian is one flat matrix, so building a block by
# hand needs the mapping spelled out; in nemos it is implicit in how ``jax.hessian``
# flattens the parameter tree. ``{1: [0], 2: [1, 2, 3]}`` is a 4x4 Hessian whose intercept
# is row and column 0 and whose three coefficients are rows and columns 1 to 3.
Layout = dict[int, np.ndarray]

# Every generated matrix takes its eigenvalues from this range, so the condition number is
# 4 and the largest eigenvalue is 2 whatever the layout. Pinning the spectrum is what makes
# the thresholds below derivable instead of measured.
_SMALLEST_EIGENVALUE, _LARGEST_EIGENVALUE = 0.5, 2.0
# The single negative eigenvalue given to an indefinite term. Weyl's inequality bounds
# lam_min(A + B) <= lam_min(A) + lam_max(B) <= -3 + 2 = -1, and lam_max(A + B) <= 4, so the
# sum's smallest eigenvalue relative to its largest is at most -1/4.
_NEGATIVE_EIGENVALUE = -3.0
_INDEFINITE_BOUND = (_NEGATIVE_EIGENVALUE + _LARGEST_EIGENVALUE) / (
    2 * _LARGEST_EIGENVALUE
)
# Floating point slack, relative to the largest eigenvalue. The definite cases clear it by
# some six orders of magnitude and the singular ones sit at 1e-16, so nothing here is tuned.
_TOL = 1e-8


def _layout(*block_sizes: int) -> Layout:
    """Give each parameter a block of consecutive indices, the intercept going first.

    Parameters
    ----------
    *block_sizes :
        How many scalars each parameter holds, in leaf id order, so the first is the
        intercept and the rest are coefficients.

    Returns
    -------
    :
        Which rows and columns each parameter occupies.
    """
    indices = np.arange(sum(block_sizes))
    edges = np.cumsum((0,) + block_sizes)
    return {
        leaf: indices[start:stop]
        for leaf, (start, stop) in enumerate(zip(edges, edges[1:]), start=1)
    }


# A one-dimensional intercept is the real GLM shape; the others make the intercept block
# wider so nothing silently depends on it being a single row.
_LAYOUTS = [
    pytest.param(_layout(1, 3), id="intercept-1-coef-3"),
    pytest.param(_layout(2, 5), id="intercept-2-coef-5"),
    pytest.param(_layout(3, 3), id="intercept-3-coef-3"),
]
# The same, with the coefficients split over two leaves.
_TRI_LEAF_LAYOUTS = [
    pytest.param(_layout(1, 3, 2), id="intercept-1-coef-3-2"),
    pytest.param(_layout(2, 2, 3), id="intercept-2-coef-2-3"),
]
_SEEDS = [0, 1, 2]


# --- Property lattice ---


@pytest.mark.parametrize(
    "prop, expected",
    [
        (PositiveDefinite, PositiveSemiDefinite),
        (PositiveSemiDefinite, PositiveSemiDefinite),
        (NegativeDefinite, Symmetric),
        (Symmetric, Symmetric),
        (General, General),
    ],
)
def test_weaken_property(prop, expected):
    """Zeroing out directions costs strictness and keeps the sign."""
    assert weaken_property(prop) is expected


@pytest.mark.parametrize("prop", _ALL_PROPERTIES)
def test_weaken_property_is_idempotent(prop):
    """Nothing is left to weaken after one pass: no strictness survives it."""
    assert weaken_property(weaken_property(prop)) is weaken_property(prop)


@pytest.mark.parametrize("prop", _ALL_PROPERTIES)
def test_weaken_property_implied_by_original(prop):
    """The weakened property must be one the original already implies."""
    assert weaken_property(prop) in _expand_property(prop)


# --- Tags used by both the algebra and the numerical tests ---

# A loss that curves on the intercept: its intercept block is the sum of the per-sample
# weights. Singular elsewhere, since a rank deficient design leaves coefficients flat.
_LOSS_DEFINITE_ON_INTERCEPT = HessianTag(
    structure=Full, property=PositiveSemiDefinite, definite_on=_INTERCEPT
)
# The same, for a loss whose second derivative can go negative (a link breaking convexity).
_LOSS_INDEFINITE = HessianTag(
    structure=Full, property=Symmetric, definite_on=_INTERCEPT
)
# A loss that claims nothing about single parameters, e.g. a softmax loss, which is flat
# along a shift of all the intercepts at once.
_LOSS_NO_CLAIM = HessianTag(structure=Full, property=PositiveSemiDefinite)
_LOSS_DEFINITE_ON_COEF = HessianTag(
    structure=Full, property=PositiveSemiDefinite, definite_on=_COEF
)
# A loss with every feature masked out: exactly zero on the coefficients, curved on the
# intercept. The only tag here that carries both claims at once.
_LOSS_ZERO_ON_COEF = HessianTag(
    structure=Full,
    property=PositiveSemiDefinite,
    flat_on=_COEF,
    definite_on=_INTERCEPT,
)
# A ridge penalty skipping the intercept: zero curvature there, curved on the coefficients.
_PENALTY_FLAT_ON_INTERCEPT = HessianTag(
    structure=Diagonal, property=PositiveSemiDefinite, flat_on=_INTERCEPT
)
# The same penalty with a zero strength: flat everywhere.
_PENALTY_FLAT_EVERYWHERE = HessianTag(
    structure=Diagonal, property=PositiveSemiDefinite, flat_on=_INTERCEPT | _COEF
)
# A penalty reaching every parameter with a positive strength: definite on its own.
_PENALTY_COVERS_ALL = HessianTag(
    structure=Diagonal, property=PositiveDefinite, flat_on=frozenset()
)

_TAGS = {
    "loss-definite-on-intercept": _LOSS_DEFINITE_ON_INTERCEPT,
    "loss-indefinite": _LOSS_INDEFINITE,
    "loss-no-claim": _LOSS_NO_CLAIM,
    "loss-definite-on-coef": _LOSS_DEFINITE_ON_COEF,
    "loss-zero-on-coef": _LOSS_ZERO_ON_COEF,
    "penalty-flat-on-intercept": _PENALTY_FLAT_ON_INTERCEPT,
    "penalty-flat-everywhere": _PENALTY_FLAT_EVERYWHERE,
    "penalty-covers-all": _PENALTY_COVERS_ALL,
}

_CASES = [
    pytest.param(
        _LOSS_DEFINITE_ON_INTERCEPT,
        _PENALTY_FLAT_ON_INTERCEPT,
        PositiveDefinite,
        id="penalty-flat-where-loss-curves",
    ),
    pytest.param(
        _LOSS_ZERO_ON_COEF,
        _PENALTY_FLAT_ON_INTERCEPT,
        PositiveDefinite,
        id="each-term-flat-where-the-other-curves",
    ),
    pytest.param(
        _LOSS_DEFINITE_ON_INTERCEPT,
        _PENALTY_FLAT_EVERYWHERE,
        PositiveSemiDefinite,
        id="zero-strength-penalty-flat-everywhere",
    ),
    pytest.param(
        _LOSS_NO_CLAIM,
        _PENALTY_FLAT_ON_INTERCEPT,
        PositiveSemiDefinite,
        id="loss-claims-nothing-where-penalty-is-flat",
    ),
    pytest.param(
        _LOSS_INDEFINITE,
        _PENALTY_FLAT_ON_INTERCEPT,
        Symmetric,
        id="indefinite-loss-is-not-promoted",
    ),
    pytest.param(
        _LOSS_DEFINITE_ON_INTERCEPT,
        _LOSS_DEFINITE_ON_COEF,
        PositiveSemiDefinite,
        id="two-curved-blocks-without-a-flat-one",
    ),
    pytest.param(
        _LOSS_NO_CLAIM,
        _PENALTY_COVERS_ALL,
        PositiveDefinite,
        id="penalty-covering-everything-needs-no-claim",
    ),
]

_WEAKER_CASES = [case for case in _CASES if case.values[2] is not PositiveDefinite]


# --- Tags for a pytree design, whose coefficients sit on more than one leaf ---

# A loss curving on the intercept and on the first group of features, saying nothing about
# the second. Its curved parameters are then strictly more than a penalty is flat on, which
# is the one relation a two-leaf tree cannot express.
_LOSS_DEFINITE_ON_INTERCEPT_AND_GROUP = HessianTag(
    structure=Full,
    property=PositiveSemiDefinite,
    definite_on=_INTERCEPT | _COEF,
)
# A ridge penalty that skips the intercept and also leaves the second group of features
# unpenalized, so it is flat on both and curves only on the first group.
_PENALTY_FLAT_ON_INTERCEPT_AND_GROUP = HessianTag(
    structure=Diagonal,
    property=PositiveSemiDefinite,
    flat_on=_INTERCEPT | _SECOND_COEF,
)

_TRI_LEAF_TAGS = {
    "loss-definite-on-intercept-and-group": _LOSS_DEFINITE_ON_INTERCEPT_AND_GROUP,
    "penalty-flat-on-intercept-and-group": _PENALTY_FLAT_ON_INTERCEPT_AND_GROUP,
}

_TRI_LEAF_CASES = [
    pytest.param(
        _LOSS_DEFINITE_ON_INTERCEPT_AND_GROUP,
        _PENALTY_FLAT_ON_INTERCEPT,
        PositiveDefinite,
        id="loss-curves-on-more-than-the-penalty-is-flat-on",
    ),
    pytest.param(
        _LOSS_DEFINITE_ON_INTERCEPT_AND_GROUP,
        _PENALTY_FLAT_ON_INTERCEPT_AND_GROUP,
        PositiveSemiDefinite,
        id="penalty-flat-on-a-group-the-loss-says-nothing-about",
    ),
]

_TRI_LEAF_WEAKER_CASES = [
    case for case in _TRI_LEAF_CASES if case.values[2] is not PositiveDefinite
]


# --- Pairing tags with the layouts that hold the parameters they name ---
#
# A tag naming a second group of coefficients has nothing to point at in a two-leaf layout,
# so the pairs are built per number of leaves instead of as a product of every combination.


def _cases_with_layouts(cases: list, layouts: list) -> list:
    """Every case against every layout, keeping the ids of both."""
    return [
        pytest.param(*case.values, layout.values[0], id=f"{case.id}-{layout.id}")
        for case in cases
        for layout in layouts
    ]


def _tags_with_layouts(tags: dict, layouts: list) -> list:
    """Every tag against every layout, keeping the tag's name and the layout's id."""
    return [
        pytest.param(tag, layout.values[0], id=f"{name}-{layout.id}")
        for name, tag in tags.items()
        for layout in layouts
    ]


_TAG_CASES = _tags_with_layouts(_TAGS, _LAYOUTS) + _tags_with_layouts(
    _TRI_LEAF_TAGS, _TRI_LEAF_LAYOUTS
)
_SUM_CASES = _cases_with_layouts(_CASES, _LAYOUTS) + _cases_with_layouts(
    _TRI_LEAF_CASES, _TRI_LEAF_LAYOUTS
)
_NECESSITY_CASES = _cases_with_layouts(_WEAKER_CASES, _LAYOUTS) + _cases_with_layouts(
    _TRI_LEAF_WEAKER_CASES, _TRI_LEAF_LAYOUTS
)


# --- Tag algebra ---


@pytest.mark.parametrize("t1, t2, expected", _CASES + _TRI_LEAF_CASES)
def test_combine_hessian_tags_property(t1, t2, expected):
    """The sum is positive definite when one term is flat only where the other curves.

    ``two-curved-blocks-without-a-flat-one`` is why a zero block is required: two terms
    curving on one block each can still add up to something singular, as
    ``A = B = [[1, 1], [1, 1]]`` does. ``loss-curves-on-more-than-the-penalty-is-flat-on``
    is why the check is a subset rather than an equality: with the coefficients on two
    leaves the loss can curve on more parameters than the penalty is flat on.
    """
    assert combine_hessian_tags(t1, t2).property is expected
    assert combine_hessian_tags(t2, t1).property is expected


def test_combine_hessian_tags_propagates_claims():
    """Curvature is zero only where both are zero, and a curved parameter stays curved."""
    both_flat = HessianTag(
        structure=Full,
        property=PositiveSemiDefinite,
        flat_on=_INTERCEPT | _COEF,
        definite_on=_INTERCEPT,
    )
    combined = combine_hessian_tags(both_flat, _PENALTY_FLAT_ON_INTERCEPT)
    assert combined.flat_on == _INTERCEPT
    assert combined.definite_on == _INTERCEPT


def test_combine_hessian_tags_without_claims_stays_unclaimed():
    """Tags that say nothing about single parameters combine into one that says nothing."""
    combined = combine_hessian_tags(_LOSS_NO_CLAIM, _LOSS_NO_CLAIM)
    assert combined.flat_on is None
    assert combined.definite_on is None


# --- Building Hessians that satisfy a tag ---


def _size(layout: Layout) -> int:
    """Side length of a Hessian over these parameters.

    Parameters
    ----------
    layout :
        Which rows and columns each parameter occupies.

    Returns
    -------
    :
        Total number of rows, i.e. the number of scalar parameters.
    """
    return sum(len(indices) for indices in layout.values())


def _indices(layout: Layout, leaves) -> np.ndarray:
    """Rows and columns belonging to ``leaves``, ready to slice a block out of a Hessian.

    Parameters
    ----------
    layout :
        Which rows and columns each parameter occupies.
    leaves :
        Leaf ids to look up, as a tag spells them in ``flat_on`` or ``definite_on``. An
        empty set gives an empty index array, so slicing with it selects nothing.

    Returns
    -------
    :
        Indices of those parameters, in increasing order of leaf id.
    """
    if not leaves:
        return np.array([], dtype=int)
    return np.concatenate([layout[leaf] for leaf in sorted(leaves)])


def _spectrum(size: int) -> np.ndarray:
    """Eigenvalues spanning a fixed range, so every generated matrix has the same scale.

    Parameters
    ----------
    size :
        How many eigenvalues are needed.

    Returns
    -------
    :
        Values from ``_SMALLEST_EIGENVALUE`` to ``_LARGEST_EIGENVALUE``, the largest always
        present so the thresholds derived from it hold whatever the size.
    """
    if size == 1:
        return np.array([_LARGEST_EIGENVALUE])
    return np.linspace(_SMALLEST_EIGENVALUE, _LARGEST_EIGENVALUE, size)


def _orthogonal(rng, size: int, first_column: np.ndarray | None = None) -> np.ndarray:
    """Random orthogonal matrix, with a prescribed unit first column when given.

    Parameters
    ----------
    rng :
        Source of the random rotation.
    size :
        Side length of the matrix.
    first_column :
        Direction to use as the first column, normalised here so the caller need not. The
        remaining columns are a random orthonormal basis of what is left. ``None`` leaves
        every column random.

    Returns
    -------
    :
        An orthogonal matrix whose first column is ``first_column`` when one was given.
    """
    m = rng.normal(size=(size, size))
    if first_column is not None:
        m[:, 0] = first_column / np.linalg.norm(first_column)
    q, r = np.linalg.qr(m)
    # QR fixes the first column up to a sign; pin it so q[:, 0] is +first_column
    return q * np.sign(np.diag(r))


def _with_spectrum(
    eigenvalues: np.ndarray, rng, flat_direction: np.ndarray | None = None
) -> np.ndarray:
    """Symmetric matrix with exactly this spectrum, built as ``Q diag(eigenvalues) Q.T``.

    Prescribing the spectrum is what makes the thresholds in this file derivable: the scale
    and the conditioning of every generated matrix are known before anything is measured.

    Parameters
    ----------
    eigenvalues :
        The spectrum the result is to have, exactly. Its length sets the matrix size.
    rng :
        Source of the random rotation ``Q``.
    flat_direction :
        Direction to pair with ``eigenvalues[0]``, by making it the first column of ``Q``.
        With a first eigenvalue of zero it becomes a null direction, which is what the name
        refers to and what lets two matrices be given the same one; with a negative first
        eigenvalue it is instead the direction along which the matrix curves downwards.
        ``None`` leaves the choice to the random rotation.

    Returns
    -------
    :
        A symmetric matrix with the requested spectrum.
    """
    q = _orthogonal(rng, len(eigenvalues), flat_direction)
    return q @ np.diag(eigenvalues) @ q.T


def _off_block_direction(layout: Layout, leaves, rng) -> np.ndarray:
    """A random direction with no component on ``leaves``.

    Parameters
    ----------
    layout :
        Which rows and columns each parameter occupies.
    leaves :
        Leaf ids to avoid. An empty set or ``None`` places no restriction.
    rng :
        Source of the random direction.

    Returns
    -------
    :
        A vector that is zero on those parameters and random on the others.
    """
    if not leaves:
        return rng.normal(size=_size(layout))
    direction = np.zeros(_size(layout))
    off = _indices(layout, set(layout) - set(leaves))
    direction[off] = rng.normal(size=off.size)
    return direction


def _hessian_for(
    tag: HessianTag, layout: Layout, rng, flat_direction: np.ndarray | None = None
) -> np.ndarray:
    """A Hessian consistent with ``tag``, made flat along ``flat_direction`` if allowed.

    Whether it is allowed is not decided here: the caller checks the result against the tag
    with ``_satisfies``, so a direction the claims forbid shows up as a matrix that fails
    its own tag rather than as a silently invalid example.

    A ``flat_on`` tag gives a matrix with nothing at all outside its curved block, and that
    is forced rather than chosen: zero curvature on those parameters means a zero diagonal
    block, and a positive semidefinite matrix with a zero diagonal block has zero rows and
    columns there too. Coupling inside the curved block is still there whenever the
    structure is not ``Diagonal``, and the other branch is dense throughout.

    Parameters
    ----------
    tag :
        The claims the result has to satisfy.
    layout :
        Which rows and columns each parameter occupies.
    rng :
        Source of the random rotation.
    flat_direction :
        Direction to make the matrix flat along, when the tag permits it. Passing the same
        one to two tags is how their null spaces are made to meet, which is the only way a
        sum of two positive semidefinite matrices comes out singular. ``None`` asks for no
        particular direction, except for an indefinite tag, where one is chosen off the
        block the tag claims curves.

    Returns
    -------
    :
        A Hessian over the parameters in ``layout``, symmetric by construction.
    """
    size = _size(layout)
    if tag.flat_on is not None:
        curved = _indices(layout, set(layout) - set(tag.flat_on))
        hess = np.zeros((size, size))
        if curved.size:
            spectrum = _spectrum(curved.size)
            block = (
                np.diag(spectrum)
                if tag.structure is Diagonal
                else _with_spectrum(spectrum, rng)
            )
            hess[np.ix_(curved, curved)] = block
        return hess
    spectrum = _spectrum(size)
    # a loss is singular where the design is rank deficient, and dips negative when the
    # link breaks convexity
    spectrum[0] = _NEGATIVE_EIGENVALUE if tag.property is Symmetric else 0.0
    if flat_direction is None and tag.property is Symmetric:
        # left to a random rotation, the negative direction can land on the block the tag
        # says curves, which would contradict the tag; keep it off that block
        flat_direction = _off_block_direction(layout, tag.definite_on, rng)
    return _with_spectrum(spectrum, rng, flat_direction=flat_direction)


def _smallest_relative_eigenvalue(hess: np.ndarray) -> float:
    """Smallest eigenvalue against the largest in magnitude, so thresholds are scale-free.

    Parameters
    ----------
    hess :
        A symmetric matrix.

    Returns
    -------
    :
        The smallest eigenvalue divided by the largest in magnitude, which is negative for
        an indefinite matrix, around zero for a singular one, and positive for a definite
        one. The divisor is floored at one so a matrix of small entries is not rescaled up.
    """
    eigenvalues = np.linalg.eigvalsh(hess)
    return eigenvalues[0] / max(1.0, abs(eigenvalues).max())


def _satisfies(tag: HessianTag, layout: Layout, hess: np.ndarray) -> bool:
    """Whether ``hess`` really has every property ``tag`` claims.

    Every generated matrix goes through here before it is used, so a test cannot pass on an
    example that does not match its own tag.

    Parameters
    ----------
    tag :
        The claims to check: symmetry, the structure, the overall definiteness, zero rows on
        ``flat_on`` with a definite block on the rest, and a definite ``definite_on`` block.
    layout :
        Which rows and columns each parameter occupies.
    hess :
        The matrix to check.

    Returns
    -------
    :
        Whether every claim holds, to within ``_TOL`` on the eigenvalues.
    """
    if not np.allclose(hess, hess.T):
        return False
    if tag.structure is Diagonal and not np.allclose(hess, np.diag(np.diag(hess))):
        return False
    smallest = _smallest_relative_eigenvalue(hess)
    if tag.property is PositiveDefinite and smallest <= _TOL:
        return False
    if tag.property is PositiveSemiDefinite and smallest < -_TOL:
        return False
    if tag.flat_on is not None:
        flat = _indices(layout, tag.flat_on)
        if flat.size and not np.allclose(hess[flat, :], 0.0):
            return False
        curved = _indices(layout, set(layout) - set(tag.flat_on))
        if (
            curved.size
            and _smallest_relative_eigenvalue(hess[np.ix_(curved, curved)]) <= _TOL
        ):
            return False
    if tag.definite_on is not None:
        curved = _indices(layout, tag.definite_on)
        if _smallest_relative_eigenvalue(hess[np.ix_(curved, curved)]) <= _TOL:
            return False
    return True


def _flat_directions(layout: Layout, rng) -> list[np.ndarray | None]:
    """Directions worth making a Hessian flat along: one per way the blocks can be hit.

    Parameters
    ----------
    layout :
        Which rows and columns each parameter occupies.
    rng :
        Source of the random directions among them.

    Returns
    -------
    :
        Directions to hand to ``_hessian_for``: none at all, then for every parameter a
        single one of its coordinates and a random direction confined to it, then one
        spread over everything and one random over everything. A tag that forbids being
        flat along one of them rejects it later, in ``_satisfies``.
    """
    size = _size(layout)
    unit = np.eye(size)
    directions = [None]
    for indices in layout.values():
        inside = np.zeros(size)
        inside[indices] = rng.normal(size=len(indices))
        directions += [unit[indices[0]], inside]
    return directions + [np.ones(size), rng.normal(size=size)]


def _honest_sums(
    t1: HessianTag, t2: HessianTag, layout: Layout, seed: int, repeats: int = 10
) -> list[float]:
    """Smallest relative eigenvalue of ``A + B`` over pairs that satisfy both tags.

    Both terms are made flat along the same direction, which is what puts that direction in
    the null space of each. Two random matrices never share a null direction, so without
    this the singular and indefinite cases would never come up.

    Parameters
    ----------
    t1 :
        Tag of the first term, typically a loss.
    t2 :
        Tag of the second term, typically a penalty.
    layout :
        Which rows and columns each parameter occupies.
    seed :
        Seed for the random rotations and directions, so a failure can be reproduced.
    repeats :
        How many pairs to build per direction, since the rotations differ each time.

    Returns
    -------
    :
        One entry per pair that satisfied both tags, holding the smallest eigenvalue of the
        sum relative to its largest. Pairs where either matrix contradicted its own tag are
        left out, so a caller has to check the list is not empty before reading anything
        into it.
    """
    rng = np.random.default_rng(seed)
    out = []
    for direction, _ in product(_flat_directions(layout, rng), range(repeats)):
        first = _hessian_for(t1, layout, rng, direction)
        second = _hessian_for(t2, layout, rng, direction)
        if _satisfies(t1, layout, first) and _satisfies(t2, layout, second):
            summed = first + second
            assert np.allclose(
                summed, summed.T
            ), "a sum of symmetric matrices must be symmetric"
            out.append(_smallest_relative_eigenvalue(summed))
    return out


# --- Numerical checks ---


@pytest.mark.parametrize("tag, layout", _TAG_CASES)
def test_generated_hessian_satisfies_its_tag(tag, layout):
    """Guards the tests below: they are vacuous if the generator ignores the claims."""
    rng = np.random.default_rng(0)
    assert _satisfies(tag, layout, _hessian_for(tag, layout, rng))


@pytest.mark.parametrize("t1, t2, expected, layout", _SUM_CASES)
@pytest.mark.parametrize("seed", _SEEDS)
def test_combined_tag_holds_of_the_actual_sum(t1, t2, expected, layout, seed):
    """Whatever the combined tag claims must be true of every honest pair of Hessians."""
    smallest = _honest_sums(t1, t2, layout, seed)
    assert smallest, "no pair satisfied both tags, so nothing was checked"
    if expected is PositiveDefinite:
        assert min(smallest) > _TOL
    elif expected is PositiveSemiDefinite:
        assert min(smallest) >= -_TOL
    else:
        # Symmetric claims nothing beyond symmetry, asserted in ``_honest_sums``; that the
        # sum really can be indefinite is the job of the necessity test below.
        assert expected is Symmetric


@pytest.mark.parametrize("t1, t2, expected, layout", _NECESSITY_CASES)
def test_weaker_verdict_is_necessary(t1, t2, expected, layout):
    """A pair the tags allow must exist that would make a stronger verdict false.

    Without this the rule could over-promote and still pass, because singular and
    indefinite sums only show up when the two null spaces are made to coincide on purpose,
    which random matrices never do.
    """
    smallest = min(_honest_sums(t1, t2, layout, seed=0))
    if expected is PositiveSemiDefinite:
        assert smallest <= _TOL, "never singular, so positive definite was available"
    else:
        assert smallest <= _INDEFINITE_BOUND


@pytest.mark.parametrize("layout", _LAYOUTS + _TRI_LEAF_LAYOUTS)
def test_promotion_rules_out_its_own_counterexample(layout):
    """A loss curving on the intercept cannot be flat along an intercept direction.

    This is what makes the promotion sound: the penalty's null space sits inside the
    intercept, and the loss's claim to curve there forbids sharing it.
    """
    rng = np.random.default_rng(0)
    inside_intercept = np.eye(_size(layout))[layout[_INTERCEPT_LEAF][0]]
    hess = _hessian_for(_LOSS_DEFINITE_ON_INTERCEPT, layout, rng, inside_intercept)
    assert not _satisfies(_LOSS_DEFINITE_ON_INTERCEPT, layout, hess)
