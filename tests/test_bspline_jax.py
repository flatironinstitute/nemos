"""Equivalence of the two B-spline implementations.

``bspline`` evaluates the basis with scipy's ``splev`` and ``bspline_jax`` with a
pure-jax recursion. The second is used whenever the samples are traced, so the
two must agree on every input the basis can be handed.
"""

import jax
import numpy as np
import pytest

import nemos.basis as basis
from nemos.basis._spline_basis import bspline, bspline_jax

# the scipy path builds float64 arrays, so the comparison needs x64
pytestmark = pytest.mark.requires_x64

ORDERS = (2, 3, 4, 5, 6)

# ``splev`` accepts 0 <= der <= k, with k = order - 1 the degree
ORDER_DER = [(order, der) for order in ORDERS for der in range(order)]


def open_knots(n_basis, order):
    """Knot vector of ``BSplineEval``, boundary knots repeated ``order`` times."""
    return basis.BSplineEval(n_basis, order=order)._generate_knots(is_cyclic=False)


def cyclic_knots(n_basis, order):
    """Knot vector of ``CyclicBSplineEval``, extended below zero and not repeated."""
    knot_locs = np.unique(
        basis.CyclicBSplineEval(n_basis, order=order)._generate_knots(is_cyclic=True)
    )
    nk = knot_locs.shape[0]
    return np.hstack(
        (knot_locs[0] - knot_locs[-1] + knot_locs[nk - order : nk - 1], knot_locs)
    )


SAMPLES = {
    "grid": np.linspace(0, 1, 41),
    "random": np.sort(np.random.default_rng(0).uniform(0, 1, 23)),
    "with-nan": np.concatenate([[np.nan], np.linspace(0, 1, 19), [np.nan]]),
    "out-of-range": np.linspace(-0.4, 1.4, 31),
    "single-point": np.array([0.5]),
}


def assert_same_basis(expected, actual):
    """Same shape, same NaN pattern, same values."""
    assert expected.shape == actual.shape
    np.testing.assert_array_equal(
        np.isnan(expected), np.isnan(actual), err_msg="NaN patterns differ"
    )
    np.testing.assert_allclose(expected, actual, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("order, der", ORDER_DER)
@pytest.mark.parametrize("n_basis", [6, 12])
@pytest.mark.parametrize("sample_label", list(SAMPLES))
def test_matches_splev_open_knots(order, der, n_basis, sample_label):
    """The jax recursion reproduces ``splev`` on the standard knot vector."""
    knots = open_knots(n_basis, order)
    sample_pts = SAMPLES[sample_label]
    expected = bspline(sample_pts.copy(), knots.copy(), order=order, der=der)
    actual = np.asarray(bspline_jax(sample_pts, knots, order=order, der=der))
    assert_same_basis(expected, actual)
    assert actual.shape == (sample_pts.shape[0], knots.shape[0] - order)


@pytest.mark.parametrize("order, der", ORDER_DER)
@pytest.mark.parametrize("n_basis", [8, 12])
@pytest.mark.parametrize("sample_label", list(SAMPLES))
def test_matches_splev_cyclic_knots(order, der, n_basis, sample_label):
    """The knot vector of the cyclic basis is not the repeated-boundary one."""
    knots = cyclic_knots(n_basis, order)
    sample_pts = SAMPLES[sample_label]
    expected = bspline(sample_pts.copy(), knots.copy(), order=order, der=der)
    actual = np.asarray(bspline_jax(sample_pts, knots, order=order, der=der))
    assert_same_basis(expected, actual)


@pytest.mark.parametrize("order, der", ORDER_DER)
def test_matches_splev_unsorted_knots(order, der):
    """Both implementations sort the knots before evaluating."""
    knots = open_knots(10, order)
    shuffled = np.random.default_rng(0).permutation(knots)
    expected = bspline(shuffled.copy(), knots.copy(), order=order, der=der)
    actual = np.asarray(bspline_jax(SAMPLES["grid"], shuffled, order=order, der=der))
    assert_same_basis(
        bspline(SAMPLES["grid"].copy(), knots.copy(), order=order, der=der), actual
    )
    assert expected is not None  # the scipy call must not raise on unsorted input


@pytest.mark.parametrize("order, der", ORDER_DER)
def test_jit_matches_eager(order, der):
    """Tracing ``bspline_jax`` does not change its output."""
    knots = open_knots(10, order)
    sample_pts = SAMPLES["with-nan"]
    eager = np.asarray(bspline_jax(sample_pts, knots, order=order, der=der))
    jitted = np.asarray(
        jax.jit(lambda x: bspline_jax(x, knots, order=order, der=der))(sample_pts)
    )
    assert_same_basis(eager, jitted)


@pytest.mark.parametrize("order", ORDERS)
def test_bspline_dispatches_to_jax_when_traced(order):
    """``bspline`` swaps ``splev`` for the jax recursion on traced samples."""
    knots = open_knots(10, order)
    sample_pts = SAMPLES["grid"]
    expected = bspline(sample_pts.copy(), knots.copy(), order=order, der=0)
    traced = np.asarray(
        jax.jit(lambda x: bspline(x, knots, order=order, der=0))(sample_pts)
    )
    assert_same_basis(expected, traced)


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("offset", [0, 1, 4])
def test_der_above_degree_rejected_by_both(order, offset):
    """``bspline_jax`` enforces the same ``der`` range as ``splev``."""
    knots = open_knots(10, order)
    der = order + offset
    with pytest.raises(ValueError, match="0 <= der <= order - 1"):
        bspline_jax(SAMPLES["grid"], knots, order=order, der=der)
    with pytest.raises(ValueError, match="der"):
        bspline(SAMPLES["grid"].copy(), knots.copy(), order=order, der=der)


@pytest.mark.parametrize("der", [-1, -3])
def test_negative_der_rejected(der):
    """Negative derivatives are rejected rather than silently mishandled."""
    with pytest.raises(ValueError, match="0 <= der <= order - 1"):
        bspline_jax(SAMPLES["grid"], open_knots(10, 4), order=4, der=der)


@pytest.mark.parametrize("order", ORDERS)
def test_out_of_knot_range_is_nan(order):
    """Samples outside the knots range, NaNs included, are filled with NaN."""
    knots = open_knots(10, order)
    sample_pts = np.array([np.nan, -0.5, 0.0, 0.5, 1.0, 1.5])
    actual = np.asarray(bspline_jax(sample_pts, knots, order=order))
    expected_nan = np.array([True, True, False, False, False, True])
    np.testing.assert_array_equal(np.isnan(actual).all(axis=1), expected_nan)
    np.testing.assert_array_equal(np.isnan(actual).any(axis=1), expected_nan)


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("n_basis", [6, 12])
def test_partition_of_unity(order, n_basis):
    """A B-spline basis of any degree sums to one over the knots range."""
    knots = open_knots(n_basis, order)
    actual = np.asarray(bspline_jax(np.linspace(0, 1, 101), knots, order=order))
    np.testing.assert_allclose(actual.sum(axis=1), 1.0, rtol=1e-10)


@pytest.mark.parametrize("order", [3, 4, 5, 6])
@pytest.mark.parametrize("der", [1, 2])
def test_derivative_matches_finite_difference(order, der):
    """The derivative recursion is the derivative, not only a match to ``splev``."""
    knots = open_knots(10, order)
    step = 1e-6
    sample_pts = np.linspace(0.2, 0.8, 25)
    lower = np.asarray(bspline_jax(sample_pts - step, knots, order=order, der=der - 1))
    upper = np.asarray(bspline_jax(sample_pts + step, knots, order=order, der=der - 1))
    actual = np.asarray(bspline_jax(sample_pts, knots, order=order, der=der))
    np.testing.assert_allclose((upper - lower) / (2 * step), actual, atol=1e-4)
