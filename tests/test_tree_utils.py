import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nemos import tree_utils


@pytest.mark.parametrize(
    "array, expected",
    [
        (jnp.array([[1, 2], [3, 4]]), jnp.array([True, True])),
        (jnp.array([[1, jnp.inf], [3, 4]]), jnp.array([False, True])),
        (jnp.array([[1, 2], [jnp.inf, jnp.inf]]), jnp.array([True, False])),
    ],
)
def test_get_not_inf(array, expected):
    """Test _get_not_inf function for correctly identifying non-infinite values."""
    assert jnp.array_equal(tree_utils._get_not_inf(array), expected)


@pytest.mark.parametrize(
    "array, expected",
    [
        (jnp.array([[1, 2], [3, 4]]), jnp.array([True, True])),
        (jnp.array([[1, jnp.nan], [3, 4]]), jnp.array([False, True])),
        (jnp.array([[1, 2], [jnp.nan, jnp.nan]]), jnp.array([True, False])),
    ],
)
def test_get_not_nan(array, expected):
    """Test _get_not_nan function for correctly identifying non-NaN values."""
    assert jnp.array_equal(tree_utils._get_not_nan(array), expected)


@pytest.mark.parametrize(
    "tree, expected_shape",
    [
        (jnp.array([[1], [2], [3], [4]]), 4),
        ({"x": {"y": jnp.array([1, 2, jnp.nan])}}, 3),
    ],
)
def test_check_valid_length(tree, expected_shape):
    """Test that validation of trees returns an array of the right first shape."""
    valid = tree_utils._get_valid_tree(tree)
    assert valid.shape[0] == expected_shape


@pytest.mark.parametrize(
    "tree",
    [(jnp.array([[1], [2], [3], [4]])), ({"x": {"y": jnp.array([1, 2, jnp.nan])}})],
)
def test_check_flat_array(tree):
    """Test that validation of trees returns an array of the right dimensionality."""
    valid = tree_utils._get_valid_tree(tree)
    assert valid.ndim == 1


@pytest.mark.parametrize(
    "tree, expected",
    [
        (
            {"a": jnp.array([[1, 2], [3, 4]]), "b": jnp.array([[5, 6], [7, 8]])},
            jnp.array([True, True]),
        ),
        (
            {"a": jnp.array([[1, 2], [jnp.nan, 4]]), "b": jnp.array([[5, 6], [7, 8]])},
            jnp.array([True, False]),
        ),
        (
            {
                "a": jnp.array([[1, jnp.nan], [3, 4]]),
                "b": jnp.array([[5, 6], [jnp.inf, 8]]),
            },
            jnp.array([False, False]),
        ),
    ],
)
def test_get_valid_tree(tree, expected):
    """Test _get_valid_tree function for filtering valid tree entries."""
    assert jnp.array_equal(tree_utils._get_valid_tree(tree), expected)


@pytest.mark.parametrize(
    "trees, expected",
    [
        (
            ({"a": jnp.array([[1, 2], [3, 4]])}, {"b": jnp.array([[5, 6], [7, 8]])}),
            jnp.array([True, True]),
        ),
        (
            (
                {"a": jnp.array([[1, 2], [3, 4]])},
                {"b": jnp.array([[5, 6], [jnp.nan, 8]])},
            ),
            jnp.array([True, False]),
        ),
        (
            (
                {"a": jnp.array([[1, jnp.nan], [3, 4]])},
                {"b": jnp.array([[4, 6], [7, jnp.inf]])},
            ),
            jnp.array([False, False]),
        ),
    ],
)
def test_get_valid_multitree(trees, expected):
    """Test get_valid_multitree function for filtering valid entries across multiple trees."""
    assert jnp.array_equal(tree_utils.get_valid_multitree(*trees), expected)


@pytest.mark.parametrize(
    "idx",
    [
        slice(2, 5),  # Slice indexing
        np.array([1, 3, 5]),  # Integer list indexing
        np.array(
            [True, False, True, False, True, False, True, False, True, False]
        ),  # Boolean array indexing
        (slice(1, 3), slice(0, 2)),  # Mixed indexing (simple example with slices)
    ],
)
def test_tree_slice(idx):
    mydict = {
        "array1": np.random.rand(10, 3),
        "array2": np.random.rand(10, 2),
        "array3": np.random.rand(10, 4),
        "array4": jnp.arange(30).reshape(10, 3),
    }
    result = tree_utils.tree_slice(mydict, idx)
    for key in mydict:
        expected = mydict[key][idx]
        assert jnp.all(result[key] == expected)


@pytest.mark.parametrize(
    "tree, expected",
    [
        (jnp.array([1.0, 2.0, 3.0]), True),
        (jnp.array([1.0, jnp.nan, 3.0]), False),
        (jnp.array([1.0, jnp.inf, 3.0]), False),
        (jnp.array([1.0, -jnp.inf, 3.0]), False),
        (
            {
                "a": jnp.array([1.0, 2.0]),
                "b": {"c": jnp.array([3.0, 4.0])},
            },
            True,
        ),
        (
            {
                "a": jnp.array([1.0, 2.0]),
                "b": {"c": jnp.array([3.0, jnp.nan])},
            },
            False,
        ),
        (
            {
                "a": jnp.array([1.0, jnp.inf]),
                "b": jnp.array([3.0, 4.0]),
            },
            False,
        ),
        (
            {
                "float": jnp.array([1.0, 2.0]),
                "integer": jnp.array([1, 2]),
                "boolean": jnp.array([True, False]),
            },
            True,
        ),
    ],
)
def test_tree_all_finite(tree, expected):
    """Test whether all elements across all PyTree leaves are finite."""
    result = tree_utils.tree_all_finite(tree)

    assert result.shape == ()
    assert jnp.array_equal(result, jnp.asarray(expected))


@pytest.mark.parametrize(
    "tree, expected",
    [
        ({"a": jnp.array([1.0, 2.0])}, True),
        ({"a": jnp.array([1.0, jnp.nan])}, False),
        ({"a": jnp.array([1.0, jnp.inf])}, False),
    ],
)
def test_tree_all_finite_jit(tree, expected):
    """Test that tree_all_finite works under JIT compilation."""
    result = jax.jit(tree_utils.tree_all_finite)(tree)

    assert result.shape == ()
    assert jnp.array_equal(result, jnp.asarray(expected))


def test_tree_all_finite_empty_tree():
    """Test that an empty PyTree is considered finite."""
    result = tree_utils.tree_all_finite({})

    assert result.shape == ()
    assert jnp.array_equal(result, jnp.asarray(True))


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.int32, "float32"])
def test_tree_astype_casts_every_leaf(dtype):
    trees = ({"a": jnp.ones(2)}, [np.zeros(3), jnp.arange(4)])
    out = tree_utils.tree_astype(*trees, dtype=dtype)
    assert len(out) == len(trees)
    leaves = jax.tree_util.tree_leaves(out)
    assert leaves and all(leaf.dtype == jnp.dtype(dtype) for leaf in leaves)


def test_tree_astype_always_returns_a_tuple():
    """One argument in, a one-element tuple out: no special case for a single tree."""
    (out,) = tree_utils.tree_astype(jnp.ones(3), dtype=jnp.float16)
    assert out.dtype == jnp.float16


def test_tree_astype_passes_none_through():
    """None is an empty pytree node, so optional arguments need no guard."""
    array, missing = tree_utils.tree_astype(jnp.ones(3), None, dtype=jnp.float16)
    assert array.dtype == jnp.float16
    assert missing is None


def test_tree_astype_default_dtype_is_a_noop():
    (out,) = tree_utils.tree_astype({"a": jnp.ones(2, dtype=jnp.int16)})
    assert out["a"].dtype == jnp.int16
