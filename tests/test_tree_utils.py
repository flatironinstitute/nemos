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
