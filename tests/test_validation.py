from contextlib import nullcontext as does_not_raise

import jax.numpy as jnp
import pytest

from nemos import validation


@pytest.mark.parametrize(
    "tree, expectation",
    [
        (jnp.array([[1], [2], [3]]), does_not_raise()),
        (
            jnp.array([[1], [2], [jnp.nan]]),
            pytest.raises(ValueError, match="The provided trees contain Nans"),
        ),
        (
            jnp.array([[1], [jnp.inf], [3]]),
            pytest.raises(ValueError, match="The provided trees contain Infs"),
        ),
        (
            jnp.array([[1], [jnp.inf], [jnp.nan]]),
            pytest.raises(ValueError, match="The provided trees contain Infs and Nans"),
        ),
    ],
)
def test_error_invalid_entry(tree, expectation):
    """Test validation of trees generates the correct exceptions."""
    valid_data = jnp.array([[1], [2], [3]])
    with expectation:
        validation.error_invalid_entry(valid_data, tree)


@pytest.mark.parametrize(
    "fill_val, expectation",
    [
        (0, does_not_raise()),
        (
            jnp.inf,
            pytest.raises(
                ValueError, match="At least a NaN or an Inf at all sample points"
            ),
        ),
        (
            jnp.nan,
            pytest.raises(
                ValueError, match="At least a NaN or an Inf at all sample points"
            ),
        ),
    ],
)
def test_error_all_invalid(fill_val, expectation):
    """Test error when all samples have at least a nan."""
    inp = jnp.array(
        [[fill_val, 0, 0], [0, 0, fill_val], [0, 0, fill_val], [0, fill_val, 0]]
    )
    valid_inp = jnp.zeros((inp.shape[0], 1))
    # some structured tree
    trees = [[[valid_inp], inp], valid_inp]
    with expectation:
        validation.error_all_invalid(*trees)


# Sample pytree for testing
sample_pytree = {"a": jnp.array([1, 2, 3]), "b": {"c": jnp.array([[1], [2]])}}

sample_pytree_same_shape = {
    "a": jnp.array([[1, 2, 3]]),
    "b": {"c": jnp.array([[1, 2]])},
}


@pytest.mark.parametrize(
    "x, expected_len, err_message, expectation",
    [
        ([1, 2], 2, "", does_not_raise()),
        ([1, 2, 3], 2, "Length does not match expected.", pytest.raises(ValueError)),
    ],
)
def test_check_length(x, expected_len, err_message, expectation):
    with expectation:
        validation.check_length(x, expected_len, err_message)


@pytest.mark.parametrize(
    "tree, err_message, data_type, expectation",
    [
        (sample_pytree, "", jnp.float32, does_not_raise()),
        (
            [1, 2, "invalid"],
            "Conversion failed.",
            jnp.float32,
            pytest.raises(TypeError),
        ),
    ],
)
def test_convert_tree_leaves_to_jax_array(tree, err_message, data_type, expectation):
    with expectation:
        validation.convert_tree_leaves_to_jax_array(tree, err_message, data_type)


@pytest.mark.parametrize(
    "tree, expected_dim, err_message, expectation",
    [
        (sample_pytree_same_shape, 2, "", does_not_raise()),
        (
            {"a": jnp.array([[1], [2]]), "b": jnp.array([1, 2, 3])},
            1,
            "Dimensionality mismatch.",
            pytest.raises(ValueError),
        ),
    ],
)
def test_check_tree_leaves_dimensionality(tree, expected_dim, err_message, expectation):
    with expectation:
        validation.check_tree_leaves_dimensionality(tree, expected_dim, err_message)


@pytest.mark.parametrize(
    "arrays, axis, err_message, expectation",
    [
        ((jnp.array([1, 2]), jnp.array([3, 4])), 0, "", does_not_raise()),
        (
            (jnp.array([1, 2]), jnp.array([[3, 4, 3], [5, 6, 1]])),
            0,
            "",
            does_not_raise(),
        ),
        (
            (jnp.array([1, 2, 3]), jnp.array([[3, 4, 3], [5, 6, 1]])),
            0,
            "Shape mismatch on axis.",
            pytest.raises(ValueError),
        ),
    ],
)
def test_check_same_shape_on_axis(arrays, axis, err_message, expectation):
    with expectation:
        validation.check_same_shape_on_axis(*arrays, axis=axis, err_message=err_message)


@pytest.mark.parametrize(
    "tree, array, axis, err_message, expectation",
    [
        (sample_pytree_same_shape, jnp.array([[1, 2, 3]]), 0, "", does_not_raise()),
        (sample_pytree, jnp.array([1, 2, 3]), 0, "", pytest.raises(ValueError)),
        (
            {"a": jnp.array([1, 2, 3]), "b": {"c": jnp.array([[1], [2]])}},
            jnp.array([1, 2]),
            0,
            "",
            pytest.raises(ValueError),
        ),
    ],
)
def test_check_array_shape_match_tree(tree, array, axis, err_message, expectation):
    with expectation:
        validation.check_array_shape_match_tree(tree, array, axis, err_message)


@pytest.mark.parametrize(
    "tree_1, tree_2, axis_1, axis_2, err_message, expectation",
    [
        (sample_pytree, sample_pytree, 0, 0, "", does_not_raise()),
        (
            {"a": jnp.array([1, 2, 3])},
            {"b": jnp.array([[1], [2]])},
            0,
            1,
            "Axis consistency mismatch.",
            pytest.raises(ValueError),
        ),
    ],
)
def test_check_tree_axis_consistency(
    tree_1, tree_2, axis_1, axis_2, err_message, expectation
):
    with expectation:
        validation.check_tree_axis_consistency(
            tree_1, tree_2, axis_1, axis_2, err_message
        )


@pytest.mark.parametrize(
    "tree_1, tree_2, err_message, expectation",
    [
        (sample_pytree, sample_pytree, "", does_not_raise()),
        (
            sample_pytree,
            {"a": [1, 2, 3], "b": {"c": [1, 2]}},
            "Tree structure mismatch.",
            pytest.raises(TypeError),
        ),
    ],
)
def test_check_tree_structure(tree_1, tree_2, err_message, expectation):
    with expectation:
        validation.check_tree_structure(tree_1, tree_2, err_message)
