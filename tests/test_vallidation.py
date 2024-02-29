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
    "tree, expectation",
    [
        (jnp.array([[1], [2], [3]]), does_not_raise()),
        (
            jnp.array([[1], [2], [jnp.nan]]),
            pytest.warns(UserWarning, match="The provided trees contain Nans"),
        ),
        (
            jnp.array([[1], [jnp.inf], [3]]),
            pytest.warns(UserWarning, match="The provided trees contain Infs"),
        ),
        (
            jnp.array([[1], [jnp.inf], [jnp.nan]]),
            pytest.warns(UserWarning, match="The provided trees contain Infs and Nans"),
        ),
    ],
)
def test_warn_invalid_entry(tree, expectation):
    """Test validation of trees generates the correct exceptions."""
    valid_data = jnp.array([[1], [2], [3]])
    with expectation:
        validation.warn_invalid_entry(valid_data, tree)


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
