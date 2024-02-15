from contextlib import nullcontext as does_not_raise

import jax.numpy as jnp
import pytest

from nemos import validation

@pytest.mark.parametrize(
    "tree, expectation",
    [
        (jnp.array([[1], [2], [3]]), does_not_raise()),
        (jnp.array([[1], [2], [jnp.nan]]), pytest.raises(ValueError, match="The provided trees contain Nans")),
        (jnp.array([[1], [jnp.inf], [3]]), pytest.raises(ValueError, match="The provided trees contain Infs")),
        (jnp.array([[1], [jnp.inf], [jnp.nan]]),
         pytest.raises(ValueError, match="The provided trees contain Infs and Nans"))
    ]
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
        (jnp.array([[1], [2], [jnp.nan]]), pytest.warns(UserWarning, match="The provided trees contain Nans")),
        (jnp.array([[1], [jnp.inf], [3]]), pytest.warns(UserWarning, match="The provided trees contain Infs")),
        (jnp.array([[1], [jnp.inf], [jnp.nan]]),
         pytest.warns(UserWarning, match="The provided trees contain Infs and Nans"))
    ]
)
def test_warn_invalid_entry(tree, expectation):
    """Test validation of trees generates the correct exceptions."""
    valid_data = jnp.array([[1], [2], [3]])
    with expectation:
        validation.warn_invalid_entry(valid_data, tree)
