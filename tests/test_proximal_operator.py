import jax
import jax.numpy as jnp

from nemos.proximal_operator import _vmap_norm2_masked_2, prox_group_lasso


def test_prox_group_lasso_returns_tuple(example_data_prox_operator):
    """Test whether prox_group_lasso returns a tuple."""
    params, alpha, mask, scaling = example_data_prox_operator
    updated_params = prox_group_lasso(params, alpha, mask, scaling)
    assert isinstance(updated_params, tuple)


def test_prox_group_lasso_tuple_length(example_data_prox_operator):
    """Test whether the tuple returned by prox_group_lasso has a length of 2."""
    params, alpha, mask, scaling = example_data_prox_operator
    updated_params = prox_group_lasso(params, alpha, mask, scaling)
    assert len(updated_params) == 2


def test_prox_group_lasso_weights_shape(example_data_prox_operator):
    """Test whether the shape of the weights in prox_group_lasso is correct."""
    params, alpha, mask, scaling = example_data_prox_operator
    updated_params = prox_group_lasso(params, alpha, mask, scaling)
    assert updated_params[0].shape == params[0].shape


def test_prox_group_lasso_intercepts_shape(example_data_prox_operator):
    """Test whether the shape of the intercepts in prox_group_lasso is correct."""
    params, alpha, mask, scaling = example_data_prox_operator
    updated_params = prox_group_lasso(params, alpha, mask, scaling)
    assert updated_params[1].shape == params[1].shape


def test_vmap_norm2_masked_2_returns_array(example_data_prox_operator):
    """Test whether _vmap_norm2_masked_2 returns a NumPy array."""
    params, _, mask, _ = example_data_prox_operator
    l2_norm = _vmap_norm2_masked_2(params[0], mask)
    assert isinstance(l2_norm, jnp.ndarray)


def test_vmap_norm2_masked_2_shape(example_data_prox_operator):
    """Test whether the shape of the result from _vmap_norm2_masked_2 is correct."""
    params, _, mask, _ = example_data_prox_operator
    l2_norm = _vmap_norm2_masked_2(params[0], mask)
    assert l2_norm.shape == (params[0].shape[0], mask.shape[0])


def test_vmap_norm2_masked_2_non_negative(example_data_prox_operator):
    """Test whether all elements of the result from _vmap_norm2_masked_2 are non-negative."""
    params, _, mask, _ = example_data_prox_operator
    l2_norm = _vmap_norm2_masked_2(params[0], mask)
    assert jnp.all(l2_norm >= 0)

