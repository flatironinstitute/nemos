import jax
import jax.numpy as jnp

from nemos.proximal_operator import _vmap_norm2_masked_2, prox_group_lasso, prox_group_lasso_pytree
from nemos.utils import pytree_map_and_reduce
from nemos.pytrees import FeaturePytree

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


def test_compare_group_lasso(example_data_prox_operator):
    params, regularizer_strength, mask, scaling = example_data_prox_operator
    # create a pytree version of params
    params_tree = FeaturePytree(**{f"{k}": params[0][:, jnp.array(msk, dtype=bool)] for k, msk in enumerate(mask)})
    # create a regularizer tree with the same struct as params_tree
    treedef = jax.tree_util.tree_structure(params_tree)
    # make sure the leaves are arrays (otherwise FeaturePytree cannot be instantiated)
    alpha_tree = jax.tree_util.tree_unflatten(treedef, [jnp.atleast_1d(regularizer_strength)] * treedef.num_leaves)
    # compute updates using both functions
    updated_params = prox_group_lasso(params, regularizer_strength, mask, scaling)
    updated_params_tree = prox_group_lasso_pytree((params_tree, params[1]), alpha_tree, scaling)
    # check agreement
    check_updates = [
        jnp.all(updated_params[0][:, jnp.array(msk, dtype=bool)] == updated_params_tree[0][f"{k}"])
        for k, msk in enumerate(mask)
    ]
    assert all(check_updates)
    assert all(updated_params_tree[1] == updated_params[1])

