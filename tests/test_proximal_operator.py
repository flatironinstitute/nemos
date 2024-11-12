import jax.numpy as jnp
import pytest

from nemos.proximal_operator import _vmap_norm2_masked_2, prox_group_lasso, prox_lasso


@pytest.mark.parametrize("prox_operator", [prox_group_lasso, prox_lasso])
def test_prox_operator_returns_tuple(prox_operator, example_data_prox_operator):
    """Test whether the proximal operator returns a tuple."""
    args = example_data_prox_operator
    args = args if prox_operator is prox_group_lasso else (*args[:2], *args[3:])
    params_new = prox_operator(*args)
    assert isinstance(params_new, tuple)


@pytest.mark.parametrize("prox_operator", [prox_group_lasso, prox_lasso])
def test_prox_operator_returns_tuple_multineuron(
    prox_operator, example_data_prox_operator_multineuron
):
    """Test whether the tuple returned by the proximal operator has a length of 2."""
    args = example_data_prox_operator_multineuron
    args = args if prox_operator is prox_group_lasso else (*args[:2], *args[3:])
    params_new = prox_operator(*args)
    assert isinstance(params_new, tuple)


@pytest.mark.parametrize("prox_operator", [prox_group_lasso, prox_lasso])
def test_prox_operator_tuple_length(prox_operator, example_data_prox_operator):
    """Test whether the tuple returned by the proximal operator has a length of 2."""
    args = example_data_prox_operator
    args = args if prox_operator is prox_group_lasso else (*args[:2], *args[3:])
    params_new = prox_operator(*args)
    assert len(params_new) == 2


@pytest.mark.parametrize("prox_operator", [prox_group_lasso, prox_lasso])
def test_prox_operator_tuple_length_multineuron(
    prox_operator, example_data_prox_operator_multineuron
):
    """Test whether the tuple returned by the proximal operator has a length of 2."""
    args = example_data_prox_operator_multineuron
    args = args if prox_operator is prox_group_lasso else (*args[:2], *args[3:])
    params_new = prox_operator(*args)
    assert len(params_new) == 2


@pytest.mark.parametrize("prox_operator", [prox_group_lasso, prox_lasso])
def test_prox_operator_weights_shape(prox_operator, example_data_prox_operator):
    """Test whether the shape of the weights in the proximal operator is correct."""
    args = example_data_prox_operator
    args = args if prox_operator is prox_group_lasso else (*args[:2], *args[3:])
    params_new = prox_operator(*args)
    assert params_new[0].shape == args[0][0].shape


@pytest.mark.parametrize("prox_operator", [prox_group_lasso, prox_lasso])
def test_prox_operator_weights_shape_multineuron(
    prox_operator, example_data_prox_operator_multineuron
):
    """Test whether the shape of the weights in the proximal operator is correct."""
    args = example_data_prox_operator_multineuron
    args = args if prox_operator is prox_group_lasso else (*args[:2], *args[3:])
    params_new = prox_operator(*args)
    assert params_new[0].shape == args[0][0].shape


@pytest.mark.parametrize("prox_operator", [prox_group_lasso, prox_lasso])
def test_prox_operator_intercepts_shape(prox_operator, example_data_prox_operator):
    """Test whether the shape of the intercepts in the proximal operator is correct."""
    args = example_data_prox_operator
    args = args if prox_operator is prox_group_lasso else (*args[:2], *args[3:])
    params_new = prox_operator(*args)
    assert params_new[1].shape == args[0][1].shape


@pytest.mark.parametrize("prox_operator", [prox_group_lasso, prox_lasso])
def test_prox_operator_intercepts_shape_multineuron(
    prox_operator, example_data_prox_operator_multineuron
):
    """Test whether the shape of the intercepts in the proximal operator is correct."""
    args = example_data_prox_operator_multineuron
    args = args if prox_operator is prox_group_lasso else (*args[:2], *args[3:])
    params_new = prox_operator(*args)
    assert params_new[1].shape == args[0][1].shape


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


def test_vmap_norm2_masked_2_shape_multineuron(example_data_prox_operator_multineuron):
    """Test whether the shape of the result from _vmap_norm2_masked_2 is correct."""
    params, _, mask, _ = example_data_prox_operator_multineuron
    l2_norm = _vmap_norm2_masked_2(params[0].T, mask)
    assert l2_norm.shape == (params[0].shape[1], mask.shape[0])


def test_vmap_norm2_masked_2_non_negative(example_data_prox_operator):
    """Test whether all elements of the result from _vmap_norm2_masked_2 are non-negative."""
    params, _, mask, _ = example_data_prox_operator
    l2_norm = _vmap_norm2_masked_2(params[0], mask)
    assert jnp.all(l2_norm >= 0)


def test_vmap_norm2_masked_2_non_negative_multineuron(
    example_data_prox_operator_multineuron,
):
    """Test whether all elements of the result from _vmap_norm2_masked_2 are non-negative."""
    params, _, mask, _ = example_data_prox_operator_multineuron
    l2_norm = _vmap_norm2_masked_2(params[0].T, mask)
    assert jnp.all(l2_norm >= 0)


def test_prox_operator_shrinks_only_masked(example_data_prox_operator):
    params, _, mask, _ = example_data_prox_operator
    mask = mask.at[:, 1].set(jnp.zeros(2))
    params_new = prox_group_lasso(params, 0.05, mask)
    assert params_new[0][1] == params[0][1]
    assert all(params_new[0][i] < params[0][i] for i in [0, 2, 3])


def test_prox_operator_shrinks_only_masked_multineuron(
    example_data_prox_operator_multineuron,
):
    params, _, mask, _ = example_data_prox_operator_multineuron
    mask = mask.astype(float)
    mask = mask.at[:, 1].set(jnp.zeros(2))
    params_new = prox_group_lasso(params, 0.05, mask)
    assert jnp.all(params_new[0][1] == params[0][1])
    assert all(jnp.all(params_new[0][i] < params[0][i]) for i in [0, 2, 3])
