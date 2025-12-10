import jax.numpy as jnp
import pytest

from nemos.proximal_operator import _vmap_norm2_masked_2, prox_group_lasso, prox_lasso


@pytest.mark.parametrize(
    "prox_operator,input_data,expected_type",
    [
        (prox_lasso, lambda d: (d[0], d[1], d[3]), tuple),  # (params, regularizer_strength, scaling)
        (prox_group_lasso, lambda d: (d[0][0], d[1], d[2], d[3]), jnp.ndarray),  # (weights, regularizer_strength, mask, scaling)
    ],
)
def test_prox_operator_returns_correct_type(prox_operator, input_data, expected_type, example_data_prox_operator):
    """Test whether the proximal operator returns the correct type."""
    args = input_data(example_data_prox_operator)
    result = prox_operator(*args)
    assert isinstance(result, expected_type)


@pytest.mark.parametrize(
    "prox_operator,input_data,expected_type",
    [
        (prox_lasso, lambda d: (d[0], d[1], d[3]), tuple),  # (params, regularizer_strength, scaling)
        (prox_group_lasso, lambda d: (d[0][0], d[1], d[2], d[3]), jnp.ndarray),  # (weights, regularizer_strength, mask, scaling)
    ],
)
def test_prox_operator_returns_correct_type_multineuron(
    prox_operator, input_data, expected_type, example_data_prox_operator_multineuron
):
    """Test whether the proximal operator returns the correct type."""
    args = input_data(example_data_prox_operator_multineuron)
    result = prox_operator(*args)
    assert isinstance(result, expected_type)


def test_prox_lasso_tuple_length(example_data_prox_operator):
    """Test whether the tuple returned by prox_lasso has a length of 2."""
    params, regularizer_strength, _, scaling = example_data_prox_operator
    params_new = prox_lasso(params, regularizer_strength, scaling)
    assert len(params_new) == 2


def test_prox_lasso_tuple_length_multineuron(example_data_prox_operator_multineuron):
    """Test whether the tuple returned by prox_lasso has a length of 2."""
    params, regularizer_strength, _, scaling = example_data_prox_operator_multineuron
    params_new = prox_lasso(params, regularizer_strength, scaling)
    assert len(params_new) == 2


@pytest.mark.parametrize(
    "prox_operator,input_data,shape_getter",
    [
        (prox_lasso, lambda d: (d[0], d[1], d[3]), lambda result, d: (result[0].shape, d[0][0].shape)),  # (params, regularizer_strength, scaling)
        (prox_group_lasso, lambda d: (d[0][0], d[1], d[2], d[3]), lambda result, d: (result.shape, d[0][0].shape)),  # (weights, regularizer_strength, mask, scaling)
    ],
)
def test_prox_operator_weights_shape(prox_operator, input_data, shape_getter, example_data_prox_operator):
    """Test whether the shape of the weights in the proximal operator is correct."""
    args = input_data(example_data_prox_operator)
    result = prox_operator(*args)
    result_shape, expected_shape = shape_getter(result, example_data_prox_operator)
    assert result_shape == expected_shape


@pytest.mark.parametrize(
    "prox_operator,input_data,shape_getter",
    [
        (prox_lasso, lambda d: (d[0], d[1], d[3]), lambda result, d: (result[0].shape, d[0][0].shape)),  # (params, regularizer_strength, scaling)
        (prox_group_lasso, lambda d: (d[0][0], d[1], d[2], d[3]), lambda result, d: (result.shape, d[0][0].shape)),  # (weights, regularizer_strength, mask, scaling)
    ],
)
def test_prox_operator_weights_shape_multineuron(
    prox_operator, input_data, shape_getter, example_data_prox_operator_multineuron
):
    """Test whether the shape of the weights in the proximal operator is correct."""
    args = input_data(example_data_prox_operator_multineuron)
    result = prox_operator(*args)
    result_shape, expected_shape = shape_getter(result, example_data_prox_operator_multineuron)
    assert result_shape == expected_shape


def test_prox_lasso_intercepts_shape(example_data_prox_operator):
    """Test whether the shape of the intercepts returned by prox_lasso is correct."""
    params, regularizer_strength, _, scaling = example_data_prox_operator
    params_new = prox_lasso(params, regularizer_strength, scaling)
    assert params_new[1].shape == params[1].shape


def test_prox_lasso_intercepts_shape_multineuron(example_data_prox_operator_multineuron):
    """Test whether the shape of the intercepts returned by prox_lasso is correct."""
    params, regularizer_strength, _, scaling = example_data_prox_operator_multineuron
    params_new = prox_lasso(params, regularizer_strength, scaling)
    assert params_new[1].shape == params[1].shape


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


def test_prox_group_lasso_shrinks_only_masked(example_data_prox_operator):
    """Test that prox_group_lasso only shrinks features that belong to groups (masked)."""
    params, _, mask, _ = example_data_prox_operator
    mask = mask.at[:, 1].set(jnp.zeros(2))
    weights_new = prox_group_lasso(params[0], 0.05, mask)
    assert weights_new[1] == params[0][1]
    assert all(weights_new[i] < params[0][i] for i in [0, 2, 3])


def test_prox_group_lasso_shrinks_only_masked_multineuron(
    example_data_prox_operator_multineuron,
):
    """Test that prox_group_lasso only shrinks features that belong to groups (masked) for multiple neurons."""
    params, _, mask, _ = example_data_prox_operator_multineuron
    mask = mask.astype(float)
    mask = mask.at[:, 1].set(jnp.zeros(2))
    weights_new = prox_group_lasso(params[0], 0.05, mask)
    assert all(jnp.all(weights_new[i] < params[0][i]) for i in [0, 2, 3])
