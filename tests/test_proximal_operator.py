import jax.numpy as jnp
import pytest

from nemos.glm.params import GLMParams
from nemos.proximal_operator import masked_norm_2, prox_group_lasso, prox_lasso


@pytest.mark.parametrize(
    "prox_operator,input_data,expected_type",
    [
        (
            prox_lasso,
            lambda d: (d[0], d[1], d[3]),
            GLMParams,
        ),  # (params, regularizer_strength, scaling)
        (
            prox_group_lasso,
            lambda d: (d[0], d[1], d[2], d[3]),
            GLMParams,
        ),  # (params, regularizer_strength, mask, scaling)
    ],
)
def test_prox_operator_returns_correct_type(
    prox_operator, input_data, expected_type, example_data_prox_operator
):
    """Test whether the proximal operator returns the correct type."""
    args = input_data(example_data_prox_operator)
    result = prox_operator(*args)
    assert isinstance(result, expected_type)


@pytest.mark.parametrize(
    "prox_operator,input_data,expected_type",
    [
        (
            prox_lasso,
            lambda d: (d[0], d[1], d[3]),
            GLMParams,
        ),  # (params, regularizer_strength, scaling)
        (
            prox_group_lasso,
            lambda d: (d[0], d[1], d[2], d[3]),
            GLMParams,
        ),  # (params, regularizer_strength, mask, scaling)
    ],
)
def test_prox_operator_returns_correct_type_multineuron(
    prox_operator, input_data, expected_type, example_data_prox_operator_multineuron
):
    """Test whether the proximal operator returns the correct type."""
    args = input_data(example_data_prox_operator_multineuron)
    result = prox_operator(*args)
    assert isinstance(result, expected_type)


def test_prox_lasso_has_coef_and_intercept(example_data_prox_operator):
    """Test whether the returned GLMParams has both coef and intercept."""
    params, regularizer_strength, _, scaling = example_data_prox_operator
    params_new = prox_lasso(params, regularizer_strength, scaling)
    assert hasattr(params_new, 'coef')
    assert hasattr(params_new, 'intercept')


def test_prox_lasso_has_coef_and_intercept_multineuron(example_data_prox_operator_multineuron):
    """Test whether the returned GLMParams has both coef and intercept."""
    params, regularizer_strength, _, scaling = example_data_prox_operator_multineuron
    params_new = prox_lasso(params, regularizer_strength, scaling)
    assert hasattr(params_new, 'coef')
    assert hasattr(params_new, 'intercept')


@pytest.mark.parametrize(
    "prox_operator,input_data,shape_getter",
    [
        (
            prox_lasso,
            lambda d: (d[0], d[1], d[3]),
            lambda result, d: (result.coef.shape, d[0].coef.shape),
        ),  # (params, regularizer_strength, scaling)
        (
            prox_group_lasso,
            lambda d: (d[0], d[1], d[2], d[3]),
            lambda result, d: (result.coef.shape, d[0].coef.shape),
        ),  # (params, regularizer_strength, mask, scaling)
    ],
)
def test_prox_operator_weights_shape(
    prox_operator, input_data, shape_getter, example_data_prox_operator
):
    """Test whether the shape of the weights in the proximal operator is correct."""
    args = input_data(example_data_prox_operator)
    result = prox_operator(*args)
    result_shape, expected_shape = shape_getter(result, example_data_prox_operator)
    assert result_shape == expected_shape


@pytest.mark.parametrize(
    "prox_operator,input_data,shape_getter",
    [
        (
            prox_lasso,
            lambda d: (d[0], d[1], d[3]),
            lambda result, d: (result.coef.shape, d[0].coef.shape),
        ),  # (params, regularizer_strength, scaling)
        (
            prox_group_lasso,
            lambda d: (d[0], d[1], d[2], d[3]),
            lambda result, d: (result.coef.shape, d[0].coef.shape),
        ),  # (params, regularizer_strength, mask, scaling)
    ],
)
def test_prox_operator_weights_shape_multineuron(
    prox_operator, input_data, shape_getter, example_data_prox_operator_multineuron
):
    """Test whether the shape of the weights in the proximal operator is correct."""
    args = input_data(example_data_prox_operator_multineuron)
    result = prox_operator(*args)
    result_shape, expected_shape = shape_getter(
        result, example_data_prox_operator_multineuron
    )
    assert result_shape == expected_shape


def test_prox_lasso_intercepts_shape(example_data_prox_operator):
    """Test whether the shape of the intercepts returned by prox_lasso is correct."""
    params, regularizer_strength, _, scaling = example_data_prox_operator
    params_new = prox_lasso(params, regularizer_strength, scaling)
    assert params_new.intercept.shape == params.intercept.shape


def test_prox_lasso_intercepts_shape_multineuron(
    example_data_prox_operator_multineuron,
):
    """Test whether the shape of the intercepts returned by prox_lasso is correct."""
    params, regularizer_strength, _, scaling = example_data_prox_operator_multineuron
    params_new = prox_lasso(params, regularizer_strength, scaling)
    assert params_new.intercept.shape == params.intercept.shape


def test_masked_norm_2_returns_array(example_data_prox_operator):
    """Test whether masked_norm_2 returns a NumPy array."""
    params, _, mask, _ = example_data_prox_operator
    l2_norm = masked_norm_2(params, mask)
    assert isinstance(l2_norm, jnp.ndarray)


def test_masked_norm_2_shape(example_data_prox_operator):
    """Test whether the shape of the result from masked_norm_2 is correct."""
    params, _, mask, _ = example_data_prox_operator
    l2_norm = masked_norm_2(params, mask)
    # For single neuron: shape should be (n_groups,)
    assert l2_norm.shape == (mask.coef.shape[0],)


def test_masked_norm_2_shape_multineuron(example_data_prox_operator_multineuron):
    """Test whether the shape of the result from masked_norm_2 is correct."""
    params, _, mask, _ = example_data_prox_operator_multineuron
    l2_norm = masked_norm_2(params, mask)
    # For multi-neuron: shape should be (n_groups,)
    assert l2_norm.shape == (mask.coef.shape[0],)


def test_masked_norm_2_non_negative(example_data_prox_operator):
    """Test whether all elements of the result from masked_norm_2 are non-negative."""
    params, _, mask, _ = example_data_prox_operator
    l2_norm = masked_norm_2(params, mask)
    assert jnp.all(l2_norm >= 0)


def test_masked_norm_2_non_negative_multineuron(
    example_data_prox_operator_multineuron,
):
    """Test whether all elements of the result from masked_norm_2 are non-negative."""
    params, _, mask, _ = example_data_prox_operator_multineuron
    l2_norm = masked_norm_2(params, mask)
    assert jnp.all(l2_norm >= 0)


def test_prox_group_lasso_shrinks_only_masked(example_data_prox_operator):
    """Test that prox_group_lasso only shrinks features that belong to groups (masked)."""
    params, _, mask, _ = example_data_prox_operator
    # Set feature 1 to have no group (all zeros in mask)
    # For single neuron case: shape is (n_groups, n_features)
    n_groups = mask.coef.shape[0]
    mask_array = mask.coef.at[:, 1].set(jnp.zeros(n_groups))
    mask = GLMParams(mask_array, mask.intercept)
    params_new = prox_group_lasso(params, 0.05, mask)
    # Feature 1 should not be shrunk (no group assignment)
    assert params_new.coef[1] == params.coef[1]
    # Other features should be shrunk
    assert all(params_new.coef[i] < params.coef[i] for i in [0, 2, 3])


def test_prox_group_lasso_shrinks_only_masked_multineuron(
    example_data_prox_operator_multineuron,
):
    """Test that prox_group_lasso only shrinks features that belong to groups (masked) for multiple neurons."""
    params, _, mask, _ = example_data_prox_operator_multineuron
    # Set feature 1 to have no group (all zeros in mask)
    # For multi-neuron case: shape is (n_groups, n_features, n_neurons)
    n_groups = mask.coef.shape[0]
    n_neurons = mask.coef.shape[2]
    mask_array = mask.coef.at[:, 1].set(jnp.zeros((n_groups, n_neurons)))
    mask = GLMParams(mask_array, mask.intercept)
    params_new = prox_group_lasso(params, 0.05, mask)
    # Feature 1 should not be shrunk across all neurons
    assert jnp.all(params_new.coef[1] == params.coef[1])
    # Other features should be shrunk
    assert all(jnp.all(params_new.coef[i] < params.coef[i]) for i in [0, 2, 3])
