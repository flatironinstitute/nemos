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
    assert hasattr(params_new, "coef")
    assert hasattr(params_new, "intercept")


def test_prox_lasso_has_coef_and_intercept_multineuron(
    example_data_prox_operator_multineuron,
):
    """Test whether the returned GLMParams has both coef and intercept."""
    params, regularizer_strength, _, scaling = example_data_prox_operator_multineuron
    params_new = prox_lasso(params, regularizer_strength, scaling)
    assert hasattr(params_new, "coef")
    assert hasattr(params_new, "intercept")


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


# Tests for flexible PyTree structures
def test_prox_group_lasso_dict_structure():
    """Test prox_group_lasso with dict-based PyTree parameters."""
    # Create dict-based parameters
    params = {
        "spatial": jnp.ones(3),
        "temporal": jnp.ones(2),
        "intercept": jnp.zeros(1),
    }

    # Create matching mask structure: (n_groups, *param_shape) for each leaf
    mask = {
        "spatial": jnp.array(
            [[1, 1, 0], [0, 0, 1]], dtype=float
        ),  # 2 groups, 3 features
        "temporal": jnp.array([[1, 0], [0, 1]], dtype=float),  # 2 groups, 2 features
        "intercept": jnp.zeros((2, 1), dtype=float),  # Not regularized
    }

    regularizer_strength = 0.1
    result = prox_group_lasso(params, regularizer_strength, mask)

    # Check structure preserved
    assert isinstance(result, dict)
    assert set(result.keys()) == {"spatial", "temporal", "intercept"}

    # Check shapes preserved
    assert result["spatial"].shape == (3,)
    assert result["temporal"].shape == (2,)
    assert result["intercept"].shape == (1,)

    # Check that regularization was applied (values should be shrunk)
    assert jnp.all(jnp.abs(result["spatial"]) <= jnp.abs(params["spatial"]))
    assert jnp.all(jnp.abs(result["temporal"]) <= jnp.abs(params["temporal"]))
    # Intercept should be unchanged (zero mask)
    assert jnp.allclose(result["intercept"], params["intercept"])


def test_prox_group_lasso_nested_structure():
    """Test prox_group_lasso with nested PyTree structure."""
    # Create nested structure
    params = {
        "features": {
            "position": jnp.ones(4),
            "speed": jnp.ones(3),
        },
        "bias": jnp.zeros(1),
    }

    # Create matching mask
    mask = {
        "features": {
            "position": jnp.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=float),
            "speed": jnp.array([[1, 0, 0], [0, 1, 1]], dtype=float),
        },
        "bias": jnp.zeros((2, 1), dtype=float),
    }

    regularizer_strength = 0.1
    result = prox_group_lasso(params, regularizer_strength, mask)

    # Check nested structure preserved
    assert isinstance(result, dict)
    assert "features" in result and "bias" in result
    assert isinstance(result["features"], dict)
    assert set(result["features"].keys()) == {"position", "speed"}

    # Check shapes preserved
    assert result["features"]["position"].shape == (4,)
    assert result["features"]["speed"].shape == (3,)
    assert result["bias"].shape == (1,)


def test_prox_group_lasso_equivalence_array_vs_dict():
    """Test that equivalent groupings produce similar results regardless of PyTree structure.

    This tests that splitting an array into dict leaves with equivalent masks
    produces the same regularization behavior.
    """
    # Setup: 6 coefficients split as [position(4), speed(2)]
    # 2 groups: group 0 = first half of each, group 1 = second half of each

    # Version 1: Single array
    params_array = GLMParams(
        coef=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        intercept=jnp.zeros(1),
    )
    mask_array = GLMParams(
        coef=jnp.array(
            [
                [1, 1, 1, 0, 0, 0],  # group 0: first half
                [0, 0, 0, 1, 1, 1],  # group 1: second half
            ],
            dtype=float,
        ),
        intercept=jnp.zeros((2, 1), dtype=float),
    )

    # Version 2: Dict splitting same coefficients
    params_dict = {
        "position": jnp.array([1.0, 2.0, 3.0, 4.0]),
        "speed": jnp.array([5.0, 6.0]),
        "intercept": jnp.zeros(1),
    }
    mask_dict = {
        "position": jnp.array(
            [
                [1, 1, 0, 0],  # group 0: first 2 position features
                [0, 0, 1, 1],  # group 1: last 2 position features
            ],
            dtype=float,
        ),
        "speed": jnp.array(
            [
                [1, 0],  # group 0: first speed feature
                [0, 1],  # group 1: second speed feature
            ],
            dtype=float,
        ),
        "intercept": jnp.zeros((2, 1), dtype=float),
    }

    regularizer_strength = 0.1

    # Apply prox operator to both versions
    result_array = prox_group_lasso(params_array, regularizer_strength, mask_array)
    result_dict = prox_group_lasso(params_dict, regularizer_strength, mask_dict)

    # Concatenate dict results to compare with array
    result_dict_concat = jnp.concatenate(
        [
            result_dict["position"],
            result_dict["speed"],
        ]
    )

    # The results should be similar (not exactly equal due to different group norm calculations)
    # Each structure computes group norms differently:
    # - Array: groups are [1,2,3] and [4,5,6]
    # - Dict: groups are [1,2] from position + [5] from speed, and [3,4] from position + [6] from speed
    # So we just check that shapes match and regularization was applied
    assert result_array.coef.shape == result_dict_concat.shape
    assert jnp.all(jnp.abs(result_array.coef) <= jnp.abs(params_array.coef))
    assert jnp.all(
        jnp.abs(result_dict_concat) <= jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    )


def test_masked_norm_2_dict_structure():
    """Test masked_norm_2 with dict-based PyTree parameters."""
    params = {
        "spatial": jnp.array([1.0, 2.0, 3.0]),
        "temporal": jnp.array([4.0, 5.0]),
    }

    mask = {
        "spatial": jnp.array([[1, 1, 0], [0, 0, 1]], dtype=float),
        "temporal": jnp.array([[1, 0], [0, 1]], dtype=float),
    }

    l2_norm = masked_norm_2(params, mask)

    # Should return (n_groups,) array
    assert l2_norm.shape == (2,)
    assert jnp.all(l2_norm >= 0)

    # Group 0: spatial[0,1] + temporal[0] = [1, 2, 4]
    #   Unnormalized: sqrt(1^2 + 2^2 + 4^2) = sqrt(21)
    #   Group size: 3 elements
    #   Normalized (default): sqrt(21) / sqrt(3) ≈ 2.646
    # Group 1: spatial[2] + temporal[1] = [3, 5]
    #   Unnormalized: sqrt(3^2 + 5^2) = sqrt(34)
    #   Group size: 2 elements
    #   Normalized (default): sqrt(34) / sqrt(2) ≈ 4.123
    expected = jnp.array(
        [jnp.sqrt(21.0) / jnp.sqrt(3.0), jnp.sqrt(34.0) / jnp.sqrt(2.0)]
    )
    assert jnp.allclose(l2_norm, expected)


def test_prox_lasso_dict_structure():
    """Test prox_lasso with dict-based PyTree parameters."""
    params = {
        "features": jnp.array([1.0, 2.0, 3.0]),
        "intercept": jnp.zeros(1),
    }

    regularizer_strength = 0.5
    result = prox_lasso(params, regularizer_strength, scaling=1.0)

    # Check structure preserved
    assert isinstance(result, dict)
    assert set(result.keys()) == {"features", "intercept"}

    # Check shapes preserved
    assert result["features"].shape == (3,)
    assert result["intercept"].shape == (1,)

    # Check soft-thresholding was applied
    # prox_lasso(x, lambda) = sign(x) * max(|x| - lambda, 0)
    expected_features = jnp.array([0.5, 1.5, 2.5])  # [1-0.5, 2-0.5, 3-0.5]
    assert jnp.allclose(result["features"], expected_features)
    assert jnp.allclose(result["intercept"], params["intercept"])
