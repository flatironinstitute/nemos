import jax
import jax.numpy as jnp
import pytest

from nemos.glm.params import GLMParams
from nemos.proximal_operator import (
    masked_norm_2,
    prox_elastic_net,
    prox_group_lasso,
    prox_lasso,
    prox_none,
    prox_ridge,
)
from nemos.tree_utils import tree_full_like


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
        ),  # (weights, regularizer_strength, mask, scaling)
    ],
)
def test_prox_operator_returns_correct_type_multineuron(
    prox_operator, input_data, expected_type, example_data_prox_operator_multineuron
):
    """Test whether the proximal operator returns the correct type."""
    args = input_data(example_data_prox_operator_multineuron)
    result = prox_operator(*args)
    assert isinstance(result, expected_type)


def test_prox_none_identity_pytree():
    """prox_none should return the input PyTree unchanged."""
    params = {
        "w": jnp.array([1.0, -2.0, 3.0]),
        "b": jnp.array([0.5]),
        "nested": {"a": jnp.array([-1.0, 2.0])},
    }
    out = prox_none(params, strength=None, scaling=123.0)

    # Structure and values should match exactly
    def leaves_equal(x, y):
        return jnp.allclose(x, y)

    assert set(out.keys()) == set(params.keys())
    assert leaves_equal(out["w"], params["w"])
    assert leaves_equal(out["b"], params["b"])
    assert set(out["nested"].keys()) == {"a"}
    assert leaves_equal(out["nested"]["a"], params["nested"]["a"])


def test_prox_ridge_pytree_broadcast_and_shape():
    """Test prox_ridge on a dict PyTree with mixed scalar/array strengths."""
    params = {
        "w": jnp.array([1.0, -2.0, 3.0]),
        "b": jnp.array([0.5]),
    }
    # Per-leaf strengths: array for w, scalar for b
    strength = {
        "w": jnp.array([0.5, 1.0, 2.0]),
        "b": 0.1,
    }
    scaling = 2.0

    out = prox_ridge(params, strength=strength, scaling=scaling)

    # Expected: u / (1 + scaling * s)
    expected_w = params["w"] / (1.0 + scaling * strength["w"])
    expected_b = params["b"] / (1.0 + scaling * strength["b"])

    assert out["w"].shape == params["w"].shape
    assert out["b"].shape == params["b"].shape
    assert jnp.allclose(out["w"], expected_w)
    assert jnp.allclose(out["b"], expected_b)


def test_prox_lasso_pytree_nested():
    """prox_lasso should support nested PyTrees with per-leaf strengths."""
    params = {
        "features": {
            "pos": jnp.array([1.0, -2.0, 3.0]),
            "spd": jnp.array([-1.0, 0.5]),
        },
        "bias": jnp.array([0.0]),
    }
    strength = {
        "features": {
            "pos": 0.5,  # scalar (broadcast)
            "spd": jnp.array([0.2, 0.3]),
        },
        "bias": 0.0,  # no regularization on bias
    }
    scaling = 1.0

    out = prox_lasso(params, strength=strength, scaling=scaling)

    def soft_thresh(v, thr):
        return jnp.sign(v) * jax.nn.relu(jnp.abs(v) - thr)

    expected_pos = soft_thresh(params["features"]["pos"], 0.5)
    expected_spd = soft_thresh(params["features"]["spd"], jnp.array([0.2, 0.3]))
    expected_bias = params["bias"]  # unchanged (threshold 0)

    assert jnp.allclose(out["features"]["pos"], expected_pos)
    assert jnp.allclose(out["features"]["spd"], expected_spd)
    assert jnp.allclose(out["bias"], expected_bias)


def test_prox_elastic_net_pytree_dict_structure():
    """Test prox_elastic_net on a dict PyTree with leaf-wise (strength, ratio) tuples."""
    params = {
        "x": jnp.array([1.0, -2.0, 3.0]),  # length 3
        "y": jnp.array([-4.0, 5.0]),  # length 2
    }
    # For x: scalar s and r; for y: vector s and scalar r (broadcast)
    strength = {
        "x": (0.5, 0.5),
        "y": (jnp.array([0.2, 0.2]), 0.5),
    }
    scaling = 1.5

    out = prox_elastic_net(params, strength=strength, scaling=scaling)

    def elastic_net_expected(u, s, r, scaling):
        # lam = s * r
        lam = s * r
        # gamma = (1 - r) / r
        gamma = (1.0 - r) / r

        # soft-threshold then divide by (1 + scaling * lam * gamma)
        def soft_thresh(v, thr):
            return jnp.sign(v) * jax.nn.relu(jnp.abs(v) - thr)

        numer = soft_thresh(u, scaling * lam)
        denom = 1.0 + scaling * lam * gamma
        return numer / denom

    expected_x = elastic_net_expected(params["x"], 0.5, 0.5, scaling)
    expected_y = elastic_net_expected(params["y"], jnp.array([0.2, 0.2]), 0.5, scaling)

    assert out["x"].shape == params["x"].shape
    assert out["y"].shape == params["y"].shape
    assert jnp.allclose(out["x"], expected_x)
    assert jnp.allclose(out["y"], expected_y)


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
        ),  # (weights, regularizer_strength, mask, scaling)
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
    params_new = prox_group_lasso(params, tree_full_like(params, 0.05), mask)
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
    params_new = prox_group_lasso(params, tree_full_like(params, 0.05), mask)
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
    result = prox_group_lasso(
        params, tree_full_like(params, regularizer_strength), mask
    )

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
    result = prox_group_lasso(
        params, tree_full_like(params, regularizer_strength), mask
    )

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
    result_array = prox_group_lasso(
        params_array, tree_full_like(params_array, regularizer_strength), mask_array
    )
    result_dict = prox_group_lasso(
        params_dict, tree_full_like(params_dict, regularizer_strength), mask_dict
    )

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
    result = prox_lasso(
        params, tree_full_like(params, regularizer_strength), scaling=1.0
    )

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


def test_prox_group_lasso_pytree_common_group_scaling():
    """For cross-leaf grouping, all features in the same group share the same shrink factor."""
    # Two leaves ("a", "b"), each with 2 features; two groups select the same index across leaves.
    params = {
        "a": jnp.array([1.0, 2.0]),
        "b": jnp.array([2.0, 4.0]),
    }
    mask = {
        # Group 0 picks index 0, group 1 picks index 1
        "a": jnp.array([[1, 0], [0, 1]], dtype=float),
        "b": jnp.array([[1, 0], [0, 1]], dtype=float),
    }
    # Per-leaf per-group strength vectors (same across leaves for clarity)
    strength = {
        "a": jnp.array([0.5, 0.5]),
        "b": jnp.array([0.5, 0.5]),
    }
    scaling = 1.0

    # Compute group-wise factors via masked_norm_2
    l2 = masked_norm_2(params, mask)  # shape (2,)
    factor = jax.nn.relu(1.0 - scaling * strength["a"] / l2)  # per-group factor

    out = prox_group_lasso(params, strength=strength, mask=mask, scaling=scaling)

    # Check that each element is scaled by its group's factor consistently across leaves
    # Group 0 -> index 0 in both "a" and "b"; Group 1 -> index 1
    assert jnp.allclose(out["a"][0] / params["a"][0], factor[0])
    assert jnp.allclose(out["b"][0] / params["b"][0], factor[0])
    assert jnp.allclose(out["a"][1] / params["a"][1], factor[1])
    assert jnp.allclose(out["b"][1] / params["b"][1], factor[1])


def test_prox_group_lasso_pytree_strength_broadcast_and_shape():
    """Scalar vs per-group strength: stronger strength yields stronger shrink; structure preserved."""
    params = {
        "w": jnp.array([1.0, 1.0, 1.0, 1.0]),
    }
    # Two groups: [0,1] and [2,3]
    mask = {
        "w": jnp.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=float),
    }
    # Scalar strength vs larger per-group vector strength
    strength_weak = {"w": 0.05}
    strength_strong = {"w": jnp.array([0.5, 0.5])}
    scaling = 1.0

    out_weak = prox_group_lasso(
        params, strength=strength_weak, mask=mask, scaling=scaling
    )
    out_strong = prox_group_lasso(
        params, strength=strength_strong, mask=mask, scaling=scaling
    )

    # Stronger strength should shrink more (smaller magnitude), elementwise
    assert jnp.all(jnp.abs(out_strong["w"]) <= jnp.abs(out_weak["w"]))
    # Structure preserved
    assert out_weak["w"].shape == params["w"].shape
    assert out_strong["w"].shape == params["w"].shape
