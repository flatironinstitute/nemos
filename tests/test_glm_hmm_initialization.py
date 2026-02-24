"""Tests for glm_hmm/initialize_parameters.py"""

import itertools
from unittest.mock import create_autospec

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nemos.glm_hmm.initialize_parameters import (
    DEFAULT_INIT_FUNCTION,
    _is_native_init_registry,
    _resolve_dirichlet_priors,
    _resolve_init_func,
    _resolve_init_funcs_registry,
    _resolve_init_kwargs,
    _resolve_init_kwargs_registry,
    glm_hmm_initialization,
    ones_scale_init,
    random_glm_params_init,
    sticky_transition_proba_init,
    uniform_initial_proba_init,
    uniform_transition_proba_init,
)


class TestRandomGLMParamsInitialization:
    """Test random initialization of GLM parameters for GLM-HMM."""

    @pytest.mark.parametrize("n_states", [1, 2, 3])
    @pytest.mark.parametrize(
        "n_samples, n_features, n_neurons",
        [
            (100, 5, 1),  # Single neuron
            (100, 5, 3),  # Multiple neurons
            (50, 10, 1),  # Different dimensions
        ],
    )
    def test_expected_output_shape(self, n_states, n_samples, n_features, n_neurons):
        """Test that output shapes match expected dimensions."""
        X = jnp.ones((n_samples, n_features))
        y = jnp.ones((n_samples, n_neurons)) if n_neurons > 1 else jnp.ones(n_samples)
        inverse_link = lambda x: x  # Identity link

        coef, intercept = random_glm_params_init(
            n_states, X, y, inverse_link, random_key=jax.random.PRNGKey(123)
        )

        # Check shapes
        if n_neurons == 1:
            assert coef.shape == (n_features, n_states)
            assert intercept.shape == (n_states,)
        else:
            assert coef.shape == (n_features, n_neurons, n_states)
            assert intercept.shape == (n_neurons, n_states)

    @pytest.mark.parametrize("std_dev", [0.0, 1.0, None])
    def test_std_dev_param(self, std_dev):
        X = jnp.ones((10, 2))
        y = jnp.ones((10, 3))
        inverse_link = lambda x: x  # Identity link

        kwargs = {} if std_dev is None else {"std_dev": std_dev}
        coef, intercept = random_glm_params_init(
            4, X, y, inverse_link, random_key=jax.random.PRNGKey(123), **kwargs
        )
        if std_dev == 0:
            assert np.all(coef == 0)

    @pytest.mark.parametrize(
        "X, y",
        [
            (np.ones((100, 5)), np.ones(100)),
            (jnp.ones((100, 5)), jnp.ones(100)),
        ],
    )
    def test_expected_output_type(self, X, y):
        """Test that outputs are JAX arrays regardless of input type."""
        n_states = 2
        inverse_link = lambda x: x

        coef, intercept = random_glm_params_init(
            n_states, X, y, inverse_link, random_key=jax.random.PRNGKey(123)
        )

        assert isinstance(coef, jnp.ndarray)
        assert isinstance(intercept, jnp.ndarray)

    def test_randomization(self):
        """Test that different seeds produce different coef but same intercept."""
        n_states = 3
        X = jnp.ones((100, 5))
        y = jnp.ones(100)
        inverse_link = lambda x: x

        seed1 = jax.random.PRNGKey(41)
        seed2 = jax.random.PRNGKey(42)

        coef1, intercept1 = random_glm_params_init(
            n_states, X, y, inverse_link, random_key=seed1
        )
        coef2, intercept2 = random_glm_params_init(
            n_states, X, y, inverse_link, random_key=seed2
        )

        # Different seeds should give different random coefficients
        assert not jnp.allclose(coef1, coef2)

        # Intercept is deterministic (based on mean of y), so should be identical
        assert jnp.allclose(intercept1, intercept2)

    def test_coef_magnitude(self):
        """Test that coefficients are small (scaled by 0.001)."""
        n_states = 3
        X = jnp.ones((100, 5))
        y = jnp.ones(100)
        inverse_link = lambda x: x

        coef, _ = random_glm_params_init(
            n_states, X, y, inverse_link, random_key=jax.random.PRNGKey(123)
        )

        # Coefficients should be small (0.001 * normal values)
        # Most values should be within [-0.01, 0.01] (roughly 3 std devs)
        assert jnp.abs(coef).max() < 0.01

    def test_intercept_matches_mean_rate(self):
        """Test that intercept is initialized to match mean rate of y."""
        n_states = 3
        X = jnp.ones((100, 5))
        y_mean = 2.5
        y = jnp.full(100, y_mean)

        # Identity link: inverse_link(x) = x, so intercept should equal mean
        inverse_link = lambda x: x

        _, intercept = random_glm_params_init(
            n_states, X, y, inverse_link, random_key=jax.random.PRNGKey(123)
        )

        # All states should have same intercept (tiled), equal to mean of y
        assert jnp.allclose(intercept, y_mean)

    def test_intercept_tiled_across_states(self):
        """Test that intercept is the same across all states."""
        n_states = 3
        X = jnp.ones((100, 5))
        y = jnp.array([1.0, 2.0, 3.0] * 33 + [1.0])  # Non-uniform values
        inverse_link = lambda x: x

        _, intercept = random_glm_params_init(
            n_states, X, y, inverse_link, random_key=jax.random.PRNGKey(123)
        )

        # All states should have identical intercept values
        assert jnp.allclose(intercept[0], intercept)

    @pytest.mark.parametrize("n_neurons", [1, 3])
    def test_inverse_link_function_usage(self, n_neurons):
        """Test that inverse_link_function is used for intercept initialization."""
        n_states = 2
        X = jnp.ones((100, 5))
        y = jnp.full((100, n_neurons) if n_neurons > 1 else 100, 10.0)

        # Exp link: inverse_link(x) = exp(x), so intercept = log(mean(y))
        inverse_link = jnp.exp

        _, intercept = random_glm_params_init(
            n_states, X, y, inverse_link, random_key=jax.random.PRNGKey(123)
        )

        # intercept should be log(10.0) for exp inverse link
        expected = jnp.log(10.0)
        assert jnp.allclose(intercept, expected)


class TestOnesScaleInitialization:
    """Test ones initialization for scale parameters."""

    @pytest.mark.parametrize("n_states", [1, 2, 3, 5])
    @pytest.mark.parametrize("n_samples, n_neurons", [(100, 1), (100, 3), (50, 10)])
    def test_expected_output_shape(self, n_states, n_samples, n_neurons):
        """Test that output shapes match expected dimensions."""
        X = jnp.ones((n_samples, 5))
        y = jnp.ones((n_samples, n_neurons)) if n_neurons > 1 else jnp.ones(n_samples)

        scale = ones_scale_init(n_states, X, y, random_key=jax.random.PRNGKey(124))

        # Check shape
        if n_neurons == 1:
            assert scale.shape == (n_states,)
        else:
            assert scale.shape == (n_neurons, n_states)

    @pytest.mark.parametrize(
        "X, y",
        [
            (np.ones((100, 5)), np.ones(100)),
            (jnp.ones((100, 5)), jnp.ones(100)),
        ],
    )
    def test_expected_output_type(self, X, y):
        """Test that output is a JAX array regardless of input type."""
        n_states = 2

        scale = ones_scale_init(n_states, X, y, random_key=jax.random.PRNGKey(124))

        assert isinstance(scale, jnp.ndarray)

    @pytest.mark.parametrize("n_states", [1, 3, 5])
    @pytest.mark.parametrize("n_neurons", [1, 3])
    def test_all_values_are_ones(self, n_states, n_neurons):
        """Test that all scale values are initialized to 1.0."""
        X = jnp.ones((100, 5))
        y = jnp.ones((100, n_neurons)) if n_neurons > 1 else jnp.ones(100)

        scale = ones_scale_init(n_states, X, y, random_key=jax.random.PRNGKey(124))

        # All values should be exactly 1.0
        assert jnp.all(scale == 1.0)

    def test_deterministic(self):
        """Test that output is deterministic (same across different calls)."""
        n_states = 3
        X = jnp.ones((100, 5))
        y = jnp.ones(100)

        scale1 = ones_scale_init(n_states, X, y, random_key=jax.random.PRNGKey(124))
        scale2 = ones_scale_init(n_states, X, y, random_key=jax.random.PRNGKey(999))

        # Should be identical regardless of random key
        assert jnp.array_equal(scale1, scale2)


class TestStickyTransitionProbaInitialization:
    """Test sticky initialization for transition probabilities."""

    @pytest.mark.parametrize("n_states", [1, 2, 3, 5])
    def test_expected_output_shape(self, n_states):
        """Test that output shape is (n_states, n_states)."""
        X = jnp.ones((100, 5))
        y = jnp.ones(100)

        transition_prob = sticky_transition_proba_init(
            n_states, X, y, random_key=jax.random.PRNGKey(123)
        )

        assert transition_prob.shape == (n_states, n_states)

    @pytest.mark.parametrize(
        "X, y",
        [
            (np.ones((100, 5)), np.ones(100)),
            (jnp.ones((100, 5)), jnp.ones(100)),
        ],
    )
    def test_expected_output_type(self, X, y):
        """Test that output is a JAX array regardless of input type."""
        n_states = 2

        transition_prob = sticky_transition_proba_init(
            n_states, X, y, random_key=jax.random.PRNGKey(123)
        )

        assert isinstance(transition_prob, jnp.ndarray)

    @pytest.mark.parametrize("n_states", [2, 3, 5])
    def test_off_diagonal_values(self, n_states):
        """Test that off-diagonal values are (1 - prob_stay) / (n_states - 1)."""
        X = jnp.ones((100, 5))
        y = jnp.ones(100)
        prob_stay = 0.95

        transition_prob = sticky_transition_proba_init(
            n_states, X, y, random_key=jax.random.PRNGKey(123), prob_stay=prob_stay
        )

        # Off-diagonal should be (1 - prob_stay) / (n_states - 1)
        expected_off_diag = (1 - prob_stay) / (n_states - 1)

        # Check all off-diagonal elements
        for i in range(n_states):
            for j in range(n_states):
                if i != j:
                    assert jnp.isclose(transition_prob[i, j], expected_off_diag)

    @pytest.mark.parametrize("n_states", [2, 3, 5])
    def test_rows_sum_to_one(self, n_states):
        """Test that each row sums to 1 (valid probability distribution)."""
        X = jnp.ones((100, 5))
        y = jnp.ones(100)

        transition_prob = sticky_transition_proba_init(
            n_states, X, y, random_key=jax.random.PRNGKey(123)
        )

        # Each row should sum to 1
        row_sums = jnp.sum(transition_prob, axis=1)
        assert jnp.allclose(row_sums, 1.0)

    def test_single_state_edge_case(self):
        """Test that n_states=1 returns [[prob_stay]]."""
        X = jnp.ones((100, 5))
        y = jnp.ones(100)
        n_states = 1
        prob_stay = 0.95

        transition_prob = sticky_transition_proba_init(
            n_states, X, y, random_key=jax.random.PRNGKey(123), prob_stay=prob_stay
        )

        # For single state, implementation returns [[prob_stay]] (not normalized)
        assert transition_prob.shape == (1, 1)
        assert jnp.isclose(transition_prob[0, 0], prob_stay)

    @pytest.mark.parametrize("prob_stay", [0.8, 0.9, 0.95, 0.99])
    def test_custom_prob_stay(self, prob_stay):
        """Test that custom prob_stay values work correctly."""
        X = jnp.ones((100, 5))
        y = jnp.ones(100)
        n_states = 3

        transition_prob = sticky_transition_proba_init(
            n_states, X, y, random_key=jax.random.PRNGKey(123), prob_stay=prob_stay
        )

        # Diagonal should match custom prob_stay
        diagonal = jnp.diag(transition_prob)
        assert jnp.allclose(diagonal, prob_stay)

    def test_deterministic(self):
        """Test that output is deterministic (same across different calls)."""
        n_states = 3
        X = jnp.ones((100, 5))
        y = jnp.ones(100)

        transition_prob1 = sticky_transition_proba_init(
            n_states, X, y, random_key=jax.random.PRNGKey(123)
        )
        transition_prob2 = sticky_transition_proba_init(
            n_states, X, y, random_key=jax.random.PRNGKey(999)
        )

        # Should be identical regardless of random key
        assert jnp.allclose(transition_prob1, transition_prob2)


class TestUniformTransitionProbaInitialization:
    """Test sticky initialization for transition probabilities."""

    @pytest.mark.parametrize("n_states", [1, 2, 3, 5])
    def test_expected_output_shape(self, n_states):
        """Test that output shape is (n_states, n_states)."""
        X = jnp.ones((100, 5))
        y = jnp.ones(100)

        transition_prob = uniform_transition_proba_init(
            n_states, X, y, random_key=jax.random.PRNGKey(123)
        )

        assert transition_prob.shape == (n_states, n_states)
        assert jnp.array_equal(
            transition_prob, jnp.full((n_states, n_states), 1 / n_states)
        )

    @pytest.mark.parametrize(
        "X, y",
        [
            (np.ones((100, 5)), np.ones(100)),
            (jnp.ones((100, 5)), jnp.ones(100)),
        ],
    )
    def test_expected_output_type(self, X, y):
        """Test that output is a JAX array regardless of input type."""
        n_states = 2

        transition_prob = uniform_transition_proba_init(
            n_states, X, y, random_key=jax.random.PRNGKey(123)
        )

        assert isinstance(transition_prob, jnp.ndarray)
        assert jnp.array_equal(
            transition_prob, jnp.full((n_states, n_states), 1 / n_states)
        )


class TestUniformInitialProbaInitialization:
    """Test uniform initialization for initial state probabilities."""

    @pytest.mark.parametrize("n_states", [1, 2, 3, 5, 10])
    def test_expected_output_shape(self, n_states):
        """Test that output shape is (n_states,)."""
        X = jnp.ones((100, 5))
        y = jnp.ones(100)

        initial_prob = uniform_initial_proba_init(
            n_states, X, y, random_key=jax.random.PRNGKey(124)
        )

        assert initial_prob.shape == (n_states,)

    @pytest.mark.parametrize(
        "X, y",
        [
            (np.ones((100, 5)), np.ones(100)),
            (jnp.ones((100, 5)), jnp.ones(100)),
        ],
    )
    def test_expected_output_type(self, X, y):
        """Test that output is a JAX array regardless of input type."""
        n_states = 2

        initial_prob = uniform_initial_proba_init(
            n_states, X, y, random_key=jax.random.PRNGKey(124)
        )

        assert isinstance(initial_prob, jnp.ndarray)

    @pytest.mark.parametrize("n_states", [1, 2, 3, 5, 10])
    def test_uniform_distribution(self, n_states):
        """Test that all probabilities are equal (uniform distribution)."""
        X = jnp.ones((100, 5))
        y = jnp.ones(100)

        initial_prob = uniform_initial_proba_init(
            n_states, X, y, random_key=jax.random.PRNGKey(124)
        )

        # All values should be equal to 1/n_states
        expected_value = 1.0 / n_states
        assert jnp.allclose(initial_prob, expected_value)

    @pytest.mark.parametrize("n_states", [1, 2, 3, 5, 10])
    def test_sums_to_one(self, n_states):
        """Test that probabilities sum to 1."""
        X = jnp.ones((100, 5))
        y = jnp.ones(100)

        initial_prob = uniform_initial_proba_init(
            n_states, X, y, random_key=jax.random.PRNGKey(124)
        )

        # Should sum to 1
        assert jnp.isclose(jnp.sum(initial_prob), 1.0)

    def test_deterministic(self):
        """Test that output is deterministic (same across different calls)."""
        n_states = 3
        X = jnp.ones((100, 5))
        y = jnp.ones(100)

        initial_prob1 = uniform_initial_proba_init(
            n_states, X, y, random_key=jax.random.PRNGKey(124)
        )
        initial_prob2 = uniform_initial_proba_init(
            n_states, X, y, random_key=jax.random.PRNGKey(999)
        )

        # Should be identical regardless of random key
        assert jnp.allclose(initial_prob1, initial_prob2)


class TestGLMHMMInitialization:
    """Test full GLM-HMM parameter initialization."""

    @pytest.mark.parametrize("n_states", [1, 2, 3])
    @pytest.mark.parametrize("n_neurons", [1, 3])
    def test_default_initialization_shape(self, n_states, n_neurons):
        """Test that default initialization returns correct shapes."""
        n_samples = 100
        n_features = 5
        X = jnp.ones((n_samples, n_features))
        y = jnp.ones((n_samples, n_neurons)) if n_neurons > 1 else jnp.ones(n_samples)
        inverse_link = lambda x: x

        coef, intercept, scale, initial_prob, transition_prob = glm_hmm_initialization(
            n_states, X, y, inverse_link, random_key=jax.random.PRNGKey(123)
        )

        # Check shapes
        if n_neurons == 1:
            assert coef.shape == (n_features, n_states)
            assert intercept.shape == (n_states,)
            assert scale.shape == (n_states,)
        else:
            assert coef.shape == (n_features, n_neurons, n_states)
            assert intercept.shape == (n_neurons, n_states)
            assert scale.shape == (n_neurons, n_states)

        assert initial_prob.shape == (n_states,)
        assert transition_prob.shape == (n_states, n_states)

    def test_default_initialization_types(self):
        """Test that all outputs are JAX arrays."""
        n_states = 2
        X = jnp.ones((100, 5))
        y = jnp.ones(100)
        inverse_link = lambda x: x

        coef, intercept, scale, initial_prob, transition_prob = glm_hmm_initialization(
            n_states, X, y, inverse_link, random_key=jax.random.PRNGKey(123)
        )

        assert isinstance(coef, jnp.ndarray)
        assert isinstance(intercept, jnp.ndarray)
        assert isinstance(scale, jnp.ndarray)
        assert isinstance(initial_prob, jnp.ndarray)
        assert isinstance(transition_prob, jnp.ndarray)

    def test_default_initialization_values(self):
        """Test that default initialization uses expected default functions."""
        n_states = 3
        X = jnp.ones((100, 5))
        y = jnp.full(100, 2.0)
        inverse_link = lambda x: x

        coef, intercept, scale, initial_prob, transition_prob = glm_hmm_initialization(
            n_states, X, y, inverse_link, random_key=jax.random.PRNGKey(123)
        )

        # Check default values:
        # - coef: small random (< 0.01)
        assert jnp.abs(coef).max() < 0.01

        # - intercept: mean of y (2.0)
        assert jnp.allclose(intercept, 2.0)

        # - scale: ones
        assert jnp.all(scale == 1.0)

        # - initial_prob: uniform (1/n_states)
        assert jnp.allclose(initial_prob, 1.0 / n_states)

        # - transition_prob: sticky (diagonal 0.95)
        assert jnp.allclose(jnp.diag(transition_prob), 0.95)

    def test_custom_registry_with_mocks(self):
        """Test that custom registry functions are called with correct arguments."""
        n_states = 3
        n_features = 5
        n_samples = 100
        X = jnp.ones((n_samples, n_features))
        y = jnp.ones(n_samples)
        inverse_link = lambda x: x

        # Create mock tracking
        mock_calls = {
            "glm_params": [],
            "scale": [],
            "initial_prob": [],
            "transition_prob": [],
        }

        # Create mock functions with DISTINCTIVE outputs (zeros, ones, twos, threes, fours)
        def mock_glm_params(n_states, X, y, inverse_link_function, random_key):
            mock_calls["glm_params"].append(
                (n_states, X, y, inverse_link_function, random_key)
            )
            return jnp.zeros((n_features, n_states)), jnp.ones(n_states)

        def mock_scale(n_states, X, y, random_key):
            mock_calls["scale"].append((n_states, X, y, random_key))
            return jnp.full(n_states, 2.0)

        def mock_initial_prob(n_states, X, y, random_key):
            mock_calls["initial_prob"].append((n_states, X, y, random_key))
            return jnp.full(n_states, 3.0)

        def mock_transition_prob(n_states, X, y, random_key):
            mock_calls["transition_prob"].append((n_states, X, y, random_key))
            return jnp.full((n_states, n_states), 4.0)

        custom_registry = {
            "glm_params_init": mock_glm_params,
            "scale_init": mock_scale,
            "initial_proba_init": mock_initial_prob,
            "transition_proba_init": mock_transition_prob,
        }

        coef, intercept, scale, initial_prob, transition_prob = glm_hmm_initialization(
            n_states,
            X,
            y,
            inverse_link,
            random_key=jax.random.PRNGKey(123),
            initialization_funcs=custom_registry,
        )

        # Verify all mocks were called
        assert len(mock_calls["glm_params"]) == 1
        assert len(mock_calls["scale"]) == 1
        assert len(mock_calls["initial_prob"]) == 1
        assert len(mock_calls["transition_prob"]) == 1

        # Verify they were called with correct arguments
        call_args = mock_calls["glm_params"][0]
        assert call_args[0] == n_states  # n_states
        assert jnp.array_equal(call_args[1], X)  # X
        assert jnp.array_equal(call_args[2], y)  # y
        assert call_args[3] == inverse_link  # inverse_link_function

        # Verify that the distinctive outputs are actually used
        assert jnp.all(coef == 0.0)  # zeros
        assert jnp.all(intercept == 1.0)  # ones
        assert jnp.all(scale == 2.0)  # twos
        assert jnp.all(initial_prob == 3.0)  # threes
        assert jnp.all(transition_prob == 4.0)  # fours

    def test_partial_registry(self):
        """Test that partial registry merges with defaults."""
        n_states = 2
        n_features = 5
        X = jnp.ones((100, n_features))
        y = jnp.ones(100)
        inverse_link = lambda x: x

        # Create tracking for mock
        mock_called = {"called": False}

        # Only override glm_params_init
        def mock_glm_params(n_states, X, y, inverse_link_function, random_key):
            mock_called["called"] = True
            return jnp.zeros((n_features, n_states)), jnp.full(n_states, 5.0)

        partial_registry = {
            "glm_params_init": mock_glm_params,
        }

        coef, intercept, scale, initial_prob, transition_prob = glm_hmm_initialization(
            n_states,
            X,
            y,
            inverse_link,
            random_key=jax.random.PRNGKey(123),
            initialization_funcs=partial_registry,
        )

        # Mock was used for glm_params
        assert mock_called["called"]
        assert jnp.allclose(intercept, 5.0)  # From mock

        # Defaults were used for others
        assert jnp.all(scale == 1.0)  # Default ones_scale_init
        assert jnp.allclose(initial_prob, 0.5)  # Default uniform
        assert jnp.allclose(jnp.diag(transition_prob), 0.95)  # Default sticky

    def test_random_key_splitting(self):
        """Test that random key is properly split so each init function gets different subkey."""
        n_states = 3
        n_features = 5
        X = jnp.ones((100, n_features))
        y = jnp.ones(100)
        inverse_link = lambda x: x

        # Create mock functions that ALL return the SAME SHAPE (10,) array
        # This allows us to check that all outputs differ within a single call
        fixed_shape = (10,)

        def random_glm_params(n_states, X, y, inverse_link_function, random_key):
            # Split key to return two different arrays of fixed shape
            key1, key2 = jax.random.split(random_key)
            return jax.random.normal(key1, fixed_shape), jax.random.normal(
                key2, fixed_shape
            )

        def random_scale(n_states, X, y, random_key):
            return jax.random.normal(random_key, fixed_shape)

        def random_initial_prob(n_states, X, y, random_key):
            return jax.random.normal(random_key, fixed_shape)

        def random_transition_prob(n_states, X, y, random_key):
            return jax.random.normal(random_key, fixed_shape)

        random_registry = {
            "glm_params_init": random_glm_params,
            "scale_init": random_scale,
            "initial_proba_init": random_initial_prob,
            "transition_proba_init": random_transition_prob,
        }

        # Single call with one seed
        coef, intercept, scale, initial_prob, transition_prob = glm_hmm_initialization(
            n_states,
            X,
            y,
            inverse_link,
            random_key=jax.random.PRNGKey(123),
            initialization_funcs=random_registry,
        )

        # All outputs should differ from each other (different subkeys were used)
        for p1, p2 in itertools.combinations(
            [coef, intercept, scale, initial_prob, transition_prob], 2
        ):
            assert not jnp.allclose(p1, p2)

        # Run with different seed - all should differ from first call
        coef2, intercept2, scale2, initial_prob2, transition_prob2 = (
            glm_hmm_initialization(
                n_states,
                X,
                y,
                inverse_link,
                random_key=jax.random.PRNGKey(456),
                initialization_funcs=random_registry,
            )
        )

        assert not jnp.allclose(coef, coef2)
        assert not jnp.allclose(intercept, intercept2)
        assert not jnp.allclose(scale, scale2)
        assert not jnp.allclose(initial_prob, initial_prob2)
        assert not jnp.allclose(transition_prob, transition_prob2)

    def test_inverse_link_function_passed_to_glm_init(self):
        """Test that inverse_link_function is passed to glm_params_init."""
        n_states = 2
        X = jnp.ones((100, 5))
        y = jnp.full(100, 10.0)

        # Use exp inverse link
        inverse_link = jnp.exp

        _, intercept, _, _, _ = glm_hmm_initialization(
            n_states, X, y, inverse_link, random_key=jax.random.PRNGKey(123)
        )

        # Intercept should be log(mean(y)) = log(10) for exp link
        expected = jnp.log(10.0)
        assert jnp.allclose(intercept, expected)

    @pytest.mark.parametrize("n_neurons", [1, 3])
    def test_population_vs_single_neuron(self, n_neurons):
        """Test initialization works for both single neuron and population GLMs."""
        n_states = 2
        n_features = 5
        n_samples = 100
        X = jnp.ones((n_samples, n_features))
        y = jnp.ones((n_samples, n_neurons)) if n_neurons > 1 else jnp.ones(n_samples)
        inverse_link = lambda x: x

        coef, intercept, scale, initial_prob, transition_prob = glm_hmm_initialization(
            n_states, X, y, inverse_link, random_key=jax.random.PRNGKey(123)
        )

        # All parameters should be properly shaped
        if n_neurons == 1:
            assert coef.ndim == 2  # (n_features, n_states)
            assert intercept.ndim == 1  # (n_states,)
            assert scale.ndim == 1  # (n_states,)
        else:
            assert coef.ndim == 3  # (n_features, n_neurons, n_states)
            assert intercept.ndim == 2  # (n_neurons, n_states)
            assert scale.ndim == 2  # (n_neurons, n_states)

    def test_returns_tuple_of_five_elements(self):
        """Test that function returns exactly 5 elements."""
        n_states = 2
        X = jnp.ones((100, 5))
        y = jnp.ones(100)
        inverse_link = lambda x: x

        result = glm_hmm_initialization(
            n_states, X, y, inverse_link, random_key=jax.random.PRNGKey(123)
        )

        assert isinstance(result, tuple)
        assert len(result) == 5

    @pytest.mark.parametrize(
        "registry",
        [
            {"glm_params_init": "random"},
            {"scale_init": "ones"},
            {"transition_proba_init": "sticky"},
            {"initial_proba_init": "uniform"},
        ],
    )
    def test_string_lookup_in_registry(self, registry):
        """Test that string lookups work for built-in functions."""
        n_states = 2
        X = jnp.ones((100, 5))
        y = jnp.ones(100)
        inverse_link = lambda x: x

        # Should not raise
        result = glm_hmm_initialization(
            n_states,
            X,
            y,
            inverse_link,
            random_key=jax.random.PRNGKey(123),
            initialization_funcs=registry,
        )

        assert len(result) == 5

    def test_init_kwargs_prob_stay(self):
        """Test that prob_stay kwarg is passed to transition_proba_init."""
        n_states = 3
        X = jnp.ones((100, 5))
        y = jnp.ones(100)
        inverse_link = lambda x: x

        # Use custom prob_stay
        custom_prob_stay = 0.8
        init_kwargs = {"transition_proba_init": {"prob_stay": custom_prob_stay}}

        _, _, _, _, transition_prob = glm_hmm_initialization(
            n_states,
            X,
            y,
            inverse_link,
            random_key=jax.random.PRNGKey(123),
            initialization_kwargs=init_kwargs,
        )

        # Diagonal should be prob_stay
        assert jnp.allclose(jnp.diag(transition_prob), custom_prob_stay)

        # Off-diagonal should be (1 - prob_stay) / (n_states - 1)
        expected_off_diag = (1 - custom_prob_stay) / (n_states - 1)
        off_diag = transition_prob[~jnp.eye(n_states, dtype=bool)]
        assert jnp.allclose(off_diag, expected_off_diag)

    def test_init_kwargs_std_dev(self):
        """Test that std_dev kwarg is passed to glm_params_init."""
        n_states = 2
        n_features = 5
        X = jnp.ones((100, n_features))
        y = jnp.ones(100)
        inverse_link = lambda x: x

        # Use large std_dev to make effect measurable
        large_std_dev = 1.0
        init_kwargs = {"glm_params_init": {"std_dev": large_std_dev}}

        coef_large, _, _, _, _ = glm_hmm_initialization(
            n_states,
            X,
            y,
            inverse_link,
            random_key=jax.random.PRNGKey(123),
            initialization_kwargs=init_kwargs,
        )

        # Use small std_dev
        small_std_dev = 0.001
        init_kwargs = {"glm_params_init": {"std_dev": small_std_dev}}

        coef_small, _, _, _, _ = glm_hmm_initialization(
            n_states,
            X,
            y,
            inverse_link,
            random_key=jax.random.PRNGKey(123),
            initialization_kwargs=init_kwargs,
        )

        # Large std_dev should produce larger magnitude coefficients
        assert jnp.abs(coef_large).max() > jnp.abs(coef_small).max()
        # Small std_dev should be close to 0
        assert jnp.abs(coef_small).max() < 0.01

    def test_init_kwargs_multiple_functions(self):
        """Test that init_kwargs can pass kwargs to multiple init functions."""
        n_states = 3
        X = jnp.ones((100, 5))
        y = jnp.ones(100)
        inverse_link = lambda x: x

        init_kwargs = {
            "glm_params_init": {"std_dev": 0.5},
            "transition_proba_init": {"prob_stay": 0.7},
        }

        coef, _, _, _, transition_prob = glm_hmm_initialization(
            n_states,
            X,
            y,
            inverse_link,
            random_key=jax.random.PRNGKey(123),
            initialization_kwargs=init_kwargs,
        )

        # Check std_dev effect
        assert jnp.abs(coef).max() > 0.01  # Larger than default 0.001

        # Check prob_stay effect
        assert jnp.allclose(jnp.diag(transition_prob), 0.7)

    def test_init_kwargs_empty_dict(self):
        """Test that empty init_kwargs dict works (uses defaults)."""
        n_states = 2
        X = jnp.ones((100, 5))
        y = jnp.ones(100)
        inverse_link = lambda x: x

        # Empty dict should behave like None
        result_empty = glm_hmm_initialization(
            n_states,
            X,
            y,
            inverse_link,
            random_key=jax.random.PRNGKey(123),
            initialization_kwargs={},
        )

        result_none = glm_hmm_initialization(
            n_states,
            X,
            y,
            inverse_link,
            random_key=jax.random.PRNGKey(123),
            initialization_kwargs=None,
        )

        # Results should be identical
        for r1, r2 in zip(result_empty, result_none):
            assert jnp.allclose(r1, r2)


class TestResolveDirichletPriors:
    """Test _resolve_dirichlet_priors validation function."""

    def test_none_input_returns_none(self):
        """Test that None input returns None."""
        result = _resolve_dirichlet_priors(None, (3,))
        assert result is None

    @pytest.mark.parametrize(
        "alphas, expected_shape",
        [
            (np.array([1.0, 1.0, 1.0]), (3,)),
            (np.array([2.0, 3.0]), (2,)),
            (jnp.array([[1.0, 2.0], [3.0, 4.0]]), (2, 2)),
        ],
    )
    def test_valid_array_input(self, alphas, expected_shape):
        """Test that valid array inputs are converted to JAX arrays."""
        result = _resolve_dirichlet_priors(alphas, expected_shape)
        assert isinstance(result, jnp.ndarray)
        assert result.shape == expected_shape

    def test_shape_mismatch_raises_value_error(self):
        """Test that shape mismatch raises ValueError."""
        alphas = jnp.array([1.0, 2.0, 3.0])
        expected_shape = (2,)

        with pytest.raises(ValueError, match="must have shape"):
            _resolve_dirichlet_priors(alphas, expected_shape)

    def test_values_less_than_one_raises_value_error(self):
        """Test that alpha values < 1 raise ValueError."""
        alphas = jnp.array([1.0, 0.5, 2.0])
        expected_shape = (3,)

        with pytest.raises(ValueError, match="must be >= 1"):
            _resolve_dirichlet_priors(alphas, expected_shape)

    def test_invalid_type_raises_type_error(self):
        """Test that invalid types raise TypeError."""
        alphas = "invalid"
        expected_shape = (3,)

        with pytest.raises(TypeError, match="Invalid type"):
            _resolve_dirichlet_priors(alphas, expected_shape)


class TestResolveInitFunc:
    """Test _resolve_init_func validation function."""

    def test_none_returns_default(self):
        """Test that None returns the default function."""
        result = _resolve_init_func("glm_params_init", None)
        assert result is random_glm_params_init

    @pytest.mark.parametrize(
        "func_name, string_name, expected_func",
        [
            ("glm_params_init", "random", random_glm_params_init),
            ("scale_init", "ones", ones_scale_init),
            ("transition_proba_init", "sticky", sticky_transition_proba_init),
            ("initial_proba_init", "uniform", uniform_initial_proba_init),
        ],
    )
    def test_string_lookup(self, func_name, string_name, expected_func):
        """Test that string names resolve to correct functions."""
        result = _resolve_init_func(func_name, string_name)
        assert result is expected_func

    def test_unknown_string_raises_value_error(self):
        """Test that unknown string names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown initialization method"):
            _resolve_init_func("glm_params_init", "unknown_method")

    def test_callable_with_correct_signature_accepted(self):
        """Test that callable with correct signature is accepted."""

        def custom_init(n_states, X, y, random_key):
            return jnp.ones(n_states)

        result = _resolve_init_func("scale_init", custom_init)
        assert result is custom_init

    def test_callable_too_few_params_raises_value_error(self):
        """Test that callable with too few parameters raises ValueError."""

        def bad_init(n_states, X):
            return jnp.ones(n_states)

        with pytest.raises(ValueError, match="must have at least"):
            _resolve_init_func("scale_init", bad_init)

    def test_callable_extra_params_without_defaults_raises_value_error(self):
        """Test that extra parameters without defaults raise ValueError."""

        def bad_init(n_states, X, y, random_key, extra_param):
            return jnp.ones(n_states)

        with pytest.raises(ValueError, match="must have default values"):
            _resolve_init_func("scale_init", bad_init)

    def test_callable_extra_params_with_defaults_accepted(self):
        """Test that extra parameters with defaults are accepted."""

        def custom_init(n_states, X, y, random_key, extra_param=1.0):
            return jnp.ones(n_states)

        result = _resolve_init_func("scale_init", custom_init)
        assert result is custom_init

    def test_glm_params_init_requires_five_params(self):
        """Test that glm_params_init requires 5 parameters."""

        def bad_glm_init(n_states, X, y, random_key):
            return jnp.ones(n_states), jnp.ones(n_states)

        with pytest.raises(ValueError, match="must have at least 5"):
            _resolve_init_func("glm_params_init", bad_glm_init)

    def test_invalid_type_raises_type_error(self):
        """Test that invalid types raise TypeError."""
        with pytest.raises(TypeError, match="Invalid initialization function"):
            _resolve_init_func("scale_init", 123)


class TestResolveInitFuncsRegistry:
    """Test _resolve_init_funcs_registry validation function."""

    def test_none_returns_defaults(self):
        """Test that None returns default registry."""
        result = _resolve_init_funcs_registry(None)
        assert result["glm_params_init"] is random_glm_params_init
        assert result["scale_init"] is ones_scale_init
        assert result["transition_proba_init"] is sticky_transition_proba_init
        assert result["initial_proba_init"] is uniform_initial_proba_init

    def test_invalid_key_raises_key_error(self):
        """Test that invalid registry keys raise KeyError."""
        invalid_registry = {"invalid_key": lambda: None}

        with pytest.raises(KeyError, match="Invalid key"):
            _resolve_init_funcs_registry(invalid_registry)

    def test_valid_keys_with_one_invalid_raises_key_error(self):
        """Test that registry with valid keys plus one invalid key raises KeyError."""
        mixed_registry = {
            "glm_params_init": random_glm_params_init,
            "scale_init": ones_scale_init,
            "transition_proba_init": sticky_transition_proba_init,
            "initial_proba_init": uniform_initial_proba_init,
            "invalid_key": lambda: None,
        }

        with pytest.raises(KeyError, match="Invalid key"):
            _resolve_init_funcs_registry(mixed_registry)

    def test_partial_registry_merges_with_defaults(self):
        """Test that partial registry is merged with defaults."""

        def custom_scale(n_states, X, y, random_key):
            return jnp.full(n_states, 2.0)

        partial_registry = {"scale_init": custom_scale}

        result = _resolve_init_funcs_registry(partial_registry)

        # Custom function used
        assert result["scale_init"] is custom_scale
        # Defaults for others
        assert result["glm_params_init"] is random_glm_params_init
        assert result["transition_proba_init"] is sticky_transition_proba_init
        assert result["initial_proba_init"] is uniform_initial_proba_init


class TestIsNativeInitRegistry:
    """Test _is_native_init_registry helper function."""

    def test_native_registry_returns_true(self):
        """Test that registry with all native functions returns True."""
        native_registry = {
            "glm_params_init": random_glm_params_init,
            "scale_init": ones_scale_init,
            "transition_proba_init": sticky_transition_proba_init,
            "initial_proba_init": uniform_initial_proba_init,
        }
        assert _is_native_init_registry(native_registry) is True

    def test_partial_native_registry_returns_true(self):
        """Test that partial registry with native functions returns True."""
        partial_registry = {
            "glm_params_init": random_glm_params_init,
            "scale_init": ones_scale_init,
        }
        assert _is_native_init_registry(partial_registry) is True

    def test_custom_function_returns_false(self):
        """Test that registry with custom function returns False."""

        def custom_scale(n_states, X, y, random_key):
            return jnp.ones(n_states)

        custom_registry = {
            "scale_init": custom_scale,
        }
        assert _is_native_init_registry(custom_registry) is False


# =============================================================================
# Mock infrastructure for testing init kwargs validation
# =============================================================================


# Template functions for creating mocks with create_autospec
# GLM params init requires 5 mandatory params (n_states, X, y, inverse_link_function, key)
def _glm_template_no_extra(n_states, X, y, inverse_link_function, random_key):
    pass


def _glm_template_one_extra(
    n_states, X, y, inverse_link_function, random_key, param1=None
):
    pass


def _glm_template_two_extra(
    n_states, X, y, inverse_link_function, random_key, param1=None, param2=None
):
    pass


def _glm_template_three_extra(
    n_states, X, y, inverse_link_function, random_key, alpha=None, beta=None, gamma=None
):
    pass


def _glm_template_special(
    n_states, X, y, inverse_link_function, random_key, my_special_param=None
):
    pass


# Other init funcs require 4 mandatory params (n_states, X, y, key)
def _other_template_no_extra(n_states, X, y, random_key):
    pass


def _other_template_one_extra(n_states, X, y, random_key, param1=None):
    pass


def _other_template_two_extra(n_states, X, y, random_key, param1=None, param2=None):
    pass


def _other_template_three_extra(
    n_states, X, y, random_key, alpha=None, beta=None, gamma=None
):
    pass


def _other_template_special(n_states, X, y, random_key, my_special_param=None):
    pass


# Template registry for mock creation
_GLM_TEMPLATES = {
    "no_extra": _glm_template_no_extra,
    "one_extra": _glm_template_one_extra,
    "two_extra": _glm_template_two_extra,
    "three_extra": _glm_template_three_extra,
    "special": _glm_template_special,
}

_OTHER_TEMPLATES = {
    "no_extra": _other_template_no_extra,
    "one_extra": _other_template_one_extra,
    "two_extra": _other_template_two_extra,
    "three_extra": _other_template_three_extra,
    "special": _other_template_special,
}

# Valid kwargs for each template type
MOCK_VALID_KWARGS = {
    "one_extra": {"param1": 0.5},
    "two_extra": {"param1": 0.5, "param2": 0.5},
    "three_extra": {"alpha": 0.5, "beta": 0.5, "gamma": 0.5},
    "special": {"my_special_param": 0.5},
}


def _get_mock(func_name, template_type):
    """Get a mock function with the appropriate signature."""
    templates = _GLM_TEMPLATES if func_name == "glm_params_init" else _OTHER_TEMPLATES
    return create_autospec(templates[template_type], return_value=None)


def _get_mock_registry(template_type="one_extra"):
    """Get a mock init function registry where all funcs have the same template type."""
    return {
        func_name: _get_mock(func_name, template_type)
        for func_name in DEFAULT_INIT_FUNCTION
    }


@pytest.mark.parametrize(
    "func_name",
    ["glm_params_init", "scale_init", "initial_proba_init", "transition_proba_init"],
)
class TestResolveInitKwargs:
    """Test _resolve_init_kwargs validation function."""

    def test_none_returns_empty_dict(self, func_name):
        """Test that None kwargs returns empty dict for each func type."""
        mock_func = _get_mock(func_name, "no_extra")
        result = _resolve_init_kwargs(func_name, mock_func, None)
        assert result == {}

    def test_empty_dict_returns_empty_dict(self, func_name):
        """Test that empty kwargs returns empty dict for each func type."""
        mock_func = _get_mock(func_name, "no_extra")
        result = _resolve_init_kwargs(func_name, mock_func, {})
        assert result == {}

    @pytest.mark.parametrize(
        "template_type,extra_kwargs",
        [
            ("one_extra", {"param1": 0.5}),
            ("two_extra", {"param1": 0.5, "param2": 0.5}),
            ("three_extra", {"alpha": 0.5, "beta": 0.5, "gamma": 0.5}),
        ],
    )
    def test_valid_kwargs_returned_unchanged(
        self, func_name, template_type, extra_kwargs
    ):
        """Test that valid kwargs are returned unchanged for each func type."""
        mock_func = _get_mock(func_name, template_type)
        result = _resolve_init_kwargs(func_name, mock_func, extra_kwargs)
        assert result == extra_kwargs

    def test_invalid_kwarg_raises_value_error(self, func_name):
        """Test that unrecognized kwargs raise ValueError for each func type."""
        mock_func = _get_mock(func_name, "one_extra")
        with pytest.raises(ValueError, match="Invalid keyword argument"):
            _resolve_init_kwargs(func_name, mock_func, {"invalid_param": 0.5})

    def test_error_message_shows_available_params(self, func_name):
        """Test that error message lists available parameters."""
        mock_func = _get_mock(func_name, "special")
        with pytest.raises(ValueError, match="my_special_param"):
            _resolve_init_kwargs(func_name, mock_func, {"invalid": 0.5})

    def test_function_with_no_extra_params(self, func_name):
        """Test kwargs validation for function with no optional params."""
        mock_func = _get_mock(func_name, "no_extra")
        with pytest.raises(ValueError, match="no extra keyword arguments"):
            _resolve_init_kwargs(func_name, mock_func, {"any_param": 1.0})

    def test_subset_of_kwargs_valid(self, func_name):
        """Test that providing a subset of available kwargs is valid."""
        mock_func = _get_mock(func_name, "three_extra")
        # Only provide one of the three available kwargs
        kwargs = {"beta": 0.5}
        result = _resolve_init_kwargs(func_name, mock_func, kwargs)
        assert result == kwargs


@pytest.mark.parametrize(
    "func_name",
    ["glm_params_init", "scale_init", "initial_proba_init", "transition_proba_init"],
)
class TestResolveInitKwargsRegistry:
    """Test _resolve_init_kwargs_registry validation function.

    Uses mock registries so all functions have kwargs to test.
    """

    def test_none_returns_empty_kwargs_for_all(self, func_name):
        """Test that None returns empty dict for each function."""
        mock_registry = _get_mock_registry("one_extra")
        result = _resolve_init_kwargs_registry(None, mock_registry)
        assert all(v == {} for v in result.values())
        assert set(result.keys()) == set(mock_registry.keys())

    def test_empty_dict_returns_empty_kwargs_for_all(self, func_name):
        """Test that empty dict returns empty dict for each function."""
        mock_registry = _get_mock_registry("one_extra")
        result = _resolve_init_kwargs_registry({}, mock_registry)
        assert all(v == {} for v in result.values())

    def test_single_func_kwargs_fills_others_empty(self, func_name):
        """Test that setting kwargs for one func fills others with empty dicts."""
        mock_registry = _get_mock_registry("one_extra")
        valid_kwargs = MOCK_VALID_KWARGS["one_extra"]

        result = _resolve_init_kwargs_registry({func_name: valid_kwargs}, mock_registry)

        assert result[func_name] == valid_kwargs
        for other_func in mock_registry:
            if other_func != func_name:
                assert result[other_func] == {}

    def test_invalid_kwarg_per_func_raises(self, func_name):
        """Test that invalid kwargs raise ValueError for each func."""
        mock_registry = _get_mock_registry("one_extra")
        with pytest.raises(ValueError, match="Invalid keyword argument"):
            _resolve_init_kwargs_registry(
                {func_name: {"totally_invalid_param": 0.5}}, mock_registry
            )

    def test_kwargs_passed_to_correct_func(self, func_name):
        """Test that kwargs are validated against the correct function in registry."""
        mock_registry = _get_mock_registry("one_extra")
        valid_kwargs = MOCK_VALID_KWARGS["one_extra"]

        # Should succeed - valid kwargs for this func
        result = _resolve_init_kwargs_registry({func_name: valid_kwargs}, mock_registry)
        assert result[func_name] == valid_kwargs


class TestResolveInitKwargsRegistryPairs:
    """Test _resolve_init_kwargs_registry with pairs and all functions."""

    @pytest.mark.parametrize(
        "func_pair",
        list(
            itertools.combinations(
                [
                    "glm_params_init",
                    "scale_init",
                    "initial_proba_init",
                    "transition_proba_init",
                ],
                2,
            )
        ),
    )
    def test_pairs_of_func_kwargs(self, func_pair):
        """Test setting kwargs for pairs of functions."""
        mock_registry = _get_mock_registry("one_extra")
        valid_kwargs = MOCK_VALID_KWARGS["one_extra"]
        func1, func2 = func_pair

        input_kwargs = {func1: valid_kwargs, func2: valid_kwargs}
        result = _resolve_init_kwargs_registry(input_kwargs, mock_registry)

        # Check the funcs we set
        assert result[func1] == valid_kwargs
        assert result[func2] == valid_kwargs

        # Check others are empty
        for fn in mock_registry:
            if fn not in func_pair:
                assert result[fn] == {}

    def test_all_funcs_kwargs_at_once(self):
        """Test setting kwargs for all functions."""
        mock_registry = _get_mock_registry("one_extra")
        valid_kwargs = MOCK_VALID_KWARGS["one_extra"]

        input_kwargs = {fn: valid_kwargs for fn in mock_registry}
        result = _resolve_init_kwargs_registry(input_kwargs, mock_registry)

        for fn in mock_registry:
            assert result[fn] == valid_kwargs

    def test_all_funcs_with_mixed_valid_empty(self):
        """Test setting all funcs, some with kwargs some empty."""
        mock_registry = _get_mock_registry("one_extra")
        valid_kwargs = MOCK_VALID_KWARGS["one_extra"]

        input_kwargs = {
            "glm_params_init": valid_kwargs,
            "scale_init": {},
            "initial_proba_init": {},
            "transition_proba_init": valid_kwargs,
        }

        result = _resolve_init_kwargs_registry(input_kwargs, mock_registry)

        assert result["glm_params_init"] == valid_kwargs
        assert result["scale_init"] == {}
        assert result["initial_proba_init"] == {}
        assert result["transition_proba_init"] == valid_kwargs
