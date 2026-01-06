"""Tests for glm_hmm/initialize_parameters.py"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nemos.glm_hmm.initialize_parameters import (
    ones_scale_init,
    random_glm_params_init,
    sticky_transition_proba_init,
    uniform_initial_proba_init,
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
