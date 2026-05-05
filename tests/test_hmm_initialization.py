"""Tests for hmm/initialize_parameters.py"""

from contextlib import nullcontext as does_not_raise

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from sklearn.datasets import make_blobs

from nemos.hmm.initialize_parameters import (
    DEFAULT_INIT_FUNCTIONS,
    KMeansInitializer,
    _resolve_dirichlet_priors,
    generate_hmm_initial_params,
    kmeans_initial_proba_init,
    kmeans_transition_proba_init,
    random_initial_proba_init,
    random_transition_proba_init,
    setup_hmm_initialization,
    sticky_transition_proba_init,
    uniform_initial_proba_init,
    uniform_transition_proba_init,
)


@pytest.fixture
def use_method_for_test(method, method_subset):
    if method not in method_subset:
        pytest.skip()


@pytest.mark.parametrize(
    "method",
    [
        uniform_initial_proba_init,
        random_initial_proba_init,
    ],
)
class TestInitialProbaInitialization:
    """Test uniform initialization for initial state probabilities."""

    @pytest.mark.parametrize("n_states", [1, 2, 3, 5, 10])
    def test_expected_output_shape(self, method, n_states):
        """Test that output shape is (n_states,)."""

        initial_prob = method(n_states, random_key=jax.random.PRNGKey(124))

        assert initial_prob.shape == (n_states,)

    def test_expected_output_type(self, method):
        """Test that output is a JAX array regardless of input type."""
        n_states = 2

        initial_prob = method(n_states, random_key=jax.random.PRNGKey(124))

        assert isinstance(initial_prob, jnp.ndarray)

    @pytest.mark.parametrize("n_states", [1, 2, 3, 5, 10])
    def test_sums_to_one(self, method, n_states):
        """Test that probabilities sum to 1."""

        initial_prob = method(n_states, random_key=jax.random.PRNGKey(124))

        # Should sum to 1
        assert jnp.isclose(jnp.sum(initial_prob), 1.0)

    @pytest.mark.parametrize("method_subset", [[uniform_initial_proba_init]])
    @pytest.mark.parametrize("n_states", [1, 2, 3, 5, 10])
    def test_uniform_distribution(
        self, method, method_subset, use_method_for_test, n_states
    ):
        """Test that all probabilities are equal (uniform distribution)."""

        initial_prob = method(n_states, random_key=jax.random.PRNGKey(124))

        # All values should be equal to 1/n_states
        expected_value = 1.0 / n_states
        assert jnp.allclose(initial_prob, expected_value)

    @pytest.mark.parametrize("method_subset", [[uniform_initial_proba_init]])
    def test_deterministic(self, method, method_subset, use_method_for_test):
        """Test that output is deterministic (same across different calls)."""
        n_states = 3

        initial_prob1 = method(n_states, random_key=jax.random.PRNGKey(124))
        initial_prob2 = method(n_states, random_key=jax.random.PRNGKey(999))

        # Should be identical regardless of random key
        assert jnp.allclose(initial_prob1, initial_prob2)

    @pytest.mark.parametrize("method_subset", [[random_initial_proba_init]])
    def test_non_deterministic(self, method, method_subset, use_method_for_test):
        """Test that output is non-deterministic (different across different calls)."""
        n_states = 3

        initial_prob1 = method(n_states, random_key=jax.random.PRNGKey(124))
        initial_prob2 = method(n_states, random_key=jax.random.PRNGKey(999))

        assert not jnp.allclose(initial_prob1, initial_prob2)


@pytest.mark.parametrize(
    "method",
    [
        sticky_transition_proba_init,
        uniform_transition_proba_init,
        random_transition_proba_init,
    ],
)
class TestTransitionProbaInitialization:
    """Test initialization for transition probabilities."""

    @pytest.mark.parametrize("n_states", [1, 2, 3, 5])
    def test_expected_output_shape(self, method, n_states):
        """Test that output shape is (n_states, n_states)."""
        transition_prob = method(n_states, random_key=jax.random.PRNGKey(123))

        assert transition_prob.shape == (n_states, n_states)

    def test_expected_output_type(self, method):
        """Test that output is a JAX array regardless of input type."""
        n_states = 2

        transition_prob = method(n_states, random_key=jax.random.PRNGKey(123))

        assert isinstance(transition_prob, jnp.ndarray)

    @pytest.mark.parametrize("n_states", [2, 3, 5])
    def test_rows_sum_to_one(self, method, n_states):
        """Test that each row sums to 1 (valid probability distribution)."""

        transition_prob = method(n_states, random_key=jax.random.PRNGKey(123))

        # Each row should sum to 1
        row_sums = jnp.sum(transition_prob, axis=1)
        assert jnp.allclose(row_sums, 1.0)

    @pytest.mark.parametrize("method_subset", [[sticky_transition_proba_init]])
    @pytest.mark.parametrize("n_states", [2, 3, 5])
    def test_off_diagonal_values(
        self, method, method_subset, use_method_for_test, n_states
    ):
        """Test that off-diagonal values are (1 - prob_stay) / (n_states - 1)."""
        prob_stay = 0.95

        transition_prob = method(
            n_states, random_key=jax.random.PRNGKey(123), prob_stay=prob_stay
        )

        # Off-diagonal should be (1 - prob_stay) / (n_states - 1)
        expected_off_diag = (1 - prob_stay) / (n_states - 1)

        # Check all off-diagonal elements
        for i in range(n_states):
            for j in range(n_states):
                if i != j:
                    assert jnp.isclose(transition_prob[i, j], expected_off_diag)

    @pytest.mark.parametrize("method_subset", [[sticky_transition_proba_init]])
    def test_single_state_edge_case(self, method, method_subset, use_method_for_test):
        """Test that n_states=1 returns [[1]]."""
        n_states = 1
        prob_stay = 0.95

        transition_prob = method(
            n_states, random_key=jax.random.PRNGKey(123), prob_stay=prob_stay
        )

        # For single state, implementation returns [[prob_stay]] (not normalized)
        assert transition_prob.shape == (1, 1)
        assert jnp.isclose(transition_prob[0, 0], 1.0)

    @pytest.mark.parametrize("method_subset", [[sticky_transition_proba_init]])
    @pytest.mark.parametrize("prob_stay", [0.8, 0.9, 0.95, 0.99])
    def test_custom_prob_stay(
        self, method, method_subset, use_method_for_test, prob_stay
    ):
        """Test that custom prob_stay values work correctly."""
        n_states = 3

        transition_prob = method(
            n_states, random_key=jax.random.PRNGKey(123), prob_stay=prob_stay
        )

        # Diagonal should match custom prob_stay
        diagonal = jnp.diag(transition_prob)
        assert jnp.allclose(diagonal, prob_stay)

    @pytest.mark.parametrize(
        "method_subset", [[sticky_transition_proba_init, uniform_transition_proba_init]]
    )
    def test_deterministic(self, method, method_subset, use_method_for_test):
        """Test that output is deterministic (same across different calls)."""
        n_states = 3

        transition_prob1 = method(n_states, random_key=jax.random.PRNGKey(123))
        transition_prob2 = method(n_states, random_key=jax.random.PRNGKey(999))

        # Should be identical regardless of random key
        assert jnp.allclose(transition_prob1, transition_prob2)

    @pytest.mark.parametrize("method_subset", [[random_transition_proba_init]])
    def test_non_deterministic(self, method, method_subset, use_method_for_test):
        """Test that output is deterministic (same across different calls)."""
        n_states = 3

        transition_prob1 = method(n_states, random_key=jax.random.PRNGKey(123))
        transition_prob2 = method(n_states, random_key=jax.random.PRNGKey(999))

        assert not jnp.allclose(transition_prob1, transition_prob2)


def generate_kmeans_data(n_states=3, min_prob=0.02):
    n_samples = n_states * 5000
    is_new_session = np.zeros((n_samples,), dtype=bool)
    is_new_session[0::340] = True

    # generate data from sklearn and sort it to create temporal dynamics
    X, y = make_blobs(
        n_samples=np.repeat(5000, n_states),
        random_state=170,
        cluster_std=0.5,
        n_features=4,
    )
    y2 = y + np.repeat(np.arange(0, n_states * int(n_samples / 100), n_states), 100)
    y = y[np.argsort(y2)]
    y_pop = np.concatenate([y[:, None], np.ones_like(y[:, None])], axis=-1)
    X = X[np.argsort(y2), :]
    X_tree = {"feature1": X[:, :2], "feature2": X[:, 2:]}

    state = jax.nn.one_hot(y, n_states)
    initial_probability = state[is_new_session].sum(axis=0)
    initial_probability = (initial_probability / initial_probability.sum()) + min_prob
    initial_probability = initial_probability / initial_probability.sum()

    transition_probability = (
        state[:-1][~is_new_session[1:]].T @ state[1:][~is_new_session[1:]]
    )
    transition_probability = (
        transition_probability / transition_probability.sum(axis=1, keepdims=True)
    ) + min_prob
    transition_probability = transition_probability / transition_probability.sum(
        axis=1, keepdims=True
    )
    return (
        X,
        X_tree,
        y,
        y_pop,
        is_new_session,
        initial_probability,
        transition_probability,
    )


class TestKMeansInitialization:
    """Test k-means initialization for HMM parameters."""

    @pytest.mark.parametrize("n_states", [2, 3, 5])
    @pytest.mark.parametrize("pytree", [0, 1])
    @pytest.mark.parametrize("population", [0, 1])
    def test_kmeans_initializer(self, n_states, pytree, population):
        X, y = [[], []], [[], []]
        (
            X[0],
            X[1],
            y[0],
            y[1],
            is_new_session,
            expected_initial_prob,
            expected_transition_prob,
        ) = generate_kmeans_data(n_states)
        initializer = KMeansInitializer(
            n_states,
            X[pytree],
            y[population],
            is_new_session,
            random_key=jax.random.PRNGKey(123),
        )
        initial_prob, transition_prob = (
            initializer.initial_probability(),
            initializer.transition_probability(),
        )

        assert np.allclose(np.sort(initial_prob), np.sort(expected_initial_prob))
        assert np.allclose(
            np.sort(transition_prob.ravel()), np.sort(expected_transition_prob.ravel())
        )

    @pytest.mark.parametrize("n_states", [2, 3, 5])
    @pytest.mark.parametrize("pytree", [0, 1])
    @pytest.mark.parametrize("population", [0, 1])
    def test_output_shape_and_type(self, n_states, pytree, population):
        X, y = [[], []], [[], []]
        X[0], X[1], y[0], y[1], is_new_session, _, _ = generate_kmeans_data(n_states)
        initial_prob = kmeans_initial_proba_init(
            n_states, X[pytree], y[population], is_new_session
        )
        transition_prob = kmeans_transition_proba_init(
            n_states, X[pytree], y[population], is_new_session
        )

        assert initial_prob.shape == (n_states,)
        assert transition_prob.shape == (n_states, n_states)
        assert isinstance(initial_prob, jnp.ndarray)
        assert isinstance(transition_prob, jnp.ndarray)

    def test_nondeterministic_output(self):
        """Test that k-means initialization is non-deterministic across different random keys."""
        # the sorted values will be the same, but the state identities will differ
        n_states = 5
        X, _, y, _, is_new_session, _, _ = generate_kmeans_data(n_states)

        initial_prob1 = kmeans_initial_proba_init(
            n_states, X, y, is_new_session, random_key=jax.random.PRNGKey(1)
        )
        transition_prob1 = kmeans_transition_proba_init(
            n_states, X, y, is_new_session, random_key=jax.random.PRNGKey(1)
        )

        initial_prob2 = kmeans_initial_proba_init(
            n_states, X, y, is_new_session, random_key=jax.random.PRNGKey(2)
        )
        transition_prob2 = kmeans_transition_proba_init(
            n_states, X, y, is_new_session, random_key=jax.random.PRNGKey(2)
        )

        assert not jnp.allclose(initial_prob1, initial_prob2)
        assert not jnp.allclose(transition_prob1, transition_prob2)

    def test_shared_initializer(self):
        """Test that k-means initializer is used when specified in setup_hmm_initialization."""
        n_states = 5
        X, _, y, _, is_new_session, _, _ = generate_kmeans_data(n_states)
        default_initializer = KMeansInitializer(n_states, X, y, is_new_session)
        used_initializer = KMeansInitializer(
            n_states, X, y, is_new_session, random_key=jax.random.PRNGKey(2)
        )
        # initial probability
        init_prob = kmeans_initial_proba_init(
            n_states, X, y, is_new_session, initializer=used_initializer
        )
        assert not jnp.allclose(init_prob, default_initializer.initial_probability())
        assert jnp.allclose(init_prob, used_initializer.initial_probability())
        # transition probability
        transition_prob = kmeans_transition_proba_init(
            n_states, X, y, is_new_session, initializer=used_initializer
        )
        assert not jnp.allclose(
            transition_prob, default_initializer.transition_probability()
        )
        assert jnp.allclose(transition_prob, used_initializer.transition_probability())


class TestSetupHMMInitialization:
    """Test setup_hmm_initialization function for validating initialization protocols"""

    @pytest.mark.parametrize(
        "init_str, expectation, method",
        [
            ("uniform", does_not_raise(), uniform_initial_proba_init),
            ("random", does_not_raise(), random_initial_proba_init),
            ("kmeans", does_not_raise(), kmeans_initial_proba_init),
            (None, does_not_raise(), DEFAULT_INIT_FUNCTIONS["initial_proba_init"]),
            (
                "invalid",
                pytest.raises(ValueError, match="Invalid initialization"),
                None,
            ),
            (
                ["invalid"],
                pytest.raises(TypeError, match="either a string or a callable"),
                None,
            ),
        ],
    )
    def test_initial_proba_init_str(self, init_str, expectation, method):
        with expectation:
            init_funcs = setup_hmm_initialization(initial_proba_init=init_str)
            # check that the correct function is set in init_funcs
            assert init_funcs["initial_proba_init"] == method

    @pytest.mark.parametrize(
        "init_func, expectation",
        [
            (
                lambda n_states, X, y, random_key: jnp.ones((n_states,)) / n_states,
                does_not_raise(),
            ),
            (
                lambda n_states: jnp.ones((n_states,)) / n_states,
                pytest.raises(ValueError, match="must have the parameters"),
            ),
        ],
    )
    def test_initial_proba_init_custom(self, init_func, expectation):
        with expectation:
            init_funcs = setup_hmm_initialization(initial_proba_init=init_func)
            # check that the correct function is set in init_funcs
            assert init_funcs["initial_proba_init"] == init_func

    @pytest.mark.parametrize(
        "init_str, expectation, method",
        [
            ("sticky", does_not_raise(), sticky_transition_proba_init),
            ("uniform", does_not_raise(), uniform_transition_proba_init),
            ("random", does_not_raise(), random_transition_proba_init),
            ("kmeans", does_not_raise(), kmeans_transition_proba_init),
            (None, does_not_raise(), DEFAULT_INIT_FUNCTIONS["transition_proba_init"]),
            (
                "invalid",
                pytest.raises(ValueError, match="Invalid initialization"),
                None,
            ),
        ],
    )
    def test_transition_proba_init_str(self, init_str, expectation, method):
        with expectation:
            init_funcs = setup_hmm_initialization(transition_proba_init=init_str)
            # check that the correct function is set in init_funcs
            assert init_funcs["transition_proba_init"] == method

    @pytest.mark.parametrize(
        "init_func, expectation",
        [
            (
                lambda n_states, X, y, random_key: jnp.ones((n_states, n_states))
                / n_states,
                does_not_raise(),
            ),
            (
                lambda n_states: jnp.ones((n_states, n_states)) / n_states,
                pytest.raises(ValueError, match="must have the parameters"),
            ),
        ],
    )
    def test_transition_proba_init_custom(self, init_func, expectation):
        with expectation:
            init_funcs = setup_hmm_initialization(transition_proba_init=init_func)
            # check that the correct function is set in init_funcs
            assert init_funcs["transition_proba_init"] == init_func

    @pytest.mark.parametrize(
        "key", ["initial_proba_init_kwargs", "transition_proba_init_kwargs"]
    )
    @pytest.mark.parametrize("kwargs", ["n_states", "X", "y", "random_key"])
    def test_init_kwargs_reserved(self, key, kwargs):
        with pytest.raises(
            ValueError, match=f"Keyword argument '{kwargs}' is reserved"
        ):
            setup_hmm_initialization(**{key: {kwargs: 123}})

    @pytest.fixture
    def custom_init_func(self, init_func, key):
        if init_func:
            return (
                lambda n_states, X, y, random_key, extra_key=2: jnp.ones(
                    shape=((n_states, n_states) if "transition" in key else (n_states,))
                )
                / n_states
            )
        else:
            return None

    @pytest.mark.parametrize(
        "key", ["initial_proba_init_kwargs", "transition_proba_init_kwargs"]
    )
    @pytest.mark.parametrize(
        "init_func, init_kwargs, expectation",
        [
            (0, {"prob_stay": 0.7}, "transition_only"),
            (
                0,
                {"extra_key": 123},
                pytest.raises(ValueError, match="Invalid keyword argument"),
            ),
            (
                1,
                {"extra_key": 123},
                does_not_raise(),
            ),
            (
                1,
                {
                    "extra_key": 123,
                    "another_key": 456,
                },
                pytest.raises(ValueError, match="Invalid keyword argument"),
            ),
        ],
    )
    def test_init_kwargs(
        self, key, init_func, custom_init_func, init_kwargs, expectation
    ):
        if expectation == "transition_only":
            if key == "initial_proba_init_kwargs":
                expectation = pytest.raises(
                    ValueError, match="Invalid keyword argument"
                )
            else:
                expectation = does_not_raise()
        with expectation:
            init_funcs = setup_hmm_initialization(
                **{key[:-7]: custom_init_func, key: init_kwargs}
            )
            # check that the kwargs are set
            assert init_funcs[key] == init_kwargs

    @pytest.mark.parametrize(
        "key1, value1, key2, value2",
        [
            (
                "initial_proba_init",
                "kmeans",
                "initial_proba_init_kwargs",
                {"is_new_session": None},
            ),
            (
                "transition_proba_init",
                "sticky",
                "transition_proba_init_kwargs",
                {"prob_stay": 0.7},
            ),
            (
                "initial_proba_init",
                lambda n_states, X, y, random_key, extra_kwarg=1: jnp.ones((n_states,))
                / n_states,
                "initial_proba_init_kwargs",
                {"extra_kwarg": 2},
            ),
            (
                "transition_proba_init",
                lambda n_states, X, y, random_key, extra_kwarg=1: jnp.ones(
                    (n_states, n_states)
                )
                / n_states,
                "transition_proba_init_kwargs",
                {"extra_kwarg": 2},
            ),
            (
                "transition_proba_init",
                "random",
                "initial_proba_init",
                "random",
            ),
        ],
    )
    def test_update_init_funcs(self, key1, value1, key2, value2):
        first_dict = setup_hmm_initialization(**{key1: value1})
        second_dict = setup_hmm_initialization(**{key2: value2}, init_funcs=first_dict)
        assert first_dict[key1] == second_dict[key1]
        assert first_dict[key2] != second_dict[key2]

    @pytest.mark.parametrize(
        "key, value",
        [
            (
                "initial_proba_init",
                "random",
            ),
            (
                "transition_proba_init",
                "random",
            ),
            (
                "initial_proba_init",
                lambda n_states, X, y, random_key: jnp.ones((n_states,)) / n_states,
            ),
            (
                "transition_proba_init",
                lambda n_states, X, y, random_key: jnp.ones((n_states, n_states))
                / n_states,
            ),
        ],
    )
    def test_reset_init_kwargs(self, key, value):
        first_dict = setup_hmm_initialization(
            **{key: "kmeans", key + "_kwargs": {"is_new_session": None}}
        )
        second_dict = setup_hmm_initialization(**{key: value}, init_funcs=first_dict)
        assert first_dict[key + "_kwargs"] == {"is_new_session": None}
        assert second_dict[key + "_kwargs"] == {}

    def test_default_initialization(self):
        """Test that default initialization functions are set correctly."""
        # I'm putting this at the end to make sure that the default dictionary is not modified by other tests
        init_funcs = setup_hmm_initialization()
        assert init_funcs == DEFAULT_INIT_FUNCTIONS


class TestGenerateHMMInitParams:
    """Test generate_hmm_initial_params function"""

    @pytest.mark.parametrize(
        "init_funcs, expectation",
        [
            ({}, does_not_raise()),
            ({"initial_proba_init": random_initial_proba_init}, does_not_raise()),
            (
                {"invalid_key": None},
                pytest.raises(KeyError, match="Unexpected or unknown keys"),
            ),
            (
                {"initial_prob_init": random_initial_proba_init},
                pytest.raises(KeyError, match="Did you mean"),
            ),
        ],
    )
    def test_init_funcs_keys(self, init_funcs, expectation):
        with expectation:
            generate_hmm_initial_params(
                n_states=3, X=None, y=None, init_funcs=init_funcs
            )

    def test_init_funcs_none_values(self):
        """Test that None values in init_funcs are replaced by defaults."""
        init_funcs = {
            "initial_proba_init": None,
            "initial_proba_init_kwargs": None,
            "transition_proba_init": None,
            "transition_proba_init_kwargs": None,
        }
        result1 = generate_hmm_initial_params(
            n_states=3, X=None, y=None, init_funcs=init_funcs
        )
        result2 = generate_hmm_initial_params(n_states=3, X=None, y=None)
        assert jnp.allclose(jnp.vstack(result1), jnp.vstack(result2))

    @pytest.mark.parametrize("n_states", [1, 2, 3, 5])
    def test_output_shapes_and_types(self, n_states):
        """Test that output shapes and types are correct."""
        initial_prob, transition_prob = generate_hmm_initial_params(
            n_states=n_states, X=None, y=None
        )

        assert initial_prob.shape == (n_states,)
        assert transition_prob.shape == (n_states, n_states)
        assert isinstance(initial_prob, jnp.ndarray)
        assert isinstance(transition_prob, jnp.ndarray)

    @pytest.mark.parametrize(
        "key, value, expectation",
        [
            (
                "initial_proba_init",
                lambda n_states, X, y, random_key: jnp.ones((n_states,)) / n_states,
                does_not_raise(),
            ),
            (
                "transition_proba_init",
                lambda n_states, X, y, random_key: jnp.ones((n_states, n_states))
                / n_states,
                does_not_raise(),
            ),
            (
                "initial_proba_init",
                lambda n_states, X, y, random_key: jnp.ones((n_states - 1,)),
                pytest.raises(ValueError, match="must return an array of shape"),
            ),
            (
                "transition_proba_init",
                lambda n_states, X, y, random_key: jnp.ones(
                    (n_states - 1, n_states - 1)
                ),
                pytest.raises(ValueError, match="must return an array of shape"),
            ),
            (
                "initial_proba_init",
                lambda n_states, X, y, random_key: jnp.ones((n_states,)),
                pytest.raises(ValueError, match="must sum to 1"),
            ),
            (
                "transition_proba_init",
                lambda n_states, X, y, random_key: jnp.ones((n_states, n_states)),
                pytest.raises(ValueError, match="rows that sum to 1"),
            ),
        ],
    )
    def test_validate_custom_func_output(self, key, value, expectation):
        with expectation:
            generate_hmm_initial_params(
                n_states=3,
                X=None,
                y=None,
                init_funcs={key: value, key + "_custom": True},
            )


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
