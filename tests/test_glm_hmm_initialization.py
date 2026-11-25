"""Tests for glm_hmm/initialize_parameters.py"""

from contextlib import nullcontext as does_not_raise

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nemos._observation_model_builder import (
    AVAILABLE_OBSERVATION_MODELS,
    instantiate_observation_model,
)
from nemos.glm import GLM
from nemos.glm_hmm.initialize_parameters import (
    KMeansInitializerGLM,
    constant_parameter_init,
    generate_glm_hmm_initial_model_params,
    random_glm_params_init,
)
from nemos.inverse_link_function_utils import resolve_inverse_link_function
from nemos.observation_models import BernoulliObservations, PoissonObservations

# =============================================================================
# Tests for random_glm_params_init
# =============================================================================


class TestRandomGLMParamsInitialization:
    """Test random initialization of GLM parameters for GLM-HMM."""

    @pytest.mark.parametrize("n_states", [1, 2, 3])
    @pytest.mark.parametrize(
        "n_samples, n_features, n_neurons",
        [
            (100, 5, 1),
            (100, 5, 3),
            (50, 10, 1),
        ],
    )
    def test_expected_output_shape(self, n_states, n_samples, n_features, n_neurons):
        X = jnp.ones((n_samples, n_features))
        y = jnp.ones((n_samples, n_neurons)) if n_neurons > 1 else jnp.ones(n_samples)
        inverse_link = lambda x: x

        coef, intercept = random_glm_params_init(
            n_states,
            X,
            y,
            inverse_link,
            PoissonObservations(),
            random_key=jax.random.PRNGKey(123),
        )

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
        coef, intercept = random_glm_params_init(
            2,
            X,
            y,
            lambda x: x,
            PoissonObservations(),
            random_key=jax.random.PRNGKey(123),
        )
        assert isinstance(coef, jnp.ndarray)
        assert isinstance(intercept, jnp.ndarray)

    def test_randomization(self):
        """Different seeds give different coef but identical intercept."""
        X = jnp.ones((100, 5))
        y = jnp.ones(100)
        inverse_link = lambda x: x

        coef1, intercept1 = random_glm_params_init(
            3,
            X,
            y,
            inverse_link,
            PoissonObservations(),
            random_key=jax.random.PRNGKey(41),
        )
        coef2, intercept2 = random_glm_params_init(
            3,
            X,
            y,
            inverse_link,
            PoissonObservations(),
            random_key=jax.random.PRNGKey(42),
        )

        assert not jnp.allclose(coef1, coef2)
        assert jnp.allclose(intercept1, intercept2)

    @pytest.mark.parametrize("std_dev", [0.0, 1.0])
    def test_std_dev_param(self, std_dev):
        X = jnp.ones((10, 2))
        y = jnp.ones((10, 3))
        coef, _ = random_glm_params_init(
            4,
            X,
            y,
            lambda x: x,
            PoissonObservations(),
            random_key=jax.random.PRNGKey(123),
            std_dev=std_dev,
        )
        if std_dev == 0.0:
            assert jnp.all(coef == 0)

    def test_coef_magnitude(self):
        """Default std_dev produces small coefficients."""
        X = jnp.ones((100, 5))
        y = jnp.ones(100)
        coef, _ = random_glm_params_init(
            3,
            X,
            y,
            lambda x: x,
            PoissonObservations(),
            random_key=jax.random.PRNGKey(123),
        )
        assert jnp.abs(coef).max() < 0.01

    def test_intercept_matches_mean_rate(self):
        """Intercept initialized to match mean rate of y (identity link)."""
        y_mean = 2.5
        y = jnp.full(100, y_mean)
        _, intercept = random_glm_params_init(
            3,
            jnp.ones((100, 5)),
            y,
            lambda x: x,
            PoissonObservations(),
            random_key=jax.random.PRNGKey(123),
        )
        assert jnp.allclose(intercept, y_mean)

    def test_intercept_tiled_across_states(self):
        """All states get the same intercept value."""
        _, intercept = random_glm_params_init(
            3,
            jnp.ones((100, 5)),
            jnp.array([1.0, 2.0, 3.0] * 33 + [1.0]),
            lambda x: x,
            PoissonObservations(),
            random_key=jax.random.PRNGKey(123),
        )
        assert jnp.allclose(intercept[0], intercept)

    def test_pytree_x_unwraps_feature_pytree(self):
        """FeaturePytree X is unwrapped to its underlying dict; coef mirrors that structure."""
        from nemos.pytrees import FeaturePytree

        n_samples, n_states = 30, 3
        X = FeaturePytree(a=jnp.ones((n_samples, 2)), b=jnp.ones((n_samples, 4)))
        y = jnp.ones(n_samples)
        coef, intercept = random_glm_params_init(
            n_states,
            X,
            y,
            lambda x: x,
            PoissonObservations(),
            random_key=jax.random.PRNGKey(0),
        )
        # Line 93-94: FeaturePytree -> X.data (dict).
        assert isinstance(coef, dict)
        assert set(coef) == {"a", "b"}
        assert coef["a"].shape == (2, n_states)
        assert coef["b"].shape == (4, n_states)
        assert intercept.shape == (n_states,)

    @pytest.mark.parametrize("n_neurons", [1, 3])
    def test_inverse_link_function_usage(self, n_neurons):
        """Exp inverse link → intercept = log(mean(y))."""
        y = jnp.full((100, n_neurons) if n_neurons > 1 else 100, 10.0)
        _, intercept = random_glm_params_init(
            2,
            jnp.ones((100, 5)),
            y,
            jax.nn.softplus,
            PoissonObservations(),
            random_key=jax.random.PRNGKey(123),
        )
        link_func = resolve_inverse_link_function(jax.nn.softplus, None)
        assert jnp.allclose(intercept, link_func(10.0))

    @pytest.mark.parametrize(
        "observation_model, expectation",
        [
            (PoissonObservations(), does_not_raise()),
            (
                BernoulliObservations(),
                pytest.raises(ValueError, match="Mean activity must lie within"),
            ),
        ],
    )
    def test_rate_range_read_from_observation_model(
        self, observation_model, expectation
    ):
        """A mean of 2.5 is valid for Poisson and out of range for Bernoulli."""
        with expectation:
            random_glm_params_init(
                3,
                jnp.ones((100, 5)),
                jnp.full(100, 2.5),
                lambda x: x,
                observation_model,
                random_key=jax.random.PRNGKey(123),
            )


# =============================================================================
# Tests for constant_scale_init
# =============================================================================


class TestConstantScaleInitialization:
    """Test constant initialization for scale parameters."""

    @pytest.mark.parametrize("n_states", [1, 2, 3, 5])
    @pytest.mark.parametrize("n_samples, n_neurons", [(100, 1), (100, 3), (50, 10)])
    def test_expected_output_shape(self, n_states, n_samples, n_neurons):
        X = jnp.ones((n_samples, 5))
        y = jnp.ones((n_samples, n_neurons)) if n_neurons > 1 else jnp.ones(n_samples)

        scale = constant_parameter_init(
            n_states, X, y, lambda x: x, random_key=jax.random.PRNGKey(124)
        )

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
        scale = constant_parameter_init(
            2, X, y, lambda x: x, random_key=jax.random.PRNGKey(124)
        )
        assert isinstance(scale, jnp.ndarray)

    @pytest.mark.parametrize("n_states", [1, 3, 5])
    @pytest.mark.parametrize("n_neurons", [1, 3])
    def test_default_value_is_one(self, n_states, n_neurons):
        y = jnp.ones((100, n_neurons)) if n_neurons > 1 else jnp.ones(100)
        scale = constant_parameter_init(
            n_states,
            jnp.ones((100, 5)),
            y,
            lambda x: x,
            random_key=jax.random.PRNGKey(124),
        )
        assert jnp.all(scale == 1.0)

    @pytest.mark.parametrize("scale_val", [0.5, 1.0, 2.0, 10.0])
    @pytest.mark.parametrize("n_neurons", [1, 3])
    def test_custom_scale_value(self, scale_val, n_neurons):
        y = jnp.ones((100, n_neurons)) if n_neurons > 1 else jnp.ones(100)
        scale = constant_parameter_init(
            3,
            jnp.ones((100, 5)),
            y,
            lambda x: x,
            random_key=jax.random.PRNGKey(124),
            fill_value=scale_val,
        )
        assert jnp.all(scale == scale_val)

    def test_deterministic(self):
        """Output is identical regardless of random key."""
        X = jnp.ones((100, 5))
        y = jnp.ones(100)
        scale1 = constant_parameter_init(
            3, X, y, lambda x: x, random_key=jax.random.PRNGKey(1)
        )
        scale2 = constant_parameter_init(
            3, X, y, lambda x: x, random_key=jax.random.PRNGKey(999)
        )
        assert jnp.array_equal(scale1, scale2)


# =============================================================================
# Tests for KMeansInitializerGLM
# =============================================================================


@pytest.fixture
def kmeans_mock(monkeypatch, request):
    """Patch GLM.fit to avoid slow JAX optimization. Uses small data so sklearn KMeans is fast."""
    n_states = 3
    n_samples = 30
    n_features = 4
    n_neurons = getattr(request, "param", 1)

    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_samples, n_features))
    if n_neurons == 1:
        y = np.ones(n_samples)

        def fake_glm_fit(self, X, y, **kwargs):
            n_feat = X.shape[1]
            self.coef_ = jnp.zeros(n_feat)
            self.intercept_ = jnp.array([0.0])

        expected_scale_shape = (n_states,)

    else:
        y = np.ones((n_samples, n_neurons))

        def fake_glm_fit(self, X, y, **kwargs):
            n_feat = X.shape[1]
            self.coef_ = jnp.zeros((n_feat, n_neurons))
            self.intercept_ = jnp.array([0.0] * n_neurons)

        expected_scale_shape = (n_neurons, n_states)

    monkeypatch.setattr(GLM, "fit", fake_glm_fit)
    return n_states, X, y, n_features, expected_scale_shape


class TestKMeansInitializerGLM:
    """Test KMeans-based GLM parameter initialization."""

    def test_glm_params_output_shape(self, kmeans_mock):
        n_states, X, y, n_features, _ = kmeans_mock
        initializer = KMeansInitializerGLM(
            n_states, X, y, jnp.exp, "Poisson", random_key=0
        )
        coef, intercept = initializer.glm_params()
        assert coef.shape == (n_features, n_states)
        assert intercept.shape == (n_states,)

    @pytest.mark.parametrize(
        "kmeans_mock",
        [1, 5],
        indirect=True,
        ids=["n_neurons=1", "n_neurons=5"],
    )
    def test_scale_output_shape(self, kmeans_mock):
        """Poisson GLM has fixed scale=1, so scale() returns ones without fitting."""
        n_states, X, y, _, expected_shape = kmeans_mock
        initializer = KMeansInitializerGLM(
            n_states, X, y, jnp.exp, "Poisson", random_key=0
        )
        scale = initializer.scale()
        assert scale.shape == expected_shape

    def test_empty_state_raises(self, kmeans_mock):
        """fit() raises when KMeans leaves a state with zero assigned samples."""
        n_states, X, y, _, _ = kmeans_mock
        initializer = KMeansInitializerGLM(
            n_states, X, y, jnp.exp, "Poisson", random_key=0
        )
        # Force all samples into state 0 so states 1..n_states-1 are empty
        labels = jnp.zeros(X.shape[0], dtype=int)
        initializer.states = jax.nn.one_hot(labels, num_classes=n_states)
        with pytest.raises(ValueError, match="0 samples to state"):
            initializer.fit()

    def test_unwraps_feature_pytree_x(self, kmeans_mock):
        """FeaturePytree X is unwrapped to its underlying dict at __init__ (line 158-159)."""
        from nemos.pytrees import FeaturePytree

        n_states, X_arr, y, _, _ = kmeans_mock
        X = FeaturePytree(a=jnp.asarray(X_arr))
        initializer = KMeansInitializerGLM(
            n_states, X, y, jnp.exp, "Poisson", random_key=0
        )
        assert not isinstance(initializer._X, FeaturePytree)
        assert isinstance(initializer._X, dict)
        assert "a" in initializer._X

    def test_population_branch_uses_population_glm(self, monkeypatch):
        """y.ndim>1 selects PopulationGLM (lines 174-183) and glm_params honours
        the is_population branch (line 218-219)."""
        from nemos.glm import PopulationGLM

        n_states, n_samples, n_features, n_neurons = 3, 30, 4, 2
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n_samples, n_features))
        y = np.ones((n_samples, n_neurons))

        def fake_pop_fit(self, X, y, **kw):
            self.coef_ = jnp.zeros((X.shape[1], y.shape[1]))
            self.intercept_ = jnp.zeros(y.shape[1])

        monkeypatch.setattr(PopulationGLM, "fit", fake_pop_fit)
        initializer = KMeansInitializerGLM(
            n_states, X, y, jnp.exp, "Poisson", random_key=0
        )
        assert isinstance(initializer._glm_models[0], PopulationGLM)
        assert initializer._is_population is True
        coef, intercept = initializer.glm_params()
        assert coef.shape == (n_features, n_neurons, n_states)
        assert intercept.shape == (n_neurons, n_states)

    def test_scale_non_fixed_obs_model_fits_glm(self, monkeypatch):
        """Non-fixed-scale obs model (Gaussian) exercises the ``scale_ is None ->
        self.fit()`` branch and the per-state scale extraction (lines 236-241)."""
        n_states, n_samples, n_features = 3, 30, 4
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n_samples, n_features))
        y = rng.standard_normal(n_samples)
        scale_val = 2.5

        def fake_glm_fit(self, X, y, **kw):
            self.coef_ = jnp.zeros(X.shape[1])
            self.intercept_ = jnp.array([0.0])
            self.scale_ = jnp.array(scale_val)

        monkeypatch.setattr(GLM, "fit", fake_glm_fit)
        initializer = KMeansInitializerGLM(
            n_states, X, y, lambda x: x, "Gaussian", random_key=0
        )
        # Call scale() before glm_params() so the ``scale_ is None`` guard fires.
        scale = initializer.scale()
        assert scale.shape == (n_states,)
        assert jnp.all(scale == scale_val)


# =============================================================================
# Tests for generate_glm_hmm_initial_model_params
# =============================================================================


class TestGenerateGLMHMMInitialParams:
    """Test generate_glm_hmm_initial_model_params function."""

    @pytest.mark.parametrize("n_states", [1, 2, 3, 5])
    @pytest.mark.parametrize("n_neurons", [1, 3])
    @pytest.mark.parametrize("obs_model", AVAILABLE_OBSERVATION_MODELS)
    def test_output_shapes_and_types(self, n_states, n_neurons, obs_model):
        n_features = 5
        X = jnp.ones((50, n_features))
        y = jnp.ones((50, n_neurons)) if n_neurons > 1 else jnp.ones(50)

        obs = instantiate_observation_model(obs_model)
        user_params = generate_glm_hmm_initial_model_params(
            n_states, X, y, jnp.exp, obs
        )
        coef = user_params["coef"]
        intercept = user_params["intercept"]
        extras = {k: user_params[k] for k in obs.extra_params}

        if n_neurons == 1:
            assert coef.shape == (n_features, n_states)
            assert intercept.shape == (n_states,)
            assert all(
                v.shape == (n_states,) for v in jax.tree_util.tree_leaves(extras)
            )
        else:
            assert coef.shape == (n_features, n_neurons, n_states)
            assert intercept.shape == (n_neurons, n_states)
            assert all(
                v.shape == (n_neurons, n_states)
                for v in jax.tree_util.tree_leaves(extras)
            )

        for arr in [coef, intercept]:
            assert isinstance(arr, jnp.ndarray)
        assert all(
            isinstance(arr, jnp.ndarray) for arr in jax.tree_util.tree_leaves(extras)
        )

    @pytest.mark.parametrize("obs_model", AVAILABLE_OBSERVATION_MODELS)
    def test_returns_three_elements(self, obs_model):
        obs = instantiate_observation_model(obs_model)
        result = generate_glm_hmm_initial_model_params(
            2, jnp.ones((10, 3)), jnp.ones(10), lambda x: x, obs
        )
        assert isinstance(result, dict)
        # coef, intercept + extras for the obs model
        assert len(result) == 2 + len([k for k in obs.extra_params])

    @pytest.mark.parametrize("obs_model", AVAILABLE_OBSERVATION_MODELS)
    def test_none_init_funcs_uses_defaults(self, obs_model):
        if obs_model == "Bernoulli":
            pytest.skip(
                "Requires (0,1) rate. fix how it is computed before running this test."
            )
        obs = instantiate_observation_model(obs_model)
        X = jnp.ones((50, 5))
        y = jnp.full(50, 2.0)
        par = generate_glm_hmm_initial_model_params(3, X, y, lambda x: x, obs)
        coef = par.pop("coef")
        intercept = par.pop("intercept")

        assert jnp.abs(coef).max() < 0.01
        assert jnp.allclose(intercept, 2.0)
        assert all(jnp.all(par[k] == v.default) for k, v in obs.extra_params.items())

    def test_random_key_affects_output(self):
        """Different random keys produce different GLM coefficient initializations."""
        X = jnp.ones((50, 5))
        y = jnp.ones(50)
        par1 = generate_glm_hmm_initial_model_params(
            3, X, y, lambda x: x, PoissonObservations(), random_key=1
        )
        par2 = generate_glm_hmm_initial_model_params(
            3, X, y, lambda x: x, PoissonObservations(), random_key=2
        )
        assert not jnp.allclose(par1["coef"], par2["coef"])
