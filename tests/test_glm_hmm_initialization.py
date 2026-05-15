"""Tests for glm_hmm/initialize_parameters.py"""

from contextlib import nullcontext as does_not_raise
from unittest.mock import create_autospec

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nemos.glm import GLM
from nemos.glm_hmm.initialize_parameters import (
    DEFAULT_INIT_FUNCTIONS_GLMHMM,
    KMeansInitializerGLM,
    constant_scale_init,
    generate_glm_hmm_initial_model_params,
    kmeans_glm_params_init,
    kmeans_scale_init,
    random_glm_params_init,
    setup_glm_hmm_initialization,
)
from nemos.glm_hmm.validation import GLMHMMValidator
from nemos.hmm.hmm import BaseHMM
from nemos.inverse_link_function_utils import resolve_inverse_link_function

# =============================================================================
# Minimal GLMHMM mock for key-validation tests (exercises initialization_funcs setter)
# =============================================================================


class MockGLMHMM(BaseHMM):
    _validator_class = GLMHMMValidator
    _model_default_init_dict = DEFAULT_INIT_FUNCTIONS_GLMHMM

    def __init__(
        self, n_states, hmm_initialization_funcs=None, model_initialization_funcs=None
    ):
        BaseHMM.__init__(
            self, n_states=n_states, hmm_initialization_funcs=hmm_initialization_funcs
        )
        self.model_initialization_funcs = model_initialization_funcs
        self.coef_ = self.intercept_ = self.scale_ = None

    def _model_setup(self, **kwargs):
        self._model_initialization_funcs = setup_glm_hmm_initialization(
            init_funcs=self._model_initialization_funcs,
        )

    def _check_model_is_fit(self):
        pass

    def _get_model_params(self):
        pass

    def _set_model_params(self, params):
        pass

    def _log_likelihood(self, params, X, y):
        pass

    def _model_params_initialization(self, X, y, is_new_session, random_key=None):
        pass

    def fit(self, *a, **kw):
        pass

    def _initialize_optimizer_and_state(self, *a, **kw):
        pass

    def _compute_loss(self, *a, **kw):
        pass

    def _get_optimal_solver_params_config(self, *a, **kw):
        pass

    def predict(self, *a, **kw):
        pass

    def simulate(self, *a, **kw):
        pass

    def save_params(self, *a, **kw):
        pass

    def update(self, *a, **kw):
        pass

    def score(self, *a, **kw):
        pass


# =============================================================================
# Mock infrastructure
# =============================================================================


# GLM-type templates: 6 mandatory params
# (n_states, X, y, inverse_link_function, is_new_session, random_key)
def _glm_template_no_extra(
    n_states, X, y, inverse_link_function, is_new_session, random_key
):
    pass


def _glm_template_one_extra(
    n_states, X, y, inverse_link_function, is_new_session, random_key, param1=None
):
    pass


def _glm_template_two_extra(
    n_states,
    X,
    y,
    inverse_link_function,
    is_new_session,
    random_key,
    param1=None,
    param2=None,
):
    pass


def _glm_template_special(
    n_states,
    X,
    y,
    inverse_link_function,
    is_new_session,
    random_key,
    my_special_param=None,
):
    pass


# HMM-type templates: 5 mandatory params
# (n_states, X, y, is_new_session, random_key)
def _hmm_template_no_extra(n_states, X, y, is_new_session, random_key):
    pass


def _hmm_template_one_extra(n_states, X, y, is_new_session, random_key, param1=None):
    pass


def _hmm_template_two_extra(
    n_states, X, y, is_new_session, random_key, param1=None, param2=None
):
    pass


def _hmm_template_special(
    n_states, X, y, is_new_session, random_key, my_special_param=None
):
    pass


_GLM_TEMPLATES = {
    "no_extra": _glm_template_no_extra,
    "one_extra": _glm_template_one_extra,
    "two_extra": _glm_template_two_extra,
    "special": _glm_template_special,
}
_HMM_TEMPLATES = {
    "no_extra": _hmm_template_no_extra,
    "one_extra": _hmm_template_one_extra,
    "two_extra": _hmm_template_two_extra,
    "special": _hmm_template_special,
}

_GLM_FUNC_NAMES = {"glm_params_init", "scale_init"}
_ALL_FUNC_NAMES = [
    "glm_params_init",
    "scale_init",
    "initial_proba_init",
    "transition_proba_init",
]

MOCK_VALID_KWARGS = {
    "one_extra": {"param1": 0.5},
    "two_extra": {"param1": 0.5, "param2": 0.5},
}


def _get_mock(func_name, template_type):
    """Get a mock with the correct signature for the given function name."""
    templates = _GLM_TEMPLATES if func_name in _GLM_FUNC_NAMES else _HMM_TEMPLATES
    return create_autospec(templates[template_type], return_value=None)


def _get_mock_registry(template_type="one_extra"):
    """Get a mock registry where all functions share the same template type."""
    return {fn: _get_mock(fn, template_type) for fn in _ALL_FUNC_NAMES}


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
            n_states, X, y, inverse_link, random_key=jax.random.PRNGKey(123)
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
            2, X, y, lambda x: x, random_key=jax.random.PRNGKey(123)
        )
        assert isinstance(coef, jnp.ndarray)
        assert isinstance(intercept, jnp.ndarray)

    def test_randomization(self):
        """Different seeds give different coef but identical intercept."""
        X = jnp.ones((100, 5))
        y = jnp.ones(100)
        inverse_link = lambda x: x

        coef1, intercept1 = random_glm_params_init(
            3, X, y, inverse_link, random_key=jax.random.PRNGKey(41)
        )
        coef2, intercept2 = random_glm_params_init(
            3, X, y, inverse_link, random_key=jax.random.PRNGKey(42)
        )

        assert not jnp.allclose(coef1, coef2)
        assert jnp.allclose(intercept1, intercept2)

    @pytest.mark.parametrize("std_dev", [0.0, 1.0])
    def test_std_dev_param(self, std_dev):
        X = jnp.ones((10, 2))
        y = jnp.ones((10, 3))
        coef, _ = random_glm_params_init(
            4, X, y, lambda x: x, random_key=jax.random.PRNGKey(123), std_dev=std_dev
        )
        if std_dev == 0.0:
            assert jnp.all(coef == 0)

    def test_coef_magnitude(self):
        """Default std_dev produces small coefficients."""
        X = jnp.ones((100, 5))
        y = jnp.ones(100)
        coef, _ = random_glm_params_init(
            3, X, y, lambda x: x, random_key=jax.random.PRNGKey(123)
        )
        assert jnp.abs(coef).max() < 0.01

    def test_intercept_matches_mean_rate(self):
        """Intercept initialized to match mean rate of y (identity link)."""
        y_mean = 2.5
        y = jnp.full(100, y_mean)
        _, intercept = random_glm_params_init(
            3, jnp.ones((100, 5)), y, lambda x: x, random_key=jax.random.PRNGKey(123)
        )
        assert jnp.allclose(intercept, y_mean)

    def test_intercept_tiled_across_states(self):
        """All states get the same intercept value."""
        _, intercept = random_glm_params_init(
            3,
            jnp.ones((100, 5)),
            jnp.array([1.0, 2.0, 3.0] * 33 + [1.0]),
            lambda x: x,
            random_key=jax.random.PRNGKey(123),
        )
        assert jnp.allclose(intercept[0], intercept)

    @pytest.mark.parametrize("n_neurons", [1, 3])
    def test_inverse_link_function_usage(self, n_neurons):
        """Exp inverse link → intercept = log(mean(y))."""
        y = jnp.full((100, n_neurons) if n_neurons > 1 else 100, 10.0)
        _, intercept = random_glm_params_init(
            2,
            jnp.ones((100, 5)),
            y,
            jax.nn.softplus,
            random_key=jax.random.PRNGKey(123),
        )
        link_func = resolve_inverse_link_function(jax.nn.softplus, None)
        assert jnp.allclose(intercept, link_func(10.0))


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

        scale = constant_scale_init(
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
        scale = constant_scale_init(
            2, X, y, lambda x: x, random_key=jax.random.PRNGKey(124)
        )
        assert isinstance(scale, jnp.ndarray)

    @pytest.mark.parametrize("n_states", [1, 3, 5])
    @pytest.mark.parametrize("n_neurons", [1, 3])
    def test_default_value_is_one(self, n_states, n_neurons):
        y = jnp.ones((100, n_neurons)) if n_neurons > 1 else jnp.ones(100)
        scale = constant_scale_init(
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
        scale = constant_scale_init(
            3,
            jnp.ones((100, 5)),
            y,
            lambda x: x,
            random_key=jax.random.PRNGKey(124),
            scale_val=scale_val,
        )
        assert jnp.all(scale == scale_val)

    def test_deterministic(self):
        """Output is identical regardless of random key."""
        X = jnp.ones((100, 5))
        y = jnp.ones(100)
        scale1 = constant_scale_init(
            3, X, y, lambda x: x, random_key=jax.random.PRNGKey(1)
        )
        scale2 = constant_scale_init(
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

    def test_shared_initializer(self, kmeans_mock):
        """Providing a pre-built initializer skips creating a new one."""
        n_states, X, y, _, _ = kmeans_mock
        initializer = KMeansInitializerGLM(
            n_states, X, y, jnp.exp, "Poisson", random_key=7
        )
        result = kmeans_glm_params_init(
            n_states, X, y, jnp.exp, "Poisson", initializer=initializer
        )
        expected = initializer.glm_params()
        coef_r, int_r = result
        coef_e, int_e = expected
        assert jnp.allclose(coef_r, coef_e)
        assert jnp.allclose(int_r, int_e)


# =============================================================================
# Tests for setup_glm_hmm_initialization
# =============================================================================


class TestSetupGLMHMMInitialization:
    """Test setup_glm_hmm_initialization for GLM-specific initialization functions."""

    @pytest.mark.parametrize(
        "init_str, expectation, method",
        [
            ("random", does_not_raise(), random_glm_params_init),
            ("kmeans", does_not_raise(), kmeans_glm_params_init),
            (None, does_not_raise(), DEFAULT_INIT_FUNCTIONS_GLMHMM["glm_params_init"]),
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
    def test_glm_params_init_str(self, init_str, expectation, method):
        with expectation:
            init_funcs = setup_glm_hmm_initialization(glm_params_init=init_str)
            assert init_funcs["glm_params_init"] == method

    @pytest.mark.parametrize(
        "init_str, expectation, method",
        [
            ("constant", does_not_raise(), constant_scale_init),
            ("kmeans", does_not_raise(), kmeans_scale_init),
            (None, does_not_raise(), DEFAULT_INIT_FUNCTIONS_GLMHMM["scale_init"]),
            (
                "invalid",
                pytest.raises(ValueError, match="Invalid initialization"),
                None,
            ),
        ],
    )
    def test_scale_init_str(self, init_str, expectation, method):
        with expectation:
            init_funcs = setup_glm_hmm_initialization(scale_init=init_str)
            assert init_funcs["scale_init"] == method

    @pytest.mark.parametrize(
        "init_func, expectation",
        [
            (
                lambda n_states, X, y, inverse_link_function, is_new_session, random_key: (
                    jnp.zeros((1, n_states)),
                    jnp.zeros(n_states),
                ),
                does_not_raise(),
            ),
            (
                lambda n_states: (jnp.zeros((1, n_states)), jnp.zeros(n_states)),
                pytest.raises(ValueError, match="must have the parameters"),
            ),
        ],
    )
    def test_glm_params_init_custom(self, init_func, expectation):
        with expectation:
            init_funcs = setup_glm_hmm_initialization(glm_params_init=init_func)
            assert init_funcs["glm_params_init"] == init_func

    @pytest.mark.parametrize(
        "init_func, expectation",
        [
            (
                lambda n_states, X, y, inverse_link_function, is_new_session, random_key: jnp.ones(
                    n_states
                ),
                does_not_raise(),
            ),
            (
                lambda n_states: jnp.ones(n_states),
                pytest.raises(ValueError, match="must have the parameters"),
            ),
        ],
    )
    def test_scale_init_custom(self, init_func, expectation):
        with expectation:
            init_funcs = setup_glm_hmm_initialization(scale_init=init_func)
            assert init_funcs["scale_init"] == init_func

    @pytest.mark.parametrize(
        "kwarg_name",
        ["n_states", "X", "y", "inverse_link_function", "is_new_session", "random_key"],
    )
    def test_glm_params_init_kwargs_reserved(self, kwarg_name):
        with pytest.raises(
            ValueError, match=f"Keyword argument '{kwarg_name}' is reserved"
        ):
            setup_glm_hmm_initialization(glm_params_init_kwargs={kwarg_name: 123})

    @pytest.mark.parametrize(
        "kwarg_name",
        ["n_states", "X", "y", "inverse_link_function", "is_new_session", "random_key"],
    )
    def test_scale_init_kwargs_reserved(self, kwarg_name):
        with pytest.raises(
            ValueError, match=f"Keyword argument '{kwarg_name}' is reserved"
        ):
            setup_glm_hmm_initialization(scale_init_kwargs={kwarg_name: 123})

    def test_unknown_init_funcs_key(self):
        with pytest.raises(KeyError, match="Unexpected or unknown keys"):
            MockGLMHMM(
                n_states=2, model_initialization_funcs={"totally_invalid_key": None}
            )

    @pytest.mark.parametrize(
        "key, init_str, kwargs_key, kwargs_val",
        [
            (
                "glm_params_init",
                "random",
                "glm_params_init_kwargs",
                {"std_dev": 0.01},
            ),
            (
                "scale_init",
                "constant",
                "scale_init_kwargs",
                {"scale_val": 2.0},
            ),
        ],
    )
    def test_reset_kwargs_when_func_changes(
        self, key, init_str, kwargs_key, kwargs_val
    ):
        """Providing a new init function resets its kwargs to {}."""
        first = setup_glm_hmm_initialization(**{key: init_str, kwargs_key: kwargs_val})
        second = setup_glm_hmm_initialization(
            **{key: init_str},
            init_funcs={
                k: v for k, v in first.items() if k in DEFAULT_INIT_FUNCTIONS_GLMHMM
            },
        )
        assert first[kwargs_key] == kwargs_val
        assert second[kwargs_key] == {}

    def test_default_initialization(self):
        """No-arg call returns DEFAULT_INIT_FUNCTIONS_GLMHMM."""
        init_funcs = setup_glm_hmm_initialization()
        assert init_funcs == DEFAULT_INIT_FUNCTIONS_GLMHMM

    # def test_hmm_params_delegated(self):
    #     """HMM init functions (initial_proba, transition_proba) are set correctly."""
    #     init_funcs = setup_glm_hmm_initialization(
    #         initial_proba_init="uniform",
    #         transition_proba_init="sticky",
    #     )
    #     assert init_funcs["initial_proba_init"] == uniform_initial_proba_init
    #     assert init_funcs["transition_proba_init"] == sticky_transition_proba_init


@pytest.mark.parametrize(
    "func_name",
    ["glm_params_init", "scale_init"],
)
class TestSetupGLMHMMInitializationKwargs:
    """Test kwargs validation in setup_glm_hmm_initialization via mock registries."""

    def test_none_returns_empty_kwargs(self, func_name, recwarn):
        mock_registry = _get_mock_registry("one_extra")
        init_funcs = setup_glm_hmm_initialization(
            **{func_name: mock_registry[func_name]}
        )
        assert init_funcs[func_name + "_kwargs"] == {}

    @pytest.mark.parametrize(
        "template_type, extra_kwargs",
        [
            ("one_extra", {"param1": 0.5}),
            ("two_extra", {"param1": 0.5, "param2": 0.5}),
        ],
    )
    def test_valid_kwargs_accepted(self, func_name, template_type, extra_kwargs):
        mock_func = _get_mock(func_name, template_type)
        init_funcs = setup_glm_hmm_initialization(
            **{func_name: mock_func, func_name + "_kwargs": extra_kwargs}
        )
        assert init_funcs[func_name + "_kwargs"] == extra_kwargs

    def test_invalid_kwarg_raises(self, func_name):
        mock_func = _get_mock(func_name, "one_extra")
        with pytest.raises(ValueError, match="Invalid keyword argument"):
            setup_glm_hmm_initialization(
                **{func_name: mock_func, func_name + "_kwargs": {"invalid_param": 0.5}}
            )


# =============================================================================
# Tests for generate_glm_hmm_initial_model_params
# =============================================================================


class TestGenerateGLMHMMInitialParams:
    """Test generate_glm_hmm_initial_model_params function."""

    @pytest.mark.parametrize("n_states", [1, 2, 3, 5])
    @pytest.mark.parametrize("n_neurons", [1, 3])
    def test_output_shapes_and_types(self, n_states, n_neurons):
        n_features = 5
        X = jnp.ones((50, n_features))
        y = jnp.ones((50, n_neurons)) if n_neurons > 1 else jnp.ones(50)

        coef, intercept, scale = generate_glm_hmm_initial_model_params(
            n_states, X, y, jnp.exp
        )

        if n_neurons == 1:
            assert coef.shape == (n_features, n_states)
            assert intercept.shape == (n_states,)
            assert scale.shape == (n_states,)
        else:
            assert coef.shape == (n_features, n_neurons, n_states)
            assert intercept.shape == (n_neurons, n_states)
            assert scale.shape == (n_neurons, n_states)

        for arr in [coef, intercept, scale]:
            assert isinstance(arr, jnp.ndarray)

    def test_returns_three_elements(self):
        result = generate_glm_hmm_initial_model_params(
            2, jnp.ones((10, 3)), jnp.ones(10), lambda x: x
        )
        assert isinstance(result, tuple)
        assert len(result) == 3

    @pytest.mark.parametrize(
        "init_funcs, expectation",
        [
            ({}, does_not_raise()),
            ({"glm_params_init": random_glm_params_init}, does_not_raise()),
        ],
    )
    def test_init_funcs_key_validation(self, init_funcs, expectation):
        with expectation:
            generate_glm_hmm_initial_model_params(
                3,
                jnp.ones((10, 5)),
                jnp.ones(10),
                lambda x: x,
                init_funcs=init_funcs,
            )

    def test_init_funcs_unknown_key_raises(self):
        with pytest.raises(KeyError, match="Unexpected or unknown keys"):
            MockGLMHMM(
                n_states=2, model_initialization_funcs={"totally_invalid_key": None}
            )

    def test_none_init_funcs_uses_defaults(self):
        X = jnp.ones((50, 5))
        y = jnp.full(50, 2.0)
        coef, intercept, scale = generate_glm_hmm_initial_model_params(
            3, X, y, lambda x: x
        )
        assert jnp.abs(coef).max() < 0.01
        assert jnp.allclose(intercept, 2.0)
        assert jnp.all(scale == 1.0)

    def test_random_key_affects_output(self):
        """Different random keys produce different GLM coefficient initializations."""
        X = jnp.ones((50, 5))
        y = jnp.ones(50)
        coef1, *_ = generate_glm_hmm_initial_model_params(
            3, X, y, lambda x: x, random_key=1
        )
        coef2, *_ = generate_glm_hmm_initial_model_params(
            3, X, y, lambda x: x, random_key=2
        )
        assert not jnp.allclose(coef1, coef2)
