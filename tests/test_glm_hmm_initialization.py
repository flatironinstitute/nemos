"""Tests for glm_hmm/initialize_parameters.py"""

import itertools
from contextlib import nullcontext as does_not_raise
from unittest.mock import create_autospec

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nemos.glm import GLM
from nemos.glm_hmm.initialize_parameters import (
    DEFAULT_INIT_FUNCTIONS_GLMHMM,
    GLM_INIT_FUCS,
    KMeansInitializerGLM,
    constant_scale_init,
    generate_glm_hmm_initial_params,
    kmeans_glm_params_init,
    kmeans_scale_init,
    random_glm_params_init,
    setup_glm_hmm_initialization,
)
from nemos.hmm.initialize_parameters import (
    sticky_transition_proba_init,
    uniform_initial_proba_init,
)
from nemos.inverse_link_function_utils import resolve_inverse_link_function

# =============================================================================
# Mock infrastructure
# =============================================================================


# GLM-type templates: 5 mandatory params (used for glm_params_init and scale_init)
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


def _glm_template_special(
    n_states, X, y, inverse_link_function, random_key, my_special_param=None
):
    pass


# HMM-type templates: 4 mandatory params (used for initial_proba_init and transition_proba_init)
def _hmm_template_no_extra(n_states, X, y, random_key):
    pass


def _hmm_template_one_extra(n_states, X, y, random_key, param1=None):
    pass


def _hmm_template_two_extra(n_states, X, y, random_key, param1=None, param2=None):
    pass


def _hmm_template_special(n_states, X, y, random_key, my_special_param=None):
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
def kmeans_mock(monkeypatch):
    """Patch GLM.fit to avoid slow JAX optimization. Uses small data so sklearn KMeans is fast."""
    n_states = 3
    n_samples = 30
    n_features = 4

    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_samples, n_features))
    y = np.ones(n_samples)

    def fake_glm_fit(self, X, y, **kwargs):
        n_feat = X.shape[1]
        self.coef_ = jnp.zeros(n_feat)
        self.intercept_ = jnp.array([0.0])

    monkeypatch.setattr(GLM, "fit", fake_glm_fit)
    return n_states, X, y, n_features


class TestKMeansInitializerGLM:
    """Test KMeans-based GLM parameter initialization."""

    def test_glm_params_output_shape(self, kmeans_mock):
        n_states, X, y, n_features = kmeans_mock
        initializer = KMeansInitializerGLM(n_states, X, y, jnp.exp, random_key=0)
        coef, intercept = initializer.glm_params()
        assert coef.shape == (n_features, n_states)
        assert intercept.shape == (n_states,)

    def test_scale_output_shape(self, kmeans_mock):
        """Poisson GLM has fixed scale=1, so scale() returns ones without fitting."""
        n_states, X, y, _ = kmeans_mock
        initializer = KMeansInitializerGLM(n_states, X, y, jnp.exp, random_key=0)
        scale = initializer.scale()
        assert scale.shape == (n_states,)

    def test_shared_initializer(self, kmeans_mock):
        """Providing a pre-built initializer skips creating a new one."""
        n_states, X, y, _ = kmeans_mock
        initializer = KMeansInitializerGLM(n_states, X, y, jnp.exp, random_key=7)
        result = kmeans_glm_params_init(
            n_states, X, y, jnp.exp, initializer=initializer
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
                lambda n_states, X, y, inverse_link_function, random_key: (
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
                lambda n_states, X, y, inverse_link_function, random_key: jnp.ones(
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
        "kwarg_name", ["n_states", "X", "y", "inverse_link_function", "random_key"]
    )
    def test_glm_params_init_kwargs_reserved(self, kwarg_name):
        with pytest.raises(
            ValueError, match=f"Keyword argument '{kwarg_name}' is reserved"
        ):
            setup_glm_hmm_initialization(glm_params_init_kwargs={kwarg_name: 123})

    @pytest.mark.parametrize(
        "kwarg_name", ["n_states", "X", "y", "inverse_link_function", "random_key"]
    )
    def test_scale_init_kwargs_reserved(self, kwarg_name):
        with pytest.raises(
            ValueError, match=f"Keyword argument '{kwarg_name}' is reserved"
        ):
            setup_glm_hmm_initialization(scale_init_kwargs={kwarg_name: 123})

    def test_unknown_init_funcs_key(self):
        with pytest.raises(KeyError, match="Unexpected or unknown keys"):
            setup_glm_hmm_initialization(init_funcs={"totally_invalid_key": None})

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
            init_funcs={k: v for k, v in first.items() if k in GLM_INIT_FUCS},
        )
        assert first[kwargs_key] == kwargs_val
        assert second[kwargs_key] == {}

    def test_default_initialization(self):
        """No-arg call returns DEFAULT_INIT_FUNCTIONS_GLMHMM."""
        init_funcs = setup_glm_hmm_initialization()
        assert init_funcs == DEFAULT_INIT_FUNCTIONS_GLMHMM

    def test_hmm_params_delegated(self):
        """HMM init functions (initial_proba, transition_proba) are set correctly."""
        init_funcs = setup_glm_hmm_initialization(
            initial_proba_init="uniform",
            transition_proba_init="sticky",
        )
        assert init_funcs["initial_proba_init"] == uniform_initial_proba_init
        assert init_funcs["transition_proba_init"] == sticky_transition_proba_init


@pytest.mark.parametrize(
    "func_name",
    ["glm_params_init", "scale_init", "initial_proba_init", "transition_proba_init"],
)
class TestSetupGLMHMMInitializationKwargs:
    """Test kwargs validation in setup_glm_hmm_initialization via mock registries."""

    def test_none_returns_empty_kwargs(self, func_name, recwarn):
        mock_registry = _get_mock_registry("one_extra")
        # inject a mock function then check kwargs validation
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
# Tests for generate_glm_hmm_initial_params
# =============================================================================


class TestGenerateGLMHMMInitialParams:
    """Test generate_glm_hmm_initial_params function."""

    @pytest.mark.parametrize("n_states", [1, 2, 3, 5])
    @pytest.mark.parametrize("n_neurons", [1, 3])
    def test_output_shapes_and_types(self, n_states, n_neurons):
        n_features = 5
        X = jnp.ones((50, n_features))
        y = jnp.ones((50, n_neurons)) if n_neurons > 1 else jnp.ones(50)

        coef, intercept, scale, initial_probs, transition_matrix = (
            generate_glm_hmm_initial_params(n_states, X, y, jnp.exp)
        )

        if n_neurons == 1:
            assert coef.shape == (n_features, n_states)
            assert intercept.shape == (n_states,)
            assert scale.shape == (n_states,)
        else:
            assert coef.shape == (n_features, n_neurons, n_states)
            assert intercept.shape == (n_neurons, n_states)
            assert scale.shape == (n_neurons, n_states)

        assert initial_probs.shape == (n_states,)
        assert transition_matrix.shape == (n_states, n_states)

        for arr in [coef, intercept, scale, initial_probs, transition_matrix]:
            assert isinstance(arr, jnp.ndarray)

    def test_returns_five_elements(self):
        result = generate_glm_hmm_initial_params(
            2, jnp.ones((10, 3)), jnp.ones(10), lambda x: x
        )
        assert isinstance(result, tuple)
        assert len(result) == 5

    @pytest.mark.parametrize(
        "init_funcs, expectation",
        [
            ({}, does_not_raise()),
            ({"glm_params_init": random_glm_params_init}, does_not_raise()),
            (
                {"totally_invalid_key": None},
                pytest.raises(KeyError, match="Unexpected or unknown keys"),
            ),
        ],
    )
    def test_init_funcs_key_validation(self, init_funcs, expectation):
        with expectation:
            generate_glm_hmm_initial_params(
                3,
                jnp.ones((10, 5)),
                jnp.ones(10),
                lambda x: x,
                init_funcs=init_funcs,
            )

    def test_none_init_funcs_uses_defaults(self):
        X = jnp.ones((50, 5))
        y = jnp.full(50, 2.0)
        coef, intercept, scale, initial_probs, transition_matrix = (
            generate_glm_hmm_initial_params(3, X, y, lambda x: x)
        )
        assert jnp.abs(coef).max() < 0.01
        assert jnp.allclose(intercept, 2.0)
        assert jnp.all(scale == 1.0)
        assert jnp.allclose(initial_probs, 1.0 / 3)
        assert jnp.allclose(jnp.diag(transition_matrix), 0.95)

    def test_random_key_affects_output(self):
        """Different random keys produce different GLM coefficient initializations."""
        X = jnp.ones((50, 5))
        y = jnp.ones(50)
        coef1, *_ = generate_glm_hmm_initial_params(3, X, y, lambda x: x, random_key=1)
        coef2, *_ = generate_glm_hmm_initial_params(3, X, y, lambda x: x, random_key=2)
        assert not jnp.allclose(coef1, coef2)

    @pytest.mark.parametrize(
        "custom_func, custom_flag, expectation, X",
        [
            # Valid coef+intercept, plain array X
            (
                lambda n_states, X, y, inverse_link_function, random_key: (
                    jnp.zeros((X.shape[1], n_states)),
                    jnp.zeros(n_states),
                ),
                "glm_params_init_custom",
                does_not_raise(),
                jnp.ones((10, 5)),
            ),
            # Wrong coef shape
            (
                lambda n_states, X, y, inverse_link_function, random_key: (
                    jnp.zeros((X.shape[1] + 1, n_states)),
                    jnp.zeros(n_states),
                ),
                "glm_params_init_custom",
                pytest.raises(ValueError, match="mis-shaped"),
                jnp.ones((10, 5)),
            ),
            # Wrong intercept shape
            (
                lambda n_states, X, y, inverse_link_function, random_key: (
                    jnp.zeros((X.shape[1], n_states)),
                    jnp.zeros(n_states + 1),
                ),
                "glm_params_init_custom",
                pytest.raises(ValueError, match="incorrect shape"),
                jnp.ones((10, 5)),
            ),
            # Wrong coef type
            (
                lambda n_states, X, y, inverse_link_function, random_key: (
                    "not_an_array",
                    jnp.zeros(n_states),
                ),
                "glm_params_init_custom",
                pytest.raises(TypeError, match="did not return a pytree of arrays"),
                jnp.ones((10, 5)),
            ),
            # Wrong intercept type
            (
                lambda n_states, X, y, inverse_link_function, random_key: (
                    jnp.zeros((X.shape[1], n_states)),
                    "not_an_array",
                ),
                "glm_params_init_custom",
                pytest.raises(TypeError, match="did not return an array"),
                jnp.ones((10, 5)),
            ),
            # Valid scale
            (
                lambda n_states, X, y, inverse_link_function, random_key: jnp.ones(
                    n_states
                ),
                "scale_init_custom",
                does_not_raise(),
                jnp.ones((10, 5)),
            ),
            # Wrong scale shape
            (
                lambda n_states, X, y, inverse_link_function, random_key: jnp.ones(
                    n_states + 1
                ),
                "scale_init_custom",
                pytest.raises(ValueError, match="incorrect shape"),
                jnp.ones((10, 5)),
            ),
            # Wrong scale type
            (
                lambda n_states, X, y, inverse_link_function, random_key: "not_an_array",
                "scale_init_custom",
                pytest.raises(TypeError, match="must return an array"),
                jnp.ones((10, 5)),
            ),
            # Valid pytree coef: dict X with matching dict coef and correct leaf shapes
            (
                lambda n_states, X, y, inverse_link_function, random_key: (
                    {k: jnp.zeros((v.shape[1], n_states)) for k, v in X.items()},
                    jnp.zeros(n_states),
                ),
                "glm_params_init_custom",
                does_not_raise(),
                {"feature_a": jnp.ones((10, 3)), "feature_b": jnp.ones((10, 2))},
            ),
            # Wrong pytree structure: dict X but plain-array coef
            (
                lambda n_states, X, y, inverse_link_function, random_key: (
                    jnp.zeros((5, n_states)),
                    jnp.zeros(n_states),
                ),
                "glm_params_init_custom",
                pytest.raises(ValueError, match="tree structure"),
                {"feature_a": jnp.ones((10, 3))},
            ),
        ],
    )
    def test_validate_custom_func_output(
        self, custom_func, custom_flag, expectation, X
    ):
        func_key = custom_flag.replace("_custom", "")
        with expectation:
            generate_glm_hmm_initial_params(
                3,
                X,
                jnp.ones(10),
                lambda x: x,
                init_funcs={func_key: custom_func, custom_flag: True},
            )


# =============================================================================
# Tests for _validate_custom_glm_params_output
# =============================================================================


class TestValidateCustomGLMParamsOutput:
    """Test custom GLM params output validation via generate_glm_hmm_initial_params."""

    @pytest.mark.parametrize("n_neurons", [1, 3])
    def test_valid_output(self, n_neurons):
        n_states, n_features = 3, 5
        X = jnp.ones((10, n_features))
        y = jnp.ones((10, n_neurons)) if n_neurons > 1 else jnp.ones(10)

        if n_neurons == 1:
            coef = jnp.zeros((n_features, n_states))
            intercept = jnp.zeros(n_states)
        else:
            coef = jnp.zeros((n_features, n_neurons, n_states))
            intercept = jnp.zeros((n_neurons, n_states))

        def custom_init(n_states, X, y, inverse_link_function, random_key, **kwargs):
            return coef, intercept

        init_funcs = setup_glm_hmm_initialization(glm_params_init=custom_init)
        generate_glm_hmm_initial_params(n_states, X, y, jnp.exp, init_funcs=init_funcs)

    @pytest.mark.parametrize("n_neurons", [1, 3])
    def test_wrong_coef_shape_raises(self, n_neurons):
        n_states, n_features = 3, 5
        X = jnp.ones((10, n_features))
        y = jnp.ones((10, n_neurons)) if n_neurons > 1 else jnp.ones(10)
        coef = jnp.zeros((n_features + 1, n_states))  # wrong n_features
        intercept = jnp.zeros((n_neurons, n_states) if n_neurons > 1 else (n_states,))

        def custom_init(n_states, X, y, inverse_link_function, random_key, **kwargs):
            return coef, intercept

        init_funcs = setup_glm_hmm_initialization(glm_params_init=custom_init)
        with pytest.raises(ValueError, match="mis-shaped"):
            generate_glm_hmm_initial_params(n_states, X, y, jnp.exp, init_funcs=init_funcs)

    @pytest.mark.parametrize("n_neurons", [1, 3])
    def test_wrong_intercept_shape_raises(self, n_neurons):
        n_states, n_features = 3, 5
        X = jnp.ones((10, n_features))
        y = jnp.ones((10, n_neurons)) if n_neurons > 1 else jnp.ones(10)
        coef = jnp.zeros(
            (n_features, n_neurons, n_states)
            if n_neurons > 1
            else (n_features, n_states)
        )
        intercept = jnp.zeros(n_states + 1)  # wrong shape

        def custom_init(n_states, X, y, inverse_link_function, random_key, **kwargs):
            return coef, intercept

        init_funcs = setup_glm_hmm_initialization(glm_params_init=custom_init)
        with pytest.raises(ValueError, match="incorrect shape"):
            generate_glm_hmm_initial_params(n_states, X, y, jnp.exp, init_funcs=init_funcs)

    def test_wrong_coef_type_raises(self):
        n_states, n_features = 3, 5
        X = jnp.ones((10, n_features))
        y = jnp.ones(10)

        def custom_init(n_states, X, y, inverse_link_function, random_key, **kwargs):
            return "not_an_array", jnp.zeros(n_states)

        init_funcs = setup_glm_hmm_initialization(glm_params_init=custom_init)
        with pytest.raises(TypeError, match="did not return a pytree of arrays"):
            generate_glm_hmm_initial_params(n_states, X, y, jnp.exp, init_funcs=init_funcs)

    def test_wrong_intercept_type_raises(self):
        n_states, n_features = 3, 5
        X = jnp.ones((10, n_features))
        y = jnp.ones(10)

        def custom_init(n_states, X, y, inverse_link_function, random_key, **kwargs):
            return jnp.zeros((n_features, n_states)), "not_an_array"

        init_funcs = setup_glm_hmm_initialization(glm_params_init=custom_init)
        with pytest.raises(TypeError, match="did not return an array"):
            generate_glm_hmm_initial_params(n_states, X, y, jnp.exp, init_funcs=init_funcs)

    def test_error_message_shows_shapes(self):
        """Error message includes both actual and expected shapes."""
        n_states, n_features = 3, 5
        X = jnp.ones((10, n_features))
        y = jnp.ones(10)
        coef = jnp.zeros((n_features + 2, n_states))

        def custom_init(n_states, X, y, inverse_link_function, random_key, **kwargs):
            return coef, jnp.zeros(n_states)

        init_funcs = setup_glm_hmm_initialization(glm_params_init=custom_init)
        with pytest.raises(ValueError, match="Actual shapes"):
            generate_glm_hmm_initial_params(n_states, X, y, jnp.exp, init_funcs=init_funcs)

        with pytest.raises(ValueError, match="Expected shapes"):
            generate_glm_hmm_initial_params(n_states, X, y, jnp.exp, init_funcs=init_funcs)


# =============================================================================
# Tests for _validate_custom_scale_output
# =============================================================================


class TestValidateCustomScaleOutput:
    """Test custom scale output validation via generate_glm_hmm_initial_params."""

    @pytest.mark.parametrize("n_neurons", [1, 3])
    def test_valid_output(self, n_neurons):
        n_states = 3
        X = jnp.ones((10, 5))
        y = jnp.ones((10, n_neurons)) if n_neurons > 1 else jnp.ones(10)
        scale = jnp.ones((n_states, n_neurons) if n_neurons > 1 else (n_states,))

        def custom_scale(n_states, X, y, inverse_link_function, random_key, **kwargs):
            return scale

        init_funcs = setup_glm_hmm_initialization(scale_init=custom_scale)
        generate_glm_hmm_initial_params(n_states, X, y, jnp.exp, init_funcs=init_funcs)

    @pytest.mark.parametrize("n_neurons", [1, 3])
    def test_wrong_shape_raises(self, n_neurons):
        n_states = 3
        X = jnp.ones((10, 5))
        y = jnp.ones((10, n_neurons)) if n_neurons > 1 else jnp.ones(10)
        scale = jnp.ones(n_states + 1)

        def custom_scale(n_states, X, y, inverse_link_function, random_key, **kwargs):
            return scale

        init_funcs = setup_glm_hmm_initialization(scale_init=custom_scale)
        with pytest.raises(ValueError, match="incorrect shape"):
            generate_glm_hmm_initial_params(n_states, X, y, jnp.exp, init_funcs=init_funcs)

    def test_wrong_type_raises(self):
        n_states = 3
        X = jnp.ones((10, 5))
        y = jnp.ones(10)

        def custom_scale(n_states, X, y, inverse_link_function, random_key, **kwargs):
            return "not_an_array"

        init_funcs = setup_glm_hmm_initialization(scale_init=custom_scale)
        with pytest.raises(TypeError, match="must return an array"):
            generate_glm_hmm_initial_params(n_states, X, y, jnp.exp, init_funcs=init_funcs)

    def test_error_message_shows_expected_shape(self):
        n_states = 3
        X = jnp.ones((10, 5))
        y = jnp.ones(10)
        scale = jnp.ones(n_states + 2)

        def custom_scale(n_states, X, y, inverse_link_function, random_key, **kwargs):
            return scale

        init_funcs = setup_glm_hmm_initialization(scale_init=custom_scale)
        with pytest.raises(ValueError, match=rf"\({n_states},\)"):
            generate_glm_hmm_initial_params(n_states, X, y, jnp.exp, init_funcs=init_funcs)
