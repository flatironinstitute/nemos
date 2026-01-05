from contextlib import nullcontext as does_not_raise
from numbers import Number
from typing import Callable
from unittest.mock import patch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pynapple as nap
import pytest
from conftest import instantiate_base_regressor_subclass
from test_base_regressor_subclasses import (
    INSTANTIATE_MODEL_AND_SIMULATE,
    INSTANTIATE_MODEL_ONLY,
)

import nemos as nmo
from nemos._observation_model_builder import (
    instantiate_observation_model,
)
from nemos._regularizer_builder import instantiate_regularizer
from nemos.glm_hmm.expectation_maximization import GLMHMMState, em_glm_hmm, em_step
from nemos.glm_hmm.params import GLMHMMParams, GLMParams
from nemos.pytrees import FeaturePytree
from nemos.utils import _get_name

# ============================================================================
# Tests for GLMHMM.__init__ property setters
# ============================================================================


class TestGLMHMMInit:
    """Tests for GLMHMM class initialization and property setters."""

    # -------------------------------------------------------------------------
    # n_states setter tests
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "n_states, expectation",
        [
            (1, does_not_raise()),
            (2, does_not_raise()),
            (10, does_not_raise()),
            (3.0, does_not_raise()),  # float with no decimals is allowed
            (0, pytest.raises(ValueError, match="must be a positive integer")),
            (-1, pytest.raises(ValueError, match="must be a positive integer")),
            (2.5, pytest.raises(TypeError, match="must be a positive integer")),
            ("3", pytest.raises(TypeError, match="must be a positive integer")),
            (None, pytest.raises(TypeError, match="must be a positive integer")),
            ([3], pytest.raises(TypeError, match="must be a positive integer")),
        ],
    )
    def test_n_states_setter(self, n_states, expectation):
        """Test n_states validation accepts positive integers only."""
        with expectation:
            model = nmo.glm_hmm.GLMHMM(n_states=n_states)
            assert model.n_states == int(n_states)

    def test_n_states_creates_validator(self):
        """Test that setting n_states creates a GLMHMMValidator."""
        model = nmo.glm_hmm.GLMHMM(n_states=3)
        assert hasattr(model, "_validator")
        assert model._validator.n_states == 3

    # -------------------------------------------------------------------------
    # maxiter setter tests
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "maxiter, expectation",
        [
            (1, does_not_raise()),
            (100, does_not_raise()),
            (1000, does_not_raise()),
            (10.0, does_not_raise()),  # float with no decimals is allowed
            (0, pytest.raises(ValueError, match="must be a strictly positive integer")),
            (
                -1,
                pytest.raises(ValueError, match="must be a strictly positive integer"),
            ),
            (
                10.5,
                pytest.raises(ValueError, match="must be a strictly positive integer"),
            ),
            (
                "100",
                pytest.raises(ValueError, match="must be a strictly positive integer"),
            ),
            (
                None,
                pytest.raises(ValueError, match="must be a strictly positive integer"),
            ),
        ],
    )
    def test_maxiter_setter(self, maxiter, expectation):
        """Test maxiter validation accepts positive integers only."""
        with expectation:
            model = nmo.glm_hmm.GLMHMM(n_states=2, maxiter=maxiter)
            assert model.maxiter == int(maxiter)

    # -------------------------------------------------------------------------
    # tol setter tests
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "tol, expectation",
        [
            (1e-8, does_not_raise()),
            (0.001, does_not_raise()),
            (1.0, does_not_raise()),
            (1, does_not_raise()),  # int is allowed (converted to float)
            (0, pytest.raises(ValueError, match="must be a strictly positive float")),
            (
                -1e-8,
                pytest.raises(ValueError, match="must be a strictly positive float"),
            ),
            (
                "0.001",
                pytest.raises(ValueError, match="must be a strictly positive float"),
            ),
            (
                None,
                pytest.raises(ValueError, match="must be a strictly positive float"),
            ),
        ],
    )
    def test_tol_setter(self, tol, expectation):
        """Test tol validation accepts positive numbers only."""
        with expectation:
            model = nmo.glm_hmm.GLMHMM(n_states=2, tol=tol)
            assert model.tol == float(tol)

    # -------------------------------------------------------------------------
    # seed setter tests
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "seed, expectation",
        [
            (jax.random.PRNGKey(0), does_not_raise()),
            (jax.random.PRNGKey(123), does_not_raise()),
            (jax.random.PRNGKey(999999), does_not_raise()),
        ],
    )
    def test_seed_setter_valid(self, seed, expectation):
        """Test seed validation accepts valid JAX PRNG keys."""
        with expectation:
            model = nmo.glm_hmm.GLMHMM(n_states=2, seed=seed)
            assert jnp.array_equal(model.seed, seed)

    @pytest.mark.parametrize(
        "seed",
        [
            123,  # plain int
            np.array([1, 2, 3]),  # wrong shape
            jnp.array([1.0, 2.0]),  # wrong dtype
            "seed",  # string
            None,  # None
        ],
    )
    def test_seed_setter_invalid(self, seed):
        """Test seed validation rejects invalid inputs."""
        with pytest.raises(TypeError, match="seed must be a JAX PRNG key"):
            nmo.glm_hmm.GLMHMM(n_states=2, seed=seed)

    # -------------------------------------------------------------------------
    # observation_model setter tests
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "obs_model",
        ["Poisson", "Bernoulli", "Gamma", "Gaussian", "NegativeBinomial"],
    )
    def test_observation_model_setter_string(self, obs_model):
        """Test observation_model accepts valid string names."""
        model = nmo.glm_hmm.GLMHMM(n_states=2, observation_model=obs_model)
        assert obs_model in model.observation_model.__class__.__name__

    def test_observation_model_setter_instance(self):
        """Test observation_model accepts observation model instances."""
        obs_instance = nmo.observation_models.PoissonObservations()
        model = nmo.glm_hmm.GLMHMM(n_states=2, observation_model=obs_instance)
        assert model.observation_model is obs_instance

    def test_observation_model_setter_invalid_string(self):
        """Test observation_model rejects invalid string names."""
        with pytest.raises(ValueError, match="Unknown observation model"):
            nmo.glm_hmm.GLMHMM(n_states=2, observation_model="InvalidModel")

    # -------------------------------------------------------------------------
    # dirichlet_prior_alphas_init_prob setter tests
    # -------------------------------------------------------------------------
    def test_dirichlet_prior_init_prob_none(self):
        """Test that None is accepted for dirichlet prior."""
        model = nmo.glm_hmm.GLMHMM(n_states=3, dirichlet_prior_alphas_init_prob=None)
        assert model.dirichlet_prior_alphas_init_prob is None

    def test_dirichlet_prior_init_prob_valid(self):
        """Test valid dirichlet prior alphas."""
        alphas = jnp.array([1.0, 2.0, 3.0])
        model = nmo.glm_hmm.GLMHMM(n_states=3, dirichlet_prior_alphas_init_prob=alphas)
        assert jnp.array_equal(model.dirichlet_prior_alphas_init_prob, alphas)

    def test_dirichlet_prior_init_prob_wrong_shape(self):
        """Test that wrong shape raises ValueError."""
        alphas = jnp.array([1.0, 2.0])  # n_states=3 but only 2 elements
        with pytest.raises(ValueError, match="must have shape"):
            nmo.glm_hmm.GLMHMM(n_states=3, dirichlet_prior_alphas_init_prob=alphas)

    def test_dirichlet_prior_init_prob_values_less_than_one(self):
        """Test that alpha values < 1 raise ValueError."""
        alphas = jnp.array([1.0, 0.5, 2.0])
        with pytest.raises(ValueError, match="must be >= 1"):
            nmo.glm_hmm.GLMHMM(n_states=3, dirichlet_prior_alphas_init_prob=alphas)

    # -------------------------------------------------------------------------
    # dirichlet_prior_alphas_transition setter tests
    # -------------------------------------------------------------------------
    def test_dirichlet_prior_transition_none(self):
        """Test that None is accepted for dirichlet prior."""
        model = nmo.glm_hmm.GLMHMM(n_states=3, dirichlet_prior_alphas_transition=None)
        assert model.dirichlet_prior_alphas_transition is None

    def test_dirichlet_prior_transition_valid(self):
        """Test valid dirichlet prior alphas for transitions."""
        alphas = jnp.ones((3, 3))
        model = nmo.glm_hmm.GLMHMM(n_states=3, dirichlet_prior_alphas_transition=alphas)
        assert jnp.array_equal(model.dirichlet_prior_alphas_transition, alphas)

    def test_dirichlet_prior_transition_wrong_shape(self):
        """Test that wrong shape raises ValueError."""
        alphas = jnp.ones((2, 3))  # n_states=3 but wrong shape
        with pytest.raises(ValueError, match="must have shape"):
            nmo.glm_hmm.GLMHMM(n_states=3, dirichlet_prior_alphas_transition=alphas)

    # -------------------------------------------------------------------------
    # initialization_funcs setter tests
    # -------------------------------------------------------------------------
    def test_initialization_funcs_none_uses_defaults(self):
        """Test that None uses default initialization functions."""
        model = nmo.glm_hmm.GLMHMM(n_states=2, initialization_funcs=None)
        assert model.initialization_funcs is not None
        assert "glm_params_init" in model.initialization_funcs

    def test_initialization_funcs_custom(self):
        """Test that custom initialization functions are accepted."""

        def custom_scale(n_states, X, y, random_key):
            return jnp.full(n_states, 2.0)

        model = nmo.glm_hmm.GLMHMM(
            n_states=2, initialization_funcs={"scale_init": custom_scale}
        )
        assert model.initialization_funcs["scale_init"] is custom_scale

    def test_initialization_funcs_invalid_key(self):
        """Test that invalid registry keys raise KeyError."""
        with pytest.raises(KeyError, match="Invalid key"):
            nmo.glm_hmm.GLMHMM(
                n_states=2, initialization_funcs={"invalid_key": lambda: None}
            )

    # -------------------------------------------------------------------------
    # inverse_link_function setter tests
    # -------------------------------------------------------------------------
    def test_inverse_link_function_none_uses_default(self):
        """Test that None uses observation model's default inverse link."""
        model = nmo.glm_hmm.GLMHMM(
            n_states=2, observation_model="Poisson", inverse_link_function=None
        )
        # Poisson default is exp
        assert model.inverse_link_function is not None

    def test_inverse_link_function_custom(self):
        """Test that custom inverse link functions are accepted."""
        custom_link = lambda x: x**2
        model = nmo.glm_hmm.GLMHMM(n_states=2, inverse_link_function=custom_link)
        assert model.inverse_link_function is custom_link

    # -------------------------------------------------------------------------
    # Default values tests
    # -------------------------------------------------------------------------
    def test_default_values(self):
        """Test that default values are set correctly."""
        model = nmo.glm_hmm.GLMHMM(n_states=3)

        assert model.n_states == 3
        assert model.maxiter == 1000
        assert model.tol == 1e-8
        assert "Bernoulli" in model.observation_model.__class__.__name__
        assert model.dirichlet_prior_alphas_init_prob is None
        assert model.dirichlet_prior_alphas_transition is None

    def test_fit_attributes_initialized_to_none(self):
        """Test that fit attributes are initialized to None."""
        model = nmo.glm_hmm.GLMHMM(n_states=3)

        assert model.coef_ is None
        assert model.intercept_ is None
        assert model.scale_ is None
        assert model.initial_prob_ is None
        assert model.transition_prob_ is None
        assert model.solver_state_ is None
        assert model.dof_resid_ is None

    # -------------------------------------------------------------------------
    # repr test
    # -------------------------------------------------------------------------
    def test_repr(self):
        """Test that repr returns a string representation."""
        model = nmo.glm_hmm.GLMHMM(n_states=3, observation_model="Poisson")
        repr_str = repr(model)
        assert "GLMHMM" in repr_str
        assert "n_states" in repr_str


# FILTER FOR GLM HMM
INSTANTIATE_MODEL_ONLY = INSTANTIATE_MODEL_ONLY.copy()
INSTANTIATE_MODEL_ONLY = [v for v in INSTANTIATE_MODEL_ONLY if "GLMHMM" in v["model"]]
INSTANTIATE_MODEL_AND_SIMULATE = INSTANTIATE_MODEL_AND_SIMULATE.copy()
INSTANTIATE_MODEL_AND_SIMULATE = [
    v for v in INSTANTIATE_MODEL_AND_SIMULATE if "GLMHMM" in v["model"]
]

DEFAULT_GLM_COEF_SHAPE = {
    "GLMHMM": (2, 3),
}


@pytest.mark.parametrize(
    "instantiate_base_regressor_subclass",
    INSTANTIATE_MODEL_ONLY,
    indirect=True,
)
def test_get_fit_attrs(instantiate_base_regressor_subclass):
    fixture = instantiate_base_regressor_subclass
    expected_state = {
        "coef_": None,
        "dof_resid_": None,
        "initial_prob_": None,
        "intercept_": None,
        "scale_": None,
        "solver_state_": None,
        "transition_prob_": None,
    }
    assert fixture.model._get_fit_state() == expected_state
    fixture.model.solver_kwargs = {"maxiter": 1}
    fixture.model.fit(fixture.X, fixture.y)
    assert all(val is not None for val in fixture.model._get_fit_state().values())
    assert fixture.model._get_fit_state().keys() == expected_state.keys()


@pytest.mark.parametrize(
    "instantiate_base_regressor_subclass",
    INSTANTIATE_MODEL_AND_SIMULATE,
    indirect=True,
)
class TestGLMHMM:
    """
    Unit tests for the GLMHMM class that do not depend on the observation model.
    i.e. tests that do not call observation model methods, or tests that do not check the output when
    observation model methods are called (e.g. error testing for input validation)
    """

    @pytest.fixture
    def fit_weights_dimensionality_expectation(
        self, instantiate_base_regressor_subclass
    ):
        """
        Fixture to define the expected behavior for test_fit_weights_dimensionality based on the type of GLM class.
        """
        model_cls = instantiate_base_regressor_subclass.model.__class__
        if "Population" in model_cls.__name__:
            # FILL IN WHEN POPULATION CLASS IS DEFINED
            # NOTE THAT THE FIXTURE WILL MAKE TESTS FAIL FOR POPULATION, WHICH IS
            # ENOUGH TO REMIND US TO FILL THIS IN
            return {}
        else:
            return {
                0: pytest.raises(
                    ValueError,
                    match=r"Invalid parameter dimensionality",
                ),
                1: pytest.raises(
                    ValueError,
                    match=r"Invalid parameter dimensionality",
                ),
                2: does_not_raise(),
                3: pytest.raises(
                    ValueError,
                    match=r"Invalid parameter dimensionality",
                ),
            }

    @pytest.mark.parametrize("dim_weights", [0, 1, 2, 3])
    @pytest.mark.requires_x64
    def test_fit_weights_dimensionality(
        self,
        dim_weights,
        instantiate_base_regressor_subclass,
        fit_weights_dimensionality_expectation,
    ):
        """
        Test the `fit` method with weight matrices of different dimensionalities.
        Check for correct dimensionality.
        """
        fixture = instantiate_base_regressor_subclass
        expectation = fit_weights_dimensionality_expectation[dim_weights]
        n_samples, n_features = fixture.X.shape

        if dim_weights == 0:
            init_w = jnp.array([])
        elif dim_weights == 1:
            init_w = jnp.zeros((n_features,))
        elif dim_weights == 2:
            init_w = jnp.zeros(DEFAULT_GLM_COEF_SHAPE[fixture.model.__class__.__name__])
        else:
            init_w = jnp.zeros(
                DEFAULT_GLM_COEF_SHAPE[fixture.model.__class__.__name__]
                + (1,) * (dim_weights - 2)
            )
        with expectation:
            fixture.model.fit(
                fixture.X,
                fixture.y,
                init_params=(
                    init_w,
                    fixture.params.glm_params.intercept,
                    jnp.exp(fixture.params.glm_scale.log_scale),
                    jnp.exp(fixture.params.hmm_params.log_initial_prob),
                    jnp.exp(fixture.params.hmm_params.log_transition_prob),
                ),
            )

    @pytest.mark.parametrize(
        "dim_intercepts, expectation",
        [
            (
                0,
                pytest.raises(ValueError, match=r"Invalid parameter dimensionality"),
            ),
            (1, does_not_raise()),
            (
                2,
                pytest.raises(ValueError, match=r"Invalid parameter dimensionality"),
            ),
            (
                3,
                pytest.raises(ValueError, match=r"Invalid parameter dimensionality"),
            ),
        ],
    )
    @pytest.mark.requires_x64
    def test_fit_intercepts_dimensionality(
        self, dim_intercepts, expectation, instantiate_base_regressor_subclass
    ):
        """
        Test the `fit` method with intercepts of different dimensionalities. Check for correct dimensionality.
        """
        fixture = instantiate_base_regressor_subclass
        if dim_intercepts == 1:
            init_b = jnp.ones(
                DEFAULT_GLM_COEF_SHAPE[fixture.model.__class__.__name__][1]
            )
        else:
            init_b = jnp.ones((1,) * dim_intercepts)
        with expectation:
            fixture.model.fit(
                fixture.X,
                fixture.y,
                init_params=(
                    fixture.params.glm_params.coef,
                    init_b,
                    jnp.exp(fixture.params.glm_scale.log_scale),
                    jnp.exp(fixture.params.hmm_params.log_initial_prob),
                    jnp.exp(fixture.params.hmm_params.log_transition_prob),
                ),
            )

    """
    Parameterization used by test_fit_init_params_type and test_initialize_solver_init_params_type
    Contains the expected behavior and separate initial parameters for regular and population GLMs

    Note: init_params is expected to be a 5-tuple (coef, intercept, scale, initial_prob, transition_prob)
    """
    fit_init_params_type_init_params = (
        "expectation, init_params_glm, init_params_population_glm",
        [
            # Valid case: proper arrays for all 5 params
            (
                does_not_raise(),
                (
                    jnp.zeros((2, 3)),
                    jnp.zeros((3,)),
                    jnp.ones((3,)),  # scale
                    jnp.ones(3) / 3,
                    jnp.ones((3, 3)) / 3,
                ),
                (
                    jnp.zeros((2, 3, 3)),
                    jnp.zeros((3, 3)),
                    jnp.ones((3, 3)),  # scale for population
                    jnp.ones(3) / 3,
                    jnp.ones((3, 3)) / 3,
                ),
            ),
            # Wrong length tuple (not 5 elements)
            (
                pytest.raises(ValueError, match="Params must have length 5"),
                (jnp.zeros((1, 2, 3)), jnp.zeros((3,))),  # Only 2 elements
                (jnp.zeros((1, 2, 3)), jnp.zeros((3, 3))),
            ),
            # Dict instead of tuple for coef (X is array, so dict coef raises error)
            # Different observation models may raise AttributeError or TypeError
            (
                pytest.raises((AttributeError, TypeError)),
                (
                    dict(p1=jnp.zeros((1, 3)), p2=jnp.zeros((1, 3))),
                    jnp.zeros((3,)),
                    jnp.ones((3,)),  # scale
                    jnp.ones(3) / 3,
                    jnp.ones((3, 3)) / 3,
                ),
                (
                    dict(p1=jnp.zeros((2, 3, 3)), p2=jnp.zeros((2, 2, 3))),
                    jnp.zeros((3, 3)),
                    jnp.ones((3, 3)),  # scale
                    jnp.ones(3) / 3,
                    jnp.ones((3, 3)) / 3,
                ),
            ),
            # FeaturePytree for coef (X is array, so FeaturePytree coef raises TypeError)
            (
                pytest.raises(TypeError, match=r"X and coef have mismatched structure"),
                (
                    FeaturePytree(p1=jnp.zeros((1, 3)), p2=jnp.zeros((1, 3))),
                    jnp.zeros((3,)),
                    jnp.ones((3,)),  # scale
                    jnp.ones(3) / 3,
                    jnp.ones((3, 3)) / 3,
                ),
                (
                    FeaturePytree(p1=jnp.zeros((1, 3, 3)), p2=jnp.zeros((1, 2, 3))),
                    jnp.zeros((3, 3)),
                    jnp.ones((3, 3)),  # scale
                    jnp.ones(3) / 3,
                    jnp.ones((3, 3)) / 3,
                ),
            ),
            # Scalar instead of tuple (wrong type)
            (pytest.raises(ValueError, match="Params must have length 5"), 0, 0),
            # Set instead of tuple (wrong type, not subscriptable, raises ValueError about length)
            (
                pytest.raises(ValueError, match="Params must have length 5"),
                {0, 1},
                {0, 1},
            ),
            # String as intercept (wrong type)
            (
                pytest.raises(
                    TypeError, match="Failed to convert parameters to JAX arrays"
                ),
                (
                    jnp.zeros((2, 3)),
                    "",
                    jnp.ones((3,)),
                    jnp.ones(3) / 3,
                    jnp.ones((3, 3)) / 3,
                ),
                (
                    jnp.zeros((2, 3, 3)),
                    "",
                    jnp.ones((3, 3)),
                    jnp.ones(3) / 3,
                    jnp.ones((3, 3)) / 3,
                ),
            ),
            # String as coef (wrong type)
            (
                pytest.raises(
                    TypeError, match="Failed to convert parameters to JAX arrays"
                ),
                (
                    "",
                    jnp.zeros((3,)),
                    jnp.ones((3,)),
                    jnp.ones(3) / 3,
                    jnp.ones((3, 3)) / 3,
                ),
                (
                    "",
                    jnp.zeros((3, 3)),
                    jnp.ones((3, 3)),
                    jnp.ones(3) / 3,
                    jnp.ones((3, 3)) / 3,
                ),
            ),
        ],
    )

    @pytest.mark.parametrize(*fit_init_params_type_init_params)
    @pytest.mark.requires_x64
    def test_fit_init_glm_params_type(
        self,
        instantiate_base_regressor_subclass,
        expectation,
        init_params_glm,
        init_params_population_glm,
    ):
        """
        Test the `fit` method with various types of initial parameters. Ensure that the provided initial parameters
        are array-like.
        """
        fixture = instantiate_base_regressor_subclass
        if "Population" in fixture.model.__class__.__name__:
            init_params = init_params_population_glm
        else:
            init_params = init_params_glm

        with expectation:
            fixture.model.fit(fixture.X, fixture.y, init_params=init_params)

    @pytest.mark.parametrize(
        "delta_n_features, expectation",
        [
            (-1, pytest.raises(ValueError, match="Inconsistent number of features")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="Inconsistent number of features")),
        ],
    )
    @pytest.mark.requires_x64
    def test_fit_n_feature_consistency_weights(
        self,
        delta_n_features,
        instantiate_base_regressor_subclass,
        expectation,
    ):
        """
        Test the `fit` method for inconsistencies between data features and initial weights provided.
        Ensure the number of features align.
        """
        fixture = instantiate_base_regressor_subclass
        if "Population" in fixture.model.__class__.__name__:
            raise RuntimeError("Fill in the test case for population glmhmm")
        else:
            init_w = jnp.zeros((fixture.X.shape[1] + delta_n_features, 3))
            init_b = jnp.ones(
                3,
            )
        with expectation:
            fixture.model.fit(
                fixture.X,
                fixture.y,
                init_params=(
                    init_w,
                    init_b,
                    jnp.exp(fixture.params.glm_scale.log_scale),
                    jnp.exp(fixture.params.hmm_params.log_initial_prob),
                    jnp.exp(fixture.params.hmm_params.log_transition_prob),
                ),
            )

    @pytest.fixture
    def initialize_solver_weights_dimensionality_expectation(
        self, instantiate_base_regressor_subclass
    ):
        name = instantiate_base_regressor_subclass.model.__class__.__name__
        if "Population" in name:
            return {
                0: pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
                1: pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
                2: pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
                3: does_not_raise(),
            }
        else:
            return {
                0: pytest.raises(
                    ValueError,
                    match=r"Inconsistent number of features",
                ),
                1: pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
                2: does_not_raise(),
                3: pytest.raises(
                    ValueError,
                    match=r"params\[0\] must be an array or .* of shape \(n_features",
                ),
            }

    @pytest.mark.parametrize("dim_weights", [0, 1, 2, 3])
    def test_initialize_solver_weights_dimensionality(
        self,
        dim_weights,
        instantiate_base_regressor_subclass,
        fit_weights_dimensionality_expectation,
    ):
        """
        Test the `initialize_solver` method with weight matrices of different dimensionalities.
        Check for correct dimensionality.
        """
        fixture = instantiate_base_regressor_subclass
        expectation = fit_weights_dimensionality_expectation[dim_weights]
        n_samples, n_features = fixture.X.shape

        if dim_weights == 0:
            init_w = jnp.array([])
        elif dim_weights == 1:
            init_w = jnp.zeros((n_features,))
        elif dim_weights == 2:
            init_w = jnp.zeros((n_features, 3))
        elif dim_weights == 3:
            init_w = jnp.zeros(
                (n_features, fixture.y.shape[1] if fixture.y.ndim > 1 else 1, 3)
            )
        else:
            init_w = jnp.zeros((n_features, 3) + (1,) * (dim_weights - 2))
        with expectation:
            params = fixture.model.initialize_params(
                fixture.X,
                fixture.y,
            )
            params = tuple([init_w, *params[1:]])
            # check that params are set
            init_state = fixture.model.initialize_optimization_and_state(
                fixture.X, fixture.y, params
            )

    @pytest.mark.parametrize(
        "dim_intercepts",
        [0, 1, 2, 3],
    )
    def test_initialize_solver_intercepts_dimensionality(
        self,
        dim_intercepts,
        instantiate_base_regressor_subclass,
    ):
        """
        Test the `initialize_solver` method with intercepts of different dimensionalities.

        Check for correct dimensionality.
        """
        fixture = instantiate_base_regressor_subclass
        n_samples, n_features = fixture.X.shape
        is_population = "Population" in fixture.model.__class__.__name__
        if (dim_intercepts == 2 and is_population) or (
            dim_intercepts == 1 and not is_population
        ):
            expectation = does_not_raise()
        elif dim_intercepts == 0:
            expectation = pytest.raises(
                ValueError, match=r"Invalid parameter dimensionality"
            )
        else:

            expectation = pytest.raises(
                ValueError, match=r"Invalid parameter dimensionality"
            )
        if is_population:
            raise RuntimeError("Fill in the test case for population glmhmm")
        else:
            init_b = (
                jnp.zeros((3,) + (1,) * (dim_intercepts - 1))
                if dim_intercepts > 0
                else jnp.array([])
            )
            init_w = jnp.zeros((n_features, 3))
        with expectation:
            params = fixture.model.initialize_params(
                fixture.X,
                fixture.y,
            )
            params = tuple([init_w, init_b, *params[2:]])
            # check that params are set
            init_state = fixture.model.initialize_optimization_and_state(
                fixture.X, fixture.y, params
            )

    @pytest.mark.parametrize(*fit_init_params_type_init_params)
    def test_initialize_solver_init_glm_params_type(
        self,
        instantiate_base_regressor_subclass,
        expectation,
        init_params_glm,
        init_params_population_glm,
    ):
        """
        Test the `initialize_solver` method with various types of initial parameters.

        Ensure that the provided initial parameters are array-like.
        """
        fixture = instantiate_base_regressor_subclass
        if "Population" in fixture.model.__class__.__name__:
            init_params = init_params_population_glm
        else:
            init_params = init_params_glm
        with expectation:
            # check that params are set
            init_state = fixture.model.initialize_optimization_and_state(
                fixture.X, fixture.y, init_params
            )

    @pytest.mark.parametrize(
        "delta_n_features, expectation",
        [
            (-1, pytest.raises(ValueError, match="Inconsistent number of features")),
            (0, does_not_raise()),
            (1, pytest.raises(ValueError, match="Inconsistent number of features")),
        ],
    )
    def test_initialize_solver_n_feature_consistency_glm_coef(
        self,
        delta_n_features,
        expectation,
        instantiate_base_regressor_subclass,
    ):
        """
        Test the `initialize_solver` method for inconsistencies between data features and initial weights provided.
        Ensure the number of features align.
        """
        fixture = instantiate_base_regressor_subclass
        if "Population" in fixture.model.__class__.__name__:
            raise RuntimeError("Fill in the test case for population glmhmm")
        else:
            init_w = jnp.zeros((fixture.X.shape[1] + delta_n_features, 3))
            init_b = jnp.ones(3)
        with expectation:
            params = fixture.model.initialize_params(
                fixture.X,
                fixture.y,
            )
            params = tuple([init_w, init_b, *params[2:]])
            # check that params are set
            init_state = fixture.model.initialize_optimization_and_state(
                fixture.X, fixture.y, params
            )


@pytest.mark.parametrize(
    "X, y, expected_new_session",
    [
        (np.ones((3, 1)), np.ones((3,)), jnp.array([1, 0, 0])),
        (np.ones((3, 1)), np.array([0, np.nan, 0]), jnp.array([1, 0, 1])),
        (
            nap.TsdFrame(
                t=np.arange(3),
                d=np.zeros((3, 3)),
            ),
            nap.Tsd(
                t=np.arange(3),
                d=np.zeros((3,)),
            ),
            jnp.array([1, 0, 0]),
        ),
        (
            nap.TsdFrame(
                t=np.arange(3),
                d=np.zeros((3, 3)),
                time_support=nap.IntervalSet([0, 1.5], [1.0, 2.0]),
            ),
            nap.Tsd(
                t=np.arange(3),
                d=np.zeros((3,)),
                time_support=nap.IntervalSet([0, 1.5], [1.0, 2.0]),
            ),
            jnp.array([1, 0, 1]),
        ),
        (
            nap.TsdFrame(
                t=np.arange(5),
                d=np.zeros((5, 3)),
                time_support=nap.IntervalSet([0, 1.5], [1.0, 5.0]),
            ),
            nap.Tsd(
                t=np.arange(5),
                d=np.array([0, 0, np.nan, np.nan, 0]),
                time_support=nap.IntervalSet([0, 1.5], [1.0, 5.0]),
            ),
            jnp.array([1, 0, 1, 1, 1]),
        ),
        (
            nap.TsdFrame(
                t=np.arange(6),
                d=np.zeros((6, 3)),
                time_support=nap.IntervalSet([0, 1.5], [1.0, 5.0]),
            ),
            nap.Tsd(
                t=np.arange(6),
                d=np.array([0, 0, np.nan, np.nan, 0, 0]),
                time_support=nap.IntervalSet([0, 1.5], [1.0, 5.0]),
            ),
            jnp.array([1, 0, 1, 1, 1, 0]),
        ),
        (
            nap.TsdFrame(
                t=np.arange(6),
                d=np.array([[0], [0], [np.nan], [np.nan], [0], [0]]),
                time_support=nap.IntervalSet([0, 1.5], [1.0, 5.0]),
            ),
            nap.Tsd(
                t=np.arange(6),
                d=np.zeros(6),
                time_support=nap.IntervalSet([0, 1.5], [1.0, 5.0]),
            ),
            jnp.array([1, 0, 1, 1, 1, 0]),
        ),
    ],
)
@pytest.mark.parametrize(
    "instantiate_base_regressor_subclass",
    [{"model": "GLMHMM", "obs_model": "Poisson", "simulate": False}],
    indirect=True,
)
def test_is_new_session(
    X, y, expected_new_session, instantiate_base_regressor_subclass
):
    """Test initialization of new session."""
    fixture = instantiate_base_regressor_subclass
    model = fixture.model
    is_new_session = model._get_is_new_session(X, y)
    assert jnp.all(is_new_session == expected_new_session)

    # -------------------------------------------------------------------------
    # Tests for _initialize_optimization_and_state internal setup
    # -------------------------------------------------------------------------

    def test_initialize_solver_sets_optimization_run(
        self,
        instantiate_base_regressor_subclass,
    ):
        """Test that _optimization_run wraps em_glm_hmm with correct arguments."""
        fixture = instantiate_base_regressor_subclass
        params = fixture.model.initialize_params(fixture.X, fixture.y)
        fixture.model.initialize_optimization_and_state(fixture.X, fixture.y, params)

        assert hasattr(fixture.model, "_optimization_run")
        assert isinstance(fixture.model._optimization_run, eqx.Partial)
        # Check wrapped function
        assert fixture.model._optimization_run.func is em_glm_hmm
        # Check bound keywords
        expected_keys = {
            "inverse_link_function",
            "log_likelihood_func",
            "m_step_fn_glm_params",
            "m_step_fn_glm_scale",
            "maxiter",
            "tol",
        }
        assert set(fixture.model._optimization_run.keywords.keys()) == expected_keys
        # Check maxiter and tol values
        assert (
            fixture.model._optimization_run.keywords["maxiter"] == fixture.model.maxiter
        )
        assert fixture.model._optimization_run.keywords["tol"] == fixture.model.tol

    def test_initialize_solver_sets_optimization_update(
        self,
        instantiate_base_regressor_subclass,
    ):
        """Test that _optimization_update wraps em_step with correct arguments."""
        fixture = instantiate_base_regressor_subclass
        params = fixture.model.initialize_params(fixture.X, fixture.y)
        fixture.model.initialize_optimization_and_state(fixture.X, fixture.y, params)

        assert hasattr(fixture.model, "_optimization_update")
        assert isinstance(fixture.model._optimization_update, eqx.Partial)
        # Check wrapped function
        assert fixture.model._optimization_update.func is em_step
        # Check bound keywords
        expected_keys = {
            "inverse_link_function",
            "log_likelihood_func",
            "m_step_fn_glm_params",
            "m_step_fn_glm_scale",
        }
        assert set(fixture.model._optimization_update.keywords.keys()) == expected_keys

    def test_initialize_solver_sets_optimization_init_state(
        self,
        instantiate_base_regressor_subclass,
    ):
        """Test that _optimization_init_state is set as a callable after initialize_optimization_and_state."""
        fixture = instantiate_base_regressor_subclass
        params = fixture.model.initialize_params(fixture.X, fixture.y)
        fixture.model.initialize_optimization_and_state(fixture.X, fixture.y, params)

        assert hasattr(fixture.model, "_optimization_init_state")
        assert callable(fixture.model._optimization_init_state)

    def test_initialize_solver_returns_glmhmm_state(
        self,
        instantiate_base_regressor_subclass,
    ):
        """Test that initialize_optimization_and_state returns a GLMHMMState."""
        fixture = instantiate_base_regressor_subclass
        params = fixture.model.initialize_params(fixture.X, fixture.y)
        init_state = fixture.model.initialize_optimization_and_state(
            fixture.X, fixture.y, params
        )

        assert isinstance(init_state, GLMHMMState)

    def test_initialize_solver_state_initial_values(
        self,
        instantiate_base_regressor_subclass,
    ):
        """Test that the returned GLMHMMState has correct initial values."""
        fixture = instantiate_base_regressor_subclass
        params = fixture.model.initialize_params(fixture.X, fixture.y)
        init_state = fixture.model.initialize_optimization_and_state(
            fixture.X, fixture.y, params
        )

        # Check initial log-likelihoods are -inf
        assert jnp.isinf(init_state.data_log_likelihood)
        assert init_state.data_log_likelihood < 0
        assert jnp.isinf(init_state.previous_data_log_likelihood)
        assert init_state.previous_data_log_likelihood < 0

        # Check iterations starts at 0
        assert init_state.iterations == 0

        # Check log_likelihood_history has correct shape and is filled with NaN
        assert init_state.log_likelihood_history.shape == (fixture.model.maxiter,)
        assert jnp.all(jnp.isnan(init_state.log_likelihood_history))

    def test_initialize_solver_init_state_fn_returns_same_structure(
        self,
        instantiate_base_regressor_subclass,
    ):
        """Test that _optimization_init_state returns a state with the same structure."""
        fixture = instantiate_base_regressor_subclass
        params = fixture.model.initialize_params(fixture.X, fixture.y)
        init_state = fixture.model.initialize_optimization_and_state(
            fixture.X, fixture.y, params
        )

        # Call _optimization_init_state and verify it returns same structure
        new_state = fixture.model._optimization_init_state()
        assert isinstance(new_state, GLMHMMState)
        assert (
            new_state.log_likelihood_history.shape
            == init_state.log_likelihood_history.shape
        )

    @pytest.mark.parametrize("maxiter", [10, 100, 500])
    def test_initialize_solver_respects_maxiter(
        self,
        maxiter,
        instantiate_base_regressor_subclass,
    ):
        """Test that maxiter is correctly used in state initialization."""
        fixture = instantiate_base_regressor_subclass
        fixture.model.maxiter = maxiter
        params = fixture.model.initialize_params(fixture.X, fixture.y)
        init_state = fixture.model.initialize_optimization_and_state(
            fixture.X, fixture.y, params
        )

        # log_likelihood_history should have shape (maxiter,)
        assert init_state.log_likelihood_history.shape == (maxiter,)


@pytest.mark.parametrize(
    "obs_model, expected_solver_calls, y",
    [
        ("Gaussian", 1, jnp.ones(100)),
        ("Poisson", 2, jnp.ones(100)),
        (
            "Bernoulli",
            2,
            jax.random.choice(
                jax.random.PRNGKey(0), jnp.array([0.0, 1.0]), shape=(100,)
            ),
        ),
        ("Gamma", 2, jnp.ones(100)),
        ("NegativeBinomial", 2, jnp.ones(100)),
    ],
)
def test_initialize_solver_scale_update_method(obs_model, expected_solver_calls, y):
    """Test analytical vs numerical scale update selection.

    Gaussian uses analytical scale update (_instantiate_solver called once).
    All others use numerical scale update (_instantiate_solver called twice).
    """
    model = nmo.glm_hmm.GLMHMM(n_states=3, observation_model=obs_model)
    X = jnp.ones((100, 2))
    params = model.initialize_params(X, y)

    with patch.object(
        model, "_instantiate_solver", wraps=model._instantiate_solver
    ) as mock_solver:
        model.initialize_optimization_and_state(X, y, params)
        assert mock_solver.call_count == expected_solver_calls
