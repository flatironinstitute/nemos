from contextlib import nullcontext as does_not_raise
from copy import deepcopy
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
from nemos import tree_utils
from nemos._observation_model_builder import (
    instantiate_observation_model,
)
from nemos._regularizer_builder import instantiate_regularizer
from nemos.glm_hmm.expectation_maximization import GLMHMMState, em_glm_hmm, em_step
from nemos.glm_hmm.params import GLMHMMParams, GLMParams
from nemos.pytrees import FeaturePytree
from nemos.tree_utils import pytree_map_and_reduce
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

    def test_fit_pynapple_tsd(self, instantiate_base_regressor_subclass):
        """Check that pynapple fit works."""
        fixture = instantiate_base_regressor_subclass
        model_epoch = deepcopy(fixture.model)
        X = fixture.X
        y = fixture.y
        X = nap.TsdFrame(t=np.arange(y.shape[0]), d=X)
        y = nap.Tsd(t=np.arange(y.shape[0]), d=y)
        X = X[:20]
        y = y[:20]
        is_new_session = np.zeros(20)
        is_new_session[0] = 1.0
        # fit via model
        model = fixture.model
        params = model.initialize_params(X, y)
        model.fit(X, y, init_params=params)

        # run em passing the is new session directly
        params_new, _ = model._optimization_run(
            model._validator.to_model_params(params),
            X=X.d,
            y=y.d,
            is_new_session=is_new_session,
        )
        assert not pytree_map_and_reduce(
            jnp.array_equal,
            all,
            params_new,
            model._validator.to_model_params(params),
        )
        assert pytree_map_and_reduce(
            jnp.array_equal,
            all,
            params_new.glm_params,
            GLMParams(model.coef_, model.intercept_),
        )

        # add an epoch and run fits again
        ep = nap.IntervalSet([0, 8], [6, 20])
        X_ep = X.restrict(ep)
        y_ep = y.restrict(ep)
        is_new_session = np.zeros(y_ep.shape[0])
        is_new_session[0] = 1.0
        is_new_session[len(y.restrict(ep[0]))] = 1

        model_epoch.fit(X_ep, y_ep, init_params=params)
        # run em passing the is new session directly
        params_new_epoch, _ = model_epoch._optimization_run(
            model._validator.to_model_params(params),
            X=X_ep.d,
            y=y_ep.d,
            is_new_session=is_new_session,
        )
        assert not pytree_map_and_reduce(
            jnp.array_equal,
            all,
            params_new_epoch,
            params_new,
        )
        assert pytree_map_and_reduce(
            jnp.array_equal,
            all,
            params_new_epoch.glm_params,
            GLMParams(model_epoch.coef_, model_epoch.intercept_),
        )

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
        "X, y, expectation",
        [
            (np.array([[np.nan], [0]]), np.array([0, 1]), does_not_raise()),
            (np.array([[0], [np.nan]]), np.array([0, 1]), does_not_raise()),
            (np.array([[0], [0]]), np.array([np.nan, 1]), does_not_raise()),
            (np.array([[0], [0]]), np.array([0, np.nan]), does_not_raise()),
            (
                np.array([[0], [np.nan], [0]]),
                np.array([0, 1, 2]),
                pytest.raises(
                    ValueError, match="GLM-HMM requires continuous time-series data"
                ),
            ),
            (
                np.array([[0], [0], [0]]),
                np.array([0, np.nan, 2]),
                pytest.raises(
                    ValueError, match="GLM-HMM requires continuous time-series data"
                ),
            ),
            (
                nap.TsdFrame(
                    t=np.arange(5),
                    d=np.array([[0], [np.nan], [0], [0], [0]]),
                    time_support=nap.IntervalSet([0, 3], [2, 5]),
                ),
                np.array([0, 1, 2, 4, 5]),
                pytest.raises(
                    ValueError, match="GLM-HMM requires continuous time-series data"
                ),
            ),
            (
                nap.TsdFrame(
                    t=np.arange(5),
                    d=np.array([[0], [0], [np.nan], [0], [0]]),
                    time_support=nap.IntervalSet([0, 3], [2, 5]),
                ),
                np.array([0, 1, 2, 4, 5]),
                does_not_raise(),
            ),
            (
                np.zeros((5, 1)),
                nap.Tsd(
                    t=np.arange(5),
                    d=np.array([0, np.nan, 2, 4, 5]),
                    time_support=nap.IntervalSet([0, 3], [2, 5]),
                ),
                pytest.raises(
                    ValueError, match="GLM-HMM requires continuous time-series data"
                ),
            ),
            (
                np.zeros((5, 1)),
                nap.Tsd(
                    t=np.arange(5),
                    d=np.array([0, 1, np.nan, 4, 5]),
                    time_support=nap.IntervalSet([0, 3], [2, 5]),
                ),
                does_not_raise(),
            ),
            # Multiple consecutive NaNs in middle - should fail
            (
                np.array([[0], [np.nan], [np.nan], [0]]),
                np.array([0, 1, 2, 3]),
                pytest.raises(
                    ValueError, match="GLM-HMM requires continuous time-series data"
                ),
            ),
            # Multiple consecutive NaNs at start - should pass
            (
                np.array([[np.nan], [np.nan], [0]]),
                np.array([0, 1, 2]),
                does_not_raise(),
            ),
            # Multiple consecutive NaNs at end - should pass
            (
                np.array([[0], [np.nan], [np.nan]]),
                np.array([0, 1, 2]),
                does_not_raise(),
            ),
            # All NaNs - should fail (caught by parent validation)
            (
                np.array([[np.nan], [np.nan]]),
                np.array([np.nan, np.nan]),
                pytest.raises(ValueError),
            ),
            # No NaNs - should pass
            (np.array([[0], [1]]), np.array([0, 1]), does_not_raise()),
            # NaN at start of second epoch - should pass
            (
                nap.TsdFrame(
                    t=np.arange(5),
                    d=np.array([[0], [0], [np.nan], [0], [0]]),
                    time_support=nap.IntervalSet([0, 2], [2, 5]),
                ),
                np.zeros(5),
                does_not_raise(),
            ),
            # Both X and y with NaNs in middle at different positions - should fail
            (
                np.array([[0], [np.nan], [0], [0]]),
                np.array([0, 1, np.nan, 3]),
                pytest.raises(
                    ValueError, match="GLM-HMM requires continuous time-series data"
                ),
            ),
            # Both X and y with NaNs in middle at same position - should fail
            (
                np.array([[0], [np.nan], [0]]),
                np.array([0, np.nan, 2]),
                pytest.raises(
                    ValueError, match="GLM-HMM requires continuous time-series data"
                ),
            ),
        ],
    )
    def test_nan_between_epochs(
        self,
        instantiate_base_regressor_subclass,
        X,
        y,
        expectation,
    ):
        """Test that NaN values are only allowed at epoch boundaries, not in the middle.

        The GLM-HMM forward-backward algorithm requires continuous time-series data within
        each epoch. This validation ensures data quality by rejecting datasets with NaN
        values that would break the algorithm's recurrence relations.

        Test coverage:
        - NaNs at start/end of data → allowed (epoch boundaries)
        - NaNs in middle of epoch → rejected (breaks continuity)
        - Multiple consecutive NaNs at borders → allowed
        - Multiple consecutive NaNs in middle → rejected
        - All NaN data → rejected (caught by parent validation)
        - No NaNs → allowed (trivial case)
        - NaNs at epoch boundaries in multi-epoch pynapple data → allowed
        - NaNs in both X and y → properly combined and validated

        This strict validation prevents runtime errors in the forward-backward algorithm
        and ensures users provide properly formatted time-series data.
        """
        fixture = instantiate_base_regressor_subclass
        model = fixture.model
        with expectation:
            model._validator.validate_inputs(X, y)


@pytest.mark.parametrize(
    "regularizer", ["Ridge", "UnRegularized", "Lasso", "ElasticNet"]
)
@pytest.mark.parametrize(
    "obs_model",
    [
        "PoissonObservations",
        "BernoulliObservations",
        "GammaObservations",
    ],
)
@pytest.mark.parametrize(
    "solver_name",
    [
        "GradientDescent",
        "BFGS",
        "LBFGS",
        "NonlinearCG",
        "ProximalGradient",
        "SVRG",
        "ProxSVRG",
    ],
)
@pytest.mark.parametrize(
    "model_class, fit_state_attrs",
    [
        (
            nmo.glm_hmm.GLMHMM,
            {
                "coef_": jnp.zeros((2, 3)),
                "intercept_": jnp.array([1.0, 1.0, 1.0]),
                "scale_": 2.0 * jnp.ones(3),
                "solver_state_": None,
                "transition_prob_": None,
                "initial_prob_": None,
                "dof_resid_": None,
            },
        ),
    ],
)
def test_save_and_load(
    regularizer,
    obs_model,
    solver_name,
    tmp_path,
    fit_state_attrs,
    model_class,
):
    """
    Test saving and loading a model with various observation models and regularizers.
    Ensure all parameters are preserved.
    """
    if (
        regularizer == "Lasso"
        or regularizer == "GroupLasso"
        or regularizer == "ElasticNet"
        and solver_name not in ["ProximalGradient", "ProxSVRG"]
    ):
        pytest.skip(
            f"Skipping {solver_name} for Lasso type regularizer; not an approximate solver."
        )

    kwargs = dict(
        n_states=3,
        observation_model=obs_model,
        solver_name=solver_name,
        regularizer=regularizer,
        regularizer_strength=2.0,
        solver_kwargs={"tol": 10**-6},
    )

    if regularizer == "UnRegularized":
        kwargs.pop("regularizer_strength")

    model = model_class(**kwargs)

    initial_params = model.get_params()
    # set fit states
    for key, val in fit_state_attrs.items():
        setattr(model, key, val)
        initial_params[key] = val

    # Save
    save_path = tmp_path / "test_model.npz"
    model.save_params(save_path)

    # Load
    loaded_model = nmo.load_model(save_path)
    loaded_params = loaded_model.get_params()
    fit_state = loaded_model._get_fit_state()
    loaded_params.update(fit_state)

    # Assert matching keys and values
    assert set(initial_params.keys()) == set(
        loaded_params.keys()
    ), "Parameter keys mismatch after load."

    for key in initial_params:
        init_val = initial_params[key]
        load_val = loaded_params[key]
        if isinstance(init_val, (int, float, str, type(None))):
            assert init_val == load_val, f"{key} mismatch: {init_val} != {load_val}"
        elif isinstance(init_val, dict):
            assert (
                init_val == load_val
            ), f"{key} dict mismatch: {init_val} != {load_val}"
        elif isinstance(init_val, (np.ndarray, jnp.ndarray)):
            assert np.allclose(
                np.array(init_val), np.array(load_val)
            ), f"{key} array mismatch"
        elif isinstance(init_val, Callable):
            assert _get_name(init_val) == _get_name(
                load_val
            ), f"{key} function mismatch: {_get_name(init_val)} != {_get_name(load_val)}"


@pytest.mark.parametrize("regularizer", ["Ridge"])
@pytest.mark.parametrize(
    "obs_model",
    [
        "PoissonObservations",
    ],
)
@pytest.mark.parametrize(
    "solver_name",
    [
        "ProxSVRG",
    ],
)
@pytest.mark.parametrize(
    "model_class, fit_state_attrs",
    [
        (
            nmo.glm_hmm.GLMHMM,
            {
                "coef_": jnp.zeros((2, 3)),
                "intercept_": jnp.array([1.0, 1.0, 1.0]),
                "scale_": 2.0,
                "dof_resid_": None,
                "solver_state_": None,
                "transition_prob_": None,
                "initial_prob_": None,
            },
        ),
    ],
)
@pytest.mark.parametrize(
    "mapping_dict, expectation",
    [
        ({}, does_not_raise()),
        (
            {
                "observation_model": nmo.observation_models.GammaObservations,
                "regularizer": nmo.regularizer.Lasso,
                "inverse_link_function": lambda x: x**2,
            },
            pytest.warns(UserWarning, match="The following keys have been replaced"),
        ),
        (
            {
                "observation_model": nmo.observation_models.GammaObservations(),  # fails, only class or callable
                "regularizer": nmo.regularizer.Lasso,
                "inverse_link_function": lambda x: x**2,
            },
            pytest.raises(ValueError, match="Invalid map parameter types detected"),
        ),
        (
            {
                "observation_model": "GammaObservations",  # fails, only class or callable
                "regularizer": nmo.regularizer.Lasso,
            },
            pytest.raises(ValueError, match="Invalid map parameter types detected"),
        ),
        (
            {
                "regularizer": nmo.regularizer.Lasso,
                "regularizer_strength": 3.0,  # fails, only class or callable
            },
            pytest.raises(ValueError, match="Invalid map parameter types detected"),
        ),
        (
            {
                "solver_kwargs": {"tol": 10**-1},
            },
            pytest.raises(ValueError, match="Invalid map parameter types detected"),
        ),
        (
            {
                "some__nested__dictionary": {"tol": 10**-1},
            },
            pytest.raises(
                ValueError,
                match="The following keys in your mapping do not match",
            ),
        ),
        # valid mapping dtype, invalid name
        (
            {
                "some__nested__dictionary": nmo.regularizer.Ridge,
            },
            pytest.raises(
                ValueError,
                match="The following keys in your mapping do not match ",
            ),
        ),
    ],
)
def test_save_and_load_with_custom_mapping(
    regularizer,
    obs_model,
    solver_name,
    mapping_dict,
    tmp_path,
    fit_state_attrs,
    model_class,
    expectation,
):
    """
    Test saving and loading a model with various observation models and regularizers.
    Ensure all parameters are preserved.
    """

    if (
        regularizer == "Lasso"
        or regularizer == "GroupLasso"
        and solver_name not in ["ProximalGradient", "SVRG", "ProxSVRG"]
    ):
        pytest.skip(
            f"Skipping {solver_name} for Lasso type regularizer; not an approximate solver."
        )

    model = model_class(
        n_states=3,
        observation_model=obs_model,
        solver_name=solver_name,
        regularizer=regularizer,
        regularizer_strength=2.0,
    )

    initial_params = model.get_params()
    # set fit states
    for key, val in fit_state_attrs.items():
        setattr(model, key, val)
        initial_params[key] = val

    # Save
    save_path = tmp_path / "test_model.npz"
    model.save_params(save_path)

    # Load
    with expectation:
        loaded_model = nmo.load_model(save_path, mapping_dict=mapping_dict)
        loaded_params = loaded_model.get_params()
        fit_state = loaded_model._get_fit_state()
        loaded_params.update(fit_state)

        # Assert matching keys and values
        assert set(initial_params.keys()) == set(
            loaded_params.keys()
        ), "Parameter keys mismatch after load."

        unexpected_keys = set(mapping_dict) - set(initial_params)
        raise_exception = bool(unexpected_keys)
        if raise_exception:
            with pytest.raises(
                ValueError, match="mapping_dict contains unexpected keys"
            ):
                raise ValueError(
                    f"mapping_dict contains unexpected keys: {unexpected_keys}"
                )

        for key in initial_params:
            init_val = initial_params[key]
            load_val = loaded_params[key]

            if key == "observation_model__inverse_link_function":
                if "observation_model" in mapping_dict:
                    continue
            if key in mapping_dict:
                if key == "observation_model":
                    if isinstance(mapping_dict[key], str):
                        mapping_obs = instantiate_observation_model(mapping_dict[key])
                    else:
                        mapping_obs = mapping_dict[key]
                    assert _get_name(mapping_obs) == _get_name(
                        load_val
                    ), f"{key} observation model mismatch: {mapping_dict[key]} != {load_val}"
                elif key == "regularizer":
                    if isinstance(mapping_dict[key], str):
                        mapping_reg = instantiate_regularizer(mapping_dict[key])
                    else:
                        mapping_reg = mapping_dict[key]
                    assert _get_name(mapping_reg) == _get_name(
                        load_val
                    ), f"{key} regularizer mismatch: {mapping_dict[key]} != {load_val}"
                elif key == "solver_name":
                    assert (
                        mapping_dict[key] == load_val
                    ), f"{key} solver name mismatch: {mapping_dict[key]} != {load_val}"
                elif key == "regularizer_strength":
                    assert (
                        mapping_dict[key] == load_val
                    ), f"{key} regularizer strength mismatch: {mapping_dict[key]} != {load_val}"
                continue

        if isinstance(init_val, (int, float, str, type(None))):
            assert init_val == load_val, f"{key} mismatch: {init_val} != {load_val}"

        elif isinstance(init_val, dict):
            assert (
                init_val == load_val
            ), f"{key} dict mismatch: {init_val} != {load_val}"

        elif isinstance(init_val, (np.ndarray, jnp.ndarray)):
            assert np.allclose(
                np.array(init_val), np.array(load_val)
            ), f"{key} array mismatch"

        elif isinstance(init_val, Callable):
            assert _get_name(init_val) == _get_name(
                load_val
            ), f"{key} function mismatch: {_get_name(init_val)} != {_get_name(load_val)}"


def test_save_and_load_nested_class(nested_regularizer, tmp_path):
    """Test that save and load works with nested classes."""
    model = nmo.glm_hmm.GLMHMM(
        n_states=3, regularizer=nested_regularizer, regularizer_strength=1.0
    )
    save_path = tmp_path / "test_model.npz"
    model.save_params(save_path)

    mapping_dict = {
        "regularizer": nested_regularizer.__class__,
        "regularizer__func": jnp.exp,
    }
    with pytest.warns(UserWarning, match="The following keys have been replaced"):
        loaded_model = nmo.load_model(save_path, mapping_dict=mapping_dict)

    assert isinstance(loaded_model.regularizer, nested_regularizer.__class__)
    assert isinstance(
        loaded_model.regularizer.sub_regularizer,
        nested_regularizer.sub_regularizer.__class__,
    )
    assert loaded_model.regularizer.func == mapping_dict["regularizer__func"]

    # change mapping
    mapping_dict = {
        "regularizer": nested_regularizer.__class__,
        "regularizer__sub_regularizer": nmo.regularizer.Ridge,
        "regularizer__func": lambda x: x**2,
    }
    with pytest.warns(UserWarning, match="The following keys have been replaced"):
        loaded_model = nmo.load_model(save_path, mapping_dict=mapping_dict)
    assert isinstance(loaded_model.regularizer, nested_regularizer.__class__)
    assert isinstance(loaded_model.regularizer.sub_regularizer, nmo.regularizer.Ridge)
    assert loaded_model.regularizer.func == mapping_dict["regularizer__func"]


@pytest.mark.parametrize(
    "instantiate_base_regressor_subclass",
    [{"model": "GLMHMM", "obs_model": "Bernoulli", "simulate": True}],
    indirect=True,
)
def test_save_and_load_fitted_model(instantiate_base_regressor_subclass, tmp_path):
    """
    Test saving and loading a fitted model with various observation models and regularizers.
    Ensure all parameters are preserved.
    """
    fixture = instantiate_base_regressor_subclass
    fixture.model.coef_ = fixture.params.glm_params.coef
    fixture.model.intercept_ = fixture.params.glm_params.intercept
    fixture.model.transition_prob_ = jnp.exp(
        fixture.params.hmm_params.log_transition_prob
    )
    fixture.model.initial_prob_ = jnp.exp(fixture.params.hmm_params.log_initial_prob)
    fixture.model.scale_ = jnp.zeros_like(fixture.params.glm_params.intercept) * 11.0
    fixture.model.dof_resid_ = 1.0
    initial_params = fixture.model.get_params()
    fit_state = fixture.model._get_fit_state()
    initial_params.update(fit_state)

    # Save
    save_path = tmp_path / "test_model.npz"
    fixture.model.save_params(save_path)

    # Load
    loaded_model = nmo.load_model(save_path)
    loaded_params = loaded_model.get_params()
    fit_state = loaded_model._get_fit_state()
    fit_state.pop("solver_state_")
    initial_params.pop("solver_state_")
    loaded_params.update(fit_state)

    # Assert states are close
    for k, v in fit_state.items():
        print(f"CHECKING {k}")
        msg = f"{k} mismatch after load."
        if isinstance(v, jnp.ndarray) or isinstance(v, Number):
            assert np.allclose(initial_params[k], v), msg
        else:
            assert all(np.allclose(a, b) for a, b in zip(initial_params[k], v)), msg


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
        # NaN at the very end of data
        (
            np.ones((4, 1)),
            np.array([0, 1, 2, np.nan]),
            jnp.array([1, 0, 0, 0]),
        ),
        # X and y both have NaNs at different positions
        (
            np.array([[0], [np.nan], [2], [3], [np.nan]]),
            np.array([0, 1, np.nan, 3, 4]),
            jnp.array([1, 0, 1, 1, 0]),
        ),
        # X and y both have NaNs at same position
        (
            np.array([[0], [np.nan], [2], [3]]),
            np.array([0, np.nan, 2, 3]),
            jnp.array([1, 0, 1, 0]),
        ),
        # Entire epoch is NaN (with pynapple)
        (
            nap.TsdFrame(
                t=np.arange(6),
                d=np.zeros((6, 1)),
                time_support=nap.IntervalSet([0, 2, 4], [1, 3, 5]),
            ),
            nap.Tsd(
                t=np.arange(6),
                d=np.array([0, 0, np.nan, np.nan, 3, 4]),
                time_support=nap.IntervalSet([0, 2, 4], [1, 3, 5]),
            ),
            jnp.array([1, 0, 1, 1, 1, 0]),
        ),
        # Multiple NaNs at the end
        (
            np.ones((5, 1)),
            np.array([0, 1, 2, np.nan, np.nan]),
            jnp.array([1, 0, 0, 0, 1]),
        ),
    ],
)
@pytest.mark.parametrize(
    "instantiate_base_regressor_subclass",
    [{"model": "GLMHMM", "obs_model": "Poisson", "simulate": False}],
    indirect=True,
)
def test__get_is_new_session(
    X, y, expected_new_session, instantiate_base_regressor_subclass
):
    """Test that session boundaries are correctly identified from epoch starts and NaN positions.

    The GLM-HMM requires proper session segmentation to reset hidden states at discontinuities.
    This test verifies that:
    1. Epoch start times (from pynapple time_support) are marked as new sessions
    2. Positions immediately following NaN values are marked as new sessions
    3. Combined NaNs from both X and y are handled correctly

    This ensures the forward-backward algorithm can properly handle gaps in time-series data
    without incorrectly propagating state information across discontinuities.
    """
    fixture = instantiate_base_regressor_subclass
    model = fixture.model
    is_new_session = model._get_is_new_session(X, y)
    assert jnp.all(is_new_session == expected_new_session)


@pytest.mark.parametrize(
    "X, y",
    [
        (
            nap.TsdFrame(
                t=np.arange(6),
                d=np.zeros((6, 1)),
                time_support=nap.IntervalSet([0, 1.5], [1.0, 5.0]),
            ),
            nap.Tsd(
                t=np.arange(6),
                d=np.arange(6),
                time_support=nap.IntervalSet([0, 1.5], [1.0, 5.0]),
            ),
        ),
        (
            nap.TsdFrame(
                t=np.arange(6),
                d=np.zeros((6, 1)),
                time_support=nap.IntervalSet([0, 1.5], [1.0, 5.0]),
            ),
            nap.Tsd(
                t=np.arange(6),
                d=np.array([0, 1, np.nan, np.nan, 2, 3]),
                time_support=nap.IntervalSet([0, 1.5], [1.0, 5.0]),
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "instantiate_base_regressor_subclass",
    [{"model": "GLMHMM", "obs_model": "Poisson", "simulate": False}],
    indirect=True,
)
def test__get_is_new_session_and_drop_nan(X, y, instantiate_base_regressor_subclass):
    """Test that session markers remain valid after NaN removal.

    Critical integration test verifying that the NaN handling pipeline preserves session
    boundaries correctly. When NaNs are dropped from the data, the new_session indicators
    must still point to the correct first valid sample of each epoch.

    This test ensures:
    1. Number of sessions equals number of epochs after NaN removal
    2. Session markers correctly identify the first valid value of each epoch

    This is essential because the model marks positions *after* NaNs as new sessions
    before dropping them, ensuring session boundaries survive the filtering step.
    """
    fixture = instantiate_base_regressor_subclass
    model = fixture.model
    is_new_session = model._get_is_new_session(X, y)

    _, drop_y, is_new_session = tree_utils.drop_nans(X.d, y.d, is_new_session)
    assert is_new_session.sum() == len(X.time_support)
    first_valid_per_epoch = []
    for ep in y.time_support:
        yep = np.asarray(y.get(ep.start[0], ep.end[0]))
        Xep = np.asarray(X.get(ep.start[0], ep.end[0])).reshape(yep.shape[0], -1)
        is_nan = np.isnan(yep) | np.any(np.isnan(Xep), axis=1)
        first_valid_per_epoch.append(yep[~is_nan][0])
    assert np.all(
        np.array(first_valid_per_epoch) == drop_y[is_new_session.astype(bool)]
    )


# ============================================================================
# Tests for smooth_proba method
# ============================================================================


@pytest.mark.parametrize(
    "instantiate_base_regressor_subclass",
    [
        {"model": "GLMHMM", "obs_model": "Bernoulli", "simulate": True},
        {"model": "GLMHMM", "obs_model": "Poisson", "simulate": True},
        {"model": "GLMHMM", "obs_model": "Gamma", "simulate": True},
        {"model": "GLMHMM", "obs_model": "Gaussian", "simulate": True},
    ],
    indirect=True,
)
@pytest.mark.requires_x64
class TestInferenceMethods:
    """Test suite for inference methods (smooth_proba, filter_proba, decode_state)."""

    @staticmethod
    def _get_expected_shape(method_name, kwargs, n_samples, n_states):
        """Helper to compute expected output shape based on method and kwargs."""
        if method_name in ["smooth_proba", "filter_proba"]:
            return (n_samples, n_states)
        elif method_name == "decode_state":
            if kwargs.get("output_format") == "index":
                return (n_samples,)
            else:  # one-hot (default)
                return (n_samples, n_states)
        else:
            raise ValueError(f"Unknown method: {method_name}")

    @pytest.mark.parametrize(
        "drop_attr",
        ["coef_", "intercept_", "scale_", "initial_prob_", "transition_prob_"],
    )
    @pytest.mark.parametrize(
        "method_config",
        [
            pytest.param(("smooth_proba", {}), id="smooth_proba"),
            pytest.param(("filter_proba", {}), id="filter_proba"),
            pytest.param(("decode_state", {}), id="decode_state-onehot"),
            pytest.param(("decode_state", {"output_format": "index"}), id="decode_state-index"),
        ],
    )
    def test_not_fitted_raises_error(
        self, instantiate_base_regressor_subclass, drop_attr, method_config
    ):
        """Test that inference methods raise an error when model is not fitted."""
        method_name, kwargs = method_config
        fixture = instantiate_base_regressor_subclass
        model = fixture.model
        setattr(model, drop_attr, None)
        with pytest.raises(
            ValueError,
            match=rf"This GLMHMM instance is not fitted yet. .+ \['{drop_attr}'\]",
        ):
            getattr(model, method_name)(fixture.X, fixture.y, **kwargs)

    @pytest.mark.parametrize(
        "method_config",
        [
            pytest.param(("smooth_proba", {}), id="smooth_proba"),
            pytest.param(("filter_proba", {}), id="filter_proba"),
            pytest.param(("decode_state", {}), id="decode_state-onehot"),
            pytest.param(("decode_state", {"output_format": "index"}), id="decode_state-index"),
        ],
    )
    def test_returns_correct_shape(
        self, instantiate_base_regressor_subclass, method_config
    ):
        """Test that inference methods return arrays with correct shapes."""
        method_name, kwargs = method_config
        fixture = instantiate_base_regressor_subclass
        model = fixture.model

        # Get output
        out = getattr(model, method_name)(fixture.X, fixture.y, **kwargs)

        # Check shape
        n_samples = (
            ~np.isnan(np.sum(fixture.y, axis=tuple(range(1, fixture.y.ndim))))
        ).sum()
        n_states = model.n_states
        expected_shape = self._get_expected_shape(method_name, kwargs, n_samples, n_states)
        assert out.shape == expected_shape, f"Expected shape {expected_shape}, got {out.shape}"

    @pytest.mark.parametrize("method_name", ["smooth_proba", "filter_proba"])
    def test_posterior_proba_returns_valid_probabilities(
        self, instantiate_base_regressor_subclass, method_name
    ):
        """Test that smooth_proba returns valid probabilities (between 0 and 1)."""
        fixture = instantiate_base_regressor_subclass
        model = fixture.model

        # Get posteriors
        posteriors = getattr(model, method_name)(fixture.X, fixture.y)

        # Check all values are between 0 and 1
        assert jnp.all(posteriors >= 0), "Some posteriors are negative"
        assert jnp.all(posteriors <= 1), "Some posteriors are greater than 1"

    @pytest.mark.parametrize("method_name", ["smooth_proba", "filter_proba"])
    def test_posterior_proba_probabilities_sum_to_one(
        self, instantiate_base_regressor_subclass, method_name
    ):
        """Test that probabilities sum to 1 across states for each sample."""
        fixture = instantiate_base_regressor_subclass
        model = fixture.model

        # Get posteriors
        posteriors = getattr(model, method_name)(fixture.X, fixture.y)

        # Check sum across states
        row_sums = jnp.sum(posteriors, axis=1)
        assert jnp.allclose(
            row_sums, 1.0, rtol=1e-5
        ), f"Probabilities don't sum to 1. Min: {row_sums.min()}, Max: {row_sums.max()}"

    @pytest.mark.parametrize(
        "method_config",
        [
            pytest.param(("smooth_proba", {}), id="smooth_proba"),
            pytest.param(("filter_proba", {}), id="filter_proba"),
            pytest.param(("decode_state", {}), id="decode_state-onehot"),
            pytest.param(("decode_state", {"output_format": "index"}), id="decode_state-index"),
        ],
    )
    def test_with_arrays(self, instantiate_base_regressor_subclass, method_config):
        """Test inference methods with numpy/jax arrays return jax array."""
        method_name, kwargs = method_config
        fixture = instantiate_base_regressor_subclass
        model = fixture.model

        # Test with numpy array
        out = getattr(model, method_name)(fixture.X, fixture.y, **kwargs)
        assert isinstance(out, jnp.ndarray), f"Expected jnp.ndarray, got {type(out)}"

    @pytest.mark.parametrize("input_type", ["X", "y", "both"])
    @pytest.mark.parametrize(
        "method_config",
        [
            pytest.param(("smooth_proba", {}), id="smooth_proba"),
            pytest.param(("filter_proba", {}), id="filter_proba"),
            pytest.param(("decode_state", {}), id="decode_state-onehot"),
            pytest.param(("decode_state", {"output_format": "index"}), id="decode_state-index"),
        ],
    )
    def test_with_pynapple_returns_tsdframe(
        self, instantiate_base_regressor_subclass, input_type, method_config
    ):
        """Test that inference methods return TsdFrame/Tsd when input is pynapple."""
        method_name, kwargs = method_config
        fixture = instantiate_base_regressor_subclass
        model = fixture.model

        # Convert to pynapple
        n_samples = fixture.X.shape[0]
        time = np.linspace(0, n_samples / 100, n_samples)

        X_input = fixture.X
        y_input = fixture.y

        if input_type in ["X", "both"]:
            X_input = nap.TsdFrame(t=time, d=fixture.X)
        if input_type in ["y", "both"]:
            y_input = nap.Tsd(t=time, d=fixture.y)

        # Get output
        out = getattr(model, method_name)(X_input, y_input, **kwargs)

        # Check return type - decode_state with index format returns Tsd, others return TsdFrame
        if method_name == "decode_state" and kwargs.get("output_format") == "index":
            assert isinstance(out, nap.Tsd), f"Expected nap.Tsd, got {type(out)}"
            assert out.shape == (n_samples,)
        else:
            assert isinstance(out, nap.TsdFrame), f"Expected nap.TsdFrame, got {type(out)}"
            assert out.shape == (n_samples, model.n_states)
        assert jnp.allclose(out.t, time)

    @pytest.mark.parametrize(
        "method_config",
        [
            pytest.param(("smooth_proba", {}), id="smooth_proba"),
            pytest.param(("filter_proba", {}), id="filter_proba"),
            pytest.param(("decode_state", {}), id="decode_state-onehot"),
            pytest.param(("decode_state", {"output_format": "index"}), id="decode_state-index"),
        ],
    )
    def test_with_multiple_sessions(
        self, instantiate_base_regressor_subclass, method_config
    ):
        """Test inference methods with multiple sessions (pynapple epochs)."""
        method_name, kwargs = method_config
        fixture = instantiate_base_regressor_subclass
        model = fixture.model

        # Create multi-session data
        n_samples = fixture.X.shape[0]
        session_1_end = n_samples // 2

        time = np.linspace(0, n_samples / 100, n_samples)
        epochs = nap.IntervalSet(
            start=[time[0], time[session_1_end]],
            end=[time[session_1_end - 1], time[-1]],
        )

        X_tsd = nap.TsdFrame(t=time, d=fixture.X, time_support=epochs)
        y_tsd = nap.Tsd(t=time, d=fixture.y, time_support=epochs)

        # Get output
        out = getattr(model, method_name)(X_tsd, y_tsd, **kwargs)

        # Check shape and type
        if method_name == "decode_state" and kwargs.get("output_format") == "index":
            assert isinstance(out, nap.Tsd)
            assert out.shape == (n_samples,)
        else:
            assert isinstance(out, nap.TsdFrame)
            assert out.shape == (n_samples, model.n_states)

        # Check probabilities are valid for proba methods
        if method_name in ["smooth_proba", "filter_proba"]:
            assert jnp.all(out.values >= 0)
            assert jnp.all(out.values <= 1)
            row_sums = jnp.sum(out.values, axis=1)
            assert jnp.allclose(row_sums, 1.0, rtol=1e-5)

    @pytest.mark.parametrize(
        "method_config",
        [
            pytest.param(("smooth_proba", {}), id="smooth_proba"),
            pytest.param(("filter_proba", {}), id="filter_proba"),
            pytest.param(("decode_state", {}), id="decode_state-onehot"),
            pytest.param(("decode_state", {"output_format": "index"}), id="decode_state-index"),
        ],
    )
    def test_consistency_across_calls(
        self, instantiate_base_regressor_subclass, method_config
    ):
        """Test that inference methods return consistent results across multiple calls."""
        method_name, kwargs = method_config
        fixture = instantiate_base_regressor_subclass
        model = fixture.model

        # Get output twice
        out_1 = getattr(model, method_name)(fixture.X, fixture.y, **kwargs)
        out_2 = getattr(model, method_name)(fixture.X, fixture.y, **kwargs)

        # Check consistency
        assert jnp.allclose(
            out_1, out_2
        ), f"{method_name} returns different results on consecutive calls"

    @pytest.mark.parametrize(
        "method_name", ["smooth_proba", "filter_proba", "decode_state"]
    )
    def test_single_sample(self, instantiate_base_regressor_subclass, method_name):
        """Test smooth_proba with a single sample."""
        fixture = instantiate_base_regressor_subclass
        model = fixture.model

        # Get posteriors for single sample
        X_single = fixture.X[:1]
        y_single = fixture.y[:1]
        out = getattr(model, method_name)(X_single, y_single)

        # Check shape
        assert out.shape == (1, model.n_states)

        if method_name != "decode_state":
            # Check probabilities are valid
            assert jnp.all(out >= 0)
            assert jnp.all(out <= 1)
            assert jnp.allclose(jnp.sum(out), 1.0, rtol=1e-5)

    @pytest.mark.parametrize(
        "method_name", ["smooth_proba", "filter_proba", "decode_state"]
    )
    def test_with_nans_filtered(self, instantiate_base_regressor_subclass, method_name):
        """Test that smooth_proba handles NaNs properly by filtering them."""
        fixture = instantiate_base_regressor_subclass
        model = fixture.model

        # Create data with NaNs
        X_with_nan = fixture.X.copy()
        y_with_nan = fixture.y.copy()

        # Add NaNs at specific indices
        nan_indices = [0, 1, 2]
        X_with_nan[nan_indices] = np.nan

        # This should work - NaNs get filtered internally
        posteriors = getattr(model, method_name)(X_with_nan, y_with_nan)

        # Check that we get valid output (NaN rows filtered)
        assert posteriors.shape[1] == model.n_states
        # After filtering NaNs, shape[0] should be reduced
        assert posteriors.shape[0] == fixture.X.shape[0]

    @pytest.mark.parametrize(
        "method_name", ["smooth_proba", "filter_proba", "decode_state"]
    )
    def test_different_observation_models(
        self, instantiate_base_regressor_subclass, method_name
    ):
        """Test smooth_proba works with different observation models."""
        fixture = instantiate_base_regressor_subclass
        model = fixture.model
        obs_model_name = model.observation_model.__class__.__name__

        # Get posteriors
        out = getattr(model, method_name)(fixture.X, fixture.y)

        # Basic checks
        assert out.shape == (
            fixture.X.shape[0],
            model.n_states,
        ), f"Shape check failed for {obs_model_name}"

        if method_name != "decode_state":
            assert jnp.all(out >= 0), f"Negative probabilities for {obs_model_name}"
            assert jnp.all(out <= 1), f"Probabilities > 1 for {obs_model_name}"
            row_sums = jnp.sum(out, axis=1)
            assert jnp.allclose(
                row_sums, 1.0, rtol=1e-5
            ), f"Probabilities don't sum to 1 for {obs_model_name}"

    @pytest.mark.parametrize("method_name", ["smooth_proba", "filter_proba"])
    @pytest.mark.convergence
    def test_posterior_proba_maxiter_effect(
        self, instantiate_base_regressor_subclass, method_name
    ):
        """Test that smooth_proba results depend on model fit quality (maxiter)."""
        fixture = instantiate_base_regressor_subclass

        # Fit with very few iterations
        model_poor = nmo.glm_hmm.GLMHMM(
            n_states=fixture.model.n_states,
            observation_model=fixture.model.observation_model,
            solver_kwargs={"maxiter": 1},
        )
        model_poor.fit(fixture.X, fixture.y)
        posteriors_poor = model_poor.smooth_proba(fixture.X, fixture.y)

        # Fit with more iterations
        model_better = nmo.glm_hmm.GLMHMM(
            n_states=fixture.model.n_states,
            observation_model=fixture.model.observation_model,
            solver_kwargs={"maxiter": 10},
        )
        model_better.fit(fixture.X, fixture.y)
        posteriors_better = model_better.smooth_proba(fixture.X, fixture.y)

        # Both should be valid probabilities
        for posteriors in [posteriors_poor, posteriors_better]:
            assert jnp.all(posteriors >= 0)
            assert jnp.all(posteriors <= 1)
            row_sums = jnp.sum(posteriors, axis=1)
            assert jnp.allclose(row_sums, 1.0, rtol=1e-5)

    @pytest.mark.parametrize(
        "method_name", ["smooth_proba", "filter_proba", "decode_state"]
    )
    @pytest.mark.parametrize("nan_location", [[], [0, 1, 10, 11, 12]])
    def test_pynapple_in_pynapple_out_X(
        self, instantiate_base_regressor_subclass, method_name, nan_location
    ):
        fixture = instantiate_base_regressor_subclass
        X = fixture.X.copy()
        X[nan_location] = np.nan
        ep = nap.IntervalSet([0, 10], [9, 500])
        X = nap.TsdFrame(t=np.arange(X.shape[0]), d=X, time_support=ep)
        y = fixture.y
        model = fixture.model
        out = getattr(model, method_name)(X, y)
        assert isinstance(out, nap.TsdFrame), f"Did not return pynapple!"
        assert np.all(
            np.isnan(out[nan_location])
        ), f"Not returning NaNs in the expected location!"

    @pytest.mark.parametrize(
        "method_name", ["smooth_proba", "filter_proba", "decode_state"]
    )
    @pytest.mark.parametrize("nan_location", [[], [0, 1, 10, 11, 12]])
    def test_pynapple_in_pynapple_out_y(
        self, instantiate_base_regressor_subclass, method_name, nan_location
    ):
        fixture = instantiate_base_regressor_subclass
        X = fixture.X
        y = np.array(fixture.y.copy(), dtype=float)
        y[nan_location] = np.nan
        ep = nap.IntervalSet([0, 10], [9, 500])
        y = nap.Tsd(t=np.arange(y.shape[0]), d=y, time_support=ep)
        model = fixture.model
        posteriors = getattr(model, method_name)(X, y)
        assert isinstance(posteriors, nap.TsdFrame), f"Did not return pynapple!"
        assert np.all(
            np.isnan(posteriors[nan_location])
        ), f"Not returning NaNs in the expected location!"

    @pytest.mark.parametrize(
        "method_name", ["smooth_proba", "filter_proba", "decode_state"]
    )
    def test_int_vs_float_y(self, instantiate_base_regressor_subclass, method_name):
        """Test that integer and float y with same values give same posteriors.

        This is a regression test for a bug where y.dtype was used to cast params
        before preprocessing, causing integer y to round float params to integers.
        """
        fixture = instantiate_base_regressor_subclass
        X = fixture.X
        y = round(fixture.y.copy())
        y_float = y.astype(float)
        y_int = y.astype(int)
        model = fixture.model

        # Get posteriors with float y
        out_float = getattr(model, method_name)(X, y_float)

        # Get posteriors with int y (same values)
        out_int = getattr(model, method_name)(X, y_int)

        # Posteriors should be identical regardless of y dtype
        np.testing.assert_allclose(
            out_float,
            out_int,
            rtol=1e-10,
            err_msg=f"{method_name} gives different results for int vs float y with same values",
        )

    def test_onehot_vs_index_decode(self, instantiate_base_regressor_subclass):
        fixture = instantiate_base_regressor_subclass
        X = fixture.X
        y = fixture.y
        model = fixture.model
        out_onehot = model.decode_state(X, y, output_format="one-hot")
        out_index = model.decode_state(X, y, output_format="index")
        assert jnp.all(jnp.where(out_onehot == 1)[1] == out_index), "index and one-hot do not match!"
        assert jnp.all(out_onehot.sum(axis=1) == 1), "more than one hot value in one-hot array!"

@pytest.mark.parametrize("n_states", [2, 3, 5])
@pytest.mark.parametrize(
    "method_name", ["smooth_proba", "filter_proba", "decode_state"]
)
def test_different_n_states(n_states, method_name):
    """Test smooth_proba with different numbers of states."""
    np.random.seed(123)
    n_samples, n_features = 100, 2
    X = np.random.randn(n_samples, n_features)
    y = np.random.poisson(2, size=n_samples)

    model = nmo.glm_hmm.GLMHMM(
        n_states=n_states,
        observation_model="Poisson",
        solver_kwargs={"maxiter": 2},
    )
    model.fit(X, y)

    out = getattr(model, method_name)(X, y)

    # Check shape
    assert out.shape == (
        n_samples,
        n_states,
    ), f"Expected shape ({n_samples}, {n_states}), got {out.shape}"

    # Check probabilities are valid
    assert jnp.all(out >= 0)
    assert jnp.all(out <= 1)
    row_sums = jnp.sum(out, axis=1)
    assert jnp.allclose(row_sums, 1.0)

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
    @pytest.mark.convergence
    def test_initialize_solver_respects_maxiter(
        self,
        maxiter,
        instantiate_base_regressor_subclass,
    ):
        """Test that maxiter is correctly used in state initialization."""
        fixture = instantiate_base_regressor_subclass
        fixture.model.maxiter = maxiter
        fixture.model.solver_kwargs.update({"maxiter": 500})
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
