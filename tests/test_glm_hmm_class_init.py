"""Tests for GLMHMM class __init__ and property setters."""

from contextlib import nullcontext as does_not_raise
from unittest.mock import create_autospec

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nemos as nmo

# =============================================================================
# Mock infrastructure for testing init kwargs (shared with test_glm_hmm_initialization.py)
# =============================================================================


def _glm_mock_template(n_states, X, y, inverse_link_function, random_key, param1=None):
    """Template for glm_params_init mock with extra kwarg."""
    pass


def _other_mock_template(n_states, X, y, random_key, param1=None):
    """Template for other init mocks with extra kwarg."""
    pass


FUNC_NAMES = [
    "glm_params_init",
    "scale_init",
    "initial_proba_init",
    "transition_proba_init",
]
MOCK_VALID_KWARGS = {"param1": 0.5}


def _get_mock_func(func_name):
    """Get a mock function with the appropriate signature and extra kwargs."""
    template = (
        _glm_mock_template if func_name == "glm_params_init" else _other_mock_template
    )
    return create_autospec(template, return_value=None)


def _get_mock_registry():
    """Get a mock init function registry where all funcs have param1 kwarg."""
    return {fn: _get_mock_func(fn) for fn in FUNC_NAMES}


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

    # -------------------------------------------------------------------------
    # initialization_kwargs setter tests
    # -------------------------------------------------------------------------
    def test_initialization_kwargs_none_uses_defaults(self):
        """Test that None uses empty kwargs for all functions."""
        model = nmo.glm_hmm.GLMHMM(n_states=2, initialization_kwargs=None)
        assert model.initialization_kwargs is not None
        assert all(v == {} for v in model.initialization_kwargs.values())

    def test_initialization_kwargs_empty_dict_uses_defaults(self):
        """Test that empty dict uses empty kwargs for all functions."""
        model = nmo.glm_hmm.GLMHMM(n_states=2, initialization_kwargs={})
        assert model.initialization_kwargs is not None
        assert all(v == {} for v in model.initialization_kwargs.values())


@pytest.mark.parametrize("func_name", FUNC_NAMES)
class TestInitializationKwargsWithMocks:
    """Test initialization_kwargs using mock functions so all funcs have kwargs."""

    def test_valid_kwargs_accepted(self, func_name):
        """Test that valid kwargs are accepted for each func."""
        mock_registry = _get_mock_registry()
        model = nmo.glm_hmm.GLMHMM(
            n_states=2,
            initialization_funcs=mock_registry,
            initialization_kwargs={func_name: MOCK_VALID_KWARGS},
        )
        assert model.initialization_kwargs[func_name] == MOCK_VALID_KWARGS

    def test_invalid_kwargs_raises(self, func_name):
        """Test that invalid kwargs raise ValueError for each func."""
        mock_registry = _get_mock_registry()
        with pytest.raises(ValueError, match="Invalid keyword argument"):
            nmo.glm_hmm.GLMHMM(
                n_states=2,
                initialization_funcs=mock_registry,
                initialization_kwargs={func_name: {"invalid_param": 0.8}},
            )

    def test_kwargs_setter_valid(self, func_name):
        """Test setting kwargs via setter for each func."""
        mock_registry = _get_mock_registry()
        model = nmo.glm_hmm.GLMHMM(n_states=2, initialization_funcs=mock_registry)

        model.initialization_kwargs = {func_name: MOCK_VALID_KWARGS}
        assert model.initialization_kwargs[func_name] == MOCK_VALID_KWARGS

    def test_kwargs_setter_invalid_raises(self, func_name):
        """Test that setting invalid kwargs via setter raises for each func."""
        mock_registry = _get_mock_registry()
        model = nmo.glm_hmm.GLMHMM(n_states=2, initialization_funcs=mock_registry)

        with pytest.raises(ValueError, match="Invalid"):
            model.initialization_kwargs = {func_name: {"invalid": 1}}

    def test_single_func_kwargs_others_empty(self, func_name):
        """Test that setting kwargs for one func leaves others empty."""
        mock_registry = _get_mock_registry()
        model = nmo.glm_hmm.GLMHMM(
            n_states=2,
            initialization_funcs=mock_registry,
            initialization_kwargs={func_name: MOCK_VALID_KWARGS},
        )

        assert model.initialization_kwargs[func_name] == MOCK_VALID_KWARGS
        for other_fn in FUNC_NAMES:
            if other_fn != func_name:
                assert model.initialization_kwargs[other_fn] == {}


class TestInitializationKwargsMultiple:
    """Test initialization_kwargs with multiple functions."""

    def test_all_funcs_kwargs_at_once(self):
        """Test setting kwargs for all functions at once."""
        mock_registry = _get_mock_registry()
        all_kwargs = {fn: MOCK_VALID_KWARGS for fn in FUNC_NAMES}
        model = nmo.glm_hmm.GLMHMM(
            n_states=2,
            initialization_funcs=mock_registry,
            initialization_kwargs=all_kwargs,
        )

        for fn in FUNC_NAMES:
            assert model.initialization_kwargs[fn] == MOCK_VALID_KWARGS

    def test_all_pairs_of_funcs_kwargs(self):
        """Test setting kwargs for all pairs of functions."""
        import itertools

        mock_registry = _get_mock_registry()

        for func1, func2 in itertools.combinations(FUNC_NAMES, 2):
            pair_kwargs = {func1: MOCK_VALID_KWARGS, func2: MOCK_VALID_KWARGS}
            model = nmo.glm_hmm.GLMHMM(
                n_states=2,
                initialization_funcs=mock_registry,
                initialization_kwargs=pair_kwargs,
            )

            assert model.initialization_kwargs[func1] == MOCK_VALID_KWARGS
            assert model.initialization_kwargs[func2] == MOCK_VALID_KWARGS
            for other_fn in FUNC_NAMES:
                if other_fn not in (func1, func2):
                    assert model.initialization_kwargs[other_fn] == {}


@pytest.mark.parametrize("func_name", FUNC_NAMES)
class TestInitializationKwargsReset:
    """Test that kwargs are reset when initialization functions change.

    Uses mock functions so all func_names can be tested uniformly.
    """

    def test_warning_when_func_changes_with_kwargs(self, func_name):
        """Test that warning is raised when function changes and kwargs exist."""
        mock_registry = _get_mock_registry()
        model = nmo.glm_hmm.GLMHMM(
            n_states=2,
            initialization_funcs=mock_registry,
            initialization_kwargs={func_name: MOCK_VALID_KWARGS},
        )

        # Create a NEW mock function (different object) to trigger the change
        new_mock = _get_mock_func(func_name)

        with pytest.warns(UserWarning, match="changed"):
            model.initialization_funcs = {func_name: new_mock}

        # kwargs should be reset
        assert model.initialization_kwargs[func_name] == {}

    def test_no_warning_when_func_changes_without_kwargs(self, func_name, recwarn):
        """Test no warning when function changes but no kwargs were set."""
        mock_registry = _get_mock_registry()
        model = nmo.glm_hmm.GLMHMM(n_states=2, initialization_funcs=mock_registry)

        recwarn.clear()

        # Create a NEW mock function to change to
        new_mock = _get_mock_func(func_name)
        model.initialization_funcs = {func_name: new_mock}

        # Check no UserWarning was raised
        user_warnings = [w for w in recwarn if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0

    def test_no_warning_when_same_func_reassigned(self, func_name, recwarn):
        """Test no warning when same function is reassigned."""
        mock_registry = _get_mock_registry()
        original_func = mock_registry[func_name]
        model = nmo.glm_hmm.GLMHMM(
            n_states=2,
            initialization_funcs=mock_registry,
            initialization_kwargs={func_name: MOCK_VALID_KWARGS},
        )

        recwarn.clear()

        # Reassign the SAME function object
        model.initialization_funcs = {func_name: original_func}

        # Check no UserWarning was raised
        user_warnings = [w for w in recwarn if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0

        # kwargs should be preserved
        assert model.initialization_kwargs[func_name] == MOCK_VALID_KWARGS

    def test_kwargs_validated_against_current_funcs(self, func_name):
        """Test that setting new initialization_kwargs validates against current funcs."""
        mock_registry = _get_mock_registry()
        model = nmo.glm_hmm.GLMHMM(n_states=2, initialization_funcs=mock_registry)

        # Set valid kwargs
        model.initialization_kwargs = {func_name: MOCK_VALID_KWARGS}
        assert model.initialization_kwargs[func_name] == MOCK_VALID_KWARGS

        # Invalid kwargs should raise
        with pytest.raises(ValueError, match="Invalid"):
            model.initialization_kwargs = {func_name: {"invalid": 1}}

    def test_warning_message_explains_reason(self, func_name):
        """Test that warning message explains why kwargs are being reset."""
        mock_registry = _get_mock_registry()
        model = nmo.glm_hmm.GLMHMM(
            n_states=2,
            initialization_funcs=mock_registry,
            initialization_kwargs={func_name: MOCK_VALID_KWARGS},
        )

        new_mock = _get_mock_func(func_name)
        with pytest.warns(UserWarning, match="may not be compatible"):
            model.initialization_funcs = {func_name: new_mock}

    def test_partial_funcs_update_resets_others_to_defaults(self, func_name):
        """Test that partial initialization_funcs update resets other funcs to defaults.

        When setting initialization_funcs with a partial dict, the missing funcs
        are filled from DEFAULT_INIT_FUNCTION, not preserved from current. This
        means their kwargs are also validated against the new (default) funcs.
        """
        mock_registry = _get_mock_registry()
        # Set kwargs for all functions using mock registry
        all_kwargs = {fn: MOCK_VALID_KWARGS for fn in FUNC_NAMES}
        model = nmo.glm_hmm.GLMHMM(
            n_states=2,
            initialization_funcs=mock_registry,
            initialization_kwargs=all_kwargs,
        )

        # Change only this func_name - others will reset to defaults
        new_mock = _get_mock_func(func_name)

        # This will warn for the changed func AND potentially reset others
        # whose kwargs may not be valid for the default funcs
        with pytest.warns(UserWarning):
            model.initialization_funcs = {func_name: new_mock}

        # The changed func's kwargs should be reset
        assert model.initialization_kwargs[func_name] == {}


class TestInitializeParamsWithKwargs:
    """Integration tests verifying kwargs are passed to initialize_params."""

    def test_transition_proba_kwargs_applied(self):
        """Test that transition_proba_init kwargs are applied in initialize_params."""
        X = np.random.randn(10, 2)
        y = np.random.choice([0, 1], size=10)

        # Create model with custom prob_stay
        custom_prob_stay = 0.999
        model = nmo.glm_hmm.GLMHMM(
            n_states=3,
            initialization_kwargs={
                "transition_proba_init": {"prob_stay": custom_prob_stay}
            },
        )

        params = model.initialize_params(X, y)
        transition_proba = params[
            4
        ]  # (coef, intercept, scale, initial_prob, transition_prob)

        # Check diagonal is prob_stay
        assert jnp.allclose(jnp.diag(transition_proba), custom_prob_stay)

    def test_glm_params_kwargs_applied(self):
        """Test that glm_params_init kwargs are applied in initialize_params."""
        X = np.random.randn(10, 2)
        y = np.random.choice([0, 1], size=10)

        # Create model with zero std_dev (should produce zero coefficients)
        model = nmo.glm_hmm.GLMHMM(
            n_states=3,
            initialization_kwargs={"glm_params_init": {"std_dev": 0.0}},
        )

        params = model.initialize_params(X, y)
        coef = params[0]

        # With std_dev=0, coefficients should be exactly zero
        assert jnp.allclose(coef, 0.0)

    def test_kwargs_setter_then_initialize_params(self):
        """Test setting kwargs via setter then calling initialize_params."""
        X = np.random.randn(10, 2)
        y = np.random.choice([0, 1], size=10)

        model = nmo.glm_hmm.GLMHMM(n_states=3)

        # Set kwargs after init
        model.initialization_kwargs = {"transition_proba_init": {"prob_stay": 0.999}}

        params = model.initialize_params(X, y)
        transition_proba = params[4]

        assert jnp.allclose(jnp.diag(transition_proba), 0.999)

    def test_multiple_kwargs_applied(self):
        """Test multiple kwargs are all applied in initialize_params."""
        X = np.random.randn(10, 2)
        y = np.random.choice([0, 1], size=10)

        model = nmo.glm_hmm.GLMHMM(
            n_states=3,
            initialization_kwargs={
                "transition_proba_init": {"prob_stay": 0.999},
                "glm_params_init": {"std_dev": 0.0},
            },
        )

        params = model.initialize_params(X, y)
        coef = params[0]
        transition_proba = params[4]

        # Both kwargs should be applied
        assert jnp.allclose(coef, 0.0)
        assert jnp.allclose(jnp.diag(transition_proba), 0.999)
