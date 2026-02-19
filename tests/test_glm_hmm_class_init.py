"""Tests for GLMHMM class __init__ and property setters."""

from contextlib import nullcontext as does_not_raise

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nemos as nmo


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
            (-1, pytest.raises(ValueError, match="must be a strictly positive integer")),
            (10.5, pytest.raises(ValueError, match="must be a strictly positive integer")),
            ("100", pytest.raises(ValueError, match="must be a strictly positive integer")),
            (None, pytest.raises(ValueError, match="must be a strictly positive integer")),
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
            (-1e-8, pytest.raises(ValueError, match="must be a strictly positive float")),
            ("0.001", pytest.raises(ValueError, match="must be a strictly positive float")),
            (None, pytest.raises(ValueError, match="must be a strictly positive float")),
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
        model = nmo.glm_hmm.GLMHMM(
            n_states=3, dirichlet_prior_alphas_init_prob=None
        )
        assert model.dirichlet_prior_alphas_init_prob is None

    def test_dirichlet_prior_init_prob_valid(self):
        """Test valid dirichlet prior alphas."""
        alphas = jnp.array([1.0, 2.0, 3.0])
        model = nmo.glm_hmm.GLMHMM(
            n_states=3, dirichlet_prior_alphas_init_prob=alphas
        )
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
        model = nmo.glm_hmm.GLMHMM(
            n_states=3, dirichlet_prior_alphas_transition=None
        )
        assert model.dirichlet_prior_alphas_transition is None

    def test_dirichlet_prior_transition_valid(self):
        """Test valid dirichlet prior alphas for transitions."""
        alphas = jnp.ones((3, 3))
        model = nmo.glm_hmm.GLMHMM(
            n_states=3, dirichlet_prior_alphas_transition=alphas
        )
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
        model = nmo.glm_hmm.GLMHMM(
            n_states=2, inverse_link_function=custom_link
        )
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
