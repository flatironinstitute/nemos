"""Tests for GLMHMM.fit and related fit-path validation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nemos as nmo


class TestClassifierGLMHMM:
    """
    Unit tests specific to classifier GLM.
    """

    @pytest.mark.solver_related
    @pytest.mark.parametrize("seed", [0, 123])
    @pytest.mark.parametrize("n_states", [2, 3])
    @pytest.mark.requires_x64
    def test_fit_glmhmm_matches_bernoulli(self, seed, n_states):
        """
        Ensure that the model fit matches the Bernoulli GLMHMM.
        Since it needs to be unregularized, we only check n_states=2 to reduce chance of numerical instability.
        """
        np.random.seed(seed)
        n_classes = 2
        X = np.random.normal(size=(100, 5))
        b_true = np.zeros((n_classes,))
        w_true = np.random.normal(size=(5, n_classes))
        rate = jax.nn.log_softmax(jnp.einsum("ki,tk->ti", w_true, X) + b_true)
        key = jax.random.PRNGKey(seed)
        y = jax.random.categorical(key, rate)

        model = nmo.glm_hmm.ClassifierGLMHMM(
            n_states=n_states,
            n_classes=n_classes,
            regularizer="Ridge",
            regularizer_strength=1.0,
            solver_name="LBFGS",
            solver_kwargs={"tol": 10**-8},
        ).fit(X, y)
        flat_coef = model.coef_[:, 1, :] - model.coef_[:, 0, :]
        flat_intercept = model.intercept_[1, :] - model.intercept_[0, :]

        model_check = nmo.glm_hmm.GLMHMM(
            n_states=n_states,
            observation_model="Bernoulli",
            regularizer="Ridge",
            regularizer_strength=0.5,
            solver_name="LBFGS",
            solver_kwargs={"tol": 10**-8},
        ).fit(X, y)
        np.testing.assert_array_almost_equal(
            np.sort(flat_coef.ravel()), np.sort(model_check.coef_.ravel()), decimal=4
        )
        np.testing.assert_array_almost_equal(
            np.sort(flat_intercept), np.sort(model_check.intercept_), decimal=5
        )
        np.testing.assert_array_almost_equal(
            np.sort(model.transition_prob_.ravel()),
            np.sort(model_check.transition_prob_.ravel()),
            decimal=5,
        )
        np.testing.assert_array_almost_equal(
            np.sort(model.initial_prob_), np.sort(model_check.initial_prob_), decimal=5
        )

    @pytest.mark.solver_related
    @pytest.mark.parametrize("seed", [0, 123])
    @pytest.mark.parametrize("n_classes", [2, 3])
    @pytest.mark.requires_x64
    def test_fit_glmhmm_matches_glm(self, seed, n_classes):
        """
        Ensure that the model fit matches the Bernoulli GLMHMM.
        Since it needs to be unregularized, we only check n_states=2 to reduce chance of numerical instability.
        """
        np.random.seed(seed)
        X = np.random.normal(size=(100, 5))
        b_true = np.zeros((n_classes,))
        w_true = np.random.normal(size=(5, n_classes))
        rate = jax.nn.log_softmax(jnp.einsum("ki,tk->ti", w_true, X) + b_true)
        key = jax.random.PRNGKey(seed)
        y = jax.random.categorical(key, rate)

        # force same initialization point as GLM
        init_params = (
            jnp.zeros((5, n_classes, 1)),
            jnp.zeros((n_classes, 1)),
            jnp.ones((n_classes, 1)),
            jnp.ones(1),
            jnp.ones((1, 1)),
        )
        model = nmo.glm_hmm.ClassifierGLMHMM(
            n_states=1,
            n_classes=n_classes,
            regularizer="UnRegularized",
            solver_name="LBFGS",
            solver_kwargs={"tol": 10**-8},
        ).fit(X, y, init_params=init_params)

        model_check = nmo.glm.ClassifierGLM(
            n_classes=n_classes,
            regularizer="UnRegularized",
            solver_name="LBFGS",
            solver_kwargs={"tol": 10**-8},
        ).fit(X, y)

        np.testing.assert_array_almost_equal(model.coef_[:, :, 0], model_check.coef_)
        np.testing.assert_array_almost_equal(
            model.intercept_[:, 0], model_check.intercept_
        )
