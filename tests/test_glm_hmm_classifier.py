"""Tests for methods of ClassifierGLMHMM."""

import inspect
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nemos.glm import ClassifierGLM
from nemos.glm_hmm import GLMHMM, ClassifierGLMHMM


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

        model = ClassifierGLMHMM(
            n_states=n_states,
            n_classes=n_classes,
            regularizer="Ridge",
            regularizer_strength=1.0,
            solver_name="LBFGS",
            solver_kwargs={"tol": 10**-8},
        ).fit(X, y)
        flat_coef = model.coef_[:, 1, :] - model.coef_[:, 0, :]
        flat_intercept = model.intercept_[1, :] - model.intercept_[0, :]

        model_check = GLMHMM(
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
        model = ClassifierGLMHMM(
            n_states=1,
            n_classes=n_classes,
            regularizer="UnRegularized",
            solver_name="LBFGS",
            solver_kwargs={"tol": 10**-8},
        ).fit(X, y, init_params=init_params)

        model_check = ClassifierGLM(
            n_classes=n_classes,
            regularizer="UnRegularized",
            solver_name="LBFGS",
            solver_kwargs={"tol": 10**-8},
        ).fit(X, y)

        np.testing.assert_array_almost_equal(model.coef_[:, :, 0], model_check.coef_)
        np.testing.assert_array_almost_equal(
            model.intercept_[:, 0], model_check.intercept_
        )

    @pytest.mark.requires_x64
    def test_n_update_steps_matches_fit(self):
        """fit(maxiter=N) and N manual update() calls from the same init produce identical params."""
        rng = np.random.default_rng(0)
        n, k, c, s = 80, 3, 2, 2
        X = rng.standard_normal((n, k))
        y = rng.binomial(1, 0.4, size=n)
        session_starts = jnp.zeros(n, dtype=bool).at[0].set(True)

        # shared init params so both paths start from exactly the same point
        seed = jax.random.PRNGKey(7)
        init_coef = jnp.zeros((k, c, s))
        init_intercept = jnp.zeros((c, s))
        init_scale = jnp.ones((c, s))
        init_initial_prob = jnp.ones(s) / s
        init_transition_prob = jnp.ones((s, s)) / s
        init_params = (
            init_coef,
            init_intercept,
            init_scale,
            init_initial_prob,
            init_transition_prob,
        )

        n_steps = 3

        # --- path A: fit with maxiter=n_steps ---
        model_fit = ClassifierGLMHMM(n_states=s, maxiter=n_steps, tol=1e-300, seed=seed)
        model_fit.fit(X, y, init_params=init_params)

        # --- path B: manual update loop ---
        model_update = ClassifierGLMHMM(
            n_states=s, maxiter=n_steps, tol=1e-300, seed=seed
        )
        model_update.set_classes(y)
        opt_state = model_update.initialize_optimizer_and_state(init_params, X, y)

        params = init_params
        for _ in range(n_steps):
            params, opt_state = model_update.update(
                params, opt_state, X, y, session_starts
            )

        np.testing.assert_allclose(model_fit.coef_, model_update.coef_)
        np.testing.assert_allclose(model_fit.intercept_, model_update.intercept_)
        np.testing.assert_allclose(model_fit.initial_prob_, model_update.initial_prob_)
        np.testing.assert_allclose(
            model_fit.transition_prob_, model_update.transition_prob_
        )

    def test_validator_extra_params(self):
        """``_get_validator_extra_params`` passes both n_classes and n_states along."""
        model = ClassifierGLMHMM(n_states=4, n_classes=3)
        assert model._get_validator_extra_params() == {"n_classes": 3, "n_states": 4}
        assert model._validator.extra_params == {"n_classes": 3, "n_states": 4}


class TestClassifierGLMHMMLabeling:

    @pytest.fixture
    def classifier_glm_hmm_labeled(self):
        """Two equivalent fits of the same data, one with default and one with string labels.

        Returns ``(X, y, labels, model_int, model_str)`` where ``labels[y] == y_str``, so
        every method output of ``model_str`` must map onto the ``model_int`` one through
        the label encoding.
        """
        np.random.seed(0)
        X = np.random.normal(size=(60, 3))
        y = np.random.binomial(n=1, p=0.5, size=60)
        labels = np.array(["a", "b"])

        kwargs = dict(n_states=2, seed=jax.random.PRNGKey(123))
        model_int = ClassifierGLMHMM(**kwargs).fit(X, y)
        model_str = ClassifierGLMHMM(**kwargs).fit(X, labels[y])
        return X, y, labels, model_int, model_str

    @pytest.mark.parametrize(
        "method_name",
        [
            "score",
            "decode_state",
            "smooth_proba",
            "filter_proba",
            "simulate",
            "update",
        ],
    )
    def test_must_set_classes_before_calling(
        self, method_name, classifier_glm_hmm_labeled
    ):
        """Every label-encoding method raises before touching its inputs if classes_ is unset."""
        *_, model, _ = classifier_glm_hmm_labeled
        model = deepcopy(model)
        model.classes_ = None

        # superset of all possible required inputs
        input_dict = {
            "X": None,
            "y": None,
            "params": None,
            "random_key": None,
            "feedforward_input": None,
            "opt_state": None,
        }
        method = getattr(model, method_name)
        required = [
            name
            for name, param in inspect.signature(method).parameters.items()
            if param.default is inspect.Parameter.empty
            and param.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]
        with pytest.raises(
            RuntimeError, match=rf"Classes are not set\..*{method_name}"
        ):
            method(**{k: input_dict[k] for k in required})

    def test_fit_from_label(self, classifier_glm_hmm_labeled):
        """``fit`` sets ``classes_`` from y and encodes it, matching the default-label fit."""
        _, _, labels, model_int, model_str = classifier_glm_hmm_labeled
        np.testing.assert_array_equal(model_str.classes_, labels)
        np.testing.assert_array_equal(model_int.classes_, np.arange(2))
        np.testing.assert_allclose(model_int.coef_, model_str.coef_)
        np.testing.assert_allclose(model_int.intercept_, model_str.intercept_)

    @pytest.mark.parametrize(
        "method_name", ["score", "decode_state", "smooth_proba", "filter_proba"]
    )
    def test_method_from_label(self, method_name, classifier_glm_hmm_labeled):
        """Methods that only encode y are invariant to the label representation."""
        X, y, labels, model_int, model_str = classifier_glm_hmm_labeled
        out_int = getattr(model_int, method_name)(X, y)
        out_str = getattr(model_str, method_name)(X, labels[y])
        np.testing.assert_allclose(out_int, out_str)

    def test_simulate_from_label(self, classifier_glm_hmm_labeled):
        """``simulate`` decodes the one-hot draws back to the user's labels."""
        X, _, labels, model_int, model_str = classifier_glm_hmm_labeled
        key = jax.random.PRNGKey(1)
        y_int, proba_int, states_int = model_int.simulate(key, X)
        y_str, proba_str, states_str = model_str.simulate(key, X)

        # default labels: the one-hot is argmax-ed but left encoded
        assert y_int.shape == (X.shape[0],)
        assert jnp.issubdtype(y_int.dtype, jnp.integer)
        np.testing.assert_array_equal(labels[y_int], y_str)
        # only y is decoded, the other outputs pass through untouched
        np.testing.assert_allclose(proba_int, proba_str)
        np.testing.assert_array_equal(states_int, states_str)

    def test_update_from_label(self, classifier_glm_hmm_labeled):
        """A single ``update`` step is invariant to the label representation."""
        X, y, labels, *_ = classifier_glm_hmm_labeled
        init_params = (
            jnp.zeros((X.shape[1], 2, 2)),
            jnp.zeros((2, 2)),
            jnp.ones((2, 2)),
            jnp.ones(2) / 2,
            jnp.ones((2, 2)) / 2,
        )
        out = []
        for y_ in (y, labels[y]):
            model = ClassifierGLMHMM(n_states=2)
            model.set_classes(y_)
            opt_state = model.initialize_optimizer_and_state(init_params, X, y_)
            out.append(model.update(init_params, opt_state, X, y_)[0])

        for param_int, param_str in zip(*out):
            np.testing.assert_allclose(param_int, param_str)

    def test_unrecognized_label_raises(self, classifier_glm_hmm_labeled):
        """Labels outside ``classes_`` are rejected by the encoding step."""
        X, y, _, _, model_str = classifier_glm_hmm_labeled
        y_invalid = np.full(y.shape, "z")
        with pytest.raises(ValueError, match="Unrecognized label"):
            model_str.smooth_proba(X, y_invalid)
