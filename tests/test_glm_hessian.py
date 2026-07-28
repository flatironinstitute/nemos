"""
Validity tests for the analytic Fisher-scoring GLM Hessian.

Validity test checked against the autodiff Hessian evaluated at ``y = mu``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

import nemos as nmo
import nemos.inverse_link_function_utils as ilf
from nemos.glm.params import GLMParams

obs = nmo.observation_models

N_SAMPLES = 800
N_FEATURES = 4
N_NEURONS = 3
N_CLASSES = 3


def _cubic(x):
    return x**3


# Each model is tested with its canonical link and a non-canonical one
# (``g'(eta) != const``), the case where the Fisher and observed Hessians differ.
SUPPORTED_MODELS = {
    "poisson": (obs.PoissonObservations, ilf.exp, ilf.softplus),
    "gaussian": (obs.GaussianObservations, ilf.identity, _cubic),
    "gamma": (obs.GammaObservations, ilf.one_over_x, ilf.softplus),
    "bernoulli": (obs.BernoulliObservations, ilf.logistic, ilf.norm_cdf),
}

UNSUPPORTED_MODELS = {
    "negative_binomial": (obs.NegativeBinomialObservations, ilf.exp, ilf.softplus),
}

LINK_KINDS = {"canonical": 1, "non_canonical": 2}


def _make_params(rng, coef_shape, intercept_shape):
    coef = jnp.asarray(0.1 * rng.standard_normal(coef_shape))
    intercept = jnp.full(intercept_shape, 2.0)
    return GLMParams(coef, intercept)


def _fisher_via_autodiff(model, params, X):
    mu = jax.lax.stop_gradient(model._predict(params, X))
    flat, unravel = ravel_pytree(params)
    return np.asarray(
        jax.hessian(lambda fp: model._compute_loss(unravel(fp), X, mu))(flat)
    )


def _analytic_hessian(model, params, X):
    return np.asarray(model._get_hess_fn(params)(params, X))


def _population_block_indices(n_features, n_neurons, neuron):
    coef_idx = [i * n_neurons + neuron for i in range(n_features)]
    intercept_idx = [n_features * n_neurons + neuron]
    return np.array(coef_idx + intercept_idx)


def _categorical_neuron_indices(n_features, n_neurons, n_classes, neuron):
    coef_idx = [
        i * n_neurons * n_classes + neuron * n_classes + k
        for i in range(n_features)
        for k in range(n_classes)
    ]
    intercept_idx = [
        n_features * n_neurons * n_classes + neuron * n_classes + k
        for k in range(n_classes)
    ]
    return np.array(coef_idx + intercept_idx)


@pytest.mark.requires_x64
@pytest.mark.parametrize("model_id", list(SUPPORTED_MODELS))
@pytest.mark.parametrize("link_kind", list(LINK_KINDS))
class TestGLMHessian:
    def _build(self, model_id, link_kind, rng):
        om_factory, canonical, non_canonical = SUPPORTED_MODELS[model_id]
        link = canonical if link_kind == "canonical" else non_canonical
        X = rng.standard_normal((N_SAMPLES, N_FEATURES))
        model = nmo.glm.GLM(
            observation_model=om_factory(),
            inverse_link_function=link,
            regularizer="UnRegularized",
        )
        params = _make_params(rng, (N_FEATURES,), (1,))
        return model, params, X

    def test_matches_autodiff_fisher(self, model_id, link_kind):
        rng = np.random.default_rng(0)
        model, params, X = self._build(model_id, link_kind, rng)
        analytic = _analytic_hessian(model, params, X)
        ground_truth = _fisher_via_autodiff(model, params, X)
        np.testing.assert_allclose(analytic, ground_truth, atol=1e-6, rtol=1e-6)

    def test_symmetric(self, model_id, link_kind):
        rng = np.random.default_rng(0)
        model, params, X = self._build(model_id, link_kind, rng)
        analytic = _analytic_hessian(model, params, X)
        np.testing.assert_allclose(analytic, analytic.T, atol=1e-8)

    def test_positive_semidefinite(self, model_id, link_kind):
        rng = np.random.default_rng(0)
        model, params, X = self._build(model_id, link_kind, rng)
        analytic = _analytic_hessian(model, params, X)
        eigvals = np.linalg.eigvalsh(0.5 * (analytic + analytic.T))
        assert eigvals.min() >= -1e-8


@pytest.mark.requires_x64
@pytest.mark.parametrize("model_id", list(SUPPORTED_MODELS))
@pytest.mark.parametrize("link_kind", list(LINK_KINDS))
class TestPopulationGLMHessian:
    def _build(self, model_id, link_kind, rng):
        om_factory, canonical, non_canonical = SUPPORTED_MODELS[model_id]
        link = canonical if link_kind == "canonical" else non_canonical
        X = rng.standard_normal((N_SAMPLES, N_FEATURES))
        model = nmo.glm.PopulationGLM(
            observation_model=om_factory(),
            inverse_link_function=link,
            regularizer="UnRegularized",
        )
        params = _make_params(rng, (N_FEATURES, N_NEURONS), (N_NEURONS,))
        return model, params, X

    def test_blocks_match_autodiff_fisher(self, model_id, link_kind):
        rng = np.random.default_rng(0)
        model, params, X = self._build(model_id, link_kind, rng)
        analytic = _analytic_hessian(model, params, X)
        full = _fisher_via_autodiff(model, params, X)

        assert analytic.shape == (N_NEURONS, N_FEATURES + 1, N_FEATURES + 1)
        for neuron in range(N_NEURONS):
            idx = _population_block_indices(N_FEATURES, N_NEURONS, neuron)
            block = full[np.ix_(idx, idx)]
            np.testing.assert_allclose(analytic[neuron], block, atol=1e-6, rtol=1e-6)

    def test_cross_neuron_blocks_are_zero(self, model_id, link_kind):
        rng = np.random.default_rng(0)
        model, params, X = self._build(model_id, link_kind, rng)
        full = _fisher_via_autodiff(model, params, X)
        for a in range(N_NEURONS):
            for b in range(a + 1, N_NEURONS):
                ia = _population_block_indices(N_FEATURES, N_NEURONS, a)
                ib = _population_block_indices(N_FEATURES, N_NEURONS, b)
                np.testing.assert_allclose(full[np.ix_(ia, ib)], 0.0, atol=1e-8)


@pytest.mark.requires_x64
@pytest.mark.parametrize("model_id", list(SUPPORTED_MODELS))
def test_ridge_hessian_matches_penalized_autodiff(model_id):
    rng = np.random.default_rng(0)
    om_factory, canonical, _ = SUPPORTED_MODELS[model_id]
    strength = 0.7
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    model = nmo.glm.GLM(
        observation_model=om_factory(),
        inverse_link_function=canonical,
        regularizer="Ridge",
        regularizer_strength=strength,
    )
    params = _make_params(rng, (N_FEATURES,), (1,))

    analytic = _analytic_hessian(model, params, X)

    penalized_loss = model.regularizer.penalized_loss(
        model._compute_loss, params=params, strength=strength
    )
    mu = jax.lax.stop_gradient(model._predict(params, X))
    flat, unravel = ravel_pytree(params)
    ground_truth = np.asarray(
        jax.hessian(lambda fp: penalized_loss(unravel(fp), X, mu))(flat)
    )
    np.testing.assert_allclose(analytic, ground_truth, atol=1e-6, rtol=1e-6)


@pytest.mark.requires_x64
@pytest.mark.parametrize("model_id", list(UNSUPPORTED_MODELS))
@pytest.mark.parametrize("link_kind", list(LINK_KINDS))
def test_unsupported_observation_models_have_no_analytic_hessian(model_id, link_kind):
    rng = np.random.default_rng(0)
    om_factory, canonical, non_canonical = UNSUPPORTED_MODELS[model_id]
    link = canonical if link_kind == "canonical" else non_canonical
    model = nmo.glm.GLM(
        observation_model=om_factory(),
        inverse_link_function=link,
        regularizer="UnRegularized",
    )
    params = _make_params(rng, (N_FEATURES,), (1,))
    with pytest.raises(NotImplementedError, match="variance function"):
        model._get_hess_fn(params)


@pytest.mark.requires_x64
def test_classifier_glm_defers_to_autodiff_hessian():
    rng = np.random.default_rng(0)
    model = nmo.glm.ClassifierGLM(n_classes=N_CLASSES, regularizer="UnRegularized")
    model.set_classes(np.arange(N_CLASSES))
    params = _make_params(rng, (N_FEATURES, N_CLASSES), (N_CLASSES,))
    assert model._get_hess_fn(params) is None


@pytest.mark.requires_x64
def test_classifier_population_blocks_match_autodiff():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    labels = rng.integers(0, N_CLASSES, (N_SAMPLES, N_NEURONS))
    y = jax.nn.one_hot(labels, N_CLASSES)

    model = nmo.glm.ClassifierPopulationGLM(
        n_classes=N_CLASSES, regularizer="UnRegularized"
    )
    model.set_classes(np.arange(N_CLASSES))
    params = _make_params(
        rng, (N_FEATURES, N_NEURONS, N_CLASSES), (N_NEURONS, N_CLASSES)
    )

    blocks = np.asarray(model._get_hess_fn(params)(params, X, y))
    flat, unravel = ravel_pytree(params)
    full = np.asarray(
        jax.hessian(lambda fp: model._compute_loss(unravel(fp), X, y))(flat)
    )

    block_dim = (N_FEATURES + 1) * N_CLASSES
    assert blocks.shape == (N_NEURONS, block_dim, block_dim)
    for neuron in range(N_NEURONS):
        idx = _categorical_neuron_indices(N_FEATURES, N_NEURONS, N_CLASSES, neuron)
        sub = full[np.ix_(idx, idx)]
        # eigenvalues are invariant to the per-neuron parameter ordering convention
        np.testing.assert_allclose(
            np.linalg.eigvalsh(blocks[neuron]), np.linalg.eigvalsh(sub), atol=1e-6
        )
    for a in range(N_NEURONS):
        for b in range(a + 1, N_NEURONS):
            ia = _categorical_neuron_indices(N_FEATURES, N_NEURONS, N_CLASSES, a)
            ib = _categorical_neuron_indices(N_FEATURES, N_NEURONS, N_CLASSES, b)
            np.testing.assert_allclose(full[np.ix_(ia, ib)], 0.0, atol=1e-8)
