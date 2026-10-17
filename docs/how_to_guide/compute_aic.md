---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
nemos_tags:
  model: [GLM, ClassifierGLM]
  observation_model: [Poisson, Gamma, Categorical]
  signal: [spike counts, continuous, behavior choices]
  topic: [model comparison]
  data: simulated
  description: Penalize the log-likelihood with the parameter count, for a Poisson, a Gamma and a Categorical fit.
---

(compute-aic)=
# Compute the AIC

Adding predictors can only raise the log-likelihood, so a comparison between fits of different size must account for the number of parameters. Akaike's information criterion penalizes the log-likelihood by two nats per parameter,

$$
\text{AIC} = -2\mathcal{L} + 2k,
$$

with $\mathcal{L}$ the total log-likelihood and $k$ the number of parameters the fit estimates. Among models compared on the same observations, the one with the lowest AIC is preferred.

An exponential-family distribution is specified by its mean and, for some families, by a scale: for example, the Gamma and the Gaussian have one, while the Poisson, the Bernoulli and the Categorical fix it at 1. A GLM estimates the mean through the coefficients and the intercept, and the scale afterwards, from the residuals of the fitted mean. Both count as model parameters when computing $k$.

Each example below computes the log-likelihood twice, once from the [observation model](../user_guide/observation_models.md) and once from the model. The first route is pedagogical: it spells out every step, including the scale estimation that a fit performs internally.

```{code-cell} ipython3
import jax
import jax.numpy as jnp

import nemos as nmo
```

## Poisson GLM

[`score`](nemos.glm.GLM.score) and [`log_likelihood`](nemos.observation_models.Observations.log_likelihood) reduce the per-sample log-likelihoods with `aggregate_sample_scores`, which defaults to their mean. The AIC is defined on the total, so pass `jnp.sum`.

```{code-cell} ipython3
X, y = nmo.datasets.tuned_poisson(n_samples=500)

model = nmo.glm.GLM().fit(X, y)
obs = model.observation_model

ll = obs.log_likelihood(y, model.predict(X), aggregate_sample_scores=jnp.sum)

print("observation model:", ll)
print("model.score:", model.score(X, y, aggregate_sample_scores=jnp.sum))
```

The Poisson scale is fixed at 1, so the parameters are the coefficients and the intercept.

```{code-cell} ipython3
n_params = model.coef_.size + model.intercept_.size

print("AIC:", -2 * ll + 2 * n_params)
```

## Gamma GLM

The Gamma estimates its scale from the residuals of the fitted mean, so it is a parameter of its own and $k$ gains one. [`estimate_scale`](nemos.observation_models.Observations.estimate_scale) is the call the fit makes internally: it takes the residual degrees of freedom, the number of samples minus the number of mean parameters, and returns the same value as `scale_`.

```{code-cell} ipython3
X, y = nmo.datasets.tuned_gamma(n_samples=500)

model = nmo.glm.GLM(observation_model="Gamma").fit(X, y)
obs = model.observation_model

rate = model.predict(X)
n_mean_params = model.coef_.size + model.intercept_.size
scale = obs.estimate_scale(y, rate, dof_resid=X.shape[0] - n_mean_params)

print("estimate_scale:", scale)
print("model.scale_:", model.scale_)
```

The scale enters the log-likelihood, so it has to be passed explicitly when scoring from the observation model.

```{code-cell} ipython3
ll = obs.log_likelihood(y, rate, scale, aggregate_sample_scores=jnp.sum)

print("observation model:", ll)
print("model.score:", model.score(X, y, aggregate_sample_scores=jnp.sum))
```

```{code-cell} ipython3
# coefficients and intercept, plus one scale per neuron
n_params = n_mean_params + scale.size

print("AIC:", -2 * ll + 2 * n_params)
```

## Categorical GLM

The [`ClassifierGLM`](nemos.glm.ClassifierGLM) predicts one probability per class, so its parameters form a `(n_features, n_classes)` matrix. The observation model takes the observations one-hot encoded and the rate as log-probabilities, which is what [`predict_proba`](nemos.glm.ClassifierGLM.predict_proba) returns by default.

```{code-cell} ipython3
X, y = nmo.datasets.tuned_categorical(n_samples=500, n_classes=3)

model = nmo.glm.ClassifierGLM(classes=3).fit(X, y)
obs = model.observation_model

ll = obs.log_likelihood(
    jax.nn.one_hot(y.astype(int), 3),
    model.predict_proba(X),
    aggregate_sample_scores=jnp.sum,
)

print("observation model:", ll)
print("model.score:", model.score(X, y, aggregate_sample_scores=jnp.sum))
```

Here `coef_.size + intercept_.size` overcounts: the class probabilities sum to one, so only $C - 1$ of the $C$ columns are free. Each free column holds one coefficient per feature and one intercept, so the fit estimates $(n_{\text{features}} + 1)(C - 1)$ parameters.

```{code-cell} ipython3
n_features, n_classes = model.coef_.shape
n_params = (n_features + 1) * (n_classes - 1)

print("all coefficients:", model.coef_.size + model.intercept_.size)
print("free parameters:", n_params)
print("AIC:", -2 * ll + 2 * n_params)
```
