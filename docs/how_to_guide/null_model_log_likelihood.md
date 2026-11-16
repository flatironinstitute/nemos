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
  model: []
  observation_model: [Poisson]
  signal: [spike counts]
  topic: [model comparison]
  data: simulated
  description: Score the constant-rate model with an observation model object alone, without fitting a GLM.
---

(null-model-log-likelihood)=
# Compute the Null-Model Log-Likelihood

The null model predicts the same rate at every time point, the mean of the observations. It serves as a minimal baseline: a common question is how much better a model does than the mean rate, and that is what the [pseudo-$R^2$](pseudo-r2) and the [bits per spike](./log_likelihood_bits_per_spike.md) measure.

Nothing has to be fitted to get it: an [observation model](../user_guide/observation_models.md) scores any rate you hand it, so the null-model log-likelihood is one call on a constant array.

```{code-cell} ipython3
import jax.numpy as jnp

import nemos as nmo

_, y = nmo.datasets.tuned_poisson(n_samples=500)

obs = nmo.observation_models.PoissonObservations()
```

The null rate is the mean of `y`, repeated at every sample,

```{code-cell} ipython3
null_rate = jnp.full_like(y, y.mean())
```

and [`log_likelihood`](nemos.observation_models.Observations.log_likelihood) evaluates it like any other rate.

```{code-cell} ipython3
ll_null = obs.log_likelihood(y, null_rate)

print("null log-likelihood:", ll_null)
```

## Several neurons

Each neuron gets its own mean, so take the mean along the sample axis and broadcast it back.

```{code-cell} ipython3
_, y_pop = nmo.datasets.tuned_poisson(n_neurons=3, n_samples=500)

null_rate_pop = jnp.broadcast_to(y_pop.mean(axis=0), y_pop.shape)

print("null log-likelihood:", obs.log_likelihood(y_pop, null_rate_pop))
```

## Families with a free scale

The Poisson and the Bernoulli have a fixed scale, but the Gamma and the Gaussian estimate one, and the null model has its own: it is estimated on the null rate, with $n - 1$ residual degrees of freedom, and passed to `log_likelihood`.

```{code-cell} ipython3
_, y_gamma = nmo.datasets.tuned_gamma(n_samples=500)

gamma = nmo.observation_models.GammaObservations()
null_rate_gamma = jnp.full_like(y_gamma, y_gamma.mean())
null_scale = gamma.estimate_scale(
    y_gamma, null_rate_gamma, dof_resid=y_gamma.shape[0] - 1
)

ll_null_gamma = gamma.log_likelihood(y_gamma, null_rate_gamma, null_scale)

print("null log-likelihood:", ll_null_gamma)
```
