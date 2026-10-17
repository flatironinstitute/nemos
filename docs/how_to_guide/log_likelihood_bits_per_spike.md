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
  model: [GLM]
  observation_model: [Poisson]
  signal: [spike counts]
  topic: [model comparison]
  data: simulated
  description: Express the fit quality as the information a model carries about each spike.
---

(bits-per-spike)=
# Compute the Log-Likelihood in Bits per Spike

Bits per spike rescales the log-likelihood gain of a model over the [null model](./null_model_log_likelihood.md) by the number of spikes, giving a quantity that is comparable across neurons and recordings:

$$
\text{bits/spike} = \frac{\mathcal{L}_{\text{model}} - \mathcal{L}_{\text{null}}}{N_{\text{spikes}} \log 2},
$$

where both log-likelihoods are summed over all samples, and the $\log 2$ converts nats to bits.

```{code-cell} ipython3
import jax.numpy as jnp

import nemos as nmo

X, y = nmo.datasets.tuned_poisson(n_samples=500)

model = nmo.glm.GLM().fit(X, y)
```

[`score`](nemos.glm.GLM.score) gives the first term. The second one is scored by the observation model, since no model predicts the null rate. Both take `aggregate_sample_scores=jnp.sum`, so they are totals rather than per-sample averages.

```{code-cell} ipython3
ll_model = model.score(X, y, aggregate_sample_scores=jnp.sum)
ll_null = model.observation_model.log_likelihood(
    y, jnp.full_like(y, y.mean()), model.scale_, aggregate_sample_scores=jnp.sum
)
```

Dividing by the spike count and by $\log 2$ gives the bits per spike.

```{code-cell} ipython3
bits_per_spike = (ll_model - ll_null) / (y.sum() * jnp.log(2))

print("bits/spike:", bits_per_spike)
```

:::{warning}
The score is undefined for a neuron that never fires, where `y.sum()` is zero.
:::
