---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

# Manual update loop for more control

+++

While `GLM.stochastic_fit` provides a convenient interface for most usecases, in some situations fine-grained manual control during the optimization loop might be required.

+++

As a starting point for more advanced users needing such control, we show how to manually set up a training loop performing stochastic gradient descent (SGD).

+++

:::{warning}
SVRG is particularly suited for stochastic optimization, but its optimization loop is more involved than for SGD, so using `GLM.stochastic_fit` is highly recommended.
:::

+++

## Set up the data and model

```{code-cell} ipython3
import jax
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import seaborn as sns
from _toy_data_generation import _simulate_batching_data

import nemos as nmo

jax.enable_x64()
nap.nap_config.suppress_conversion_warnings = True
np.random.seed(123)
```

This approach doesn't require the use of a NeMoS data loader, you can load batches of data any way you want. Here, we will use in-memory data and `ArrayDataLoader` for simplicity.

```{code-cell} ipython3
units, spike_trains, X = _simulate_batching_data()
```

```{code-cell} ipython3
loader = nmo.batching.ArrayDataLoader(X, spike_trains, batch_size=100)
```

Use gradient descent as the solver with a constant stepsize and acceleration disabled:

```{code-cell} ipython3
glm = nmo.glm.PopulationGLM(
    solver_name="GradientDescent",
    solver_kwargs={"stepsize": 0.01, "acceleration": False},
)
```

## Manual update loop calling `GLM.update`

+++

Basic SGD can effectively be reproduced by the following manual loop with score logging included after every batch:

```{code-cell} ipython3
n_epochs = 10
```

```{code-cell} ipython3
X_sample, y_sample = loader.sample_batch()
params = glm.initialize_params(X_sample, y_sample)
opt_state = glm.initialize_optimizer_and_state(params, X_sample, y_sample)

scores = []
for i in range(n_epochs):
    for X_batch, y_batch in loader:
        params, opt_state = glm.update(params, opt_state, X_batch, y_batch)
        scores.append(glm.score(X, spike_trains))
```

Plotting the resulting scores shows a similar curve as in previous sections using `GLM.stochastic_fit`:

```{code-cell} ipython3
fig, ax = plt.subplots()

ax.plot(scores)
ax.set_xlabel("Batch number")
ax.set_ylabel("Log-likelihood")

sns.despine(ax=ax)
```
