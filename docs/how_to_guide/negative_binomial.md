---
jupyter:
  jupytext:
    format_version: 0.13
    formats: ipynb,md
    jupytext_version: 1.16.4
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: nemos
    language: python
    name: python3
---

# Comparing fit using Negative Binomial and Poisson GLMS


:::{admonition} The Poisson GLM is pretty good - Why should I be interested in this? 

As stated in [1]:
- Poisson distributions assume a variance equal to the mean, an assumption which does not hold in many brain areas.
- Moreover, one could be interested in performing Bayesian inference, which cannot occurr using a Poisson model, as the posterior formed under Poisson likelihood and Gaussian prior has no tractable representation

To curve these limitations, one could use the negative-binomial (NB) distribution, which constitutes a generalization of the Poisson with a scale parameter that modulates the tradeoff between mean and variance.
:::


Before digging into the comparison, we can first import the packages we will use for this tutorial

```python
# Imports
import jax
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap
import nemos as nmo
import sklearn
from scipy.stats import nbinom, poisson
```
And generate some synthetic data. We will create three sets of data, each with different levels of dispersion.

```python
# Set seed for reproducibility
np.random.seed(111)

# Parameters
mu = 10  # Mean count
n_samples = 1000  # Number of trials

# Poisson data (no overdispersion)
poisson_data = np.random.poisson(mu, size=n_samples)

# Negative Binomial data with different dispersion (r)
# Note: NB parameterization: r = dispersion, p = success prob
# Mean = r * (1 - p) / p → solve for p
def generate_nb_data(mu, r, size):
    p = r / (mu + r)
    return nbinom.rvs(r, p, size=size)

nb_data_low = generate_nb_data(mu, r=10000000000000000, size=n_samples)  # slight overdispersion
nb_data_med = generate_nb_data(mu, r=5, size=n_samples)   # moderate
nb_data_high = generate_nb_data(mu, r=1, size=n_samples)  # heavy overdispersion
```

```python
# random design tensor. Shape (n_time_points, n_features).
X = 0.5*np.random.normal(size=(n_samples, 5))
```

We can take a look at it

```python
# Plot histograms
plt.figure(figsize=(10, 6))
sns.histplot(poisson_data, kde=False, color="blue", label="Poisson", stat="density", bins=30)
sns.histplot(nb_data_low, kde=False, color="green", label="NegBin r=20", stat="density", bins=30)
sns.histplot(nb_data_med, kde=False, color="orange", label="NegBin r=5", stat="density", bins=30)
sns.histplot(nb_data_high, kde=False, color="red", label="NegBin r=1", stat="density", bins=30)

plt.title("Simulated Count Data with Increasing Overdispersion")
plt.xlabel("Count")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()
```

## Create models
```python
poisson_model = nmo.glm.GLM()

binom_model = nmo.glm.GLM(
    observation_model="Negative"
)
```

:::{admonition}
We are not passing an `observation_model` when initializing our `poisson_model` because the Poisson observation model is default.
:::




```python

```

:::{admonition} What is overdispersion?

:::

```python
binom_model
```

```python

```

# References
[1] Pillow, Jonathan, and James Scott. "Fully Bayesian inference for neural models with negative-binomial spiking." Advances in neural information processing systems 25 (2012).
