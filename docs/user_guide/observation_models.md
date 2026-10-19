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
---

```{code-cell} ipython3
:tags: [hide-input]

%matplotlib inline
import warnings

warnings.filterwarnings(
    "ignore",
    message="plotting functions contained within `_documentation_utils` are intended for nemos's documentation.",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message="Ignoring cached namespace 'core'",
    category=UserWarning,
)
```

(observation-models)=
# Observation Models

An `Observations` instance models one distribution of the [exponential family](https://en.wikipedia.org/wiki/Exponential_family). Given observations and a predicted mean it returns the log-likelihood, the deviance, the pseudo-$R^2$, an estimate of the scale, and samples from the distribution.

Models such as [`GLM`](nemos.glm.GLM) and [`GLMHMM`](nemos.glm_hmm.GLMHMM) store one such object as an attribute, and use it to define their loss function and to compute model statistics. This page describes the object itself, which is of interest if you are writing a model of your own, or if you want these statistics for a model you fit elsewhere.

```{contents}
:local:
:depth: 2
```

## Supported Noise Distributions

The available families are:

| Observation model | Data modeled |
| --- | --- |
| [`PoissonObservations`](nemos.observation_models.PoissonObservations) | spike counts |
| [`NegativeBinomialObservations`](nemos.observation_models.NegativeBinomialObservations) | overdispersed counts |
| [`GammaObservations`](nemos.observation_models.GammaObservations) | positive continuous, e.g. calcium traces |
| [`GaussianObservations`](nemos.observation_models.GaussianObservations) | unbounded continuous |
| [`BernoulliObservations`](nemos.observation_models.BernoulliObservations) | binary events |
| [`CategoricalObservations`](nemos.observation_models.CategoricalObservations) | one of $K$ labels |


## Background

### The exponential family

A distribution belongs to the exponential family when its density can be written as

$$
f(y; \theta, \phi) = h(y, \phi) \exp\left( a(\phi) \left[ y\theta - k(\theta) \right] \right),
$$

with $\theta$ the canonical parameter, $k$ the cumulant function, $\phi$ the scale, and $h$ the factor that normalizes the density.

It can be shown that the first two moments of $Y$ are both obtained by differentiating $k$. The mean is

$$
\mu = \mathbb{E}[Y] = k'(\theta),
$$

and the variance is

$$
\operatorname{var}(Y) = \frac{k''(\theta)}{a(\phi)}.
$$

The mean determines $\theta$ through the first identity, so the variance can be expressed as a function of the mean alone, $\operatorname{var}(Y) = V(\mu)/a(\phi)$, with $V = k'' \circ (k')^{-1}$ the variance function.

Both identities are obtained by differentiating the integral of the density under the parameter and using the score function. The argument is worked out step by step in Chapter 3 of [An Introduction to Generalized Linear Models](https://www.routledge.com/An-Introduction-to-Generalized-Linear-Models/Dobson-Barnett/p/book/9781138741515) by Dobson and Barnett, under "Properties of distributions in the exponential family".

### The parametrization used here

Every method below takes the mean and the scale of the distribution as its `predicted_rate` and `scale` arguments, with `predicted_rate` named after the spike-count case where $\mu$ is a firing rate. To map back to the textbook parametrization, recover the canonical parameter by inverting $\mu = k'(\theta)$.

Anything that produces a predicted mean can use these objects, however that mean was obtained: a weighted sum of predictors, the output layer of a network, the emission of a state space model.

:::{dropdown} Can $\mu = k'(\theta)$ always be inverted?
:color: info
:icon: info

Yes, for every family here. Differentiating twice gives $k''(\theta) = a(\phi)\operatorname{var}(Y)$, which is positive wherever $a(\phi) > 0$ and $\operatorname{var}(Y) > 0$. So $k'$ is strictly increasing, and it maps the canonical parameter onto the range of $\mu$ one to one.
:::

Each family names $\mu$ and $\phi$ differently:

| Distribution | $\mu$ | Range of $\mu$ | $\phi$ |
| --- | --- | --- | --- |
| Poisson | rate $\lambda$ | $(0, \infty)$ | fixed to 1 |
| Negative Binomial | mean | $(0, \infty)$ | fixed to 1; the dispersion is a separate parameter |
| Gamma | mean | $(0, \infty)$ | $1/\phi$ the shape, $\mu\phi$ the scale |
| Gaussian | mean | $(-\infty, \infty)$ | variance $\sigma^2$ |
| Bernoulli | success probability $p$ | $(0, 1)$ | fixed to 1 |
| Categorical | probability $\mu_k$ of each label | $(0, 1)$ | fixed to 1 |

The Categorical methods take $\log \mu_k$ rather than $\mu_k$, which is what `log_softmax` produces, because fitting in log space is numerically more stable.

### The log-likelihood of each family

Every panel below gives the log-likelihood of a single observation in terms of $\mu$ and $\phi$, which is what `log_likelihood` computes, and states what `predicted_rate` is.

::::{tab-set}

:::{tab-item} Poisson

$$
\ell(y; \mu) = y \log\mu - \mu - \log(y!)
$$

`predicted_rate` is $\mu$, the rate. The Poisson log-likelihood depends only on $\mu$, and `scale` is fixed to 1 by convention.
:::

:::{tab-item} Negative Binomial

$$
\ell(y; \mu, r) = \log\Gamma(y + r) - \log\Gamma(y + 1) - \log\Gamma(r) + y \log\frac{\mu}{\mu + r} + r \log\frac{r}{\mu + r}
$$

`predicted_rate` is $\mu$, the mean, and $r$ is the shape, the reciprocal of the `dispersion` set at initialization. The variance is $\mu + \mu^2/r$. `scale` is fixed to 1.
:::

:::{tab-item} Gamma

Shape $1/\phi$ and mean $\mu$, so the rate is $1/(\mu\phi)$,

$$
\ell(y; \mu, \phi) = \left( \frac{1}{\phi} - 1 \right)\log y - \frac{y}{\mu\phi} - \frac{1}{\phi}\log(\mu\phi) - \log\Gamma\!\left(\frac{1}{\phi}\right)
$$

`predicted_rate` is $\mu$, the mean. `scale` is $\phi$, and the variance is $\phi\mu^2$.
:::

:::{tab-item} Gaussian

$$
\ell(y; \mu, \phi) = -\frac{(y - \mu)^2}{2\phi} - \frac{1}{2}\log(2\pi\phi)
$$

`predicted_rate` is $\mu$, the mean. `scale` is the variance $\phi$.
:::

:::{tab-item} Bernoulli

$$
\ell(y; \mu) = y \log\mu + (1 - y)\log(1 - \mu)
$$

`predicted_rate` is $\mu$, the success probability. The Bernoulli log-likelihood depends only on $\mu$, and `scale` is fixed to 1 by convention.
:::

:::{tab-item} Categorical

With $y$ one-hot over $K$ labels,

$$
\ell(y; \mu) = \sum_{k=1}^{K} y_k \log \mu_k
$$

`predicted_rate` is $\log \mu_k$, one log-probability per label. The Categorical log-likelihood depends only on $\log \mu_k$, and `scale` is fixed to 1 by convention.

Since $y$ is one-hot encoded, multiplying $[\log\mu_1, \ldots, \log\mu_K]$ elementwise by $[y_1, \ldots, y_K]$ keeps only the entry of the label that occurred, and zeroes the rest. The log-likelihood of a sample is therefore the log-probability of the observed label. The likelihood needs only $\log\mu_k$, so `predicted_rate` is $\log\mu_k$. Computing a small $\mu_k$ first and then taking its logarithm loses precision.
:::

::::

:::{admonition} The Negative Binomial dispersion selects the distribution
:class: note

"Negative Binomial" names a collection of distributions, one for each shape $r$, the reciprocal of the dispersion. Fix $r$ and the result is an exponential family in the mean, with $\theta = \log\frac{\mu}{\mu + r}$, $k(\theta) = -r\log(1 - e^{\theta})$ and $a(\phi) = 1$. Differentiating gives $k'(\theta) = \mu$ and $k''(\theta) = \mu + \mu^2/r$, the mean and the variance.

Let $r$ vary and the collection is no longer one exponential family. The variance would have to be $V(\mu)/a(r)$ for a single variance function $V$, and the ratio of $\mu + \mu^2/r$ at two values $r_1 \neq r_2$ is $(1 + \mu/r_1)/(1 + \mu/r_2)$, which depends on $\mu$ while $a(r_2)/a(r_1)$ does not.

So $r$ is not passed to the methods beside the mean and the scale: it decides which distribution the instance models, and is set once, at construction.
:::

## Initialization

Most observation models take no argument at initialization.

```{code-cell} ipython3
import nemos as nmo

poisson = nmo.observation_models.PoissonObservations()
gamma = nmo.observation_models.GammaObservations()

print(poisson)
print(gamma)
```

The only exception is `NegativeBinomialObservations`, which takes a `dispersion`. The model does not learn it from the data, so you must set it at initialization. Writing the dispersion as $1/r$, the variance of the distribution is $\mu + \mu^2/r$: a larger `dispersion` means counts more overdispersed relative to a Poisson of the same mean, and a `dispersion` approaching zero recovers the Poisson.

```{code-cell} ipython3
neg_binomial = nmo.observation_models.NegativeBinomialObservations(dispersion=4.0)

neg_binomial
```

The value is exposed as a read-only property, so it cannot be changed after construction.

```{code-cell} ipython3
neg_binomial.dispersion
```

## Class attributes

`rate_range` and `glm_convexity_preserving_links` are class attributes: their value depends only on the observation model class.

### `rate_range`

The interval the mean of the observations, $\mu = \mathbb{E}[Y]$, has to lie in.

```{code-cell} ipython3
gaussian = nmo.observation_models.GaussianObservations()
poisson = nmo.observation_models.PoissonObservations()
categorical = nmo.observation_models.CategoricalObservations()
print("Gaussian mean range: ", gaussian.rate_range)
print("Poisson mean range: ", poisson.rate_range)
# the mean of a one-hot observation is the frequency of that label
print("Categorical mean range: ", categorical.rate_range)
```

### `glm_convexity_preserving_links`

A GLM predicts the mean as $\mu = f(Xw + c)$, and $\mathbf{f}$ is what NeMoS calls the **inverse link function**. The log-likelihood is a function of $\mu$, so $f$ is part of it too. This means that the properties of the log-likelihood depends on which inverse link $f$ is chosen.

Not every choice leaves the log-likelihood convex, and convexity is worth having: it guarantees a unique optimal solution, independent of the intial parameter guess. Each class lists the non-linearities known to preserve convexity.

```{code-cell} ipython3
nb = nmo.observation_models.NegativeBinomialObservations()

print("GLM convexity preserving inverse links:")
print("Poisson:          ", [f.__name__ for f in poisson.glm_convexity_preserving_links])
print("NegativeBinomial: ", [f.__name__ for f in nb.glm_convexity_preserving_links])
```

The Poisson keeps convexity with either one. The Negative Binomial keeps it only with `exp`, and `softplus` loses it.

## Methods

The examples below use a small Poisson dataset and the rate predicted by a GLM fit to it.

```{code-cell} ipython3
X, y = nmo.datasets.tuned_poisson(n_samples=500)

glm = nmo.glm.GLM().fit(X, y)
predicted_rate = glm.predict(X)
```

### Log-likelihood

[`log_likelihood`](nemos.observation_models.Observations.log_likelihood) computes the mean log-likelihood of the observations given the predicted mean. Both arrays have the same shape, one entry per sample.

```{code-cell} ipython3
poisson.log_likelihood(y, predicted_rate)
```

The number returned is the mean log-likelihood per sample.

The per-sample values are passed through `aggregate_sample_scores`, which can be any function and defaults to the mean. For the total log-likelihood,

```{code-cell} ipython3
import jax.numpy as jnp

poisson.log_likelihood(y, predicted_rate, aggregate_sample_scores=jnp.sum)
```

For the log-likelihood of each sample,

```{code-cell} ipython3
per_sample = poisson.log_likelihood(
    y, predicted_rate, aggregate_sample_scores=lambda x: x
)
per_sample.shape
```

[`likelihood`](nemos.observation_models.Observations.likelihood) is the exponential of the same quantity.

:::{admonition} Objective function
:class: note

A model objective is minimized, so what a model optimizes is the negative log-likelihood. The full log-likelihood usually includes normalization terms that do not depend on the parameters, and dropping them does not move the minimum. Models therefore optimize the private `_negative_log_likelihood`, which computes the parameter-dependent part alone.
:::

### Scale estimation

The scale $\phi$ sets the variance left over once the mean is fixed. [`estimate_scale`](nemos.observation_models.Observations.estimate_scale) estimates it from the residuals, correcting for the number of parameters in the model.

The Gamma is a family that has one. Fit a Gamma GLM to Gamma-distributed data,

```{code-cell} ipython3
X_gamma, y_gamma = nmo.datasets.tuned_gamma(n_samples=500)

glm_gamma = nmo.glm.GLM(observation_model="Gamma").fit(X_gamma, y_gamma)
rate_gamma = glm_gamma.predict(X_gamma)
```

and estimate the scale from the residuals,

```{code-cell} ipython3
# the parameters free to vary in the model:
# one per column of X, plus the intercept
n_parameters = X_gamma.shape[1] + 1
n_samples = X_gamma.shape[0]

dof_resid = n_samples - n_parameters

gamma.estimate_scale(y_gamma, rate_gamma, dof_resid=dof_resid)
```

A fitted model does this for you and stores the result in `scale_`.

```{code-cell} ipython3
glm_gamma.scale_, glm_gamma.dof_resid_
```

The Poisson has no scale to estimate: it is fixed at 1 by the family, and the method returns 1 whatever the data.

```{code-cell} ipython3
poisson.estimate_scale(y, predicted_rate, dof_resid=dof_resid)
```

The Bernoulli, the Categorical and the Negative Binomial behave the same way, and the Gaussian is estimated like the Gamma. The Negative Binomial `dispersion` is a separate parameter, and is not estimated either.

### Sampling

[`sample_generator`](nemos.observation_models.Observations.sample_generator) draws from the distribution at a given mean. It takes a [JAX random key](https://docs.jax.dev/en/latest/random-numbers.html), so the same key and the same rate always give the same draw.

```{code-cell} ipython3
import jax

poisson.sample_generator(jax.random.key(0), predicted_rate)[:10]
```

### Deviance

The [deviance](https://en.wikipedia.org/wiki/Deviance_(statistics)) measures how far the model is from a saturated model, one that fits every observation exactly:

$$
D(y, \hat{\mu}) = 2 \left[ \log p(y \mid y) - \log p(y \mid \hat{\mu}) \right].
$$

It is zero for a perfect fit and grows as the fit worsens, which makes it the natural residual for a non-Gaussian family: for the Gaussian it reduces to the sum of squared residuals. [`deviance`](nemos.observation_models.Observations.deviance) returns one value per sample.

```{code-cell} ipython3
deviance = poisson.deviance(y, predicted_rate)

print(deviance.shape)
print(deviance.sum())  # total residual deviance
```

(pseudo-r2)=
### Pseudo-$R^2$

The log-likelihood is unbounded, so its magnitude is not informative on its own. [`pseudo_r2`](nemos.observation_models.Observations.pseudo_r2) compares the model to a null model that predicts the mean of `y` for every sample, giving a number close to 1 for a good fit and close to 0 for a model no better than the null.

```{code-cell} ipython3
print(poisson.pseudo_r2(y, predicted_rate, score_type="pseudo-r2-McFadden"))
print(poisson.pseudo_r2(y, predicted_rate, score_type="pseudo-r2-Cohen"))
```

## Observation models as model attributes

A model takes its observation model by name,

```{code-cell} ipython3
model = nmo.glm.GLM(observation_model="Gamma")

model.observation_model
```

and instantiates the object internally, so the attribute is the object itself and all the methods above are reachable from the model.

```{code-cell} ipython3
type(model.observation_model)
```

Passing an instance instead is what you need when the family has a parameter to set, which among the families above is the Negative Binomial dispersion.

```{code-cell} ipython3
obs = nmo.observation_models.NegativeBinomialObservations(dispersion=2.0)

model = nmo.glm.GLM(observation_model=obs)
model.observation_model
```
