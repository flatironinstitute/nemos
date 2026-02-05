---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
:tags: [hide-input]

%matplotlib inline
import warnings

# Ignore the first specific warning
warnings.filterwarnings(
    "ignore",
    message="plotting functions contained within `_documentation_utils` are intended for nemos's documentation.",
    category=UserWarning,
)

# Ignore the second specific warning
warnings.filterwarnings(
    "ignore",
    message="Ignoring cached namespace 'core'",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=(
        "invalid value encountered in div "
    ),
    category=RuntimeWarning,
)
```

# Regularizing parameters with different strengths

NeMoS allows for regularizing individual parameters with different regulariation strengths.
By passing structures of regularization strengths that match the parameter structure, you can get fine control over how parameters are regularized.


## Traditional regularization: all parameters are regularized equally
We will first generate some synthetic data with two feature groups:
:::{note}
We will store the features in a [`FeaturePytrees`](nemos.pytrees.FeaturePytree), take a look at [the tutorial on pytrees](/how_to_guide/plot_07_glm_pytree.md) to find out what they are.
:::


```{code-cell} ipython3
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import nemos as nmo

np.random.seed(123)
n_samples = 500
n_features = 7

# random design matrix, containing two feature groups: f1 and f2
X = nmo.pytrees.FeaturePytree(
    f1=0.5*np.random.normal(size=(n_samples, 5)),
    f2=0.5*np.random.normal(size=(n_samples, 2)),
)

# log-rates & weights
b_true = 1.0
w1_true = np.random.uniform(size=(5,))
w2_true = np.random.uniform(size=(2,))

# generate counts (spikes will be (n_samples, )
rate = jnp.exp(jnp.dot(X['f1'], w1_true) + jnp.dot(X['f2'], w2_true)+ b_true)
spikes = np.random.poisson(rate)

print(spikes.shape)
```

Let's start with the traditional case where every parameter is regularized with the same strength.
We'll fit a GLM with [`Ridge`](nemos.regularizer.Ridge) regression (this all works identically for [`Lasso`](nemos.regularizer.Lasso) regression):
```{code-cell} ipython3
glm = nmo.glm.GLM(regularizer="Ridge", regularizer_strength=0.1)
glm.fit(X, spikes)
```

In this case, the regularization strength of `0.1` is used for all 7 parameters (the intercept is not regularized).

## Group-wise regularization: all parameters within a group are regularized equally
If we want different regularization strengths for the two parameter groups, we can pass a dictionary matching the design matrix:

```{code-cell} ipython3
glm = nmo.glm.GLM(
    regularizer="Ridge", 
    regularizer_strength=dict(f1=0.1, f2=0.2)
)
glm.fit(X, spikes)
```

## Parameter-wise regularization: every parameter has their own regularization strength
If we want even finer control over regularization, we can pass arrays within the dictionary that match the design matrix:

```{code-cell} ipython3
glm = nmo.glm.GLM(
    regularizer="Ridge", 
    regularizer_strength=dict(
        f1=[0.1, 0.3, 0.3, 0.1, 1.0], 
        f2=[0.2, 0.1]
    )
)
glm.fit(X, spikes)
```

You can also mix different approaches, such as passing a single value for one group, and a list for the other:
```{code-cell} ipython3
glm = nmo.glm.GLM(
    regularizer="Ridge", 
    regularizer_strength=dict(
        f1=0.1, 
        f2=[0.2, 0.1]
    )
)
glm.fit(X, spikes)
```

## Special cases
There are a couple special cases to keep in mind!

### ElasticNet regularization
[`ElasticNet`](nemos.regularizer.ElasticNet) regularization  combines L1 and L2 regularization, introducing a `ratio` parameter that determines the relative contribution of either.
In the traditional case, you can pass the strength and ratio as a tuple:

```{code-cell} ipython3
glm = nmo.glm.GLM(regularizer="ElasticNet", regularizer_strength=(1.0, 0.5))
glm.fit(X, spikes)
```

However, if you want finer control, you can again pass a dictionary matching the parameter structure: this time one for the strenghts, and one for the ratios:
```{code-cell} ipython3
glm = nmo.glm.GLM(
    regularizer="ElasticNet", 
    regularizer_strength=(
        dict( # strength
            f1=[0.1, 0.3, 0.3, 0.1, 1.0], 
            f2=[0.2, 0.1]
        ),
        dict( # ratio
            f1=[0.5, 0.3, 0.5, 0.5, 0.5], 
            f2=[0.5, 0.4]
        ),
    )
)
glm.fit(X, spikes)
```

### GroupLasso regularization
[`GroupLasso`](nemos.regularizer.GroupLasso) works like [`Lasso`](nemos.regularizer.Lasso), but it works on groups of features instead of individual features.
It either keeps all features in a group, or shrinks the whole group to zero.

Regularizing individual parameters differently in [`GroupLasso`](nemos.regularizer.GroupLasso) does not make sense.
Instead, NeMoS allows for regularizing each group differently.
Again, you pass a dictionary, but now matching the groups, instead of the parameters explicitly:

```{code-cell} ipython3
glm = nmo.glm.GLM(
    regularizer="GroupLasso", 
    regularizer_strength=[0.1, 0.4],
)
glm.fit(X, spikes)
```
By default, [`GroupLasso`](nemos.regularizer.GroupLasso) generates masks that match the group stucture in the design matrix `X`.
If you pass your own mask, you need to make sure the regularizer strength matches its structure.

### PopulationGLM
A [`PopulationGLM`](nemos.glm.PopulationGLM) models many neurons simultaneously. 
Internally, this means it will have a set of parameters per neuron.
All regularization strategies above work for a [`PopulationGLM`](nemos.glm.PopulationGLM) as well:
```{code-cell} ipython3
# we'll create a second neuron with twice the amount of spikes
spikes = jnp.stack([spikes, spikes*2], axis=1)
glm = nmo.glm.PopulationGLM(
    regularizer="Ridge", 
    regularizer_strength=dict(
        f1=[[0.1, 0.2], [0.1, 0.2], [0.1, 0.2], [0.1, 0.2], [0.1, 0.2]],
        f2=[[0.1, 0.2], [0.1, 0.2]]
    )
)
glm.fit(X, spikes)
```
For every parameter of every feature group (5 for f1 and 2 for f1) we are now passing two regularization strengths, one for each neuron.
It is a bit tedious, but: this model is fitting two neurons at the same time and regularizing the parameters for each neuron differently!
