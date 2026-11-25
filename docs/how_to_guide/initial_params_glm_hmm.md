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
import jax

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
jax.config.update("jax_enable_x64", True)
```

# Controlling Parameter Initialization in GLM-HMM

## Generating Different Initial Parameter Samples
The GLM-HMM objective is non-convex. This means that multiple local-optimum solutions may exist and which of those solutions the fit converges to may depend on the initial parameters chosen.

NeMoS provides the following default initialization methods:

- `"initial_prob"`: uniform, each state has probability `1 / n_states`. Deterministic.
- `"transition_prob"`: sticky, 0.95 on the diagonal and the remaining 0.05 split evenly across the other states. Deterministic.
- `"coef"`: drawn from a normal distribution with mean 0 and standard deviation 0.001. Random.
- `"intercept"`: the link function applied to the mean of `y`, the same for every state. Since the coefficients start close to zero, every state initially predicts a mean close to the empirical mean of `y`. Deterministic.
- `"scale"` (Gamma, Gaussian) and `"dispersion"` (NegativeBinomial): set to 1.0. Deterministic.

All the default initialization functions are deterministic, with the exception of the one for the coefficients.
By changing the `seed` parameter of the model, you can generate different coefficient samples.

```{code-cell} ipython3
import numpy as np
import nemos as nmo
import jax

rng = np.random.default_rng()
X = rng.standard_normal((15,2))
y = rng.choice([0., 1.], size=15)

# initialize a model & compute the initial parameters
model = nmo.glm_hmm.GLMHMM(3, seed=jax.random.PRNGKey(0))
param = model.initialize_params(X, y)
print("coefficients for seed 0:\n", param["coef"])

# Change seed and recompute parameters
model.seed = jax.random.PRNGKey(1)
param = model.initialize_params(X, y)

# Compare the coefficients
print("coefficients for seed 1:\n", param["coef"])
```

## Specifying the Initial Parameters

As for the GLMs, the GLM-HMM fit method exposes an `init_params` argument. The expected parameter format is a dictionary with the following key and value pairs:

- `"initial_prob"`: a 1-D array with the initial probabilities of shape `(n_states,)` that must sum to one.
- `"transition_prob"`: a 2-D array with the transition probabilities of shape `(n_states, n_states)` that must sum to one over the rows.
- `"coef"`: a 2-D array with the coefficients of shape `(n_features, n_states)`.
- `"intercept"`: a 1-D array with the intercepts of shape `(n_states,)`.

Gamma and Gaussian observation models require an additional `"scale"` entry, and the NegativeBinomial a `"dispersion"` both of which are arrays of dimension `(n_states,)` with strictly positive values.

```{code-cell} ipython3

custom_params = param.copy()

# example: change the initial probabilities
custom_params["initial_prob"] = np.array([0.2, 0.5, 0.3])
model.fit(X, y, init_params=custom_params)

# switch to negative binomial & define custom dispersion
model.observation_model = "NegativeBinomial"
custom_params = model.initialize_params(X, y)
print("Default dispersion: ", custom_params["dispersion"])

custom_params["dispersion"] = np.full(3, 2.)
model.fit(X, y, init_params=custom_params)
```

## Cross-validating with user-provided parameters

If you are cross-validating or creating pipelines with `scikit-learn`, you can still provide parameters for model fitting.

`init_params` reaches the model `fit` through scikit-learn metadata routing. `sklearn.set_config(enable_metadata_routing=True)` switches routing on globally, and it stays on for the rest of the Python session; `model.set_fit_request(init_params=True)` tells the cross-validation object to forward `init_params` to `model.fit`.

```{code-cell} ipython3
from sklearn.model_selection import GridSearchCV
import sklearn

# configure the parameter routing
sklearn.set_config(enable_metadata_routing=True)
model.set_fit_request(init_params=True)

# set a ridge regularizer and cross-validate strength
model.regularizer="Ridge"
cls = GridSearchCV(model, param_grid={"regularizer_strength": [1., 10.]})
cls.fit(X, y, init_params=custom_params)
```

## Using the Available Initialization Functions

NeMoS provides additional initialization functions in `nemos.hmm.initialize_parameters` and `nemos.glm_hmm.initialize_parameters`. Call them to build the parameters and pass the result to `fit`.

### Random Initial and Transition Probabilities

`random_initial_proba_init` and `random_transition_proba_init` draw the probabilities at random. Use `dirichlet_initial_proba_init` and `dirichlet_transition_proba_init` to draw them from a Dirichlet distribution instead.

```{code-cell} ipython3
from nemos.hmm.initialize_parameters import (
    random_initial_proba_init,
    random_transition_proba_init,
)

model = nmo.glm_hmm.GLMHMM(3)
params = model.initialize_params(X, y)

# draw initial and transition probabilities at random
key_init, key_trans = jax.random.split(jax.random.PRNGKey(0))
params["initial_prob"] = random_initial_proba_init(3, random_key=key_init)
params["transition_prob"] = random_transition_proba_init(3, random_key=key_trans)
model.fit(X, y, init_params=params)
```

### K-means Initialization

`KMeansInitializerGLM` clusters `X` and `y` with k-means, assigns each sample to a state, and fits one GLM per state. It provides the coefficients, the intercepts, and the initial and transition probabilities.

```{code-cell} ipython3
from nemos.glm_hmm.initialize_parameters import KMeansInitializerGLM

kmeans = KMeansInitializerGLM(
    3, X, y, model.inverse_link_function, model.observation_model
)
coef, intercept = kmeans.glm_params()
params = {
    "coef": coef,
    "intercept": intercept,
    "initial_prob": kmeans.initial_probability(),
    "transition_prob": kmeans.transition_probability(),
}
model.fit(X, y, init_params=params)
```
