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

# Additional Available Initialization Functions in GLM-HMM

NeMoS provides additional initialization functions in `nemos.glm_hmm`. Call them to build the parameters and pass the result to `fit`.

Initial probabilities, `"initial_prob"`:

- `uniform_initial_proba_init`: each state has probability `1 / n_states` (default).
- `random_initial_proba_init`: uniform random draws, normalized to sum to one.
- `dirichlet_initial_proba_init`: a draw from a Dirichlet distribution with concentration `alphas`.

Transition probabilities, `"transition_prob"`:

- `sticky_transition_proba_init`: `prob_stay` on the diagonal, the rest split evenly across the other states (default, `prob_stay=0.95`).
- `uniform_transition_proba_init`: every transition has probability `1 / n_states`.
- `random_transition_proba_init`: uniform random draws, each row normalized to sum to one.
- `dirichlet_transition_proba_init`: each row drawn from a Dirichlet distribution with concentration `alphas`.

GLM parameters:

- `random_glm_params_init`: `"coef"` drawn from a normal distribution with standard deviation `std_dev`, `"intercept"` set to the link function of the mean of `y` (default, `std_dev=0.001`).
- `constant_parameter_init`: `"scale"` or `"dispersion"` filled with `fill_value` (default, `fill_value=1.0`).

All parameters at once:

- `KMeansInitializerGLM`: assigns states with k-means and fits one GLM per state.

The two examples below show the random probabilities and the k-means initialization.

```{code-cell} ipython3
import numpy as np
import nemos as nmo

rng = np.random.default_rng()
X = rng.standard_normal((15, 2))
y = rng.choice([0., 1.], size=15)

model = nmo.glm_hmm.GLMHMM(3)
```

## Random Initial and Transition Probabilities

`random_initial_proba_init` and `random_transition_proba_init` draw the probabilities at random. Use `dirichlet_initial_proba_init` and `dirichlet_transition_proba_init` to draw them from a Dirichlet distribution instead.

```{code-cell} ipython3
params = model.initialize_params(X, y)

# draw initial and transition probabilities at random
key_init, key_trans = jax.random.split(jax.random.PRNGKey(0))
params["initial_prob"] = nmo.glm_hmm.random_initial_proba_init(3, random_key=key_init)
params["transition_prob"] = nmo.glm_hmm.random_transition_proba_init(
    3, random_key=key_trans
)
model.fit(X, y, init_params=params)
```

## K-means Initialization

`KMeansInitializerGLM` clusters `X` and `y` with k-means, assigns each sample to a state, and fits one GLM per state. It provides the coefficients, the intercepts, and the initial and transition probabilities.

```{code-cell} ipython3
kmeans = nmo.glm_hmm.KMeansInitializerGLM(
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
