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

(handle-silent-neurons)=
# Handle Silent Neurons During Initialization

A neuron that never fires in your recording (a *silent* neuron) is an edge case for GLM initialization. This guide explains why it is a problem, how NeMoS handles it for you, and what to do if you use the `Newton` solver.

```{code-cell} ipython3
import warnings

import numpy as np

import nemos as nmo
```

## Why silent neurons are a problem

By default, [`GLM.initialize_params`](nemos.glm.GLM.initialize_params) starts the fit from parameters that match the mean of the observations. It sets the coefficients to zero and chooses the intercept so that the rate predicted by the model equals the mean rate of each neuron.

A GLM maps the linear predictor to a rate through its **inverse link function** $g$ (`model.inverse_link_function`). To find the intercept that reproduces a given mean rate, initialization therefore applies the *inverse of that function* to the mean rate $\bar{y}$:

$$
b_0 = g^{-1}(\bar{y})
$$

For the default Poisson GLM the inverse link function is the exponential, $g = \exp$, so $g^{-1} = \log$ and the initial intercept is $\log(\bar{y})$. A silent neuron has $\bar{y} = 0$, and $\log(0) = -\infty$. The initial intercept would not be finite, and the fit could not even start.

```{code-cell} ipython3
with np.errstate(divide="ignore"):
    print(np.log(0.0))
```

## How initialization handles it

Each observation model declares the range its mean can take in the `rate_range` class attribute (see the [developer notes](../developers_notes/05-observation_models.md)). For the Poisson model this is $(0, \infty)$.

When the mean activity of an output sits exactly on the boundary of that range, as it does for a silent neuron, the initialization shifts it slightly into the interior before applying the inverse link, and emits a `UserWarning`. For a range of $(0, \infty)$ and $n$ samples the shifted mean is

$$
\bar{y} \;\rightarrow\; \bar{y} + \frac{1}{2n}.
$$

The intercept is then finite. Only the starting point changes: the data and the loss function are untouched.

## `initialize_params` in practice

Let's initialize a model on a neuron that never fires.

```{code-cell} ipython3
n_samples = 100
X = np.random.randn(n_samples, 2)
y_silent = np.zeros(n_samples)

model = nmo.glm.GLM()

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    coef, intercept = model.initialize_params(X, y_silent)

print(f"warning: {caught[0].message}")
print(f"initial coefficients: {coef}")
print(f"initial intercept: {intercept}")
```

The intercept is finite and close to $\log\left(\frac{1}{2n}\right)$, i.e. the log of a very small firing rate:

```{code-cell} ipython3
print(np.log(1 / (2 * n_samples)))
```

The same applies to a [`PopulationGLM`](nemos.glm.PopulationGLM), where each neuron is handled independently. Only the silent columns are shifted.

```{code-cell} ipython3
y_pop = np.random.poisson(1.0, size=(n_samples, 3)).astype(float)
y_pop[:, 1] = 0  # neuron 1 is silent

pop_model = nmo.glm.PopulationGLM()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _, pop_intercept = pop_model.initialize_params(X, y_pop)

print(pop_intercept)
```

Other observation models are treated according to their own `rate_range`. For example, a Bernoulli model whose output is always 1 has a mean on the upper boundary of $(0, 1)$, and the mean is shifted back inside the interval, so the logit is finite.

:::{note}
If you would rather choose the starting point yourself, pass your own parameters to `fit` with the `init_params` argument; initialization is then skipped.
:::

## Fitting with the Newton solver

Initialization is only the first step. A silent Poisson neuron has no finite maximum-likelihood intercept: the loss and its gradient approach zero as the intercept decreases, and the exact optimum is only approached at an infinite intercept. Most solvers still converge, because the gradient becomes small enough to meet the stopping tolerance.

The `Newton` solver is more fragile. At every step it solves a linear system with the Hessian, using a Cholesky factorization by default. When the intercept is very negative, the rates of the silent neurons underflow to zero and the Hessian becomes numerically singular, so the factorization fails. A ridge penalty does not guarantee that this is avoided.

:::{warning}
Newton's Cholesky solve can fail with a singular Hessian when the rates of silent neurons underflow to zero.
:::

The example below makes this happen with a population in which neuron 0 fires a few times and neurons 1 and 2 are silent. We start from very negative, user-supplied intercepts, so the rates of the silent neurons are zero from the first step, and we use a `Ridge` penalty to show that the regularizer alone does not fix it.

```{code-cell} ipython3
:tags: [raises-exception]

X_silent = np.random.default_rng(0).normal(size=(40, 2))
y_silent = np.zeros((40, 3))
y_silent[[1, 5, 7], 0] = 1
init_silent = (np.zeros((2, 3)), np.array([-4.0, -1000.0, -1000.0]))

silent_model = nmo.glm.PopulationGLM(
    solver_name="Newton",
    regularizer="Ridge",
    regularizer_strength=0.1,
    solver_kwargs={"linear_solver": "cholesky"},
)
silent_model.fit(X_silent, y_silent, init_params=init_silent)
```

The linear solve used by Newton is configurable through `solver_kwargs`, so you can replace the Cholesky factorization with a more robust one. Select `identity_shift`, which adds a small multiple of the identity to the Hessian, to stabilize the solve. `eigh`, an eigendecomposition, is another available option.

```{code-cell} ipython3
silent_model.solver_kwargs = {"linear_solver": "identity_shift"}
silent_model.fit(X_silent, y_silent, init_params=init_silent)

print(silent_model.intercept_)
```

If a silent neuron is not informative for your analysis, you can also exclude it from the population before fitting.