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

# Approximate leave-one-out cross-validation

```{code-cell} ipython3
:tags: [hide-input]

%matplotlib inline
import warnings

warnings.filterwarnings(
    "ignore",
    message="plotting functions contained within `_documentation_utils` are intended for nemos's documentation.",
    category=UserWarning,
)
```

Leave-one-out cross-validation (LOO-CV) is a natural way to estimate the
out-of-sample performance of a GLM, but computing it exactly requires refitting
the model $n$ times, once per held-out observation. NeMoS provides an
**approximate** LOO-CV that recovers the same per-observation held-out
predictions from a *single* full-data fit plus a cheap $O(p^2)$ correction per
observation (where $p$ is the number of features).

The approximation is the *infinitesimal jackknife* / *one-step Newton* estimator
(Pregibon, 1981; Rad & Maleki, 2020). At the fitted solution $\hat\beta$, with
Fisher working weights $w_i = g'(\eta_i)^2 / V(\mu_i)$ (for the canonical-link
Poisson model $w_i = \mu_i$), curvature $A = X^\top W X + \text{penalty}$, and
hat-matrix diagonal $h_{ii}$, the held-out linear predictor is approximated as

$$
\eta_i^{(-i)} \approx \eta_i + \frac{s_i\, x_i^\top A^{-1} x_i}{1 - h_{ii}},
\qquad \mu_i^{(-i)} = g^{-1}\!\left(\eta_i^{(-i)}\right),
$$

where $s_i = \partial_{\eta_i}[-\log p(y_i\mid\mu_i)]$ is the score contribution
of observation $i$ (for Poisson, $s_i = \mu_i - y_i$).

## Fitting a model and running approximate LOO

We simulate a small Poisson GLM dataset and fit a {class}`~nemos.glm.GLM`.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np

import nemos as nmo

np.random.seed(0)
n_samples, n_features = 200, 4
X = np.random.normal(size=(n_samples, n_features))
true_coef = np.array([0.5, -0.3, 0.25, 0.0])
y = np.random.poisson(np.exp(X @ true_coef - 0.5)).astype(float)

model = nmo.glm.GLM().fit(X, y)
```

Calling {meth}`~nemos.glm.GLM.approximate_loo` returns a named tuple with the
per-observation LOO predicted mean, linear predictor, log-likelihood, deviance,
and leverage.

```{code-cell} ipython3
loo = model.approximate_loo(X, y)

print("fields:", loo._fields)
print("approx. LOO mean log-likelihood:", float(loo.log_likelihood.mean()))
print("approx. LOO mean deviance:      ", float(loo.deviance.mean()))
```

The same result is available as a standalone function,
{func}`nemos.model_selection.approximate_loo`, which also works for
{class}`~nemos.glm.PopulationGLM` (returning arrays with a trailing neuron axis).

## How close is the approximation?

To check the approximation we compute *exact* LOO by refitting the model on each
leave-one-out subset. Because NeMoS GLMs are scikit-learn-compatible estimators,
we can get this directly from {class}`~sklearn.model_selection.LeaveOneOut` and
{func}`~sklearn.model_selection.cross_val_predict`, which refit a fresh model on
each training fold and predict the held-out sample. We use a smaller dataset so
the $n$ refits are quick.

```{code-cell} ipython3
from sklearn.model_selection import LeaveOneOut, cross_val_predict

n_small = 60
Xs, ys = X[:n_small], y[:n_small]

approx = nmo.glm.GLM().fit(Xs, ys).approximate_loo(Xs, ys)

# exact LOO: sklearn's cross_val_predict still performs n refits under the hood,
# which is substantially slower.
exact_mean = cross_val_predict(nmo.glm.GLM(), Xs, ys, cv=LeaveOneOut())

approx_mean = np.asarray(approx.predicted_mean)
print("max abs. difference:", float(np.abs(approx_mean - exact_mean).max()))
```

```{code-cell} ipython3
:tags: [hide-input]

fig, axs = plt.subplots(1, 2, figsize=(9, 4))

axs[0].scatter(exact_mean, approx_mean, s=18, alpha=0.7)
lims = [
    min(exact_mean.min(), approx_mean.min()),
    max(exact_mean.max(), approx_mean.max()),
]
axs[0].plot(lims, lims, "k--", lw=1, label="identity")
axs[0].set(
    xlabel="exact LOO predicted mean",
    ylabel="approximate LOO predicted mean",
    title="Approximate vs. exact LOO",
)
axs[0].legend()

axs[1].scatter(np.asarray(approx.leverage), np.abs(approx_mean - exact_mean), s=18, alpha=0.7)
axs[1].set(
    xlabel=r"leverage $h_{ii}$",
    ylabel="|approximate - exact|",
    title="Error grows with leverage",
)
fig.tight_layout()
```

The approximate LOO predictions closely track the exact refit-based values. The
right panel shows the characteristic behavior of the infinitesimal jackknife:
the approximation is a first-order expansion, so its error grows for
**high-leverage** points ($h_{ii} \to 1$). The returned `leverage` array lets you
flag those observations.

## Limitations

- **High leverage.** As shown above, the accuracy can degrade for high-leverage
  points. For a definitive estimate at those points, fall back to an exact refit.
- **Regularization.** Ridge penalties are supported (folded into the curvature
  $A$). Non-smooth penalties — {class}`~nemos.regularizer.Lasso`,
  {class}`~nemos.regularizer.ElasticNet`, {class}`~nemos.regularizer.GroupLasso` —
  raise a `NotImplementedError`, because the infinitesimal-jackknife formula
  assumes a twice-differentiable objective (Rad & Maleki, 2020).
- **Observation models.** Supported where a variance function is defined
  (Poisson, Gamma, Gaussian, Bernoulli).

## References

- Pregibon, D. (1981). Logistic regression diagnostics. *The Annals of
  Statistics*, 9(4), 705-724.
- Rad, K. R., & Maleki, A. (2020). A scalable estimate of the out-of-sample
  prediction error via approximate leave-one-out cross-validation. *Journal of
  the Royal Statistical Society: Series B*, 82(4), 965-996.
