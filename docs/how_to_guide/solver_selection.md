---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(solver-selection)=
# Benchmarking GLM Configurations

In this note we will compares solvers performance in GLM problems on simulated data and neural recordings (from [1]_), for all combinations of:

- **devices:** cpu vs cuda.
- **solvwera:**  NeMoS JIT compiled solvers and a scipy wrapper of `L-BFGS-B`, see table below.
- **problem sizes:** (simulation only) by varying number of samples, features and neurons.



:::{admonition} JIT vs `scipy.minimize`
:class: note

NeMoS native solvers JIT-compile the full optimization when `GLM.fit` is called, incurring a one-time compilation cost before the first iteration. `L-BFGS-B` from `scipy.minimize` calls pre-compiled Fortran routines but invokes Python at each iteration to evaluate the GLM likelihood and gradient. As a result, most JIT-compiled solvers run faster per-iteration once compiled, while `scipy` avoids the compilation cost entirely. If the optimization loop dominates over compilation (large problems, many iterations), prefer the JIT-compiled solvers; for small problems, the scipy wrapper provided below is likely faster.
:::

## Solvers

| Solver | Backend | Notes |
|--------|---------|-------|
| [`GradientDescent`](https://en.wikipedia.org/wiki/Gradient_descent) | [optimistix](https://docs.kidger.site/optimistix/) | First-order (gradient only). Memory scales linearly with parameter count. Smooth penalties only; typically requires many iterations. |
| [`BFGS`](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm) | [optimistix](https://docs.kidger.site/optimistix/) | Quasi-Newton; maintains a dense Hessian approximation. Memory scales quadratically with parameter count — impractical for large parameter spaces. Fewer iterations than first-order methods. |
| [`LBFGS`](https://en.wikipedia.org/wiki/Limited-memory_BFGS) | [optax](https://optax.readthedocs.io/) + [optimistix](https://docs.kidger.site/optimistix/) | Limited-memory quasi-second-order; stores the last *m* gradient vectors. Memory scales linearly. Fewer iterations than first-order methods. Recommended default for smooth problems. |
| [`LBFGS`](https://en.wikipedia.org/wiki/Limited-memory_BFGS) | scipy | L-BFGS-B via [`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html). No JIT compilation cost; Python callback per iteration. Preferred for small problems or many repeated `fit()` calls. |
| [`ProximalGradient`](https://en.wikipedia.org/wiki/Proximal_gradient_method) | [optimistix](https://docs.kidger.site/optimistix/) | First-order + proximal step for non-smooth penalties (Lasso, GroupLasso, ElasticNet). Memory scales linearly. Typically requires many iterations. |
| [`SVRG`](https://en.wikipedia.org/wiki/Stochastic_variance_reduction_gradient) | nemos | Inner loop of fast mini-batch gradient steps followed by a full-gradient anchor step each epoch. Memory scales linearly. The nested inner-outer loop structure can be slow on GPU. |
| [`ProxSVRG`](https://en.wikipedia.org/wiki/Stochastic_variance_reduction_gradient) | nemos | Proximal variant of SVRG for non-smooth penalties. Same inner-outer loop structure and GPU caveats as `SVRG`. |

## Scipy L-BFGS-B reference adapter

The scipy comparison above uses a custom adapter wrapping `scipy.optimize.minimize` with
method `"L-BFGS-B"`, following the same pattern as the [Powell adapter](custom_solvers.md)
in the custom solvers guide. Unlike Powell, L-BFGS-B requires gradients; here they are
computed via `jax.grad` so gradient and likelihood evaluations run in JAX while the
optimizer loop runs in scipy. The adapter is not shipped with NeMoS.

```{code-cell} ipython3
:tags: [hide-input]

import jax
import scipy.optimize
from typing import NamedTuple

from nemos.utils import get_flattener_unflattener


class ScipySolverState(NamedTuple):
    iter_num: int
    res: dict
    converged: bool


class ScipySolver:
    """General adapter for scipy.optimize.minimize as a NeMoS solver."""

    def __init__(
        self,
        unregularized_loss,
        regularizer,
        regularizer_strength,
        has_aux,
        init_params,
        method: str,
        maxiter: int = 1000,
        tol: float = 1e-8,
    ) -> None:
        assert not has_aux, "Auxiliary output not supported."
        self.fun = regularizer.penalized_loss(
            unregularized_loss, init_params, regularizer_strength
        )
        self.grad = jax.grad(self.fun)
        self.method = method
        self.tol = tol
        self.maxiter = maxiter

    @classmethod
    def get_accepted_arguments(cls):
        return {"method", "maxiter", "tol"}

    def init_state(self, init_params, *args):
        return ScipySolverState(0, {}, False)

    def run(self, init_params, *args):
        params, res = self._run_for_n_steps(init_params, self.maxiter, *args)
        return params, ScipySolverState(res.nit, {**res}, res.success), None

    def update(self, params, state, *args):
        params, res = self._run_for_n_steps(params, 1, *args)
        return params, ScipySolverState(state.iter_num + res.nit, {**res}, False), None

    def _run_for_n_steps(self, params, n_steps, *args):
        flatten, unflatten = get_flattener_unflattener(params)
        x0 = flatten(params)

        def flat_obj(x, *args):
            return self.fun(unflatten(x), *args)

        def flat_grad(x, *args):
            return flatten(self.grad(unflatten(x), *args))

        res = scipy.optimize.minimize(
            fun=flat_obj,
            jac=flat_grad,
            x0=x0,
            args=args,
            method=self.method,
            options={"maxiter": n_steps},
            tol=self.tol,
        )
        return unflatten(res.x), res


class ScipyLBFGS(ScipySolver):
    """Scipy L-BFGS-B solver adapter for NeMoS."""

    def __init__(
        self,
        unregularized_loss,
        regularizer,
        regularizer_strength,
        has_aux,
        init_params,
        maxiter: int = 1000,
        tol: float = 1e-8,
    ):
        super().__init__(
            unregularized_loss, regularizer, regularizer_strength,
            has_aux, init_params, "L-BFGS-B", maxiter, tol,
        )
```

To use the scipy adapter defined in the code cell above, register it and declare compatibility with the desired regularizer:

```{code-cell} ipython3
import nemos as nmo
import numpy as np

# Add the solver to the registry and allow for Ridge
nmo.solvers.register("LBFGS", ScipyLBFGS, backend="scipy")
nmo.regularizer.Ridge.allow_solver("LBFGS")

# Generate data
rng = np.random.default_rng(0)
n_samples, n_features = 5_000, 50

X = rng.standard_normal((n_samples, n_features))
true_coef = rng.standard_normal(n_features) * 0.5
y = rng.poisson(np.exp(X @ true_coef)).astype(float)

# Fit using the scipy wrapper
model = nmo.glm.GLM(regularizer="Ridge", solver_name="LBFGS[scipy]")
model.fit(X, y)
```


## References

[1] [Peyrache A, Lacroix MM, Petersen PC, Buzsáki G. Internally organized mechanisms of the head direction sense. Nat Neurosci. 2015 Apr;18(4):569-75. doi: 10.1038/nn.3968. Epub 2015 Mar 2. PMID: 25730672; PMCID: PMC4376557.](https://pubmed.ncbi.nlm.nih.gov/25730672/)

https://doi.org/10.7554/eLife.85786.2
[2] [Gower, Robert M., Mark Schmidt, Francis Bach, and Peter Richtárik. "Variance-Reduced Methods for Machine Learning." arXiv preprint arXiv:2010.00892 (2020).](https://arxiv.org/abs/2010.00892)
