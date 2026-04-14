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
# Solver Selection: Performance and Device Considerations

This note compares solver and device configurations to help identify which setup is most
efficient for a given problem.

In particular, we will focus on how to choose a configuration along the axis:

- **Device:** cpu vs cuda.
- **Algorithm:** first (`GradientDescent`) vs second order (`LBFGS`, `BFGS`) vs stochastic (`SVRG`) optimization methods.
- **Compilation:** fully compiled (`optax` or `optimistix` solvers) vs a wrapper of `L-BFGS-B` from `scipy`.


:::{admonition} Compilation
:info:

JAX jit-compiles the full optimization just-in-time at each model re-instantiation and change in the input structure (shape, data type etc.). The `L-BFGS-B` from `scipy.minimize` is ahead-of-time compiled, meaning that the compile binary is readily available, saving the compilation costs entirely. This may be advantageous when fitting multiple small models.
:::

## Short Answer

In this section we will provide a quick summary of the take home messages of our benchmarking analysis.  We divided our recommendation into smooth vs non-smooth problems, since the algorithms requirements are different for these two problem types.

### Smooth Optimization (Unregularized or Ridge GLM)

For smooth optimizations, NeMoS provides several second order (`BFGS`, `LBFGS`,...) and first order methods (`GradientDescent`). We generally recommend `LBFGS` since it is performant and, in terms of memory footprint, scales well with problem size.

- For problem medium to large size problems (>10k samples, >10 parameters), use or default `LBFGS` which is fully JIT-compiled, and a GPU if available and if the problem fit in memory.
- For small problems <1k samples, tens of model parameters, use the scipy wrapper provided below; It executes a `LBFGS` solve without JIT-compilation costs. Likely faster with a JAX CPU backend (if you have a JAX cuda, set `jax.config.update("jax_platform_name", "cpu")` at the beginning of your script). The GPU is likely detrimental, and the JIT compilation may dominate the computational cost.
- For problems that don't fit in memory, we recommend trying `SVRG`. Call repeatedly `model.update` passing data batches. This implements a form of stochastic optimization with convergence guarantees under certain assumptions, see [1]_ for details. For a simpler alternative one can use stochastic gradient descent (SGD), see our note [batching](batching) that explain how to set SGD up.

### Non-Smooth Optimization (Lasso, GroupLasso or Elastic Net)

Nemos provide two first order proximal gradient methods, `ProxGradientDescent` and `ProxSVRG`.

- For problems that fit in memory, we recommend `ProximalGradientDescent`.
  - For small problems, <1k samples, tens of model parameters, configure JAX for CPU.
  - For larger problems, >10k samples, rely on the GPU if available.
- For very large problems, we recommend `ProxSVRG`.


## Interpreting the benchmarking results

To interpret the benchmarking results, it is helpful to decompose the `model.fit()` time into different contributions. An approximate but useful decomposition is the following:

```
total_time ≈ t_compile + n_iter × t_iter
```

where,

- **`t_compile`** — NeMoS JAX-based solvers compiles at every `fit()` call.
This cost is fixed for a given problem size (the size of `X` and `y`) and depends on the device as well as the algorithm chosen. The `scipy` based wrapper has no such cost.

- **`t_iter`** — Time per solver step. For a fixed problem shape and device, `t_iter` is roughly
constant. It scales with problem size and depends strongly on both the algorithm and the device.

**`n_iter`** — Iterations to convergence. This is the least predictable term: it depends on
both problem difficulty and the algorithm chosen.



-  `t_compile` is spent at each `fit` call: the more fit calls (as in a grid-search) the more time will be spent compiling. For small problems, the compilation time may be the highest cost, and a `scipy` adapter may be preferred.


## Benchmarking Smooth Optimization

For smooth penalties (Ridge, UnRegularized, and others), multiple solvers are available.
The benchmarks below compare `LBFGS[optax+optimistix]` and `GradientDescent[optimistix]`,
the two most commonly used, together with a scipy L-BFGS-B reference
(see [Scipy L-BFGS-B reference adapter](#scipy-lbfgs-b-reference-adapter)).

Set `CSV_PATH` to the CSV produced by `scripts/benchmarking/benchmarking_glm.py`:

```{code-cell} ipython3
from pathlib import Path

CSV_PATH = Path("../../benchmarking_results/20260413_182029_benchmarking_ef6bffe.csv")
```

```{code-cell} ipython3
import pandas as pd

df = pd.read_csv(CSV_PATH)
df = df[df["rep"] != 0]   # rep 0 is a warmup run

synth = df[df["data_source"] == "synthetic"].copy()
real  = df[df["data_source"] != "synthetic"].copy()

synth["t_iter"] = synth["fit_s"] / synth["iter_num"]
real["t_iter"]  = real["fit_s"]  / real["iter_num"]
```

### Compilation overhead

```{code-cell} ipython3
compile_tbl = (
    synth.groupby(["solver_name", "device"])["compilation_s"]
    .median()
    .unstack("device")
    .rename(columns={"cpu": "CPU (s)", "gpu": "GPU (s)"})
)
compile_tbl.index.name = "Solver"
compile_tbl
```

### Per-iteration time

```{code-cell} ipython3
t_iter_tbl = (
    synth.groupby(["solver_name", "device", "sample_size"])["t_iter"]
    .median()
    .mul(1e3)   # convert to milliseconds
    .unstack("sample_size")
)
t_iter_tbl.index.names = ["Solver", "Device"]
t_iter_tbl.columns.name = "n"
t_iter_tbl.round(3)
```

JAX solver iteration time on GPU is nearly flat from n = 100 to n = 100 k, while CPU time
grows roughly linearly with *n*. The scipy adapter benefits from GPU at large *n* (~2.6×
at n = 100 k) because gradients and likelihoods are evaluated in JAX on-device, but the C
optimizer loop limits the gain compared to native JAX solvers; at small *n* the dispatch
overhead makes scipy slower on GPU than on CPU.

GPU speedup over CPU at n = 100 k: `GradientDescent` ~46×, `LBFGS[optax+optimistix]` ~9×,
scipy L-BFGS-B ~2.6×. The large per-iteration speedup of `GradientDescent` on GPU does not
translate into end-to-end gains because of the higher iteration count required; see below.

### Number of iterations

**Synthetic data**

```{code-cell} ipython3
iter_synth = (
    synth.groupby("solver_name")["iter_num"]
    .median()
    .rename("Median iterations (synthetic)")
    .to_frame()
)
iter_synth.index.name = "Solver"
iter_synth
```

**Real data**

```{code-cell} ipython3
iter_real = (
    real.groupby(["solver_name", "device"])[["iter_num", "end_to_end_s"]]
    .median()
    .rename(columns={"iter_num": "Iterations", "end_to_end_s": "End-to-end (s)"})
)
iter_real.index.names = ["Solver", "Device"]
iter_real.round(1)
```

On the real-data benchmark `GradientDescent` hit the iteration cap on both devices without
converging. LBFGS required hundreds of iterations — far more than on synthetic data — but
its low `t_iter` kept total runtime manageable, particularly on GPU.

:::{note}
Always check `model.solver_state_.converged` and `model.solver_state_.num_steps` after
fitting. `GradientDescent` convergence can be improved by increasing `max_iter` and tuning
the learning rate via `solver_kwargs`, but LBFGS is the safer choice for real neural
recordings.
:::



## Benchmarking Non-smooth optimization

*Coming soon.*



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

[1] [Gower, Robert M., Mark Schmidt, Francis Bach, and Peter Richtárik. "Variance-Reduced Methods for Machine Learning." arXiv preprint arXiv:2010.00892 (2020).](https://arxiv.org/abs/2010.00892)
