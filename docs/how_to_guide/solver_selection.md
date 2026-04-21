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
```{code-cell}
:tags: [hide-input]

import pynapple as nap
import warnings

nap.nap_config.suppress_conversion_warnings = True

warnings.filterwarnings(
    "ignore",
    message="DataFrame is not sorted by start",
    category=UserWarning,
)

```


(solver-selection)=
# Benchmarking GLM Configurations

```{contents}
:depth: 2
:local:
```

In this note we will compare solvers performance in GLM problems on simulated data and neural recordings (from [[1]](#ref-1)), for all combinations of:

- **devices:** cpu vs cuda.
- **solvers:**  NeMoS JIT compiled solvers and a scipy wrapper of `L-BFGS-B`, see the [table below](table_solvers) for a complete list.
- **problem sizes:** (simulation only) by varying number of samples, features and neurons.

Additionally, we compared against scikit-learn's `PoissonRegressor` with the `"newton-cholevsky"` solver, which is the most efficient for the configuration tested.

:::{admonition} JIT vs `scipy.minimize`
:class: note

NeMoS native solvers JIT-compile the full optimization when `GLM.fit` is called, incurring a one-time compilation cost before the first iteration. `L-BFGS-B` from `scipy.minimize` calls pre-compiled Fortran routines but invokes Python at each iteration to evaluate the GLM likelihood and gradient. As a result, most JIT-compiled solvers run faster per-iteration once compiled, while `scipy` avoids the compilation cost entirely. If the optimization loop dominates over compilation (large problems, many iterations), prefer the JIT-compiled solvers; for small problems, the scipy wrapper provided below is likely faster.
:::


## Results

We present the results in two section: simulations and real data. The difference in how algorithm performs under the two conditions is striking, but the interpretation is pretty obvious:

- **Simulations**: fitting simulated data requires a smaller number of optimization steps, and the compilation cost may dominate over the optimization loop, especially true for smaller dataset; For this reason the `scipy.minimize` wrapper is the most efficient algorithm for small problem sizes, while `BFGS` is comes up on top for larger problems.

- **Neural Recordings**: fitting neural recordings often requires hundreds or even thousands of optimization steps, depending on the algorithm. The JIT compiled solvers tends to be more efficient since the compilation cost is negligible. In our experience, this is commonly the case in real applications.

- **GPU vs CPU**: GPU compilation is slower than its CPU counterpart, however, the optimization updates scale very well with the problem size. For this reason, the GPU optimization is the most performant option for large  problems.


Before digging into the data, some let's set up some `pandas` configurations and helper functions.

```{code-cell}

import pandas as pd
from pathlib import Path

pd.set_option("display.max_rows", 25)
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", "{:.3f}".format)

def load(csv_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (synthetic_df, recordings_df) split on data_source, warmup rep dropped."""
    df = pd.read_csv(csv_path)
    df = df[df["rep"] != 0]  # rep 0 is a warmup run — discard before any stats
    df["compilation_s"] = df["compilation_s"].fillna(0.0)
    synth = df[df["data_source"] == "synthetic"].copy()
    recordings = df[df["data_source"] != "synthetic"].copy()
    return synth, recordings


def filter_and_compute_averages(df: pd.DataFrame, query: str | None=None):
    """Prepare benchmarking result summary.

    This function filter the result dataframe with a query, and average the
    end-to-end fit time over fit repetitions, and returns a styled dataframe.
    """
    # filter data
    if query is not None:
        df = simulations.query(query)
    df = df.copy()

    # compute the fraction of the end-to-end fit time spent on compilation
    df.loc[:, "compile_time_fraction"]  = df["compilation_s"] / (df["solver_init_s"] + df["compilation_s"] + df["fit_s"])

    # compute the mean fit time per repetition of the fit, as well as the mean
    # number of iterations (constant over repetition since it is algorithm dependent)
    result = (
        df.groupby(["device", "solver_name"])
        .agg(
            fit_time_s=("end_to_end_s", "mean"),
            converged=("converged", "all"),
            iter_num=("iter_num", "mean"),
            compile_time_fraction=("compile_time_fraction", "mean"),
        )
        .sort_values("fit_time_s")
        .reset_index()
    )
    return (
      result.style
      .hide(axis="index")
      .set_table_styles([
          {"selector": "th", "props": [("text-align", "center")]},
          {"selector": "td", "props": [("text-align", "center")]},
          {"selector": "tr:nth-child(even)", "props": [("background-color", "#f2f2f2")]},
          {"selector": "tr:nth-child(odd)", "props": [("background-color", "white")]},
      ])
      .format("{:.3f}", subset=["fit_time_s", "compile_time_fraction"])
      .format("{:.0f}", subset=["iter_num"])
      )

```

### Simulations

When fitting simulated data, the compilation time represent a significant fraction of the total compute time. That's because the number of iteration required by the numerical solvers to converge to the optimal solution is relatively low. This is more evident in for the smallest problem size, as we can see by comparing the result of the smallest and the largest simulated problem.


#### Small problem size

- Sample size: 100
- Feature dimension: 1
- Number of neurons: 1


```{code-cell}

# TODO: replace with download from www folder
path = "/Users/ebalzani/Code/nemos/benchmarking_results/20260420_170138_benchmarking_921bdd8.csv"

# load the benchmarking results for the recordings and the simulations
simulations, recordings = load(path)

query = "sample_size == 1e2 and pop_size == 1  and feature_dim == 1"
filter_and_compute_averages(simulations, query)
```

#### Large problem size

- Sample size: $10^5$
- Feature dimension: 100
- Number of neurons: 20


```{code-cell}

query = "sample_size == 1e5 and pop_size == 20  and feature_dim == 100"
filter_and_compute_averages(simulations, query)
```

### Neural Recordings

- Sample size: 195820
- Feature dimension: 95
- Number of neurons: 19

```{code-cell}

filter_and_compute_averages(recordings)
```

## Comparison with `scikit-learn`

Finally, let's take a look on how NeMoS performs compared to `scikit-learn` `PoissonRegressor` on the same neural recording used for benchmarking. Note that this example runs on a completely different machine than the benchmarking, therefore the actual numbers won't be directly comparable to the table above.

```{code-cell}
from sklearn.linear_model import PoissonRegressor
import pynapple as nap
import numpy as np
import nemos as nmo
from time import perf_counter
import jax.numpy as jnp


def get_data():
    """Model design."""
    path = nmo.fetch.fetch_data("Mouse32-140822.nwb")
    data = nap.load_file(path)
    spikes = data["units"]
    epochs = data["epochs"]
    wake_ep = epochs[epochs.tags == "wake"]
    spikes = spikes.getby_category("location")["adn"]
    spikes = spikes.restrict(wake_ep).getby_threshold("rate", 1.)
    y = spikes.count(0.01, ep=wake_ep)
    X = nmo.basis.RaisedCosineLogConv(
        5, window_size=80
    ).compute_features(y)
    X, y = X.d, y.d
    keep = np.all(~np.isnan(X), axis=1)
    return X[keep], y[keep]


X, y = get_data()

skl_glm = PoissonRegressor(alpha=0.001, tol=1e-6, max_iter=1000)
nemos_glm = nmo.glm.GLM(
    regularizer="Ridge",
    regularizer_strength=0.001,
    solver_name="BFGS",
    solver_kwargs={"tol":1e-6, "maxiter": 1000},
)

t0 = perf_counter()
skl_glm.fit(X, y[:, 0])
sklearn_time = perf_counter() - t0

X, y = jnp.array(X), jnp.array(y)
t0 = perf_counter()
nemos_glm.fit(X, y[:, 0])
nemos_time = perf_counter() - t0

print(f"scikit-learn fit duration: {np.round(sklearn_time, 2)} sec")
print(f"NeMoS fit duration: {np.round(nemos_time, 2)} sec")
print(f"NeMoS is {np.round(sklearn_time/nemos_time, 2)}x faster then scikit-learn")
```



(table_solvers)=
## Benchmarked Solvers

| Solver | Backend | Notes                                                                                                                                                                                                                                             |
|--------|---------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`GradientDescent`](https://en.wikipedia.org/wiki/Gradient_descent) | [optimistix](https://docs.kidger.site/optimistix/) | First-order (gradient only). Memory scales linearly with parameter count. Smooth penalties only; typically requires many iterations.                                                                                                              |
| [`BFGS`](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm) | [optimistix](https://docs.kidger.site/optimistix/) | Quasi-Newton; maintains a dense Hessian approximation. Memory scales quadratically with parameter count — impractical for large parameter spaces. Fewer iterations than first-order methods.                                                      |
| [`LBFGS`](https://en.wikipedia.org/wiki/Limited-memory_BFGS) | [optax](https://optax.readthedocs.io/) + [optimistix](https://docs.kidger.site/optimistix/) | Limited-memory quasi-second-order; stores the last *m* gradient vectors. Memory scales linearly. Fewer iterations than first-order methods. Recommended default for smooth problems.                                                              |
| [`LBFGS`](https://en.wikipedia.org/wiki/Limited-memory_BFGS) | scipy | L-BFGS-B via [`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html). No JIT compilation cost; Python callback per iteration. Preferred for small problems or many repeated `fit()` calls. |
| [`ProximalGradient`](https://en.wikipedia.org/wiki/Proximal_gradient_method) | [optimistix](https://docs.kidger.site/optimistix/) | First-order + proximal step for non-smooth penalties (Lasso, GroupLasso, ElasticNet). Memory scales linearly. Typically requires many iterations.                                                                                                 |
| [`SVRG`](https://en.wikipedia.org/wiki/Stochastic_variance_reduction_gradient) | nemos | Inner loop of fast mini-batch gradient steps followed by a full-gradient anchor step each epoch. Memory scales linearly. The nested inner-outer loop structure can be slow on GPU. [[2]](#ref-2)                                                  |
| [`ProxSVRG`](https://en.wikipedia.org/wiki/Stochastic_variance_reduction_gradient) | nemos | Proximal variant of SVRG for non-smooth penalties. Same inner-outer loop structure and GPU caveats as `SVRG`.                                                                                                                                     |

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

[1] <span id="ref-1"><a href="https://pubmed.ncbi.nlm.nih.gov/25730672/">Peyrache A, Lacroix MM, Petersen PC, Buzsáki G. Internally organized mechanisms of the head direction sense. Nat Neurosci. 2015 Apr;18(4):569-75. doi: 10.1038/nn.3968. Epub 2015 Mar 2. PMID: 25730672; PMCID: PMC4376557.</a></span>

[2] <span id="ref-2"><a href="https://arxiv.org/abs/2010.00892">Gower, Robert M., Mark Schmidt, Francis Bach, and Peter Richtárik. "Variance-Reduced Methods for Machine Learning." arXiv preprint arXiv:2010.00892 (2020).</a></span>
