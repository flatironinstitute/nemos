(solver-selection)=
# Benchmarking

In this note we compare solver performance in GLM problems on simulated data and neural recordings (from [[1]](#ref-1)), for all combinations of:

- **devices:** cpu vs cuda.
- **solvers:** NeMoS JIT compiled solvers and a scipy wrapper of `L-BFGS-B`, see the [table below](table_solvers) for a complete list.
- **problem sizes:** (simulation only) by varying number of samples, features and neurons.

Additionally, we compared against scikit-learn's `PoissonRegressor` with the `"newton-cholesky"` solver, which was their most efficient option for all the configurations tested.

:::{admonition} JIT vs `scipy.minimize`
:class: note

NeMoS native solvers JIT-compile the full optimization when `GLM.fit` is called, incurring a one-time compilation cost before the first iteration. `L-BFGS-B` from `scipy.minimize` calls pre-compiled Fortran routines but invokes Python at each iteration to evaluate the GLM likelihood and gradient. As a result, most JIT-compiled solvers run faster per-iteration once compiled, while `scipy` avoids the compilation cost entirely. If the optimization loop dominates over compilation (large problems, many iterations), prefer the JIT-compiled solvers; for small problems, the scipy wrapper is likely faster.
:::

```{contents}
:local:
:depth: 2
```

## Results

- **Simulations**: fitting simulated data requires fewer optimization steps; compilation cost can dominate, especially for small datasets. `scipy.minimize` and `scikit-learn` are the most efficient options for small problems, while native NeMoS solvers are significantly faster for large ones.
- **Neural recordings**: fitting real recordings often requires hundreds to thousands of optimization steps. JIT-compiled solvers are more efficient since the compilation cost is negligible — and this is the common case in practice.
- **GPU vs CPU**: GPU compilation is slower, but the optimization loop scales very well with problem size, making GPU the most performant option for large problems.

(summary-chart)=
### Summary chart

```{raw} html
<div style="display:flex; align-items:center; gap:1.5em; margin-bottom:1.2em; flex-wrap:wrap;">
  <div>
    <label for="filter-version"><strong>Version:&nbsp;</strong></label>
    <select id="filter-version"></select>
  </div>
  <div>
    <label for="filter-dataset"><strong>Dataset:&nbsp;</strong></label>
    <select id="filter-dataset">
      <option value="all">All</option>
      <option value="recordings">Recordings</option>
      <option value="small">Small sim</option>
      <option value="large">Large sim</option>
    </select>
  </div>
  <div>
    <label for="filter-sort"><strong>Sort by:&nbsp;</strong></label>
    <select id="filter-sort">
      <option value="gpu">GPU</option>
      <option value="cpu">CPU</option>
    </select>
  </div>
  <div>
    <label for="filter-scale"><strong>Y-scale:&nbsp;</strong></label>
    <select id="filter-scale">
      <option value="linear">Linear</option>
      <option value="log">Log</option>
    </select>
  </div>
  <div>
    <label><input type="checkbox" id="filter-share-y" checked>&nbsp;<strong>Share Y</strong></label>
  </div>
</div>
```

```{raw} html
<div id="benchmark-chart" style="width:100%; height:480px;"></div>
```

### Tables

(small-problem-size)=
#### Small problem size

- Sample size: 100 — Feature dimension: 1 — Neurons: 1

```{raw} html
<table id="table-small-sim" class="display" style="width:100%">
  <thead><tr>
    <th>Version</th><th>Device</th><th>Solver</th>
    <th>Fit time (s)</th><th>Converged</th><th>Avg. iterations</th><th>Compile fraction</th>
  </tr></thead>
  <tbody></tbody>
</table>
```

(large-problem-size)=
#### Large problem size

- Sample size: 100,000 — Feature dimension: 100 — Neurons: 20

```{raw} html
<table id="table-large-sim" class="display" style="width:100%">
  <thead><tr>
    <th>Version</th><th>Device</th><th>Solver</th>
    <th>Fit time (s)</th><th>Converged</th><th>Avg. iterations</th><th>Compile fraction</th>
  </tr></thead>
  <tbody></tbody>
</table>
```

(neural-recordings)=
#### Neural recordings

- Sample size: 195,820 — Feature dimension: 95 — Neurons: 19

```{raw} html
<table id="table-recordings" class="display" style="width:100%">
  <thead><tr>
    <th>Version</th><th>Device</th><th>Solver</th>
    <th>Fit time (s)</th><th>Converged</th><th>Avg. iterations</th><th>Compile fraction</th>
  </tr></thead>
  <tbody></tbody>
</table>
```

(benchmarked-solvers)=
## Benchmarked solvers

| Solver | Backend | Notes |
|--------|---------|-------|
| [`GradientDescent`](https://en.wikipedia.org/wiki/Gradient_descent) | [optimistix](https://docs.kidger.site/optimistix/) | First-order (gradient only). Memory scales linearly with parameter count. Smooth penalties only; typically requires many iterations. |
| [`BFGS`](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm) | [optimistix](https://docs.kidger.site/optimistix/) | Quasi-Newton; maintains a dense Hessian approximation. Memory scales quadratically — impractical for large parameter spaces. Fewer iterations than first-order methods. |
| [`LBFGS`](https://en.wikipedia.org/wiki/Limited-memory_BFGS) | [optax](https://optax.readthedocs.io/) + [optimistix](https://docs.kidger.site/optimistix/) | Limited-memory quasi-second-order; stores the last *m* gradient vectors. Memory scales linearly. Recommended default for smooth problems. |
| [`LBFGS`](https://en.wikipedia.org/wiki/Limited-memory_BFGS) | scipy | L-BFGS-B via [`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html). No JIT compilation cost; Python callback per iteration. Preferred for small problems or many repeated `fit()` calls. |
| [`NewtonCholesky`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html) | scikit-learn | Newton method; each iteration builds the p × p Hessian (O(n · p²)) and solves it via Cholesky (O(p³)), where n = samples and p = features. Memory O(p²). Converges in very few iterations. CPU only. |
| [`ProximalGradient`](https://en.wikipedia.org/wiki/Proximal_gradient_method) | [optimistix](https://docs.kidger.site/optimistix/) | First-order + proximal step for non-smooth penalties (Lasso, GroupLasso, ElasticNet). Memory scales linearly. Typically requires many iterations. |
| [`SVRG`](https://en.wikipedia.org/wiki/Stochastic_variance_reduction#SVRG) | nemos | Mini-batch gradient steps with a full-gradient anchor step per epoch. Memory scales linearly. The nested inner-outer loop structure can be slow on GPU. [[2]](#ref-2) |
| [`ProxSVRG`](https://en.wikipedia.org/wiki/Stochastic_variance_reduction#SVRG) | nemos | Proximal variant of SVRG for non-smooth penalties. Same inner-outer loop structure and GPU caveats as `SVRG`. |


## References

[1] <span id="ref-1"><a href="https://pubmed.ncbi.nlm.nih.gov/25730672/">Peyrache A, Lacroix MM, Petersen PC, Buzsáki G. Internally organized mechanisms of the head direction sense. Nat Neurosci. 2015 Apr;18(4):569-75.</a></span>

[2] <span id="ref-2"><a href="https://arxiv.org/abs/2010.00892">Gower, Robert M., Mark Schmidt, Francis Bach, and Peter Richtárik. "Variance-Reduced Methods for Machine Learning." arXiv preprint arXiv:2010.00892 (2020).</a></span>
