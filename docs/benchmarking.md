(solver-selection)=
# Benchmarking

In this note we compare solver performance in GLM problems on simulated data and neural recordings (from [[1]](#ref-1)), for all combinations of:

- **devices:** cpu vs cuda.
- **solvers:**  NeMoS JIT compiled solvers and a scipy wrapper of `L-BFGS-B`, see the [table below](table_solvers) for a complete list.
- **problem sizes:** (simulation only) by varying number of samples, features and neurons.

Additionally, we compared against scikit-learn's `PoissonRegressor` with the `"newton-cholesky"` solver, which was their most efficient option for all the configurations tested.

:::{admonition} JIT vs `scipy.minimize`
:class: note

NeMoS native solvers JIT-compile the full optimization when `GLM.fit` is called, incurring a one-time compilation cost before the first iteration. `L-BFGS-B` from `scipy.minimize` calls pre-compiled Fortran routines but invokes Python at each iteration to evaluate the GLM likelihood and gradient. As a result, most JIT-compiled solvers run faster per-iteration once compiled, while `scipy` avoids the compilation cost entirely. If the optimization loop dominates over compilation (large problems, many iterations), prefer the JIT-compiled solvers; for small problems, the scipy wrapper provided below is likely faster.
:::


## Results

- **Simulations**: fitting simulated data requires a smaller number of optimization steps, and the compilation cost may dominate over the optimization loop, especially for smaller datasets. For this reason the `scipy.minimize` wrapper and `scikit-learn` are the most efficient options for small problem sizes, while native NeMoS options are significantly faster for larger problems.

- **Neural Recordings**: fitting neural recordings often requires hundreds or even thousands of optimization steps, depending on the algorithm. JIT-compiled solvers tend to be more efficient since the compilation cost is negligible. In our experience, this is commonly the case in real applications.

- **GPU vs CPU**: GPU compilation is slower than its CPU counterpart; however, the optimization iteration scales very well with problem size. For this reason, GPU optimization is the most performant option for large problems.


### Small problem size

- Sample size: 100
- Feature dimension: 1
- Number of neurons: 1

```{raw} html
<div style="margin-bottom: 1em;">
  <label for="filter-small-sim"><strong>Version:&nbsp;</strong></label>
  <select id="filter-small-sim"><option value="">All</option></select>
</div>
<table id="table-small-sim" class="display" style="width:100%">
  <thead><tr>
    <th>Version</th><th>Device</th><th>Solver</th>
    <th>Fit time (s)</th><th>Converged</th><th>Avg. iterations</th><th>Compile fraction</th>
  </tr></thead>
  <tbody></tbody>
</table>
```

### Large problem size

- Sample size: 100,000
- Feature dimension: 100
- Number of neurons: 20

```{raw} html
<div style="margin-bottom: 1em;">
  <label for="filter-large-sim"><strong>Version:&nbsp;</strong></label>
  <select id="filter-large-sim"><option value="">All</option></select>
</div>
<table id="table-large-sim" class="display" style="width:100%">
  <thead><tr>
    <th>Version</th><th>Device</th><th>Solver</th>
    <th>Fit time (s)</th><th>Converged</th><th>Avg. iterations</th><th>Compile fraction</th>
  </tr></thead>
  <tbody></tbody>
</table>
```

### Neural Recordings

- Sample size: 195,820
- Feature dimension: 95
- Number of neurons: 19

```{raw} html
<div style="margin-bottom: 1em;">
  <label for="filter-recordings"><strong>Version:&nbsp;</strong></label>
  <select id="filter-recordings"><option value="">All</option></select>
</div>
<table id="table-recordings" class="display" style="width:100%">
  <thead><tr>
    <th>Version</th><th>Device</th><th>Solver</th>
    <th>Fit time (s)</th><th>Converged</th><th>Avg. iterations</th><th>Compile fraction</th>
  </tr></thead>
  <tbody></tbody>
</table>
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


## References

[1] <span id="ref-1"><a href="https://pubmed.ncbi.nlm.nih.gov/25730672/">Peyrache A, Lacroix MM, Petersen PC, Buzsáki G. Internally organized mechanisms of the head direction sense. Nat Neurosci. 2015 Apr;18(4):569-75. doi: 10.1038/nn.3968. Epub 2015 Mar 2. PMID: 25730672; PMCID: PMC4376557.</a></span>

[2] <span id="ref-2"><a href="https://arxiv.org/abs/2010.00892">Gower, Robert M., Mark Schmidt, Francis Bach, and Peter Richtárik. "Variance-Reduced Methods for Machine Learning." arXiv preprint arXiv:2010.00892 (2020).</a></span>
