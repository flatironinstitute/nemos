# Basis Function

(table_basis)=
```{eval-rst}

.. role:: raw-html(raw)
    :format: html
    
.. list-table::
   :header-rows: 1
   :name: table-basis
   :align: center

   * - **Basis**
     - **Kernel Visualization**
     - **Examples**
     - **Evaluation/Convolution**
     - **Preferred Mode**
   * - **B-Spline**
     - .. plot:: scripts/basis_figs.py plot_bspline
          :show-source-link: False
          :height: 80px
     - :ref:`Grid cells <grid_cells_nemos>`
     - :class:`~nemos.basis.BSplineEval` :raw-html:`<br />`
       :class:`~nemos.basis.BSplineConv`
     - 🟢 Eval
   * - **Cyclic B-Spline**
     - .. plot:: scripts/basis_figs.py plot_cyclic_bspline
          :show-source-link: False
          :height: 80px
     - :ref:`Place cells <basis_eval_place_cells>`
     - :class:`~nemos.basis.CyclicBSplineEval`  :raw-html:`<br />`
       :class:`~nemos.basis.CyclicBSplineConv`
     - 🟢 Eval
   * - **M-Spline**
     - .. plot:: scripts/basis_figs.py plot_mspline
          :show-source-link: False
          :height: 80px
     - :ref:`Place cells <basis_eval_place_cells>`
     - :class:`~nemos.basis.MSplineEval`  :raw-html:`<br />`
       :class:`~nemos.basis.MSplineConv`
     - 🟢 Eval
   * - **Linearly Spaced Raised Cosine**
     - .. plot:: scripts/basis_figs.py plot_raised_cosine_linear
          :show-source-link: False
          :height: 80px
     - 
     - :class:`~nemos.basis.RaisedCosineLinearEval`  :raw-html:`<br />`
       :class:`~nemos.basis.RaisedCosineLinearConv`
     - 🟢 Eval
   * - **Log Spaced Raised Cosine**
     - .. plot:: scripts/basis_figs.py plot_raised_cosine_log
          :show-source-link: False
          :height: 80px
     - :ref:`Head Direction <head_direction_reducing_dimensionality>`
     - :class:`~nemos.basis.RaisedCosineLogEval`  :raw-html:`<br />`
       :class:`~nemos.basis.RaisedCosineLogConv`
     - 🔵 Conv
   * - **Orthogonalized Exponential Decays**
     - .. plot:: scripts/basis_figs.py plot_orth_exp_basis
          :show-source-link: False
          :height: 80px
     - 
     - :class:`~nemos.basis.OrthExponentialEval`  :raw-html:`<br />`
       :class:`~nemos.basis.OrthExponentialConv`
     - 🟢 Eval
   * - **Identity Function**
     - .. plot:: scripts/basis_figs.py plot_identity_basis
          :show-source-link: False
          :height: 80px
     - :ref:`Custom Features <custom-features>`
     - :class:`~nemos.basis.IdentityEval`  :raw-html:`<br />`
     - 🟢 Eval
     
   * - **History Effects**
     - .. plot:: scripts/basis_figs.py plot_history_basis
          :show-source-link: False
          :height: 80px
     - :ref:`Coupled GLM <fully_coupled_glm_how_to>`
     - :class:`~nemos.basis.HistoryConv`  :raw-html:`<br />`
     - 🔵 Conv
```

## Overview

A basis function is a collection of simple building blocks—functions that, when combined (weighted and summed together), can represent more complex, non-linear relationships. Think of them as tools for constructing predictors in GLMs, helping to model:

1. **Non-linear mappings** between task variables (like velocity or position) and firing rates.
2. **Linear temporal effects**, such as spike history, neuron-to-neuron couplings, or how stimuli are integrated over time.

In a GLM, we assume a non-linear mapping exists between task variables and neuronal firing rates. This mapping isn’t something we can directly observe—what we do see are the inputs (task covariates) and the resulting neural activity. The challenge is to infer a "good" approximation of this hidden relationship.

Basis functions help simplify this process by representing the non-linearity as a weighted sum of fixed functions, $\psi_1(x), \dots, \psi_n(x)$, with weights $\alpha_1, \dots, \alpha_n$. Mathematically:

$$
f(x) \approx \alpha_1 \psi_1(x) + \dots + \alpha_n \psi_n(x)
$$

Here, $\approx$ means "approximately equal". 

Instead of tackling the hard problem of learning an unknown function $f(x)$ directly, we reduce it to the simpler task of learning the weights $\{\alpha_i\}$. This preserves convexity, resulting in a much simpler optimization problem.


## Basis in NeMoS

NeMoS provides a variety of basis functions (see the [table](table_basis) above). For each basis type, there are two dedicated classes of objects, corresponding to the two uses described above:

- **Eval basis objects**: For representing non-linear mappings between task variables and outputs. These objects all have names ending with `Eval`.
- **Conv basis objects**: For linear temporal effects. These objects all have names ending with `Conv`.

`Eval` and `Conv` objects can be combined to construct multi-dimensional basis functions, enabling [complex feature construction](composing_basis_function).

## Learn More

::::{grid} 1 2 2 2

:::{grid-item-card}

```{eval-rst}

.. plot:: scripts/basis_figs.py plot_1d_basis_thumbnail
   :show-source-link: False
   :height: 100px
```

```{toctree}
:maxdepth: 2

plot_01_1D_basis_function.md
```
:::

:::{grid-item-card}

```{eval-rst}

.. plot:: scripts/basis_figs.py plot_nd_basis_thumbnail
   :show-source-link: False
   :height: 100px
```

```{toctree}
:maxdepth: 2

plot_02_ND_basis_function.md
```
:::

::::
