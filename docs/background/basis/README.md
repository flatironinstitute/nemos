# Basis Function

(table_basis)=
```{table}
:name: table-basis

|           **Basis**               |                                     **Kernel Visualization**                 |                       **Examples**                       |                           **Evaluation/Convolution**                                                                                     | **Preferred Mode**   |                               
|:---------------------------------:|:----------------------------------------------------------------------------:|:--------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------:|:--------------------:|
|            **B-Spline**           | <img src="images/EvalBSpline.svg"  alt="B-spline."/>                         | [Grid cells](grid_cells_nemos)                           | [EvalBSpline](nemos.basis.basis.EvalBSpline)<br>[ConvBSpline](nemos.basis.basis.ConvBSpline)                                             | 🟢 Eval              |
|      **Cyclic B-Spline**          | <img src="images/EvalCyclicBSpline.svg"  alt="Cyclic B-spline."/>            | [Place cells](basis_eval_place_cells)                    | [EvalCyclicBSpline](nemos.basis.basis.EvalCyclicBSpline)<br>[ConvCyclicBSpline](nemos.basis.basis.ConvCyclicBSpline)                     | 🟢 Eval              |
|           **M-Spline**            | <img src="images/EvalMSpline.svg"  alt="M-spline."/>                         | [Place cells](basis_eval_place_cells)                    | [EvalMSpline](nemos.basis.basis.EvalMSpline)<br>[ConvMSpline](nemos.basis.basis.ConvMSpline)                                             | 🟢 Eval              |
| **Linearly Spaced Raised Cosine** | <img src="images/EvalRaisedCosineLinear.svg"  alt="Raised Cosine Linear."/>  |                                                          | [EvalRaisedCosineLinear](nemos.basis.basis.EvalRaisedCosineLinear)<br>[ConvRaisedCosineLinear](nemos.basis.basis.ConvRaisedCosineLinear) | 🟢 Eval              |
|    **Log Spaced Raised Cosine**   | <img src="images/EvalRaisedCosineLog.svg"  alt="Raised Cosine Log."/>        | [Head Direction](head_direction_reducing_dimensionality) | [EvalRaisedCosineLog](nemos.basis.basis.EvalRaisedCosineLog)<br>[ConvRaisedCosineLog](nemos.basis.basis.ConvRaisedCosineLog)             | 🔵 Conv              |
| **Orthogonalized Exponential Decays** | <img src="images/EvalOrthExponential.svg"  alt="Orth Exponential Decays"/> |         | [EvalOrthExponential](nemos.basis.basis.EvalOrthExponential)<br>[ConvOrthExponential](nemos.basis.basis.ConvOrthExponential)    | 🟢 Eval              |
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

Instead of tackling the hard problem of learning an unknown function $f(x)$ directly, we reduce it to the simpler task of learning the weights $\{\alpha_i\}$.


## Basis in NeMoS

NeMoS provides a variety of basis functions (see the [table](table_basis) above). For each basis type, there are two dedicated classes of objects, corresponding to the two key uses described in the overview:

- **Eval-basis objects**: For representing non-linear mappings between task variables and outputs. These objects are identified by names starting with `Eval`.
- **Conv-basis objects**: For linear temporal effects. These objects are identified by names starting with `Conv`.

`Eval` and `Conv` objects can be combined to construct multi-dimensional basis functions, enabling complex feature construction.

## Learn More

::::{grid} 1 2 2 2

:::{grid-item-card}

<figure>
<img src="../../_static/thumbnails/background/plot_01_1D_basis_function.svg" style="height: 100px", alt="One-Dimensional Basis."/>
</figure>

```{toctree}
:maxdepth: 2

plot_01_1D_basis_function.md
```
:::

:::{grid-item-card}

<figure>
<img src="../../_static/thumbnails/background/plot_02_ND_basis_function.svg" style="height: 100px", alt="N-Dimensional Basis."/>
</figure>

```{toctree}
:maxdepth: 2

plot_02_ND_basis_function.md
```
:::

::::