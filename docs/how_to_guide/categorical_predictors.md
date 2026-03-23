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

(categorical_design_matrices)=
# Construct Design Matrices for Categorical Features

## Splitting a Continuous Variable by Category

The primary use of the `Category` basis in NeMoS is to estimate category-specific
tuning curves by multiplying it with a continuous basis:
```{code-cell} ipython3
import numpy as np
import nemos as nmo

# Simulate data: 4 samples, two context labels, a continuous speed variable
context = np.array(["L", "L", "R", "R"])
speed   = np.array([10., 3., 2., 20.])
counts  = np.array([10, 5, 2, 0])

# Category * continuous basis: one set of basis functions per category
bas = nmo.basis.Category(["L", "R"]) * nmo.basis.RaisedCosineLinearEval(3)
X = bas.compute_features(context, speed)
print("X.shape: ", X.shape)  # (4, 6): 3 basis functions × 2 categories

```

## Standalone Categorical Predictors

To add a category as a main effect, drop one column after calling
`compute_features`. The dropped category becomes the reference level and all
remaining coefficients are contrasts against it:
```{code-cell} ipython3
cat_basis = nmo.basis.Category(["L", "R"])
X_cat = cat_basis.compute_features(context)
X_cat = X_cat[:, 1:]  # "L" is the reference; remaining column codes "R" vs "L"
```

:::{warning}
NeMoS GLMs include an intercept. Including all columns of a `Category` basis
as a standalone predictor introduces perfect collinearity — the column sum
equals the intercept column. Always drop one column per categorical variable
when using categories as main effects.
For a detailed discussion of identifiability and the effect of regularization,
see the `Category` basis docstring.
:::

## Complex Designs

:::{dropdown} Additional requirements
:color: warning
:icon: alert

To run this section, you may need to install the `patsy` package:
```bash
pip install patsy
```

Alternatively, to install all dependencies required for running any of our
notebooks, execute:
```bash
pip install "nemos[examples]"
```
:::

For designs involving multiple categorical variables, higher-order interactions,
or non-default contrast coding (sum-to-zero, Helmert, etc.), use
[`patsy`](https://patsy.readthedocs.io) or [`formulaic`](https://matthew.wardrop.casa/formulaic/latest/)
to construct the design matrix. Those libraries resolve redundancies automatically and
support a wide range of coding schemes.

```{code-cell} ipython3
import pandas as pd
from patsy import dmatrix

data = pd.DataFrame({
    'stimulus': ['Tri', 'Sq', 'Tri', 'Sq'],
    'context':  ['C',   'C',   'S',  'S'],
    'counts': [10, 5, 2, 0],
})

formula = "stimulus + context + stimulus:context"
design_df = dmatrix(formula, data, return_type="dataframe")

# patsy adds an intercept; drop it since NeMoS GLMs include one implicitly
design_df.drop(columns=["Intercept"], inplace=True)
design_df
```

:::{dropdown} Understanding `patsy`'s output
:color: info
:icon: info

- `T.` indicates **treatment coding**: `stimulus[T.Sq]` is 1 when
  `stimulus == Sq`, 0 otherwise, with `Tri` as the reference.
- One column is dropped per variable to avoid collinearity with the intercept.
- The interaction term `stimulus[T.Sq]:context[T.S]` captures the joint effect
  of `Sq` and context `S`; the other combinations are absorbed by the reference
  categories.

See the [`patsy` docs](https://patsy.readthedocs.io/en/latest/formulas.html)
for sum-to-zero, Helmert, and other coding schemes.
:::
```{code-cell} ipython3
model = nmo.glm.GLM().fit(design_df, counts)
```

:::{note}

NeMoS basis product of two categories is equivalent to the following `patsy` dmatrix construction:


```{code-block}

>>> import nemos as nmo
>>> import pandas as pd
>>> data = pd.DataFrame({
...     'stimulus': ['Tri', 'Sq', 'Tri', 'Sq'],
...     'context':  ['C',   'C',   'S',  'S'],
...     'counts': [10, 5, 2, 0],
... })
>>> interaction = nmo.basis.Category(["Tri","Sq"]) * nmo.basis.Category(["C","S"])
>>> interaction.compute_features(data["stimulus"], data["context"])
Array([[0., 0., 1., 0.],
       [1., 0., 0., 0.],
       [0., 0., 0., 1.],
       [0., 1., 0., 0.]], dtype=float32)
>>> dmatrix("0+context:stimulus", data, return_type="dataframe")
   context[C]:stimulus[Sq]  context[S]:stimulus[Sq]  context[C]:stimulus[Tri]  context[S]:stimulus[Tri]
0                      0.0                      0.0                       1.0                       0.0
1                      1.0                      0.0                       0.0                       0.0
2                      0.0                      0.0                       0.0                       1.0
3                      0.0                      1.0                       0.0                       0.0
```
:::
