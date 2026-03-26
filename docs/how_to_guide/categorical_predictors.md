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

The primary use of the [`Category`](nemos.basis.Category) basis in NeMoS is to estimate category-specific
tuning curves by multiplying it with a continuous basis.

For example, consider an experiment where a subject performs either a leftward or rightward turn on each trial, and we want to learn separate coefficients for each motion type. You can use the `Category` basis to produce an appropriate design matrix:

```{code-cell} ipython3
import numpy as np
import nemos as nmo

# Simulate data: 4 samples, two context labels, a continuous speed variable
context = np.array(["L", "L", "R", "R"])
speed   = np.array([10., 3., 2., 20.])
counts  = np.array([10, 5, 10, 0])

# Category * continuous basis: one set of basis functions per category
bas = nmo.basis.Category(["L", "R"]) * nmo.basis.RaisedCosineLinearEval(3)
X = bas.compute_features(context, speed)
print("X.shape: ", X.shape)  # (4, 6): 3 basis functions × 2 categories

```

## Standalone Categorical Predictors: Why do we recommend dropping a column?

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
see [Identifiability of Categorical Predictors](categorical_identifiability).
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
[`patsy`](https://patsy.readthedocs.io) or [`formulaic`](https://matthewwardrop.github.io/formulaic/)
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
design_df = design_df.drop(columns=["Intercept"])
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

Full one-hot encoding of each term in the formula — the two categorical variables and their interaction — would have produced 8 columns, 4 of which would be redundant. `patsy` detects and drops all redundant columns automatically, guaranteeing that model coefficients are identifiable.

```{code-cell} ipython3
model = nmo.glm.GLM().fit(design_df, counts)
```


## NeMoS `Category` vs `patsy`

NeMoS [`Category`](nemos.basis.Category) basis provides a simple dummy coding of categorical variables. This is just one of the many encoding schemes that `patsy` provides.

For example, the dummy code for one categorical predictor in NeMoS,

```{code-cell} ipython3
nmo.basis.Category(["Tri","Sq"]).compute_features(data["stimulus"])
```

is equivalent to `patsy`'s,

```{code-cell} ipython3
dmatrix("0 + stimulus", data, return_type="dataframe")
```

Similarly, the dummy code for the interaction of two categories,

```{code-cell} ipython3
interaction = nmo.basis.Category(["Tri","Sq"]) * nmo.basis.Category(["C","S"])
interaction.compute_features(data["stimulus"], data["context"])
```

is equivalent to `patsy`'s

```{code-cell} ipython3
dmatrix("0 + context:stimulus", data, return_type="dataframe")
```

NeMoS `Category` covers only basic encodings; for more complex design schemes, see [`patsy`](https://patsy.readthedocs.io) and [`formulaic`](https://matthewwardrop.github.io/formulaic/).
