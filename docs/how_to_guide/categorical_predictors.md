---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Construct Design Matrices for Categorical Features

## Design Matrix with `patsy`

:::{dropdown} Additional requirements
:color: warning
:icon: alert

To run this notebook, you may need to install the `patsy` package:

```bash
pip install patsy
```

Alternatively, to install all dependencies required for running any of our notebooks, execute:

```bash
pip install "nemos[examples]"
```
:::

If you have categorized your trials and want to capture a change in the firing rate with the trial category, you can create your design matrix using [`patsy`](https://patsy.readthedocs.io/en/latest/).

Let's assume we have a simple dataset with four samples and two labels assigned to each sample. You can think of these labels as characteristics of the trial to which the samples belong, such as stimulus 1 vs stimulus 2 or context 1 vs context 2.

To make it concrete, let's assume that the stimulus can be the image of a red triangle (Tri) a blue square (Sq), and the context defines the task, for example report either the color (C) or the shape (S) of the stimulus.

```{code-cell} ipython3
# Example data
import pandas as pd
from patsy import dmatrix
import numpy as np

# Example data 
data = pd.DataFrame({
    'stimulus': ['Tri', 'Tri', 'Sq', 'Sq'],  # Stimulus
    'context': ['C', 'C', 'S', 'S']   # Context
})

# Spike counts
counts = np.array([10, 5, 2, 0])
```

You can use a `patsy` formula to structure the design matrix:

```{code-cell} ipython3

# (a:b) indicates interaction between a and b
formula = "stimulus + context + stimulus:context"
```

To construct the design matrix:

```{code-cell} ipython3
# Generate design matrix
design_df = dmatrix(formula, data, return_type="dataframe")

design_df
```

The feature in the design matrix are 1-hot encoded categories.
Note that `patsy` adds an intercept and drops the reference `Tri`. This is done by design to avoid perfect collinearity (the sum of the `Tri` and the `Sq` column would be equal to the intercept). 

NeMoS GLMs, however, implicitly specify an intercept term, making the first column of this dataframe redundant. Let's drop it.

```{code-cell} ipython3
design_df.drop(columns=["Intercept"], inplace=True)

design_df
```

:::{dropdown} Understanding `patsy`'s Output
:color: info
:icon: info

In the design matrix:
- `T.` indicates **treatment coding**, which is used to encode categorical variables. For example, `stimulus[T.Sq]` represents the presence (`1`) or absence (`0`) of the category `Sq` compared to the reference category `Tri`.
- The reference category (`Tri`) is dropped to avoid collinearity, as including all categories would result in redundancy.
- Similarly, for `context`, columns `context[C]`, and `context[T.S]` is dropped to avoid collinearity.
- The interaction term `stimulus[T.Sq]:context[T.S]` represents the combined effect of `stimulus = Sq` and `context = S`. Only one column is needed for the interaction, as the other combinations are implicitly represented by the reference categories (`Tri` and `C`).

See [`patsy` docs](https://patsy.readthedocs.io/en/latest/formulas.html#the-formula-language) for more details. 
:::


### Fit the GLM

We are now ready to fit a GLM model using the 1-hot encoded categories as predictor. The model will learn a different 
firing rate for each condition. 


```{code-cell} ipython3
import nemos as nmo

# Fit the GLM model
model = nmo.glm.GLM().fit(design_df, counts)
```
:::{note}

You can directly pass a [`pandas.DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) to the 
`fit` method, as long as it can be internally converted to a floating-point array (that means, for example, that you can't have a column whose values are strings). The conversion is handled automatically, by calling `design_df.values`,
so no additional steps are required on your part.
:::

## Categorical Predictors and Basis Functions

Assume you have an additional time series that you want to process with a basis:

```{code-cell} ipython3
# Example time series
speed = np.array([10., 3., 2., 20.])
```

You can create a composite basis to combine predictors:

```{code-cell} ipython3
# Identity basis combined with a B-spline basis
bas = nmo.basis.IdentityEval() + nmo.basis.RaisedCosineLinearEval(3)

# Compute features
X = bas.compute_features(design_df, speed)

# convert to DataFrame for readability
# bi(speed) indicates that the speed is passed through the i-th raised cosine basis function
pd.DataFrame(X, columns=design_df.columns.to_list() + [f"b{i}(speed)" for i in range(3)])
```

Here the `IdentityEval` basis just passes through the design_df (converting it to an array), while the `speed` is processed by the `RaisedCosineLinearEval`.
The output of both bases is concatenated to form a single design matrix.

:::{admonition} Basis Composition

Check out this [background note](composing_basis_function) to learn more about basis composition.
:::

### Interaction with Basis algebra

To add the interaction To add the interaction between the `S` context and speed to your design matrix, you can take advantage of basis multiplication:

```{code-cell} ipython3
# Interaction basis
bas = nmo.basis.IdentityEval() * nmo.basis.RaisedCosineLinearEval(3)

# Compute features for interaction
X2 = bas.compute_features(design_df["context[T.S]"], speed)

# convert to DataFrame for readability
pd.DataFrame(X2, columns=[f"context[T.S] * b{i}(speed)" for i in range(3)])
```

