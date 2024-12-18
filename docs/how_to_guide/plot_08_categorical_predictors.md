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

```{code-cell} ipython3
:tags: [hide-input]

%matplotlib inline
import warnings

# Ignore the first specific warning
warnings.filterwarnings(
    "ignore",
    message="plotting functions contained within `_documentation_utils` are intended for nemos's documentation.",
    category=UserWarning,
)

# Ignore the second specific warning
warnings.filterwarnings(
    "ignore",
    message="Ignoring cached namespace 'core'",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=(
        "invalid value encountered in div "
    ),
    category=RuntimeWarning,
)
```

# Trial Type Effects

## Create A Design Matrix with `patsy`

If you have categorized your trials and want capture a change in the firing rate with the trial category, you can
create your design matrix taking advantage of `patsy`.

Let's assume that we have a very simple dataset with 4 samples, and two labels that we can attach to each sample. 
You can think of this label as marking a characteristic of the trial to which the samples belong, for example stimulus 
1 vs stimulus 2 or context 1 vs context 2 etc.


```{code-cell} ipython3
# Example data
import pandas as pd
from patsy import dmatrix
import numpy as np

# Example data 
data = pd.DataFrame({
    'stimulus': ['s1', 's1', 's2', 's2'],  # Stimulus
    'context': ['c1', 'c2', 'c1', 'c2']   # Context
})

# Spike counts
counts = np.array([10, 5, 2, 0])
```

You can use the `patsy` formula for structuring the design matrix:


```{code-cell} ipython3

# 0 means no intercept (NeMoS GLM has already an intercept term)
# (a:b) means interaction between a and b
formula = "0 + stimulus + context + stimulus:context"
```

To construct the design:

```{code-cell} ipython3

design_df = dmatrix(formula, data, return_type="dataframe")

design_df
```

Fit the GLM

```{code-cell} ipython3
import nemos as nmo

model = nmo.glm.GLM().fit(np.asarray(design_df), counts)
```

## Categorical Predictors And Basis

Assume that you have an extra time series that you want to process via basis:

```{code-cell} ipython3

speed = np.array([10., 3., 2., 20.])
```

Create a composite basis concatenating both predictors ,

```{code-cell} ipython3

bas = nmo.basis.IdentityEval() + nmo.basis.BSplineEval(5)
X = bas.compute_features(np.asarray(design_df), speed)
```

Interactions with speed and category:


```{code-cell} ipython3

bas = nmo.basis.IdentityEval() * nmo.basis.BSplineEval(5)
X2 = bas.compute_features(np.asarray(design_df), speed)
```

