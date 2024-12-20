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

(custom-features)=
# Precomputed Features

## Incorporating Precomputed Features into GLM Design

In some cases, your data may contain features that are not directly computable through the available bases in NeMoS, for instance a Principal Component Analysis (PCA) of high-dimensional signals. 
The `IdentityEval` basis allows you to incorporate these precomputed features into the model design, and combine them with other predictors through basis composition.

For example, let's assume that you want to compute a Principal Component Analysis (PCA) of some signals, and use 
the first 2 Principal Components as GLM features. 

Currently, NeMoS doesn't provide a PCA basis, but you can compute the Principal Components in `sklearn`.

```{code-cell}
import numpy as np
from sklearn.decomposition import PCA

n_samples = 100
n_signals = 10

# generate some random signals
high_dim_singals = np.random.randn(n_samples, n_signals)

# generate some counts
counts = np.random.poisson(size=n_samples)

# compute the first 2 pcs
pcs = PCA(2).fit_transform(high_dim_singals)

```

Now, let's see how to use the `IdentityEval` basis to model jointly the PCs and a spike history filter.

```{code-cell}
import nemos as nmo

# create a composite basis 
pc_basis = nmo.basis.IdentityEval()
history_basis = nmo.basis.RaisedCosineLogConv(3, window_size=10)
composite_basis = pc_basis + history_basis

# create the model design
X = composite_basis.compute_features(pcs, counts)

print(f"Design matrix shape: {X.shape}")

# fit the glm
model = nmo.glm.GLM().fit(X, counts)

```