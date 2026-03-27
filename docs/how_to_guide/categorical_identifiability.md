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

(categorical_identifiability)=
# Resolving Redundancy in Categorical Design Matrices

## Why Does Redundancy Arise?

When the [`Category`](nemos.basis.Category) basis is used as a standalone main-effect predictor
together with a NeMoS GLM (which always includes an intercept), there are infinitely many coefficient vectors that produce the same linear predictor $X \cdot \mathbf{w}$. Let's see why:

```{code-cell} ipython3
import numpy as np
import nemos as nmo

category = np.array(["L", "L", "L", "L", "R", "R", "R", "R"])
cat_basis = nmo.basis.Category(["L", "R"])
X = cat_basis.compute_features(category)

print("One-hot encoding:\n", X)
print("Sum over columns:", X.sum(axis=1))
```

Because the one-hot columns sum to a vector of ones, we can find coefficients $\mathbf{w}$ and
intercept $c$ for which $X \cdot \mathbf{w} + c = 0$:

```{code-cell} ipython3
c = -1
w = np.array([1, 1])

print("X @ w + c:", X @ w + c)
```

Stacking $c$ and $\mathbf{w}$ into a single vector $\mathbf{v} = [c, \mathbf{w}]$ and prepending
an all-ones column to $X$ gives the compact form $X_\text{aug} \cdot \mathbf{v} = 0$:

```{code-cell} ipython3
v = np.hstack([c, w])
X_aug = np.column_stack([np.ones(len(category)), X])  # [1 | X]

print("X_aug @ v:", X_aug @ v)  # = 0  →  v is in the null space of X_aug
```

In mathematical terms, we can say that $\mathbf{v}$ is in the null space of the augmented matrix $X_\text{aug}$. This happens when at least one column of $X_\text{aug}$ can be expressed as a weighted sum of the other columns—that is, the columns are **linearly dependent**. The representation is therefore redundant, and we can add any multiple of $\mathbf{v}$ to the parameters without changing the predictions:

```{code-cell} ipython3
params = np.array([2., 0.5, -0.3]) # arbitrary parameters [intercept, coef]
alpha = 5.0                        # arbitrary shift along the null direction

print("X_aug @ params:             ", X_aug @ params)
print("X_aug @ (params + alpha*v): ", X_aug @ (params + alpha * v))
```

Models with parameters $[c, \mathbf{w}]$ and $[c, \mathbf{w}] + \alpha \cdot \mathbf{v}$ predict the same firing rate for any $\alpha \in \mathbb{R}$.
This is **non-identifiability**: the data alone cannot distinguish between them, and there is no unique optimal solution.

A practical way to detect this redundancy is to check whether the rank of `X_aug = [1 | X]` is strictly less than number of its columns.
**The rank is the number of linearly independent columns**, so if it is smaller than the total number of columns, some columns must be redundant. In that case, a non-trivial null space exists, i.e., there are non-zero vectors $\mathbf{v}$ such that $X_\text{aug} \cdot \mathbf{v} = 0$:

```{code-cell} ipython3
print(f"Rank of [1 | X]:   {np.linalg.matrix_rank(X_aug)}")
print(f"Number of columns: {X_aug.shape[1]}")
# rank < n_cols  →  null space exists  →  model is non-identifiable
```

## Reference Coding: A Simple Way to Recover Identifiability

Drop one column from the full encoding. The retained columns become *contrasts* against the
dropped (reference) category. All columns are now linearly independent:

```{code-cell} ipython3
X_ref = X[:, 1:]  # drop "L"; the remaining column codes "R" vs "L"
X_ref_aug = np.column_stack([np.ones(len(category)), X_ref])
# rank == n_cols → full rank
print(f"Rank of [1 | X_ref]: {np.linalg.matrix_rank(X_ref_aug)}")
print(f"Number of columns:   {1 + X_ref.shape[1]}")
```

The reference level is arbitrary from a model-fit perspective, but determines coefficient
interpretation: every retained coefficient is the effect of that category *relative to the
reference*. See [^1] for a full discussion of contrast coding schemes.

## When Redundancy Does Not Arise

Multiplying a `Category` basis with a continuous basis produces *category-specific tuning curves*.
The intercept is not involved — the interaction columns are already linearly independent — so no
column needs to be dropped:

```{code-cell} ipython3
speed = np.array([10., 3., 2., 20., 5., 8., 15., 1.])
speed_by_context = nmo.basis.Category(["L", "R"]) * nmo.basis.RaisedCosineLinearEval(3)
X_interact = speed_by_context.compute_features(category, speed)
print("X_interact.shape:", X_interact.shape)  # (8, 6): 3 basis functions × 2 categories

# Full-rank even without dropping anything
print("Rank:", np.linalg.matrix_rank(np.c_[np.ones(len(category)), X_interact]))
```

See [Construct Design Matrices for Categorical Features](categorical_design_matrices) for worked
examples of this pattern.

## Effect of Regularization

If all columns are retained (no reference dropped), whether the fitted coefficients are unique
depends on the regularizer:

| Regularizer | Unique solution? | Notes                                                                                                                                                                                                                                                                              |
|---|---|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| None (unregularized) | No | Any redistribution of the total effect across redundant columns that preserves predictions is equally valid. Do not interpret individual coefficients.                                                                                                                             |
| Ridge (L2) | Yes | The L2 penalty is symmetric; it selects a unique solution by shrinking coefficients, effectively imposing a specific linear relationship between the intercept and category coefficients. The solution is well-defined but the coefficients do not have a contrast interpretation. |
| Lasso (L1) | No | The L1 penalty does not restore uniqueness along the degenerate directions. multiple solutions can achieve the same objective value. Coefficient values are solver-dependent.                                                                                                      |
| Elastic net | Yes | The L2 component restores strict convexity [^2]. Unique, but less interpretable than pure Ridge; dropping a reference is recommended whenever interpretation matters.                                                                                                              |

In practice, we advise to always drop a reference column when using `Category` as a standalone
predictor, regardless of regularizer. This makes coefficients interpretable as contrasts and
avoids solver-dependent results.

[^1]: UCLA Othis ARC: Coding Systems for Categorical Variables.
    <https://stats.oarc.ucla.edu/r/library/r-library-contrast-coding-systems-for-categorical-variables/>
[^2]: Zou, H. & Hastie, T. (2005). Regularization and variable selection via the elastic net.
    *Journal of the Royal Statistical Society: Series B*, 67(2), 301–320.
