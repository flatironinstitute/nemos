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

When the [`Category`](nemos.basis.Category) basis is used as a standalone main-effect predictor
together with a NeMoS GLM (which always includes an intercept), you can find infinitely many
coefficient vectors that produce the exact same linear combination $X \cdot w$. Let's see why:

```{code-cell} ipython3
import numpy as np
import nemos as nmo

category = np.array(["L", "L", "R", "R"])
cat_basis = nmo.basis.Category(["L", "R"])
X = cat_basis.compute_features(category)

print("One-hot encoding:\n", X)
print("Sum over columns:", X.sum(axis=1))
```

Because the one-hot columns sum to a vector of ones, we can find a set of coefficients $w$ and intercept $c$ for which $ X \cdot w + c = 0$,

```{code-cell} ipython3

c = -1
w = np.array([1, 1])

print(r"X @ w + c:", X @ w + c)

# this is equivalent to stacking c and w in a single vector
# and multiply this vector with X_agu = [1 | X]

v = np.hstack([c, w])
X_aug = np.column_stack([np.ones(len(category)), X])

print(r"X_aug @ v:", X_aug @ v)  # = 0  →  v is in the null space of X_aug
```

Since $X_{\text{aug}} \cdot v = 0$, adding any multiple of $v$ to the parameters leaves predictions
unchanged:

```{code-cell} ipython3
params = np.array([2., 0.5, -0.3]) # arbitrary parameters [intercept, coef]
alpha = 5.0                        # arbitrary shift along the null direction

print("X_aug @ params:             ", X_aug @ params)
print("X_aug @ (params + alpha*v): ", X_aug @ (params + alpha * v))
```

Models with parameters $\text{coef} = [c, w]$ and $\text{coef} + \alpha \cdot v$ predict the same firing rate for any $\alpha$.
This is **non-identifiability**: the data alone cannot distinguish between them, and there is
no unique optimal solution. Vectors like $v$ are said to be in the null space of the augmented matrix `[1 | X]`.

A practical way to detect this is to check whether the rank of `[1 | X]` is strictly
less than the number of columns — if so, a non-trivial null space exists, i.e., there are non-zero vectors $v$ such that $X_\text{aug} \cdot v = 0$:

```{code-cell} ipython3
print("Rank of [1 | X]:", np.linalg.matrix_rank(X_aug))
print("Number of columns:           ", X_aug.shape[1])
# rank < n_cols  →  null space exists  →  model is non-identifiable
```

## Reference Coding: A Simple Way to Recover Identifiability

Drop one column from the full encoding. The retained columns become *contrasts* against the
dropped (reference) category. All columns are now linearly independent:

```{code-cell} ipython3
X_ref = X[:, 1:]  # drop "L"; the remaining column codes "R" vs "L"
print("Rank of [1 | X_ref]:", np.linalg.matrix_rank(np.c_[np.ones(len(category)), X_ref]))
print("Number of columns:      ", 1 + X_ref.shape[1])  # rank == n_cols → full rank
```

The reference level is arbitrary from a model-fit perspective, but determines coefficient
interpretation: every retained coefficient is the effect of that category *relative to the
reference*. See [^1] for a full discussion of contrast coding schemes.

## When Redundancy Does Not Arise

Multiplying a `Category` basis with a continuous basis produces *category-specific tuning curves*.
The intercept is not involved — the interaction columns are already linearly independent — so no
column needs to be dropped:

```{code-cell} ipython3
speed = np.array([10., 3., 2., 20.])
speed_by_context = nmo.basis.Category(["L", "R"]) * nmo.basis.RaisedCosineLinearEval(3)
X_interact = speed_by_context.compute_features(category, speed)
print("X_interact.shape:", X_interact.shape)  # (4, 6): 3 basis functions × 2 categories

# Full-rank even without dropping anything
print("Rank:", np.linalg.matrix_rank(np.c_[np.ones(len(category)), X_interact]))
```

See [Construct Design Matrices for Categorical Features](categorical_design_matrices) for worked
examples of this pattern.

## Effect of Regularization

If all columns are retained (no reference dropped), whether the fitted coefficients are unique
depends on the regularizer:

| Regularizer | Unique solution? | Notes |
|---|---|---|
| None (unregularized) | No | Any redistribution of the total effect across redundant columns that preserves predictions is equally valid. Do not interpret individual coefficients. |
| Ridge (L2) | Yes | The L2 penalty is symmetric; it forces a specific linear relationship between the intercept and category coefficients. The solution is well-defined but the coefficients do not have a contrast interpretation. |
| Lasso (L1) | No | The L1 penalty does not restore uniqueness along the degenerate directions. Coefficient values are solver-dependent. |
| Elastic net | Yes | The L2 component restores strict convexity [^2]. Unique, but less interpretable than pure Ridge; dropping a reference is recommended whenever interpretation matters. |

The practical rule: **always drop a reference column when using `Category` as a standalone
predictor**, regardless of regularizer. This makes coefficients interpretable as contrasts and
avoids solver-dependent results.

[^1]: UCLA OARC: Coding Systems for Categorical Variables.
    <https://stats.oarc.ucla.edu/r/library/r-library-contrast-coding-systems-for-categorical-variables/>
[^2]: Zou, H. & Hastie, T. (2005). Regularization and variable selection via the elastic net.
    *Journal of the Royal Statistical Society: Series B*, 67(2), 301–320.
