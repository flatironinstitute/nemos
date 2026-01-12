---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Selecting Covariates via Group Lasso

Fitting an encoding model—such as a Generalized Linear Model (GLM)—with multiple predictors is common practice (e.g., speed and position, head direction and theta phase). In this note, we demonstrate how to use the [GroupLasso](~nemos.regularizer.GroupLasso) regularizer to identify the most informative covariates.

We begin by generating artificial data consisting of an animal’s position on a linear maze, its speed, and simultaneously recorded spike counts.

```{code-cell}
import nemos as nmo
import numpy as np
np.random.seed(123)

n_samples = 100

# Dummy behavioral data
position = np.random.randn(n_samples)
speed = np.random.randn(n_samples)

# Nonlinear response to position
n_basis = 5
bas = nmo.basis.BSplineEval(n_basis)
coef = np.random.randn(n_basis)
X = bas.compute_features(position)

# Simulated response
firing = np.exp(np.dot(X, coef))
counts = np.random.poisson(firing)
```

Next, we model the neuronal response using a NeMoS [basis](table-basis) and [FeaturePytree](pytrees_howto) and fit a [Group Lasso–regularized](~nemos.regularizer.GroupLasso) GLM across a range of regularization strengths.

```{code-cell}
import matplotlib.pyplot as plt

bas = (
    nmo.basis.RaisedCosineLinearEval(6, label="position") +
    nmo.basis.RaisedCosineLinearEval(6, label="speed")
)
predictors = bas.compute_features(position, speed)
print(predictors.shape)

# Create a FeaturePytree of predictors
predictors = nmo.pytrees.FeaturePytree(**bas.split_by_feature(predictors))
print("Pytree predictors:", predictors)

# Define a GroupLasso GLM
# Each element in the FeaturePytree is treated as a group and shrunk jointly
model = nmo.glm.GLM(regularizer="GroupLasso", solver_kwargs={"maxiter": 5000})

# Range of regularization strengths
reg_strengths = np.geomspace(1e-4, 1, 10)

# Containers for coefficient norms
speed_coef_norm = np.zeros(len(reg_strengths))
position_coef_norm = np.zeros(len(reg_strengths))

# Fit model for each regularization strength
for i, reg in enumerate(reg_strengths):
    model.regularizer_strength = reg
    model.fit(predictors, counts)
    speed_coef_norm[i] = np.linalg.norm(model.coef_["speed"])
    position_coef_norm[i] = np.linalg.norm(model.coef_["position"])

# Plot results
plt.figure()
plt.title("Regularization Path of Grouped Coefficients")
plt.xscale("log")
plt.plot(reg_strengths, speed_coef_norm, label="speed")
plt.plot(reg_strengths, position_coef_norm, label="position")
plt.xlabel("Regularization strength")
plt.ylabel("Coefficient norm")
plt.legend()
plt.tight_layout()
```

As the regularization strength increases, the Group Lasso progressively shrinks entire groups of coefficients toward zero. In this example, the weaker predictor is eliminated first, and the Group Lasso correctly identifies `position` as the most informative covariate.
