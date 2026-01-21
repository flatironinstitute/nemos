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

# Fit GLM for Classification

The [`CategoricalGLM`](nemos.glm.CategoricalGLM) models categorical outcomes such as behavioral choices.

**Key differences from standard GLM:**
- [`predict`](nemos.glm.CategoricalGLM.predict) returns predicted category labels
- [`predict_proba`](nemos.glm.CategoricalGLM.predict_proba) returns (log-)probabilities for each category

## Generate Synthetic Data

```{code-cell}
import jax
import numpy as np
import nemos as nmo

np.random.seed(200)

n_samples, n_features, n_categories = 1000, 5, 3
X = np.random.randn(n_samples, n_features)

# simulate categorical choices using known coefficients
true_coef = 2 * np.random.randn(n_features, n_categories - 1)
true_intercept = np.zeros(n_categories - 1)

model = nmo.glm.CategoricalGLM(n_categories)
model.coef_ = true_coef
model.intercept_ = true_intercept
true_choice, _ = model.simulate(jax.random.PRNGKey(124), X)
```

## Fit the Model

```{code-cell}
model = nmo.glm.CategoricalGLM(n_categories)

train_samples = 500
model.fit(X[:train_samples], true_choice[:train_samples])
```

## Predict and Evaluate

```{code-cell}
predicted_choice = model.predict(X)

# get class probabilities
probs = model.predict_proba(X)
print(f"Probability shape: {probs.shape}")  # (n_samples, n_categories)
```

## Visualize Results

```{code-cell}
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

cm = confusion_matrix(true_choice[train_samples:], predicted_choice[train_samples:])
disp = ConfusionMatrixDisplay(cm)
disp.plot(text_kw=dict(fontsize=15))
plt.title("Confusion Matrix", fontsize=20)
plt.xlabel(disp.ax_.get_xlabel(), fontsize=15)
plt.ylabel(disp.ax_.get_ylabel(), fontsize=15)
plt.show()
```


```{code-cell} ipython3
:tags: [hide-input]

# save image for thumbnail
from pathlib import Path
import os

fig = disp.figure_
root = os.environ.get("READTHEDOCS_OUTPUT")
if root:
   path = Path(root) / "html/_static/thumbnails/how_to_guide"
# if local store in ../_build/html/...
else:
   path = Path("../_build/html/_static/thumbnails/how_to_guide")

# make sure the folder exists if run from build
if root or Path("../assets/stylesheets").exists():
   path.mkdir(parents=True, exist_ok=True)

if path.exists():
  fig.savefig(path / "glm_for_classification.svg")
```
