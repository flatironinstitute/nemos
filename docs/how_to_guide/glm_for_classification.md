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

The logistic regression classification model is a type of GLM where the observations are modeled as a categorical random variable. In NeMoS, this model goes under the name of [`CategoricalGLM`](nemos.glm.CategoricalGLM). The [`CategoricalGLM`](nemos.glm.CategoricalGLM) follows a very similar syntax to the GLM but with few key differences:

- [predict](nemos.glm.CategoricalGLM.predict): The predict method of the model returns the predicted categories (instead of the predicted rate).
- [predict_proba](nemos.glm.CategoricalGLM.predict_proba): An additional method returning either the (log-)probabilities of each category.

```{code-cell}
import jax
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import nemos as nmo

np.random.seed(200)

# Generate some dummy data and simulate choice
n_samples, n_features, n_categories = 1000, 5, 3
X = np.random.randn(n_samples, n_features)
true_coef = 2 * np.random.randn(n_features, n_categories - 1)
true_intercept = np.zeros(n_categories - 1)

# define a categorical model
model = nmo.glm.CategoricalGLM(n_categories)
# set the coef and intercept, and simulate choices
model.coef_ = true_coef
model.intercept_ = true_intercept
true_choice, _ = model.simulate(jax.random.PRNGKey(124), X)

# define a new model and fit the coefficients
model = nmo.glm.CategoricalGLM(n_categories)

# fit on the first half
train_samples = 500
model.fit(X[:train_samples], true_choice[:train_samples])

# predicted the last 100 choice
predicted_choice = model.predict(X)


# Plot the confusion matrix on the second half
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
