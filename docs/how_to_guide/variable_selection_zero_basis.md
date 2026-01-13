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

(variable-selection)=
# Model Selection: Cross-validate over Inputs

When modeling neural activity with multiple inputs, you may want to determine which inputs are necessary. The `Zero` basis acts as a placeholder that contributes no features, allowing you to systematically test different input combinations.

## Load data

We'll use place cell data to test whether position, theta phase, or both are needed to predict neural responses. See the [place cells tutorial](place-cells-data) for details on this dataset.

```{code-cell} ipython3
import nemos as nmo
import pynapple as nap
from sklearn.model_selection import cross_val_score

# Fetch data
path = nmo.fetch.fetch_data("Achilles_10252013.nwb")
data = nap.load_file(path)

# Get spikes, position, and theta phase
spikes = data["units"].getby_category("cell_type")["pE"]
position = data["position"].restrict(data["trials"])
theta = data["theta_phase"]

# Select one neuron and bin spikes
neuron = spikes[92]
bin_size = 0.1
counts = neuron.count(bin_size, ep=position.time_support)

# Align position and theta to spike counts
position = position.interpolate(counts, ep=counts.time_support)
theta = theta.interpolate(counts, ep=counts.time_support)
```

## Cross-validate over inputs

We'll use [scikit-learn's cross-validation](sklearn-how-to) to compare models with different input combinations.

```{code-cell} ipython3
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np

# Define complete basis configurations
position_basis = nmo.basis.BSplineEval(n_basis_funcs=10)
theta_basis = nmo.basis.CyclicBSplineEval(n_basis_funcs=8)

# Use Zero as placeholder for excluded inputs
basis_both = position_basis + theta_basis
basis_position = position_basis + nmo.basis.Zero()
basis_theta = nmo.basis.Zero() + theta_basis

basis_both.label = "both"
basis_position.label = "position"
basis_theta.label = "theta"


# Set up pipeline
pipeline = Pipeline([
    ("basis", basis_both.to_transformer()),
    ("glm", nmo.glm.GLM())
])

# Test different input combinations
param_grid = {
    "basis__basis": [
        basis_both,     # position + theta
        basis_position, # position only
        basis_theta     # theta only
    ],
}

# Run grid search
gridsearch = GridSearchCV(pipeline, param_grid=param_grid, cv=5)
X = np.column_stack([position.d, theta.d])
gridsearch.fit(X, counts.d)

# Display results
import pandas as pd
results = pd.DataFrame(gridsearch.cv_results_)
print(results[["params", "mean_test_score"]])
```

The results show which inputs contribute to predicting neural activity.
