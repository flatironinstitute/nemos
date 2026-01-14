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

```{code-cell} ipython3
:tags: [hide-input]

%matplotlib inline
import warnings
import jax

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

# Ignore convergence (irrelevant for this note)
# For real neural data analyis, increase solver maxiter if warning is raised.
warnings.filterwarnings(
    "ignore",
    message=(
        "The fit did not converge"
    ),
    category=RuntimeWarning,
)

jax.config.update("jax_enable_x64", True)
```

(variable_selection)=
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
neuron = spikes[82]
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
    ("glm", nmo.glm.GLM(solver_name="LBFGS"))
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
X = np.column_stack([position, theta])
gridsearch.fit(X, counts.d)
gridsearch
```

The most predictive encoding model for this neuron includes position only. Below the comparison of the tuning curves.

```{code-cell} ipython3
import matplotlib.pyplot as plt

# Compute and plot tuning curves
tc_position = nap.compute_tuning_curves(
    neuron, position, bins=10, feature_names=["position"]
)
tc_position_model = nap.compute_tuning_curves(
    gridsearch.predict(X) * X.rate, position, bins=10, feature_names=["position"]
)

# Plot tuning curves
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
tc_position.squeeze().plot(ax=ax, linewidth=2, markersize=6, label="true")
tc_position_model.squeeze().plot(ax=ax, linewidth=3, markersize=6, label="model")
ax.set_ylabel('Firing rate (Hz)', fontsize=15)
ax.set_xlabel('Position', fontsize=15)
ax.set_title(f'Unit {tc_position.coords["unit"].values[0]}', fontsize=20)
ax.grid(True, alpha=0.3)
plt.legend(fontsize=15)
plt.tight_layout()
```

```{code-cell} ipython3
:tags: [hide-input]

# save image for thumbnail
from pathlib import Path
import os

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
  fig.savefig(path / "variable_selection_zero_basis.svg")
```
