---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# GLM

The theory behind the GLM, and the classes implementing it for a single neuron, a population, and categorical responses of one or several outputs.

Fitting a GLM in NeMoS can be done in a single line of code. For instance, for a Poisson GLM,

```{code-cell}
import nemos as nmo

# Poisson GLM
X, y = nmo.datasets.tuned_poisson()
nmo.glm.GLM().fit(X, y)

```

Observation models can be easily swapped.

```{code-cell}
# Other observation Models
X, y = nmo.datasets.tuned_gaussian()
nmo.glm.GLM("Gaussian").fit(X, y)
```

## Contents

```{toctree}
:maxdepth: 1

Background <intro>
GLM: 1-d output <single_neuron>
PopulationGLM: 2-d output <population>
ClassifierGLM <classifier>
Population ClassifierGLM <population_classifier>
```
