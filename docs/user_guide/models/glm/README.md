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

## TLDR;

```{code-cell}
import nemos as nmo

# Poisson GLM
X, y = nmo.datasets.gauss_tuned_poisson()
model = nmo.model.GLM().fit(X, y)

# Other observation Models
model = nmo.model.GLM("Gaussian").fit(X, y)
mode
```

```{toctree}
:maxdepth: 1

Background <intro>
GLM: 1-d output <single_neuron>
PopulationGLM: 2-d output <population>
ClassifierGLM <classifier>
Population ClassifierGLM <population_classifier>
```
