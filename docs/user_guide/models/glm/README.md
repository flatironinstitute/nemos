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

(glm-user-guide)=
# Generalized Linear Model (GLM)

## Overview

A GLM predicts a response by passing a linear combination of the input features through an inverse link function, which gives the mean of an observation distribution such as Poisson for spike counts. NeMoS provides four GLM classes. They differ in the number of outputs and in the type of the response.

A [`GLM`](glm-single-neuron) learns one weight vector `w` for one neuron. A [`PopulationGLM`](glm-population) learns a weight matrix `W` with one column per neuron. All the neurons share the same predictors `X`.

```{image} ../../../assets/glm_vs_population.svg
:alt: GLM and PopulationGLM. A GLM maps X through a weight vector w to a single output y. A PopulationGLM maps the same X through a weight matrix W, one column per neuron, to an output Y with one column per neuron.
:class: only-light
```

```{image} ../../../assets/glm_vs_population_dark.svg
:alt: GLM and PopulationGLM. A GLM maps X through a weight vector w to a single output y. A PopulationGLM maps the same X through a weight matrix W, one column per neuron, to an output Y with one column per neuron.
:class: only-dark
```

A `GLM` predicts a count or a continuous value for each sample. A [`ClassifierGLM`](glm-classifier) predicts a discrete label for each sample. It learns one weight column per class. The softmax gives the probability of each class, and the predicted label is the most probable class.

```{image} ../../../assets/classifier_glm.svg
:alt: ClassifierGLM. It maps X through a weight matrix W, one column per class, then a softmax gives the class probabilities P, and the argmax gives one label per sample.
:class: only-light
```

```{image} ../../../assets/classifier_glm_dark.svg
:alt: ClassifierGLM. It maps X through a weight matrix W, one column per class, then a softmax gives the class probabilities P, and the argmax gives one label per sample.
:class: only-dark
```

A [`ClassifierPopulationGLM`](population_classifier.md) combines the two: it predicts one label for each neuron.

## GLM Guide Contents

::::{div} topic

```{toctree}
:maxdepth: 2

Background <intro>
GLM <single_neuron>
PopulationGLM <population>
ClassifierGLM <classifier>
ClassifierPopulationGLM <population_classifier>
```

::::

## Basic Syntax

```{code-cell} ipython3
import nemos as nmo

X, y = nmo.datasets.tuned_poisson()
model = nmo.glm.GLM().fit(X, y)
model.score(X, y)
```

Swap the observation model with a single argument:

```{code-cell} ipython3
X, y = nmo.datasets.tuned_gaussian()
nmo.glm.GLM("Gaussian").fit(X, y)
```

and classify discrete labels with `ClassifierGLM`:

```{code-cell} ipython3
X, y = nmo.datasets.tuned_categorical(n_classes=3)
nmo.glm.ClassifierGLM(classes=3).fit(X, y)
```
