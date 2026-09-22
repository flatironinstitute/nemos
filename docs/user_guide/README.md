# User Guide

NeMoS has two core modules:
- [`basis`](user-guide-feature-design), which builds model features from inputs such as position, phase, stimuli or spike counts
- the [model classes](user-guide-models), which relate those features to a measured response — spike counts, calcium traces, behavioral choices.




(user-guide-feature-design)=
## Feature design

Constructing the design matrix, and composing several inputs into one.

% The captions below group the left sidebar nav and repeat the section headers, so
% conf.py drops them from this page's body (see _SIDEBAR_ONLY_CAPTION_PAGES).

:::{card}

```{toctree}
:maxdepth: 2
:titlesonly:
:caption: Feature design

basis/README.md
```

:::

(user-guide-models)=
## Models

The model families and the estimator interface they share.

:::{card}

```{toctree}
:maxdepth: 2
:titlesonly:
:caption: Models

models/glm/README.md
models/glm_hmm/README.md
```

:::

## Model configurations

Advanced usage: what each component does, how they combine into a model, and the interface to implement your own.

:::{card}

```{toctree}
:maxdepth: 2
:titlesonly:
:caption: Model components

observation_models.md
regularizers.md
solvers.md
```

:::

## Saving and Loading

Writing a model to disk and reading it back.

:::{card}

```{toctree}
:maxdepth: 2
:titlesonly:
:caption: Saving and Loading

saving_and_loading.md
```

:::

## Interactions with other packages

Every basis and model implements the scikit-learn estimator API, so they drop into the tools built around it. Bases and models also take pynapple time-aware objects as inputs and preserve them in their outputs.

:::{card}

```{toctree}
:maxdepth: 2
:titlesonly:
:caption: Interactions with other packages

sklearn_compatibility/basis_transformer.md
sklearn_compatibility/pipeline.md
sklearn_compatibility/cross_validation.md
pynapple.md
```

:::

## Stochastic optimization

Fitting when the data does not fit in memory.

:::{card}

```{toctree}
:maxdepth: 2
:titlesonly:
:caption: Stochastic optimization

scalability/README.md
```

:::
