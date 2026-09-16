# User Guide

NeMoS has two core modules: `basis`, which builds model features from inputs such as position, phase, stimuli or spike counts, and the model classes, which relate those features to a measured response — spike counts, calcium traces, behavioral choices. Those two cover most analyses, and the chapters on them come first.

For more advanced use, NeMoS exposes a set of components — observation models, regularizers, solvers — that you can combine to build your own model. Each component defines an interface, so you can supply your own if the ones implemented do not cover your use case.

The last chapters cover saving a fitted model, hyperparameter search with scikit-learn, and recordings too large to fit in memory.

:::{dropdown} Additional requirements
:color: warning
:icon: alert

To run the code in these pages, you may need to install some additional packages used for plotting and data fetching.
You can install all of the required packages with the following command:
```
pip install nemos[examples]
```

:::

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

## Model components

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

scalability/README.md
```

:::
