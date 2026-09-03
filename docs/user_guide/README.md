# User Guide

NeMoS has two core modules: `basis`, which builds model features from inputs such as position, phase, stimuli or spike counts, and the model classes, which relate those features to a measured response — spike counts, calcium traces, behavioral choices. Those two cover most analyses, and the chapters on them come first.

For more advanced use, NeMoS exposes a set of components — observation models, regularizers, solvers — that you can combine to build your own model. Each component defines an interface, so you can supply your own if the ones implemented do not cover your use case.

The last chapters cover recordings too large to fit in memory, hyperparameter search with scikit-learn, and saving a fitted model.

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

```{toctree}
:maxdepth: 2

basis/README.md
```

## Models

The model families and the estimator interface they share.

```{toctree}
:maxdepth: 2

models/glm/README.md
models/glm_hmm/README.md
```

## Model components

Advanced usage: what each component does, how they combine into a model, and the interface to implement your own.

```{toctree}
:maxdepth: 1

observation_models.md
regularizers.md
solvers.md
```

## Scalability

Fitting when the data does not fit in memory.

```{toctree}
:maxdepth: 1

scalability/README.md
```

## scikit-learn compatibility

Every basis and model implements the scikit-learn estimator API, so they drop into the tools built around it.

```{toctree}
:maxdepth: 1

sklearn_compatibility/basis_transformer.md
sklearn_compatibility/pipeline.md
sklearn_compatibility/cross_validation.md
```

## pynapple compatibility

Passing time-aware objects to the basis and the models, and what comes back.

```{toctree}
:maxdepth: 1

pynapple.md
```

## Saving and loading

Writing a model to disk and reading it back.

```{toctree}
:maxdepth: 1

saving_and_loading.md
```
