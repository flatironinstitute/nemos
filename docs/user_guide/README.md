# User Guide

NeMoS has two core modules:
- [`basis`](user-guide-feature-design), which builds model features from inputs such as position, phase, stimuli or spike counts
- the [model classes](user-guide-models), which relate those features to a measured response — spike counts, calcium traces, behavioral choices.




(user-guide-feature-design)=
## Feature Design

Constructing the design matrix, and composing several inputs into one.

% The captions below group the left sidebar nav and repeat the section headers, so
% conf.py drops them from this page's body (see _SIDEBAR_ONLY_CAPTION_PAGES).

:::{card}

```{toctree}
:maxdepth: 2
:titlesonly:
:caption: Feature Design

Background <basis/README>
Pynapple Compatibility <pynapple>
Basis types: eval vs conv <basis/eval_vs_conv>
Handling disjoint epochs <basis/disjoint_epochs>
Multiple predictors: basis addition <basis/addition>
Multi-dim predictors: basis multiplication <basis/multiplication>
Advanced <basis/advanced>
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

GLM <models/glm/README>
```

:::

## Model components

The observation models, regularizers and solvers a model is configured with.

:::{card}

```{toctree}
:maxdepth: 2
:titlesonly:
:caption: Model components

Observation models <observation_models>
Regularizers <regularizers>
Solvers <solvers>
```

:::

## Model selection and pipelining with sklearn

Bases as transformers, pipelines of bases and models, and searching their hyperparameters.

:::{card}

```{toctree}
:maxdepth: 2
:titlesonly:
:caption: Model selection and pipelining with sklearn

Transformers <sklearn_compatibility/basis_transformer>
Pipeline <sklearn_compatibility/pipeline>
Grid searches <sklearn_compatibility/cross_validation>
```

:::

## Scalability & Performance

Fitting when the data does not fit in memory, and how fast each solver fits.

:::{card}

```{toctree}
:maxdepth: 2
:titlesonly:
:caption: Scalability & Performance

Stochastic fit <scalability/README>
Custom update <scalability/custom_update>
```

:::
