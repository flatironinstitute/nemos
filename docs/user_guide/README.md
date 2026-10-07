# User Guide

The user guide explains the building blocks of NeMoS one at a time: the models, the components they are configured with, and the features they take as input. For complete analyses of real recordings, see the [tutorials](../tutorials/README.md); for answers to specific questions, such as how to fit a dataset that does not fit in memory, see the [how-to guides](../how_to_guide/README.md). The [Example Finder](examples-overview) lists both, filterable by model, signal and topic.

(user-guide-models)=
## Models

The model families and the estimator interface they share.

% The captions below group the left sidebar nav and repeat the section headers, so
% conf.py drops them from this page's body (see _SIDEBAR_ONLY_CAPTION_PAGES).

:::{card}

```{toctree}
:maxdepth: 2
:titlesonly:
:caption: Models

GLM <models/glm/README>
GLM-HMM <models/glm_hmm/README>
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

(user-guide-feature-design)=
## Feature Design

Constructing the design matrix, and composing several inputs into one.

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
