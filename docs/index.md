(id:_home)=

```{eval-rst}
:html_theme.sidebar_secondary.remove:
```


```{toctree}
:maxdepth: 2
:hidden:

Getting Started <getting_started>
User Guide <user_guide/README>
How-To <how_to_guide/README>
Tutorials <tutorials/README>
API Reference <api/index>
Benchmarking <benchmarking>
Citation Guide <citation>
For Developers <developers_notes/README>
```


# __Neural ModelS__


NeMoS (Neural ModelS) is a statistical modeling framework optimized for systems neuroscience and powered by [JAX](https://jax.readthedocs.io/en/latest/).
It streamlines the process of defining and selecting models, through a collection of easy-to-use methods for feature design.

The core of NeMoS includes GPU-accelerated, well-tested implementations of standard statistical models for systems neuroscience.

::::{grid} auto

:::{grid-item}
```{button-ref} getting_started
:ref-type: doc
:color: primary
:shadow:

Getting Started
```
:::
:::{grid-item}
```{button-ref} user_guide/README
:ref-type: doc
:color: primary
:shadow:

User Guide
```
:::
:::{grid-item}
```{button-ref} how_to_guide/README
:ref-type: doc
:color: primary
:shadow:

How-To Guide
```
:::
:::{grid-item}
```{button-ref} tutorials/README
:ref-type: doc
:color: primary
:shadow:

Tutorials
```
:::
:::{grid-item}
```{button-ref} api/index
:ref-type: doc
:color: primary
:shadow:

API Reference
```
:::
::::


## __Models__

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} __GLM__

```{image} assets/lnp_model_colscheme.svg
:alt: Linear-Nonlinear-Poisson diagram.
:width: 100%
:class: only-light
```

```{image} assets/lnp_model_colscheme_dark.svg
:alt: Linear-Nonlinear-Poisson diagram.
:width: 100%
:class: only-dark
```

:::

:::{grid-item-card} __GLM-HMM__

```{image} assets/glm_hmm_graphical_model.svg
:alt: GLM-HMM graphical model.
:width: 100%
:class: only-light
```

```{image} assets/glm_hmm_graphical_model_dark.svg
:alt: GLM-HMM graphical model.
:width: 100%
:class: only-dark
```

:::

::::

Within the same model class, you can configure:

- __Observation models__: {class}`Poisson <nemos.observation_models.PoissonObservations>`, {class}`NegativeBinomial <nemos.observation_models.NegativeBinomialObservations>`, {class}`Gamma <nemos.observation_models.GammaObservations>`, {class}`Gaussian <nemos.observation_models.GaussianObservations>`, {class}`Bernoulli <nemos.observation_models.BernoulliObservations>`. Categorical responses are the exception, and are fit with {class}`ClassifierGLM <nemos.glm.ClassifierGLM>`, a separate class that fixes {class}`CategoricalObservations <nemos.observation_models.CategoricalObservations>` as its observation model.
- __Regularizers__: {class}`UnRegularized <nemos.regularizer.UnRegularized>`, {class}`Ridge <nemos.regularizer.Ridge>`, {class}`Lasso <nemos.regularizer.Lasso>`, {class}`GroupLasso <nemos.regularizer.GroupLasso>`, {class}`ElasticNet <nemos.regularizer.ElasticNet>`.


### __Examples__

% The model names below become links to the quick example of each model in the
% user guide, once those are written.

::::{grid} 1 2 4 4
:gutter: 3

:::{grid-item-card} __Spike counts__

```{eval-rst}

.. plot:: scripts/catalogue_figs.py plot_counts_thumbnail
   :show-source-link: False
   :height: 100px
   :class: only-light

.. plot:: scripts/catalogue_figs.py plot_counts_thumbnail_dark
   :show-source-link: False
   :height: 100px
   :class: only-dark
```

<div style="text-align: center;">

GLM<br/>GLM-HMM

</div>
:::

:::{grid-item-card} __Continuous signals__

```{eval-rst}

.. plot:: scripts/catalogue_figs.py plot_continuous_thumbnail
   :show-source-link: False
   :height: 100px
   :class: only-light

.. plot:: scripts/catalogue_figs.py plot_continuous_thumbnail_dark
   :show-source-link: False
   :height: 100px
   :class: only-dark
```

<div style="text-align: center;">

GLM<br/>GLM-HMM

</div>
:::

:::{grid-item-card} __Binary outcomes__

```{eval-rst}

.. plot:: scripts/catalogue_figs.py plot_binary_thumbnail
   :show-source-link: False
   :height: 100px
   :class: only-light

.. plot:: scripts/catalogue_figs.py plot_binary_thumbnail_dark
   :show-source-link: False
   :height: 100px
   :class: only-dark
```

<div style="text-align: center;">

GLM<br/>GLM-HMM

</div>
:::

:::{grid-item-card} __Choices and categories__

```{eval-rst}

.. plot:: scripts/catalogue_figs.py plot_choices_thumbnail
   :show-source-link: False
   :height: 100px
   :class: only-light

.. plot:: scripts/catalogue_figs.py plot_choices_thumbnail_dark
   :show-source-link: False
   :height: 100px
   :class: only-dark
```

<div style="text-align: center;">

ClassifierGLM

</div>
:::

::::


## __Building the features__

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} __Building blocks__
:link: user_guide/basis/one_dimensional.html
:link-alt: Building blocks

```{eval-rst}

.. plot:: scripts/catalogue_figs.py plot_zoo_thumbnail
   :show-source-link: False
   :height: 100px
   :class: only-light

.. plot:: scripts/catalogue_figs.py plot_zoo_thumbnail_dark
   :show-source-link: False
   :height: 100px
   :class: only-dark
```

B-splines, raised cosines, Fourier, M-splines, and many more.
:::

:::{grid-item-card} __Multiple predictors__
:link: user_guide/basis/composing.html
:link-alt: Multiple predictors

```{eval-rst}

.. plot:: scripts/catalogue_figs.py plot_addition_thumbnail
   :show-source-link: False
   :height: 100px
   :class: only-light

.. plot:: scripts/catalogue_figs.py plot_addition_thumbnail_dark
   :show-source-link: False
   :height: 100px
   :class: only-dark
```

Add bases to give each input its own block of the design matrix.
:::

:::{grid-item-card} __Higher dimension__
:link: user_guide/basis/composing.html
:link-alt: Higher dimension

```{eval-rst}

.. plot:: scripts/catalogue_figs.py plot_product_thumbnail
   :show-source-link: False
   :height: 100px
   :class: only-light

.. plot:: scripts/catalogue_figs.py plot_product_thumbnail_dark
   :show-source-link: False
   :height: 100px
   :class: only-dark
```

Multiply bases to model the joint effect of two inputs.
:::

::::


<div style="text-align: center;">

__Learning Resources:__ [<span class="iconify" data-icon="mdi:book-open-variant-outline"></span> Neuromatch Academy's Lessons](https://compneuro.neuromatch.io/tutorials/W1D3_GeneralizedLinearModels/student/W1D3_Tutorial1.html) | [<span class="iconify" data-icon="mdi:youtube"></span> Cosyne 2018 Tutorial](https://www.youtube.com/watch?v=NFeGW5ljUoI&t=424s) <br>
__Useful Links:__ [<span class="iconify" data-icon="mdi:chat-question"></span> Getting Help](getting_help.md) | [<span class="iconify" data-icon="mdi:alert-circle-outline"></span> Issue Tracker](https://github.com/flatironinstitute/nemos/issues) | [<span class="iconify" data-icon="mdi:order-bool-ascending-variant"></span> Contributing Guidelines](https://github.com/flatironinstitute/nemos/blob/main/CONTRIBUTING.md)

</div>


## <span class="iconify" data-icon="mdi:scale-balance" style="width: 1em"></span>  __License__

Open source, [licensed under MIT](https://github.com/flatironinstitute/nemos/blob/main/LICENSE).

## <span class="iconify" data-icon="mdi:lead-pencil" style="width: 1em"></span>  __Cite Us__

If you use NeMoS in academic work, please cite the software. See the [](citation-doc) for more details.

## __Support__

This package is supported by:

- The Center for Computational Neuroscience, in the Flatiron Institute of the Simons Foundation.
- The NIH BRAIN Initiative (1RF1MH133778).

```{image} assets/logo_flatiron_white.svg
:alt: Flatiron Center for Computational Neuroscience logo White.
:class: only-dark
:width: 200px
:target: https://www.simonsfoundation.org/flatiron/center-for-computational-neuroscience/
```

```{image} assets/CCN-logo-wText.png
:alt: Flatiron Center for Computational Neuroscience logo.
:class: only-light
:width: 200px
:target: https://www.simonsfoundation.org/flatiron/center-for-computational-neuroscience/
```
