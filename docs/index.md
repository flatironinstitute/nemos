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
Reference <reference/README>
For Developers <developers_notes/README>
```


# __Neural ModelS__


NeMoS (Neural ModelS) is a statistical modeling framework for systems neuroscience, powered by [JAX](https://jax.readthedocs.io/en/latest/).
At its core are GPU-accelerated, well-tested implementations of the standard models. They follow [scikit-learn's API](https://scikit-learn.org/stable/), so they work with its cross-validation and model-selection tools, and they take data as numpy arrays or as [pynapple](https://pynapple.org) objects.

::::{grid} auto
:class-container: landing-buttons

:::{grid-item}
```{button-ref} installation
:ref-type: doc
:color: primary
:shadow:

Install
```
:::
:::{grid-item}
```{button-ref} quickstart
:ref-type: doc
:color: primary
:shadow:

Quickstart
```
:::
:::{grid-item}
```{button-ref} reference/benchmarking
:ref-type: doc
:color: primary
:shadow:

Benchmarking
```
:::
:::{grid-item}
```{button-ref} reference/citation
:ref-type: doc
:color: primary
:shadow:

Citation Guide
```
:::
::::


::::::{grid} 1 1 2 2
:gutter: 3
:class-container: landing-band landing-captioned

:::::{grid-item}
:columns: 12 12 6 6

```{rubric} Models
:class: band-title
```
:::::

:::::{grid-item}
:columns: 12 12 6 6

```{rubric} Building the features
:class: band-title
```
:::::

:::::{grid-item-card} __GLM__
:columns: 12 12 6 6
:link: user_guide/models/glm/README.html
:link-alt: GLM

```{image} assets/lnp_model_colscheme.svg
:alt: Linear-Nonlinear-Poisson diagram.
:class: only-light
```

```{image} assets/lnp_model_colscheme_dark.svg
:alt: Linear-Nonlinear-Poisson diagram.
:class: only-dark
```


Encoding models for single neurons and populations.
:::::

:::::{grid-item-card} __Basis functions__
:columns: 12 12 6 6
:class-card: inset-plot
:link: user_guide/basis/one_dimensional.html
:link-alt: Basis functions

```{eval-rst}

.. plot:: scripts/catalogue_figs.py plot_basis_scheme_thumbnail
   :show-source-link: False
   :class: only-light

.. plot:: scripts/catalogue_figs.py plot_basis_scheme_thumbnail_dark
   :show-source-link: False
   :class: only-dark
```

B-splines, raised cosines, Fourier, and many more.
:::::

:::::{grid-item-card} __GLM-HMM__
:columns: 12 12 6 6
:link: user_guide/models/glm_hmm/README.html
:link-alt: GLM-HMM

```{image} assets/glm_hmm_graphical_model.svg
:alt: GLM-HMM graphical model.
:class: only-light
```

```{image} assets/glm_hmm_graphical_model_dark.svg
:alt: GLM-HMM graphical model.
:class: only-dark
```


Encoding models for non-stationary responses.
:::::

:::::{grid-item}
:columns: 12 12 6 6

% The pair sits in a grid of its own, so the space between the two of them
% stays tight while the gap down the middle of the band is wide.

::::{grid} 1 2 2 2
:gutter: 3
:class-container: landing-band

:::{grid-item-card} __Basis addition__
:link: user_guide/basis/composing.html
:link-alt: Multiple predictors

```{eval-rst}

.. plot:: scripts/catalogue_figs.py plot_addition_thumbnail
   :show-source-link: False
   :height: 125px
   :class: only-light

.. plot:: scripts/catalogue_figs.py plot_addition_thumbnail_dark
   :show-source-link: False
   :height: 125px
   :class: only-dark
```

One block per input.
:::

:::{grid-item-card} __Basis multiplication__
:link: user_guide/basis/composing.html
:link-alt: Higher dimension

```{eval-rst}

.. plot:: scripts/catalogue_figs.py plot_product_thumbnail
   :show-source-link: False
   :height: 125px
   :class: only-light

.. plot:: scripts/catalogue_figs.py plot_product_thumbnail_dark
   :show-source-link: False
   :height: 125px
   :class: only-dark
```

Joint effects of two inputs.
:::

::::

:::::

::::::


## __Examples__

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


## __Model components__

```{rubric} Observation models
```

The observation model can be swapped for any of these, with one exception: categorical responses, which are fit with {class}`ClassifierGLM <nemos.glm.ClassifierGLM>`, a separate class that fixes {class}`CategoricalObservations <nemos.observation_models.CategoricalObservations>`.


% Six across, so each set is one row of the same grid; the four regularizers
% take the first four columns of theirs.

::::{grid} 2 3 6 6
:gutter: 2
:class-container: landing-band

:::{grid-item-card} `Poisson`
:class-card: catalogue-card
:link: api/generated/observation_models/nemos.observation_models.PoissonObservations
:link-type: doc
:link-alt: Poisson

```{eval-rst}

.. plot:: scripts/catalogue_figs.py plot_poisson_thumbnail
   :show-source-link: False
   :class: only-light

.. plot:: scripts/catalogue_figs.py plot_poisson_thumbnail_dark
   :show-source-link: False
   :class: only-dark
```

:::
:::{grid-item-card} `NegativeBinomial`
:class-card: catalogue-card
:link: api/generated/observation_models/nemos.observation_models.NegativeBinomialObservations
:link-type: doc
:link-alt: NegativeBinomial

```{eval-rst}

.. plot:: scripts/catalogue_figs.py plot_neg_binomial_thumbnail
   :show-source-link: False
   :class: only-light

.. plot:: scripts/catalogue_figs.py plot_neg_binomial_thumbnail_dark
   :show-source-link: False
   :class: only-dark
```

:::
:::{grid-item-card} `Gamma`
:class-card: catalogue-card
:link: api/generated/observation_models/nemos.observation_models.GammaObservations
:link-type: doc
:link-alt: Gamma

```{eval-rst}

.. plot:: scripts/catalogue_figs.py plot_gamma_thumbnail
   :show-source-link: False
   :class: only-light

.. plot:: scripts/catalogue_figs.py plot_gamma_thumbnail_dark
   :show-source-link: False
   :class: only-dark
```

:::
:::{grid-item-card} `Gaussian`
:class-card: catalogue-card
:link: api/generated/observation_models/nemos.observation_models.GaussianObservations
:link-type: doc
:link-alt: Gaussian

```{eval-rst}

.. plot:: scripts/catalogue_figs.py plot_gaussian_thumbnail
   :show-source-link: False
   :class: only-light

.. plot:: scripts/catalogue_figs.py plot_gaussian_thumbnail_dark
   :show-source-link: False
   :class: only-dark
```

:::
:::{grid-item-card} `Bernoulli`
:class-card: catalogue-card
:link: api/generated/observation_models/nemos.observation_models.BernoulliObservations
:link-type: doc
:link-alt: Bernoulli

```{eval-rst}

.. plot:: scripts/catalogue_figs.py plot_bernoulli_thumbnail
   :show-source-link: False
   :class: only-light

.. plot:: scripts/catalogue_figs.py plot_bernoulli_thumbnail_dark
   :show-source-link: False
   :class: only-dark
```

:::
:::{grid-item-card} `Categorical`
:class-card: catalogue-card
:link: api/generated/observation_models/nemos.observation_models.CategoricalObservations
:link-type: doc
:link-alt: Categorical

```{eval-rst}

.. plot:: scripts/catalogue_figs.py plot_categorical_thumbnail
   :show-source-link: False
   :class: only-light

.. plot:: scripts/catalogue_figs.py plot_categorical_thumbnail_dark
   :show-source-link: False
   :class: only-dark
```

:::

::::


```{rubric} Regularizers
```

Swapping one for another is a single argument: `regularizer="Ridge"`, with a `regularizer_strength` to set how hard it bites. {class}`UnRegularized <nemos.regularizer.UnRegularized>` is the default, and each of the others is drawn below as the set of coefficients it admits.


::::{grid} 2 3 6 6
:gutter: 2
:class-container: landing-band

:::{grid-item-card} `Ridge`
:class-card: catalogue-card
:link: api/generated/regularizer/nemos.regularizer.Ridge
:link-type: doc
:link-alt: Ridge

```{eval-rst}

.. plot:: scripts/catalogue_figs.py plot_ridge_thumbnail
   :show-source-link: False
   :class: only-light

.. plot:: scripts/catalogue_figs.py plot_ridge_thumbnail_dark
   :show-source-link: False
   :class: only-dark
```

:::
:::{grid-item-card} `Lasso`
:class-card: catalogue-card
:link: api/generated/regularizer/nemos.regularizer.Lasso
:link-type: doc
:link-alt: Lasso

```{eval-rst}

.. plot:: scripts/catalogue_figs.py plot_lasso_thumbnail
   :show-source-link: False
   :class: only-light

.. plot:: scripts/catalogue_figs.py plot_lasso_thumbnail_dark
   :show-source-link: False
   :class: only-dark
```

:::
:::{grid-item-card} `GroupLasso`
:class-card: catalogue-card
:link: api/generated/regularizer/nemos.regularizer.GroupLasso
:link-type: doc
:link-alt: GroupLasso

```{eval-rst}

.. plot:: scripts/catalogue_figs.py plot_group_lasso_thumbnail
   :show-source-link: False
   :class: only-light

.. plot:: scripts/catalogue_figs.py plot_group_lasso_thumbnail_dark
   :show-source-link: False
   :class: only-dark
```

:::
:::{grid-item-card} `ElasticNet`
:class-card: catalogue-card
:link: api/generated/regularizer/nemos.regularizer.ElasticNet
:link-type: doc
:link-alt: ElasticNet

```{eval-rst}

.. plot:: scripts/catalogue_figs.py plot_elastic_net_thumbnail
   :show-source-link: False
   :class: only-light

.. plot:: scripts/catalogue_figs.py plot_elastic_net_thumbnail_dark
   :show-source-link: False
   :class: only-dark
```

:::

::::


<div style="text-align: center;">

__Learning Resources:__ [<span class="iconify" data-icon="mdi:book-open-variant-outline"></span> Neuromatch Academy's Lessons](https://compneuro.neuromatch.io/tutorials/W1D3_GeneralizedLinearModels/student/W1D3_Tutorial1.html) | [<span class="iconify" data-icon="mdi:youtube"></span> Cosyne 2018 Tutorial](https://www.youtube.com/watch?v=NFeGW5ljUoI&t=424s) <br>
__Useful Links:__ [<span class="iconify" data-icon="mdi:chat-question"></span> Getting Help](getting-help) | [<span class="iconify" data-icon="mdi:alert-circle-outline"></span> Issue Tracker](https://github.com/flatironinstitute/nemos/issues) | [<span class="iconify" data-icon="mdi:order-bool-ascending-variant"></span> Contributing Guidelines](https://github.com/flatironinstitute/nemos/blob/main/CONTRIBUTING.md)

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
