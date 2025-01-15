(id:_home)=

```{eval-rst}
:html_theme.sidebar_secondary.remove:
```


```{toctree}
:maxdepth: 2
:hidden:

Install <installation>
Quickstart <quickstart>
Background <background/README>
How-To Guide <how_to_guide/README>
Tutorials <tutorials/README>
Getting Help <getting_help>
API Reference <api_reference>
For Developers <developers_notes/README>
```


# __Neural ModelS__


NeMoS (Neural ModelS) is a statistical modeling framework optimized for systems neuroscience and powered by [JAX](https://jax.readthedocs.io/en/latest/). 
It streamlines the process of defining and selecting models, through a collection of easy-to-use methods for feature design.

The core of NeMoS includes GPU-accelerated, well-tested implementations of standard statistical models, currently 
focusing on the Generalized Linear Model (GLM). 

We provide a **Poisson GLM** for analyzing spike counts, and a **Gamma GLM** for calcium or voltage imaging traces.


::::{grid} 1 2 3 3

:::{grid-item-card} <span class="iconify" data-icon="mdi:hammer-wrench"></span> &nbsp; **Installation Instructions**
:link: installation.html
:link-alt: Install
---

Run the following `pip` command in your virtual environment.

```{code-block}

pip install nemos

```

:::

:::{grid-item-card} <span class="iconify" data-icon="mdi:clock-fast"></span> &nbsp; **Getting Started**
:link: quickstart.html
:link-alt: Quickstart

---

New to NeMoS? Get the ball rolling with our quickstart.

:::

:::{grid-item-card} <span class="iconify" data-icon="mdi:book-open-variant-outline"></span> &nbsp; **Background**
:link: background/README.html
:link-alt: Background

---

Refresh your theoretical knowledge before diving into data analysis with our notes.

:::

:::{grid-item-card} <span class="iconify" data-icon="mdi:lightbulb-on-10"></span> &nbsp; **How-to Guide**
:link: how_to_guide/README.html
:link-alt: How-to-Guide

---

Already familiar with the concepts? Learn how you to process and analyze your data with NeMoS.


<div class="card-footer-content">

*Requires familiarity with the theory.*

</div>

:::

:::{grid-item-card} <span class="iconify" data-icon="mdi:brain"></span> &nbsp; **Neural Modeling**
:link: tutorials/README.html
:link-alt: Tutorials

---

Explore fully worked examples to learn how to analyze neural recordings from scratch.

<div class="card-footer-content">

*Requires familiarity with the theory.*

</div>

:::

:::{grid-item-card} <span class="iconify" data-icon="mdi:cog"></span> &nbsp; **API Reference**
:link: api_reference.html
:link-alt: API Reference

---

Access a detailed description of each module and function, including parameters and functionality.

:::

::::


<div style="text-align: center;">

__Learning Resources:__ [<span class="iconify" data-icon="mdi:book-open-variant-outline"></span> Neuromatch Academy's Lessons](https://compneuro.neuromatch.io/tutorials/W1D3_GeneralizedLinearModels/student/W1D3_Tutorial1.html) | [<span class="iconify" data-icon="mdi:youtube"></span> Cosyne 2018 Tutorial](https://www.youtube.com/watch?v=NFeGW5ljUoI&t=424s) <br> 
__Useful Links:__ [<span class="iconify" data-icon="mdi:chat-question"></span> Getting Help](getting_help.md) | [<span class="iconify" data-icon="mdi:alert-circle-outline"></span> Issue Tracker](https://github.com/flatironinstitute/nemos/issues) | [<span class="iconify" data-icon="mdi:order-bool-ascending-variant"></span> Contributing Guidelines](https://github.com/flatironinstitute/nemos/blob/main/CONTRIBUTING.md)

</div>


## <span class="iconify" data-icon="mdi:scale-balance" style="width: 1em"></span>  __License__

Open source, [licensed under MIT](https://github.com/flatironinstitute/nemos/blob/main/LICENSE).


## Support

This package is supported by the Center for Computational Neuroscience, in the Flatiron Institute of the Simons Foundation.  

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
