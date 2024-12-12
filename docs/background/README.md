# Background

These notes aim to provide the essential background knowledge needed to understand the models and data processing techniques implemented in NeMoS.

:::{dropdown} Additional requirements
:color: warning
:icon: alert

To run the tutorials, you may need to install some additional packages used for plotting and data fetching.
You can install all of the required packages with the following command:
```
pip install nemos[examples]
```

:::


::::{grid} 1 2 3 3

:::{grid-item-card}

<figure>
<img src="../_static/lnp_model.svg" style="height: 100px", alt="Linear-Non Linear-Poisson illustration."/>
</figure>


```{toctree}
:maxdepth: 2

plot_00_conceptual_intro.md
```
:::

:::{grid-item-card}

```{eval-rst}

.. plot:: scripts/basis_figs.py plot_raised_cosine_linear
   :show-source-link: False
   :height: 100px
```

```{toctree}
:maxdepth: 3

basis/README.md
```

:::

:::{grid-item-card}

<figure>
<img src="../_static/thumbnails/background/plot_03_1D_convolution.svg" style="height: 100px", alt="One-Dimensional Convolutions."/>
</figure>

```{toctree}
:maxdepth: 2

plot_03_1D_convolution.md
```
:::

::::
