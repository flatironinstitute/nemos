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

<figure>
<img src="../_static/thumbnails/background/one_dim_bspline_basis.svg" style="height: 100px", alt="Linear-Non Linear-Poisson illustration."/>
</figure>

```{toctree}
:maxdepth: 2

plot_01_1D_basis_function.md
```
:::

:::{grid-item-card}

<figure>
<img src="../_static/thumbnails/background/multiplicative_basis.svg" style="height: 100px", alt="Linear-Non Linear-Poisson illustration."/>
</figure>

```{toctree}
:maxdepth: 2

plot_02_ND_basis_function.md
```
:::

:::{grid-item-card}

<figure>
<img src="../_static/thumbnails/background/convolutions_valid_mode.svg" style="height: 100px", alt="Linear-Non Linear-Poisson illustration."/>
</figure>

```{toctree}
:maxdepth: 2

plot_03_1D_convolution.md
```
:::

::::
