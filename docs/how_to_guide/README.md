
# How-To Guide

Familiarize with NeMoS modules and learn how to take advantage of the `pynapple` and `scikit-learn` compatibility.

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
<img src="../_static/thumbnails/how_to_guide/plot_02_glm_demo.svg" style="height: 100px", alt="GLM demo."/>
</figure>

```{toctree}
:maxdepth: 2

plot_02_glm_demo.md
```
:::

:::{grid-item-card}

<figure>
<img src="../_static/thumbnails/how_to_guide/plot_03_population_glm.svg" style="height: 100px", alt="Population GLM."/>
</figure>

```{toctree}
:maxdepth: 2

plot_03_population_glm.md
```
:::

:::{grid-item-card}

<figure>
<img src="../_static/thumbnails/how_to_guide/plot_04_batch_glm.svg" style="height: 100px", alt="Batched GLM."/>
</figure>

```{toctree}
:maxdepth: 2

plot_04_batch_glm.md
```
:::

:::{grid-item-card}

<figure>
<img src="../_static/nemos_sklearn.svg" style="height: 100px", alt="NeMoS vs sklearn."/>
</figure>

```{toctree}
:maxdepth: 2

plot_05_transformer_basis.md
```
:::

:::{grid-item-card}

<figure>
<img src="../_static/thumbnails/how_to_guide/plot_06_sklearn_pipeline_cv_demo.svg" style="height: 100px", alt="PyTrees."/>
</figure>

```{toctree}
:maxdepth: 2

plot_06_sklearn_pipeline_cv_demo.md
```

:::

:::{grid-item-card}

<figure>
<img src="../_static/thumbnails/how_to_guide/plot_07_glm_pytree.svg" style="height: 100px", alt="PyTrees."/>
</figure>

```{toctree}
:maxdepth: 2

plot_07_glm_pytree.md
```

:::

:::{grid-item-card}

```{eval-rst}

.. plot:: scripts/glm_predictors.py plot_categorical_var_design_matrix
   :show-source-link: False
   :height: 100px
```

```{toctree}
:maxdepth: 2

categorical_predictors.md
```

:::

:::{grid-item-card}

```{eval-rst}

.. plot:: scripts/glm_predictors.py plot_custom_features
   :show-source-link: False
   :height: 100px
```

```{toctree}
:maxdepth: 2

custom_predictors.md
```

:::

:::{grid-item-card}

<figure>
<img src="../_static/glm_population_scheme.svg" style="height: 100px", alt="Coupled GLM."/>
</figure>

```{toctree}
:maxdepth: 2

raw_history_feature.md
```

:::

::::
