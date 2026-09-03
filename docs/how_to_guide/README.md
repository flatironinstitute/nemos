
# How-To Guide

Task-oriented recipes: each page answers a single "how do I ...?" question. The concepts behind them are covered in the [user guide](../user_guide/README.md).

:::{dropdown} Additional requirements
:color: warning
:icon: alert
To run these guides, you may need to install some additional packages used for plotting and data fetching.
You can install all of the required packages with the following command:
```
pip install nemos[examples]
```
:::

## Fitting Models

::::{grid} 1 2 3 3

:::{grid-item-card}

<figure>
<a href="raw_history_feature.html">
<img src="../_static/glm_population_scheme.svg" style="height: 100px", alt="Coupled GLM."/>
</a>
</figure>

```{toctree}
:maxdepth: 2

raw_history_feature.md
```
:::

:::{grid-item-card}

<figure>
<a href="glm_for_classification.html">
<img src="../_static/thumbnails/how_to_guide/glm_for_classification.svg" style="height: 100px", alt="Confusion Matrix."/>
</a>
</figure>

```{toctree}
:maxdepth: 2

glm_for_classification.md
```
:::

:::{grid-item-card}

```{toctree}
:maxdepth: 2

simulate_coupled_population.md
```
:::

::::

## Feature Engineering

::::{grid} 1 2 3 3

:::{grid-item-card}

```{eval-rst}

.. plot:: scripts/basis_figs.py plot_laguerre_basis
   :show-source-link: False
   :height: 100px
```

```{toctree}
:maxdepth: 2

define_a_custom_basis.md
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

```{toctree}
:maxdepth: 2

pytree_predictors.md
```

:::

::::

## Model Selection

::::{grid} 1 2 3 3

:::{grid-item-card}

<figure>
<img src="../_static/thumbnails/how_to_guide/variable_selection_zero_basis.svg" style="height: 100px", alt="Model Selection."/>
</figure>

```{toctree}
:maxdepth: 2

variable_selection_zero_basis.md
```

:::

:::{grid-item-card}

<figure>
<a href="variable_selection_group_lasso.html">
<img src="../_static/thumbnails/how_to_guide/variable_selection_group_lasso.svg" style="height: 100px", alt="Variable selection."/>
</a>
</figure>

```{toctree}
:maxdepth: 2

variable_selection_group_lasso.md
```

:::

::::

## Performance and Scaling

::::{grid} 1 2 3 3

:::{grid-item-card}

<figure>
<a href="convolve_large_arrays.html">
<img src="../_static/convolve_batching_scheme.svg" style="height: 100px", alt="Batching scheme."/>
</a>
</figure>

```{toctree}
:maxdepth: 2

convolve_large_arrays.md
```

:::

:::{grid-item-card}

```{toctree}
:maxdepth: 2

stochastic_fit.md
```
:::

:::{grid-item-card}

```{toctree}
:maxdepth: 2

custom_dataloader.md
```
:::

:::{grid-item-card}

<figure>
<img src="../_static/thumbnails/how_to_guide/batch_glm_loss_curve.svg" style="height: 100px", alt="Batched GLM."/>
</figure>

```{toctree}
:maxdepth: 2

custom_callbacks_and_termination.md
```
:::

:::{grid-item-card}

```{toctree}
:maxdepth: 2

manual_batching_loop.md
```
:::

:::{grid-item-card}


```{toctree}
:maxdepth: 2

custom_solvers.md
```

:::

::::
