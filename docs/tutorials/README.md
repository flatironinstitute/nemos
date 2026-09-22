# Tutorials

A gallery of fully worked out tutorials analyzing neural recordings from different brain regions and recording modalities.

:::{dropdown} Additional requirements
:color: warning
:icon: alert
:open:
To run the tutorials, you may need to install some additional packages used for plotting and data fetching.
You can install all of the required packages with the following command:
```
pip install nemos[examples]
```
:::

% Only the first toctree of each group carries the :caption:, so the sidebar shows one
% group header per model family; conf.py keeps captions out of this page's body.

## GLM

Encoding models of intracellular current injection, head direction, grid and place fields, V1 responses and calcium transients.

::::{grid} 1 2 3 3

:::{grid-item-card}

<figure>
<img src="../_static/thumbnails/tutorials/plot_01_current_injection.svg" style="height: 100px", alt="Current Injection."/>
</figure>

```{toctree}
:maxdepth: 2
:caption: GLM

glm/plot_01_current_injection.md
```
:::

:::{grid-item-card}

<figure>
<img src="../_static/thumbnails/tutorials/plot_02_head_direction.svg" style="height: 100px", alt="Head direction."/>
</figure>

```{toctree}
:maxdepth: 2

glm/plot_02_head_direction.md
```
:::

:::{grid-item-card}

<figure>
<img src="../_static/thumbnails/tutorials/plot_03_grid_cells.svg" style="height: 100px", alt="Grid Cells."/>
</figure>

```{toctree}
:maxdepth: 2

glm/plot_03_grid_cells.md
```
:::

:::{grid-item-card}

<figure>
<img src="../_static/thumbnails/tutorials/plot_04_v1_cells.svg" style="height: 100px", alt="V1 cells."/>
</figure>

```{toctree}
:maxdepth: 2

glm/plot_04_v1_cells.md
```
:::

:::{grid-item-card}

<figure>
<img src="../_static/thumbnails/tutorials/plot_05_place_cells.svg" style="height: 100px", alt="Place cells."/>
</figure>

```{toctree}
:maxdepth: 2

glm/plot_05_place_cells.md
```
:::

:::{grid-item-card}

<figure>
<img src="../_static/thumbnails/tutorials/plot_06_calcium_imaging.svg" style="height: 100px", alt="Calcium imaging."/>
</figure>

```{toctree}
:maxdepth: 2

glm/plot_06_calcium_imaging.md
```
:::

::::

## GLM-HMM

Inferring the behavioral states an animal switches between during a decision-making task.

::::{grid} 1 2 3 3

:::{grid-item-card}

<figure>
<img src="../_static/glm_hmm_graphical_model.svg" style="height: 100px", alt="GLM-HMM graphical model."/>
</figure>

```{toctree}
:maxdepth: 2
:caption: GLM-HMM

glm_hmm/plot_07_behavioral_states.md
```
:::

::::
