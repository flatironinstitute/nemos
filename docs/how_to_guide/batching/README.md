# Stochastic optimization

Datasets from experiments spanning many hours, or models that couple together many neurons, can produce more data than fits in memory.
In that case we can split the data into chunks called batches and loop over them, updating the model parameters one batch at a time.
Fitting this way -- on batches, or in the extreme case on individual data points -- is called stochastic optimization.

NeMoS provides some functionality to make fitting GLMs this way easier:
- [`GLM.stochastic_fit`](nemos.glm.GLM.stochastic_fit) as a simple interface.
- Built-in data loaders for common data sources.
- Monitoring and convergence criteria using callbacks.
- Straightforward customization through custom data loaders and callbacks.

::::{grid} 1 2 2 2

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

::::
