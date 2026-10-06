(developers-example-tags)=
# Example tags

Every executable page under `docs/tutorials/` and `docs/how_to_guide/` includes a `nemos_tags` block in its frontmatter, next to `jupytext` and `kernelspec`.
The lists of examples the landing page links to are generated from it, so a page with no block, or with a value outside the vocabulary below, won't be listed.
```yaml
nemos_tags:
  model: [PopulationGLM]
  observation_model: [Poisson]
  signal: [spike counts]
  topic: [stochastic fit, scalability]
  data: simulated
  description: Fit a population GLM by mini-batch stochastic optimization on data read from an NWB file.
```

| field | values                                                                                         |
| --- |------------------------------------------------------------------------------------------------|
| `model` | `GLM`, `PopulationGLM`, `GLMHMM`, `ClassifierGLM`                                              |
| `observation_model` | `Poisson`, `Gamma`, `Gaussian`, `Bernoulli`, `NegativeBinomial`, `Categorical`                 |
| `signal` | `spike counts`, `calcium imaging`, `lfp`, `behavior choices`, `behavior tracking`, `continuous` |
| `topic` | `feature design`, `variable selection`, `cross-validation`, `regularization`, `stochastic fit`, `scalability`, `functional connectivity`, `simulation`, `custom components`, `receptive fields`, `latent states` |
| `data` | `recorded`, `simulated`                                                                        |
| `description` | free text, one sentence                                                                        |

- A new example **must** carry all six fields, with `model`, `observation_model`, `signal` and `topic` as lists, empty where nothing applies — `convolve_large_arrays.md` fits no model and leaves `model` and `observation_model` empty.
- `description` **must** be a non-empty string; it is shown next to the example's title in the examples table, so it **should** say what the page fits or builds in one sentence.
- A new value **should** be added to this note in the same change, so the vocabulary stays discoverable.

Sphinx exposes the block as `env.metadata[docname]["nemos_tags"]`, where `myst_parser` leaves it as a **JSON string**; the generator has to `json.loads` it rather than index it as a dict.

`write_examples_index` in `conf.py` runs on `build-finished` and writes `_build/html/_static/examples.json`, one entry per tagged page: its tags plus `title` and `url`, the latter relative to the site root.
It is written into the build and not the sources, so it never needs committing.
