(developers-example-tags)=
# Example tags

Every executable page under `docs/tutorials/` and `docs/how_to_guide/` includes a `nemos_tags` block in its frontmatter, next to `jupytext` and `kernelspec`.
The lists of examples the landing page links to are generated from it, so a page with no block, or with a value outside the vocabulary below, won't be listed.
```yaml
nemos_tags:
  model: [PopulationGLM]
  observation_model: [Poisson]
  signal: [spike-counts]
  data: simulated
```

| field | values                                                                                         |
| --- |------------------------------------------------------------------------------------------------|
| `model` | `GLM`, `PopulationGLM`, `GLMHMM`, `ClassifierGLM`                                              |
| `observation_model` | `Poisson`, `Gamma`, `Gaussian`, `Bernoulli`, `NegativeBinomial`, `Categorical`                 |
| `signal` | `spike-counts`, `calcium-imaging`, `lfp`, `behavior-choices`, `behavior-tracking`, `continuous` |
| `data` | `recorded`, `simulated`                                                                        |

- A new example **must** carry all four fields, with `model`, `observation_model` and `signal` as lists, empty where nothing applies — `convolve_large_arrays.md` fits no model and leaves `model` and `observation_model` empty.
- A new value **should** be added to this note in the same change, so the vocabulary stays discoverable.

Sphinx exposes the block as `env.metadata[docname]["nemos_tags"]`, where `myst_parser` leaves it as a **JSON string**; the generator has to `json.loads` it rather than index it as a dict.
