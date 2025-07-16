---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Saving and Loading

## Saving and Loading a Model

In nemos, you can save a model by calling the `save_params` method, which writes `npz` file, a NumPy specific binary.

```{code-cell}
import nemos as nmo

# define a ridge regularized glm, with a Gradient descent solver
model = nmo.glm.GLM(
    regularizer="Ridge",
    solver_name="GradientDescent"
)

# save
model.save_params("ridge_glm_params.npz")
model
```

The output file is loaded without pickling, because NeMoS does not store objects directly. Instead, it stores only a string representing the object’s class and its parameters. This approach is both more robust and safer than pickling the objects themselves.

```{code-cell}

loaded_model = nmo.load_model("ridge_glm_params.npz")
loaded_model
```

## Inspecting the `npz`
You can check the content of the stored `npz` by calling the `nemos.inspect_npz` function.
The `inspect_npz` function shows the saved object’s metadata and parameter keys:

```{code-cell}

nmo.inspect_npz("ridge_glm_params.npz")
```


## Save and Load Custom Objects

Advanced users may want to specify custom models and still be able to save and load. For example, one could try a different inverse link function (non-linearity) or a custom  `Regularizer`.
```{code-cell}

def custom_link(x):
    return x**2

class CustomRegularizer(nmo.regularizer.Ridge):
    def __init__(self, new_param):
        self.new_param = new_param

obs_model = nmo.observation_models.PoissonObservations(custom_link)
model = nmo.glm.GLM(regularizer=CustomRegularizer(10))
model.save_params("custom_regularizer_params.npz")

nmo.inspect_npz("custom_regularizer_params.npz")
```

As you can see, the regularizer class is stored as a string, `"{object_class.__module__}.{object_class.__name__}"`. This means that trying to load this model directly will result in an error, because NeMoS doesn’t pickle objects and therefore doesn’t know how to recreate the `CustomRegularizer` automatically.

```{code-cell}
:tags: [raises-exception]

loaded_model = nmo.load_model("custom_regularizer_params.npz")
```

You can circumvent this error, by providing a mapping to the class and to the callable.

```{code-cell}

mapping = {
    "regularizer": CustomRegularizer,
    "observation_model__inverse_link_function": custom_link
}
loaded_model = nmo.load_model("custom_regularizer_params.npz", mapping_dict=mapping)
loaded_model
```
::: {tip}

Always keep your custom classes and functions in importable modules if you plan to load saved models later. This makes it easier to share models with collaborators or run them on different machines.
:::

:::{admonition} Allowed Mappings
:class: warning

Mapping is allowed **only** for callables (functions) and classes, because these cannot be stored directly without pickling.
Other values (like numbers, strings, or arrays) are always stored directly in the `.npz` and cannot be remapped.

⚠️ When mapping a custom class, you must provide the class **type**, not an instance (e.g., `mapping = {"regularizer": CustomRegularizer}` is valid, but `CustomRegularizer()` is not).
Providing an instance would overwrite the saved parameters, which could lead to inconsistencies.
:::
