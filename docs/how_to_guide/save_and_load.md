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

The output file can be loaded without pickling, since do not store directly NeMoS objects, but only their type and the parameters for specifying them. This approach is both more robust and safer than pickling the objects directly.

```{code-cell}

loaded_model = nmo.load_model("ridge_glm_params.npz")
loaded_model
```

## Inspecting the `npz`
You can check the content of the stored `npz` by calling the `nemos.inspect_npz` function.
This will print some environment meta-information, as well as the stored parameters.

```{code-cell}

nmo.inspect_npz("ridge_glm_params.npz")
```


## Save and Load Custom Objects

Advanced users may want to specify custom models while still be able to save and load. For example, one could try out a different inverse link function (non-linearity) or a custom  `Regularizer`.

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

As you can see, the regularizer class is stored as `"{object_class.__module__}.{object_class.__name__}"`. Trying to load this model directly will result in an error, since `NeMoS` doesn't pickle your object and doesn't know how to instantiate the `CustomRegularizer`.

```{code-cell}
:tags: [raises-exception]

loaded_model = nmo.load_model("custom_regularizer_params.npz")
```

You can circumvent this, by providing a mapping to the class.

```{code-cell}

mapping = {"regularizer": CustomRegularizer}
loaded_model = nmo.load_model("custom_regularizer_params.npz", mapping_dict=mapping)
loaded_model
```

Note that if you provide a regularizer instance, it would use it as is, discarding the saved `new_param` value.

```{code-cell}

mapping = {"regularizer": CustomRegularizer(7)}
nmo.load_model("custom_regularizer_params.npz", mapping_dict=mapping)
```
