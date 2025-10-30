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

In nemos, you can save a model by calling the {py:meth}`~nemos.glm.GLM.save_params` method, which writes a {py:func}`npz file <numpy.savez>`, a NumPy-specific binary format.


```{code-cell}
import nemos as nmo

# define a ridge regularized glm, with a Gradient descent solver
model = nmo.glm.GLM(
    regularizer="Ridge",
    solver_name="GradientDescent"
)

# save
model.save_params("ridge_glm_params.npz")


# load
loaded_model = nmo.load_model("ridge_glm_params.npz")

print("Original Model: \n", model)
print("\nLoaded Model: \n", loaded_model)

```

## Saving and Loading a Fitted Model

The same workflow works for fitted models, meaning the learned coefficients and intercepts are also saved and restored:

```{code-cell}
import numpy as np

# generate some data
np.random.seed(123)
X, weights = np.random.randn(50, 1), 0.1 * np.random.randn(1)
counts = np.random.poisson(np.exp(X @ weights))

# fit and save
model.fit(X, counts)
model.save_params("ridge_glm_params_fitted.npz")

# load
loaded_model = nmo.load_model("ridge_glm_params_fitted.npz")

print("Original coefficient and intercept:", model.coef_, model.intercept_)
print("Loaded coefficient and intercept:", loaded_model.coef_, loaded_model.intercept_)
```

## Inspecting the `npz`

You can inspect the contents of a saved `.npz` file with {py:func}`~nemos.io.inspect_npz`, which displays the stored metadata and parameter keys—useful for debugging (e.g., when loading fails) or verifying saved models.

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

model = nmo.glm.GLM(inverse_link_function=custom_link, regularizer=CustomRegularizer(10))
model.save_params("custom_regularizer_params.npz")

nmo.inspect_npz("custom_regularizer_params.npz")
```

As you can see, the regularizer class is stored as a string, `"{object_class.__module__}.{object_class.__name__}"`. This means that trying to load this model directly will result in an error, because NeMoS doesn’t pickle objects and therefore doesn’t know how to recreate the `CustomRegularizer` automatically.

:::{admonition} Why prevent pickling?
:class: warning

Unpickling typically involves executing code, which can pose a security risk.
A third party could tamper with a pickled file to insert malicious code that runs whenever the object is unpickled.

For a real-world example, see [this discussion](https://news.ycombinator.com/item?id=41901475).
:::

```{code-cell}
:tags: [raises-exception]

loaded_model = nmo.load_model("custom_regularizer_params.npz")
```

As the error explains, you can tell nemos how to load the custom objects by providing a mapping between the saved string and to the callable.

```{code-cell}

mapping = {
    "regularizer": CustomRegularizer,
    "inverse_link_function": custom_link
}
loaded_model = nmo.load_model("custom_regularizer_params.npz", mapping_dict=mapping)
loaded_model
```

:::{admonition} Allowed Mappings
:class: warning

Mapping is allowed **only** for callables (functions) and classes, because these cannot be stored directly without pickling.
Other values (like numbers, strings, or arrays) are always stored directly in the `.npz` and cannot be remapped.

When mapping a custom class, you must pass the **class itself** (e.g., `mapping = {"regularizer": CustomRegularizer}`), not an instance (`CustomRegularizer()`).
Passing an instance would overwrite the saved parameters and could lead to inconsistencies.

:::
