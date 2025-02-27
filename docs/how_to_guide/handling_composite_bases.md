---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Handling Composite Bases

## Structure of Composite Basis

Composite basis, aka objects of type `AdditiveBasis` or `MultiplicativeBasis`, are containers of multiple "atomic" one-dimensional basis, organized in a tree structure. Every time we add or multiplied two bases, they will be stored as attributes of the `AdditiveBasis` or `MultiplicativeBasis` respectively.

```{code-cell} ipython3
import nemos as nmo

# define a composite basis
add = nmo.basis.RaisedCosineLinearEval(5, label="input1") + nmo.basis.BSplineEval(6, label="input2")

# `add` stores the two 1dimensional bases as attributes
print(add)
print(add.basis1)
print(add.basis2)
```

Composing even more, will result in more nesting of attributes.

```{code-cell} ipython3
add = add + nmo.basis.MSplineEval(4, label="input3")
print(add)
print(add.basis1.basis1)
print(add.basis1.basis2)
print(add.basis2)
```

## Retrieving Basis Components and Their Parameters
In principle, nesting makes the process of retrieving or setting the parameters of individual components quite cumbersome.  

```{code-cell} ipython3
# retreive the number of basis funciton for input2 basis
add.basis1.basis2.n_basis_funcs
```

However, if you associated a label to the basis, you can use it to get the corresponding basis element.

```{code-cell} ipython3
add["input2"]
```

And its parameters can be easily accessed.

```{code-cell} ipython3
add["input2"].n_basis_funcs
```

This works for any sub-element, including the one that are composite.

```{code-cell} ipython3
# get input1 + input2
add["(input1 + input2)"]
```

Note that the label of this composite basis is assigned automatically. You can overwrite that with a custom label.

```{code-cell} ipython3
add["(input1 + input2)"].label = "my_custom_label"
add
```

A label can be specified at initialization if the composite basis is defined directly.

```{code-cell} ipython3
nmo.basis.AdditiveBasis(
    nmo.basis.BSplineEval(5),
    nmo.basis.MSplineEval(5), 
    label="my_custom_label"
)
```

And if you are asking yourself what happens when two bases with the same label are composed, well, this results in an error.
This guarantees that the labels are always unique and you can always retrieve a basis using its label.

```{code-cell} ipython3
:tags: [raises-exception]

nmo.basis.BSplineEval(5, label="x") + nmo.basis.MSplineEval(5, label="x")
```

Because we ensure that all basis labels are unique, you can always retrieve a specific basis using its label, even when the composite basis is made up of many individual basis objects.

```{code-cell} ipython3
# add 10 basis
composite_bas = nmo.basis.MSplineEval(4, label="label_0")
for k in range(1, 10):
    composite_bas = composite_bas + nmo.basis.MSplineEval(4, label=f"label_{k}")

# retreive one of them using the label
composite_bas["label_5"]
```

## Get and Set Composite Basis Parameters

When working with composite bases, often times one wants to re-configurate specific components. Again, the easiest way to achieve this is labeling each element and using the label to retrieve the basis.

```{code-cell} ipython3
# get the basis function parameter
print(add["input2"].n_basis_funcs)

# set a new value for the parameter
add["input2"].n_basis_funcs = 8
print(add["input2"].n_basis_funcs)
```

This change is reflected on the composite basis.

```{code-cell} ipython3
# check that the input2 basis has now 8 basis funcs
add
```
Note that if you don't provide a label, basis class name is used to construct the keys. If the same basis is repeated, the key is disambiguated by appending an extra numerical identifier.

```{code-cell} ipython3
nmo.basis.BSplineEval(10) + nmo.basis.BSplineEval(5)
```

### Modifying Basis Parameters with `get_params` and `set_params`
Another way to get and set the basis parameter is via the `get_params` and `set_params` methods. This is how `scikit-learn` interacts with basis objects, and so enables cross-validation.

The `get_params` method returns a dictionary, containing all the parameters. The dictionary keys start with the basis label, followed by a double underscore and the name of the parameter.

```{code-cell} ipython3
add.get_params()
```

Each of the key can be used as keyword argument to the `set_method` which in turns sets one or more of the parameter values.

```{code-cell} ipython3
add.set_params(input3__order=3, input1__bounds=(-1,1))
```

:::{admonition} Grid definition
:class: info

The parameter keys retrieved by `get_params` are the one needed to define a parameter grid when cross-validating your hyper-parameters with scikit-learn. Learn how to cross-validate basis parameters using [pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) with [this notebook](sklearn-how-to). 
:::


As noted above, when labels are not provided, `get_params` retrieves the auto-generated ones.

```{code-cell} ipython3
basis = nmo.basis.BSplineEval(10) + nmo.basis.BSplineEval(5)
basis.get_params()
```

Setting the parameters is still possible, but we recommend to always provide informative labels in order to improve code readability.

```{code-cell} ipython3
basis.set_params(BSplineEval_1__n_basis_funcs=12)
```
