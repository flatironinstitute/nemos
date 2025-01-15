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

# Using bases as scikit-learn transformers

(tansformer-vs-nemos-basis)=
## scikit-learn Transformers and NeMoS Basis

`scikit-learn` is a powerful machine learning library that provides advanced tools for creating data analysis pipelines, from input transformations to model fitting and cross-validation.

All of `scikit-learn`'s machinery relies on strict assumptions about input structure. In particular, all `scikit-learn` 
objects require inputs to be arrays of at most two dimensions, where the first dimension represents the time (or samples) 
axis, and the second dimension represents features. 
While this may feel rigid, it enables transformations to be seamlessly chained together, greatly simplifying the 
process of building stable, complex pipelines.

They can accept arrays or `pynapple` time series data,  which can take any shape as long as the time (or sample) axis is the first of each array. 
Furthermore, `NeMoS` design favours object composability: one can combine bases into [`CompositeBasis`](composing_basis_function) objects to compute complex features, with a user-friendly interface that can accept a separate array/time series for each input type (e.g., an array with the spike counts, an array for the animal's position, etc.).

Both approaches to data transformation are valuable and each has its own advantages. Wouldn't it be great if one could combine the two? Well, this is what NeMoS `TransformerBasis` is for!


## From Basis to TransformerBasis

:::{admonition} Composite Basis
:class: note

To learn more on composite basis, take a look at [this note](composing_basis_function).
:::

With NeMoS, you can easily create a basis which accepts two inputs. Let's assume that we want to process neural activity stored in a 2-dimensional spike count array of shape `(n_samples, n_neurons)` and a second array containing the speed of an animal, with shape `(n_samples,)`.

```{code-cell} ipython3
import numpy as np
import nemos as nmo

# create the arrays
n_samples, n_neurons = 100, 5
counts = np.random.poisson(size=(100, 5))
speed = np.random.normal(size=(100))

# create a composite basis
counts_basis = nmo.basis.RaisedCosineLogConv(5, window_size=10)
speed_basis = nmo.basis.BSplineEval(5)
composite_basis = counts_basis + speed_basis

# compute the features
X = composite_basis.compute_features(counts, speed)

```

### Converting NeMoS `Basis` to a transformer

Now, imagine that we want to use this basis as a step in a `scikit-learn` pipeline. 
In this standard (for NeMoS) form, it would not be possible as the `composite_basis` object requires two inputs. We need to convert it first into a `scikit-learn`-compliant  transformer. This can be achieved through the [`TransformerBasis`](nemos.basis._transformer_basis.TransformerBasis) wrapper class.

Instantiating a [`TransformerBasis`](nemos.basis._transformer_basis.TransformerBasis) can be done either by using the constructor directly or with [`Basis.to_transformer()`](nemos.basis._basis.Basis.to_transformer):


```{code-cell} ipython3
bas = nmo.basis.RaisedCosineLinearConv(5, window_size=5)

# initalize using the constructor
trans_bas = nmo.basis.TransformerBasis(bas)

# equivalent initialization via "to_transformer"
trans_bas = bas.to_transformer()

```

[`TransformerBasis`](nemos.basis._transformer_basis.TransformerBasis) provides convenient access to the underlying [`Basis`](nemos.basis._basis.Basis) object's attributes:


```{code-cell} ipython3
print(bas.n_basis_funcs, trans_bas.n_basis_funcs)
```

We can also set attributes of the underlying [`Basis`](nemos.basis._basis.Basis). Note that -- because [`TransformerBasis`](nemos.basis._transformer_basis.TransformerBasis) is created with a copy of the [`Basis`](nemos.basis._basis.Basis) object passed to it -- this does not change the original [`Basis`](nemos.basis._basis.Basis), nor does changing the original [`Basis`](nemos.basis._basis.Basis) modify the [`TransformerBasis`](nemos.basis._transformer_basis.TransformerBasis) we created:


```{code-cell} ipython3
trans_bas.n_basis_funcs = 10
bas.n_basis_funcs = 100

print(bas.n_basis_funcs, trans_bas.n_basis_funcs)
```

As with any `sckit-learn` transformer, the `TransformerBasis` implements `fit`, a preparation step, `transform`, the actual feature computation, and `fit_transform` which chains `fit` and `transform`. These methods comply with the `scikit-learn` input structure convention, and therefore they all accept a single 2D array.

## Setting up the TransformerBasis

At this point we have an object equipped with the correct methods, so now, all we have to do is concatenate the inputs into a unique array and call `fit_transform`, right? 

```{code-cell} ipython3

# reinstantiate the basis transformer for illustration porpuses
composite_basis = counts_basis + speed_basis
trans_bas = (composite_basis).to_transformer()
# concatenate the inputs
inp = np.concatenate([counts, speed[:, np.newaxis]], axis=1)
print(inp.shape)

try:
    trans_bas.fit_transform(inp)
except RuntimeError as e:
    print(repr(e))
    
```

...Unfortunately, not yet. The problem is that the basis doesn't know which columns of `inp` should be processed by `count_basis` and which by `speed_basis`.

You can provide this information by calling the `set_input_shape` method of the basis. 

This can be called before or after the transformer basis is defined. The method extracts and stores the number of columns for each input. There are multiple ways to call this method:

- It directly accepts the input: `composite_basis.set_input_shape(counts, speed)`.
- If the input is 1D or 2D, it also accepts the number of columns: `composite_basis.set_input_shape(5, 1)`.
- A tuple containing the shapes of all except the first: `composite_basis.set_input_shape((5,), (1,))`.
- A mix of the above methods: `composite_basis.set_input_shape(counts, 1)`.

:::{note}

Note that what `set_input_shapes` requires are the dimensions of the input stimuli, with the exception of the sample 
axis. For example, if the input is a 4D tensor, one needs to provide the last 3 dimensions:

```{code} ipython3
# generate a 4D input
x = np.random.randn(10, 3, 2, 1)

# define and setup the basis
basis = nmo.basis.BSplineEval(5).set_input_shape((3, 2, 1))

X = basis.to_transformer().transform(
    x.reshape(10, -1)  # reshape to 2D
)
```
:::

You can also invert the order of operations and call `to_transform` first and then set the input shapes. 
```{code-cell} ipython3

trans_bas = composite_basis.to_transformer()
trans_bas.set_input_shape(5, 1) 
out = trans_bas.fit_transform(inp)
```

:::{note}

If you define a basis and call `compute_features` on your inputs, internally, it will store its shapes, 
and the `TransformerBasis` will be ready to process without any direct call to `set_input_shape`.
:::

:::{warning}

If for some reason you need to provide an input of different shape to an already set-up transformer, you must reset the 
`TransformerBasis` with `set_input_shape`.
:::

```{code-cell} ipython3

# define inputs with different shapes and concatenate
x, y = np.random.poisson(size=(10, 3)), np.random.randn(10, 2, 3) 
inp2 = np.concatenate([x, y.reshape(10, 6)], axis=1)

trans_bas = composite_basis.to_transformer()
trans_bas.set_input_shape(3, (2, 3)) 
out2 = trans_bas.fit_transform(inp2)
```


### Learn more

If you want to learn more about how to select basis' hyperparameters with `sklearn` pipelining and cross-validation, check out [this how-to guide](sklearn-how-to).

