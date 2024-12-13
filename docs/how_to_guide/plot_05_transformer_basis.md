# Converting NeMoS Bases To scikit-learn Transformers

## scikit-learn Transformers and NeMoS Basis

`scikit-learn` is a great machine learning package that provides advanced tooling for creating data analysis pipelines, from input transformations to model fitting and cross-validation.

All of `scikit-learn` machinery relies on very strong assumptions on how one should structure the inputs to each processing step. 
In particular, all `scikit-learn` objects requires inputs in the form of arrays of at most two-dimensions, where the first dimension always represents time (or samples) dimension, and the other features.
This may feel a bit rigid at first, but what this buys you is that any transformation can be chained to any other, greatly simplifying the process of building stable complex pipelines.

In `scikit-learn`, the data transformation steps are performed by object called `transformers`.  


On the other hand, NeMoS basis are powerful feature constructors that allow a high degree of flexibility in terms of the required input structure. 
Depending on the basis type, it can accept one or more input arrays or `pynapple` time series data, each of which can have any shape as long as the time (or sample) axis is the first of each array;
NeMoS design favours object composability, one can combine any two or more bases to compute complex features, and a user-friendly interface can accept a separate array/time series for each input type (e.g., an array with the spike counts, an array for the animal's position, etc.).

Both approaches to data transformations are valuable and have their own advantages. 
Wouldn't it be great if one could combine them? Well, this is what NeMoS `TransformerBasis` are for!


## From Basis to TransformerBasis


With NeMoS, you can easily create a basis accepting two inputs. Let's assume that we want to process the neural activity as a 2-dimensional spike count array of shape `(n_samples, n_neurons)` and a second array with the speed of an animal of shape `(n_samples,)`.

```{code-block} ipython3
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
In this standard (for NeMoS) form, it would not be possible the `composite_basis` object requires two inputs. We need to convert it first into a compliant scikit-learn transformer. This can be achieved through the [`TransformerBasis`](nemos.basis._trans_basis.TransformerBasis) wrapper class.

Instantiating a [`TransformerBasis`](nemos.basis._trans_basis.TransformerBasis) can be done either using by the constructor directly or with [`Basis.to_transformer()`](nemos.basis._basis.Basis.to_transformer):


```{code-cell} ipython3
bas = nmo.basis.RaisedCosineLinearConv(5, window_size=5)

# initalize using the constructor
trans_bas = nmo.basis.TransformerBasis(bas)

# equivalent initialization via "to_transformer"
trans_bas = bas.to_transformer()

```

[`TransformerBasis`](nemos.basis._trans_basis.TransformerBasis) provides convenient access to the underlying [`Basis`](nemos.basis._basis.Basis) object's attributes:


```{code-cell} ipython3
print(bas.n_basis_funcs, trans_bas.n_basis_funcs)
```

We can also set attributes of the underlying [`Basis`](nemos.basis._basis.Basis). Note that -- because [`TransformerBasis`](nemos.basis._trans_basis.TransformerBasis) is created with a copy of the [`Basis`](nemos.basis._basis.Basis) object passed to it -- this does not change the original [`Basis`](nemos.basis._basis.Basis), and neither does changing the original [`Basis`](nemos.basis._basis.Basis) change [`TransformerBasis`](nemos.basis._trans_basis.TransformerBasis) we created:


```{code-cell} ipython3
trans_bas.n_basis_funcs = 10
bas.n_basis_funcs = 100

print(bas.n_basis_funcs, trans_bas.n_basis_funcs)
```

As any `sckit-learn` tansformer, the `TransformerBasis` implements `fit`, a preparation step, `transform`, the actual feature computation, and `fit_transform` which chains `fit` and `transform`. These methods comply with the `scikit-learn` input structure convention, and therefore all accepts a single 2D array.

## Setting up the TransformerBasis

At this point we have an object equipped with the correct methods, so now all we have to do is concatenate the inputs into a unique array and call `fit_transform`, right? 

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

Unfortunately not yet. The problem is that the basis has never interacted with the two separate inputs, and therefore doesn't know which columns of `inp` should be processed by `count_basis` and which by `speed_basis`.

There are several ways in which you can provide this information to the basis. The first one is by calling the method `set_input_shape`. 

This can be called before or after the transformer basis is defined. The method extracts and store the array shapes excluding the sample axis (which won't be affected in the concatenation).

`set_input_shape` accepts directly the inputs,

```{code-cell} ipython3

composite_basis.set_input_shape(counts, speed)
out = composite_basis.to_transformer().fit_transform(inp)
```

If the input is 1D or 2D, the number of columns,
```{code-cell} ipython3

trans_bas = composite_basis.set_input_shape(5, 1).transformer()
out = composite_basis.to_transformer().fit_transform(inp)
```

A tuple containing the shapes of all axis other than the first,
```{code-cell} ipython3

composite_basis.set_input_shape((5,), (1,))
out = composite_basis.to_transformer().fit_transform(inp)
```

Or a mix of the above.
```{code-cell} ipython3

composite_basis.set_input_shape(counts, 1)
out = composite_basis.to_transformer().fit_transform(inp)
```

You can also invert the order and call `to_transform` first and set the input shapes after. 
```{code-cell} ipython3

trans_bas = composite_basis.to_transformer()
trans_bas.set_input_shape(5, 1) 
out = trans_bas.fit_transform(inp)
```

:::{note}

If you define a NeMoS basis and call `compute_features` on your inputs, internally, the basis will store the
input shapes, and the `TransformerBasis` will be ready to process without any direct call to `set_input_shape`.
:::

If for some reason you will need to provide an input of different shape to the transformer, you must setup the 
`TransformerBasis` again.

```{code-cell} ipython3

# define inputs with different shapes and concatenate
x, y = np.random.poisson(size=(10, 3)), np.random.randn(10, 2, 3) 
inp2 = np.concatenate([x, y.reshape(10, 6)], axis=1)

trans_bas = composite_basis.to_transformer()
trans_bas.set_input_shape(3, (2, 3)) 
out2 = trans_bas.fit_transform(inp2)
```


### Learn more

If you want to learn more about basis how to select basis hyperparameters with `sklearn` pipelining and cross-validation, check out [this guide](sklearn-how-to).

