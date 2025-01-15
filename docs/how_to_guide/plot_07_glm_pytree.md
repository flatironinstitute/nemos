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

```{code-cell} ipython3
:tags: [hide-input]

%matplotlib inline
import warnings

# Ignore the first specific warning
warnings.filterwarnings(
    "ignore",
    message="plotting functions contained within `_documentation_utils` are intended for nemos's documentation.",
    category=UserWarning,
)

# Ignore the second specific warning
warnings.filterwarnings(
    "ignore",
    message="Ignoring cached namespace 'core'",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=(
        "invalid value encountered in div "
    ),
    category=RuntimeWarning,
)
```

# FeaturePytree example

This small example notebook shows how to use our custom FeaturePytree objects
instead of arrays to represent the design matrix. It will show that these two
representations are equivalent.

This demo will fit the Poisson-GLM to some synthetic data. We will first show
the simple case, with a single neuron receiving some input. We will then show a
two-neuron system, to demonstrate how FeaturePytree can make it easier to
separate examine separate types of inputs.

First, however, let's briefly discuss [`FeaturePytrees`](nemos.pytrees.FeaturePytree).

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import numpy as np

import nemos as nmo

np.random.seed(111)
```

## FeaturePytrees

A FeaturePytree is a custom NeMoS object used to represent design matrices,
GLM coefficients, and other similar variables. It is a simple
[pytree](https://jax.readthedocs.io/en/latest/pytrees.html), a dictionary
with strings as keys and arrays as values. These arrays must all have the
same number of elements along the first dimension, which represents the time
points, but can have different numbers of elements along the other dimensions
(and even different numbers of dimensions).


```{code-cell} ipython3
example_pytree = nmo.pytrees.FeaturePytree(feature_0=np.random.normal(size=(100, 1, 2)),
                                           feature_1=np.random.normal(size=(100, 2)),
                                           feature_2=np.random.normal(size=(100, 5)))
example_pytree
```

FeaturePytrees can be indexed into like dictionary, so we can grab a
single one of their features:


```{code-cell} ipython3
example_pytree['feature_0'].shape
```

We can grab the number of time points by getting the length or using the
`shape` attribute


```{code-cell} ipython3
print(len(example_pytree))
print(example_pytree.shape)
```

We can also jointly index into the FeaturePytree's leaves:


```{code-cell} ipython3
example_pytree[:10]
```

We can add new features after initialization, as long as they have the same
number of time points.


```{code-cell} ipython3
example_pytree['feature_3'] = np.zeros((100, 2, 4))
```

However, if we try to add a new feature with the wrong number of time points,
we'll get an exception:


```{code-cell} ipython3
try:
    example_pytree['feature_4'] = np.zeros((99, 2, 4))
except ValueError as e:
    print(e)
```

Similarly, if we try to add a feature that's not an array:


```{code-cell} ipython3
try:
    example_pytree['feature_4'] = "Strings are very predictive"
except ValueError as e:
    print(e)
```

FeaturePytrees are intended to be used with
[jax.tree_util.tree_map](https://jax.readthedocs.io/en/latest/_autosummary/jax.tree_util.tree_map.html),
a useful function for performing computations on arbitrary pytrees,
preserving their structure.




We can map lambda functions:


```{code-cell} ipython3
mapped = jax.tree_util.tree_map(lambda x: x**2, example_pytree)
print(mapped)
mapped['feature_1']
```

Or functions from jax or numpy that operate on arrays:


```{code-cell} ipython3
mapped = jax.tree_util.tree_map(jnp.exp, example_pytree)
print(mapped)
mapped['feature_1']
```

We can change the dimensionality of our pytree:


```{code-cell} ipython3
mapped = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=-1), example_pytree)
print(mapped)
mapped['feature_1']
```

Or the number of time points:


```{code-cell} ipython3
mapped = jax.tree_util.tree_map(lambda x: x[::10], example_pytree)
print(mapped)
mapped['feature_1']
```

If we map something whose output cannot be a FeaturePytree (because its
values are scalars or non-arrays), we return a dictionary of arrays instead:


```{code-cell} ipython3
print(jax.tree_util.tree_map(jnp.mean, example_pytree))
print(jax.tree_util.tree_map(lambda x: x.shape, example_pytree))
import matplotlib.pyplot as plt
import pynapple as nap

nap.nap_config.suppress_conversion_warnings = True
```

## FeaturePytrees and GLM

These properties make FeaturePytrees useful for representing design matrices
and similar objects for the [`GLM`](nemos.glm.GLM).

First, let's get our dataset and do some initial exploration of it. To do so,
we'll use pynapple to [stream
data](https://pynapple.org/examples/tutorial_pynapple_dandi.html)
from the DANDI archive.

:::{attention}

We need some additional packages for this portion, which you can install
with `pip install dandi pynapple`
:::

```{code-cell} ipython3
io = nmo.fetch.download_dandi_data(
    "000582",
    "sub-11265/sub-11265_ses-07020602_behavior+ecephys.nwb",
)

nwb = nap.NWBFile(io.read(), lazy_loading=False)

print(nwb)
```

This data set has cells that are tuned for head direction and 2d position.
Let's compute some simple tuning curves to see if we can find a cell that
looks tuned for both.


```{code-cell} ipython3
tc, binsxy = nap.compute_2d_tuning_curves(nwb['units'], nwb['SpatialSeriesLED1'].dropna(), 20)
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for i, ax in zip(tc.keys(), axes.flatten()):
    ax.imshow(tc[i], origin="lower", aspect="auto")
    ax.set_title("Unit {}".format(i))
axes[-1,-1].remove()
plt.tight_layout()

# compute head direction.
diff = nwb['SpatialSeriesLED1'].values-nwb['SpatialSeriesLED2'].values
head_dir = np.arctan2(*diff.T)
head_dir = nap.Tsd(nwb['SpatialSeriesLED1'].index, head_dir)

tune_head = nap.compute_1d_tuning_curves(nwb['units'], head_dir.dropna(), 30)

fig, axes = plt.subplots(3, 3, figsize=(9, 9), subplot_kw={'projection': 'polar'})
for i, ax in zip(tune_head.columns, axes.flatten()):
    ax.plot(tune_head.index, tune_head[i])
    ax.set_title("Unit {}".format(i))
axes[-1,-1].remove()
```

```{code-cell} ipython3
:tags: [hide-input]

# save image for thumbnail
from pathlib import Path
import os

root = os.environ.get("READTHEDOCS_OUTPUT")
if root:
   path = Path(root) / "html/_static/thumbnails/how_to_guide"
# if local store in ../_build/html/...
else:
   path = Path("../_build/html/_static/thumbnails/how_to_guide")
 
# make sure the folder exists if run from build
if root or Path("../assets/stylesheets").exists():
   path.mkdir(parents=True, exist_ok=True)

if path.exists():
  fig.savefig(path / "plot_07_glm_pytree.svg")
```


Okay, let's use unit number 7.

Now let's set up our design matrix. First, let's fit the head direction by
itself. Head direction is a circular variable (pi and -pi are adjacent to
each other), so we need to use a basis that has this property as well.
[`CyclicBSplineEval`](nemos.basis.CyclicBSplineEval) is one such basis.

Let's create our basis and then arrange our data properly.


```{code-cell} ipython3
unit_no = 7
spikes = nwb['units'][unit_no]

basis = nmo.basis.CyclicBSplineEval(10, order=5)
x = np.linspace(-np.pi, np.pi, 100)
plt.figure()
plt.plot(x, basis.compute_features(x))

# Find the interval on which head_dir has no NaNs
head_dir = head_dir.dropna()
# Grab the second (of two), since the first one is really short
valid_data= head_dir.time_support.loc[[1]]
head_dir = head_dir.restrict(valid_data)
# Count spikes at the same rate as head direction, over the same epoch
spikes = spikes.count(bin_size=1/head_dir.rate, ep=valid_data)
# the time points for spike are in the middle of these bins (whereas for
# head_dir they're at the ends), so use interpolate to shift head_dir to the
# center.
head_dir = head_dir.interpolate(spikes)

X = nmo.pytrees.FeaturePytree(head_direction=basis.compute_features(head_dir))
```

Now we'll fit our GLM and then see what our head direction tuning looks like:


```{code-cell} ipython3
model = nmo.glm.GLM(regularizer="Ridge", regularizer_strength=0.001)
model.fit(X, spikes)
print(model.coef_['head_direction'])

bs_vis = basis.compute_features(x)
tuning = jnp.einsum('b, tb->t', model.coef_['head_direction'], bs_vis)
plt.figure()
plt.polar(x, tuning)
```

This looks like a smoothed version of our tuning curve, like we'd expect!

For a more direct comparison, we can plot the tuning function based on the model predicted
firing rates with that estimated from the counts.


```{code-cell} ipython3
# predict rates and convert back to pynapple
rates_nap = nap.TsdFrame(t=head_dir.t, d=np.asarray(model.predict(X)))
# compute tuning function
tune_head_model = nap.compute_1d_tuning_curves_continuous(rates_nap, head_dir, 30)
# compare model prediction with data
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'})
ax.plot(tune_head[7], label="counts")
# multiply by the sampling rate for converting to spike/sec.
ax.plot(tune_head_model * rates_nap.rate, label="model")

# Let's compare this to using arrays, to see what it looks like:

model = nmo.glm.GLM()
model.fit(X['head_direction'], spikes)
model.coef_
```

We can see that the solution is identical, as is the way of interacting with
the GLM object.

However, with a single type of feature, it's unclear why exactly this is
helpful. Let's add a feature for the animal's position in space. For this
feature, we need a 2d basis. Let's use some raised cosine bumps and organize
our data similarly.


```{code-cell} ipython3
pos_basis = nmo.basis.RaisedCosineLinearEval(10) * nmo.basis.RaisedCosineLinearEval(10)
spatial_pos = nwb['SpatialSeriesLED1'].restrict(valid_data)

X['spatial_position'] = pos_basis.compute_features(*spatial_pos.values.T)
```

Running the GLM is identical to before, but we can see that our coef_
FeaturePytree now has two separate keys, one for each feature type.


```{code-cell} ipython3
model = nmo.glm.GLM(solver_name="LBFGS")
model.fit(X, spikes)
model.coef_
```

Let's visualize our tuning. Head direction looks pretty much the same (though
the values are slightly different, as we can see when printing out the
coefficients).


```{code-cell} ipython3
bs_vis = basis.compute_features(x)
tuning = jnp.einsum('b,nb->n', model.coef_['head_direction'], bs_vis)
print(model.coef_['head_direction'])
plt.figure()
plt.polar(x, tuning.T)
```

And the spatial tuning again looks like a smoothed version of our earlier
tuning curves.


```{code-cell} ipython3
_, _, pos_bs_vis = pos_basis.evaluate_on_grid(50, 50)
pos_tuning = jnp.einsum('b,ijb->ij', model.coef_['spatial_position'], pos_bs_vis)
plt.figure()
plt.imshow(pos_tuning)
```

We could do all this with matrices as well, but we have to pay attention to
indices in a way that is annoying:


```{code-cell} ipython3
X_mat = nmo.utils.pynapple_concatenate_jax([X['head_direction'], X['spatial_position']], -1)

model = nmo.glm.GLM()
model.fit(X_mat, spikes)
model.coef_[..., :basis.n_basis_funcs]
```
