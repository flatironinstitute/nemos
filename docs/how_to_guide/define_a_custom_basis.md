---
jupytext:
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

# How to Define A Custom Basis Class

If you want to design features that are not covered by our collection of basis function, you can design a custom basis class with `nemos.basis.CustomBasis`. The `CustomBasis` can be composed as usual with any other basis.

## Example: Laguerre Polynomials

```{code-cell} ipython3

import jax.numpy as jnp
import numpy as np
from scipy.special import laguerre
import matplotlib.pyplot as plt
import nemos as nmo
from functools import partial

x = jnp.linspace(0, 30, 1000)
c = 1.0
N = 5
P = np.zeros((N, N))
for n in range(N):
    P[n, :(n+1)] = laguerre(n).coef[::-1]
P = jnp.array(P)

def laguerre_poly(poly_coef, decay_rate, x):
    """
    Laguerre polynomial.

    Evaluate a single basis function with polynomial coefficients `p` at
    position `x` with decay time constant `c`.
    """
    exp_decay = jnp.exp(-decay_rate * x/2)
    return exp_decay * jnp.polyval(poly_coef[::-1], decay_rate * x)

funcs = [partial(laguerre_poly, p, c) for p in P]

bas = nmo.basis.CustomBasis(funcs=funcs, label="Laguerre")

features = bas.compute_features(x)

# Plot basis functions.
plt.plot(x, bas.compute_features(x))
plt.show()

# Add two Laguerre Poly
add = bas + nmo.basis.CustomBasis(funcs=funcs, label="Laguerre-2")
print(add.compute_features(x, x).shape)
```

:::{admonition} Python sharp bit
:class: warning

Replacing `functools.partial` with a `lambda` function would not work.

```{code} ipython
funcs = [lambda x: laguerre_poly(p, c, x) for p in P]
```

This will create a list of identical Laguerre polynomial functions. Why? Because p is captured by reference, not by value. When each lambda is called, it uses the value of `p` at that moment — which will be the last value in `P`, for all functions.

In contrast, `functools.partial` evaluates its arguments immediately, so each function correctly captures its own `p` value, avoiding this issue.
:::

## Multi-dimensional Outputs

Custom basis works with multi-dimensional outputs as well. Continuing on the Laguerre polynomial example, let's assume that we want to take advantage of the `JAX` vmap capability for efficiency. We can create a single basis that maps a sample to a 5-dimensional output as follows.

```{code-cell} ipython3
import jax

# vmap_laguarre: R -> R^5
vmap_laguerre = jax.vmap(laguerre_poly, in_axes=(0, None, None), out_axes=1)

# a single function can be provided directly (i.e. not wrapped in a list)
bas_vmap = nmo.basis.CustomBasis(funcs=partial(vmap_laguerre, P, c), label="Laguerre-vmap")

# Plot basis functions.
plt.plot(x, bas_vmap.compute_features(x))
plt.show()
```

## Multi-dimensional Inputs

A custom basis can receive a multi-dimensional input too. As an example, let's write down a basis that acts on image inputs, and compute the dot product of an image with a bank of filter masks.

```{code-cell} ipython3

# generate 100 random noise 50 x 50 images
imgs = np.random.randn(100, 50, 50)

def image_dot_product(img, mask):
    return jnp.sum(img * mask[None], axis=(1,2))

# define masks using a nemos 2D basis
basis_2d = nmo.basis.RaisedCosineLinearEval(8)**2
_, _, masks = basis_2d.evaluate_on_grid(50, 50)
funcs = [partial(image_dot_product, mask=m) for m in masks.T]

# specify the the expected for each sample is 2D
bas_img = nmo.basis.CustomBasis(funcs=funcs, ndim_input=2, label="Image-dot")

print(bas_img.compute_features(imgs).shape)
```
