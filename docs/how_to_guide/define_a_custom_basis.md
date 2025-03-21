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

If you want to design features that are not covered by our collection of basis function, you can design a custom basis class with `nemos.basis.CustomBasis`.

## Custom 1D basis Example: Laguerre Polynomials

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

def laguerre_poly(x, poly_coef, decay_rate):
    """
    Evaluate a single basis function with polynomial coefficients `p` at position `x` with decay time constant `c`.
    """
    return jnp.exp(-decay_rate * x/2) * jnp.polyval(poly_coef[::-1], decay_rate * x)

funcs = [partial(laguerre_poly, poly_coef=p, decay_rate=c) for p in P]

bas = nmo.basis.CustomBasis(funcs=funcs, label="Laguerre")

features = bas.compute_features(x)

# Plot basis functions.
plt.plot(x, bas.compute_features(x))
plt.show()
```

:::{admonition} Python sharp bit

Replacing `functools.partial` with a `lambda` function would not work. 

```{code} ipython
funcs = [lambda x: laguerre_poly, p, decay_rate=c) for p in P]
```

Will create a list of identical Laguerre polynomials. Why? Because `p` is captured as a reference, not as a value. When the `lambda` funciton is called, the reference is the last `p` in the loop for all the functions. On the other hand, `functools.partial` evaluate its arguments immediately, preventing the issue.
:::

## Custom Basis with multi-dimensional input and output

Custom basis works with multi-dimensional outputs as well. Continuing on the Laguerre polynomial example, let's assume that we want to take advantage of the `JAX` vmap capability for efficiency. We can create a single basis that maps a sample to a 5-dimensional output as follows.

```{code-cell} ipython3
import jax

# vmap_laguarre: R -> R^5
vmap_laguerre = jax.vmap(lambda x, p: laguerre_poly(x, p, c), in_axes=(None, 0), out_axes=1)

bas_vmap = nmo.basis.CustomBasis(funcs=[lambda x: vmap_laguerre(x, P)], label="Laguerre-vmap")

# Plot basis functions.
plt.plot(x, bas_vmap.compute_features(x))
plt.show()
```