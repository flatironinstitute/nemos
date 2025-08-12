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

# Fourier Basis

## One-Dimensional Fourier Basis

A one-dimensional Fourier basis is a complex basis whose elements are

$$
a_n(x) \;=\; \cos\!\Bigl(2\pi \frac{n}{P}\,x\Bigr) \;+\; i\,\sin\!\Bigl(2\pi \frac{n}{P}\,x\Bigr),
$$

where $P>0$ is the period, $x$ is the input variable, and $n\in\mathbb{N}$ is the frequency index.

### Fourier Basis in NeMoS

In NeMoS, you can define a `FourierEval` basis object with the following syntax:
```{code-cell} ipython3
from nemos.basis import FourierEval
import numpy as np
import matplotlib.pyplot as plt

# 5 frequencies, from 0 to 4
fourier_1d = FourierEval(frequencies=5)

x = np.linspace(0, 1, 400)
X = fourier_1d.compute_features(x)

print("frequencies:", fourier_1d.masked_frequencies[0].tolist())
print("design matrix shape:", X.shape)  # (n_samples, n_features)
```

The `compute_features` method of the basis returns a **real** design matrix that splits real and imaginary parts into separate columns, first the cosine columns then the sine.

:::{admonition} How we get this shape (the DC term)
:class: info

Let the selected frequencies be the sorted set $\mathcal F=\{n_1<\cdots<n_K\}$.

Each frequency contributes a cosine and a sine, so with $K$ frequencies you’d expect $2K$ columns. This is the case when the 0 frequency - DC term - is not included. If the DC term is included, we have that the corresponding column is null, since $\sin(0)=0$. For this reason, the column is omitted — giving $2K-1$. Summarizing,

- $n_1=0$ (with DC):

  $$
  \bigl[\;1,\,\cos(2\pi \frac{n_2}{P}x),\,\ldots,\,\cos(2\pi \frac{n_K}{P}x),\ \ \sin(2\pi \frac{n_2}{P}x),\,\ldots,\,\sin(2\pi \frac{n_K}{P}x)\;\bigr],
  $$

  total columns $=2K-1$.

- $n_1>0$ (without DC):

  $$
  \bigl[\;\cos(2\pi \frac{n_1}{P}x),\,\ldots,\,\cos(2\pi \frac{n_K}{P}x),\ \ \sin(2\pi \frac{n_1}{P}x),\,\ldots,\,\sin(2\pi \frac{n_K}{P}x)\;\bigr],
  $$

  total columns $=2K$.
:::

:::{admonition} About the period $P$
:class: tip
By default, NeMoS sets the period based on the input range as $P = \max(x) - \min(x)$.
You can override this by setting a fixed period via `bounds`. See the [section](fourier-period) below for details.
:::


```{code-cell} ipython3
# NOTE: here and below, '5' = number of frequencies (K in the math)
print("5 frequencies including the DC term:")
print("frequencies:", fourier_1d.frequencies)
print("output features: 5 * 2 - 1 = ",  fourier_1d.n_basis_funcs)

# `evaluate_on_grid` calls `compute_features` on a grid of point
_, X = fourier_1d.evaluate_on_grid(100)
f, axs = plt.subplots(1, 5, figsize=(10, 2), sharey=True)
for freq in fourier_1d.frequencies[0]:
    axs[int(freq)].set_title(f"frequency = {int(freq)}")
    axs[int(freq)].plot(X[:, int(freq)], label="cosine")
    if freq != 0:
        # to get the corresponding sin column
        # add the num of frequencies (5), minus dc term (1)
        idx_sin = int(freq) + 5 - 1
        axs[int(freq)].plot(X[:, idx_sin], label="sine")
plt.legend()
plt.tight_layout()
plt.show()

```

**Without DC (1..5): same pair**

```{code-cell} ipython3
# drop the DC term (5 frequencies)
fourier_1d = FourierEval(frequencies=(1, 6))
print("\n5 frequencies without the DC term:")
print("frequencies:", fourier_1d.frequencies)
print("output features: 5 * 2 = ",  fourier_1d.n_basis_funcs)

f, axs = plt.subplots(1, 5, figsize=(10, 2), sharey=True)

# `evaluate_on_grid` calls `compute_features` on a grid of point
_, X = fourier_1d.evaluate_on_grid(100)
for freq in fourier_1d.frequencies[0]:
    idx_freq = int(freq) - 1
    axs[idx_freq].set_title(f"frequency = {int(freq)}")
    axs[idx_freq].plot(X[:, idx_freq], label="cosine")
    # to get the corresponding sin column
    # add the num of frequencies (5), no dc term to subtract
    idx_sin = idx_freq + 5
    axs[idx_freq].plot(X[:, idx_sin], label="sine")
plt.legend()
plt.tight_layout()
plt.show()

```

(fourier-period)=
### Fixing the Period of the Basis

When evaluating the basis at some values $\boldsymbol{x} = \{x_1,...,x_t\}$, NeMoS assumes that the period of the basis is $P = \max(\boldsymbol{x}) - \min(\boldsymbol{x})$. The basis element with frequency equal to $n$ will therefore oscillate $n$ times over the range of values covered by $\boldsymbol{x}$.


```{code-cell} ipython3
fourier_1d = FourierEval(frequencies=5)

# generate an input ranging [-2, 2]
x = np.linspace(-2, 2, 100)

# evaluate the basis
X = fourier_1d.compute_features(x)

f, axs = plt.subplots(1, 3, figsize=(10, 3), sharey=True, sharex=True)
for freq in [0, 1, 2]:
    axs[freq].set_title(f"frequency = {freq}")
    axs[freq].plot(x, X[:, freq])
    axs[freq].set_xlabel("x")
    axs[freq].set_ylabel(f"$a_{{{freq}}}(x)$")
plt.tight_layout()
plt.show()
```

To fix a domain for the basis, for example $[0, 2 \pi]$, you can provide the `bounds` parameter.

```{code-cell} ipython3

# fix bounds for the range of the input
fourier_1d.bounds = (0, 2*np.pi)

# generate an input not covering the whole range
x = np.linspace(0, np.pi, 100)

# evaluate the basis
X = fourier_1d.compute_features(x)

f, axs = plt.subplots(1, 3, figsize=(10, 3), sharey=True, sharex=True)
for freq in [0, 1, 2]:
    axs[freq].set_title(f"frequency = {freq}")
    axs[freq].plot(x, X[:, freq])
    axs[freq].set_xlabel("x")
    axs[freq].set_ylabel(f"$a_{{{freq}}}(x)$")
    axs[freq].set_xlim(0, 2 * np.pi)
plt.tight_layout()
plt.show()
```

With `bounds=(0, 2π)` fixed but $\boldsymbol{x} \in [0, \pi]$, each frequency $n$ is defined to complete $n$ cycles over the full domain $[0,2\pi]$. Since we are only sampling **half** the domain, each curve shows only the **first half** of those $n$ cycles (e.g., for $n=2$ you see one full cycle).

The bounds can be provided at initialization as well.

```{code-cell} ipython3
fourier_1d = FourierEval(5, bounds=(0, 2*np.pi))
fourier_1d
```

### Selecting The Frequencies

Frequencies can be provided as:

- An integer $n$, that will result in frequencies from $0$ to $n-1$.
- A range $(n, m)$, that will result in frequencies from $n$ to $m-1$.
- An array of integers.

```{code-cell} ipython3

fourier_1d = FourierEval(frequencies=5)
print("- frequencies=5: ", fourier_1d.frequencies)

fourier_1d = FourierEval(frequencies=(5, 10))
print("- frequencies=(5, 10): ", fourier_1d.frequencies)

fourier_1d = FourierEval(frequencies=np.array([1, 3, 5]))
print("- frequencies=np.array([1, 3, 5]): ", fourier_1d.frequencies)
```

## Multi-Dimensional Fourier Basis

Fourier bases extend to $D$ dimensions. Let $\mathbf{x}=(x_1,\dots,x_D)$, per-axis periods $\mathbf{P}=(P_1,\dots,P_D)$, and multi-index $\mathbf{n}=(n_1,\dots,n_D)\in\mathbb{N}_0^D$.

A $D$-dimensional **basis element** is

$$
a_{\mathbf{n}}(\mathbf{x}) \;=\; \cos\!\left( 2\pi \sum_{d=1}^{D} \frac{n_d}{P_d}\, x_d \right)
\;+\; i\,\sin\!\left( 2\pi \sum_{d=1}^{D} \frac{n_d}{P_d}\, x_d \right).
$$

:::{note}

For simplicity, in the rest of the session we will focus on a 2D example, but everything holds true for a general D-dimensional basis.
:::

### Definition in NeMoS

First, let's specify the notation to the 2D case used in the examples. We can write $\mathbf{x}=(x,y)$, and $\mathbf{n}=(n,m)$; we’ll use $n$ for the $x$-axis frequency and $m$ for the $y$-axis frequency.

Defining a two-dimensional Fourier basis follows the syntax:

```{code-cell} ipython3

# 2D basis with n=5 (x-axis) and m=4 (y-axis) frequencies
fourier_2d = FourierEval(frequencies=[5, 4], ndim=2)
```

The total number of output features is $5\cdot4\cdot2-1=39$, and the DC term corresponds to the pair $\mathbf{n}=\mathbf{0}=(0,0)$.

```{code-cell} ipython3

fourier_2d.n_basis_funcs
```

All the frequency pairs are stored in the `masked_frequencies` array of shape `(ndim, n_frequency_pairs)`.

:::{info}

`masked_frequencies` lists the frequency pairs that are currently active in the basis. If you later restrict the grid of frequencies, this array will update to include only the kept pairs. Details follow in the [frequency selection section](select-fourier-freqs-ndim).
:::

```{code-cell} ipython3

print("first 5 frequency pairs:\n", fourier_2d.masked_frequencies[:, :5])
print("shape of the `masked_frequencies` array:", fourier_2d.masked_frequencies.shape)
```

:::{note}

You can check for the presence of the DC term by assessing if `fourier_2d.masked_frequencies[:, 0]` is all zeros.
:::

### Setting the Periodicities

By default, each axis uses its own input span as the period, reusing the 1D rule per axis:
$P_d=\max(x_d)-\min(x_d)$.

```{code-cell} ipython3
x, y = np.meshgrid(
    np.linspace(-2, 2,100),
    np.linspace(0, 1, 100),
)
X = fourier_2d.compute_features(x.flatten(), y.flatten())
# reshape to match the (100, 100) grid
X = X.reshape(100, 100, fourier_2d.n_basis_funcs)

# select frequencies n=2, m=1
idx = np.where(
    (fourier_2d.masked_frequencies[0] == 2) &
    (fourier_2d.masked_frequencies[1] == 1)
)[0][0]

# plot the output
f, axs = plt.subplots(1, 3, figsize=(10, 3))

# 2-dimensional basis
axs[0].pcolormesh(x, y, X[..., idx], shading='gouraud', cmap='viridis')
axs[0].set_title("two-dimensional basis")

# 1-dimensional projections
axs[1].plot(x[0], X[0, :, idx])
axs[1].set_title("x projection\nfrequency = 2")

axs[2].plot(y[:, 0], X[:, 0, idx])
axs[2].set_title("y projection\nfrequency = 1")
plt.tight_layout()
plt.show()
```

As we can see, the $x$-projection, the basis element with $n=2$ completes two cycles across the sampled $x$ range, and on the $y$ projection, the basis element with $m=1$ completes one cycle.

One can set the period by providing a single ``bounds`` tuple that applies to all dimensions, or one tuple per dimension.

```{code-cell} ipython3

# assign a different domain per dimension
fourier_2d.bounds = [(0, 2*np.pi), (0, np.pi)]

x, y = np.meshgrid(
    np.linspace(0, np.pi,100),  # x spans the half of the domain
    np.linspace(0, np.pi, 100), # y spans the whole domain
)

X = fourier_2d.compute_features(x.flatten(), y.flatten())
# reshape to match the (100, 100) grid
X = X.reshape(100, 100, fourier_2d.n_basis_funcs)

# select frequencies n=2, m=2
idx = np.where(
    (fourier_2d.masked_frequencies[0] == 2) &
    (fourier_2d.masked_frequencies[1] == 2)
)[0][0]

# plot the values
f, axs = plt.subplots(1, 3, figsize=(10, 3))
axs[0].pcolormesh(x, y, X[..., idx], shading='gouraud', cmap='viridis')
axs[0].set(xlim=(0, 2*np.pi), ylim=(0, np.pi))
axs[0].set_title("two-dimensional bases")

axs[1].plot(x[0], X[0, :, idx])
axs[1].set_title("x projection\nfrequency = 2")
# x domain [0, 2 pi]
axs[1].set_xlim(0, 2*np.pi)

axs[2].plot(y[:, 0], X[:, 0, idx])
axs[2].set_title("y projection\nfrequency = 2")
# y domain [0, pi]
axs[2].set_xlim(0, np.pi)
plt.tight_layout()
plt.show()
```

With bounds $[(0, 2\pi), (0, \pi)]$ and samples on $x \in [0, \pi]$, the $x$-projection for $n=2$ shows only the first half of its defined period (one full cycle over $[0,\pi]$). The $y$-projection covers its full domain $[0,\pi]$, so for $m=2$ it shows two cycles.

(select-fourier-freqs-ndim)=
### Selecting The Frequencies

The `frequencies` argument specifies, **per axis**, which integer frequencies to use. In $D$ dimensions, pass a list of $D$ arrays (one per axis); their Cartesian product defines the grid of multi-indices $\mathbf{n}$ (and thus the basis elements), which are listed in `masked_frequencies`.

```{code-cell} ipython3

fourier_2d = FourierEval(frequencies=[np.array([1,2,3]), np.array([4,5])], ndim=2)

print(fourier_2d.masked_frequencies)
```

In these examples we use pairs $(n,m)$ with $n \in N = \{1,2,3\}$ (x-axis frequencies) and $m \in M = \{4,5\}$ (y-axis frequencies).

This defines the basis elements $a_{(n,m)}$ for $(n,m) \in \{(1,4),(1,5),(2,4),(2,5),(3,4),(3,5)\}$.

You can subselect specific pairs by **masking** the 2D frequency grid. The mask can be:

1. A boolean array of shape $(|N|, |M|) = (3, 2)$ (rows = $n$, columns = $m$).
2. A callable predicate `f(n, m) -> True/False`.

#### Mask With a Boolean Array

**Pairs grid**

|     | m=4  | m=5  |
|-----|------|------|
| n=1 | (1,4) | (1,5) |
| n=2 | (2,4) | (2,5) |
| n=3 | (3,4) | (3,5) |

**Example mask selecting $(1,4), (2,4), (2,5)$**

|     | m=4 | m=5 |
|-----|-----|-----|
| n=1 | 1   | 0   |
| n=2 | 1   | 1   |
| n=3 | 0   | 0   |


```{code-cell} ipython3

frequency_mask = np.zeros((3,2))
frequency_mask[:2, 0] = 1
frequency_mask[1, 1] = 1
print("frequency mask")
print(frequency_mask)


fourier_2d.frequency_mask = frequency_mask
print("\nmasked frequencies")
print(fourier_2d.masked_frequencies)
```

#### Mask With a Callable

Alternatively, we can specify complex masking rules by defining a mask function. For example, let's filter for the frequency pairs that lies inside a circle of radius of 4.5.

```{code-cell} ipython3
frequency_mask = lambda x, y: np.sqrt(x**2 + y**2) < 4.5
fourier_2d.frequency_mask = frequency_mask

print("\nmasked frequencies")
print(fourier_2d.masked_frequencies)
```

:::{admonition} More on Masking with Callables

- Write the predicate as `f(n, m)` for 2D. The first argument maps to `masked_frequencies[0]` (x-axis, $n$), the second to `masked_frequencies[1]` (y-axis, $m$). In $D$ dimensions use `f(n1, ..., nD)` in the same row order as `masked_frequencies`.
- NeMoS applies the predicate over the Cartesian product of per-axis frequencies (conceptually `np.meshgrid(..., indexing='ij')`). Treat inputs as NumPy arrays and use elementwise operations.
- Return a boolean grid of shape `(len(n_values), len(m_values))`: `True` keeps $(n,m)$, `False` drops it. In $D$ dimensions, return a boolean tensor with one axis per dimension. .
:::
