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

A one dimensional Fourier basis is a complex basis whose elements takes the form,

$$
a_n(x) = \cos (2 \pi \frac{n}{P} x) + i \cdot \sin  (2 \pi \frac{n}{P}x),
$$

with $P$ the period of the basis elements, $x$ an angle value, and $n$ is the frequency
of the basis element.

In NeMoS, you can define a one-dimensional Fourier basis as follows,

```{code-cell} ipython3
from nemos.basis import FourierEval
import numpy as np
import matplotlib.pyplot as plt

# 5 frequencies, form 0 to 4
fourier_1d = FourierEval(frequencies=5)
```

As any other basis, NeMoS `compute_features` method returns a **real** design matrix, with the real and imaginary components - the cosines and sines - of the basis on separate columns. For this reason one may expect to have twice as many output features as the number of frequencies. When 0-th frequency - DC term - is included, the actual number of output features is $2 \times \text{num frequencies} - 1$, where $-1$ is because we drop the sine column corresponding to the DC term which will be a column of zeros, since $\sin(0) = 0$.

```{code-cell} ipython3

print("5 frequencies including the DC term:")
print("frequencies:", fourier_1d.frequencies)
print("number of features: 5 * 2 - 1 = ",  fourier_1d.n_basis_funcs)

_, X = fourier_1d.evaluate_on_grid(100)
f, axs = plt.subplots(1, 5, figsize=(10, 2))
for freq in fourier_1d.frequencies[0]:
    axs[int(freq)].plot(X[:, int(freq)], label="cosine")
    if freq != 0:
        # to get the corresponding sin column
        # add the num of frequencies, minus dc terms
        idx_sin = int(freq) + 5 - 1
        axs[int(freq)].plot(X[:, idx_sin], label="sine")
plt.legend()
plt.tight_layout()
plt.show()
```

If the DC term is not included then the number of features is actually $2 \times \text{num frequencies}$.

```{code-cell} ipython3
# drop the DC term (5 frequencies)
fourier_1d = FourierEval(frequencies=(1, 6))
print("\n5 frequencies without the DC term:")
print("frequencies:", fourier_1d.frequencies)
print("number of features: 5 * 2 = ",  fourier_1d.n_basis_funcs)

f, axs = plt.subplots(1, 5, figsize=(10, 2))
_, X = fourier_1d.evaluate_on_grid(100)
for freq in fourier_1d.frequencies[0]:
    idx_freq = int(freq) - 1
    axs[idx_freq].plot(X[:, idx_freq], label="cosine")
    # to get the corresponding sin column
    # add the num of frequencies, no dc term
    idx_sin = idx_freq + 5
    axs[idx_freq].plot(X[:, idx_sin], label="sine")
plt.legend()
plt.tight_layout()
plt.show()
```

### Setting The Domain of The Basis

When evaluating the basis at some values $\boldsymbol{x} = \{x_1,...,x_t\}$, NeMoS assumes that the domain of the basis is $[\min(\boldsymbol{x}), \max(\boldsymbol{x})]$. The basis element with frequency equal to $n$ will therefore oscillate n-times over the range of values covered by $\boldsymbol{x}$.


```{code-cell} ipython3

# generate an input ranging [-2, 2]
x = np.linspace(-2, 2, 100)

# evaluate the basis
X = fourier_1d.compute_features(x)

f, axs = plt.subplots(1, 3, figsize=(10, 3), sharey=True, sharex=True)
for freq in [0, 1, 2]:
    axs[freq].set_title(f"Frequency = {freq}")
    axs[freq].plot(x, X[:, freq])
    axs[freq].set_xlabel("x")
    axs[freq].set_ylabel(f"$a_{{{freq}}}(x)$")
plt.tight_layout()
plt.show()
```

To fix a domain for the basis, for example $[0, 2 \pi]$, you set provide the `bounds` parameter.

```{code-cell} ipython3

# fix bounds for the range of the input
fourier_1d.bounds = (0, 2*np.pi)

# generate an input not covering the whole range
x = np.linspace(0, np.pi, 100)

# evaluate the basis
X = fourier_1d.compute_features(x)

f, axs = plt.subplots(1, 3, figsize=(10, 3), sharey=True, sharex=True)
for freq in [0, 1, 2]:
    axs[freq].set_title(f"Frequency = {freq}")
    axs[freq].plot(x, X[:, freq])
    axs[freq].set_xlabel("x")
    axs[freq].set_ylabel(f"$a_{{{freq}}}(x)$")
    axs[freq].set_xlim(0, 2 * np.pi)
plt.tight_layout()
plt.show()
```
Bounds can be provided at initialization too, `fourier_1d = FourierEval(5, bounds=(0, 2*np.pi))`.

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

Fourier basis can be extended to multiple dimension. For simplicity, let's focus on the two-dimensional case, but everything discussed below applies to an N-dimensional Fourier basis as well. A two-dimensional basis element $a_{nm}(x,y)$ is defined as follows,

$$
a_{nm}(x,y) = \cos \left(2 \pi (\frac{n}{P_x} x + \frac{m}{P_y} y) \right) + i \cdot \sin \left(2 \pi (\frac{n}{P_x} x + \frac{m}{P_y} y) \right)
$$

Where $P_x$, $P_y$ and $n$, $m$ specifies the periodicity and the frequency of the basis on each axis.

Defining a two-dimensional Fourier basis follows a similar syntax:

```{code-cell} ipython3

# define a 2-dimensional basis with
# n=5 and m=4 frequencies
fourier_2d = FourierEval(frequencies=[5, 4], ndim=2)
```

The total number of basis is $5 \cdot 4 \cdot 2 - 1 = 39$, and the DC terms corresponds to the frequency pair $n = m = 0$.

```{code-cell} ipython3

fourier_2d.n_basis_funcs
```
Note that all the frequency pairs are stored in the `masked_frequencies` array of shape `(ndim, n_frequency_pairs)`.

:::{info}

The reason for the name `masked_frequencies` will become evident in the section about frequency selection with `feature_mask`.
:::

```{code-cell} ipython3

print("first 5 frequency pairs:\n", fourier_2d.masked_frequencies[:, :5])
print("shape of the `masked_frequencies` array:", fourier_2d.masked_frequencies.shape)
```

:::{note}

You can check for the presence of the DC term by assessing if `fourier_2d.masked_frequencies[:, 0]` is all zeros.
:::

### Setting the Domain

As for the one-dimensional case, by default the domain is inferred from the input range.

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

# plot the values
f, axs = plt.subplots(1, 3, figsize=(10, 3))
axs[0].pcolormesh(x, y, X[..., idx], shading='gouraud', cmap='viridis')
axs[0].set_title("two-dimensional bases")

axs[1].plot(x[0], X[0, :, idx])
axs[1].set_xlim(0, 2*np.pi)
axs[1].set_title("x projection\nfrequency = 2")

axs[2].plot(y[:, 0], X[:, 0, idx])
axs[2].set_title("y projection\nfrequency = 1")
plt.tight_layout()
plt.show()
```

And it can be fixed by providing a ``single`` bounds tuple that will be applied to all dimension, or a tuple per dimension.

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

### Selecting The Frequencies

The `frequencies` parameter required at initialization defines the range of frequency spanned by each axis, defining a grid of frequency values. Providing `ndim`-arrays of frequencies defines a grid of frequencies that defines the basis elements.

```{code-cell} ipython3

fourier_2d = FourierEval(frequencies=[np.array([1,2,3]), np.array([4,5])], ndim=2)

print(fourier_2d.masked_frequencies)
```

In the example snippets, we are defining the following basis elements, $a_{nm}$ for $(n, m) \in \{(1, 4), (1, 5), (2, 4), (2, 5), \dots (3, 5)\}$.

It is sub-select specific pairs by masking the grid of frequcnies. The mask can be specified in two ways:

1. As a $n \times m$ boolean array.
2. As a function that maps frequencies into a boolean, i.e. $f: R^{\text{ndim}} \rightarrow \{0, 1\}$, where `ndim` is the dimensionality of the basis.

For example, if our pairs are

(1, 4)   (1, 5)
(2, 4)   (2, 5)
(3, 4)   (2, 5)

If we want to select the following frequency pairs: (1,4), (2, 4), (2, 5), we can define a mask

1        0
1        1
0        0

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

Alternatively, we can specify complex masking rules by defining a mask function. For example, let's filter for the frequency pairs that lies inside a circle of radius of 4.5.

```{code-cell} ipython3
frequency_mask = lambda x, y: np.sqrt(x**2 + y**2) < 4.5
fourier_2d.frequency_mask = frequency_mask

print("\nmasked frequencies")
print(fourier_2d.masked_frequencies)
```
