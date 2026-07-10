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

# Convolve Large Arrays on the GPU

Operations that can be vectorized—such as convolving multiple arrays—can benefit significantly from GPU acceleration. However, when the input arrays are large, full vectorization may cause excessive memory allocation on the GPU, potentially leading to runtime errors or causing the operation to fall back to CPU execution, see this [related issue](https://github.com/flatironinstitute/nemos/issues/345) for more details.

By default, NeMoS vectorizes convolutions over all dimensions except the sample axis. On large arrays, this default behavior may exceed the GPU’s memory capacity. To mitigate this, you can control the memory footprint by specifying batch sizes for the convolution along the following dimensions:

- **`batch_size_channels`**: batches the operation over time series channels.

- **`batch_size_basis`**: batches over basis kernels.

- **`batch_size_samples`**: performs the convolution in sliding windows over time, with the given sample size per batch.

Use these keyword arguments inside `conv_kwargs` when initializing a convolutional basis to enable batched processing.

<figure markdown>
<!-- note that the src here has an extra ../ compared to other images, necessary when specifying path directly in html -->
<img src="../_static/convolve_batching_scheme.svg" style="width: 100%", alt="Batched dimensions scheme."/>
<figcaption>Schematic of the batched dimensions.</figcaption>
</figure>

:::{note} CPU vs GPU memory allocation

On the CPU, the vectorization process does not result in excessive memory allocation. As a result, specifying batch sizes has little to no effect on the overall memory footprint.

On the GPU, however, specifying smaller batch sizes **significantly reduces** the memory allocated during computation by limiting the size of intermediate tensors.
:::

## Example

```{code-cell} ipython3
import numpy as np
import nemos as nmo

# vectorize over 5 channels and 2 basis
batch_size_dict = dict(
    batch_size_samples=2000,
    batch_size_channels=5,
    batch_size_basis=2
)

# define the arrays
n_samples, n_channels, n_basis, window_size = 10_000, 10, 8, 100

time_series = np.random.randn(n_samples, n_channels)

# define a basis in conv mode sepecifying the batch sizes
basis = nmo.basis.RaisedCosineLogConv(n_basis, window_size, conv_kwargs=batch_size_dict)

# performe the convolution as usual
out = basis.compute_features(time_series)

# note that this works for n-dimensional array (not only 2-dimensional arrays)
# here an example with a 3D array:
out2 = basis.compute_features(
    np.random.randn(n_samples, n_channels, 2)
)
```

## Speeding up convolution on long recordings with `use_fft`

Besides memory, the other lever on large arrays is compute time. By default NeMoS convolves directly (`use_fft=False`), which is the fastest option across the array sizes typical of neural data. For **long kernels applied to long recordings**, computing the convolution in the frequency domain can be much faster. Enable it through the `use_fft` convolution keyword — the only change to the usual syntax:

```{code-cell} ipython3
import numpy as np
import nemos as nmo

n_samples, window_size, n_basis = 500_000, 256, 8
x = np.random.randn(n_samples)

# direct convolution (the default)
basis = nmo.basis.RaisedCosineLogConv(n_basis, window_size)
X_direct = basis.compute_features(x)

# FFT convolution: pass use_fft in conv_kwargs
basis_fft = nmo.basis.RaisedCosineLogConv(
    n_basis, window_size, conv_kwargs={"use_fft": True}
)
X_fft = basis_fft.compute_features(x)

# same design matrix, up to float32 round-off
np.allclose(X_direct, X_fft, atol=1e-4, equal_nan=True)
```

Whether FFT is worth it depends on the kernel length and the number of samples. The plot below times both backends (on CPU) as the recording length grows, for a fixed `window_size=256`. To show the FFT at its best, the sample counts are chosen so the internal transform length (`n_samples + window_size - 1`) is a power of two — the length at which the FFT is fastest.

```{code-cell} ipython3
import time
import matplotlib.pyplot as plt

def median_time(basis, x, repeats=3):
    basis.compute_features(x).block_until_ready()  # warmup / compile
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        basis.compute_features(x).block_until_ready()
        times.append(time.perf_counter() - t0)
    return min(times)

window_size, n_basis = 256, 8
direct = nmo.basis.RaisedCosineLogConv(n_basis, window_size)
fft = nmo.basis.RaisedCosineLogConv(
    n_basis, window_size, conv_kwargs={"use_fft": True}
)

# sample counts giving a power-of-two transform length
samples = [2**k - window_size + 1 for k in range(14, 21)]
t_direct = [median_time(direct, np.random.randn(n)) for n in samples]
t_fft = [median_time(fft, np.random.randn(n)) for n in samples]

fig, ax = plt.subplots()
ax.loglog(samples, t_direct, "-o", label="direct (use_fft=False)")
ax.loglog(samples, t_fft, "-o", label="FFT (use_fft=True)")
ax.set_xlabel("number of samples")
ax.set_ylabel("compute_features time (s)")
ax.set_title(f"direct vs FFT convolution (window_size={window_size})")
ax.legend()
fig.tight_layout()
```

Below roughly $10^5$ samples the direct convolution is faster; beyond that the FFT wins, and by $10^6$ samples it is several times faster. For the short kernels common in GLM analyses (tens of samples) the direct convolution is faster at every size, which is why `use_fft=False` is the default — reach for `use_fft=True` only when you have both a long kernel and a long recording.
```
