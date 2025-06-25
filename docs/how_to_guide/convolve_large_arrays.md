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
