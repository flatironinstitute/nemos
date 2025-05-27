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

Operations that can be vectorized (parallelized), such as convolutions of multiple arrays, will likely benefit from
GPU acceleration. However, when the size of the convolved arrays is large, the vectorization process may allocate
 pre-allocate too much memory, causing the operation to break or fall-back to the CPU.

NeMoS by default tries to vectorize over all the available dimensions, which is any axis.
