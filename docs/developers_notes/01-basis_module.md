# The `basis` Module

## Introduction

The `nemos.basis` module provides objects that allow users to construct and evaluate basis functions of various types. The classes are hierarchically organized as follows:

```
Abstract Class Basis
|
├─ Concrete Subclass AdditiveBasis
│
├─ Concrete Subclass MultiplicativeBasis
│
├─ Abstract Subclass SplineBasis
│   │
│   ├─ Concrete Subclass MSplineBasis
│   │
│   ├─ Concrete Subclass BSplineBasis
│   │
│   └─ Concrete Subclass CyclicBSplineBasis
│
├─ Concrete Subclass RaisedCosineBasisLinear 
│   │
│   └─ Concrete Subclass RaisedCosineBasisLog
│
└─ Concrete Subclass OrthExponentialBasis
```

The super-class `Basis` provides two public methods, [`compute_features`](#the-public-method-evaluate) and [`evaluate_on_grid`](#the-public-method-evaluate_on_grid). These methods perform checks on both the input provided by the user and the output of the evaluation to ensure correctness, and are thus considered "safe". They both make use of the abstract method `__call__` that is specific for each concrete class. See below for more details.

## The Class `nemos.basis.Basis`

### The Public Method `compute_features`

The `compute_features` method checks input consistency and applies the basis function to the inputs. 
`Basis` can operate in two modes defined at initialization: `"eval"` and `"conv"`. When a basis is in mode `"eval"`,
`compute_features` evaluates the basis at the given input samples. When in mode `"conv"`, it will convolve the samples
with a bank of kernels, one per basis function.

It accepts one or more NumPy array or pynapple `Tsd` object as input, and performs the following steps:

1. Checks that the inputs all have the same sample size `M`, and raises a `ValueError` if this is not the case.
2. Checks that the number of inputs matches what the basis being evaluated expects (e.g., one input for a 1-D basis, N inputs for an N-D basis, or the sum of N 1-D bases), and raises a `ValueError` if this is not the case.
3. In `"eval"` mode, calls the `__call__` method on the input, which is the subclass-specific implementation of the basis set evaluation. In `"conv"` mode, generates a filter bank using `evaluate_on_grid` and then applies the convolution to the input with `nemos.convolve.create_convolutional_predictor`.
4. Returns a NumPy array or  pynapple `TsdFrame` of shape `(M, n_basis_funcs)`, with each basis element evaluated at the samples.

!!! note "Multiple epochs"
    Note that the convolution works gracefully with multiple disjoint epochs, when a pynapple time series is used as 
    input.

### The Public Method `evaluate_on_grid`

The `evaluate_on_grid` method evaluates the basis set on a grid of equidistant sample points. The user specifies the input as a series of integers, one for each dimension of the basis function, that indicate the number of sample points in each coordinate of the grid.

This method performs the following steps:

1. Checks that the number of inputs matches what the basis being evaluated expects (e.g., one input for a 1-D basis, N inputs for an N-D basis, or the sum of N 1-D bases), and raises a `ValueError` if this is not the case.
2. Calls `_get_samples` method, which returns equidistant samples over the domain of the basis function. The domain may depend on the type of basis.
3. Calls the `__call__` method.
4. Returns both the sample grid points of shape `(m1, ..., mN)`, and the evaluation output at each grid point of shape `(m1, ..., mN, n_basis_funcs)`, where `mi` is the number of sample points for the i-th axis of the grid.

### Abstract Methods

The `nemos.basis.Basis` class has the following abstract methods, which every concrete subclass must implement:

1. `__call__`: Evaluates a basis over some specified samples.
2. `_check_n_basis_min`: Checks the minimum number of basis functions required. This requirement can be specific to the type of basis.

## Contributors Guidelines

### Implementing Concrete Basis Objects
To write a usable (i.e., concrete, non-abstract) basis object, you

- **Must** inherit the abstract superclass `Basis`
- **Must** define the `__call__` and `_check_n_basis_min` methods with the expected input/output format, see [Code References](../../reference/nemos/basis/) for the specifics.
- **Should not** overwrite the `compute_features` and `evaluate_on_grid` methods inherited from `Basis`.
- **May** inherit any number of abstract intermediate classes (e.g., `SplineBasis`). 

