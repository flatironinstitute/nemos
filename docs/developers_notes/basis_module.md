# The Basis Module

The `neurostatslib.basis` module provides objects that allow users to construct and evaluate basis functions of various types. The classes are hierarchically organized as follows:

```
Abstract Class Basis
|
├─ Abstract Subclass AdditiveBasis
│
├─ Abstract Subclass MultiplicativeBasis
│
├─ Abstract Subclass SplineBasis
│   │
│   └─ Subclass MSplineBasis
│
├─ Abstract Subclass RaisedCosineBasis
│   │
│   ├─ Subclass RaisedCosineBasisLinear
│   │
│   └─ Subclass RaisedCosineBasisLog
│
└─ Class OrthExponentialBasis
```

We've used abstract classes to ensure that any basis object inheriting from the superclass will implement the abstract methods of the superclass. This guarantees that if the inputs and outputs of those methods conform to the requirements specified by the abstract superclass, any new basis class implementation will be able to be dropped in as replacements for the currently-implented Basis objects (e.g., in `GLM`).

The user only needs to instantiate the non-abstract subclasses located at the bottom of the hierarchy. These classes provide two public methods, `evaluate` and `evaluate_on_grid`, both of which are defined in the superclass `Basis`. These methods perform checks on both the input provided by the user and the output of the evaluation to ensure correctness, and are thus considered "safe."

Additionally, a user can combine multiple objects that are subclasses of Basis, creating a higher-dimensional basis set object. Basis objects can be combined using the `__add__` and `__mul__` methods, which allow for the following syntax:

```
basis_add = basis_a + basis_b # calls basis_a.__add__(basis_b)

basis_add = basis_a * basis_b # calls basis_b.__mul_(basis_b)
```

## The Class `neurostatslib.basis.Basis`

### The Public Method `evaluate`

The `evaluate` method checks input consistency and evaluates the basis function at some sample points. It accepts one or more numpy arrays as input, which represent the sample points at which the basis will be evaluated, and performs the following steps:

1. Checks that the inputs all have the same sample size `N`, and raises a `ValueError` if this is not the case.
2. Checks that the number of inputs matches what the basis being evaluated expects (e.g., one input for a 1-D basis, N inputs for an N-D basis, or the sum of N 1-D bases), and raises a `ValueError` if this is not the case.
3. Checks that the output of the method does not exceed `gb_limit` GB of memory size. This size limit is specified at class initialization.
4. Calls the `_evaluate` method on the input, which is the subclass-specific implementation of the basis set evaluation.
5. Returns a numpy array of dimension `N x number of bases`, with each basis element evaluated at the samples.

### The Public Method `evaluate_on_grid`

The `evaluate_on_grid` method evaluates the basis set on a grid of equidistant sample points. The user specifies the input as a series of integers, one for each dimension of the basis function, that indicate the number of sample points in each coordinate of the grid.

This method performs the following steps:

1. Checks that the number of inputs matches what the basis being evaluated expects (e.g., one input for a 1-D basis, N inputs for an N-D basis, or the sum of N 1-D bases), and raises a `ValueError` if this is not the case.
2. Checks that the output of the method does not exceed `gb_limit` GB of memory size. This size limit is specified at class initialization.
3. Calls the subclass-specific `_get_samples` method, which returns equidistant samples over the domain of the basis function. The domain may depend on the type of basis.
4. Calls the subclass-specific `_evaluate` method.
5. Returns both the sample grid points and the evaluation output at each grid point.

### Abstract Methods

The `neurostatslib.basis.Basis` class has the following abstract methods, which every non-abstract subclass must implement:

1. `_evaluate`: Evaluates a basis over some specified samples.
2. `_get_samples`: Returns a tuple of equidistant samples (as flat numpy arrays) for each dimension of the basis function. These samples are used to form a grid over which the basis is evaluated.
3. `_check_n_basis_min`: Checks the minimum number of basis functions required. This requirement can be specific to the type of basis.


## The `AdditiveBasis` Class

This class extends the abstract `Basis` class and represents a set of basis functions that are combined additively. It overrides the `_evaluate`, `_get_samples`, and `_check_n_basis_min` methods with specific implementations for additive basis functions.

### Overridden Methods

1. `_evaluate`: For an `AdditiveBasis` object, this method should evaluate each basis function at the specified samples and then add the results together.
2. `_get_samples`: This method should return equidistant samples over the domain of each basis function. The samples should then be summed together.
3. `_check_n_basis_min`: This method checks whether the number of basis functions meets the minimum requirement for an additive basis set.

## The `MultiplicativeBasis` Class

This class extends the `Basis` abstract class and represents a set of basis functions that are combined multiplicatively. It overrides the `_evaluate`, `_get_samples`, and `_check_n_basis_min` methods with specific implementations for multiplicative basis functions.

### Overridden Methods

1. `_evaluate`: For a `MultiplicativeBasis` object, this method should evaluate each basis function at the specified samples and then multiply the results together.
2. `_get_samples`: This method should return equidistant samples over the domain of each basis function. The samples should then be multiplied together.
3. `_check_n_basis_min`: This method checks whether the number of basis functions meets the minimum requirement for a multiplicative basis set.

## The `SplineBasis` and `MSplineBasis` Classes

The `SplineBasis` class is an abstract subclass of `Basis` that represents a spline basis function. The `MSplineBasis` class is a concrete subclass of `SplineBasis` that implements a specific type of spline basis, the M-spline.

### Overridden Methods for `MSplineBasis`

1. `_evaluate`: For an `MSplineBasis` object, this method should evaluate the M-spline at the specified samples.
2. `_get_samples`: This method should return equidistant samples over the domain of the M-spline.
3. `_check_n_basis_min`: This method checks whether the number of M-spline basis functions meets the minimum requirement.

## The `RaisedCosineBasis`, `RaisedCosineBasisLinear`, and `RaisedCosineBasisLog` Classes

The `RaisedCosineBasis` class is an abstract subclass of `Basis` that represents a raised cosine basis function. The `RaisedCosineBasisLinear` and `RaisedCosineBasisLog` classes are concrete subclasses of `RaisedCosineBasis` that implement specific types of raised cosine basis functions.

### Overridden Methods for `RaisedCosineBasisLinear` and `RaisedCosineBasisLog`

1. `_evaluate`: For these objects, this method should evaluate the raised cosine function (with either linear or logarithmic spacing) at the specified samples.
2. `_get_samples`: This method should return equidistant samples over the domain of the raised cosine function.
3. `_check_n_basis_min`: This method checks whether the number of raised cosine basis functions meets the minimum requirement.

## The `OrthExponentialBasis` Class

The `OrthExponentialBasis` class extends the `Basis` abstract class and represents an orthogonal exponential basis set. It overrides the `_evaluate`, `_get_samples`, and `_check_n_basis_min` methods with specific implementations for orthogonal exponential basis functions.

### Overridden Methods

1. `_evaluate`: For an `OrthExponentialBasis` object, this method should evaluate each basis function at the specified samples and then combine them orthogonally.
2. `_get_samples`: This method should return equidistant samples over the domain of each basis function. The samples are then combined orthogonally.
3. `_check_n_basis_min`: This method checks whether the number of basis functions meets the minimum requirement for an orthogonal exponential basis set.


## Developer Guidelines

Developers are welcome focus on either develop non-abstract classes for new basis function types
as we as add additional checks at evaluation, as well as improve the documentation and readability
of the code. What we won't advise is to alter the abstract-classes structure, which may affect
the overall module and how each of the objects interacts with each other.

Feel free however to work on any piece of code you fell you can improve. More importantly, remember
to thoroughly test all your classes and functions, and provide clear, detailed comments in your code. 
This will not only help others use your library, but also facilitate future maintenance and development.