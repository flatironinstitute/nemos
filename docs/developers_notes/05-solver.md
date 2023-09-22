# The `solver` Module

## Introduction

The `solver` module introduces an archetype class `Solver` which provides the structural components for each concrete sub-class.

Objects of type `Solver` provide methods to define an optimization objective, and instantiate a solver for it.

Solvers are typically optimizers from the `jaxopt` package, but in principle they could be custom optimization routines as long as they respect the `jaxopt` api (i.e., have a `run` and `update` method with the appropriate input/output types).

Each solver object defines a set of allowed optimizers, which in turn depends on the loss function characteristics (smooth vs non-smooth) and/or the optimization type (constrained, un-constrained, batched, etc.).

!!! note
    If we need advanced adaptive optimizers (e.g., Adam, LAMB etc.) in the future, we should consider adding [`Optax`](https://optax.readthedocs.io/en/latest/) as a dependency, which is compatible with `jaxopt`, see [here](https://jaxopt.github.io/stable/_autosummary/jaxopt.OptaxSolver.html#jaxopt.OptaxSolver).

## The Abstract Class `Solver`

The abstract class `Solver` enforces the implementation of the `instantiate_solver` method on any concrete realization of a `Solver` object. `Solver` objects are equipped with a method for instantiating a solver runner, i.e., a function that receives as input the initial parameters, the endogenous and the exogenous variables, and outputs the optimization results.

Additionally, the class provides auxiliary methods for checking that the solver and loss function specifications are valid.

### Abstract Methods

`Solver` objects define the following abstract method:

- **`instantiate_solver`**: Instantiate a solver runner for a provided loss function. The loss function must be a `Callable` from either the `jax` or the `neurostatslib` namespace. In particular, this method prepares the arguments, calls, and returns the output of the `get_runner` public method, see **Public Methods** below.

### Public Methods

- **`get_runner`**: Configure and return a `solver_run` callable. The method accepts two dictionary arguments, `solver_kwargs` and `run_kwargs`, which are meant to hold additional keyword arguments for the instantiation and execution of the solver, respectively. These keyword arguments are prepared by the concrete implementation of `instantiate_solver`, which is solver-type specific.

### Auxiliary Methods

- **`_check_solver`**: This method ensures that the provided solver name is in the list of allowed optimizers for the specific `Solver` object. This is crucial for maintaining consistency and correctness in the solver's operation.

- **`_check_solver_kwargs`**: This method checks if the provided keyword arguments are valid for the specified solver. This helps in catching and preventing potential errors in solver configuration.

- **`_check_is_callable_from_jax`**: This method checks if the provided function is callable and whether it belongs to the `jax` or `neurostatslib` namespace, ensuring compatibility and safety when using jax-based operations.
