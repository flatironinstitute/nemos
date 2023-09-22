# The `sovler` Module

## Indtroduction

The `solver` module introduces an archetype class `Sover` which provides the structural components of each concrete sub-classes.

Objects of type `Solver` provide methods to define an optimization objective, and instantiate a solver for it. 

Solvers are typically optimizers from the `jaxopt` package, but in principle they could be custom optimization routines as long as they respect the `jaxopt` api (i.e. have a `run` and `update` method with the appropriate input/output types).

Each solver object defines a set of allowed optimizers, which in turn depends on the loss function characteristics (smooth vs non-smooth) and/or the optimization type (constrained, un-constrained, batched, etc.).

!!! note
    If we will need advanced adaptive optimizers (e.g. Adam, LAMB etc.) in the future, we should consider adding [`Optax`](https://optax.readthedocs.io/en/latest/) as a dependency, which is compatible with `jaxopt`, see [here](`https://jaxopt.github.io/stable/_autosummary/jaxopt.OptaxSolver.html#jaxopt.OptaxSolver`).

## The Abstract Class `Solver`

The abstract class  `Solver` enforces the implementation of the `instantiate_solver` method on any concrete realization of a `Solver` object. `Solver` object are equipped with a method for instantiating a solver runner, i.e. a function that receives as input the initial parameters, the endogenous and the exogenous variable and outputs the optimization results.

Additionally, the class provides auxiliary methods for checking that the solver and loss function specifications are valid.

### Abstract Methods

`Solver` objects defines the following abstract method:

- **`instantiate_solver`**: Instantiate a solver runner for a provided a loss function. The loss function must be a `Callable` from either the `jax` or the `neurostatslib` namespace. In particular, this method prepares the arguments, calls, and returns the output of the `get_runner` public method, see **Public Methods** below. 

### Public Methods

- **`get_runner`**: Configure and return a `solver_run` callable. The method accepts two dictionary arguments, `solver_kwargs` and `run_kwargs`, which are meant to hold additional keyword arguments for the instantiation and execution of the solver, respectively. These keyword arguments are prepared by the concrete implementation of `instantiate_solver`, which is solver-type specific. 

### Auxiliary methods

- **_check_solver**:

- **_check_solver_kwargs**:

- **_check_is_callable_from_jax**: