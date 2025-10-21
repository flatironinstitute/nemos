# The `solvers` Module

## Background

In the earlier versions, NeMoS relied on [JAXopt](https://jaxopt.github.io/stable/) as its optimization backend.
As [JAXopt is no longer maintained](https://github.com/google/jaxopt?tab=readme-ov-file#status), we added support for alternative optimization backends.

To support flexibility and long-term maintenance, NeMoS now has a backend-agnostic solver interface, allowing the use of solvers from different backend libraries with different interfaces.

In particular, NeMoS's solvers interface is designed to be compatible with solvers from JAXopt, Google's [Optax](https://optax.readthedocs.io/en/latest/), and the community-run [Optimistix](https://docs.kidger.site/optimistix/).

## `AbstractSolver` interface
This interface is defined by `AbstractSolver` and mostly follows the JAXopt API.
All solvers implemented in NeMoS are subclasses of `AbstractSolver`, however subclassing is not strictly required for implementing solvers that can be used with NeMoS. (See [custom solvers](#custom-solvers))

The `AbstractSolver` interface requires implementing the following methods:
- `__init__`: Construct a solver object. All solver parameters and settings (tolerance, maximum number of steps, etc.) are passed here. The other methods only take the solver state, current or initial solution (model parameters), and the input data for the objective function.
- `init_state`: Initialize the solver state.
- `update`: Take one step of the optimization algorithm.
- `run`: Run a full optimization.
- `get_accepted_arguments`: Set of argument names that can be passed to `__init__`. These will be the parameters users can change by passing `solver_kwargs` to NeMoS models (e.g., `GLM`).
- `get_optim_info`: Collect diagnostic information about the optimization run into an `OptimizationInfo` namedtuple, [described in the next section](#optimization-info).

`AbstractSolver` is a generic class parametrized by `SolverState` and `StepResult`.
`SolverState` in concrete subclasses should be the type of the solver state.
`StepResult` is the type of what is returned by each step of the solver. Typically this is a tuple of the parameters and the solver state.

(optimization-info)=
### Optimization info
Because different libraries store info about the optimization run in different places, we decided to standardize some common diagnostics.
These are accessed using the `get_optim_info` method which takes the solver state and returns an `OptimizationInfo`.

`OptimizationInfo` holds the following fields:
- `function_val`: The final value of the objective function. As not all solvers store this by default, and as it's potentially expensive to evaluate, this field is optional.
- `num_steps`: The number of steps taken by the solver.
- `converged`: Whether the optimization converged according to the solver's criteria.
- `reached_max_steps`: Whether the solver reached the maximum number of steps allowed.

## Adapters
Support for existing solvers from external libraries and the custom implementation of (Prox-)SVRG is done through adapters that "translate" between the interfaces of these external solvers and the `AbstractSolver` interface.

Creating adapters for existing solvers can be done in multiple ways.
In our experience wrapping solver objects through adapters provides a clean way of doing that, and adapters in NeMoS follow this pattern.

Currently there are adapters implemented for two optimization backends:
- `OptimistixAdapter` wraps Optimistix solvers.
- `JaxoptAdapter` wraps JAXopt solvers. As `SVRG` and `ProxSVRG` follow the JAXopt interface, these are also wrapped with `JaxoptAdapter`.

Both of these are subclasses of `SolverAdapter` that provides common methods for wrapping existing solvers.
Each subclass of `SolverAdapter` defines the methods of `AbstractInterface`, as well as a `_solver_cls` class variable signaling the type of solver wrapped by it.
During construction they set a `_solver` attribute that is a concrete instance of `_solver_cls`.

Default method implementations in `SolverAdapter`:
- `get_accepted_arguments` returns the arguments to `__init__`, `_solver_cls`, and `_solver_cls.__init__`, and discarding the ones required by `AbstractSolver.__init__`.
- `__getattr__` dispatches every attribute call to the wrapped `_solver`.
- `__init_subclass__` generates a docstring for the adapter including accepted arguments and the wrapped solver's documentation. Extra notes about accepted arguments can be included in docstrings of subclasses using `_note_about_accepted_arguments`. This is used by `OptimistixAdapter` to add a note about the different naming of the tolerance parameter.

## List of available solvers

```
Abstract Class AbstractSolver
│
├─ Abstract Subclass SolverAdapter
│ │
│ ├─ Abstract Subclass OptimistixAdapter
│ │ │
│ │ ├─ Concrete Subclass OptimistixBFGS
│ │ ├─ Concrete Subclass OptimistixLBFGS
│ │ ├─ Concrete Subclass OptimistixNonlinearCG
│ │ └─ Abstract Subclass AbstractOptimistixOptaxSolver
│ │   │
│ │   ├─ Concrete Subclass OptimistixOptaxLBFGS
│ │   ├─ Concrete Subclass OptimistixOptaxGradientDescent
│ │   └─ Concrete Subclass OptimistixOptaxProximalGradient
│ │
│ └─ Abstract Subclass JaxoptAdapter
│   │
│   ├─ Concrete Subclass JaxoptLBFGS
│   ├─ Concrete Subclass JaxoptGradientDescent
│   ├─ Concrete Subclass JaxoptProximalGradient
│   ├─ Concrete Subclass JaxoptBFGS
│   ├─ Concrete Subclass JaxoptNonlinearCG
│   │
│   ├─ Concrete Subclass WrappedSVRG
│   └─ Concrete Subclass WrappedProxSVRG
```

`OptaxOptimistixSolver` is an adapter for Optax solvers, relying on `optimistix.OptaxMinimiser` to run the full optimization loop.
Optimistix does not have implementations of Nesterov acceleration, so gradient descent is implemented by wrapping `optax.sgd` which does support it.
(Although what Optax calls Nesterov acceleration is not the [original method developed for convex optimization](https://hengshuaiyao.github.io/papers/nesterov83.pdf) but the [version adapted for training deep networks with SGD](https://proceedings.mlr.press/v28/sutskever13.html). JAXopt did implement the original method, and [a port of this is planned to be added to NeMoS](https://github.com/flatironinstitute/nemos/issues/380).)

Available solvers and which implementation they dispatch to are defined in the solver registry.
A list of available solvers is provided by {py:func}`nemos.solvers.list_available_solvers`, and extended documentation about each solver can be accessed using {py:func}`nemos.solvers.get_solver_documentation`.

(custom-solvers)=
## Custom solvers
Currently, the solver registry defines the list of available algorithms and their implementation, but in the future we are [planning to support passing any solver to `BaseRegressor`](https://github.com/flatironinstitute/nemos/issues/378).

If someone wants to use their own solver in `nemos`, they just have to write a solver that adheres to the `AbstractSolver` interface, and it should be straightforward to plug in.
While it is not necessary, a way to ensure adherence to the interface is subclassing `AbstractSolver`.

## Stochastic optimization
To run stochastic (mini-batch) optimization, JAXopt used a `run_iterator` method.
Instead of the full input data `run_iterator` accepts a generator / iterator that provides batches of data.

For information on how stochastic optimization is planned to be supported in NeMOS, see the [issue tracking the stochastic optimization interface](https://github.com/flatironinstitute/nemos/issues/376).

:::{admonition} Stochastic optimization interface for (Prox-)SVRG
:class: warning

Note that (Prox-)SVRG is especially well-suited for running stochastic optimization, however it currently requires the optimization loop to be implemented separately as it is a bit more involved than what is done by `run_iterator`.  
:::
