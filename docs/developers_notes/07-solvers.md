(developers-solvers)=
# The `solvers` Module

## Background

In the earlier versions, NeMoS relied on [JAXopt](https://jaxopt.github.io/stable/) as its optimization backend.
As [JAXopt is no longer maintained](https://github.com/google/jaxopt?tab=readme-ov-file#status), we added support for alternative optimization backends.
JAXopt remains optionally supported as an extra dependency.

To support flexibility and long-term maintenance, NeMoS now has a backend-agnostic solver interface, allowing the use of solvers from different backend libraries with different interfaces.

In particular, NeMoS's solvers interface is designed to be compatible with solvers from JAXopt, Google's [Optax](https://optax.readthedocs.io/en/latest/), and the community-run [Optimistix](https://docs.kidger.site/optimistix/).

## `AbstractSolver` interface
This interface is defined by [`AbstractSolver`](nemos.solvers._abstract_solver.AbstractSolver) and mostly follows the JAXopt API.
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
`StepResult` is the type of what is returned by each step of the solver. Typically this is a tuple of the parameters, the solver state, and auxiliary variables returned by the objective function.

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
- `JaxoptAdapter` wraps JAXopt solvers when the optional `jaxopt` dependency is installed. As `SVRG` and `ProxSVRG` follow the JAXopt-style interface, these are also wrapped with `JaxoptAdapter` even without JAXopt installed.

Both of these are subclasses of `SolverAdapter` that provides common methods for wrapping existing solvers.
Each subclass of `SolverAdapter` defines the methods of `AbstractInterface`, as well as a `_solver_cls` class variable signaling the type of solver wrapped by it.
During construction they set a `_solver` attribute that is a concrete instance of `_solver_cls`.

Default method implementations in `SolverAdapter`:
- `get_accepted_arguments` returns the arguments to `__init__`, `_solver_cls`, and `_solver_cls.__init__`, and discarding the ones required by `AbstractSolver.__init__`.
- `__getattr__` dispatches every attribute call to the wrapped `_solver`.
- `__init_subclass__` generates a docstring for the adapter including accepted arguments and the wrapped solver's documentation. Extra notes about accepted arguments can be included in docstrings of subclasses using `_note_about_accepted_arguments`. This is used by `OptimistixAdapter` to add a note about the different naming of the tolerance parameter.

## List of available solvers

The following diagram shows the solver class hierarchy. Solvers marked with `[S]` support stochastic optimization via `stochastic_run`.

```
Abstract Class AbstractSolver
│
├─ Abstract Subclass SolverAdapter
│ │
│ ├─ Abstract Subclass OptimistixAdapter
│ │ │
│ │ ├─ Concrete Subclass OptimistixBFGS
│ │ ├─ Concrete Subclass OptimistixFISTA [S]
│ │ ├─ Concrete Subclass OptimistixNAG [S]
│ │ ├─ Concrete Subclass OptimistixNonlinearCG
│ │ └─ Abstract Subclass AbstractOptimistixOptaxSolver
│ │   │
│ │   ├─ Concrete Subclass OptimistixOptaxLBFGS
│ │   └─ Concrete Subclass OptimistixOptaxGradientDescent [S]
│ │
│ └─ Abstract Subclass JaxoptAdapter
│   │
│   ├─ Concrete Subclass JaxoptLBFGS (optional)
│   ├─ Concrete Subclass JaxoptGradientDescent (optional) [S]
│   ├─ Concrete Subclass JaxoptProximalGradient (optional) [S]
│   ├─ Concrete Subclass JaxoptBFGS (optional)
│   ├─ Concrete Subclass JaxoptNonlinearCG (optional)
│   │
│   ├─ Concrete Subclass WrappedSVRG [S]
│   └─ Concrete Subclass WrappedProxSVRG [S]
```

`OptaxOptimistixSolver` is an adapter for Optax solvers, relying on `optimistix.OptaxMinimiser` to run the full optimization loop. If there is a need, this can be used to wrap adaptive solvers (e.g. Adam).

Gradient descent is implemented by two classes:
- One is wrapping `optax.sgd` which supports momentum and acceleration.
Note that what Optax calls Nesterov acceleration is not the [original method developed for convex optimization](https://hengshuaiyao.github.io/papers/nesterov83.pdf) but the [version adapted for training deep networks with SGD](https://proceedings.mlr.press/v28/sutskever13.html).
- As JAXopt implemented the original method, a [port of JAXopt's `GradientDescent` was added to NeMoS](https://github.com/flatironinstitute/nemos/pull/411) as `OptimistixNAG`.

Similarly to NAG, an accelerated proximal gradient algorithm ([FISTA](https://www.ceremade.dauphine.fr/~carlier/FISTA)) was [ported from JAXopt](https://github.com/flatironinstitute/nemos/pull/411) as `OptimistixFISTA`.

Available solvers and which implementation they dispatch to are defined in the solver registry.
A list of available algorithms is provided by {py:func}`nemos.solvers.list_available_algorithms`.
All solvers in the registry can be listed with {py:func}`nemos.solvers.list_available_solvers`, and extended documentation about each solver can be accessed using {py:func}`nemos.solvers.get_solver_documentation`.


(custom-solvers)=
## Custom solvers
The solver registry -- implemented in `nemos.solvers._solver_registry` -- the list of available algorithms and their implementation.

Alternatively, users can use their own solvers to fit NeMoS models, they just have to write a solver that adheres to the `AbstractSolver` interface, and it should be straightforward to plug in.

Fitting models using this custom solver can be done by:
1. Registering the class implementing the solver in the solver registry: \
`nemos.solvers.register("Fancy-Algorithm", MyCustomSolverClass, backend="custom")`\
Please note that not a solver instance but a class/type has to be passed.
2. Declaring the algorithm's compatibility with the appropriate regularizers: \
`nemos.regularizer.UnRegularized.allow_solver("Fancy-Algorithm")`.
3. Referring to the algorithm by name when creating a `GLM` (or any `BaseRegressor`): \
`GLM(solver_name="Fancy-Algorithm[custom]")`

When registering a solver, NeMoS does basic checks validating the custom solver's compatibility by checking if the required methods are implemented, i.e. if the class implements the  and that their signatures match [`AbstractSolver`](nemos.solvers.AbstractSolver) (and [`SolverProtocol`](nemos.solvers.SolverProtocol)).
There are also options in [`nemos.solvers.register`](nemos.solvers.register) to run a small ridge regression problem, testing that the solver's methods can be used as intended.
To validate a solver without registering, the [`nemos.solvers.validate_solver_class`](nemos.solvers.validate_solver_class) can be used.
While it is not necessary, a way to ensure adherence to the interface is subclassing `AbstractSolver`.

## Stochastic optimization

NeMoS provides a high-level interface for stochastic (mini-batch) optimization through the `stochastic_fit` method on GLM models and the `stochastic_run` method on solvers.

### Using `stochastic_fit`

The simplest way to use stochastic optimization is through the `stochastic_fit` method on GLM models:

```python
import jax.numpy as jnp
import nemos as nmo
from nemos.batching import ArrayDataLoader

# Create data loader
X = jnp.ones((10000, 50))
y = jnp.ones((10000,))
loader = ArrayDataLoader(X, y, batch_size=128, shuffle=True)

# Fit model using stochastic optimization
model = nmo.glm.GLM(
    solver_name="GradientDescent",
    solver_kwargs={"stepsize": 0.01}
)
model.stochastic_fit(loader, num_epochs=10)
```

### DataLoader protocol

Stochastic optimization requires data to be provided through a `DataLoader` that conforms to the `nemos.batching.DataLoader` protocol:

```python
class DataLoader(Protocol):
    def __iter__(self) -> Iterator[tuple[Any, ...]]:
        """Iterate over tuples containing input and output data, e.g. (X_batch, y_batch). Must return a fresh iterator each call."""
        ...

    @property
    def n_samples(self) -> int:
        """Total number of samples in the dataset."""
        ...

    def sample_batch(self) -> tuple[Any, ...]:
        """Return a single batch for initialization purposes."""
        ...
```

NeMoS provides `ArrayDataLoader` for in-memory arrays. For out-of-core data, users can implement their own data loader following the protocol.

### Solver-level interface

At the solver level, stochastic optimization is provided through the `stochastic_run` method:

```python
params, state, aux = solver.stochastic_run(init_params, data_loader, num_epochs=10)
```

Not all solvers support stochastic optimization. Only solvers with `_supports_stochastic = True` can be used. Currently supported solvers:

- `GradientDescent` (and variants like `OptimistixNAG`, `OptimistixOptaxGradientDescent`)
- `ProximalGradient` (and `OptimistixFISTA`)
- `SVRG`
- `ProxSVRG`

Solvers that do not support stochastic optimization (e.g., `BFGS`, `LBFGS`, `NonlinearCG`) will raise `NotImplementedError` when `stochastic_run` is called.

### Implementation details

Stochastic support is implemented through the `StochasticSolverMixin` class which provides a default implementation of `_stochastic_run_impl`. This mixin iterates over the data loader for the specified number of epochs, calling `update` on each batch.

For SVRG-based solvers, `stochastic_run` in the wrappers dispatches to a `run_streaming` method that handles the more complex optimization loop that requires computing full gradients at reference points.

### Manual batching

For more control over the optimization process, you can still use the manual batching approach with `initialize_solver_and_state` and `update` as shown in the [batching how-to guide](../how_to_guide/plot_04_batch_glm.md). This approach is useful when you need custom logic between batches or want to implement learning rate schedules.
