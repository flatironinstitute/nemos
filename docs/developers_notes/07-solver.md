# The `solver` Module

Note that this is a very rough draft / braindump for now, and will be cleaned up later.

## Background

In the beginning NeMoS relied on JAXopt as its optimization backend.
As JAXopt is no longer maintained, we added support for alternative optimization backends.
Some of JAXopt's funtionality was ported to Optax by Google, and Optimistix was started by the community to fill the gaps after JAXopt's deprecation.
To support flexibility and long-term maintenance, NeMoS now has a backend-agnostic solver interface, allowing the use of solvers from different backend libraries with different interfaces.

## `AbstractSolver` interface
This interface is defined by [`AbstractSolver`](nemos.solvers.AbstractSolver) and mostly follows the JAXopt API.
All solvers in NeMoS are subclasses of `AbstractSolver`.

The `AbstractSolver` interface requires implementing the following methods:
- `__init__`: all solver parameters and settings should go here. The other methods only take the solver state, current or initial solution (model parameters), and the input data for the objective function.
- `init_state`: Initialize the solver state.
- `update`: Take one step of the optimization algorithm.
- `run`: Run a full optimization.
- `get_accepted_arguments`: Set of argument names that can be passed to `__init__`.
- `get_optim_info`: Collect diagnostic information about the optimization run into an `OptimizationInfo` namedtuple.

This is a generic class parametrized by `SolverState` and `StepResult`.
`SolverState` in concrete subclasses should be the type of the solver state.
`StepResult` is the type of what is returned by each step of the solver. Typically this is a tuple of the parameters and the solver state.


## Adapters
Creating adapters for existing solvers can be done in multiple ways.
In our experience wrapping solver objects through adapters provides a clean way of doing that, and recommend adapters for new optimization libraries to follow this pattern.

[`SolverAdapter`](nemos.solvers.SolverAdapter) provides methods for wrapping existing solvers.
A default implementation of get_accepted_arguments is provided.
Dispatches every attribute call to the wrapped `_solver`.
`__init_subclass__`: Generates a docstring for the adapter including accepted arguments and the wrapped solver's documentation.

Currently we provide adapters for two optimization backends:
- [`OptimistixAdapter`](nemos.solvers.OptimistixAdapter) wraps Optimistix solvers.
- [`JaxoptAdapter`](nemos.solvers.JaxoptAdapter) wraps JAXopt solvers. As `SVRG` and `ProxSVRG` follow the JAXopt interface, these are also wrapped with `JaxoptAdapter`.


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
│ │ └─ Concrete Subclass OptaxOptimistixSolver
│ │   │
│ │   ├─ Concrete Subclass OptaxOptimistixLBFGS
│ │   ├─ Concrete Subclass OptaxOptimistixGradientDescent
│ │   └─ Concrete Subclass OptaxOptimistixProximalGradient
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

`OptaxOptimistixSolver` is for using Optax solvers, utilizing optimistix.OptaxMinimiser to run the full optimization loop.
Optimistix does not have implementations of Nesterov acceleration, so gradient descent is implemented by wrapping `optax.sgd` which does support it.
`OptaxOptimistixSolver` allows using any solver from Optax (e.g., Adam). See `OptaxOptimistixGradientDescent` for a template of how to wrap new Optax solvers.


## Optimization info
Because different libraries store info about the optimization run in different places, we decided to standardize some common diagnostics.
Optimistix saves some things in the stats dict, Optax and Jaxopt store things in their state.
These are saved in `solver.optimization_info`

## Custom solvers
If you want to use your own solver in `nemos`, you just have to write a solver that adheres to the `AbstractSolver` interface, and it should be straightforward to plug in.
Currently, the solver registry defines which implementation to use for each algorithm, so that has to be overwritten, but in the future we are [planning to support passing any solver to `BaseRegressor`](https://github.com/flatironinstitute/nemos/issues/378).
We might also define something like an `ImplementsSolverInterface` protocol as well to easily check if user-supplied solvers define the methods required for the interface.

## Stochastic optimization
To run stochastic (~mini-batch) optimization, following JAXopt we will require a `run_iterator` method to be defined.
Instead of the full input data `run_iterator` accepts a generator / iterator that provides batches of data.
For solvers in `nemos` that can be used this way, we provide `StochasticMixin` which borrows the implementation from JAXopt.

Note that (Prox-)SVRG is especially well-suited for running stochastic optimization, however it currently requires the optimization loop to be implemented separately as it is a bit more involved than what is done by `run_iterator`.
A potential solution to this would be to provide a separate method that accepts the full data, and takes care of the batching. That might be a more convenient alternative to the current `run_iterator` as well.

## Note on line searches vs. fixed stepsize in Optimistix
By default Optimistix doesn't expose the search attribute of concrete solvers but we might want to flexibly switch between linesearches and constant learning rates depending on whether `stepsize` is passed to the solver.
A solution to this would be to create short redefinitions of the required solvers with the `search` as an argument to `__init__`, and in the adapter dealing with `stepsize` with something like:
```python
class BFGS(AbstractBFGS[Y, Aux, _Hessian]):
    rtol: float
    atol: float
    norm: Callable[[PyTree], Scalar]
    use_inverse: bool
    descent: NewtonDescent
    search: AbstractSearch
    verbose: frozenset[str]

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        use_inverse: bool = True,
        verbose: frozenset[str] = frozenset(),
        search: AbstractSearch = Zoom(initial_guess_strategy="one"),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.use_inverse = use_inverse
        self.descent = NewtonDescent(linear_solver=lx.Cholesky())
        self.search = search
        self.verbose = verbose
```

and

```python
if "stepsize" in solver_init_kwargs:
   assert "search" not in solver_init_kwargs, "Specify either search or stepsize"
   solver_init_kwargs["search"] = optx.LearningRate(
       solver_init_kwargs.pop("stepsize")
   )
```
