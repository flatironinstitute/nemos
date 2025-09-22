# The `solvers` Module

## Background

In the earlier versions, NeMoS relied on [JAXopt](https://jaxopt.github.io/stable/) as its optimization backend.
As JAXopt is no longer maintained, we added support for alternative optimization backends.

Some of JAXopt's funtionality was ported to [Optax](https://optax.readthedocs.io/en/latest/) by Google, and [Optimistix](https://docs.kidger.site/optimistix/) was started by the community to fill the gaps after JAXopt's deprecation.

To support flexibility and long-term maintenance, NeMoS now has a backend-agnostic solver interface, allowing the use of solvers from different backend libraries with different interfaces.

## `AbstractSolver` interface
This interface is defined by `AbstractSolver` and mostly follows the JAXopt API.
All solvers implemented in NeMoS are subclasses of `AbstractSolver`, however subclassing is not required for implementing solvers that can be used with NeMoS. (See [custom solvers](#custom-solvers))

The `AbstractSolver` interface requires implementing the following methods:
- `__init__`: all solver parameters and settings should go here. The other methods only take the solver state, current or initial solution (model parameters), and the input data for the objective function.
- `init_state`: Initialize the solver state.
- `update`: Take one step of the optimization algorithm.
- `run`: Run a full optimization.
- `get_accepted_arguments`: Set of argument names that can be passed to `__init__`. These will be the parameters users can change by passing `solver_kwargs` to `BaseRegressor` / `GLM`.
- `get_optim_info`: Collect diagnostic information about the optimization run into an `OptimizationInfo` namedtuple.

This is a generic class parametrized by `SolverState` and `StepResult`.
`SolverState` in concrete subclasses should be the type of the solver state.
`StepResult` is the type of what is returned by each step of the solver. Typically this is a tuple of the parameters and the solver state.

### Optimization info
Because different libraries store info about the optimization run in different places, we decided to standardize some common diagnostics.  
Optimistix saves some things in the stats dict, Optax and Jaxopt store things in their state.
These are saved in `solver.optimization_info` which is of type `OptimizationInfo`.

`OptimizationInfo` holds the following fields:
- `function_val`: The final value of the objective function. As not all solvers store this by default, and it's potentially expensive to evaluate, this field is optional.
- `num_steps`: The number of steps taken by the solver.
- `converged`: Whether the optimization converged according to the solver's criteria.
- `reached_max_steps`: Whether the solver reached the maximum number of steps allowed.

## Adapters
Support for existing solvers from external libraries and the custom implementation of (Prox-)SVRG is done through adapters that "translate" between the interfaces of these external solvers and the `AbstractSolver` interface.

Creating adapters for existing solvers can be done in multiple ways.
In our experience wrapping solver objects through adapters provides a clean way of doing that, and recommend adapters for new optimization libraries to follow this pattern.

`SolverAdapter` provides methods for wrapping existing solvers.  
Each subclass of `SolverAdapter` has to define the methods of `AbstractInterface`, as well as a `_solver_cls` class variable signaling the type of solver wrapped by it.
During construction it has to set a `_solver` attribute that is a concrete instance of `_solver_cls`.

Default method implementations:
- A default implementation of `get_accepted_arguments` is provided, returning the arguments to `__init__`, `_solver_cls`, and `_solver_cls.__init__`, and discarding the ones required by `AbstractSolver.__init__`.
- `__getattr__` dispatches every attribute call to the wrapped `_solver`.
- `__init_subclass__` generates a docstring for the adapter including accepted arguments and the wrapped solver's documentation.

Currently we provide adapters for two optimization backends:
- `OptimistixAdapter` wraps Optimistix solvers.
- `JaxoptAdapter` wraps JAXopt solvers. As `SVRG` and `ProxSVRG` follow the JAXopt interface, these are also wrapped with `JaxoptAdapter`.


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

`OptaxOptimistixSolver` is an adapter for Optax solvers, relying on `optimistix.OptaxMinimiser` to run the full optimization loop.
Optimistix does not have implementations of Nesterov acceleration, so gradient descent is implemented by wrapping `optax.sgd` which does support it.
(Although what Optax calls Nesterov acceleration is not the [original method developed for convex optimization](https://hengshuaiyao.github.io/papers/nesterov83.pdf) but the [version adapted for training deep networks with SGD](https://proceedings.mlr.press/v28/sutskever13.html). Interestingly, JAXopt did implement the original, but in practice it does not seem to be faster.)
Note that `OptaxOptimistixSolver` allows using any solver from Optax (e.g., Adam). See `OptaxOptimistixGradientDescent` for a template of how to wrap new Optax solvers.

(custom-solvers)=
## Custom solvers
If you want to use your own solver in `nemos`, you just have to write a solver that adheres to the `AbstractSolver` interface, and it should be straightforward to plug in.  
While it is not necessary, a way to ensure adherence to the interface is subclassing `AbstractSolver`.

Currently, the solver registry defines which implementation to use for each algorithm, so that has to be overwritten in order to tell `nemos` to use this custom class, but in the future we are [planning to support passing any solver to `BaseRegressor`](https://github.com/flatironinstitute/nemos/issues/378).

We might also define something like an `ImplementsSolverInterface` protocol as well to easily check if user-supplied solvers define the methods required for the interface.

## Stochastic optimization
To run stochastic (~mini-batch) optimization, JAXopt used a `run_iterator` method.
Instead of the full input data `run_iterator` accepts a generator / iterator that provides batches of data.

For solvers defined in `nemos` that can be used this way, we will likely provide `StochasticMixin` which borrows the implementation from JAXopt (Or some version of it, see below).
We will likely define an interface or protocol for this, allowing custom (user-defined) solvers to also implement their own version.
We will also have to decide on how this will be exposed to users on the level of `BaseRegressor` and `GLM`.

:::{admonition} Stochastic optimization interface for (Prox-)SVRG
:class: info

Note that (Prox-)SVRG is especially well-suited for running stochastic optimization, however it currently requires the optimization loop to be implemented separately as it is a bit more involved than what is done by `run_iterator`.  
A potential solution to this would be to provide a separate method that accepts the full data and takes care of the batching. That might be a more convenient alternative to the current `run_iterator` as well.
:::

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
