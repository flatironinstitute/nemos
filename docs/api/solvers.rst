
Solvers
-------
Functions for interacting with the JAX-based optimizers used for parameter fitting.

Solver registry functions
^^^^^^^^^^^^^^^^^^^^^^^^^
Helpers to look up or register solvers.

.. currentmodule:: nemos.solvers

.. autosummary::
    :toctree: generated/solvers
    :nosignatures:

    get_solver
    get_solver_documentation
    list_available_solvers
    list_available_algorithms
    list_algo_backends
    register
    set_default_backend
    list_stochastic_solvers
    supports_stochastic

    SolverSpec


Wrapping existing solvers
^^^^^^^^^^^^^^^^^^^^^^^^^
Adapter classes for existing solvers, especially those defined in the JAXopt, Optimistix, or Optax libraries.

.. currentmodule:: nemos.solvers

.. autosummary::
    :toctree: generated/solvers
    :nosignatures:

    _solver_adapter.SolverAdapter
    _jaxopt_solvers.JaxoptAdapter
    _optimistix_solvers.OptimistixAdapter
    _optax_optimistix_solvers.AbstractOptimistixOptaxSolver

Writing custom solvers
^^^^^^^^^^^^^^^^^^^^^^
Classes useful for creating completely custom solvers.

.. currentmodule:: nemos.solvers

.. autosummary::
    :toctree: generated/solvers
    :nosignatures:

    AbstractSolver
    OptimizationInfo
    SolverProtocol
    validate_solver_class

.. seealso::

    :doc:`The developer notes <developers_notes/07-solvers>` explain the solver contract and expected types in more detail.

Stochastic optimization
^^^^^^^^^^^^^^^^^^^^^^^
Helper classes for running stochastic optimization.

.. currentmodule:: nemos.batching

.. autosummary::
    :toctree: generated/solvers
    :nosignatures:

    DataLoader
    ArrayDataLoader

