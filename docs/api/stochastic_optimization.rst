.. _nemos_stochastic_optimization:

Stochastic optimization
-----------------------
Tools for fitting models on batched data via ``stochastic_fit``.

Data loaders
^^^^^^^^^^^^
Helper classes loading data for stochastic optimization.

.. currentmodule:: nemos.batching

.. autosummary::
    :toctree: generated/stochastic_optimization
    :nosignatures:

    DataLoader
    ArrayDataLoader
    LazyArrayDataLoader

Callbacks
^^^^^^^^^
Callback interfaces for stochastic optimization and post-fit run summaries.

.. currentmodule:: nemos.callbacks

.. autosummary::
    :toctree: generated/stochastic_optimization
    :nosignatures:

    Callback
    CallbackList
    SolverConvergenceCallback
    TestLossLogger
    TrainingContext
    StochasticFitSummary
