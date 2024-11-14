.. _api_ref:

API Guide
=========

The ``nemos.glm`` module
------------------------
Classes for creating Generalized Linear Models (GLMs) for both single neurons and neural populations.

.. currentmodule:: nemos.glm

.. autosummary::
    :toctree: generated/glm
    :recursive:
    :nosignatures:

    GLM
    PopulationGLM

The ``nemos.basis`` module
--------------------------
Provides basis function classes to construct and transform features for model inputs.

.. currentmodule:: nemos.basis

.. autosummary::
    :toctree: generated/basis
    :recursive:
    :nosignatures:

    BSplineBasis
    CyclicBSplineBasis
    MSplineBasis
    OrthExponentialBasis
    RaisedCosineBasisLinear
    RaisedCosineBasisLog
    AdditiveBasis
    MultiplicativeBasis
    TransformerBasis

The ``nemos.observation_models`` module
--------------------------------------
Statistical models to describe the distribution of neural responses or other predicted variables, given inputs.

.. currentmodule:: nemos.observation_models

.. autosummary::
    :toctree: generated/observation_models
    :recursive:
    :nosignatures:

    PoissonObservations
    GammaObservations

The ``nemos.regularizer`` module
--------------------------------
Implements various regularization techniques to constrain model parameters, which helps prevent overfitting.

.. currentmodule:: nemos.regularizer

.. autosummary::
    :toctree: generated/regularizer
    :recursive:
    :nosignatures:

    UnRegularized
    Ridge
    Lasso
    GroupLasso

The ``nemos.simulation`` module
-------------------------------
Utility functions for simulating spiking activity in recurrently connected neural populations.

.. currentmodule:: nemos.simulation

.. autosummary::
    :toctree: generated/regularizer
    :recursive:
    :nosignatures:

    simulate_recurrent
    difference_of_gammas
    regress_filter

The ``nemos.identifiability_constraints`` module
------------------------------------------------
Functions to apply identifiability constraints to rank-deficient feature matrices, ensuring the uniqueness of model
solutions.

.. currentmodule:: nemos.identifiability_constraints

.. autosummary::
    :toctree: generated/identifiability_constraints
    :recursive:
    :nosignatures:

    apply_identifiability_constraints
    apply_identifiability_constraints_by_basis_component
