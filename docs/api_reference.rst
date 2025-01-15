.. _api_ref:

API Reference
=============

.. _nemos_glm:

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

.. _nemos_basis:

The ``nemos.basis`` module
--------------------------
Provides basis function classes to construct and transform features for model inputs.
Basis can be grouped according to the mode of operation into basis that performs convolution and basis that operates
as non-linear maps.


**The Abstract Classes:**

These classes are the building blocks for the concrete basis classes.

.. currentmodule:: nemos.basis._basis

.. autosummary::
    :toctree: generated/_basis
    :recursive:
    :nosignatures:

    Basis

.. currentmodule:: nemos.basis._spline_basis
.. autosummary::
    :toctree: generated/_basis
    :recursive:
    :nosignatures:

    SplineBasis


**Bases For Convolution:**

.. currentmodule:: nemos.basis

.. autosummary::
    :toctree: generated/basis
    :recursive:
    :nosignatures:


    MSplineConv
    BSplineConv
    CyclicBSplineConv
    RaisedCosineLinearConv
    RaisedCosineLogConv
    OrthExponentialConv
    HistoryConv

.. check for a config that prints only nemos.basis.Name

**Bases For Non-Linear Mapping:**

.. currentmodule:: nemos.basis

.. autosummary::
    :toctree: generated/basis
    :recursive:
    :nosignatures:

    MSplineEval
    BSplineEval
    CyclicBSplineEval
    RaisedCosineLinearEval
    RaisedCosineLogEval
    OrthExponentialEval
    IdentityEval

**Composite Bases:**

.. currentmodule:: nemos.basis._basis

.. autosummary::
    :toctree: generated/_basis
    :recursive:
    :nosignatures:

    AdditiveBasis
    MultiplicativeBasis

**Basis As ``scikit-learn`` Tranformers:**

.. currentmodule:: nemos.basis._transformer_basis

.. autosummary::
    :toctree: generated/_transformer_basis
    :recursive:
    :nosignatures:

    TransformerBasis

.. _observation_models:

The ``nemos.observation_models`` module
---------------------------------------
Statistical models to describe the distribution of neural responses or other predicted variables, given inputs.

.. currentmodule:: nemos.observation_models

.. autosummary::
    :toctree: generated/observation_models
    :recursive:
    :nosignatures:

    Observations
    PoissonObservations
    GammaObservations

.. _regularizers:

The ``nemos.regularizer`` module
--------------------------------
Implements various regularization techniques to constrain model parameters, which helps prevent overfitting.

.. currentmodule:: nemos.regularizer

.. autosummary::
    :toctree: generated/regularizer
    :recursive:
    :nosignatures:

    Regularizer
    UnRegularized
    Ridge
    Lasso
    GroupLasso

The ``nemos.simulation`` module
-------------------------------
Utility functions for simulating spiking activity in recurrently connected neural populations.

.. currentmodule:: nemos.simulation

.. autosummary::
    :toctree: generated/simulation
    :recursive:
    :nosignatures:

    simulate_recurrent
    difference_of_gammas
    regress_filter


The ``nemos.convolve`` module
-----------------------------
Utility functions for running convolution over the sample axis.

.. currentmodule:: nemos.convolve

.. autosummary::
    :toctree: generated/regularizer
    :recursive:
    :nosignatures:

    create_convolutional_predictor
    tensor_convolve


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

The ``nemos.pytrees.FeaturePytree`` class
-----------------------------------------
Class for storing the input arrays in a dictionary. Keys are usually variable names. 
These objects can be provided as input to nemos GLM methods.

.. currentmodule:: nemos.pytrees

.. autosummary::
    :toctree: generated/pytree
    :recursive:
    :nosignatures:

    FeaturePytree
