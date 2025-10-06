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
    :nosignatures:

    GLM
    PopulationGLM

.. _nemos_io:

The NeMoS I/O module
--------------------

.. currentmodule:: nemos.io
.. autosummary::
    :toctree: generated/io
    :nosignatures:

    load_model
    inspect_npz

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
    :nosignatures:

    Basis

.. currentmodule:: nemos.basis._spline_basis
.. autosummary::
    :toctree: generated/_basis
    :nosignatures:

    SplineBasis


**Bases For Convolution:**

.. currentmodule:: nemos.basis

.. autosummary::
    :toctree: generated/basis
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
    :nosignatures:

    MSplineEval
    BSplineEval
    CyclicBSplineEval
    RaisedCosineLinearEval
    RaisedCosineLogEval
    FourierEval
    OrthExponentialEval
    IdentityEval

**Composite Bases:**

.. currentmodule:: nemos.basis._basis

.. autosummary::
    :toctree: generated/_basis
    :nosignatures:

    AdditiveBasis
    MultiplicativeBasis

**Custom defined Basis:**

Define a fully functional basis form a list of functions.

.. currentmodule:: nemos.basis._custom_basis

.. autosummary::
    :toctree: generated/_custom_basis
    :nosignatures:

    CustomBasis

**Basis As scikit-learn Tranformers:**

.. currentmodule:: nemos.basis._transformer_basis

.. autosummary::
    :toctree: generated/_transformer_basis
    :nosignatures:

    TransformerBasis

.. _observation_models:

The ``nemos.observation_models`` module
---------------------------------------
Statistical models to describe the distribution of neural responses or other predicted variables, given inputs.

.. currentmodule:: nemos.observation_models

.. autosummary::
    :toctree: generated/observation_models
    :nosignatures:

    Observations
    PoissonObservations
    NegativeBinomialObservations
    GammaObservations
    BernoulliObservations

.. _regularizers:

The ``nemos.regularizer`` module
--------------------------------
Implements various regularization techniques to constrain model parameters, which helps prevent overfitting.

.. currentmodule:: nemos.regularizer

.. autosummary::
    :toctree: generated/regularizer
    :nosignatures:

    Regularizer
    UnRegularized
    Ridge
    Lasso
    ElasticNet
    GroupLasso

The ``nemos.simulation`` module
-------------------------------
Utility functions for simulating spiking activity in recurrently connected neural populations.

.. currentmodule:: nemos.simulation

.. autosummary::
    :toctree: generated/simulation
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
    :nosignatures:

    create_convolutional_predictor


The ``nemos.solvers`` module
----------------------------
JAX-based optimizers used for parameter fitting.

.. currentmodule:: nemos.solvers

.. autosummary::
    :toctree: generated/solvers
    :nosignatures:

    get_solver_documentation
    list_available_solvers


The ``nemos.identifiability_constraints`` module
------------------------------------------------
Functions to apply identifiability constraints to rank-deficient feature matrices, ensuring the uniqueness of model
solutions.

.. currentmodule:: nemos.identifiability_constraints

.. autosummary::
    :toctree: generated/identifiability_constraints
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
    :nosignatures:

    FeaturePytree
