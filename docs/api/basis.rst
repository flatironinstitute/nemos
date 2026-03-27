.. _nemos_basis:

Bases
-----
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
    Zero

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
