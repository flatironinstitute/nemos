"""Basis module stubs."""

from ._basis import AdditiveBasis, MultiplicativeBasis
from ._category import Category
from ._custom_basis import CustomBasis
from ._fourier_basis import FourierSEBasis
from ._transformer_basis import TransformerBasis
from .basis import (
    BSplineConv,
    BSplineEval,
    CyclicBSplineConv,
    CyclicBSplineEval,
    FourierEval,
    HistoryConv,
    IdentityEval,
    MSplineConv,
    MSplineEval,
    OrthExponentialConv,
    OrthExponentialEval,
    RaisedCosineLinearConv,
    RaisedCosineLinearEval,
    RaisedCosineLogConv,
    RaisedCosineLogEval,
    Zero,
)

__all__ = [
    "AdditiveBasis",
    "MultiplicativeBasis",
    "Category",
    "CustomBasis",
    "TransformerBasis",
    "BSplineConv",
    "BSplineEval",
    "CyclicBSplineConv",
    "CyclicBSplineEval",
    "FourierEval",
    "FourierSEBasis",
    "HistoryConv",
    "IdentityEval",
    "MSplineConv",
    "MSplineEval",
    "OrthExponentialConv",
    "OrthExponentialEval",
    "RaisedCosineLinearConv",
    "RaisedCosineLinearEval",
    "RaisedCosineLogConv",
    "RaisedCosineLogEval",
    "Zero",
]
