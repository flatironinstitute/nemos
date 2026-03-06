"""Basis for encoding categorical data."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import jax
import jax.numpy as jnp
from numpy._typing import NDArray
from numpy.typing import ArrayLike

from ..label_encoder import LabelEncoder

if TYPE_CHECKING:
    from pynapple import Tsd, TsdFrame, TsdTensor

from ..typing import FeatureMatrix
from ._basis import Basis
from ._basis_mixin import AtomicBasisMixin


class CategoryBasis(AtomicBasisMixin, Basis):
    _convert_to_float = False

    def __init__(self, categories: List | NDArray | int, label: Optional[str] = None):
        n_categories = self._get_n_categories(categories)
        self._label_encoder = LabelEncoder(n_categories)
        self.categories = categories
        self._n_inputs = 1
        Basis.__init__(
            self,
        )
        AtomicBasisMixin.__init__(
            self,
            n_basis_funcs=self.n_basis_funcs,
            label=label,
        )

    @property
    def categories(self):
        return self._label_encoder.classes_

    @categories.setter
    def categories(self, categories: List | NDArray | int):
        if isinstance(categories, int):
            self._label_encoder.set_classes(
                jnp.arange(self._get_n_categories(categories))
            )
        else:
            self._label_encoder.set_classes(categories)

    @property
    def n_basis_funcs(self):
        return self._label_encoder.n_classes

    @staticmethod
    def _get_n_categories(categories: int | List | NDArray) -> int:
        return getattr(categories, "__len__", lambda: categories)()

    def evaluate(self, xi: ArrayLike | Tsd | TsdFrame | TsdTensor) -> FeatureMatrix:
        encoded = self._label_encoder.encode(xi)
        return jax.nn.one_hot(encoded, self._label_encoder.n_classes)
