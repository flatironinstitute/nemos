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

from ..type_casting import support_pynapple
from ..typing import FeatureMatrix
from ._basis import Basis
from ._basis_mixin import AtomicBasisMixin


@support_pynapple(conv_type="jax")
def one_hot_encoding(
    x: NDArray | Tsd | TsdFrame | TsdTensor, n_classes: int
) -> ArrayLike | Tsd:
    """One-hot encode with pynapple support."""
    return jax.nn.one_hot(x, n_classes)


class CategoryBasis(AtomicBasisMixin, Basis):
    """
    Base class for categorical one-hot encoding.

    Encodes a categorical variable with ``n_categories`` unique labels as a
    one-hot matrix of shape ``(n_samples, n_categories)``. Each column
    corresponds to one category: the entry is 1 when the input equals that
    category, and 0 everywhere else.

    Parameters
    ----------
    categories :
        The set of valid category labels. Accepted forms:

        - ``int``: interpreted as the number of categories; labels default to
          ``[0, 1, ..., categories-1]``.
        - ``list`` or ``NDArray``: the explicit list of unique category labels.

    label :
        The label of the basis, intended to be descriptive of the task variable
        being processed. For example: ``"trial_type"``, ``"stimulus_id"``.
    """

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
            self._label_encoder = LabelEncoder(categories)
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
        """
        Evaluate the categorical basis at the provided samples.

        Encodes each sample label as a one-hot vector of length ``n_basis_funcs``
        (equal to the number of categories).

        Parameters
        ----------
        xi :
            Array of category labels. Every value must belong to the set of
            categories defined at construction time. Shape is arbitrary; the
            returned array appends the category axis as the last dimension.

        Returns
        -------
        :
            One-hot encoded array of shape ``(*xi.shape, n_basis_funcs)``.

        Raises
        ------
        ValueError
            If any label in ``xi`` is not in the set of known categories.
        """
        # Encoded could be an array or nap tsd, with integer dtype.
        encoded = self._label_encoder.encode(xi)
        return one_hot_encoding(encoded, self._label_encoder.n_classes)

    @support_pynapple(conv_type="jax")
    def decode(self, X: ArrayLike | TsdFrame, axis: int = -1):
        """Decode the categorical basis labels for 1-hot encoding."""
        return self._label_encoder.decode(jax.numpy.argmax(X, axis=axis))
