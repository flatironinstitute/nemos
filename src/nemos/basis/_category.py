"""Basis for encoding categorical data."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer
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
          When a list is provided, it is converted to an ``NDArray`` via
          ``np.asarray``. Mixed-type lists will be cast to a common dtype
          (e.g., ``["a", 1]`` becomes ``array(['a', '1'], dtype='<U21')``).

    out_of_category:
        If False, raise if labels that do not belong to ``categories`` are provided,
        else encode the out-of-category labels as all 0s.

    label :
        The label of the basis, intended to be descriptive of the task variable
        being processed. For example: ``"trial_type"``, ``"stimulus_id"``.
    """

    _convert_to_float = False
    _is_discrete = True

    def __init__(
        self,
        categories: List | NDArray | int,
        out_of_category: str | int | float | None,
        label: Optional[str] = None,
    ):
        n_categories = self._get_n_categories(categories)
        self._label_encoder = LabelEncoder(n_categories)
        self.out_of_category = out_of_category
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
    def out_of_category(self) -> int | str | float | None:
        return self._out_of_category

    @out_of_category.setter
    def out_of_category(self, out_of_category: bool):
        if not isinstance(out_of_category, bool):
            raise TypeError("``out_of_category`` must be a boolean (True or False).")
        self._out_of_category = out_of_category

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
            cats = np.asarray(categories)
            unique = np.unique(cats)
            if len(unique) != len(cats):
                duplicates = [c for c in unique if np.sum(cats == c) > 1]
                raise ValueError(
                    f"Duplicate category labels provided: {duplicates}. "
                    "Each label must be unique."
                )
            self._label_encoder.set_classes(categories)

    @property
    def n_basis_funcs(self):
        return self._label_encoder.n_classes

    @staticmethod
    def _get_n_categories(categories: int | List | NDArray) -> int:
        return getattr(categories, "__len__", lambda: categories)()

    @support_pynapple(conv_type="numpy")
    def _set_out_of_category(self, xi: NDArray | jnp.ndarray, encoded: jnp.ndarray):
        """Set out-of-category encoding to -1.

        The method assign a -1 encoding to all categories that are not in ``self.categories``.

        Parameters
        ----------
        xi:
            Array of category labels that may contain labels that are not in self.categories.
        encoded:
            Array of integer category labels, usually output of label encoding in "unsafe" mode.

        Notes
        -----
            - ``xi`` and ``encoded`` must have the same shape.
            - The method assigns -1 to all category labels not in ``self.categories``.
        """
        # encoded is always int between 0 and n-1.
        # setting -1 will result in a 0s when 1-hot encoding
        if jnp.issubdtype(self.categories.dtype, jnp.number) and jnp.issubdtype(
            encoded.dtype, jnp.number
        ):
            encoded = jnp.where(jnp.isin(xi, self.categories), encoded, -1)
        elif jnp.issubdtype(encoded.dtype, jnp.number):
            # handle non-numeric via numpy isin
            encoded = jnp.where(np.isin(xi, self.categories), encoded, -1)
        else:
            # handle both numpy, both string - edge case in which user passed
            # string labels after setting the categories as integers.
            encoded = np.full(encoded.shape, -1)
        return encoded

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
        if not self.out_of_category and not isinstance(xi, Tracer):
            encoded = self._label_encoder.encode(xi, safe=True)
        elif not self.out_of_category and isinstance(xi, Tracer):
            raise ValueError(
                "JIT compilation not available for ``out_of_category=False``. "
                "To enable JIT compilation, please set ``out_of_category=True``."
            )
        else:
            encoded = self._label_encoder.encode(xi, safe=False)
            # set -1 to out-of category-labels
            # one_hot_encoding will assign 0s where encoded == -1
            encoded = self._set_out_of_category(xi, encoded)
        return one_hot_encoding(encoded, self._label_encoder.n_classes)

    def evaluate_on_grid(self, *n_samples: int) -> Tuple[Tuple[NDArray], NDArray]:
        """Raise for categorical basis."""
        raise NotImplementedError(
            "``evaluate_on_grid`` is not implemented for categorical basis. The method "
            "only makes sense for continuous bases, ``Category`` process discrete "
            "inputs only."
        )

    def __getattr__(self, name):
        if name == "bounds":
            raise AttributeError("Category basis has no bounds.")
        super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name == "bounds":
            raise AttributeError("Category basis has no bounds.")
        super().__setattr__(name, value)
