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
from ._basis_mixin import AtomicBasisMixin, EvalBasisMixin
from ._composition_utils import add_docstring


@support_pynapple(conv_type="jax")
def one_hot_encoding(
    x: NDArray | Tsd | TsdFrame | TsdTensor, n_classes: int
) -> ArrayLike | Tsd:
    """One-hot encode with pynapple support."""
    return jax.nn.one_hot(x, n_classes)


class Category(EvalBasisMixin, AtomicBasisMixin, Basis):
    """
    Categorical one-hot encoding basis.

    Encodes a categorical variable with ``n_categories`` unique labels as a
    one-hot feature matrix of shape ``(n_samples, n_categories)``. Each
    column corresponds to one category: the entry is 1 when the input equals
    that category, and 0 everywhere else.

    Parameters
    ----------
    categories :
        The set of valid category labels. Accepted forms:

        - ``int``: interpreted as the number of categories; labels default to
          ``[0, 1, ..., categories-1]``.
        - ``list`` or ``NDArray``: the explicit list of unique category labels. Note
          that the category labels will be sorted internally. Column ``i`` of the
          one-hot encoding will correspond to ``basis.categories[i]``. When a list
          is provided, it is converted to an ``NDArray`` via ``np.asarray``.
          Mixed-type lists will be cast to a common dtype (e.g., ``["a", 1]``
          becomes ``array(['a', '1'], dtype='<U21')``).

    out_of_category :
        If False, raise if labels that do not belong to ``categories`` are provided,
        else encode the out-of-category labels as all 0s.

    label :
        The label of the basis, intended to be descriptive of the task variable
        being processed. For example: ``"trial_type"``, ``"stimulus_id"``.

    Notes
    -----
    **Design matrix identifiability.**

    This basis produces a *full* encoding: one column per category. Because
    NeMoS GLMs include an intercept, including all columns of a
    ``Category`` basis as a standalone predictor introduces perfect
    collinearity — the column sum equals the intercept column. Always drop
    one column per categorical variable when using categories as main
    effects; the dropped category becomes the reference level and all
    retained coefficients are contrasts against it.

    When ``Category`` is multiplied with a continuous basis (the recommended
    use), the intercept is not involved and no column needs to be dropped.

    For a detailed discussion of identifiability, reference-level choice,
    and the effect of regularization, see the
    :ref:`identifiability guide <categorical_identifiability>`.

    Examples
    --------
    Encode a categorical variable with 3 integer labels:

    >>> import numpy as np
    >>> from nemos.basis import Category
    >>> basis = Category(3)
    >>> basis.n_basis_funcs
    3
    >>> labels = np.array([0, 1, 2, 0])
    >>> features = basis.compute_features(labels)
    >>> features.shape
    (4, 3)

    Standalone categorical predictor with reference coding (drop one column):

    >>> basis = Category(["Tri", "Sq"])
    >>> X = basis.compute_features(np.array(["Tri", "Sq", "Tri", "Sq"]))
    >>> X = X[:, 1:]  # "Tri" is the reference; remaining column is the "Sq" contrast
    >>> X.shape
    (4, 1)

    Category-specific tuning curves via basis product (no column dropping needed):

    >>> from nemos.basis import RaisedCosineLinearEval
    >>> speed = np.random.randn(20)
    >>> context = np.random.choice(["L", "R"], size=20)
    >>> bas = Category(["L", "R"]) * RaisedCosineLinearEval(5)
    >>> X = bas.compute_features(context, speed)

    """

    _convert_to_float = False
    _is_discrete = True

    def __init__(
        self,
        categories: List | NDArray | int,
        out_of_category: bool = True,
        label: Optional[str] = None,
    ):
        n_categories = self._get_n_categories(categories)
        self._label_encoder = LabelEncoder(n_categories)
        self.out_of_category = out_of_category
        self.categories = categories
        self._n_inputs = 1
        Basis.__init__(self)
        AtomicBasisMixin.__init__(
            self,
            n_basis_funcs=self.n_basis_funcs,
            label=label,
        )
        EvalBasisMixin.__init__(self, bounds=None)

    @property
    def out_of_category(self) -> int | str | float | None:
        return self._out_of_category

    @out_of_category.setter
    def out_of_category(self, out_of_category: bool):
        if not isinstance(out_of_category, bool):
            raise TypeError("``out_of_category`` must be a boolean.")
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
            n_categories = len(unique)
            if n_categories != len(cats):
                duplicates = [c for c in unique if np.sum(cats == c) > 1]
                raise ValueError(
                    f"Duplicate category labels provided: {duplicates}. "
                    "Each label must be unique."
                )
            if n_categories != self._label_encoder.n_classes:
                self._label_encoder = LabelEncoder(n_categories)
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

        Notes
        -----
        The `evaluate` method returns an array of shape ``(*xi.shape, n_basis_funcs)``.
        The method preserves the input shape and appends an extra basis axis.

        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import Category
        >>> basis = Category(3)
        >>> x = np.array([[0, 1, 2, 0], [2, 1, 0, 0]])
        >>> out = basis.evaluate(x)
        >>> out
        Array([[[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
                [1., 0., 0.]],

               [[0., 0., 1.],
                [0., 1., 0.],
                [1., 0., 0.],
                [1., 0., 0.]]], dtype=...)
        >>> x.shape, out.shape
        ((2, 4), (2, 4, 3))
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

    @add_docstring("_compute_features", EvalBasisMixin)
    def compute_features(self, xi: ArrayLike) -> FeatureMatrix:
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import Category
        >>> labels = np.array([0, 0, 2, 1])
        >>> basis = Category(3)
        >>> basis.compute_features(labels)
        Array([[1., 0., 0.],
               [1., 0., 0.],
               [0., 0., 1.],
               [0., 1., 0.]], dtype=float...)

        """
        # ruff: noqa: D205, D400
        return super().compute_features(xi)

    @add_docstring("split_by_feature", AtomicBasisMixin)
    def split_by_feature(
        self,
        x: NDArray,
        axis: int = 1,
    ):
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nemos.basis import Category
        >>> basis = Category(3, label="stimulus")
        >>> X = basis.compute_features(np.array([0, 1, 2, 0, 1]))
        >>> split_features = basis.split_by_feature(X, axis=1)
        >>> for feature, arr in split_features.items():
        ...     print(f"{feature}: shape {arr.shape}")
        stimulus: shape (5, 3)

        """
        # ruff: noqa: D205, D400
        return super().split_by_feature(x, axis=axis)

    @add_docstring("set_input_shape", AtomicBasisMixin)
    def set_input_shape(self, xi: int | tuple[int, ...] | NDArray):
        """
        Examples
        --------
        >>> import nemos as nmo
        >>> import numpy as np
        >>> basis = nmo.basis.Category(3)
        >>> # Configure with an integer input:
        >>> _ = basis.set_input_shape(1)
        >>> basis.n_output_features
        3
        >>> # Configure with a tuple:
        >>> _ = basis.set_input_shape((4, 5))
        >>> basis.n_output_features
        60

        """
        # ruff: noqa: D205, D400
        return AtomicBasisMixin.set_input_shape(self, xi)

    def evaluate_on_grid(self, *n_samples: int) -> Tuple[Tuple[NDArray], NDArray]:
        """Raise for categorical basis."""
        raise NotImplementedError(
            "``evaluate_on_grid`` is not implemented for categorical basis. The method "
            "only makes sense for continuous bases, ``Category`` process discrete "
            "inputs only."
        )
