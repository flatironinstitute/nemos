"""
Label encoder and decoder class.

This class handles labels and should be used across models dealing with
categorical variables. The class is for internal use, main method will be
`encode` and `decode`.
"""

from typing import Dict

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray


class LabelEncoder:
    """Label encoder and decoder class."""

    _reset_attrs = (
        ("_skip_encoding", False),
        ("_class_to_index_", None),
        ("classes_", None),
    )

    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        # skip encoding if classes are
        # integers 0,...,n-1
        self._skip_encoding = False
        # hash-map linking
        self._class_to_index_: Dict | None = None
        self.classes_: np.ndarray | NDArray | None = None

    @property
    def n_classes(self) -> int:
        """Number of unique class labels."""
        return self._n_classes

    @n_classes.setter
    def n_classes(self, n_classes: int):
        self.reset()
        self._n_classes = n_classes

    def reset(self):
        """Reset cached attributes to default values."""
        for attr, val in self._reset_attrs:
            setattr(self, attr, val)

    def set_classes(self, array: jnp.ndarray | ArrayLike):
        """
        Infer unique class labels and set the ``classes_`` attribute.

        This method infers class labels from ``y`` and sets up the internal
        encoding/decoding machinery. When labels are the default ``[0, 1, ..., n_classes-1]``,
        encoding is skipped for performance.

        Parameters
        ----------
        y
            An array that must contain all the class labels,
            i.e. ``len(np.unique(y)) == n_classes``.

        Raises
        ------
        ValueError
            If the number of unique class labels in ``y`` does not match ``n_classes``.

        Notes
        -----
        :meth:`fit` and :meth:`initialize_solver_and_state` call ``set_classes`` internally,
        making sure that the ``classes_`` attribute matches the provided input.
        If you are fitting in batches by calling :meth:`update`, make sure that the ``classes_``
        are correctly set by calling ``set_classes`` before starting the :meth:`update` loop.

        """
        if isinstance(array, jnp.ndarray):
            classes = jnp.unique(array)
        else:
            classes = np.unique(array)
        n_unique = len(classes)

        # Validation
        if n_unique > self.n_classes:
            raise ValueError(
                f"Found {n_unique} unique class labels in y, but n_classes={self.n_classes}. "
                f"Increase n_classes or check your data."
            )
        elif n_unique < self.n_classes:
            raise ValueError(
                f"Found only {n_unique} unique class labels in y, but n_classes={self.n_classes}. "
                f"To correctly set the ``classes_`` attribute, provide an array containing all the "
                f"unique class labels.",
            )

        self._skip_encoding = np.array_equal(classes, np.arange(self.n_classes))
        # Create dict lookup only when needed (non-default classes)
        self._class_to_index_ = (
            None
            if self._skip_encoding
            else {label: i for i, label in enumerate(classes)}
        )

    def encode(self, y: ArrayLike) -> NDArray[int]:
        """Convert user-provided class labels to internal indices [0, n_classes-1]."""
        if self._skip_encoding:
            return y
        # use dict lookup instead of `np.searchsorted`
        # this approach will fail for label mismatches
        if isinstance(y, jnp.ndarray):
            asarray = jnp.asarray
            fromiter = jnp.fromiter
        else:
            asarray = np.array
            fromiter = np.fromiter
        try:
            y = asarray(y)
            original_shape = y.shape
            y = fromiter(
                (self._class_to_index_[label] for label in y.ravel()),
                dtype=int,
                count=y.size,
            ).reshape(original_shape)
        except KeyError as e:
            unq_labels = np.unique(y)
            valid = list(self._class_to_index_.keys())
            invalid = [lab for lab in unq_labels if lab not in valid]
            raise ValueError(
                f"Unrecognized label(s) {invalid}. " f"Valid labels are {valid}."
            ) from e
        return y

    def decode(self, indices: NDArray[int]) -> NDArray | jnp.ndarray:
        """Convert internal indices [0, n_classes-1] back to user-provided class labels."""
        if self._skip_encoding:
            return indices
        return self.classes_[indices]
