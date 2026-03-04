"""
Label encoder and decoder class.

This class handles labels and should be used across models dealing with
categorical variables. The class is for internal use, main method will be
`encode` and `decode`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass
class _ResetAttrs:
    _skip_encoding: bool = False
    _class_to_index_: dict | None = None
    classes_: NDArray | jnp.ndarray | None = None


class LabelEncoder:
    """Label encoder and decoder class."""

    _reset_attrs = _ResetAttrs()

    def __init__(self, n_classes: int):
        self._n_classes = n_classes
        # set attrs explicitly to satisfy static checkers
        # (kept in sync with _reset_attrs via dataclass fields)
        self._skip_encoding: bool = self._reset_attrs._skip_encoding
        self._class_to_index_: dict | None = self._reset_attrs._class_to_index_
        self.classes_: np.ndarray | jnp.ndarray | None = self._reset_attrs.classes_

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
        for f in fields(self._reset_attrs):
            setattr(self, f.name, getattr(self._reset_attrs, f.name))

    def set_classes(self, array: jnp.ndarray | ArrayLike):
        """
        Infer unique class labels and set the ``classes_`` attribute.

        This method infers class labels from ``array`` and sets up the internal
        encoding/decoding machinery. When labels are the default ``[0, 1, ..., n_classes-1]``,
        encoding is skipped for performance.

        Parameters
        ----------
        array
            An array that must contain all the class labels,
            i.e. ``len(np.unique(array)) == n_classes``.

        Raises
        ------
        ValueError
            If the number of unique class labels in ``array`` does not match ``n_classes``.

        Notes
        -----
        ``set_classes`` must be called before :meth:`encode` or :meth:`decode`.
        When fitting in batches, call ``set_classes`` with an array containing all class labels
        before starting the update loop, since individual batches may not contain every class.

        """
        if isinstance(array, jnp.ndarray):
            classes = jnp.unique(array)
        else:
            classes = np.unique(array)
        n_unique = len(classes)

        # Validation
        if n_unique > self.n_classes:
            raise ValueError(
                f"Found {n_unique} unique class labels in array, but n_classes={self.n_classes}. "
                f"Increase n_classes or check your data."
            )
        elif n_unique < self.n_classes:
            raise ValueError(
                f"Found only {n_unique} unique class labels in array, but n_classes={self.n_classes}. "
                f"To correctly set the ``classes_`` attribute, provide an array containing all the "
                f"unique class labels.",
            )

        # Always store the actual classes array
        self.classes_ = classes

        self._skip_encoding = np.array_equal(classes, np.arange(self.n_classes))
        # Create dict lookup only when needed (non-default classes)
        if self._skip_encoding:
            self._class_to_index_ = {label.item(): i for i, label in enumerate(classes)}
        else:
            self._class_to_index_ = None

    def encode(self, y: ArrayLike, safe=True) -> NDArray[int]:
        """
        Convert user-provided class labels to internal indices ``[0, n_classes-1]``.

        When labels are the default ``[0, 1, ..., n_classes-1]``, the input is
        returned unchanged. Otherwise, dispatches to a numpy or JAX backend
        depending on the type of ``y``.

        Parameters
        ----------
        y :
            Array of class labels to encode.
        safe :
            If ``True`` (default), validate that all labels in ``y`` are known and
            raise on any mismatch. If ``False``, skip validation for performance:
            unknown labels will silently produce incorrect indices. Set to ``False``
            only when the caller guarantees that ``y`` contains only valid labels,
            e.g. in JIT-compiled JAX functions or inner loops where labels have
            already been validated upstream by :meth:`set_classes`.

        Returns
        -------
        :
            Integer array of indices in ``[0, n_classes-1]``.

        Raises
        ------
        ValueError
            If ``safe=True`` and ``y`` contains unrecognized labels (numpy path).
        KeyError
            If ``safe=True`` and ``y`` contains unrecognized labels (JAX path).
        """
        if self._skip_encoding:
            return y
        if isinstance(y, jnp.ndarray):
            y = self._encode_jax(y, safe=safe)
        else:
            # use dict lookup instead of `np.searchsorted`
            # this approach will fail for label mismatches
            # always safe
            y = self._encode_numpy(y, safe=safe)

        return y

    def decode(self, indices: NDArray[int]) -> NDArray | jnp.ndarray:
        """Convert internal indices [0, n_classes-1] back to user-provided class labels."""
        if self._skip_encoding:
            return indices
        return self.classes_[indices]

    def check_classes_is_set(self, method_name: str):
        """Check if the class labels are set, otherwise raise an error."""
        if self.classes_ is None:
            raise RuntimeError(
                f"Classes are not set. Must call ``set_classes`` before calling ``{method_name}``."
            )

    def _encode_numpy(self, y: ArrayLike, safe: bool = True) -> ArrayLike:
        """
        Encode labels for numpy arrays.

        Parameters
        ----------
        y :
            Array of class labels to encode.
        safe :
            If ``True``, use dict-based lookup via ``np.fromiter``, which raises
            ``ValueError`` on any unrecognized label. The dict lookup is faster
            than ``np.searchsorted`` for typical label set sizes.
            If ``False``, use ``np.searchsorted`` directly — faster but requires
            labels to be orderable (supports ``<``), and silently maps unknown
            labels to nearest indices.
        """
        if safe:
            # use from iter, this forces safety while being much more
            # efficient than a unique.
            try:
                y = np.asarray(y)
                original_shape = y.shape
                y = np.fromiter(
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
        else:
            # fastest way to return the encoded indices.
            return np.searchsorted(self.classes_, y)

    def _encode_jax(self, y: ArrayLike, safe: bool = True) -> ArrayLike:
        """
        Encode labels for JAX arrays.

        Always uses ``jnp.searchsorted``, which is vectorized and stays on device.
        ``jnp.searchsorted`` does not raise on unrecognized labels — it silently
        maps them to nearest indices — so an optional safeguard is provided.

        Parameters
        ----------
        y :
            JAX array of class labels to encode.
        safe :
            If ``True``, compute unique values of ``y``, convert to a numpy array,
            and compare against ``classes_`` to detect any unrecognized labels.
            This check is not JIT-compilable. If ``False``, skip validation
            entirely: the encoding is JIT-compilable, but unknown labels will
            produce silently incorrect indices.
        """
        if safe:
            # expensive safeguard: JAX dispatch overhead + numpy conversion
            y_unq = np.asarray(jnp.unique(y))
            valid = np.asarray(self.classes_).tolist()
            invalid = set(y_unq).difference(valid)
            if len(invalid):
                invalid = [int(lab) for lab in invalid]
                raise KeyError(
                    f"Unrecognized label(s) {invalid}. " f"Valid labels are {valid}."
                )
        y_encoded = jnp.searchsorted(self.classes_, y)
        return y_encoded
