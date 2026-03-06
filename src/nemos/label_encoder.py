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

    def __init__(self, n_classes: int):
        self._n_classes = n_classes
        # set attrs explicitly to satisfy static checkers
        # (kept in sync with _reset_attrs via dataclass fields)
        self._skip_encoding: bool = _ResetAttrs._skip_encoding
        self._class_to_index_: dict | None = _ResetAttrs._class_to_index_
        self.classes_: np.ndarray | jnp.ndarray | None = _ResetAttrs.classes_

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
        for f in fields(_ResetAttrs):
            setattr(self, f.name, getattr(_ResetAttrs, f.name))

    def set_classes(self, classes_array: jnp.ndarray | ArrayLike):
        """
        Infer unique class labels and set the ``classes_`` attribute.

        This method infers class labels from ``classes_array`` and sets up the internal
        encoding/decoding machinery. When labels are the default ``[0, 1, ..., n_classes-1]``,
        encoding is skipped for performance.

        Parameters
        ----------
        classes_array
            An array that must contain all the class labels,
            i.e. ``len(np.unique(classes_array)) == n_classes``.

        Raises
        ------
        ValueError
            If the number of unique class labels in ``classes_array`` does not match ``n_classes``.

        Notes
        -----
        ``set_classes`` must be called before :meth:`encode` or :meth:`decode`.
        When fitting in batches, call ``set_classes`` with an array containing all class labels
        before starting the update loop, since individual batches may not contain every class.

        """
        if isinstance(classes_array, jnp.ndarray):
            classes = jnp.unique(classes_array)
        else:
            classes = np.unique(classes_array)
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
            self._class_to_index_ = None
        else:
            self._class_to_index_ = {label.item(): i for i, label in enumerate(classes)}

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

        Notes
        -----
        Dispatch is based on array type (transparent to the caller) and ``safe``:

        - If ``classes_`` equals ``[0, 1, ..., n_classes-1]`` and ``safe=False``,
          ``y`` is returned unchanged. If ``safe=True``, the dtype is checked
          (must be integer) and values are validated against ``[0, n_classes-1]``.
        - Otherwise:

          - **numpy, safe=True**: dict lookup via ``np.fromiter``; raises
            ``ValueError`` on unrecognized labels; works for any hashable type.
          - **numpy, safe=False**: ``np.searchsorted``; fastest numpy path; requires
            orderable labels; silently maps unknown labels to nearest indices.
          - **JAX, safe=True**: ``jnp.unique`` + numpy set difference; raises
            ``KeyError`` on unrecognized labels; not JIT-compilable.
          - **JAX, safe=False**: ``jnp.searchsorted`` on device; JIT-compilable;
            silently maps unknown labels to nearest indices.

        Use ``safe=True`` at system boundaries where labels come from user input.
        Use ``safe=False`` in hot paths where labels are guaranteed to be a subset
        of ``classes_`` — for example, inside an :meth:`update` loop after
        :meth:`set_classes` has validated the full label set.
        """
        if self._skip_encoding:
            # always return y as provided by the user.
            # If safe==True, check for category validity.
            if safe:
                # check that the labels are ints in [0,..., n_classes-1]
                dtype = getattr(y, "dtype", None)
                if dtype is None:
                    # probably a list, tuple or number
                    y_array = np.array(y)
                    dtype = y_array.dtype
                else:
                    # already an array (either jax or numpy)
                    y_array = y
                is_all_int = np.issubdtype(dtype, np.integer) or (
                    np.issubdtype(dtype, np.floating)
                    and (y_array == y_array.astype(int)).all()
                )
                if not is_all_int:
                    raise ValueError(
                        f"Expected integer labels when classes are the default "
                        f"[0, ..., n_classes-1], got dtype {dtype} instead."
                    )
                invalid_mask = (y_array < 0) | (y_array >= self.n_classes)
                if invalid_mask.any():
                    if isinstance(y_array, jnp.ndarray):
                        unique = jnp.unique
                    else:
                        unique = np.unique
                    invalid = unique(y_array[invalid_mask]).tolist()
                    raise ValueError(
                        f"Unrecognized label(s) {invalid}. "
                        f"Valid labels are {list(range(self.n_classes))}."
                    )
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

    def _encode_jax(self, y: jnp.ndarray, safe: bool = True) -> ArrayLike:
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
            # check based on type:
            # - y is a jax.ndarray -> numeric
            # - classes_ np.array but not numeric
            if not np.issubdtype(self.classes_.dtype, np.number):
                invalid = jnp.unique(y).tolist()
                raise KeyError(
                    f"Unrecognized label(s) {invalid}. Valid labels are {self.classes_.tolist()}."
                )
            # continue execution checking the content of numeric arrays
            invalid_mask = ~jnp.isin(y, self.classes_)
            if invalid_mask.any():
                invalid = np.unique(y[invalid_mask]).tolist()
                raise KeyError(
                    f"Unrecognized label(s) {invalid}. Valid labels are {self.classes_.tolist()}."
                )
        y_encoded = jnp.searchsorted(self.classes_, y)
        return y_encoded

    def __repr__(self):
        """Represent encoder object.

        Notes
        -----
        This class is for internal use only, a simple repr is sufficient.
        Using `utils.format_repr` requires inheriting `BaseRegressor` which is an overkill.
        """
        cls_name = self.__class__.__name__
        if self.classes_ is None:
            args = f"(n_classes={self.n_classes})"
        else:
            args = (
                f"(\n    n_classes={self.n_classes},\n    classes_={self.classes_}\n)"
            )
        return cls_name + args
