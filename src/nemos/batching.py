"""Data loading utilities for stochastic optimization."""

from typing import (
    Any,
    Callable,
    Iterator,
    Optional,
    Protocol,
    TypeAlias,
    runtime_checkable,
)

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike

BatchData: TypeAlias = tuple[Any, ...]


@runtime_checkable
class DataLoader(Protocol):
    """
    Protocol for data loaders that stream batches.

    The protocol itself allows batches as tuples of any length,
    but note that ``GLM.stochastic_fit`` expects ``(X, y)`` pairs.
    The variadic batch format is used on the solver-level
    via ``AbstractSolver.stochastic_run``.

    Requirements:

    - Must be re-iterable: calling ``__iter__()`` must return a fresh iterator
      each time. This is required for ``num_epochs > 1`` and for SVRG's full
      gradient computation.
    - ``sample_batch()`` should be cheap and deterministic (e.g., return first batch).
    - Batches should have consistent, non-zero sizes.
    """

    def __iter__(self) -> Iterator[BatchData]:
        """
        Iterate over tuples containing input and output data, e.g. (X_batch, y_batch).

        Must return a fresh iterator each call (re-iterable).
        """
        ...

    @property
    def n_samples(self) -> int:
        """Total number of samples in the dataset."""
        ...

    def sample_batch(self) -> BatchData:
        """
        Return a single batch for initialization purposes.

        Should be cheap/cached and deterministic (ignore shuffle setting).
        Typically returns the first batch of data.
        """
        ...


class ArrayDataLoader:
    """
    DataLoader for in-memory arrays.

    This loader is re-iterable: each call to ``__iter__()`` returns a fresh iterator.

    Parameters
    ----------
    arrays :
        Input and output arrays (any number), each an array of
        shape (n_samples, n_features) or (n_samples, ).
    batch_size :
        Number of samples per batch.
    shuffle :
        Whether to shuffle data each epoch. Default is True.
    key :
        JAX random key for shuffling. Default is ``jax.random.key(0)``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from nemos.batching import ArrayDataLoader
    >>> X = jnp.ones((100, 5))
    >>> y = jnp.ones((100,))
    >>> loader = ArrayDataLoader(X, y, batch_size=32, shuffle=True)
    >>> for X_batch, y_batch in loader:
    ...     pass  # Train on batch
    """

    def __init__(
        self,
        *arrays: ArrayLike,
        batch_size: int,
        shuffle: bool = True,
        key: Optional[jax.Array] = None,
    ):
        if len(arrays) == 0:
            raise ValueError("Provide at least one array.")

        self.arrays = tuple(jnp.asarray(x) for x in arrays)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._key = key if key is not None else jax.random.key(0)

        # Validate
        if len(set(arr.shape[0] for arr in self.arrays)) != 1:
            raise ValueError("All arrays must have same number of samples")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

    @property
    def n_samples(self) -> int:
        """Total number of samples in the dataset."""
        return self.arrays[0].shape[0]

    def sample_batch(self) -> tuple[jnp.ndarray, ...]:
        """Return first batch, deterministic (ignores shuffle)."""
        end = min(self.batch_size, self.n_samples)
        return tuple(arr[:end] for arr in self.arrays)

    def __iter__(self) -> Iterator[tuple[jnp.ndarray, ...]]:
        """Return fresh iterator. Shuffles if enabled."""
        n = self.n_samples
        if self.shuffle:
            self._key, subkey = jax.random.split(self._key)
            perm = jax.random.permutation(subkey, n)
        else:
            perm = None

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)

            if perm is None:
                yield tuple(arr[start:end] for arr in self.arrays)
            else:
                idx = perm[start:end]
                yield tuple(arr[idx] for arr in self.arrays)


class _PreprocessedDataLoader:
    """
    Wraps a DataLoader to preprocess batches on-the-fly.

    Used internally by ``GLM.stochastic_fit`` to apply preprocessing
    (e.g., NaN dropping, type casting) to each batch.
    """

    def __init__(
        self,
        loader: DataLoader,
        preprocessing_func: Callable[..., BatchData],
    ):
        self._loader = loader
        self._preprocess_fn = preprocessing_func
        self._cached_sample: Optional[BatchData] = None

    @property
    def n_samples(self) -> int:
        """Total number of samples in the dataset."""
        return self._loader.n_samples

    def sample_batch(self) -> BatchData:
        """Return cached preprocessed sample batch."""
        if self._cached_sample is None:
            raw_batch_data = self._loader.sample_batch()
            self._cached_sample = self._preprocess_fn(*raw_batch_data)
        return self._cached_sample

    def __iter__(self) -> Iterator[BatchData]:
        """Iterate with preprocessing applied to each batch."""
        for batch_data in self._loader:
            yield self._preprocess_fn(*batch_data)


def is_data_loader(obj) -> bool:
    """Check if an object conforms to the DataLoader protocol."""
    return isinstance(obj, DataLoader)
