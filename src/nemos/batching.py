"""Data loading utilities for stochastic optimization."""

from typing import Callable, Iterator, Optional, Protocol, Tuple, runtime_checkable

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike


@runtime_checkable
class DataLoader(Protocol):
    """
    Protocol for data loaders that stream batches.

    Requirements:

    - Must be re-iterable: calling ``__iter__()`` must return a fresh iterator
      each time. This is required for ``num_epochs > 1`` and for SVRG's full
      gradient computation.
    - ``sample_batch()`` should be cheap and deterministic (e.g., return first batch).
    - Batches should have consistent, non-zero sizes.
    """

    def __iter__(self) -> Iterator[Tuple[ArrayLike, ArrayLike]]:
        """
        Iterate over (X_batch, y_batch) tuples.

        Must return a fresh iterator each call (re-iterable).
        """
        ...

    @property
    def n_samples(self) -> int:
        """Total number of samples in the dataset."""
        ...

    def sample_batch(self) -> Tuple[ArrayLike, ArrayLike]:
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
    X :
        Input features, array of shape (n_samples, n_features).
    y :
        Target values, array of shape (n_samples,) or (n_samples, n_outputs).
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
        X: ArrayLike,
        y: ArrayLike,
        batch_size: int,
        shuffle: bool = True,
        key: Optional[jax.Array] = None,
    ):
        self.X = jnp.asarray(X)
        self.y = jnp.asarray(y)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._key = key if key is not None else jax.random.key(0)

        # Validate
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

    @property
    def n_samples(self) -> int:
        """Total number of samples in the dataset."""
        return self.X.shape[0]

    def sample_batch(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Return first batch, deterministic (ignores shuffle)."""
        end = min(self.batch_size, self.n_samples)
        return self.X[:end], self.y[:end]

    def __iter__(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Return fresh iterator. Shuffles if enabled."""
        n = self.n_samples
        if self.shuffle:
            self._key, subkey = jax.random.split(self._key)
            perm = jax.random.permutation(subkey, n)
            X, y = self.X[perm], self.y[perm]
        else:
            X, y = self.X, self.y

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            yield X[start:end], y[start:end]


class _PreprocessedDataLoader:
    """
    Wraps a DataLoader to preprocess batches on-the-fly.

    Used internally by ``GLM.stochastic_fit`` to apply preprocessing
    (e.g., NaN dropping, type casting) to each batch.
    """

    def __init__(
        self,
        loader: DataLoader,
        preprocessing_func: Callable[
            [ArrayLike, ArrayLike],
            Tuple[dict[str, jnp.ndarray] | jnp.ndarray, jnp.ndarray],
        ],
    ):
        self._loader = loader
        self._preprocess_fn = preprocessing_func
        self._cached_sample: Optional[
            Tuple[dict[str, jnp.ndarray] | jnp.ndarray, jnp.ndarray]
        ] = None

    @property
    def n_samples(self) -> int:
        """Total number of samples in the dataset."""
        return self._loader.n_samples

    def sample_batch(
        self,
    ) -> Tuple[dict[str, jnp.ndarray] | jnp.ndarray, jnp.ndarray]:
        """Return cached preprocessed sample batch."""
        if self._cached_sample is None:
            raw_X, raw_y = self._loader.sample_batch()
            self._cached_sample = self._preprocess_fn(raw_X, raw_y)
        return self._cached_sample

    def __iter__(
        self,
    ) -> Iterator[Tuple[dict[str, jnp.ndarray] | jnp.ndarray, jnp.ndarray]]:
        """Iterate with preprocessing applied to each batch."""
        for X_batch, y_batch in self._loader:
            yield self._preprocess_fn(X_batch, y_batch)


def is_data_loader(obj) -> bool:
    """Check if an object conforms to the DataLoader protocol."""
    return isinstance(obj, DataLoader)
