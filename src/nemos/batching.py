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

import jax.numpy as jnp
import numpy as np
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
      each time. This is required for ``num_epochs > 1`` and because SVRG's full
      gradient computation iterates through the data an additional time per epoch.
    - ``sample_batch()`` should be cheap and deterministic (e.g., return first batch).
    - Batches should have consistent, non-zero sizes. Note that the solver's ``update``
      method will be recompiled for each unique batch size. This usually means just 2
      compilations, as the last batch is almost always of a different size unless the
      number of samples is divisible by the batch size.
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


# TODO: How can this work with pynapple's lazy laoding?
class ArrayDataLoader:
    """
    DataLoader for in-memory arrays.

    This loader is re-iterable: each call to ``__iter__()`` returns a fresh iterator.

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
        seed: int = 0,
    ):
        """
        Initialize an in-memory array data loader.

        Parameters
        ----------
        *arrays :
            Input and output arrays (any number), each an array of
            shape (n_samples, n_features) or (n_samples, ).
        batch_size :
            Number of samples per batch.
        shuffle :
            Whether to shuffle data each epoch. Default is True.
        seed :
            Random seed for shuffling. Default is 0.
        """
        if len(arrays) == 0:
            raise ValueError("Provide at least one array.")

        self.arrays = tuple(jnp.asarray(x) for x in arrays)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._rng = np.random.default_rng(seed)

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
            perm = self._rng.permutation(n)
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
        """
        Initialize a preprocessed data loader.

        Parameters
        ----------
        loader :
            The underlying data loader to wrap.
        preprocessing_func :
            Function applied to each batch. Called as
            ``preprocessing_func(*batch_data)``.
        """
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


# TODO: Is n_samples required?


class LazyArrayDataLoader:
    """
    DataLoader for lazy/out-of-core arrays (e.g. dask, zarr, HDF5).

    Unlike ``ArrayDataLoader``, this loader does not eagerly convert arrays
    to JAX arrays. Instead, it reads sequential slices from the source arrays
    and converts each batch to JAX on the fly. This keeps memory usage
    proportional to batch size rather than dataset size.

    Shuffling is approximate: chunk order is randomized each epoch and
    samples within each batch are permuted after loading. Samples within
    the same chunk always end up in the same batch.

    This loader is re-iterable: each call to ``__iter__()`` returns a fresh
    iterator.

    Examples
    --------
    >>> import numpy as np
    >>> from nemos.batching import LazyArrayDataLoader
    >>> X = np.ones((100, 5))
    >>> y = np.ones((100,))
    >>> loader = LazyArrayDataLoader(X, y, batch_size=32, shuffle=True)
    >>> for X_batch, y_batch in loader:
    ...     pass  # Train on batch
    """

    def __init__(
        self,
        *arrays: ArrayLike,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
    ):
        """
        Initialize a lazy array data loader.

        Parameters
        ----------
        *arrays :
            Input and output arrays (any number). Each must support
            ``.shape`` and sequential slicing (``arr[start:end]``).
        batch_size :
            Number of samples per batch.
        shuffle :
            Whether to shuffle chunk order and within-batch sample order
            each epoch. Default is True.
        seed :
            Random seed for shuffling. Default is 0.
        """
        if len(arrays) == 0:
            raise ValueError("Provide at least one array.")

        self.arrays = arrays
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._rng = np.random.default_rng(seed)

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
        return tuple(jnp.asarray(arr[:end]) for arr in self.arrays)

    def __iter__(self) -> Iterator[tuple[jnp.ndarray, ...]]:
        """Return fresh iterator. Shuffles chunk order and within-batch if enabled."""
        n = self.n_samples
        chunks = [
            (start, min(start + self.batch_size, n))
            for start in range(0, n, self.batch_size)
        ]

        if self.shuffle:
            self._rng.shuffle(chunks)

        for start, end in chunks:
            batch = tuple(jnp.asarray(arr[start:end]) for arr in self.arrays)

            if self.shuffle:
                local_perm = self._rng.permutation(end - start)
                batch = tuple(b[local_perm] for b in batch)

            yield batch


def is_data_loader(obj) -> bool:
    """Check if an object conforms to the DataLoader protocol."""
    return isinstance(obj, DataLoader)
