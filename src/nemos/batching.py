"""Data loading utilities for stochastic optimization."""

from itertools import chain
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

from .tree_utils import get_valid_multitree

BatchData: TypeAlias = tuple[Any, ...]

#: Supported shuffle strategies (see ``_BaseArrayDataLoader``).
SHUFFLE_STRATEGIES: tuple[str, ...] = ("none", "chunk", "full")


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
    - ``sample_batch()`` should be cheap and deterministic (e.g., the first batch
      that contains valid data).
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
        Typically returns the first batch that contains valid (non-NaN/Inf) data.
        """
        ...


class _BaseArrayDataLoader:
    """
    Shared implementation for array-backed data loaders.

    Subclasses convert/store the source arrays in their own ``__init__`` (eagerly to
    JAX for in-memory data, or lazily for out-of-core data) and implement
    ``_iter_shuffle_whole``, which differs by whether the backend needs sorted
    indices for fancy indexing.

    Three shuffle strategies are supported (see ``SHUFFLE_STRATEGIES``):

    - ``"none"``: sequential, contiguous batches.
    - ``"chunk"``: contiguous batches whose order is shuffled each epoch, with the
      samples permuted within each batch. Batch membership is fixed, so this only
      requires contiguous reads.
    - ``"full"``: every sample may land in any batch (requires random access).
    """

    def __init__(
        self,
        *arrays: ArrayLike,
        batch_size: int,
        shuffle: str,
        seed: int | None = None,
    ):
        """
        Initialize an array-backed data loader.

        Parameters
        ----------
        *arrays :
            Input and output arrays (any number), each with the same number of
            samples along axis 0.
        batch_size :
            Number of samples per batch.
        shuffle :
            Shuffle strategy: one of ``"none"``, ``"chunk"``, or ``"full"``.
        seed :
            Random seed for shuffling. Default is None.

        Raises
        ------
        ValueError
            If no arrays are provided, the arrays have mismatched lengths, or
            ``shuffle`` is not a recognized strategy.
        """
        if len(arrays) == 0:
            raise ValueError("Provide at least one array.")

        if shuffle not in SHUFFLE_STRATEGIES:
            raise ValueError(
                f"shuffle must be one of {SHUFFLE_STRATEGIES}, got {shuffle!r}."
            )

        self.arrays = tuple(arrays)

        if len(set(arr.shape[0] for arr in self.arrays)) != 1:
            raise ValueError("All arrays must have same number of samples")

        self.shuffle = shuffle
        self._rng = np.random.default_rng(seed)
        self.batch_size = batch_size

    @property
    def n_samples(self) -> int:
        """Total number of samples in the dataset."""
        return self.arrays[0].shape[0]

    @property
    def batch_size(self) -> int:
        """Number of samples in each batch."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, val: int):
        if val <= 0:
            raise ValueError("batch_size must be positive.")
        if val > self.n_samples:
            raise ValueError("batch_size cannot be larger than the number of samples.")

        self._batch_size = val

    def _materialize(self, key: slice | np.ndarray) -> tuple[jnp.ndarray, ...]:
        """Load and convert to JAX the rows selected by ``key`` from each array."""
        return tuple(jnp.asarray(arr[key]) for arr in self.arrays)

    def _iter_no_shuffle(self) -> Iterator[tuple[jnp.ndarray, ...]]:
        """Yield contiguous batches in sequential order."""
        n = self.n_samples
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            yield self._materialize(slice(start, end))

    def _iter_shuffle_chunks(self) -> Iterator[tuple[jnp.ndarray, ...]]:
        """Yield contiguous batches in shuffled order, permuted within each batch."""
        n = self.n_samples
        chunks = [
            (start, min(start + self.batch_size, n))
            for start in range(0, n, self.batch_size)
        ]
        self._rng.shuffle(chunks)

        for start, end in chunks:
            batch = self._materialize(slice(start, end))

            local_perm = self._rng.permutation(end - start)
            batch = tuple(b[local_perm] for b in batch)

            yield batch

    def _iter_shuffle_whole(self) -> Iterator[tuple[jnp.ndarray, ...]]:
        """Yield batches drawn from a full permutation of the samples."""
        raise NotImplementedError

    def __iter__(self) -> Iterator[tuple[jnp.ndarray, ...]]:
        """Return a fresh iterator over batches, dispatching on the shuffle strategy."""
        match self.shuffle:
            case "none":
                return self._iter_no_shuffle()
            case "chunk":
                return self._iter_shuffle_chunks()
            case "full":
                return self._iter_shuffle_whole()
            case _:
                raise ValueError(
                    f"shuffle must be one of {SHUFFLE_STRATEGIES}, got {self.shuffle!r}."
                )

    def sample_batch(self) -> tuple[jnp.ndarray, ...]:
        """
        Return the first batch containing at least one valid sample.

        Scans contiguous batches in sequential order (ignoring shuffle) and returns
        the first one with a sample that is finite (no NaN/Inf) across every array.
        This skips leading all-invalid regions, such as the NaN warmup of
        convolutional-basis features, while loading only as much data as needed.

        Validity uses the same ``get_valid_multitree`` notion as ``validate_inputs``
        and ``_preprocess_inputs``, so an accepted batch is exactly one that passes
        validation and stays non-empty after NaN-dropping.

        Returns
        -------
        :
            The first batch with a valid sample. If no batch has one (the whole
            dataset is invalid), the first batch is returned so that downstream
            validation raises the canonical "all samples invalid" error.
        """
        batches = self._iter_no_shuffle()

        # save first batch in case all batches are invalid
        first_batch = next(batches)

        for batch in chain([first_batch], batches):
            if bool(get_valid_multitree(*batch).any()):
                return batch

        # return invalid first batch and let downstream fail
        return first_batch


class ArrayDataLoader(_BaseArrayDataLoader):
    """
    DataLoader for in-memory arrays.

    Arrays are eagerly converted to JAX arrays at construction. This loader is
    re-iterable: each call to ``__iter__()`` returns a fresh iterator.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from nemos.batching import ArrayDataLoader
    >>> X = jnp.ones((100, 5))
    >>> y = jnp.ones((100,))
    >>> loader = ArrayDataLoader(X, y, batch_size=32, shuffle="full")
    >>> for X_batch, y_batch in loader:
    ...     pass  # Train on batch
    """

    def __init__(
        self,
        *arrays: ArrayLike,
        batch_size: int,
        shuffle: str = "full",
        seed: int | None = None,
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
            Shuffle strategy: one of ``"none"``, ``"chunk"``, or ``"full"``.
            Default is ``"full"``.
        seed :
            Random seed for shuffling. Default is None.
        """
        super().__init__(
            *(jnp.asarray(x) for x in arrays),
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )

    def _iter_shuffle_whole(self) -> Iterator[tuple[jnp.ndarray, ...]]:
        """Yield batches from a full permutation via direct fancy indexing."""
        n = self.n_samples
        perm = self._rng.permutation(n)

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            yield self._materialize(perm[start:end])


class LazyArrayDataLoader(_BaseArrayDataLoader):
    """
    DataLoader for lazy/out-of-core arrays (e.g. dask, zarr, HDF5).

    Unlike ``ArrayDataLoader``, this loader does not eagerly convert arrays to JAX
    arrays. Instead, it reads slices from the source arrays and converts each batch
    to JAX on the fly. This keeps memory usage proportional to batch size rather
    than dataset size.

    The default shuffle strategy is ``"chunk"`` (approximate): chunk order is
    randomized each epoch and samples within each batch are permuted after loading,
    but samples within the same chunk always end up in the same batch. Passing
    ``shuffle="full"`` shuffles the whole dataset like ``ArrayDataLoader``. This
    requires the arrays to support fancy indexing and may be slower than reading
    contiguous segments. Indices within each batch are sorted to support HDF5
    arrays via h5py.

    This loader is re-iterable: each call to ``__iter__()`` returns a fresh
    iterator.

    Examples
    --------
    >>> import numpy as np
    >>> from nemos.batching import LazyArrayDataLoader
    >>> X = np.ones((100, 5))
    >>> y = np.ones((100,))
    >>> loader = LazyArrayDataLoader(X, y, batch_size=32, shuffle="chunk")
    >>> for X_batch, y_batch in loader:
    ...     pass  # Train on batch
    """

    def __init__(
        self,
        *arrays: ArrayLike,
        batch_size: int,
        shuffle: str = "chunk",
        seed: int | None = None,
    ):
        """
        Initialize a lazy array data loader.

        Parameters
        ----------
        *arrays :
            Input and output arrays (any number). Each must support
            ``.shape`` and sequential slicing (``arr[start:end]``). The ``"full"``
            strategy additionally requires support for fancy indexing.
        batch_size :
            Number of samples per batch.
        shuffle :
            Shuffle strategy: one of ``"none"``, ``"chunk"``, or ``"full"``.
            Default is ``"chunk"``.
        seed :
            Random seed for shuffling. Default is None.
        """
        super().__init__(*arrays, batch_size=batch_size, shuffle=shuffle, seed=seed)

    def _iter_shuffle_whole(self) -> Iterator[tuple[jnp.ndarray, ...]]:
        """Yield batches from a full permutation, reading in sorted index order."""
        n = self.n_samples
        perm = self._rng.permutation(n)

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch_idx = perm[start:end]

            # Sort indices so backends that require monotonic fancy indexing
            # (e.g. h5py) work, then restore the shuffled order within the batch.
            # The double argsort gives the rank of each element,
            # i.e. where it ends up after sorting.
            local_perm = np.argsort(np.argsort(batch_idx))
            batch = self._materialize(np.sort(batch_idx))
            yield tuple(b[local_perm] for b in batch)


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


def is_data_loader(obj) -> bool:
    """Check if an object conforms to the DataLoader protocol."""
    return isinstance(obj, DataLoader)
