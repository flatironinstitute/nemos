"""Tests for the dataloaders used for mini-batches in stochastic optimization."""

from typing import ClassVar
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nemos.batching import (
    ArrayDataLoader,
    LazyArrayDataLoader,
    _PreprocessedDataLoader,
    is_data_loader,
)

N_SAMPLES = 1_000


def ordered_data():
    """Generate 2 arrays with ordered integer values."""

    X = np.arange(N_SAMPLES * 5).reshape(N_SAMPLES, 5)
    y = np.arange(N_SAMPLES)

    return (X, y)


def random_data(seed=None):
    """Generate 3 arrays with random values."""

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N_SAMPLES, 5))
    y = rng.standard_normal(N_SAMPLES)
    z = rng.standard_normal((N_SAMPLES, 3))

    return (X, y, z)


class SliceOnlyArray:
    """Array that supports slicing but raises on fancy indexing."""

    def __init__(self, data):
        self._data = np.asarray(data)
        self.shape = self._data.shape

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self._data[idx]
        raise TypeError("Only sequential slicing is supported")


class DataLoaderCommonTests:
    """Test common to all DataLoader variants."""

    # overwrite in child classes
    loader_cls: ClassVar

    @pytest.fixture(autouse=True)
    def init_loader(self):
        """
        Create self.make_loader and save original data.

        Runs before each test.
        """
        raise NotImplementedError("Overwrite init_loader in child classes.")

    def test_basic_creation_and_n_samples(self):
        """Test creating with X, y and that n_samples is correct."""
        loader = self.make_loader()
        assert loader.n_samples == N_SAMPLES

    def test_basic_creation_and_n_samples_variadic(self):
        """Test creating with >2 arrays and that n_samples is correct."""
        loader = self.make_loader(*random_data(), *random_data())
        assert loader.n_samples == N_SAMPLES

    def test_no_arrays_raises(self):
        """Test that providing no arrays raises error."""
        with pytest.raises(ValueError, match="Provide at least one array"):
            self.loader_cls(batch_size=32)

    def test_invalid_batch_size(self):
        """Test that batch_size <= 0 raises error."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            self.loader_cls(*random_data(), batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            self.loader_cls(*random_data(), batch_size=-1)

        with pytest.raises(ValueError, match="batch_size cannot be larger"):
            X, y = ordered_data()
            self.loader_cls(X[:100], y[:100], batch_size=1000)

    def test_mismatched_samples(self):
        """Test that mismatched X and y raises error."""
        X = np.random.randn(100, 5)
        y = np.random.randn(50)

        with pytest.raises(ValueError, match="same number of samples"):
            self.loader_cls(X, y, batch_size=32)

    @pytest.mark.parametrize("shuffle", [True, False])
    @pytest.mark.parametrize("iterate_between_calls", [True, False])
    def test_sample_batch(self, shuffle, iterate_between_calls):
        """Test sample_batch yields correct type, shapes, and is deterministic."""
        # check for a non-standard batch size
        batch_size = 14

        # any value for shuffle should still be deterministic
        loader = self.make_loader(shuffle=shuffle, batch_size=batch_size)

        X_batch1, y_batch1 = loader.sample_batch()

        if iterate_between_calls:
            _ = list(iter(loader))

        X_batch2, y_batch2 = loader.sample_batch()

        assert isinstance(X_batch1, jnp.ndarray)
        assert isinstance(X_batch2, jnp.ndarray)
        assert isinstance(y_batch1, jnp.ndarray)
        assert isinstance(y_batch2, jnp.ndarray)

        assert X_batch1.shape == (batch_size, 5)
        assert y_batch1.shape == (batch_size,)
        assert X_batch2.shape == (batch_size, 5)
        assert y_batch2.shape == (batch_size,)

        np.testing.assert_array_equal(X_batch1, X_batch2)
        np.testing.assert_array_equal(y_batch1, y_batch2)
        np.testing.assert_array_equal(X_batch1, self.X[:batch_size])
        np.testing.assert_array_equal(y_batch1, self.y[:batch_size])

    def test_sample_batch_variadic(self):
        """Test that sample_batch works with >2 arrays."""
        loader = self.loader_cls(*random_data(), batch_size=32)
        batch_data = loader.sample_batch()

        assert isinstance(batch_data, tuple)
        assert len(batch_data) == 3
        assert batch_data[0].shape == (32, 5)
        assert batch_data[1].shape == (32,)
        assert batch_data[2].shape == (32, 3)

    @pytest.mark.parametrize("shuffle", [True, False])
    def test_one_batch_smaller(self, shuffle):
        """Test that one of the batches can be smaller than batch_size."""
        batch_size = 32
        loader = self.make_loader(shuffle=shuffle, batch_size=batch_size)

        batch_sizes = set([X_b.shape[0] for X_b, _ in loader])
        assert batch_sizes == {batch_size, N_SAMPLES % batch_size}

    @pytest.mark.parametrize("shuffle", [True, False])
    def test_iteration_yields_all_data(self, shuffle):
        """Test that iteration covers all samples."""
        loader = self.make_loader(shuffle=shuffle)

        all_X = []
        all_y = []
        for X_batch, y_batch in loader:
            all_X.append(X_batch)
            all_y.append(y_batch)

        X_concat = jnp.concatenate(all_X)
        y_concat = jnp.concatenate(all_y)

        if shuffle:
            sort_idx = jnp.argsort(y_concat)
            # shuffle should actually shuffle
            assert not np.array_equal(sort_idx, jnp.arange(y_concat.shape[0]))
        else:
            sort_idx = jnp.arange(y_concat.shape[0])

        # should give the ordered data saved in self.X and self.y
        np.testing.assert_array_equal(X_concat[sort_idx], self.X)
        np.testing.assert_array_equal(y_concat[sort_idx], self.y)

    def test_iteration_variadic(self):
        """Test that iteration works with >2 data arrays."""
        loader = self.make_loader(*random_data())

        for x, y, z in loader:
            assert x.shape[0] == y.shape[0] == z.shape[0]

    def test_re_iterable(self):
        """Test that DataLoader is re-iterable."""
        loader = self.make_loader(shuffle=False)

        batches_1 = list(loader)
        batches_2 = list(loader)

        assert len(batches_1) == len(batches_2)

        for (X1, y1), (X2, y2) in zip(batches_1, batches_2):
            np.testing.assert_array_equal(X1, X2)
            np.testing.assert_array_equal(y1, y2)

    def test_shuffle_different_epochs(self):
        """Test shuffling produces different order (statistically) across epochs."""
        loader = self.make_loader(batch_size=32, shuffle=True)

        all_y1, all_y2 = [], []
        for _, y_batch in loader:
            all_y1.append(y_batch)
        for _, y_batch in loader:
            all_y2.append(y_batch)

        y_concat1 = jnp.concatenate(all_y1)
        y_concat2 = jnp.concatenate(all_y2)

        # with shuffling, subsequent iterations should have different order
        # (very unlikely to be the same)
        assert not np.array_equal(y_concat1, y_concat2)
        assert np.array_equal(jnp.sort(y_concat1), self.y)
        assert np.array_equal(jnp.sort(y_concat2), self.y)

    @pytest.mark.parametrize("seed", [0, 123])
    def test_same_seed(self, seed):
        """Test that two shuffling data loaders with the same seed produce the same order."""
        loader1 = self.make_loader(shuffle=True, seed=seed)
        loader2 = self.make_loader(shuffle=True, seed=seed)

        for b1, b2 in zip(loader1, loader2):
            np.testing.assert_array_equal(b1[0], b2[0])
            np.testing.assert_array_equal(b1[1], b2[1])

    def test_different_seeds(self):
        """Test shuffling produces different ordering for different seeds."""
        loader1 = self.make_loader(batch_size=32, shuffle=True, seed=0)
        loader2 = self.make_loader(batch_size=32, shuffle=True, seed=123)

        all_y1, all_y2 = [], []
        for _, y_batch in loader1:
            all_y1.append(y_batch)
        for _, y_batch in loader2:
            all_y2.append(y_batch)

        y_concat1 = jnp.concatenate(all_y1)
        y_concat2 = jnp.concatenate(all_y2)

        assert not np.array_equal(y_concat1, y_concat2)
        assert np.array_equal(jnp.sort(y_concat1), self.y)
        assert np.array_equal(jnp.sort(y_concat2), self.y)


class TestArrayDataLoader(DataLoaderCommonTests):
    """Tests for ArrayDataLoader class."""

    loader_cls = ArrayDataLoader

    @pytest.fixture(autouse=True)
    def init_loader(self):
        """Create self.make_loader and save original data."""
        X, y = ordered_data()

        # save originals for comparison
        self.X, self.y = X.copy(), y.copy()

        def _make_loader(*args, **kwargs):
            if "batch_size" not in kwargs:
                kwargs["batch_size"] = 32
            if not args:
                args = (X, y)
            return self.loader_cls(*args, **kwargs)

        self.make_loader = _make_loader

    def test_arrays_are_jax(self):
        """Test that ArrayDataLoader converts arrays on creation."""
        loader = self.make_loader()
        for arr in loader.arrays:
            assert isinstance(arr, jnp.ndarray)


class TestLazyArrayDataLoader(DataLoaderCommonTests):
    """Tests specific for LazyArrayDataLoader class."""

    loader_cls = LazyArrayDataLoader

    @pytest.fixture
    def _numpy_arrays(self):
        X, y = ordered_data()
        return X, y

    @pytest.fixture
    def _memmap_arrays(self, _numpy_arrays, tmp_path):
        X, y = _numpy_arrays
        X_mm = np.memmap(tmp_path / "X.dat", dtype="float64", mode="w+", shape=X.shape)
        X_mm[:] = X
        X_mm.flush()
        y_mm = np.memmap(tmp_path / "y.dat", dtype="float64", mode="w+", shape=y.shape)
        y_mm[:] = y
        y_mm.flush()
        return (
            np.memmap(tmp_path / "X.dat", dtype="float64", mode="r", shape=X.shape),
            np.memmap(tmp_path / "y.dat", dtype="float64", mode="r", shape=y.shape),
        )

    @pytest.fixture
    def _zarr_arrays(self, _numpy_arrays, tmp_path):
        zarr = pytest.importorskip("zarr")

        X, y = _numpy_arrays
        zarr.save(tmp_path / "X.zarr", X)
        zarr.save(tmp_path / "y.zarr", y)
        return (
            zarr.open(tmp_path / "X.zarr", mode="r"),
            zarr.open(tmp_path / "y.zarr", mode="r"),
        )

    @pytest.fixture
    def _hdf5_arrays(self, _numpy_arrays, tmp_path):
        h5py = pytest.importorskip("h5py")

        X, y = _numpy_arrays
        path = tmp_path / "data.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("X", data=X)
            f.create_dataset("y", data=y)
        h5 = h5py.File(path, "r")

        yield h5["X"], h5["y"]

        h5.close()

    @pytest.fixture(params=["numpy", "memmap", "zarr", "hdf5"])
    def lazy_arrays(self, request):
        """Create lazy arrays with a given backend."""
        return request.getfixturevalue(f"_{request.param}_arrays")

    @pytest.fixture(autouse=True)
    def init_loader(self, lazy_arrays, _numpy_arrays):
        """Set up data loader using lazy arrays and save the original numpy arrays."""
        X, y = lazy_arrays
        X_ref, y_ref = _numpy_arrays

        self.X, self.y = X_ref.copy(), y_ref.copy()

        def _make_loader(*args, **kwargs):
            if "batch_size" not in kwargs:
                kwargs["batch_size"] = 32
            if not args:
                args = (X, y)
            return self.loader_cls(*args, **kwargs)

        self.make_loader = _make_loader

    def test_no_eager_conversion(self, lazy_arrays):
        """Test that arrays are not eagerly converted to JAX on construction."""
        loader = LazyArrayDataLoader(*lazy_arrays, batch_size=32)

        for arr in loader.arrays:
            assert not isinstance(arr, jnp.ndarray)

    def test_batches_are_jax_arrays(self, lazy_arrays):
        """Test that yielded batches are JAX arrays."""
        loader = LazyArrayDataLoader(*lazy_arrays, batch_size=32)

        X_batch, y_batch = next(iter(loader))
        assert isinstance(X_batch, jnp.ndarray)
        assert isinstance(y_batch, jnp.ndarray)

    def test_shuffle_changes_chunk_order(self, lazy_arrays):
        """Test that chunk order changes between epochs."""
        loader = LazyArrayDataLoader(*lazy_arrays, batch_size=32, shuffle=True)

        epoch1 = list(iter(loader))
        epoch2 = list(iter(loader))

        def chunk_id(batch):
            """First element of y_batch."""
            return int(batch[1].min())

        epoch1_order = [chunk_id(b) for b in epoch1]
        epoch2_order = [chunk_id(b) for b in epoch2]

        # same chunks, but in different order
        assert set(epoch1_order) == set(epoch2_order)
        assert epoch1_order != epoch2_order

    def test_shuffle_within_batch(self, lazy_arrays):
        """Test that samples within a batch are shuffled."""
        loader = LazyArrayDataLoader(
            *lazy_arrays, batch_size=self.X.shape[0], shuffle=True
        )

        # With batch_size == n_samples, there's one chunk so chunk shuffle
        # is a no-op, but within-batch shuffle should still permute.
        X_batch, y_batch = next(iter(loader))
        assert not np.array_equal(X_batch, self.X)
        assert not np.array_equal(y_batch, self.y)

    def test_slice_only_array(self):
        """Test that with fancy indexing disabled, source arrays are only sliced, never fancy-indexed."""
        X = SliceOnlyArray(np.random.randn(100, 5))
        y = SliceOnlyArray(np.random.randn(100))
        loader = LazyArrayDataLoader(X, y, batch_size=32, shuffle=True)

        # Should not raise — shuffle happens after jnp.asarray conversion
        batches = list(loader)
        assert len(batches) == 4

    def test_fancy_index_raises_with_slice_only_array(self):
        X = SliceOnlyArray(np.random.randn(100, 5))
        y = SliceOnlyArray(np.random.randn(100))
        loader = LazyArrayDataLoader(
            X, y, batch_size=32, shuffle=True, fancy_index=True
        )

        with pytest.raises(TypeError, match="sequential"):
            next(iter(loader))

    def test_fancy_index_requires_shuffle(self):
        """Test that fancy_index=True without shuffle raises error."""
        with pytest.raises(ValueError, match="fancy_index if shuffling"):
            LazyArrayDataLoader(
                *random_data(), batch_size=32, shuffle=False, fancy_index=True
            )

    def test_fancy_index_yields_all_data(self, lazy_arrays):
        """Test that fancy_index iteration covers all samples."""
        loader = LazyArrayDataLoader(
            *lazy_arrays, batch_size=32, shuffle=True, fancy_index=True
        )

        all_X = []
        all_y = []
        for X_batch, y_batch in loader:
            all_X.append(X_batch)
            all_y.append(y_batch)

        X_concat = jnp.concatenate(all_X)
        y_concat = jnp.concatenate(all_y)
        sort_idx = jnp.argsort(y_concat)
        # should give the ordered data saved in self.X and self.y
        np.testing.assert_array_equal(X_concat[sort_idx], self.X)
        np.testing.assert_array_equal(y_concat[sort_idx], self.y)

    def test_fancy_index_shuffles(self, lazy_arrays):
        """Test that fancy_index mode actually shuffles the data."""
        X, y = lazy_arrays
        n_samples = X.shape[0]
        loader = LazyArrayDataLoader(
            X, y, batch_size=n_samples, shuffle=True, fancy_index=True
        )

        X_batch, y_batch = next(iter(loader))
        assert not np.array_equal(X_batch, self.X)
        assert not np.array_equal(y_batch, self.y)

    def test_fancy_index_re_iterable(self, lazy_arrays):
        """Test that fancy_index loader produces different shuffles across epochs."""
        X, y = lazy_arrays
        loader = LazyArrayDataLoader(
            X, y, batch_size=32, shuffle=True, fancy_index=True
        )

        y1 = jnp.concatenate([y_b for _, y_b in loader])
        y2 = jnp.concatenate([y_b for _, y_b in loader])

        # both cover all samples
        np.testing.assert_array_equal(jnp.sort(y1), np.arange(len(y)))
        np.testing.assert_array_equal(jnp.sort(y2), np.arange(len(y)))
        # but in different order
        assert not np.array_equal(y1, y2)

    def test_fancy_index_last_batch_smaller(self, lazy_arrays):
        """Test that the last batch can be smaller with fancy_index."""
        loader = LazyArrayDataLoader(
            *lazy_arrays, batch_size=32, shuffle=True, fancy_index=True
        )

        batch_sizes = set([X_b.shape[0] for X_b, _ in loader])
        assert batch_sizes == {32, N_SAMPLES % 32}


class TestPreprocessedDataLoader:
    """Tests for _PreprocessedDataLoader class."""

    def test_preprocessing_applied(self):
        """Test that preprocessing is applied to batches."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        loader = ArrayDataLoader(X, y, batch_size=32, shuffle=False)

        # Preprocessing function that scales X by 2
        def preprocess(X, y):
            return X * 2, y

        wrapped = _PreprocessedDataLoader(loader, preprocess)

        X_batch, y_batch = next(iter(wrapped))
        X_orig, _ = next(iter(loader))

        np.testing.assert_array_almost_equal(X_batch, X_orig * 2)

    def test_sample_batch_cached(self):
        """Test that sample_batch result is cached."""
        call_count = 0

        def preprocess(X, y):
            nonlocal call_count
            call_count += 1
            return X, y

        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        loader = ArrayDataLoader(X, y, batch_size=32)
        wrapped = _PreprocessedDataLoader(loader, preprocess)

        wrapped.sample_batch()
        wrapped.sample_batch()
        wrapped.sample_batch()

        assert call_count == 1  # Only called once

    def test_n_samples_delegated(self):
        """Test that n_samples is delegated to the wrapped loader."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        loader = ArrayDataLoader(X, y, batch_size=32)
        wrapped = _PreprocessedDataLoader(loader, lambda X, y: (X, y))

        assert wrapped.n_samples == loader.n_samples
        assert wrapped.n_samples == 100


class TestIsDataLoader:
    """Tests for is_data_loader function."""

    def test_array_data_loader(self):
        """Test that ArrayDataLoader is recognized."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        loader = ArrayDataLoader(X, y, batch_size=32)

        assert is_data_loader(loader)

    def test_lazy_array_data_loader(self):
        """Test that LazyArrayDataLoader is recognized."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        loader = LazyArrayDataLoader(X, y, batch_size=32)

        assert is_data_loader(loader)

    def test_dict_not_data_loader(self):
        """Test that dict is not recognized as DataLoader."""
        assert not is_data_loader({"X": np.zeros((10, 5))})

    def test_list_not_data_loader(self):
        """Test that list is not recognized as DataLoader."""
        assert not is_data_loader([np.zeros((10, 5)), np.zeros(10)])

    def test_preprocessed_data_loader_is_data_loader(self):
        """Test that _PreprocessedDataLoader is recognized."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        loader = ArrayDataLoader(X, y, batch_size=32)

        def preprocess(X_batch, y_batch):
            return X_batch, y_batch

        wrapped = _PreprocessedDataLoader(loader, preprocess)
        assert is_data_loader(wrapped)


class TestCompilation:
    """
    Tests that JIT compilation count matches the number of unique batch shapes.

    The ``process`` function is compiled at most twice, meaning that the solver's
    ``update`` is also recompiled once if the last batch is not the same size as the rest.

    Uses a trace counter (Python side effect inside @jax.jit) to count
    compilations. An alternative is the undocumented ``jitted_fn._cache_size()``
    method, which returns the number of cached compiled variants.
    """

    @pytest.fixture(params=[ArrayDataLoader, LazyArrayDataLoader])
    def loader_cls(self, request):
        return request.param

    def test_even_batches_compile_once(self, loader_cls):
        """Even split: one unique shape, one compilation."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        loader = loader_cls(X, y, batch_size=20, shuffle=False)

        trace_count = 0

        @jax.jit
        def process(x, y):
            nonlocal trace_count
            trace_count += 1
            return x.sum() + y.sum()

        for x_batch, y_batch in loader:
            process(x_batch, y_batch)

        assert trace_count == 1

    def test_uneven_batches_compile_twice(self, loader_cls):
        """Uneven split: two unique shapes, two compilations."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        loader = loader_cls(X, y, batch_size=32, shuffle=False)

        trace_count = 0

        @jax.jit
        def process(x, y):
            nonlocal trace_count
            trace_count += 1
            return x.sum() + y.sum()

        for x_batch, y_batch in loader:
            process(x_batch, y_batch)

        assert trace_count == 2
