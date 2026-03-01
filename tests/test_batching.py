"""Tests for the dataloaders used for mini-batches in stochastic optimization."""

import jax.numpy as jnp
import numpy as np
import pytest

from nemos.batching import (
    ArrayDataLoader,
    LazyArrayDataLoader,
    _PreprocessedDataLoader,
    is_data_loader,
)


class TestArrayDataLoader:
    """Tests for ArrayDataLoader class."""

    def test_basic_creation(self):
        """Test basic DataLoader creation."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        loader = ArrayDataLoader(X, y, batch_size=32)

        assert loader.n_samples == 100

    def test_basic_creation_variadic(self):
        """Test basic DataLoader creation with >2 data arrays."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        z = np.random.randn(100, 3)
        loader = ArrayDataLoader(X, y, z, batch_size=32)

        assert loader.n_samples == 100

    def test_sample_batch(self):
        """Test sample_batch returns correct shapes."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        loader = ArrayDataLoader(X, y, batch_size=32)

        X_batch, y_batch = loader.sample_batch()
        assert X_batch.shape == (32, 5)
        assert y_batch.shape == (32,)

    def test_sample_batch_variadic(self):
        """Test sample_batch returns correct shapes."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        z = np.random.randn(100, 3)
        loader = ArrayDataLoader(X, y, z, batch_size=32)

        batch_data = loader.sample_batch()
        assert isinstance(batch_data, tuple)
        assert len(batch_data) == 3
        assert batch_data[0].shape == (32, 5)
        assert batch_data[1].shape == (32,)
        assert batch_data[2].shape == (32, 3)

    def test_sample_batch_deterministic(self):
        """Test sample_batch is deterministic."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        loader = ArrayDataLoader(X, y, batch_size=32, shuffle=True)

        X_batch1, y_batch1 = loader.sample_batch()
        X_batch2, y_batch2 = loader.sample_batch()

        np.testing.assert_array_equal(X_batch1, X_batch2)
        np.testing.assert_array_equal(y_batch1, y_batch2)

    def test_iteration_yields_all_data(self):
        """Test that iteration covers all samples."""
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)
        loader = ArrayDataLoader(X, y, batch_size=32, shuffle=False)

        all_X = []
        all_y = []
        for X_batch, y_batch in loader:
            all_X.append(X_batch)
            all_y.append(y_batch)

        X_concat = jnp.concatenate(all_X)
        y_concat = jnp.concatenate(all_y)

        assert X_concat.shape[0] == 100
        assert y_concat.shape[0] == 100

    def test_iteration_variadic(self):
        """Test that iteration works with >2 data arrays."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        z = np.random.randn(100, 3)
        loader = ArrayDataLoader(X, y, z, batch_size=32)

        for x, y, z in loader:
            assert x.shape[0] == y.shape[0] == z.shape[0]

    def test_re_iterable(self):
        """Test that DataLoader is re-iterable."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        loader = ArrayDataLoader(X, y, batch_size=32, shuffle=False)

        # First iteration
        batches_1 = list(loader)
        # Second iteration
        batches_2 = list(loader)

        assert len(batches_1) == len(batches_2)
        for (X1, y1), (X2, y2) in zip(batches_1, batches_2):
            np.testing.assert_array_equal(X1, X2)
            np.testing.assert_array_equal(y1, y2)

    def test_shuffle(self):
        """Test shuffling produces different order (statistically)."""
        X = np.arange(1000).reshape(1000, 1)
        y = np.arange(1000)
        loader = ArrayDataLoader(X, y, batch_size=100, shuffle=True, seed=123)

        X_batch1, _ = next(iter(loader))
        X_batch2, _ = next(iter(loader))

        # With shuffling, subsequent iterations should have different order
        # (extremely unlikely to be the same)
        assert not np.array_equal(X_batch1, X_batch2)

    def test_invalid_batch_size(self):
        """Test that batch_size <= 0 raises error."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            ArrayDataLoader(X, y, batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            ArrayDataLoader(X, y, batch_size=-1)

    def test_mismatched_samples(self):
        """Test that mismatched X and y raises error."""
        X = np.random.randn(100, 5)
        y = np.random.randn(50)

        with pytest.raises(ValueError, match="same number of samples"):
            ArrayDataLoader(X, y, batch_size=32)


class SliceOnlyArray:
    """Array that supports slicing but raises on fancy indexing."""

    def __init__(self, data):
        self._data = np.asarray(data)
        self.shape = self._data.shape

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self._data[idx]
        raise TypeError("Only sequential slicing is supported")


class TestLazyArrayDataLoader:
    """Tests for LazyArrayDataLoader class."""

    def test_basic_creation(self):
        """Test basic creation."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        loader = LazyArrayDataLoader(X, y, batch_size=32)

        assert loader.n_samples == 100

    def test_basic_creation_variadic(self):
        """Test creation with >2 data arrays."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        z = np.random.randn(100, 3)
        loader = LazyArrayDataLoader(X, y, z, batch_size=32)

        assert loader.n_samples == 100

    def test_no_eager_conversion(self):
        """Test that arrays are not eagerly converted to JAX."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        loader = LazyArrayDataLoader(X, y, batch_size=32)

        assert isinstance(loader.arrays[0], np.ndarray)
        assert isinstance(loader.arrays[1], np.ndarray)

    def test_batches_are_jax_arrays(self):
        """Test that yielded batches are JAX arrays."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        loader = LazyArrayDataLoader(X, y, batch_size=32)

        X_batch, y_batch = next(iter(loader))
        assert isinstance(X_batch, jnp.ndarray)
        assert isinstance(y_batch, jnp.ndarray)

    def test_sample_batch(self):
        """Test sample_batch returns correct shapes."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        loader = LazyArrayDataLoader(X, y, batch_size=32)

        X_batch, y_batch = loader.sample_batch()
        assert X_batch.shape == (32, 5)
        assert y_batch.shape == (32,)
        assert isinstance(X_batch, jnp.ndarray)

    def test_sample_batch_deterministic(self):
        """Test sample_batch is deterministic (ignores shuffle)."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        loader = LazyArrayDataLoader(X, y, batch_size=32, shuffle=True)

        X_batch1, y_batch1 = loader.sample_batch()
        X_batch2, y_batch2 = loader.sample_batch()

        np.testing.assert_array_equal(X_batch1, X_batch2)
        np.testing.assert_array_equal(y_batch1, y_batch2)

    def test_iteration_yields_all_data(self):
        """Test that iteration covers all samples."""
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)
        loader = LazyArrayDataLoader(X, y, batch_size=32, shuffle=False)

        all_y = []
        for _, y_batch in loader:
            all_y.append(y_batch)

        y_concat = jnp.concatenate(all_y)
        assert y_concat.shape[0] == 100
        np.testing.assert_array_equal(jnp.sort(y_concat), np.arange(100))

    def test_iteration_variadic(self):
        """Test that iteration works with >2 data arrays."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        z = np.random.randn(100, 3)
        loader = LazyArrayDataLoader(X, y, z, batch_size=32)

        for x, y, z in loader:
            assert x.shape[0] == y.shape[0] == z.shape[0]

    def test_re_iterable(self):
        """Test that DataLoader is re-iterable."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        loader = LazyArrayDataLoader(X, y, batch_size=32, shuffle=False)

        batches_1 = list(loader)
        batches_2 = list(loader)

        assert len(batches_1) == len(batches_2)
        for (X1, y1), (X2, y2) in zip(batches_1, batches_2):
            np.testing.assert_array_equal(X1, X2)
            np.testing.assert_array_equal(y1, y2)

    def test_shuffle_changes_chunk_order(self):
        """Test that chunk order changes between epochs."""
        X = np.arange(1000).reshape(1000, 1)
        y = np.arange(1000)
        loader = LazyArrayDataLoader(X, y, batch_size=100, shuffle=True, seed=123)

        X_batch1, _ = next(iter(loader))
        X_batch2, _ = next(iter(loader))

        assert not np.array_equal(X_batch1, X_batch2)

    def test_shuffle_within_batch(self):
        """Test that samples within a batch are shuffled."""
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)
        loader = LazyArrayDataLoader(X, y, batch_size=100, shuffle=True, seed=42)

        # With batch_size == n_samples, there's one chunk so chunk shuffle
        # is a no-op, but within-batch shuffle should still permute.
        X_batch, _ = next(iter(loader))
        assert not np.array_equal(X_batch, np.arange(100).reshape(100, 1))

    def test_no_shuffle_preserves_order(self):
        """Test that shuffle=False yields data in original order."""
        X = np.arange(100).reshape(100, 1)
        y = np.arange(100)
        loader = LazyArrayDataLoader(X, y, batch_size=32, shuffle=False)

        all_y = []
        for _, y_batch in loader:
            all_y.append(y_batch)

        y_concat = jnp.concatenate(all_y)
        np.testing.assert_array_equal(y_concat, np.arange(100))

    def test_invalid_batch_size(self):
        """Test that batch_size <= 0 raises error."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            LazyArrayDataLoader(X, y, batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            LazyArrayDataLoader(X, y, batch_size=-1)

    def test_mismatched_samples(self):
        """Test that mismatched array lengths raise error."""
        X = np.random.randn(100, 5)
        y = np.random.randn(50)

        with pytest.raises(ValueError, match="same number of samples"):
            LazyArrayDataLoader(X, y, batch_size=32)

    def test_no_arrays_raises(self):
        """Test that providing no arrays raises error."""
        with pytest.raises(ValueError, match="Provide at least one array"):
            LazyArrayDataLoader(batch_size=32)

    def test_slice_only_array(self):
        """Test that source arrays are only sliced, never fancy-indexed."""
        X = SliceOnlyArray(np.random.randn(100, 5))
        y = SliceOnlyArray(np.random.randn(100))
        loader = LazyArrayDataLoader(X, y, batch_size=32, shuffle=True)

        # Should not raise — shuffle happens after jnp.asarray conversion
        batches = list(loader)
        assert len(batches) == 4

    def test_last_batch_smaller(self):
        """Test that the last batch can be smaller than batch_size."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        loader = LazyArrayDataLoader(X, y, batch_size=32, shuffle=False)

        batch_sizes = [X_b.shape[0] for X_b, _ in loader]
        assert batch_sizes == [32, 32, 32, 4]


zarr = pytest.importorskip("zarr")
h5py = pytest.importorskip("h5py")


class TestLazyArrayDataLoaderIntegration:
    """Integration tests for LazyArrayDataLoader with on-disk backends."""

    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        y = rng.standard_normal(100)
        return X, y

    @pytest.fixture
    def memmap_arrays(self, data, tmp_path):
        X, y = data
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
    def zarr_arrays(self, data, tmp_path):
        X, y = data
        zarr.save(tmp_path / "X.zarr", X)
        zarr.save(tmp_path / "y.zarr", y)
        return (
            zarr.open(tmp_path / "X.zarr", mode="r"),
            zarr.open(tmp_path / "y.zarr", mode="r"),
        )

    @pytest.fixture
    def hdf5_arrays(self, data, tmp_path):
        X, y = data
        path = tmp_path / "data.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("X", data=X)
            f.create_dataset("y", data=y)
        h5 = h5py.File(path, "r")
        yield h5["X"], h5["y"]
        h5.close()

    @pytest.fixture(params=["memmap", "zarr", "hdf5"])
    def lazy_arrays(self, request, memmap_arrays, zarr_arrays, hdf5_arrays):
        return {
            "memmap": memmap_arrays,
            "zarr": zarr_arrays,
            "hdf5": hdf5_arrays,
        }[request.param]

    def test_no_eager_conversion(self, lazy_arrays):
        """Test that arrays are stored as-is, not converted to JAX."""
        X_lazy, y_lazy = lazy_arrays
        loader = LazyArrayDataLoader(X_lazy, y_lazy, batch_size=32)

        assert loader.arrays[0] is X_lazy
        assert loader.arrays[1] is y_lazy
        assert not isinstance(loader.arrays[0], jnp.ndarray)
        assert not isinstance(loader.arrays[1], jnp.ndarray)

    def test_iteration_yields_all_data(self, data, lazy_arrays):
        """Test that all samples are yielded."""
        X_ref, _ = data
        X_lazy, y_lazy = lazy_arrays
        loader = LazyArrayDataLoader(X_lazy, y_lazy, batch_size=32, shuffle=False)

        all_y = []
        for _, y_batch in loader:
            all_y.append(y_batch)

        y_concat = jnp.concatenate(all_y)
        assert y_concat.shape[0] == X_ref.shape[0]

    def test_data_matches_original(self, data, lazy_arrays):
        """Test that loaded data matches the original arrays."""
        X_ref, y_ref = data
        X_lazy, y_lazy = lazy_arrays
        loader = LazyArrayDataLoader(X_lazy, y_lazy, batch_size=32, shuffle=False)

        all_X, all_y = [], []
        for X_batch, y_batch in loader:
            all_X.append(X_batch)
            all_y.append(y_batch)

        np.testing.assert_array_almost_equal(jnp.concatenate(all_X), X_ref)
        np.testing.assert_array_almost_equal(jnp.concatenate(all_y), y_ref)

    def test_batches_are_jax_arrays(self, lazy_arrays):
        """Test that yielded batches are JAX arrays."""
        X_lazy, y_lazy = lazy_arrays
        loader = LazyArrayDataLoader(X_lazy, y_lazy, batch_size=32)

        X_batch, y_batch = next(iter(loader))
        assert isinstance(X_batch, jnp.ndarray)
        assert isinstance(y_batch, jnp.ndarray)

    def test_re_iterable(self, lazy_arrays):
        """Test that the loader can be iterated multiple times."""
        X_lazy, y_lazy = lazy_arrays
        loader = LazyArrayDataLoader(X_lazy, y_lazy, batch_size=32, shuffle=False)

        batches_1 = list(loader)
        batches_2 = list(loader)

        assert len(batches_1) == len(batches_2)
        for (X1, _), (X2, _) in zip(batches_1, batches_2):
            np.testing.assert_array_equal(X1, X2)

    def test_shuffle(self, lazy_arrays):
        """Test that shuffling produces different batch order."""
        X_lazy, y_lazy = lazy_arrays
        loader = LazyArrayDataLoader(
            X_lazy, y_lazy, batch_size=32, shuffle=True, seed=123
        )

        X_batch1, _ = next(iter(loader))
        X_batch2, _ = next(iter(loader))

        assert not np.array_equal(X_batch1, X_batch2)

    def test_sample_batch(self, data, lazy_arrays):
        """Test that sample_batch returns correct data."""
        X_ref, _ = data
        X_lazy, y_lazy = lazy_arrays
        loader = LazyArrayDataLoader(X_lazy, y_lazy, batch_size=32)

        X_batch, y_batch = loader.sample_batch()
        assert X_batch.shape == (32, 5)
        assert isinstance(X_batch, jnp.ndarray)
        np.testing.assert_array_almost_equal(X_batch, X_ref[:32])


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
