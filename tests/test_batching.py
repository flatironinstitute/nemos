"""Tests for the dataloaders used for mini-batches in stochastic optimization."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nemos.batching import ArrayDataLoader, _PreprocessedDataLoader, is_data_loader


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

    # TODO: Not sure if this should be the intended behavior.
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
        loader = ArrayDataLoader(
            X, y, batch_size=100, shuffle=True, key=jax.random.key(123)
        )

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
