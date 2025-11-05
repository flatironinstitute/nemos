from contextlib import nullcontext as does_not_raise

import jax
import numpy as np
import pynapple as nap
import pytest

from nemos import convolve, utils


def _get_sample_axis_len(time_series, axis=0):
    return jax.tree_util.tree_leaves(time_series)[0].shape[axis]


class TestShiftTimeAxisAndConvolve:

    @pytest.mark.parametrize(
        "time_series, check_func, axis",
        [
            (np.zeros((1, 20)), lambda x: x.ndim == 3, 1),
            (np.zeros((1, 20)), lambda x: x.ndim == 3, -1),
            (np.zeros((20,)), lambda x: x.ndim == 2, 0),
            (np.zeros((20, 1)), lambda x: x.ndim == 3, 0),
            (np.zeros((1, 20, 1)), lambda x: x.ndim == 4, 0),
            (np.zeros((1, 20, 1)), lambda x: x.ndim == 4, -3),
        ],
    )
    def test_output_ndim(self, time_series, check_func, axis):
        """Check that the output dimensionality matches expectation."""
        res = convolve._shift_time_axis_and_convolve(
            time_series,
            np.zeros((1, 1)),
            axis=axis,
            batch_size_channels=1,
            batch_size_basis=1,
            batch_size_samples=_get_sample_axis_len(time_series, axis=axis),
        )
        if not utils.pytree_map_and_reduce(check_func, all, res):
            raise ValueError("Output doesn't match expected structure")

    @pytest.mark.parametrize(
        "time_series, axis, output_shape",
        [
            (np.zeros((20, 1)), 0, (20, 1, 1)),
            (np.zeros((1, 20, 1)), 1, (1, 20, 1, 1)),
        ],
    )
    def test_output_shape(self, time_series, axis, output_shape):
        """Check that the output shape matches expectation."""

        def check_func(x):
            return x.shape == output_shape

        res = convolve._shift_time_axis_and_convolve(
            time_series,
            np.zeros((1, 1)),
            axis=axis,
            batch_size_channels=1,
            batch_size_basis=1,
            batch_size_samples=_get_sample_axis_len(time_series, axis=axis),
        )
        if not utils.pytree_map_and_reduce(check_func, all, res):
            raise ValueError("Output  number of neuron doesn't match input.")

    @pytest.mark.parametrize(
        "time_series, axis",
        [
            (np.zeros((20,)), 0),
            (np.zeros((20, 1)), 0),
            (np.zeros((1, 20, 1)), 1),
        ],
    )
    @pytest.mark.parametrize("basis_matrix", [np.zeros((1, 1)), np.zeros((1, 2))])
    def test_output_num_basis(self, time_series, basis_matrix, axis):
        """Check that the number of features in input and output matches."""

        def check_func(conv):
            return basis_matrix.shape[-1] == conv.shape[-1]

        res = convolve._shift_time_axis_and_convolve(
            time_series,
            basis_matrix,
            axis=axis,
            batch_size_channels=1,
            batch_size_basis=1,
            batch_size_samples=_get_sample_axis_len(time_series, axis=axis),
        )
        if not utils.pytree_map_and_reduce(check_func, all, res):
            raise ValueError("Output  number of neuron doesn't match input.")

    @pytest.mark.parametrize(
        "basis_matrix", [np.random.normal(size=(4, 3)) for _ in range(2)]
    )
    @pytest.mark.parametrize(
        "trial_counts", [np.random.normal(size=(2, 10, 3)) for _ in range(2)]
    )
    def test_valid_convolution_output(self, basis_matrix, trial_counts):
        """Check output matches numpy convolve."""
        numpy_out = np.zeros(
            (
                trial_counts.shape[0],
                trial_counts.shape[1] - basis_matrix.shape[0] + 1,
                trial_counts.shape[2],
                basis_matrix.shape[1],
            )
        )
        for tri_i, trial in enumerate(trial_counts):
            for neu_k, vec in enumerate(trial.T):
                for bas_j, basis in enumerate(basis_matrix.T):
                    numpy_out[tri_i, :, neu_k, bas_j] = np.convolve(
                        vec, basis, mode="valid"
                    )

        utils_out = np.asarray(
            convolve._shift_time_axis_and_convolve(
                trial_counts,
                basis_matrix,
                axis=1,
                batch_size_channels=1,
                batch_size_basis=basis_matrix.shape[1],
                batch_size_samples=_get_sample_axis_len(trial_counts, axis=1),
            )
        )
        assert np.allclose(utils_out, numpy_out, rtol=10**-5, atol=10**-5), (
            "Output of utils.convolve_1d_trials "
            "does not match numpy.convolve in "
            '"valid" mode.'
        )


class TestCreateConvolutionalPredictor:

    @pytest.mark.parametrize("basis_matrix", [np.zeros((3,) * n) for n in [0, 1, 2, 3]])
    @pytest.mark.parametrize("trial_count_shape", [(1, 30, 2), (2, 10, 20)])
    def test_basis_number_of_dim(self, basis_matrix, trial_count_shape: tuple[int]):
        vec = np.ones(trial_count_shape)
        raise_exception = basis_matrix.ndim != 2
        if raise_exception:
            with pytest.raises(
                ValueError, match="basis_matrix must be a 2 dimensional"
            ):
                convolve.create_convolutional_predictor(basis_matrix, vec, axis=1)
        else:
            convolve.create_convolutional_predictor(basis_matrix, vec, axis=1)

    @pytest.mark.parametrize("basis_matrix", [np.zeros((3, 4))])
    @pytest.mark.parametrize(
        "trial_counts, expectation, axis",
        [
            (np.zeros((1, 30, 2)), does_not_raise(), 1),
            ([np.zeros((30, 2))], does_not_raise(), 0),
            ({"tr1": np.zeros((30, 2)), "tr2": np.zeros((30, 2))}, does_not_raise(), 0),
            (np.zeros((1, 30, 1, 2)), does_not_raise(), 1),
            (
                [np.array(10)],
                pytest.raises(
                    ValueError,
                    match="`time_series` should contain arrays of at least one",
                ),
                0,
            ),
            (np.zeros((30, 10)), does_not_raise(), 0),
            ([np.zeros((30, 10))], does_not_raise(), 1),
            (np.zeros(10), does_not_raise(), 0),
        ],
    )
    def test_spike_count_type(self, basis_matrix, expectation, trial_counts, axis):
        with expectation:
            convolve.create_convolutional_predictor(
                basis_matrix, trial_counts, axis=axis
            )

    @pytest.mark.parametrize("basis_matrix", [np.zeros((4, 3))])
    @pytest.mark.parametrize(
        "trial_counts",
        [
            np.zeros((1, 4, 2)),  # valid
            np.zeros((1, 40, 2)),  # valid
            np.zeros((1, 3, 2)),  # invalid
        ],
    )
    def test_sufficient_trial_duration(self, basis_matrix, trial_counts):
        warns = trial_counts.shape[1] < basis_matrix.shape[0]
        if warns:
            with pytest.warns(
                UserWarning,
                match="One or more trials are shorter",
            ):
                res = convolve.create_convolutional_predictor(
                    basis_matrix, trial_counts, axis=1
                )
                assert np.all(np.isnan(res))
        else:
            convolve.create_convolutional_predictor(basis_matrix, trial_counts, axis=1)

    @pytest.mark.parametrize("basis_matrix", [np.zeros((4, 3))])
    @pytest.mark.parametrize(
        "trial_counts, batch_samples, expectation",
        [
            (np.zeros((1, 4, 2)), None, does_not_raise()),
            (np.zeros((1, 4, 2)), 4, does_not_raise()),
            (
                np.zeros((1, 4, 2)),
                3,
                pytest.raises(
                    ValueError,
                    match="Batch size too small",
                ),
            ),
        ],
    )
    def test_sufficient_trial_duration_batching(
        self, basis_matrix, trial_counts, batch_samples, expectation
    ):
        with expectation:
            convolve.create_convolutional_predictor(
                basis_matrix, trial_counts, axis=1, batch_size_samples=batch_samples
            )

    @pytest.mark.parametrize(
        "basis_matrix, expectation",
        [
            (np.zeros((2, 1)), does_not_raise()),
            (
                np.zeros((0, 1)),
                pytest.raises(
                    ValueError, match=r"Empty array provided. At least one of dimension"
                ),
            ),
        ],
    )
    def test_empty_basis(self, basis_matrix, expectation):
        vec = np.ones((1, 10))
        with expectation:
            convolve.create_convolutional_predictor(basis_matrix, vec, axis=1)

    @pytest.mark.parametrize("window_size", [2])
    @pytest.mark.parametrize("trial_len", [4, 5])
    @pytest.mark.parametrize("array_dim", [1, 2, 3])
    def test_output_trial_length(self, window_size, trial_len, array_dim):
        basis_matrix = np.zeros((window_size, 1))
        time_series = np.zeros((trial_len,))
        sample_axis = 0

        if array_dim == 2:
            time_series = np.expand_dims(time_series, axis=0)
            sample_axis = 1
        if array_dim == 3:
            time_series = np.expand_dims(time_series, axis=(0, 2))
            sample_axis = 1

        res = convolve.create_convolutional_predictor(
            basis_matrix, time_series, axis=sample_axis
        )
        if res.shape[sample_axis] != trial_len:
            raise ValueError(
                "The output of create_convolutional_predictor should have size num_samples!"
            )

    @pytest.mark.parametrize("basis_matrix", [np.zeros((4, 3))])
    @pytest.mark.parametrize(
        "trial_counts",
        [
            np.zeros((0, 4, 3)),  # invalid
            np.zeros((1, 40, 0)),  # invalid
        ],
    )
    def test_empty_counts(self, basis_matrix, trial_counts):
        with pytest.raises(ValueError, match="Empty array provided"):
            convolve.create_convolutional_predictor(basis_matrix, trial_counts, axis=1)

    @pytest.mark.parametrize(
        "time_series, check_func, axis",
        [
            (np.zeros((1, 20)), lambda x: x.ndim == 3, 1),
            (np.zeros((20,)), lambda x: x.ndim == 2, 0),
            ([np.zeros((20,)), np.zeros((10,))], lambda x: x.ndim == 2, 0),
            (np.zeros((20, 1)), lambda x: x.ndim == 3, 0),
            (np.zeros((1, 20, 1)), lambda x: x.ndim == 4, 1),
            ([np.zeros((20, 1)), np.zeros((20, 1))], lambda x: x.ndim == 3, 0),
            ([np.zeros((10, 1)), np.zeros((20, 1))], lambda x: x.ndim == 3, 0),
        ],
    )
    def test_output_ndim(self, time_series, check_func, axis):
        res = convolve.create_convolutional_predictor(
            np.zeros((2, 1)), time_series, axis=axis
        )
        if not utils.pytree_map_and_reduce(check_func, all, res):
            raise ValueError("Output doesn't match expected structure")

    @pytest.mark.parametrize(
        "time_series, axis, output_shape",
        [
            (np.zeros((20, 1)), 0, (20, 1, 1)),
            (np.zeros((1, 20, 1)), 1, (1, 20, 1, 1)),
            ([[np.zeros((1, 1, 20))], np.zeros((1, 1, 20))], 2, (1, 1, 20, 1)),
            ([np.zeros((20, 1))], 0, (20, 1, 1)),
            ([np.zeros((10, 1))], 0, (10, 1, 1)),
            ([[np.zeros((10, 1))]], 0, (10, 1, 1)),
        ],
    )
    def test_output_shape(self, time_series, axis, output_shape):
        def check_func(x):
            return x.shape == output_shape

        res = convolve.create_convolutional_predictor(
            np.zeros((2, 1)), time_series, axis=axis
        )
        if not utils.pytree_map_and_reduce(check_func, all, res):
            raise ValueError("Output  number of neuron doesn't match input.")

    @pytest.mark.parametrize(
        "time_series, axis",
        [
            (np.zeros((20,)), 0),
            (np.zeros((20, 1)), 0),
            (np.zeros((1, 20, 1)), 1),
            ([np.zeros((20, 1)), np.zeros((20, 1))], 0),
            ([np.zeros((10, 1)), np.zeros((20, 1))], 0),
            ([np.zeros((10, 1)), np.zeros((20, 2))], 0),
        ],
    )
    @pytest.mark.parametrize("basis_matrix", [np.zeros((2, 1)), np.zeros((2, 2))])
    def test_output_num_basis(self, time_series, basis_matrix, axis):
        def check_func(conv):
            return basis_matrix.shape[-1] == conv.shape[-1]

        res = convolve.create_convolutional_predictor(
            basis_matrix, time_series, axis=axis
        )
        if not utils.pytree_map_and_reduce(check_func, all, res):
            raise ValueError("Output  number of neuron doesn't match input.")

    @pytest.mark.parametrize(
        "basis_matrix", [np.random.normal(size=(4, 3)) for _ in range(2)]
    )
    @pytest.mark.parametrize(
        "trial_counts", [{key: np.random.normal(size=(10, 3)) for key in range(2)}]
    )
    @pytest.mark.parametrize("batch_samples", [None, 5])
    @pytest.mark.parametrize("batch_channels", [None, 1, 2])
    @pytest.mark.parametrize("batch_basis", [None, 1, 2])
    def test_valid_convolution_output_tree(
        self, basis_matrix, trial_counts, batch_samples, batch_channels, batch_basis
    ):
        numpy_out = np.zeros(
            (
                len(trial_counts),
                trial_counts[0].shape[0] - basis_matrix.shape[0] + 1,
                trial_counts[0].shape[1],
                basis_matrix.shape[1],
            )
        )
        for tri_i, trial in trial_counts.items():
            for neu_k, vec in enumerate(trial.T):
                for bas_j, basis in enumerate(basis_matrix.T):
                    numpy_out[tri_i, :, neu_k, bas_j] = np.convolve(
                        vec, basis, mode="valid"
                    )
        ws = basis_matrix.shape[0]
        utils_out = convolve.create_convolutional_predictor(
            basis_matrix,
            trial_counts,
            axis=0,
            shift=False,
            batch_size_samples=batch_samples,
            batch_size_basis=batch_basis,
            batch_size_channels=batch_channels,
        )
        check = all(
            np.allclose(utils_out[k][ws - 1 :], numpy_out[k], rtol=10**-5, atol=10**-5)
            for k in utils_out
        )
        assert check, (
            "Output of utils.convolve_1d_trials "
            "does not match numpy.convolve in "
            '"valid" mode.'
        )

    @pytest.mark.parametrize(
        "trial_counts, axis",
        [
            (np.zeros((1, 30, 2)), 1),
            ([np.zeros((30, 2))], 0),
            ({"tr1": np.zeros((30, 2)), "tr2": np.zeros((30, 2))}, 0),
            (np.zeros((30, 10)), 0),
            ([np.zeros((30, 10))], 0),
            ({"nested": [{"tr1": np.zeros((30, 2)), "tr2": np.zeros((30, 2))}]}, 0),
        ],
    )
    def test_tree_structure_match(self, trial_counts, axis):
        basis_matrix = np.zeros((4, 3))
        conv = convolve.create_convolutional_predictor(
            basis_matrix, trial_counts, axis=axis
        )
        assert jax.tree_util.tree_structure(
            trial_counts
        ) == jax.tree_util.tree_structure(conv)

    @pytest.mark.parametrize("axis", [0, 1, 2])
    @pytest.mark.parametrize(
        "window_size, shift, predictor_causality, nan_idx",
        [
            (3, True, "causal", [0, 1, 2]),
            (2, True, "causal", [0, 1]),
            (3, False, "causal", [0, 1]),
            (2, False, "causal", [0]),
            (2, None, "causal", [0, 1]),
            (3, True, "anti-causal", [29, 28, 27]),
            (2, True, "anti-causal", [29, 28]),
            (3, False, "anti-causal", [29, 28]),
            (2, False, "anti-causal", [29]),
            (2, None, "anti-causal", [29, 28]),
            (3, False, "acausal", [29, 0]),
        ],
    )
    @pytest.mark.parametrize("batch_samples", [None, 5])
    @pytest.mark.parametrize("batch_channels", [None, 1, 2])
    @pytest.mark.parametrize("batch_basis", [None, 1, 2])
    def test_expected_nan(
        self,
        axis,
        window_size,
        shift,
        predictor_causality,
        nan_idx,
        batch_samples,
        batch_channels,
        batch_basis,
    ):
        shape = [1, 1, 1]
        shape[axis] = 30
        feature = np.zeros(shape)
        basis = np.zeros((window_size, 1))
        res = convolve.create_convolutional_predictor(
            basis,
            feature,
            predictor_causality=predictor_causality,
            shift=shift,
            axis=axis,
            batch_size_samples=batch_samples,
            batch_size_basis=batch_basis,
            batch_size_channels=batch_channels,
        )
        # get expected non-nan idxs
        other_idx = list(set(np.arange(res.shape[1])).difference(nan_idx))

        # take values and check
        nans = np.take(res, nan_idx, axis)
        non_nans = np.take(res, other_idx, axis)
        assert np.all(np.isnan(nans))
        assert not np.any(np.isnan(non_nans))

    @pytest.mark.parametrize(
        "tsd",
        [
            nap.Tsd(
                t=np.arange(100),
                d=np.arange(100),
                time_support=nap.IntervalSet(start=[0, 50], end=[20, 75]),
            )
        ],
    )
    @pytest.mark.parametrize(
        "window_size, shift, predictor_causality, nan_index",
        [
            (3, True, "causal", [0, 1, 2, 50, 51, 52]),
            (2, True, "causal", [0, 1, 50, 51]),
            (3, False, "causal", [0, 1, 50, 51]),
            (2, False, "causal", [0, 50]),
            (2, None, "causal", [0, 1, 50, 51]),
            (3, True, "anti-causal", [20, 19, 18, 75, 74, 73]),
            (2, True, "anti-causal", [20, 19, 75, 74]),
            (3, False, "anti-causal", [20, 19, 75, 74]),
            (2, False, "anti-causal", [20, 75]),
            (2, None, "anti-causal", [20, 19, 75, 74]),
            (3, False, "acausal", [0, 20, 50, 75]),
        ],
    )
    @pytest.mark.parametrize("batch_samples", [None, 5])
    @pytest.mark.parametrize("batch_channels", [None, 1, 2])
    @pytest.mark.parametrize("batch_basis", [None, 1, 2])
    def test_multi_epoch_pynapple(
        self,
        tsd,
        window_size,
        shift,
        predictor_causality,
        nan_index,
        batch_samples,
        batch_channels,
        batch_basis,
    ):
        """Test nan location in multi-epoch pynapple tsd."""
        basis = np.zeros((window_size, 1))
        res = convolve.create_convolutional_predictor(
            basis,
            tsd,
            predictor_causality=predictor_causality,
            shift=shift,
            batch_size_samples=batch_samples,
            batch_size_basis=batch_basis,
            batch_size_channels=batch_channels,
        )

        nan_index = np.sort(nan_index)
        times_nan_found = res[np.isnan(res.d[:, 0])].t
        assert len(times_nan_found) == len(nan_index)
        assert all(times_nan_found == np.array(nan_index))

    @pytest.mark.parametrize(
        "batch_samples, expectation",
        [
            (None, does_not_raise()),
            (4, does_not_raise()),
            (
                -1,
                pytest.raises(
                    ValueError,
                    match="When provided ``batch_size_samples`` must be a strictly positive",
                ),
            ),
            (
                "a",
                pytest.raises(
                    ValueError,
                    match="When provided ``batch_size_samples`` must be a strictly positive",
                ),
            ),
            (
                (),
                pytest.raises(
                    ValueError,
                    match="When provided ``batch_size_samples`` must be a strictly positive",
                ),
            ),
            (
                (1,),
                pytest.raises(
                    ValueError,
                    match="When provided ``batch_size_samples`` must be a strictly positive",
                ),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "ts, kernels", [(np.random.randn(10, 5), np.random.randn(4, 5))]
    )
    def test_invalid_batch_size_samples(self, ts, kernels, batch_samples, expectation):
        with expectation:
            convolve.create_convolutional_predictor(
                kernels,
                ts,
                axis=0,
                batch_size_samples=batch_samples,
            )

    @pytest.mark.parametrize(
        "batch_channels, expectation",
        [
            (None, does_not_raise()),
            (4, does_not_raise()),
            (
                -1,
                pytest.raises(
                    ValueError,
                    match="When provided ``batch_size_channels`` must be a strictly positive",
                ),
            ),
            (
                "a",
                pytest.raises(
                    ValueError,
                    match="When provided ``batch_size_channels`` must be a strictly positive",
                ),
            ),
            (
                (),
                pytest.raises(
                    ValueError,
                    match="When provided ``batch_size_channels`` must be a strictly positive",
                ),
            ),
            (
                (1,),
                pytest.raises(
                    ValueError,
                    match="When provided ``batch_size_channels`` must be a strictly positive",
                ),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "ts, kernels", [(np.random.randn(10, 5), np.random.randn(4, 5))]
    )
    def test_invalid_batch_size_channels(
        self, ts, kernels, batch_channels, expectation
    ):
        with expectation:
            convolve.create_convolutional_predictor(
                kernels,
                ts,
                axis=0,
                batch_size_channels=batch_channels,
            )

    @pytest.mark.parametrize(
        "batch_basis, expectation",
        [
            (None, does_not_raise()),
            (4, does_not_raise()),
            (
                -1,
                pytest.raises(
                    ValueError,
                    match="When provided ``batch_size_basis`` must be a strictly positive",
                ),
            ),
            (
                "a",
                pytest.raises(
                    ValueError,
                    match="When provided ``batch_size_basis`` must be a strictly positive",
                ),
            ),
            (
                (),
                pytest.raises(
                    ValueError,
                    match="When provided ``batch_size_basis`` must be a strictly positive",
                ),
            ),
            (
                (1,),
                pytest.raises(
                    ValueError,
                    match="When provided ``batch_size_basis`` must be a strictly positive",
                ),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "ts, kernels", [(np.random.randn(10, 5), np.random.randn(4, 5))]
    )
    def test_invalid_batch_size_basis(self, ts, kernels, batch_basis, expectation):
        with expectation:
            convolve.create_convolutional_predictor(
                kernels,
                ts,
                axis=0,
                batch_size_basis=batch_basis,
            )

    @pytest.mark.parametrize("input_shape", [(3, 1), (3, 2), (3, 3)])
    @pytest.mark.parametrize("batch_size", [1, 2, 3])
    def test_batch_binary_func_output(self, input_shape, batch_size):
        """Check expected output values."""
        jax.config.update("jax_enable_x64", True)

        def add(a, b):
            return a + b

        x, y = np.random.randn(*input_shape), np.arange(3)[:, np.newaxis]
        result = convolve._batch_binary_func(
            x, y, binary_func=add, batch_size=batch_size, axis=1
        )
        assert np.all(result == add(x, y))

    @pytest.mark.parametrize("input_shape", [(3, 2)])
    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("out_axis, expected_out_shape", [(0, (6, 1)), (1, (3, 2))])
    def test_batch_binary_func_out_axis(
        self, input_shape, batch_size, out_axis, expected_out_shape
    ):
        """Check expected out shape."""
        jax.config.update("jax_enable_x64", True)

        def add(a, b):
            return a + b

        x, y = np.random.randn(*input_shape), np.arange(3)[:, np.newaxis]
        result = convolve._batch_binary_func(
            x, y, binary_func=add, batch_size=batch_size, axis=1, out_axis=out_axis
        )
        assert expected_out_shape == result.shape

    @pytest.mark.parametrize("input_shape", [(3, 3)])
    @pytest.mark.parametrize("batch_size", [2])
    def test_batch_binary_func_pad(self, input_shape, batch_size):
        jax.config.update("jax_enable_x64", True)

        def add(a, b):
            return a + b

        x, y = np.random.randn(*input_shape), np.arange(3)[:, np.newaxis]
        # test that nan-padding and crop and no padding makes no difference
        result = convolve._batch_binary_func(
            x, y, binary_func=add, batch_size=batch_size, axis=1, pad_final_batch=True
        )
        result_nopad = convolve._batch_binary_func(
            x, y, binary_func=add, batch_size=batch_size, axis=1, pad_final_batch=False
        )
        assert np.all(result_nopad == result)


def numpy_tensor_convolve(array, eval_basis):
    """Naive implementation of _tensor_convolve."""
    n_samples, n_channels = array.shape
    window_size, n_basis = eval_basis.shape
    output = np.empty((n_samples - window_size + 1, n_channels, n_basis))
    for i in range(n_channels):
        for j in range(n_basis):
            output[:, i, j] = np.convolve(array[:, i], eval_basis[:, j], mode="valid")
    return output


@pytest.mark.parametrize("input_shape", [(30, 4), (50, 6)])
@pytest.mark.parametrize("basis_shape", [(5, 3), (7, 2)])
@pytest.mark.parametrize("batch_sizes", [(16, 2, 1), (25, 3, 1)])
def test_tensor_convolve(input_shape, basis_shape, batch_sizes):
    """Test different parameter combinations against naive implementation."""
    key = jax.random.PRNGKey(0)
    array = jax.random.normal(key, shape=input_shape)
    eval_basis = jax.random.normal(key, shape=basis_shape)

    batch_size_samples, batch_size_channels, batch_size_basis = batch_sizes

    result = convolve._tensor_convolve(
        array, eval_basis, batch_size_samples, batch_size_channels, batch_size_basis
    )
    expected = numpy_tensor_convolve(np.array(array), np.array(eval_basis))

    np.testing.assert_allclose(np.array(result), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "tsd, nan_placement",
    [
        # epochs: invalid, valid
        (
            nap.Tsd(
                t=np.arange(10),
                d=np.ones(10),
                time_support=nap.IntervalSet([0, 2], [1.5, 10]),
            ),
            [0, 1, 2, 3, 4],
        ),
        # epochs: valid, invalid
        (
            nap.Tsd(
                t=np.arange(10),
                d=np.ones(10),
                time_support=nap.IntervalSet([0, 9], [7, 11]),
            ),
            [0, 1, 2, 8],
        ),
        # epochs: invalid, invalid
        (
            nap.Tsd(
                t=np.arange(10),
                d=np.ones(10),
                time_support=nap.IntervalSet([0, 7.5], [1.5, 11]),
            ),
            [0, 1, 2, 3],
        ),
    ],
)
@pytest.mark.parametrize("kern", [np.random.randn(3, 1)])
def test_convolve_nap_short_iset(tsd, nan_placement, kern):
    with pytest.warns(UserWarning, match="One or more trials are shorter"):
        out = convolve.create_convolutional_predictor(kern, tsd)
        assert np.all(np.isnan(out.d[nan_placement]))
        non_nan_idx = np.setxor1d(np.arange(out.shape[0]), nan_placement)
        assert not np.any(np.isnan(out.d[non_nan_idx]))


@pytest.mark.parametrize(
    "slice_up_to, expectation",
    [
        (5, does_not_raise()),
        (2, pytest.warns(UserWarning, match="One or more trials are shorter")),
    ],
)
@pytest.mark.parametrize("kern", [np.random.randn(3, 1)])
def test_convolve_sliced_tsd(slice_up_to, expectation, kern):
    tsd = nap.Tsd(
        t=np.arange(10), d=np.ones(10), time_support=nap.IntervalSet([0, 6], [5, 10])
    )
    sliced_tsd = tsd[:slice_up_to]
    with expectation:
        convolve.create_convolutional_predictor(kern, [sliced_tsd, tsd])


def test_convolve_short_arrays():
    kern = np.random.randn(3, 1)
    arrs = [np.ones(2), np.ones(1)]
    with pytest.warns(UserWarning, match="One or more trials are shorter"):
        out = convolve.create_convolutional_predictor(kern, arrs)
        assert all(np.all(np.isnan(o)) for o in out)
