from contextlib import nullcontext as does_not_raise

import jax
import numpy as np
import pytest

import nemos.utils as utils


class Test1DConvolution:
    @pytest.mark.parametrize(
        "basis_matrix, expectation",
        [
            (np.zeros((1, 1)), does_not_raise()),
            (np.zeros((0, 1)), pytest.raises(
                ValueError, match=r"Empty array provided. At least one of dimension"))
        ]
    )
    def test_empty_basis(self, basis_matrix, expectation):
        vec = np.ones((1, 10))
        with expectation:
            utils.convolve_1d_trials(basis_matrix, vec)

    @pytest.mark.parametrize("window_size", [1, 2])
    @pytest.mark.parametrize("trial_len", [4, 5])
    @pytest.mark.parametrize("array_dim", [2, 3])
    def test_output_trial_length(self, window_size, trial_len, array_dim):
        basis_matrix = np.zeros((window_size, 1))
        time_series = np.zeros((trial_len, 1))
        sample_axis = 0

        if array_dim == 3:
            time_series = np.expand_dims(time_series, axis=0)
            sample_axis += 1

        res = utils.convolve_1d_trials(basis_matrix, time_series)
        if res.shape[sample_axis] != trial_len - window_size + 1:
            raise ValueError("The output of convolution in mode valid should be of "
                             "size num_samples - window_size + 1!")

    @pytest.mark.parametrize(
        "time_series, check_func",
        [
            (np.zeros((20, 1)), lambda x: x.ndim == 3),
            (np.zeros((1, 20, 1)), lambda x: x.ndim == 4),
            ([np.zeros((20, 1)), np.zeros((20, 1))], lambda x: x.ndim == 3),
            ([np.zeros((10, 1)), np.zeros((20, 1))], lambda x: x.ndim == 3)
        ]
    )
    def test_output_shape(self, time_series, check_func):
        res = utils.convolve_1d_trials(np.zeros((1, 1)), time_series)
        if not utils.pytree_map_and_reduce(check_func, all, res):
            raise ValueError("Output doesn't match expected structure")

    @pytest.mark.parametrize(
        "time_series",
        [
            np.zeros((20, 1)),
            np.zeros((1, 20, 1)),
            [np.zeros((20, 1))],
            [np.zeros((10, 1))],
            [np.zeros((10, 1))]
        ]
    )
    def test_output_num_neuron(self, time_series):
        def check_func(x, ts): return x.shape[-2] == ts.shape[-1]
        res = utils.convolve_1d_trials(np.zeros((1, 1)), time_series)
        if not utils.pytree_map_and_reduce(check_func, all, res, time_series):
            raise ValueError("Output  number of neuron doesn't match input.")

    @pytest.mark.parametrize(
        "time_series",
        [
            np.zeros((20, 1)),
            np.zeros((1, 20, 1)),
            [np.zeros((20, 1)), np.zeros((20, 1))],
            [np.zeros((10, 1)), np.zeros((20, 1))],
            [np.zeros((10, 1)), np.zeros((20, 2))]
        ]
    )
    @pytest.mark.parametrize(
        "basis_matrix",
        [
            np.zeros((1, 1)),
            np.zeros((1, 2))
        ]
    )
    def test_output_num_basis(self, time_series, basis_matrix):
        def check_func(conv): return basis_matrix.shape[-1] == conv.shape[-1]
        res = utils.convolve_1d_trials(basis_matrix, time_series)
        if not utils.pytree_map_and_reduce(check_func, all, res):
            raise ValueError("Output  number of neuron doesn't match input.")

    @pytest.mark.parametrize(
        "time_series",
        [
            np.zeros((1, 20, 1)),
            [np.zeros((20, 1)), np.zeros((20, 1))],
            [np.zeros((10, 1)), np.zeros((20, 1))],
            [np.zeros((10, 1)), np.zeros((20, 2))]
        ]
    )
    def test_output_num_trials(self, time_series):
        def check_func(x, ts): return x.shape[0] == ts.shape[0]
        res = utils.convolve_1d_trials(np.zeros((1, 1)), time_series)
        if not utils.pytree_map_and_reduce(check_func, all, res, time_series):
            raise ValueError("Number of trials do not match between input and output")

    @pytest.mark.parametrize("basis_matrix", [np.zeros((3,) * n) for n in [0, 1, 2, 3]])
    @pytest.mark.parametrize("trial_count_shape", [(1, 30, 2), (2, 10, 20)])
    def test_basis_number_of_dim(self, basis_matrix, trial_count_shape: tuple[int]):
        vec = np.ones(trial_count_shape)
        raise_exception = basis_matrix.ndim != 2
        if raise_exception:
            with pytest.raises(
                ValueError, match="basis_matrix must be a 2 dimensional"
            ):
                utils.convolve_1d_trials(basis_matrix, vec)
        else:
            utils.convolve_1d_trials(basis_matrix, vec)

    @pytest.mark.parametrize("basis_matrix", [np.zeros((3, 4))])
    @pytest.mark.parametrize(
        "trial_counts, expectation",
        [
            (np.zeros((1, 30, 2)), does_not_raise()),
            ([np.zeros((30, 2))], does_not_raise()),
            ({"tr1": np.zeros((30, 2)), "tr2": np.zeros((30, 2))}, does_not_raise()),
            (np.zeros((1, 30, 1, 2)), pytest.raises(ValueError, match="time_series must be an pytree of 2 dimensional array-like objects ")),
            ([np.zeros((1, 30, 2))],pytest.raises(ValueError, match="time_series must be an pytree of 2 dimensional array-like objects ")),
            (np.zeros((30, 10)), does_not_raise()),
            ([np.zeros((30, 10))], does_not_raise()),
            (np.zeros(10), pytest.raises(ValueError, match="time_series must be an pytree of 2 dimensional array-like objects ")),
        ],
    )
    def test_spike_count_type(self, basis_matrix, expectation, trial_counts):
        with expectation:
                utils.convolve_1d_trials(basis_matrix, trial_counts)

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
        raise_exception = trial_counts.shape[1] < basis_matrix.shape[0]
        if raise_exception:
            with pytest.raises(
                ValueError,
                match="Insufficient trial duration. The number of time points",
            ):
                utils.convolve_1d_trials(basis_matrix, trial_counts)
        else:
            utils.convolve_1d_trials(basis_matrix, trial_counts)

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
            utils.convolve_1d_trials(basis_matrix, trial_counts)

    @pytest.mark.parametrize(
        "basis_matrix", [np.random.normal(size=(4, 3)) for _ in range(2)]
    )
    @pytest.mark.parametrize(
        "trial_counts", [np.random.normal(size=(2, 10, 3)) for _ in range(2)]
    )
    def test_valid_convolution_output(self, basis_matrix, trial_counts):
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

        utils_out = np.asarray(utils.convolve_1d_trials(basis_matrix, trial_counts))
        assert np.allclose(utils_out, numpy_out, rtol=10**-5, atol=10**-5), (
            "Output of utils.convolve_1d_trials "
            "does not match numpy.convolve in "
            '"valid" mode.'
        )

    @pytest.mark.parametrize(
        "basis_matrix", [np.random.normal(size=(4, 3)) for _ in range(2)]
    )
    @pytest.mark.parametrize(
        "trial_counts", [{key: np.random.normal(size=(10, 3)) for key in range(2)}]
    )
    def test_valid_convolution_output_tree(self, basis_matrix, trial_counts):
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

        utils_out = utils.convolve_1d_trials(basis_matrix, trial_counts)
        check = all(np.allclose(utils_out[k], numpy_out[k], rtol=10 ** -5, atol=10 ** -5) for k in utils_out)
        assert check, (
            "Output of utils.convolve_1d_trials "
            "does not match numpy.convolve in "
            '"valid" mode.'
        )

    @pytest.mark.parametrize(
        "trial_counts",
        [
            np.zeros((1, 30, 2)),
            [np.zeros((30, 2))],
            {"tr1": np.zeros((30, 2)), "tr2": np.zeros((30, 2))},
            np.zeros((30, 10)),
            [np.zeros((30, 10))],
            {"nested": [{"tr1": np.zeros((30, 2)), "tr2": np.zeros((30, 2))}]}
        ],
    )
    def test_tree_structure_match(self, trial_counts):
        basis_matrix = np.zeros((4, 3))
        conv = utils.convolve_1d_trials(basis_matrix, trial_counts)
        assert jax.tree_structure(trial_counts) == jax.tree_structure(conv)


class TestPadding:
    @pytest.mark.parametrize(
        "pytree, expectation",
        [
            (np.zeros([1]), pytest.raises(ValueError, match="conv_trials must be an iterable of 3D arrays")),
            (np.zeros([1, 1]), pytest.raises(ValueError, match="conv_trials must be an iterable of 3D arrays")),
            (np.zeros([1, 1, 1]), does_not_raise()),
            (np.zeros([1]), pytest.raises(ValueError, match="conv_trials must be an iterable of 3D arrays")),
            ([np.zeros([1, 1, 1])], does_not_raise()),
            ({"nested": [np.zeros([1, 1, 1])]}, does_not_raise()),
            ([np.zeros([1, 1, 1, 1])], pytest.raises(ValueError, match="conv_trials must be an iterable of 3D arrays")),
        ],
    )
    @pytest.mark.parametrize("filter_type", ["causal", "acausal", "anti-causal"])
    def test_check_dim(self, pytree, expectation, filter_type):
         with expectation:
            utils.nan_pad_conv(pytree, 3, filter_type=filter_type)

    @pytest.mark.parametrize("filter_type", ["causal", "acausal", "anti-causal", ""])
    @pytest.mark.parametrize("iterable", [[np.zeros([2, 4, 5]), np.zeros([1, 1, 10])]])
    def test_conv_type(self, iterable, filter_type):
        raise_exception = not (filter_type in ["causal", "anti-causal", "acausal"])
        if raise_exception:
            with pytest.raises(ValueError, match="filter_type must be causal, acausal"):
                utils.nan_pad_conv(iterable, 3, filter_type)
        else:
            utils.nan_pad_conv(iterable, 3, filter_type)

    @pytest.mark.parametrize("iterable", [[np.zeros([2, 4, 5]), np.zeros([2, 4, 6])]])
    @pytest.mark.parametrize("window_size", [0.1, -1, 0, 1, 2, 3, 5, 6])
    def test_padding_nan_causal(self, window_size, iterable):
        raise_exception = (not isinstance(window_size, int)) or (window_size <= 0)
        if raise_exception:
            with pytest.raises(
                ValueError, match="window_size must be a positive integer!"
            ):
                utils.nan_pad_conv(iterable, window_size, "anti-causal")
        else:
            padded = utils.nan_pad_conv(iterable, window_size, "causal")
            for trial in padded:
                print(trial.shape, window_size)
            assert all(np.isnan(trial[:window_size]).all() for trial in padded), (
                "Missing NaNs at the " "beginning of the array!"
            )
            assert all(not np.isnan(trial[window_size:]).any() for trial in padded), (
                "Found NaNs at the " "end of the array!"
            )
            assert all(
                padded[k].shape[0] == iterable[k].shape[0] - 1 + window_size
                for k in range(len(padded))
            ), "Size after padding doesn't match expectation. Should be T + window_size - 1."

    @pytest.mark.parametrize("iterable", [[np.zeros([2, 5, 4]), np.zeros([2, 6, 4])]])
    @pytest.mark.parametrize("window_size", [0, 1, 2, 3, 5, 6])
    def test_padding_nan_anti_causal(self, window_size, iterable):
        raise_exception = (not isinstance(window_size, int)) or (window_size <= 0)
        if raise_exception:
            with pytest.raises(
                ValueError, match="window_size must be a positive integer!"
            ):
                utils.nan_pad_conv(iterable, window_size, "anti-causal")
        else:
            padded = utils.nan_pad_conv(iterable, window_size, "anti-causal")
            for trial in padded:
                print(trial.shape, window_size)
            assert all(
                np.isnan(trial[trial.shape[0] - window_size :]).all()
                for trial in padded
            ), ("Missing NaNs at the " "end of the array!")
            assert all(
                not np.isnan(trial[: trial.shape[0] - window_size]).any()
                for trial in padded
            ), ("Found NaNs at the " "beginning of the array!")
            assert all(
                padded[k].shape[0] == iterable[k].shape[0] - 1 + window_size
                for k in range(len(padded))
            ), "Size after padding doesn't match expectation. Should be T + window_size - 1."

    @pytest.mark.parametrize("iterable", [[np.zeros([2, 5, 4]), np.zeros([2, 6, 4])]])
    @pytest.mark.parametrize("window_size", [-1, 0.2, 0, 1, 2, 3, 5, 6])
    def test_padding_nan_acausal(self, window_size, iterable):
        raise_exception = (not isinstance(window_size, int)) or (window_size <= 0)
        if raise_exception:
            with pytest.raises(
                ValueError, match="window_size must be a positive integer!"
            ):
                utils.nan_pad_conv(iterable, window_size, "acausal")

        else:
            init_nan, end_nan = (window_size - 1) // 2, window_size - 1 - (
                window_size - 1
            ) // 2
            padded = utils.nan_pad_conv(iterable, window_size, "acausal")
            for trial in padded:
                print(trial.shape, window_size)
            assert all(np.isnan(trial[:init_nan]).all() for trial in padded), (
                "Missing NaNs at the " "beginning of the array!"
            )
            assert all(
                np.isnan(trial[trial.shape[0] - end_nan :]).all() for trial in padded
            ), ("Missing NaNs at the " "end of the array!")

            assert all(
                not np.isnan(trial[init_nan : trial.shape[0] - end_nan]).any()
                for trial in padded
            ), ("Fund NaNs in " "the middle of the array!")
            assert all(
                padded[k].shape[0] == iterable[k].shape[0] - 1 + window_size
                for k in range(len(padded))
            ), "Size after padding doesn't match expectation. Should be T + window_size - 1."
