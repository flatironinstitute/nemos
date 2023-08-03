import pytest
import numpy as np

import neurostatslib.utils as utils


class Test1DConvolution:
    @pytest.mark.parametrize("basis_matrix", [np.zeros((n, n)) for n in [0, 1, 2]])
    @pytest.mark.parametrize("trial_count_shape", [(1, 30, 2), (2, 10, 20)])
    def test_basis_matrix_type(self, basis_matrix, trial_count_shape: tuple[int]):
        vec = np.ones(trial_count_shape)
        raise_exception = any(k == 0 for k in basis_matrix.shape)
        if raise_exception:
            with pytest.raises(ValueError, match=r"Empty basis_matrix provided\. "
                                                 fr"The shape of basis_matrix is \(0, 0\)!"):
                utils.convolve_1d_trials(basis_matrix, vec)
        else:
            conv = utils.convolve_1d_trials(basis_matrix, vec)
            assert len(conv) == trial_count_shape[0], "Number of trial in the input doesn't " \
                                                      "match that of the convolution output!"
            assert conv[0].shape[1] == trial_count_shape[2], "Number of neurons in the input doesn't " \
                                                             "match that of the convolution output!"
            assert conv[0].shape[2] == basis_matrix.shape[1], "Number of basis function in the input doesn't " \
                                                              "match that of the convolution output!"
            assert (conv[0].shape[0] == vec.shape[1] - basis_matrix.shape[0] + 1), "The output of \"valid\" convolution" \
                                                                                   "should be time_points - window_size + 1." \
                                                                                   f"Expected {vec.shape[1] - basis_matrix.shape[0] + 1}, " \
                                                                                   f"{conv[0].shape[0]} obtained instead!"

    @pytest.mark.parametrize("basis_matrix", [np.zeros((3,)*n) for n in [0, 1, 2, 3]])
    @pytest.mark.parametrize("trial_count_shape", [(1, 30, 2), (2, 10, 20)])
    def test_basis_number_of_dim(self, basis_matrix, trial_count_shape:tuple[int]):
        vec = np.ones(trial_count_shape)
        raise_exception = basis_matrix.ndim != 2
        if raise_exception:
            with pytest.raises(ValueError, match="basis_matrix must be a 2 dimensional"):
                utils.convolve_1d_trials(basis_matrix, vec)
        else:
            utils.convolve_1d_trials(basis_matrix, vec)

    @pytest.mark.parametrize("basis_matrix", [np.zeros((3, 4))])
    @pytest.mark.parametrize("trial_counts", [
                                                np.zeros((1, 30, 2)),   # valid
                                                [np.zeros((30, 2))],  # valid
                                                np.zeros((1, 30, 1, 2)),  # invalid
                                                [np.zeros((1, 30, 2))],  # invalid
                                                np.zeros((1, 10)),  # invalid
                                                [np.zeros(10)]  # invalid
                                              ])
    def test_spike_count_ndim(self, basis_matrix, trial_counts):
        raise_exception = any(trial.ndim != 2 for trial in trial_counts)
        if raise_exception:
            with pytest.raises(ValueError, match="time_series must be an iterable "
                                                 "of 2 dimensional array-like objects."):
                utils.convolve_1d_trials(basis_matrix, trial_counts)
        else:
            utils.convolve_1d_trials(basis_matrix, trial_counts)

    @pytest.mark.parametrize("basis_matrix", [np.zeros((4, 3))])
    @pytest.mark.parametrize("trial_counts", [
                                    np.zeros((1, 4, 2)),  # valid
                                    np.zeros((1, 40, 2)),  # valid
                                    np.zeros((1, 3, 2)),  # invalid
                                                ])
    def test_sufficient_trial_duration(self, basis_matrix, trial_counts):
        raise_exception = trial_counts.shape[1] < basis_matrix.shape[0]
        if raise_exception:
            with pytest.raises(ValueError, match="Insufficient trial duration. The number of time points"):
                utils.convolve_1d_trials(basis_matrix, trial_counts)
        else:
            utils.convolve_1d_trials(basis_matrix, trial_counts)

    @pytest.mark.parametrize("basis_matrix", [np.zeros((4, 3))])
    @pytest.mark.parametrize("trial_counts", [
        np.zeros((0, 4, 3)),  # invalid
        np.zeros((1, 40, 0)),  # invalid
    ])
    def test_empty_counts(self, basis_matrix, trial_counts):
        with pytest.raises(ValueError, match="time_series should not contain"):
            utils.convolve_1d_trials(basis_matrix, trial_counts)

    @pytest.mark.parametrize("basis_matrix", [np.random.normal(size=(4, 3)) for _ in range(2)])
    @pytest.mark.parametrize("trial_counts", [np.random.normal(size=(2, 10, 3)) for _ in range(2)])
    def test_valid_convolution_output(self, basis_matrix, trial_counts):
        numpy_out = np.zeros((trial_counts.shape[0],
                              trial_counts.shape[1] - basis_matrix.shape[0] + 1,
                              trial_counts.shape[2],
                              basis_matrix.shape[1]))
        for tri_i, trial in enumerate(trial_counts):
            for neu_k, vec in enumerate(trial.T):
                for bas_j, basis in enumerate(basis_matrix.T):
                    numpy_out[tri_i, :, neu_k, bas_j] = np.convolve(vec, basis, mode='valid')

        utils_out = np.asarray(utils.convolve_1d_trials(basis_matrix, trial_counts))
        assert np.allclose(utils_out, numpy_out, rtol=10**-5, atol=10**-5), "Output of utils.convolve_1d_trials " \
                                                                            "does not match numpy.convolve in " \
                                                                            "\"valid\" mode."


class TestPadding:

    @pytest.mark.parametrize("filter_type", ["causal", "acausal", "anti-causal"])
    @pytest.mark.parametrize("iterable", [[np.zeros([1]*n)] * 2 for n in range(1, 6)] +
                                         [[np.zeros([1, 2, 4]), np.zeros([1, 2, 4])]] +
                                         [[np.zeros([1, 2, 4]), np.zeros([1, 1, 1, 1])]] +
                                         [[np.zeros([1, 2, 4, 5]), np.zeros([1, 1, 1, 1, 1])]]
    )
    def test_check_dim(self, iterable, filter_type):
        raise_exception = any(trial.ndim != 3 for trial in iterable)
        if raise_exception:
            with pytest.raises(ValueError, match="conv_trials must be an iterable of 3D arrays"):
                utils.nan_pad_conv(iterable, 3, filter_type)
        else:
            utils.nan_pad_conv(iterable, 3, filter_type)

    @pytest.mark.parametrize("filter_type", ["causal", "acausal", "anti-causal", ""])
    @pytest.mark.parametrize("iterable",
                             [[np.zeros([2, 4, 5]), np.zeros([1, 1, 10])]]
                             )
    def test_conv_type(self, iterable, filter_type):
        raise_exception = not (filter_type in ["causal", "anti-causal", "acausal"])
        if raise_exception:
            with pytest.raises(ValueError, match='filter_type must be "causal", "acausal"'):
                utils.nan_pad_conv(iterable, 3, filter_type)
        else:
            utils.nan_pad_conv(iterable, 3, filter_type)

    @pytest.mark.parametrize("iterable",
                             [[np.zeros([2, 4, 5]), np.zeros([2, 4, 6])]]
                             )
    @pytest.mark.parametrize("window_size", [0, 1, 2, 3, 5, 6])
    def test_padding_nan_causal(self, window_size, iterable):
        raise_exception = (not isinstance(window_size, int)) or (window_size <= 0)
        if raise_exception:
            with pytest.raises(ValueError, match="window_size must be a positive integer!"):
                utils.nan_pad_conv(iterable, window_size, "anti-causal")
        else:
            padded = utils.nan_pad_conv(iterable, window_size, "causal")
            for trial in padded:
                print(trial.shape, window_size)
            assert all(np.isnan(trial[:window_size]).all() for trial in padded), "Missing NaNs at the " \
                                                                                       "beginning of the array!"
            assert all(not np.isnan(trial[window_size:]).any() for trial in padded), "Fund NaNs at the " \
                                                                                      "end of the array!"
            assert all(padded[k].shape[0] == iterable[k].shape[0] - 1 + window_size for k in range(len(padded))), \
                "Size after padding doesn't match expectation. Should be T + window_size - 1."

    @pytest.mark.parametrize("iterable",
                             [[np.zeros([2, 5, 4]), np.zeros([2, 6, 4])]]
                             )
    @pytest.mark.parametrize("window_size", [0, 1, 2, 3, 5, 6])
    def test_padding_nan_anti_causal(self, window_size, iterable):
        raise_exception = (not isinstance(window_size, int)) or (window_size <= 0)
        if raise_exception:
            with pytest.raises(ValueError, match="window_size must be a positive integer!"):
                utils.nan_pad_conv(iterable, window_size, "anti-causal")
        else:
            padded = utils.nan_pad_conv(iterable, window_size, "anti-causal")
            for trial in padded:
                print(trial.shape, window_size)
            assert all(np.isnan(trial[trial.shape[0]-window_size:]).all() for trial in padded), "Missing NaNs at the " \
                                                                                       "end of the array!"
            assert all(not np.isnan(trial[:trial.shape[0]-window_size]).any() for trial in padded), "Found NaNs at the " \
                                                                                           "beginning of the array!"
            assert all(padded[k].shape[0] == iterable[k].shape[0] - 1 + window_size for k in range(len(padded))), \
                "Size after padding doesn't match expectation. Should be T + window_size - 1."

    @pytest.mark.parametrize("iterable",
                             [[np.zeros([2, 5, 4]), np.zeros([2, 6, 4])]]
                             )
    @pytest.mark.parametrize("window_size", [-1, 0.2, 0, 1, 2, 3, 5, 6])
    def test_padding_nan_acausal(self, window_size, iterable):
        raise_exception = (not isinstance(window_size, int)) or (window_size <= 0)
        if raise_exception:
            with pytest.raises(ValueError, match="window_size must be a positive integer!"):
                utils.nan_pad_conv(iterable, window_size, "acausal")

        else:
            init_nan, end_nan = (window_size - 1) // 2, window_size - 1 - (window_size - 1) // 2
            padded = utils.nan_pad_conv(iterable, window_size, "acausal")
            for trial in padded:
                print(trial.shape, window_size)
            assert all(
                np.isnan(trial[:init_nan]).all() for trial in padded), "Missing NaNs at the " \
                                                                             "beginning of the array!"
            assert all(
                np.isnan(trial[trial.shape[0] - end_nan:]).all() for trial in padded), "Missing NaNs at the " \
                                                                                             "end of the array!"

            assert all(
                not np.isnan(trial[init_nan: trial.shape[0] - end_nan]).any() for trial in padded), "Fund NaNs in " \
                                                                                                     "the middle of the array!"
            assert all(padded[k].shape[0] == iterable[k].shape[0] - 1 + window_size for k in range(len(padded))), \
                "Size after padding doesn't match expectation. Should be T + window_size - 1."