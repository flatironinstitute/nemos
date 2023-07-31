import pytest
import numpy as np

import neurostatslib.utils as utils


class Test1DConvolution:
    @pytest.mark.parametrize("basis_matrix", [np.zeros((n, n)) for n in [0, 1, 2]])
    @pytest.mark.parametrize("trial_count_shape", [(1, 2, 30), (2, 20, 10)])
    def test_basis_matrix_type(self, basis_matrix, trial_count_shape:tuple[int]):
        vec = np.ones(trial_count_shape)
        raise_exception = basis_matrix.shape[0] < 1
        if raise_exception:
            with pytest.raises(IndexError, match="index is out of bounds for axis"):
                utils.convolve_1d_trials(basis_matrix, vec)
        else:
            conv = utils.convolve_1d_trials(basis_matrix, vec)
            assert len(conv) == trial_count_shape[0], "Number of trial in the input doesn't " \
                                                      "match that of the convolution output!"
            assert conv[0].shape[0] == trial_count_shape[1], "Number of neurons in the input doesn't " \
                                                             "match that of the convolution output!"
            assert conv[0].shape[1] == basis_matrix.shape[0], "Number of basis function in the input doesn't " \
                                                              "match that of the convolution output!"
            assert (conv[0].shape[2] == vec.shape[2] - basis_matrix.shape[0] + 1), "The output of \"valid\" convolution" \
                                                                                   "should be time_points - window_size + 1." \
                                                                                   f"Expected {vec.shape[2] - basis_matrix.shape[0] + 1}, " \
                                                                                   f"{conv[0].shape[2]} obtained instead!"

    @pytest.mark.parametrize("basis_matrix", [np.zeros((3,)*n) for n in [0, 1, 2, 3]])
    @pytest.mark.parametrize("trial_count_shape", [(1, 2, 30), (2, 20, 10)])
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
                                                np.zeros((1, 2, 30)),   # valid
                                                [np.zeros((2, 30))],  # valid
                                                np.zeros((1, 1, 2, 30)),  # invalid
                                                [np.zeros((1, 2, 30))],  # invalid
                                                np.zeros((1, 10)),  # invalid
                                                [np.zeros(10)]  # invalid
                                              ])
    def test_spike_count_ndim(self, basis_matrix, trial_counts):
        raise_exception = any(trial.ndim != 2 for trial in trial_counts)
        if raise_exception:
            with pytest.raises(ValueError, match="trials_time_series must be an iterable "
                                                 "of 2 dimensional array-like objects."):
                utils.convolve_1d_trials(basis_matrix, trial_counts)
        else:
            utils.convolve_1d_trials(basis_matrix, trial_counts)

    @pytest.mark.parametrize("basis_matrix", [np.zeros((3, 4))])
    @pytest.mark.parametrize("trial_counts", [
                                    np.zeros((1, 2, 4)),  # valid
                                    np.zeros((1, 2, 40)),  # valid
                                    np.zeros((1, 2, 3)),  # invalid
                                                ])
    def test_sufficient_trial_duration(self, basis_matrix, trial_counts):
        raise_exception = trial_counts.shape[2] < basis_matrix.shape[1]
        if raise_exception:
            with pytest.raises(ValueError, match="Insufficient trial duration. The number of time points"):
                utils.convolve_1d_trials(basis_matrix, trial_counts)
        else:
            utils.convolve_1d_trials(basis_matrix, trial_counts)

    @pytest.mark.parametrize("basis_matrix", [np.zeros((3, 4))])
    @pytest.mark.parametrize("trial_counts", [
        np.zeros((0, 2, 4)),  # invalid
        np.zeros((1, 0, 40)),  # invalid
    ])
    def test_empty_counts(self, basis_matrix, trial_counts):
        with pytest.raises(ValueError, match="trials_time_series should not contain"):
            utils.convolve_1d_trials(basis_matrix, trial_counts)

    @pytest.mark.parametrize("basis_matrix", [np.random.normal(size=(3, 4)) for i in range(2)])
    @pytest.mark.parametrize("trial_counts", [np.random.normal(size=(2, 3, 10)) for i in range(2)])
    def test_valid_convolution_output(self, basis_matrix, trial_counts):
        numpy_out = np.zeros((trial_counts.shape[0],
                              trial_counts.shape[1],
                              basis_matrix.shape[0], trial_counts.shape[2] - basis_matrix.shape[1] + 1))
        for tri_i, trial in enumerate(trial_counts):
            for neu_k, vec in enumerate(trial):
                for bas_j, basis in enumerate(basis_matrix):
                    numpy_out[tri_i, neu_k, bas_j] = np.convolve(vec, basis, mode='valid')

        utils_out = np.asarray(utils.convolve_1d_trials(basis_matrix, trial_counts))
        assert np.allclose(utils_out, numpy_out, rtol=10**-5, atol=10**-5), "Output of utils.convolve_1d_trials " \
                                                                            "does not match numpy.convolve in " \
                                                                            "\"valid\" mode."



