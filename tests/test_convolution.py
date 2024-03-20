from contextlib import nullcontext as does_not_raise

import jax
import numpy as np
import pytest

from nemos import utils
from nemos import convolve


class TestShiftTimeAxisAndConvolve:

    @pytest.mark.parametrize(
        "time_series, check_func, axis",
        [
            (np.zeros((1, 20)), lambda x: x.ndim == 3, 1),
            (np.zeros((20, )), lambda x: x.ndim == 2, 0),
            (np.zeros((20, 1)), lambda x: x.ndim == 3, 0),
            (np.zeros((1, 20, 1)), lambda x: x.ndim == 4, 0),
        ],
    )
    def test_output_ndim(self, time_series, check_func,axis):
        """Check that the output dimensionality matches expectation."""
        res = convolve._shift_time_axis_and_convolve(time_series, np.zeros((1, 1)), axis=axis)
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

        res = convolve._shift_time_axis_and_convolve(time_series, np.zeros((1, 1)), axis=axis)
        if not utils.pytree_map_and_reduce(check_func, all, res):
            raise ValueError("Output  number of neuron doesn't match input.")

    @pytest.mark.parametrize(
        "time_series, axis",
        [
            (np.zeros((20, )), 0),
            (np.zeros((20, 1)), 0),
            (np.zeros((1, 20, 1)), 1),
        ],
    )
    @pytest.mark.parametrize("basis_matrix", [np.zeros((1, 1)), np.zeros((1, 2))])
    def test_output_num_basis(self, time_series, basis_matrix, axis):
        """Check that the number of features in input and output matches."""
        def check_func(conv):
            return basis_matrix.shape[-1] == conv.shape[-1]

        res = convolve._shift_time_axis_and_convolve(time_series, basis_matrix, axis=axis)
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

        utils_out = np.asarray(convolve._shift_time_axis_and_convolve(trial_counts, basis_matrix, axis=1))
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

        utils_out = convolve._convolve_1d_trials(basis_matrix, trial_counts, axis=0)
        check = all(
            np.allclose(utils_out[k], numpy_out[k], rtol=10**-5, atol=10**-5)
            for k in utils_out
        )
        assert check, (
            "Output of utils.convolve_1d_trials "
            "does not match numpy.convolve in "
            '"valid" mode.'
        )


# class TestCreateConvolutionalPredictor:
#
#     @pytest.mark.parametrize(
#         "basis, expectation",
#         [
#             (np.ones((3, 1)), does_not_raise()),
#             (
#                 np.ones((2, 1)),
#                 pytest.warns(
#                     UserWarning, match="With `acausal` filter, `basis_matrix.shape"
#                 ),
#             ),
#         ],
#     )
#     def test_warns_even_window(self, basis, expectation):
#         with expectation:
#             utils.create_convolutional_predictor(
#                 basis, np.zeros((1, 10, 1)), predictor_causality="acausal", shift=False
#             )
#
#     @pytest.mark.parametrize("feature", [np.ones((1, 30, 1)), np.ones((1, 20, 1))])
#     @pytest.mark.parametrize(
#         "basis",
#         [
#             np.ones((3, 1)),
#             np.ones((2, 1)),
#             np.ones((3, 2)),
#             np.ones((2, 3)),
#         ],
#     )
#     @pytest.mark.parametrize(
#         "shift",
#         [
#             True,
#             False,
#             None,
#         ],
#     )
#     @pytest.mark.parametrize(
#         "predictor_causality", ["causal", "acausal", "anti-causal"]
#     )
#     def test_preserve_first_axis_shape(
#         self, feature, basis, shift, predictor_causality
#     ):
#         if predictor_causality == "acausal" and shift:
#             return
#         res = utils.create_convolutional_predictor(
#             basis, feature, predictor_causality=predictor_causality, shift=shift
#         )
#         assert res.shape[0] == feature.shape[0]
#
#     @pytest.mark.parametrize("feature", [np.zeros((1, 30, 1))])
#     @pytest.mark.parametrize(
#         "window_size, shift, predictor_causality, nan_idx",
#         [
#             (3, True, "causal", [0, 1, 2]),
#             (2, True, "causal", [0, 1]),
#             (3, False, "causal", [0, 1]),
#             (2, False, "causal", [0]),
#             (2, None, "causal", [0, 1]),
#             (3, True, "anti-causal", [29, 28, 27]),
#             (2, True, "anti-causal", [29, 28]),
#             (3, False, "anti-causal", [29, 28]),
#             (2, False, "anti-causal", [29]),
#             (2, None, "anti-causal", [29, 28]),
#             (3, False, "acausal", [29, 0]),
#             (2, False, "acausal", [29]),
#         ],
#     )
#     def test_expected_nan(
#         self, feature, window_size, shift, predictor_causality, nan_idx
#     ):
#         basis = np.zeros((window_size, 1))
#         res = utils.create_convolutional_predictor(
#             basis, feature, predictor_causality=predictor_causality, shift=shift
#         )
#         other_idx = list(set(np.arange(res.shape[1])).difference(nan_idx))
#         assert np.all(np.isnan(res[:, nan_idx]))
#         assert not np.any(np.isnan(res[:, other_idx]))
#
#     def test_acausal_shift_error(self):
#         basis = np.zeros((3, 1))
#         feature = np.zeros((1, 30, 1))
#         with pytest.raises(
#             ValueError,
#             match="Cannot shift `predictor` when `predictor_causality` is `acausal`",
#         ):
#             utils.create_convolutional_predictor(
#                 basis, feature, predictor_causality="acausal", shift=True
#             )
#
#     def test_basis_len_one_error(self):
#         basis = np.zeros((1, 1))
#         feature = np.zeros((1, 30, 1))
#         with pytest.raises(
#             ValueError, match=r"`basis_matrix.shape\[0\]` should be at least 2"
#         ):
#             utils.create_convolutional_predictor(
#                 basis, feature, predictor_causality="acausal"
#             )
#
#     @pytest.mark.parametrize(
#         "feature", [{"1": [[np.ones((30, 1))]], "2": np.ones((20, 1))}]
#     )
#     @pytest.mark.parametrize(
#         "predictor_causality", ["causal", "acausal", "anti-causal"]
#     )
#     @pytest.mark.parametrize("shift", [True, False, None])
#     def test_conv_tree(self, feature, predictor_causality, shift):
#         if shift and predictor_causality == "acausal":
#             return
#         basis = np.zeros((2, 1))
#         with does_not_raise():
#             utils.create_convolutional_predictor(
#                 basis, feature, predictor_causality=predictor_causality, shift=shift
#             )
#
#     @pytest.mark.parametrize(
#         "feature", [{"1": [[np.ones((30, 1))]], "2": np.ones((20, 1))}]
#     )
#     @pytest.mark.parametrize(
#         "predictor_causality", ["causal", "acausal", "anti-causal"]
#     )
#     @pytest.mark.parametrize("shift", [True, False, None])
#     def test_conv_tree_shape(self, feature, predictor_causality, shift):
#         if shift and predictor_causality == "acausal":
#             return
#         basis = np.zeros((2, 1))
#         res = utils.create_convolutional_predictor(
#             basis, feature, predictor_causality=predictor_causality, shift=shift
#         )
#         arr1, arr2 = jax.tree_util.tree_flatten(res)[0]
#         assert arr1.shape[0] == 30
#         assert arr2.shape[0] == 20


class TestShiftTimeSeries:
    def test_causal_shift(self):
        # Assuming time_series shape is (1, 3, 2, 2)
        time_series = np.random.rand(1, 3, 2, 2).astype(np.float32)
        shifted_series = utils.shift_time_series(time_series, "causal")

        assert np.isnan(
            shifted_series[0, 0]
        ).all(), "First time bin should be NaN for causal shift"
        assert np.array_equal(
            shifted_series[0, 1:], time_series[0, :-1]
        ), "Causal shift did not work as expected"

    def test_anti_causal_shift(self):
        time_series = np.random.rand(1, 3, 2, 2).astype(np.float32)
        shifted_series = utils.shift_time_series(time_series, "anti-causal")

        assert np.isnan(
            shifted_series[0, -1]
        ).all(), "Last time bin should be NaN for anti-causal shift"
        assert np.array_equal(
            shifted_series[0, :-1], time_series[0, 1:]
        ), "Anti-causal shift did not work as expected"

    def test_error_on_non_float_dtype(self):
        time_series = np.random.randint(0, 10, (1, 3, 2, 2))
        with pytest.raises(ValueError, match="time_series must have a float dtype"):
            utils.shift_time_series(time_series)

    def test_error_on_invalid_causality(self):
        time_series = np.random.rand(1, 3, 2, 2).astype(np.float32)
        with pytest.raises(ValueError, match="predictor_causality must be one of"):
            utils.shift_time_series(time_series, "acausal")

    def test_error_on_invalid_dimensionality(self):
        # Dimensionality not matching expected (1, 3, 2, 2) or a valid pytree structure
        time_series = np.random.rand(3, 2, 2).astype(
            np.float32
        )  # Missing n_trials dimension
        with pytest.raises(ValueError):
            utils.shift_time_series(time_series)


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
                    ), 0
            ),
            (np.zeros((30, 10)), does_not_raise(), 0),
            ([np.zeros((30, 10))], does_not_raise(), 1),
            (np.zeros(10), does_not_raise(), 0)
        ],
    )
    def test_spike_count_type(self, basis_matrix, expectation, trial_counts, axis):
        with expectation:
            convolve.create_convolutional_predictor(basis_matrix, trial_counts, axis=axis)

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
                convolve.create_convolutional_predictor(basis_matrix, trial_counts, axis=1)
        else:
            convolve.create_convolutional_predictor(basis_matrix, trial_counts, axis=1)

    @pytest.mark.parametrize(
        "basis_matrix, expectation",
        [
            (np.zeros((1, 1)), does_not_raise()),
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
            convolve._convolve_1d_trials(basis_matrix, vec)

    @pytest.mark.parametrize("window_size", [1, 2])
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

        res = convolve._convolve_1d_trials(basis_matrix, time_series, axis=sample_axis)
        if res.shape[sample_axis] != trial_len - window_size + 1:
            raise ValueError(
                "The output of convolution in mode valid should be of "
                "size num_samples - window_size + 1!"
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
            (np.zeros((1, 20, 1)), lambda x: x.ndim == 4, 0),
            ([np.zeros((20, 1)), np.zeros((20, 1))], lambda x: x.ndim == 3, 0),
            ([np.zeros((10, 1)), np.zeros((20, 1))], lambda x: x.ndim == 3, 0),
        ],
    )
    def test_output_ndim(self, time_series, check_func, axis):
        res = convolve._shift_time_axis_and_convolve(np.zeros((1, 1)), time_series, axis=axis)
        if not utils.pytree_map_and_reduce(check_func, all, res):
            raise ValueError("Output doesn't match expected structure")

    @pytest.mark.parametrize(
        "time_series, axis, output_shape",
        [
            (np.zeros((20, 1)), 0, (20, 1, 1)),
            (np.zeros((1, 20, 1)), 1, (1, 20, 1, 1)),
            ([[np.zeros((1, 1, 20))], np.zeros((1, 1, 20))], 1, (1, 1, 20, 1)),
            ([np.zeros((20, 1))], 0, (20, 1, 1)),
            ([np.zeros((10, 1))], 0, (10, 1, 1)),
            ([[np.zeros((10, 1))]], 0, (10, 1, 1)),
        ],
    )
    def test_output_shape(self, time_series, axis, output_shape):
        def check_func(x):
            return x.shape == output_shape

        res = convolve._shift_time_axis_and_convolve(np.zeros((1, 1)), time_series, axis=axis)
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
    @pytest.mark.parametrize("basis_matrix", [np.zeros((1, 1)), np.zeros((1, 2))])
    def test_output_num_basis(self, time_series, basis_matrix, axis):
        def check_func(conv):
            return basis_matrix.shape[-1] == conv.shape[-1]

        res = convolve._shift_time_axis_and_convolve(basis_matrix, time_series, axis=axis)
        if not utils.pytree_map_and_reduce(check_func, all, res):
            raise ValueError("Output  number of neuron doesn't match input.")

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

        utils_out = np.asarray(convolve._shift_time_axis_and_convolve(basis_matrix, trial_counts, axis=1))
        assert np.allclose(utils_out, numpy_out, rtol=10 ** -5, atol=10 ** -5), (
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

        utils_out = convolve._convolve_1d_trials(basis_matrix, trial_counts, axis=0)
        check = all(
            np.allclose(utils_out[k], numpy_out[k], rtol=10 ** -5, atol=10 ** -5)
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
        conv = convolve.create_convolutional_predictor(basis_matrix, trial_counts, axis=axis)
        assert jax.tree_util.tree_structure(trial_counts) == jax.tree_structure(conv)
