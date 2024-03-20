from contextlib import nullcontext as does_not_raise

import jax
import numpy as np
import pytest

from nemos import utils


class Test1DConvolution:
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
            raise ValueError(
                "The output of convolution in mode valid should be of "
                "size num_samples - window_size + 1!"
            )

    @pytest.mark.parametrize(
        "time_series, check_func",
        [
            (np.zeros((20, 1)), lambda x: x.ndim == 3),
            (np.zeros((1, 20, 1)), lambda x: x.ndim == 4),
            ([np.zeros((20, 1)), np.zeros((20, 1))], lambda x: x.ndim == 3),
            ([np.zeros((10, 1)), np.zeros((20, 1))], lambda x: x.ndim == 3),
        ],
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
            [np.zeros((10, 1))],
        ],
    )
    def test_output_num_neuron(self, time_series):
        def check_func(x, ts):
            return x.shape[-2] == ts.shape[-1]

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
            [np.zeros((10, 1)), np.zeros((20, 2))],
        ],
    )
    @pytest.mark.parametrize("basis_matrix", [np.zeros((1, 1)), np.zeros((1, 2))])
    def test_output_num_basis(self, time_series, basis_matrix):
        def check_func(conv):
            return basis_matrix.shape[-1] == conv.shape[-1]

        res = utils.convolve_1d_trials(basis_matrix, time_series)
        if not utils.pytree_map_and_reduce(check_func, all, res):
            raise ValueError("Output  number of neuron doesn't match input.")

    @pytest.mark.parametrize(
        "time_series",
        [
            np.zeros((1, 20, 1)),
            [np.zeros((20, 1)), np.zeros((20, 1))],
            [np.zeros((10, 1)), np.zeros((20, 1))],
            [np.zeros((10, 1)), np.zeros((20, 2))],
        ],
    )
    def test_output_num_trials(self, time_series):
        def check_func(x, ts):
            return x.shape[0] == ts.shape[0]

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
            (
                np.zeros((1, 30, 1, 2)),
                pytest.raises(
                    ValueError,
                    match="time_series must be a pytree of 2 dimensional array-like objects ",
                ),
            ),
            (
                [np.zeros((1, 30, 2))],
                pytest.raises(
                    ValueError,
                    match="time_series must be a pytree of 2 dimensional array-like objects ",
                ),
            ),
            (np.zeros((30, 10)), does_not_raise()),
            ([np.zeros((30, 10))], does_not_raise()),
            (
                np.zeros(10),
                pytest.raises(
                    ValueError,
                    match="time_series must be a pytree of 2 dimensional array-like objects ",
                ),
            ),
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
        check = all(
            np.allclose(utils_out[k], numpy_out[k], rtol=10**-5, atol=10**-5)
            for k in utils_out
        )
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
            {"nested": [{"tr1": np.zeros((30, 2)), "tr2": np.zeros((30, 2))}]},
        ],
    )
    def test_tree_structure_match(self, trial_counts):
        basis_matrix = np.zeros((4, 3))
        conv = utils.convolve_1d_trials(basis_matrix, trial_counts)
        assert jax.tree_util.tree_structure(trial_counts) == jax.tree_structure(conv)


class TestPadding:
    @pytest.mark.parametrize(
        "pytree, expectation",
        [
            (
                np.zeros([1]),
                pytest.raises(
                    ValueError, match="conv_time_series must be a pytree of 3D arrays"
                ),
            ),
            (
                np.zeros([1, 1]),
                pytest.raises(
                    ValueError, match="conv_time_series must be a pytree of 3D arrays"
                ),
            ),
            (
                np.zeros([1, 1, 1]),
                does_not_raise(),
            ),
            (
                np.zeros([1]),
                pytest.raises(
                    ValueError, match="conv_time_series must be a pytree of 3D arrays"
                ),
            ),
            ([np.zeros([1, 1, 1])], does_not_raise()),
            ({"nested": [np.zeros([1, 1, 1])]}, does_not_raise()),
            (
                [np.zeros([1, 1, 1, 1])],
                pytest.raises(
                    ValueError, match="conv_time_series must be a pytree of 3D arrays"
                ),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "predictor_causality", ["causal", "acausal", "anti-causal"]
    )
    def test_check_dim(self, pytree, expectation, predictor_causality):
        with expectation:
            utils.nan_pad(pytree, 3, predictor_causality=predictor_causality)

    @pytest.mark.parametrize(
        "predictor_causality", ["causal", "acausal", "anti-causal", ""]
    )
    @pytest.mark.parametrize("iterable", [[np.zeros([2, 4, 5]), np.zeros([1, 1, 10])]])
    def test_conv_type(self, iterable, predictor_causality):
        raise_exception = not (
            predictor_causality in ["causal", "anti-causal", "acausal"]
        )
        if raise_exception:
            with pytest.raises(ValueError, match="predictor_causality must be one of"):
                utils.nan_pad(iterable, 3, predictor_causality)
        else:
            utils.nan_pad(iterable, 3, predictor_causality)

    @pytest.mark.parametrize("iterable", [[np.zeros([2, 4, 5]), np.zeros([2, 4, 6])]])
    @pytest.mark.parametrize("pad_size", [0.1, -1, 0, 1, 2, 3, 5, 6])
    def test_padding_nan_causal(self, pad_size, iterable):
        raise_exception = (not isinstance(pad_size, int)) or (pad_size <= 0)
        if raise_exception:
            with pytest.raises(
                ValueError, match="pad_size must be a positive integer!"
            ):
                utils.nan_pad(iterable, pad_size, "anti-causal")
        else:
            padded = utils.nan_pad(iterable, pad_size, "causal")
            for trial in padded:
                print(trial.shape, pad_size)
            assert all(np.isnan(trial[:pad_size]).all() for trial in padded), (
                "Missing NaNs at the " "beginning of the array!"
            )
            assert all(not np.isnan(trial[pad_size:]).any() for trial in padded), (
                "Found NaNs at the " "end of the array!"
            )
            assert all(
                padded[k].shape[0] == iterable[k].shape[0] + pad_size
                for k in range(len(padded))
            ), "Size after padding doesn't match expectation. Should be T + window_size - 1."

    @pytest.mark.parametrize("iterable", [[np.zeros([2, 5, 4]), np.zeros([2, 6, 4])]])
    @pytest.mark.parametrize("pad_size", [0, 1, 2, 3, 5, 6])
    def test_padding_nan_anti_causal(self, pad_size, iterable):
        raise_exception = (not isinstance(pad_size, int)) or (pad_size <= 0)
        if raise_exception:
            with pytest.raises(
                ValueError, match="pad_size must be a positive integer!"
            ):
                utils.nan_pad(iterable, pad_size, "anti-causal")
        else:
            padded = utils.nan_pad(iterable, pad_size, "anti-causal")
            for trial in padded:
                print(trial.shape, pad_size)
            assert all(
                np.isnan(trial[trial.shape[0] - pad_size :]).all() for trial in padded
            ), ("Missing NaNs at the " "end of the array!")
            assert all(
                not np.isnan(trial[: trial.shape[0] - pad_size]).any()
                for trial in padded
            ), ("Found NaNs at the " "beginning of the array!")
            assert all(
                padded[k].shape[0] == iterable[k].shape[0] + pad_size
                for k in range(len(padded))
            ), "Size after padding doesn't match expectation. Should be T + window_size - 1."

    @pytest.mark.parametrize("iterable", [[np.zeros([2, 5, 4]), np.zeros([2, 6, 4])]])
    @pytest.mark.parametrize("pad_size", [-1, 0.2, 0, 1, 2, 3, 5, 6])
    def test_padding_nan_acausal(self, pad_size, iterable):
        raise_exception = (not isinstance(pad_size, int)) or (pad_size <= 0)
        if raise_exception:
            with pytest.raises(
                ValueError, match="pad_size must be a positive integer!"
            ):
                utils.nan_pad(iterable, pad_size, "acausal")

        else:
            init_nan, end_nan = pad_size // 2, pad_size - pad_size // 2
            padded = utils.nan_pad(iterable, pad_size, "acausal")
            for trial in padded:
                print(trial.shape, pad_size)
            assert all(np.isnan(trial[:init_nan]).all() for trial in padded), (
                "Missing NaNs at the " "beginning of the array!"
            )
            assert all(
                np.isnan(trial[trial.shape[0] - end_nan :]).all() for trial in padded
            ), ("Missing NaNs at the " "end of the array!")

            assert all(
                not np.isnan(trial[init_nan : trial.shape[0] - end_nan]).any()
                for trial in padded
            ), ("Found NaNs in " "the middle of the array!")
            assert all(
                padded[k].shape[0] == iterable[k].shape[0] + pad_size
                for k in range(len(padded))
            ), "Size after padding doesn't match expectation. Should be T + window_size - 1."

    @pytest.mark.parametrize(
        "dtype, expectation",
        [
            (
                np.int8,
                pytest.raises(
                    ValueError, match="conv_time_series must have a float dtype"
                ),
            ),
            (
                jax.numpy.int8,
                pytest.raises(
                    ValueError, match="conv_time_series must have a float dtype"
                ),
            ),
            (np.float32, does_not_raise()),
            (jax.numpy.float32, does_not_raise()),
        ],
    )
    def test_nan_pad_conv_dtype(self, dtype, expectation):
        iterable = np.arange(100).reshape(1, 100, 1, 1).astype(dtype)
        with expectation:
            utils.nan_pad(iterable, 10)


class TestCreateConvolutionalPredictor:

    @pytest.mark.parametrize(
        "basis, expectation",
        [
            (np.ones((3, 1)), does_not_raise()),
            (
                np.ones((2, 1)),
                pytest.warns(
                    UserWarning, match="With `acausal` filter, `basis_matrix.shape"
                ),
            ),
        ],
    )
    def test_warns_even_window(self, basis, expectation):
        with expectation:
            utils.create_convolutional_predictor(
                basis, np.zeros((1, 10, 1)), predictor_causality="acausal", shift=False
            )

    @pytest.mark.parametrize("feature", [np.ones((1, 30, 1)), np.ones((1, 20, 1))])
    @pytest.mark.parametrize(
        "basis",
        [
            np.ones((3, 1)),
            np.ones((2, 1)),
            np.ones((3, 2)),
            np.ones((2, 3)),
        ],
    )
    @pytest.mark.parametrize(
        "shift",
        [
            True,
            False,
            None,
        ],
    )
    @pytest.mark.parametrize(
        "predictor_causality", ["causal", "acausal", "anti-causal"]
    )
    def test_preserve_first_axis_shape(
        self, feature, basis, shift, predictor_causality
    ):
        if predictor_causality == "acausal" and shift:
            return
        res = utils.create_convolutional_predictor(
            basis, feature, predictor_causality=predictor_causality, shift=shift
        )
        assert res.shape[0] == feature.shape[0]

    @pytest.mark.parametrize("feature", [np.zeros((1, 30, 1))])
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
            (2, False, "acausal", [29]),
        ],
    )
    def test_expected_nan(
        self, feature, window_size, shift, predictor_causality, nan_idx
    ):
        basis = np.zeros((window_size, 1))
        res = utils.create_convolutional_predictor(
            basis, feature, predictor_causality=predictor_causality, shift=shift
        )
        other_idx = list(set(np.arange(res.shape[1])).difference(nan_idx))
        assert np.all(np.isnan(res[:, nan_idx]))
        assert not np.any(np.isnan(res[:, other_idx]))

    def test_acausal_shift_error(self):
        basis = np.zeros((3, 1))
        feature = np.zeros((1, 30, 1))
        with pytest.raises(
            ValueError,
            match="Cannot shift `predictor` when `predictor_causality` is `acausal`",
        ):
            utils.create_convolutional_predictor(
                basis, feature, predictor_causality="acausal", shift=True
            )

    def test_basis_len_one_error(self):
        basis = np.zeros((1, 1))
        feature = np.zeros((1, 30, 1))
        with pytest.raises(
            ValueError, match=r"`basis_matrix.shape\[0\]` should be at least 2"
        ):
            utils.create_convolutional_predictor(
                basis, feature, predictor_causality="acausal"
            )

    @pytest.mark.parametrize(
        "feature", [{"1": [[np.ones((30, 1))]], "2": np.ones((20, 1))}]
    )
    @pytest.mark.parametrize(
        "predictor_causality", ["causal", "acausal", "anti-causal"]
    )
    @pytest.mark.parametrize("shift", [True, False, None])
    def test_conv_tree(self, feature, predictor_causality, shift):
        if shift and predictor_causality == "acausal":
            return
        basis = np.zeros((2, 1))
        with does_not_raise():
            utils.create_convolutional_predictor(
                basis, feature, predictor_causality=predictor_causality, shift=shift
            )

    @pytest.mark.parametrize(
        "feature", [{"1": [[np.ones((30, 1))]], "2": np.ones((20, 1))}]
    )
    @pytest.mark.parametrize(
        "predictor_causality", ["causal", "acausal", "anti-causal"]
    )
    @pytest.mark.parametrize("shift", [True, False, None])
    def test_conv_tree_shape(self, feature, predictor_causality, shift):
        if shift and predictor_causality == "acausal":
            return
        basis = np.zeros((2, 1))
        res = utils.create_convolutional_predictor(
            basis, feature, predictor_causality=predictor_causality, shift=shift
        )
        arr1, arr2 = jax.tree_util.tree_flatten(res)[0]
        assert arr1.shape[0] == 30
        assert arr2.shape[0] == 20


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
