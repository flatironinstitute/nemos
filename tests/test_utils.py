from contextlib import nullcontext as does_not_raise

import jax
import jax.numpy as jnp
import numpy as np
import pynapple as nap
import pytest

from nemos import utils


@pytest.mark.parametrize(
    "arrays, expected_out",
    [
        ([jnp.zeros((10, 1)), np.zeros((10, 1))], jnp.zeros((10, 2))),
        ([np.zeros((10, 1)), np.zeros((10, 1))], jnp.zeros((10, 2))),
        (
            [np.zeros((10, 1)), nap.TsdFrame(t=np.arange(10), d=np.zeros((10, 1)))],
            nap.TsdFrame(t=np.arange(10), d=np.zeros((10, 2))),
        ),
        (
            [
                nap.TsdFrame(t=np.arange(10), d=np.zeros((10, 1))),
                nap.TsdFrame(t=np.arange(10), d=np.zeros((10, 1))),
            ],
            nap.TsdFrame(t=np.arange(10), d=np.zeros((10, 2))),
        ),
        (
            [
                nap.TsdTensor(t=np.arange(10), d=np.zeros((10, 1, 2))),
                nap.TsdTensor(t=np.arange(10), d=np.zeros((10, 1, 2))),
            ],
            nap.TsdTensor(t=np.arange(10), d=np.zeros((10, 2, 2))),
        ),
    ],
)
def test_concatenate_eval(arrays, expected_out):
    """Test various combination of arrays and pyapple time series."""
    out = utils.pynapple_concatenate(arrays)
    if hasattr(expected_out, "times"):
        assert np.all(out.d == expected_out.d)
        assert np.all(out.t == expected_out.t)
        assert np.all(out.time_support.values == expected_out.time_support.values)
    else:
        assert np.all(out == expected_out)


@pytest.mark.parametrize(
    "axis, arrays, expected_shape",
    [
        (0, [jnp.zeros((10,)), np.zeros((10,))], 20),
        (0, [jnp.zeros((10, 1)), np.zeros((10, 1))], 20),
        (1, [jnp.zeros((10, 1)), np.zeros((10, 1))], 2),
        (
            2,
            [
                nap.TsdTensor(t=np.arange(10), d=np.zeros((10, 1, 2))),
                nap.TsdTensor(t=np.arange(10), d=np.zeros((10, 1, 2))),
            ],
            4,
        ),
    ],
)
def test_concatenate_axis(arrays, axis, expected_shape):
    """Test various combination of arrays and pyapple time series."""
    assert utils.pynapple_concatenate(arrays, axis).shape[axis] == expected_shape


@pytest.mark.parametrize(
    "dtype, arrays",
    [
        (np.int32, [jnp.zeros((10, 1)), np.zeros((10, 1))]),
        (np.float32, [jnp.zeros((10, 1)), np.zeros((10, 1))]),
        (
            np.int32,
            [
                nap.TsdTensor(t=np.arange(10), d=np.zeros((10, 1, 2))),
                nap.TsdTensor(t=np.arange(10), d=np.zeros((10, 1, 2))),
            ],
        ),
    ],
)
def test_concatenate_type(arrays, dtype):
    """Test various combination of arrays and pyapple time series."""
    assert utils.pynapple_concatenate(arrays, dtype=dtype).dtype == dtype


class TestPadding:

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

    @pytest.mark.parametrize("causality", ["causal", "acausal", "anti-causal"])
    @pytest.mark.parametrize("pad_size", [1, 2])
    @pytest.mark.parametrize(
        "array, axis, expectation",
        [
            (jnp.zeros((10,)), 0, does_not_raise()),
            (jnp.zeros((10, 11)), 0, does_not_raise()),
            (jnp.zeros((10, 11)), 1, does_not_raise()),
            (
                jnp.zeros((10,)),
                1,
                pytest.raises(
                    ValueError, match="'axis' must be smaller than the number "
                ),
            ),
            (
                jnp.zeros((10, 11)),
                2,
                pytest.raises(
                    ValueError, match="'axis' must be smaller than the number "
                ),
            ),
            (
                jnp.zeros((10, 11)),
                0.0,
                pytest.raises(
                    ValueError, match="`axis` must be a non negative integer"
                ),
            ),
            (
                jnp.zeros((10, 11)),
                "x",
                pytest.raises(
                    ValueError, match="`axis` must be a non negative integer"
                ),
            ),
        ],
    )
    def test_axis_compatibility(self, pad_size, array, causality, axis, expectation):
        with expectation:
            utils.nan_pad(array, pad_size, causality, axis=axis)

    @pytest.mark.parametrize("causality", ["causal", "acausal", "anti-causal"])
    @pytest.mark.parametrize(
        "pad_size, expectation",
        [
            (
                -1,
                pytest.raises(ValueError, match="pad_size must be a positive integer"),
            ),
            (
                1.0,
                pytest.raises(ValueError, match="pad_size must be a positive integer"),
            ),
            (1, does_not_raise()),
            (2, does_not_raise()),
        ],
    )
    @pytest.mark.parametrize("array", [jnp.zeros((10,)), np.zeros((10, 11))])
    def test_pad_size_type(self, pad_size, array, causality, expectation):
        with expectation:
            utils.nan_pad(array, pad_size, causality, axis=0)

    @pytest.mark.parametrize(
        "causality, pad_size, expectation",
        [
            ("causal", 1, does_not_raise()),
            (
                "acausal",
                1,
                pytest.warns(
                    UserWarning,
                    match="With acausal filter, pad_size should probably be even",
                ),
            ),
            ("anti-causal", 1, does_not_raise()),
            ("causal", 2, does_not_raise()),
            ("acausal", 2, does_not_raise()),
            ("anti-causal", 2, does_not_raise()),
        ],
    )
    def test_pad_window_size(self, pad_size, causality, expectation):
        array = jnp.zeros((10,))
        with expectation:
            utils.nan_pad(array, pad_size, causality, axis=0)


class TestShiftTimeSeries:

    @pytest.mark.parametrize(
        "predictor_causality, expectation",
        [
            ("causal", does_not_raise()),
            ("anti-causal", does_not_raise()),
            (
                "invalid",
                pytest.raises(ValueError, match="predictor_causality must be one of"),
            ),
        ],
    )
    def test_causality_validation(self, predictor_causality, expectation):
        """Ensure the function rejects invalid predictor_causality values."""
        time_series = np.array([1.0, 2.0, 3.0])
        with expectation:
            utils.shift_time_series(time_series, predictor_causality)

    @pytest.mark.parametrize(
        "dtype, expectation",
        [
            (np.float32, does_not_raise()),
            (
                np.int32,
                pytest.raises(ValueError, match="time_series must have a float dtype"),
            ),
        ],
    )
    def test_dtype_validation(self, dtype, expectation):
        """Check that the function raises an error for non-float data types."""
        time_series = np.array([1, 2, 3], dtype=dtype)
        with expectation:
            utils.shift_time_series(time_series, "causal")

    @pytest.mark.parametrize(
        "axis, expectation",
        [
            (0, does_not_raise()),  # Assuming time_series is 1D for simplicity
            (
                1,
                pytest.raises(
                    ValueError,
                    match="'axis' must be smaller than the number of dimensions",
                ),
            ),
        ],
    )
    def test_axis_validation(self, axis, expectation):
        """Validate the function's handling of the axis parameter."""
        time_series = np.array([1.0, 2.0, 3.0])
        with expectation:
            utils.shift_time_series(time_series, "causal", axis)

    @pytest.mark.parametrize(
        "shape, predictor_causality, axis, expectation",
        [
            ((5, 5), "causal", 1, does_not_raise()),
            ((5, 5), "anti-causal", 0, does_not_raise()),
            (
                (5, 5),
                "invalid",
                0,
                pytest.raises(ValueError, match="predictor_causality must be one of"),
            ),
        ],
    )
    def test_shift_in_multidimensional_array(
        self, shape, predictor_causality, axis, expectation
    ):
        """Ensure correct shifting in multidimensional arrays along a specified axis."""
        time_series = np.zeros(shape)
        with expectation:
            shifted_series = utils.shift_time_series(
                time_series, predictor_causality, axis
            )
            if expectation == does_not_raise():
                # Check for NaN at the expected location
                if predictor_causality == "causal":
                    assert np.isnan(
                        shifted_series.take(0, axis=axis)
                    ).all(), (
                        "First element along the axis should be NaN for causal shift."
                    )
                    # Ensure no NaNs elsewhere
                    assert not np.isnan(
                        shifted_series.take(
                            range(1, shifted_series.shape[axis]), axis=axis
                        )
                    ).any(), "Unexpected NaNs found in the array."
                else:  # anti-causal
                    assert np.isnan(
                        shifted_series.take(-1, axis=axis)
                    ).all(), "Last element along the axis should be NaN for anti-causal shift."
                    # Ensure no NaNs elsewhere
                    assert not np.isnan(
                        shifted_series.take(
                            range(0, shifted_series.shape[axis] - 1), axis=axis
                        )
                    ).any(), "Unexpected NaNs found in the array."

    @pytest.mark.parametrize(
        "predictor_causality, axis, expectation",
        [
            ("causal", 0, does_not_raise()),
            ("anti-causal", 0, does_not_raise()),
        ],
    )
    def test_shift_with_pytree(self, predictor_causality, axis, expectation):
        """Test shifting functionality with a pytree of arrays."""
        time_series = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
        with expectation:
            shifted_series = utils.shift_time_series(
                time_series, predictor_causality, axis
            )
            for original, shifted in zip(time_series, shifted_series):
                if predictor_causality == "causal":
                    expected = np.concatenate(([np.nan], original[:-1]))
                else:  # anti-causal
                    expected = np.concatenate((original[1:], [np.nan]))
                np.testing.assert_array_equal(shifted, expected)

    def test_causal_shift(self):
        # Assuming time_series shape is (1, 3, 2, 2)
        time_series = np.random.rand(1, 3, 2, 2).astype(np.float32)
        shifted_series = utils.shift_time_series(time_series, "causal", axis=1)

        assert np.isnan(
            shifted_series[0, 0]
        ).all(), "First time bin should be NaN for causal shift"
        assert np.array_equal(
            shifted_series[0, 1:], time_series[0, :-1]
        ), "Causal shift did not work as expected"

    def test_anti_causal_shift(self):
        time_series = np.random.rand(1, 3, 2, 2).astype(np.float32)
        shifted_series = utils.shift_time_series(time_series, "anti-causal", axis=1)

        assert np.isnan(
            shifted_series[0, -1]
        ).all(), "Last time bin should be NaN for anti-causal shift"
        assert np.array_equal(
            shifted_series[0, :-1], time_series[0, 1:]
        ), "Anti-causal shift did not work as expected"
