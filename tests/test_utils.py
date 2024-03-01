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
        ([np.zeros((10, 1)), nap.TsdFrame(t=np.arange(10), d=np.zeros((10, 1)))],
         nap.TsdFrame(t=np.arange(10), d=np.zeros((10, 2)))),
        ([nap.TsdFrame(t=np.arange(10), d=np.zeros((10, 1))), nap.TsdFrame(t=np.arange(10), d=np.zeros((10, 1)))],
         nap.TsdFrame(t=np.arange(10), d=np.zeros((10, 2)))),
        ([nap.TsdTensor(t=np.arange(10), d=np.zeros((10, 1, 2))), nap.TsdTensor(t=np.arange(10), d=np.zeros((10, 1, 2)))],
         nap.TsdTensor(t=np.arange(10), d=np.zeros((10, 2, 2)))),
    ]
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
        (0, [jnp.zeros((10, )), np.zeros((10, ))], 20),
        (0, [jnp.zeros((10, 1)), np.zeros((10, 1))], 20),
        (1, [jnp.zeros((10, 1)), np.zeros((10, 1))], 2),
        (2, [nap.TsdTensor(t=np.arange(10), d=np.zeros((10, 1, 2))),
             nap.TsdTensor(t=np.arange(10), d=np.zeros((10, 1, 2)))],4),
    ]
)
def test_concatenate_axis(arrays, axis, expected_shape):
    """Test various combination of arrays and pyapple time series."""
    assert utils.pynapple_concatenate(arrays, axis).shape[axis] == expected_shape


@pytest.mark.parametrize(
    "dtype, arrays",
    [
        (np.int32, [jnp.zeros((10, 1)), np.zeros((10, 1))]),
        (np.float32, [jnp.zeros((10, 1)), np.zeros((10, 1))]),
        (np.int32, [nap.TsdTensor(t=np.arange(10), d=np.zeros((10, 1, 2))),
                 nap.TsdTensor(t=np.arange(10), d=np.zeros((10, 1, 2)))]),
    ]
)
def test_concatenate_type(arrays, dtype):
    """Test various combination of arrays and pyapple time series."""
    assert utils.pynapple_concatenate(arrays, dtype=dtype).dtype == dtype

