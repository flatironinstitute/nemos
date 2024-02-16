import jax.numpy as jnp
import numpy as np
import pynapple as nap
import pytest

import nemos as nmo


@pytest.mark.parametrize(
    "inp, jax_func, expected_type",
    [
        ([jnp.arange(3)], lambda x: jnp.power(x, 2), jnp.ndarray),
        ([np.arange(3)], lambda x: jnp.power(x, 2), jnp.ndarray),
        ([nap.Tsd(t=np.arange(3), d=np.arange(3))], lambda x: jnp.power(x, 2), nap.Tsd),
        (
            [nap.TsdFrame(t=np.arange(3), d=np.expand_dims(np.arange(3), 1))],
            lambda x: jnp.power(x, 2),
            nap.TsdFrame,
        ),
        (
            [nap.TsdTensor(t=np.arange(3), d=np.expand_dims(np.arange(3), (1, 2, 3)))],
            lambda x: jnp.power(x, 2),
            nap.TsdTensor,
        ),
        ([nmo.glm.GLM()], lambda x: x, nmo.glm.GLM),
        (
            [nap.Tsd(t=np.arange(3), d=np.arange(3)), np.arange(3)],
            lambda x, y: (x + y, y),
            (nap.Tsd, nap.Tsd),
        ),
        (
            [nap.Tsd(t=np.arange(3), d=np.arange(3)), np.expand_dims(np.arange(3), 1)],
            lambda x, y: (np.power(x, 2), x + y),
            (nap.Tsd, nap.TsdFrame),
        ),
        (
            [
                nap.Tsd(t=np.arange(3), d=np.arange(3)),
                np.expand_dims(np.arange(3), (1, 2)),
            ],
            lambda x, y: (np.power(x, 2), x + y),
            (nap.Tsd, nap.TsdTensor),
        ),
        # Test adding a scalar to different types of inputs
        ([np.arange(3)], lambda x: x + 5, jnp.ndarray),
        ([jnp.arange(3)], lambda x: x + 5, jnp.ndarray),
        ([nap.Tsd(t=np.arange(3), d=np.arange(3))], lambda x: x + 5, nap.Tsd),

        # Test element-wise multiplication between array and Tsd
        (
                [np.arange(3), nap.Tsd(t=np.arange(3), d=np.arange(3))],
                lambda x, y: x * y,
                nap.Tsd,
        ),

        # Test concatenation of TsdFrames along a new dimension
        (
                [nap.TsdFrame(t=np.arange(3), d=np.arange(3).reshape(-1, 1)),
                 nap.TsdFrame(t=np.arange(3), d=np.arange(3, 6).reshape(-1, 1))],
                lambda x, y: np.concatenate([x, y], axis=1),
                nap.TsdFrame,
        ),

        # Test operation that reduces dimensionality, from TsdTensor to TsdFrame
        (
                [nap.TsdTensor(t=np.arange(3), d=np.random.rand(3, 2, 2))],
                lambda x: x.mean(axis=2),  # Reduce last dimension
                nap.TsdFrame,
        ),

        # Test mixing JAX array and NumPy array inputs resulting in a JAX array
        (
                [jnp.arange(3), np.arange(3)],
                lambda x, y: x + y,
                jnp.ndarray,
        ),

        # Test operation involving three inputs of mixed types
        (
                [np.arange(3), jnp.arange(3, 6), nap.Tsd(t=np.arange(6, 9), d=np.arange(6, 9))],
                lambda x, y, z: (x + y, y + z),
                (nap.Tsd, nap.Tsd),
        ),
    ],
)
def test_decorator_output_type(inp, jax_func, expected_type):
    """Validate that the `cast_jax` decorator correctly casts output types based on input types."""
    @nmo.type_casting.cast_jax
    def func(*x):
        return jax_func(*x)

    out = func(*inp)
    assert nmo.utils.pytree_map_and_reduce(
        lambda x, y: isinstance(x, y), all, out, expected_type
    )


@pytest.mark.parametrize(
    "iset",
    [
        nap.IntervalSet(start=[0], end=[1]),
        nap.IntervalSet(start=[0, 0.5], end=[0.49, 1]),
    ],
)
def test_predict_nap_output(iset, poissonGLM_model_instantiation):
    """Ensure predictions using pynapple objects correctly respect interval sets."""
    X, y, model, true_params, firing_rate = poissonGLM_model_instantiation
    n_samp = X.shape[0]
    time = jnp.linspace(0, 1, n_samp)
    tsd_X = nap.TsdTensor(t=time, d=X).restrict(iset)
    model.fit(X, y)
    # run predict on tsd
    pred_rate = model.predict(tsd_X)
    assert isinstance(pred_rate, nap.TsdFrame)
    assert np.all(iset.values == pred_rate.time_support.values)


@pytest.mark.parametrize(
    "inp, expected",
    [
        (np.array([1, 2, 3]), True),
        (jnp.array([1, 2, 3]), True),
        ([1, 2, 3], False),
        (3, False),
        ("not an array", False),
        (nap.Tsd(t=np.arange(3), d=np.arange(3)), True),  # Assuming Tsd objects have array-like properties
    ],
)
def test_is_numpy_array_like(inp, expected):
    """Check if various objects are correctly identified as numpy array-like or not."""
    assert nmo.type_casting.is_numpy_array_like(inp) == expected


@pytest.mark.parametrize(
    "inp, expected",
    [
        (nap.Tsd(t=np.arange(3), d=np.arange(3)), True),
        (nap.TsdFrame(t=np.arange(3), d=np.arange(3).reshape(-1, 1)), True),
        (nap.TsdTensor(t=np.arange(3), d=np.arange(3).reshape(-1, 1, 1)), True),
        (np.array([1, 2, 3]), False),
        ([1, 2, 3], False),
    ],
)
def test_is_pynapple_tsd(inp, expected):
    """Verify identification of pynapple Tsd/TsdFrame/TsdTensor objects amongst inputs."""
    assert nmo.type_casting.is_pynapple_tsd(inp) == expected


@pytest.mark.parametrize(
    "inputs, expected",
    [
        (
            [nap.Tsd(t=np.arange(3), d=np.arange(3)),
             nap.Tsd(t=np.arange(3), d=np.arange(3))],
            True,
        ),
        (
            [nap.Tsd(t=np.arange(3), d=np.arange(3)),
             nap.Tsd(t=np.arange(3, 6), d=np.arange(3))],
            False,
        ),
    ],
)
def test_all_same_time_info(inputs, expected):
    """Assess if provided pynapple objects share consistent time axis and support information."""
    assert nmo.type_casting.all_same_time_info(*inputs) == expected


@pytest.mark.parametrize(
    "array, time, time_support, expected_type",
    [
        (np.random.rand(3), np.arange(3), nap.IntervalSet(start=[0], end=[3]), nap.Tsd),
        (np.random.rand(3, 1), np.arange(3), nap.IntervalSet(start=[0], end=[3]), nap.TsdFrame),
        (np.random.rand(3, 2, 2), np.arange(3), nap.IntervalSet(start=[0], end=[3]), nap.TsdTensor),
    ],
)
def test_cast_to_pynapple(array, time, time_support, expected_type):
    """Check the conversion from array to the correct pynapple object type based on dimensions."""
    result = nmo.type_casting.cast_to_pynapple(array, time, time_support)
    assert isinstance(result, expected_type)


@pytest.mark.parametrize(
    "inp, expected",
    [
        (np.array([1, 2, 3]), jnp.ndarray),
        ([1, 2, 3], list),  # Assuming list should be converted
        ("not an array", str),  # Assuming strings are not converted
    ],
)
def test_jnp_asarray_if(inp, expected):
    """Evaluate conditional conversion to JAX array based on input characteristics."""
    result = nmo.type_casting.jnp_asarray_if(inp)
    assert isinstance(result, expected)
