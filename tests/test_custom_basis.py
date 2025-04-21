from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
from numpy.typing import NDArray

from nemos.basis._custom_basis import apply_f_vectorized


@pytest.fixture
def vectorize_func(request):
    ndim = request.param

    def not_vectorized_func(*xi) -> NDArray:
        if any(x.ndim != ndim for x in xi):
            raise ValueError("Non-vectorized function")
        # sum over second dimension and add results
        return sum(np.sum(x.reshape(x.shape[0], -1), axis=1) for x in xi)

    return (
        lambda *xi: apply_f_vectorized(not_vectorized_func, *xi, ndim_input=ndim),
        not_vectorized_func,
    )


@pytest.fixture
def vec_func_kwargs(request):
    kwargs = request.param

    def func(*xi, add=0) -> NDArray:
        return xi[0] + add

    return lambda *xi, **k: apply_f_vectorized(func, *xi, ndim_input=1, **k), kwargs


@pytest.mark.parametrize(
    "vectorize_func, x, expectation",
    [
        (
            1,
            np.random.randn(
                10,
            ),
            does_not_raise(),
        ),
        (
            1,
            np.random.randn(10, 2),
            pytest.raises(ValueError, match="Non-vectorized function"),
        ),
        (2, np.random.randn(10, 2), does_not_raise()),
        (
            2,
            np.random.randn(10, 2, 1),
            pytest.raises(ValueError, match="Non-vectorized function"),
        ),
    ],
    indirect=["vectorize_func"],
)
def test_vec_function_dims(vectorize_func, x, expectation):
    vec_f, regular_f = vectorize_func
    # should be always fine
    try:
        vec_f(x)
    except Exception:
        raise ValueError("Failed vectorization.")
    with expectation:
        regular_f(x)


@pytest.mark.parametrize(
    "vectorize_func, x, expected_out_shape",
    [
        (
            1,
            (
                np.random.randn(
                    10,
                ),
            ),
            (10, 1),
        ),
        # vectorized over second dimension
        (1, (np.random.randn(10, 2),), (10, 2)),
        # no vectorized dimension (mean over all axis except first)
        (2, (np.random.randn(10, 2),), (10, 1)),
        # no vectorized dimension (mean over all axis except first)
        (2, (np.random.randn(10, 2), np.random.randn(10, 3)), (10, 1)),
        # vectorize 3rd dimension
        (2, (np.random.randn(10, 2, 3), np.random.randn(10, 2, 2)), (10, 6)),
    ],
    indirect=["vectorize_func"],
)
def test_vec_function_output_shape(vectorize_func, x, expected_out_shape):
    vec_f, regular_f = vectorize_func
    out = vec_f(*x)
    assert out.shape == expected_out_shape


@pytest.mark.parametrize(
    "vec_func_kwargs, x",
    [(dict(add=1), np.zeros((10, 2))), (dict(add=0), np.zeros((10, 2)))],
    indirect=["vec_func_kwargs"],
)
def test_vec_function_kwargs(vec_func_kwargs, x):
    vec_f, kwargs = vec_func_kwargs
    out = vec_f(x, **kwargs)
    assert np.all(out == x + kwargs["add"])
