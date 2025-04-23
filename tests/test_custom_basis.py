from contextlib import nullcontext as does_not_raise

import numpy as np
import pynapple as nap
import pytest
from conftest import (
    basis_collapse_all_non_vec_axis,
    basis_with_add_kwargs,
    custom_basis,
    custom_basis_2d,
)
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


@pytest.mark.parametrize("n_basis_funcs", [4])
@pytest.mark.parametrize(
    "input_dim, expectation",
    [
        (1, does_not_raise()),
        (2, pytest.raises(ValueError, match="Each input must have at least")),
    ],
)
def test_input_dimension(n_basis_funcs, input_dim, expectation):
    bas = custom_basis(n_basis_funcs=n_basis_funcs, label=None, ndim_input=input_dim)
    with expectation:
        shape = (10, *(1 for _ in range(1 - 1)))
        bas.compute_features(*shape)


@pytest.mark.parametrize("n_basis_funcs", [4, 5])
def test_custom_basis_feature_1d_shape(n_basis_funcs):
    bas = custom_basis(n_basis_funcs=n_basis_funcs, label=None)
    out = bas.compute_features(
        np.random.randn(
            10,
        )
    )
    assert out.shape == (10, n_basis_funcs)
    out = bas.compute_features(np.random.randn(10, 2, 3))
    assert out.shape == (10, 6 * n_basis_funcs)


@pytest.mark.parametrize("n_basis_funcs", [4, 5])
def test_custom_basis_feature_2d_shape(n_basis_funcs):
    bas = custom_basis(n_basis_funcs=n_basis_funcs, label=None, ndim_input=1)
    out = bas.compute_features(
        np.random.randn(
            10,
        )
    )
    assert out.shape == (10, n_basis_funcs)
    out = bas.compute_features(np.random.randn(10, 2, 3, 2))
    assert out.shape == (10, 12 * n_basis_funcs)

    # define a basis that behaves differently on vec
    bas = basis_collapse_all_non_vec_axis(n_basis_funcs, ndim_input=1)
    # vec second dim
    out = bas.compute_features(np.random.randn(10, 2))
    assert out.shape == (10, n_basis_funcs * 2)
    out = bas.compute_features(np.random.randn(10, 2, 2))
    assert out.shape == (10, n_basis_funcs * 4)


@pytest.mark.parametrize(
    "out_shape, expectation",
    [
        (1, does_not_raise()),
        (-1, pytest.raises(ValueError, match="Output shape must be strictly")),
        (2, does_not_raise()),
        ((1, 2), does_not_raise()),
        (0.5, pytest.raises(TypeError, match="`output_shape` must be an iterable of")),
        (
            ("a", 2),
            pytest.raises(
                ValueError, match="The tuple provided contains non integer values"
            ),
        ),
    ],
)
def test_output_shape_setter(out_shape, expectation):
    with expectation:
        custom_basis(n_basis_funcs=10, output_shape=out_shape, ndim_input=1)


def test_output_shape_reset():
    bas = custom_basis(n_basis_funcs=3, output_shape=(2, 3, 4), ndim_input=1)
    assert bas.output_shape == (2, 3, 4)
    bas.compute_features(np.linspace(0, 1, 10))
    # the output shape is set to empty tuple because each basis function
    # returns a flat array
    assert bas.output_shape == ()


@pytest.mark.parametrize("kwargs", [dict(add=0), dict(add=1), dict(), None])
def test_kwargs_apply(kwargs):
    bas = basis_with_add_kwargs(basis_kwargs=kwargs)
    if kwargs:
        assert np.all(
            np.full((10, 1), kwargs["add"]) == bas.compute_features(np.zeros(10))
        )
    else:
        assert bas.basis_kwargs == {}


@pytest.mark.parametrize(
    "kwargs, expectation",
    [
        (dict(add=0), does_not_raise()),
        (1, pytest.raises(ValueError, match="`basis_kwargs` must be a dictionary")),
        (dict(), does_not_raise()),
        (None, does_not_raise()),
        (dict(ndim_input=1), pytest.raises(ValueError, match="Invalid kwargs name")),
    ],
)
def test_basis_kwargs_set(kwargs, expectation):
    with expectation:
        basis_with_add_kwargs(basis_kwargs=kwargs)
    bas = basis_with_add_kwargs(basis_kwargs=None)
    with expectation:
        bas.basis_kwargs = kwargs


def test_pynapple_support():
    bas = custom_basis(5, pynapple_support=True)
    x = nap.Tsd(np.arange(10), np.linspace(0, 1, 10))
    assert isinstance(bas.compute_features(x), nap.TsdFrame)
    bas = custom_basis(5, pynapple_support=False)
    with pytest.raises(TypeError):
        bas.compute_features(x)


@pytest.mark.parametrize("ps", [True, False, None, 1, 0, -1, np.array([1, 2])])
def test_pynapple_support_type(ps):
    """Test that any value that can be parsed to bool is valid."""
    if isinstance(ps, np.ndarray):
        expect = pytest.raises(ValueError, match="The truth")
    else:
        expect = does_not_raise()
    with expect:
        b = custom_basis(5, pynapple_support=ps)
        assert isinstance(b.pynapple_support, bool)


@pytest.mark.parametrize(
    "inp_dim, vec_shape, expected_num", [(1, [], 5), (2, [], 10), (2, [3], 30)]
)
def test_n_output_features_match(inp_dim, expected_num, vec_shape):
    shape = [10] + [2] * (inp_dim - 1) + vec_shape
    x = np.random.randn(*shape)
    bas = custom_basis(5, ndim_input=inp_dim)
    out = bas.compute_features(x)
    assert out.shape[1] == bas.n_output_features == expected_num


@pytest.mark.parametrize("ps", [True, False])
def test_basis_repr(ps):
    """Check that repr strips the expectation"""
    bas = custom_basis(5, pynapple_support=ps)
    assert (
        repr(bas)
        == f"CustomBasis(\n    funcs=[partial(power_func, 1), ..., partial(power_func, 5)],\n    ndim_input=1,\n    pynapple_support={ps}\n)"
    )
    bas = custom_basis(1, pynapple_support=ps)
    assert (
        repr(bas)
        == f"CustomBasis(\n    funcs=[partial(power_func, 1)],\n    ndim_input=1,\n    pynapple_support={ps}\n)"
    )


@pytest.mark.parametrize("input_shape", [(1,), (1, 2), (1, 2, 3), ()])
def test_split_by_features_shape(input_shape):
    bas = custom_basis(4)
    out = bas.compute_features(np.random.randn(10, *input_shape))
    split = bas.split_by_feature(out, axis=1)["CustomBasis"]
    assert split.shape == (10, *input_shape, 4)


@pytest.mark.parametrize(
    "ishape, n_out_features",
    [
        ((1, 1), 5),
        ((1, 2), 10),
        ((2, 1), 10),
        ((2, 2), 20),
        (((2, 2), 1), 20),
        ((1, (2, 2)), 20),
        (((2, 2), (2, 2)), 80),
    ],
)
def test_set_input_shape_2d(ishape, n_out_features):
    """Test that the output features match expectation when setting input shape.

    Note that the 1D case is tested in test_basis.py::TestSharedMethods.
    """
    bas = custom_basis_2d(5)
    bas.set_input_shape(*ishape)
    assert bas.n_output_features == n_out_features
