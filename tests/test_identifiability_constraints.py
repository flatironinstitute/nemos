import warnings
from contextlib import nullcontext as does_not_raise

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nemos.basis.basis import BSplineConv, BSplineEval, RaisedCosineLinearEval
from nemos.identifiability_constraints import (
    _WARN_FLOAT32_MESSAGE,
    _find_drop_column,
    _warn_if_not_float64,
    add_constant,
    apply_identifiability_constraints,
    apply_identifiability_constraints_by_basis_component,
)


@pytest.mark.parametrize(
    "dtype, expected_context, filter_type",
    [
        (
            jnp.int64,
            pytest.warns(
                UserWarning, match="The feature matrix is not of dtype `float64`"
            ),
            "default",
        ),
        (
            jnp.float32,
            pytest.warns(
                UserWarning, match="The feature matrix is not of dtype `float64`"
            ),
            "default",
        ),
        (jnp.float64, does_not_raise(), "error"),
    ],
)
def test_warn_if_not_float64(dtype, expected_context, filter_type):
    """Test warnings raised for float32 matrices."""
    jax.config.update("jax_enable_x64", True)
    matrix = jnp.ones((5, 5), dtype=dtype)
    with expected_context:
        with warnings.catch_warnings():
            warnings.filterwarnings(action=filter_type)
            _warn_if_not_float64(matrix, _WARN_FLOAT32_MESSAGE)


@pytest.mark.parametrize(
    "input_matrix, expected_output",
    [
        (jnp.array([[1, 2], [3, 4]]), jnp.array([[1, 1, 2], [1, 3, 4]])),
        (jnp.array([[0]]), jnp.array([[1, 0]])),
    ],
)
def test_add_constant(input_matrix, expected_output):
    """Test add_constant for adding intercept columns."""
    result = add_constant(input_matrix)
    assert jnp.array_equal(result, expected_output)


@pytest.mark.parametrize(
    "matrix, expected_shape, expected_columns",
    [
        (jnp.array([[1], [0]]), (2, 1), jnp.array([0])),
        (jnp.eye(5), (5, 4), jnp.array([1, 2, 3, 4])),
        (jnp.array([[1, 2], [3, 4]]), (2, 1), jnp.array([1])),
        (jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), (3, 1), jnp.array([2])),
        (jnp.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]]), (3, 2), jnp.array([0, 2])),
    ],
)
def test_apply_identifiability_constraints(matrix, expected_shape, expected_columns):
    """Test apply_identifiability_constraints for both full-rank and rank-deficient cases."""
    constrained_x, kept_columns = apply_identifiability_constraints(
        matrix.astype(float), warn_if_float32=False
    )
    assert constrained_x.shape == expected_shape
    assert jnp.array_equal(kept_columns, expected_columns)


@pytest.mark.parametrize(
    "add_intercept, expected_shape", [(True, (2, 0)), (False, (2, 1))]
)
def test_apply_identifiability_constraints_add_constant(add_intercept, expected_shape):
    out, _ = apply_identifiability_constraints(
        np.ones((2, 1)), warn_if_float32=False, add_intercept=add_intercept
    )
    assert out.shape == expected_shape


@pytest.mark.parametrize(
    "basis, input_shape, output_shape, expected_columns",
    [
        (RaisedCosineLinearEval(10, width=4), (50,), (50, 10), jnp.arange(10)),
        (
            BSplineEval(5) + BSplineEval(6),
            (20,),
            (20, 9),
            jnp.array([1, 2, 3, 4, 6, 7, 8, 9, 10]),
        ),
        (
            BSplineConv(5, window_size=10) + BSplineEval(6),
            (20,),
            (20, 10),
            jnp.array([0, 1, 2, 3, 4, 6, 7, 8, 9, 10]),
        ),
        (BSplineEval(5), (10,), (10, 4), jnp.arange(1, 5)),
    ],
)
def test_apply_identifiability_constraints_by_basis_component(
    basis, input_shape, output_shape, expected_columns
):
    """Test constraints applied by basis component."""
    x = basis.compute_features(
        *([np.random.randn(*input_shape)] * basis._n_input_dimensionality)
    )
    constrained_x, kept_columns = apply_identifiability_constraints_by_basis_component(
        basis, x
    )
    assert constrained_x.shape == output_shape
    assert jnp.array_equal(expected_columns, kept_columns)


@pytest.mark.parametrize(
    "matrix, max_drop, preproc, expected_result",
    [
        (
            jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            2,
            add_constant,
            jnp.array([True, True, False]),
        ),
        (
            jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            1,
            add_constant,
            jnp.array([True, False, False]),
        ),
        (jnp.array([[1, 0], [0, 0]]), 1, add_constant, jnp.array([False, True])),
        (jnp.array([[1, 0], [0, 0]]), 0, add_constant, jnp.array([False, False])),
        (np.eye(3, 2), 1, add_constant, jnp.array([False, False])),
        (np.eye(3), 3, add_constant, jnp.array([True, False, False])),
        (
            jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            2,
            lambda x: x,
            jnp.array([True, False, False]),
        ),
        (
            jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            1,
            lambda x: x,
            jnp.array([True, False, False]),
        ),
        (jnp.array([[1, 0], [0, 0]]), 1, lambda x: x, jnp.array([False, True])),
        (jnp.array([[1, 0], [0, 0]]), 0, lambda x: x, jnp.array([False, False])),
        (np.eye(3, 2), 1, lambda x: x, jnp.array([False, False])),
        (np.eye(3), 3, lambda x: x, jnp.array([False, False, False])),
    ],
)
def test_find_drop_column(matrix, max_drop, expected_result, preproc):
    """Test if a column is linearly dependent."""
    jax.config.update("jax_enable_x64", True)
    rank = int(jnp.linalg.matrix_rank(preproc(matrix)))
    result = _find_drop_column(
        matrix.astype(float), rank=rank, max_drop=max_drop, preprocessing_func=preproc
    )
    assert jnp.array_equal(result, expected_result)


@pytest.mark.parametrize(
    "dtype, expected_dtype",
    [
        (jnp.float32, jnp.float32),
        (jnp.float64, jnp.float64),
    ],
)
def test_feature_matrix_dtype(dtype, expected_dtype):
    """Test if the matrix retains its dtype after applying constraints."""
    jax.config.update("jax_enable_x64", True)
    x = np.random.randn(10, 5).astype(dtype)
    constrained_x, _ = apply_identifiability_constraints(x, warn_if_float32=False)
    assert constrained_x.dtype == expected_dtype


@pytest.mark.parametrize(
    "invalid_entries",
    [[np.nan, np.nan], [np.nan, np.inf], [np.inf, np.inf], [np.inf, np.inf]],
)
def test_apply_constraint_with_invalid(invalid_entries):
    """Test if the matrix retains its dtype after applying constraints."""
    jax.config.update("jax_enable_x64", True)
    x = np.random.randn(10, 5)
    # add invalid
    x[:2, 2] = invalid_entries
    # make rank deficient
    x[:, 0] = np.sum(x[:, 1:], axis=1)
    constrained_x, kept_cols = apply_identifiability_constraints(x)
    assert jnp.array_equal(kept_cols, jnp.arange(1, 5))
    assert constrained_x.shape[0] == x.shape[0]
    assert jnp.all(jnp.isnan(constrained_x[:2]))


@pytest.mark.parametrize(
    "invalid_entries",
    [[np.nan, np.nan], [np.nan, np.inf], [np.inf, np.inf], [np.inf, np.inf]],
)
def test_apply_constraint_by_basis_with_invalid(invalid_entries):
    """Test if the matrix retains its dtype after applying constraints."""
    basis = BSplineEval(5)
    x = basis.compute_features(
        np.random.randn(
            10,
        )
    )
    # add invalid
    x[:2, 2] = invalid_entries
    constrained_x, kept_cols = apply_identifiability_constraints(
        x, warn_if_float32=False
    )
    assert jnp.array_equal(kept_cols, jnp.arange(1, 5))
    assert constrained_x.shape[0] == x.shape[0]
    assert jnp.all(jnp.isnan(constrained_x[:2]))
