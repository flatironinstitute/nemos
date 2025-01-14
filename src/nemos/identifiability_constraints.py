"""Utility functions for applying identifiability constraints to rank deficient feature matrices."""

from __future__ import annotations

from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike as JaxArray
from numpy.typing import NDArray

from .basis._basis import Basis
from .tree_utils import get_valid_multitree, tree_slice
from .type_casting import support_pynapple
from .validation import _warn_if_not_float64

_WARN_FLOAT32_MESSAGE = (
    "The feature matrix is not of dtype `float64`. Consider converting it to `float64` "
    "for increased numerical precision when computing the matrix rank. You can enable "
    "float64 precision globally by adding:\n\n    jax.config.update('jax_enable_x64', True)\n"
)


def add_constant(x):
    """Add intercept term."""
    return jnp.hstack((jnp.ones((x.shape[0], 1), dtype=x.dtype), x))


@partial(jax.jit, static_argnums=(2,))
def _drop_and_compute_rank(feature_matrix, idx, preprocessing_func=add_constant):
    """Drop column idx and compute rank."""
    feature_dropped = feature_matrix.at[:, idx].set(0.0)
    rank_after_drop_column = jnp.linalg.matrix_rank(preprocessing_func(feature_dropped))
    return feature_dropped, rank_after_drop_column


@partial(jax.jit, static_argnums=(1, 2, 3))
def _find_drop_column(
    feature_matrix: JaxArray,
    rank: int,
    max_drop: int,
    preprocessing_func: Callable = add_constant,
) -> JaxArray:
    """
    Find a minimal subset linearly dependent columns that can be dropped.

    This function loops over the columns of a matrix and checks if each column is linearly dependent from the others.
    If the i-th column is linearly dependent, then drop_cols[i] is set to True, and feature_matrix[:, i] is set to 0.
    The loop is stopped when max_drop linearly dependent columns are found.

    Parameters
    ----------
    feature_matrix:
        The rank deficient feature matrix.
    rank:
        The rank of the matrix.
    max_drop:
        Number of columns to be dropped.
    preprocessing_func:
        Additional processing of the feature matrix. By default, add an intercept term. Other processing could
        entail mean-centering the columns or similar.

    Returns
    -------
    drop_cols:
        A boolean vector, True if the column should be dropped, False otherwise.

    """

    def drop_col_and_update(features, dropped_features, drop_cols, iter_num):
        """Drop feature and update drop column boolean."""
        return dropped_features, drop_cols.at[iter_num].set(True)

    def do_not_drop(features, dropped_features, drop_cols, iter_num):
        """Do not drop."""
        return features, drop_cols

    def check_column(iter_num, state):
        """Drop a column if rank is not affected."""
        matrix, _, original_rank, drop_cols, mx_drop = state
        # drop the column (by set to zero) and compute rank
        col_dropped_matrix, mat_rank = _drop_and_compute_rank(
            matrix, iter_num, preprocessing_func
        )
        # apply the change if rank stays constant, do nothing otherwise.
        matrix, drop_cols = jax.lax.cond(
            mat_rank == original_rank,  # condition
            drop_col_and_update,  # true function
            do_not_drop,  # false function
            matrix,  # parameters
            col_dropped_matrix,
            drop_cols,
            iter_num,
        )
        return matrix, mat_rank, original_rank, drop_cols, mx_drop

    def body_func(iter_num, state):
        drop_cols, max_drop = state[-2:]
        return jax.lax.cond(
            drop_cols.sum() < max_drop, check_column, lambda it, x: x, iter_num, state
        )

    init_state = (
        feature_matrix,
        jnp.array(0),
        jnp.array(rank),
        jnp.zeros(feature_matrix.shape[1], dtype=bool),
        max_drop,
    )
    final_state = jax.lax.fori_loop(0, feature_matrix.shape[1], body_func, init_state)

    return final_state[3]


def _add_invalid_entries(feature_matrix, shape_first_axis, is_valid):
    """Add invalid entries to match original shape."""
    feature_matrix = (
        jnp.full(
            (shape_first_axis, *feature_matrix.shape[1:]),
            jnp.nan,
            dtype=feature_matrix.dtype,
        )
        .at[is_valid]
        .set(feature_matrix)
    )
    return feature_matrix


def _apply_identifiability_constraints(
    feature_matrix: JaxArray,
    preprocessing_func: Callable = add_constant,
    warn_if_float32: bool = True,
) -> Tuple[JaxArray, JaxArray]:
    """
    Apply identifiability constraints to a design matrix `feature_matrix`.

    Private function that does the actual computation on a single feature_matrix.
    """
    if warn_if_float32:
        _warn_if_not_float64(feature_matrix, _WARN_FLOAT32_MESSAGE)

    shape_sample_axis = feature_matrix.shape[0]
    is_valid = get_valid_multitree(feature_matrix)

    # compute initial rank if needed
    feature_matrix = tree_slice(feature_matrix, is_valid)
    feature_matrix_with_intercept = preprocessing_func(feature_matrix)
    rank = jnp.linalg.matrix_rank(feature_matrix_with_intercept)

    # full rank, no extra computation needed
    if rank == feature_matrix_with_intercept.shape[1]:
        feature_matrix = _add_invalid_entries(
            feature_matrix, shape_sample_axis, is_valid
        )
        return feature_matrix, jnp.zeros((feature_matrix.shape[1]), dtype=bool)

    max_drop = feature_matrix_with_intercept.shape[1] - rank

    # run the search
    drop_cols = _find_drop_column(
        feature_matrix,
        rank=int(rank),
        max_drop=int(max_drop),
        preprocessing_func=preprocessing_func,
    )

    # return the output matrix and the dropped indices
    feature_matrix = _add_invalid_entries(
        feature_matrix[:, ~drop_cols],
        shape_sample_axis,
        is_valid,
    )
    return feature_matrix, drop_cols


@support_pynapple(conv_type="jax")
def apply_identifiability_constraints(
    feature_matrix: NDArray | JaxArray,
    add_intercept: bool = True,
    warn_if_float32: bool = True,
) -> Tuple[NDArray, NDArray[int]]:
    """
    Apply identifiability constraints to a design matrix ``X``.

    Removes columns from ``X`` until it is full rank to ensure the uniqueness
    of the GLM (Generalized Linear Model) maximum-likelihood solution. This is particularly
    crucial for models using bases like BSplines and CyclicBspline, which, due to their
    construction, sum to 1 and can cause rank deficiency when combined with an intercept.

    For GLMs, this rank deficiency means that different sets of coefficients might yield
    identical predicted rates and log-likelihoods, complicating parameter learning, especially
    in the absence of regularization.

    For very large feature matrices generated by a sum of low-dimensional basis components, consider
    ``apply_identifiability_constraints_by_basis_component``.

    Parameters
    ----------
    feature_matrix:
        The design matrix before applying the identifiability constraints.
    add_intercept:
        Set to True if your model will add an intercept term, False otherwise.
    warn_if_float32:
        Raise a warning if feature matrix dtype is float32.

    Returns
    -------
    constrained_x:
        The adjusted design matrix with redundant columns dropped and columns mean-centered.
    kept_columns:
        The columns that have been kept.

    Examples
    --------
    >>> import numpy as np
    >>> from nemos.identifiability_constraints import apply_identifiability_constraints
    >>> from nemos.basis import BSplineEval
    >>> from nemos.glm import GLM
    >>> import jax
    >>> jax.config.update('jax_enable_x64', True)
    >>> # define a feature matrix
    >>> bas = BSplineEval(5) + BSplineEval(6)
    >>> feature_matrix = bas.compute_features(np.random.randn(100), np.random.randn(100))
    >>> # apply constraints
    >>> constrained_x, kept_columns = apply_identifiability_constraints(feature_matrix)
    >>> constrained_x.shape
    (100, 9)
    >>> kept_columns
    array([ 1,  2,  3,  4,  6,  7,  8,  9, 10])

    Notes
    -----
    Compilation is triggered at every loop. This can be slower than pure python for low number
    of samples and low dimension for the feature matrix.
    Usually, the design matrices we work with have a large number of samples.
    Running the code on GPU will reduce the computation time significantly.
    """
    if add_intercept:
        preproc_design = add_constant
    else:

        def preproc_design(x):
            return x

    # return the output matrix and the dropped indices
    constrained_x, discarded_columns = _apply_identifiability_constraints(
        jnp.asarray(feature_matrix),
        preprocessing_func=preproc_design,
        warn_if_float32=warn_if_float32,
    )
    kept_columns = np.arange(feature_matrix.shape[1])[~discarded_columns]
    return np.asarray(constrained_x), kept_columns


@support_pynapple(conv_type="jax")
def apply_identifiability_constraints_by_basis_component(
    basis: Basis,
    feature_matrix: NDArray,
    add_intercept: bool = True,
) -> Tuple[NDArray, NDArray]:
    """Apply identifiability constraint to a design matrix for each component of an additive basis.

    Parameters
    ----------
    basis:
        The basis that computed ``feature_matrix``.
    feature_matrix:
        The feature matrix before applying the identifiability constraints.
    add_intercept:
        Set to True if your model will add an intercept term, False otherwise.

    Returns
    -------
    constrained_x:
        The adjusted feature matrix after applying the identifiability constraints as numpy array.
    kept_columns:
        Indices of the columns that are kept. This should be used for applying the same transformation
        to a feature matrix generated from different a set of inputs (as for a test set).

    Examples
    --------
    >>> import numpy as np
    >>> import jax
    >>> from nemos.identifiability_constraints import apply_identifiability_constraints_by_basis_component
    >>> from nemos.basis import BSplineEval
    >>> from nemos.glm import GLM
    >>> jax.config.update('jax_enable_x64', True)
    >>> # define a feature matrix
    >>> bas = BSplineEval(5) + BSplineEval(6)
    >>> feature_matrix = bas.compute_features(np.random.randn(100), np.random.randn(100))
    >>> # apply constraints
    >>> constrained_x, kept_columns = apply_identifiability_constraints_by_basis_component(bas, feature_matrix)
    >>> constrained_x.shape
    (100, 9)
    >>> # generate a test set, shape (20, 11)
    >>> test_x = bas.compute_features(np.random.randn(20), np.random.randn(20))
    >>> test_x.shape
    (20, 11)
    >>> # apply constraint to test set
    >>> test_x = test_x[:, kept_columns]
    >>> test_x.shape
    (20, 9)
    >>> # fit on train and predict on test set
    >>> rate = GLM().fit(constrained_x, np.random.poisson(size=100)).predict(test_x)

    """
    # gets a dictionary with feature specific feature matrices
    # stored in tensors of shape (n_samples, n_inputs, n_features)
    # n_inputs can be larger than one if basis is used to perform
    # convolutions on multiple signals (as for counts TsdFrames)
    splits_x = basis.split_by_feature(feature_matrix)

    # list the arrays
    split_x = jax.tree_util.tree_leaves(splits_x)
    # add dim if needed (the dim is at least 2, (n_samples, n_basis)
    split_x = [x if x.ndim > 2 else x[:, None] for x in split_x]
    # flatten over inputs
    split_x = [x.reshape(x.shape[0], -1, x.shape[-1]) for x in split_x]
    # list leaves and unwrap over input dimension. Additive components have shapes:
    # (n_samples, n_inputs, n_basis_funcs)
    split_by_input_x = [x[:, k] for x in split_x for k in range(x.shape[1])]

    apply_identifiability = partial(
        apply_identifiability_constraints,
        add_intercept=add_intercept,
        warn_if_float32=False,
    )

    _warn_if_not_float64(split_by_input_x, _WARN_FLOAT32_MESSAGE)

    constrained_x_and_columns = jax.tree_util.tree_map(
        apply_identifiability, split_by_input_x
    )

    # unpack the outputs into array and dropped colum indices
    def is_leaf(x):
        return isinstance(x, tuple)

    constrained_x = tree_slice(constrained_x_and_columns, idx=0, is_leaf=is_leaf)
    kept_columns = tree_slice(constrained_x_and_columns, idx=1, is_leaf=is_leaf)

    # stack the arrays back into a feature matrix
    constrained_x = np.hstack(constrained_x)

    # indices are referenced to the sub-matrices, get the absolute index in the feature matrix
    # calculate the shifts for each component
    shifts = list(
        np.cumsum(
            [0]
            + [
                sub_x.shape[1]
                for sub_x in jax.tree_util.tree_leaves(split_by_input_x)[:-1]
            ]
        )
    )
    kept_columns = jax.tree_util.tree_map(lambda x, y: x + y, kept_columns, shifts)
    kept_columns = np.hstack(kept_columns)
    return constrained_x, kept_columns
