"""Utility functions for applying identifiability constraints to rank deficient feature matrices."""

from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from .basis import Basis
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


@partial(jax.jit, static_argnums=(3,))
def _find_drop_column(feature_matrix, idx, rank, preprocessing_func=add_constant):
    """Check if the column idx is linearly dependent from the other columns."""
    rank_after_drop_column = jnp.linalg.matrix_rank(
        preprocessing_func(feature_matrix.at[:, idx].set(0.0))
    )
    return rank == rank_after_drop_column


@partial(jax.jit, static_argnums=(1,))
def _search_and_drop_columns(state, find_drop_columns):
    """Search & drop columns."""
    _, drop_cols, feature_matrix, rank, _ = state
    new_drop_cols = find_drop_columns(
        feature_matrix, jnp.arange(feature_matrix.shape[1]), rank
    )
    # get the first newly found linearly dependent column
    idx = jnp.arange(drop_cols.shape[0])[jnp.argmax(new_drop_cols & (~drop_cols))]
    # drop
    feature_matrix = feature_matrix.at[:, idx].set(0.0)
    # stopping condition
    found_cols = drop_cols.sum() < new_drop_cols.sum()
    # update
    drop_cols = drop_cols.at[idx].set(True)
    return found_cols, drop_cols, feature_matrix, rank, idx


def _apply_identifiability_constraints(
    feature_matrix: NDArray,
    preprocessing_func: Callable = add_constant,
    warn_if_float32=True,
):
    """
    Apply identifiability constraints to a design matrix `feature_matrix`.

    Private function that does the actual computation on a single feature_matrix.
    """
    if warn_if_float32:
        _warn_if_not_float64(feature_matrix, _WARN_FLOAT32_MESSAGE)

    shape_sample_axis = feature_matrix.shape[0]
    is_valid = get_valid_multitree(feature_matrix)

    # feature_matrix = tree_slice(feature_matrix, is_valid)
    # vectorize the search and drop column (efficient on GPU)
    vec_find_drop_col = jax.vmap(
        partial(_find_drop_column, preprocessing_func=preprocessing_func),
        in_axes=(None, 0, None),
        out_axes=0,
    )
    search_and_drop = partial(
        _search_and_drop_columns, find_drop_columns=vec_find_drop_col
    )

    # compute initial rank if needed
    feature_matrix_with_intercept = preprocessing_func(
        tree_slice(feature_matrix, is_valid)
    )
    rank = jnp.linalg.matrix_rank(feature_matrix_with_intercept)

    # full rank, no extra computation needed
    if rank == feature_matrix_with_intercept.shape[1]:
        return feature_matrix, jnp.zeros((feature_matrix.shape[1]), dtype=bool)

    # initialize the drop col vector to True, and the output matrix to feature_matrix
    is_column_drop_found = jnp.array(True)  # for consistency with the output of jnp.any
    drop_cols = jnp.zeros((feature_matrix.shape[1]), dtype=bool)
    state = (
        is_column_drop_found,
        drop_cols,
        tree_slice(feature_matrix, is_valid),
        rank,
        jnp.array(0),
    )

    def cond_function(state):
        return state[0]

    # run the while loop
    is_column_drop_found, drop_cols, feature_matrix, _, _ = _while_loop_scan(
        cond_function,
        body_fun=search_and_drop,
        init_val=state,
        max_iter=feature_matrix_with_intercept.shape[1] - rank,
    )

    # return the output matrix and the dropped indices
    feature_matrix = (
        jnp.full(
            (shape_sample_axis, feature_matrix.shape[1] - drop_cols.sum()),
            jnp.nan,
            dtype=feature_matrix.dtype,
        )
        .at[is_valid]
        .set(feature_matrix[:, ~drop_cols])
    )
    return feature_matrix, drop_cols


def _while_loop_scan(cond_fun, body_fun, init_val, max_iter):
    """Scan-based implementation (jit ok, reverse-mode autodiff ok)."""

    def _iter(val):
        next_val = body_fun(val)
        next_cond = cond_fun(next_val)
        return next_val, next_cond

    def _fun(tup, it):
        val, cond = tup
        # When cond is met, we start doing no-ops.
        return jax.lax.cond(cond, _iter, lambda x: (x, False), val), it

    init = (init_val, cond_fun(init_val))
    return jax.lax.scan(_fun, init, None, length=max_iter)[0][0]


@support_pynapple(conv_type="jax")
def apply_identifiability_constraints(
    feature_matrix: NDArray, add_intercept: bool = True, warn_if_float32: bool = True
):
    """
    Apply identifiability constraints to a design matrix `X`.

    Removes columns from `X` until it is full rank to ensure the uniqueness
    of the GLM (Generalized Linear Model) maximum-likelihood solution. This is particularly
    crucial for models using bases like BSplines and CyclicBspline, which, due to their
    construction, sum to 1 and can cause rank deficiency when combined with an intercept.

    For GLMs, this rank deficiency means that different sets of coefficients might yield
    identical predicted rates and log-likelihood, complicating parameter learning, especially
    in the absence of regularization.

    For very large feature matrices generated by a sum of low-dimensional basis components, consider
    `apply_identifiability_constraints_by_basis_component`.

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
    >>> from nemos.basis import BSplineBasis
    >>> from nemos.glm import GLM

    >>> # define a feature matrix
    >>> bas = BSplineBasis(5) + BSplineBasis(6)
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
        feature_matrix,
        preprocessing_func=preproc_design,
        warn_if_float32=warn_if_float32,
    )
    kept_columns = np.arange(feature_matrix.shape[1])[~discarded_columns]
    return constrained_x, kept_columns


@support_pynapple(conv_type="jax")
def apply_identifiability_constraints_by_basis_component(
    basis: Basis,
    feature_matrix: NDArray,
    add_intercept: bool = True,
) -> Tuple[NDArray, NDArray]:
    """Apply identifiability constraint to a design matrix to each component of an additive basis.

    Parameters
    ----------
    basis:
        The basis that computed X;
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
    >>> from nemos.identifiability_constraints import apply_identifiability_constraints_by_basis_component
    >>> from nemos.basis import BSplineBasis
    >>> from nemos.glm import GLM

    >>> # define a feature matrix
    >>> bas = BSplineBasis(5) + BSplineBasis(6)
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

    # list leaves and unwrap over input dimension. Additive components have shapes:
    # (n_samples, n_inputs, n_basis_funcs)
    split_by_input_x = [
        x[:, k] for x in jax.tree_util.tree_leaves(splits_x) for k in range(x.shape[1])
    ]

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
