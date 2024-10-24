import warnings
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from .basis import Basis
from .tree_utils import tree_slice


def add_constant(x):
    """Add intercept term."""
    return jnp.hstack((jnp.ones((x.shape[0], 1)), x))


# delete jit compatible
jit_delete = jax.jit(jnp.delete, static_argnames=["assume_unique_indices", "axis"])


@partial(jax.jit, static_argnums=(3,))
def find_drop_column(feature_matrix, idx, rank, preprocessing_func=add_constant):
    """Check if the column idx is linearly dependent from the other columns."""
    rank_after_drop_column = jnp.linalg.matrix_rank(
        preprocessing_func(
            jit_delete(feature_matrix, idx, axis=1, assume_unique_indices=True)
        )
    )
    return rank == rank_after_drop_column


@partial(jax.jit, static_argnums=(1,))
def search_and_drop_columns(state, find_drop_columns):
    """Search & drop columns."""
    drop_cols, feature_matrix, rank = state
    drop_cols = find_drop_columns(
        feature_matrix, jnp.arange(feature_matrix.shape[1]), rank
    )
    idx = jnp.argmax(drop_cols)
    # always delete one, otherwise cannot compile
    # the output is discarded and the while loop interrupted if no columns can be dropped
    feature_matrix = jit_delete(feature_matrix, idx, axis=1, assume_unique_indices=True)
    return jnp.any(drop_cols), feature_matrix, rank, idx


def _apply_identifiability_constraints(
    feature_matrix: NDArray, preprocessing_func: Callable = add_constant
):
    """
    Apply identifiability constraints to a design matrix `feature_matrix`.

    Private function that does the actual computation on a single feature_matrix.
    """
    if jnp.issubdtype(feature_matrix.dtype, jnp.float32):
        warnings.warn(
            category=UserWarning,
            message="The feature matrix is of dtype `float32`. Consider converting it to `float64` "
            "for increased numerical precision when computing the matrix rank, as lower "
            "precision may lead to inaccurate results. "
            "You can enable float64 precision globally in JAX by adding the following line at "
            "the beginning of your script:\n\n"
            "    jax.config.update('jax_enable_x64', True)\n",
        )

    # vectorize the search and drop column (efficient on GPU)
    vec_find_drop_col = jax.vmap(
        partial(find_drop_column, preprocessing_func=preprocessing_func),
        in_axes=(None, 0, None),
        out_axes=0,
    )
    search_and_drop = partial(
        search_and_drop_columns, find_drop_columns=vec_find_drop_col
    )

    # compute initial rank if needed
    feature_matrix_with_intercept = preprocessing_func(feature_matrix)
    rank = jnp.linalg.matrix_rank(feature_matrix_with_intercept)

    # full rank, no extra computation needed
    if rank == feature_matrix_with_intercept.shape[1]:
        return feature_matrix, jnp.arange(feature_matrix.shape[1])

    # initialize the drop col vector to True, and the output matrix to feature_matrix
    is_column_drop_found = jnp.array(True)  # for consistency with the output of jnp.any
    state = (is_column_drop_found, feature_matrix, rank)
    drop_column_index = []
    col_indexes = jnp.arange(feature_matrix.shape[1])
    while is_column_drop_found:
        # update the drop column vector and the output matrix
        is_column_drop_found, fm, rank, idx = search_and_drop(state)
        if is_column_drop_found:
            # update state if column to drop was found
            state = is_column_drop_found, fm, rank
            # store the column index
            drop_column_index.append(col_indexes[idx])
            # drop column index from available
            col_indexes = jnp.delete(col_indexes, idx, axis=0)

    # return the output matrix and the dropped indices
    return state[1], jnp.hstack(drop_column_index)


def apply_identifiability_constraints(
    feature_matrix: NDArray, add_intercept: bool = True
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

    Parameters
    ----------
    feature_matrix:
        The design matrix before applying the identifiability constraints.
    add_intercept:
        Set to True if your model will add an intercept term, False otherwise.

    Returns
    -------
    :
        The adjusted design matrix with redundant columns dropped and columns mean-centered.

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
    return _apply_identifiability_constraints(
        feature_matrix, preprocessing_func=preproc_design
    )


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
    constrained_x = [
        x[:, k] for x in jax.tree_util.tree_leaves(splits_x) for k in range(x.shape[1])
    ]

    apply_identifiability = partial(
        apply_identifiability_constraints, add_intercept=add_intercept
    )
    constrained_x_and_indices = jax.tree_util.tree_map(
        apply_identifiability, constrained_x
    )

    # unpack the outputs into array and dropped colum indices
    def is_leaf(x):
        return isinstance(x, tuple)

    constrained_x = tree_slice(constrained_x_and_indices, idx=0, is_leaf=is_leaf)
    dropped_indices = tree_slice(constrained_x_and_indices, idx=1, is_leaf=is_leaf)

    # stack the arrays back into a feature matrix
    constrained_x = np.hstack(constrained_x)

    # indices are referenced to the sub-matrices, get the absolute index in the feature matrix
    dropped_indices = np.hstack(dropped_indices)
    # calculate the shifts for each component
    shifts = np.hstack(
        (
            np.cumsum(
                [0]
                + [sub_x.shape[2] for sub_x in jax.tree_util.tree_leaves(splits_x)[:-1]]
            )
        )
    )
    dropped_indices += shifts
    # get kept columns
    kept_columns = np.delete(
        np.arange(feature_matrix.shape[1]), dropped_indices, axis=0
    )
    return constrained_x, kept_columns
