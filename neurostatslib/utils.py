import jax
import jax.numpy as jnp
from functools import partial


# Broadcasted 1d convolution operations.
# [[n x t],[w]] -> [n x (t - w + 1)]
_corr1 = jax.vmap(partial(jnp.convolve, mode='valid'), (0, None), 0)
# [[n x t],[p x w]] -> [n x p x (t - w + 1)]
_corr2 = jax.vmap(_corr1, (None, 0), 0)


def convolve_1d_basis(basis_matrix, time_series):
    """
    Parameters
    ----------
    basis_matrix : array
    	Matrix holding 1d basis functions, 
    	shape == (B, W).
    time_series : array 
        Matrix holding multivariate time series,
        shape == (N, T).

    Returns
    -------
    convolved_series : array
        Result of convolution between all pairs of
        features and basis functions,
        shape == (N, B, T - W + 1).

    Notes
    -----
    For example, `time_series` could be a matrix of
    spike counts with `N` neurons and `T` timebins,
    and `basis_matrix` could be a matrix of `B`
    temporal basis functions with a window size of `W`.
    """
    return _corr2(
    	jnp.atleast_2d(basis_matrix),
    	jnp.atleast_2d(time_series)
    )
