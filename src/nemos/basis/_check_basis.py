from typing import TYPE_CHECKING, Literal, Tuple

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..tree_utils import has_matching_axis_pytree
from ._composition_utils import infer_input_dimensionality

if TYPE_CHECKING:
    from ._basis import Basis
    from ._basis_mixin import BasisMixin


def _has_zero_samples(n_samples: Tuple[int, ...]) -> bool:
    return any([n <= 0 for n in n_samples])


def _check_zero_samples(
    n_samples, err_message="All sample provided must be non empty."
):
    if _has_zero_samples(n_samples):
        raise ValueError(err_message)


def _check_input_dimensionality(bas: "BasisMixin | Basis", xi: Tuple) -> None:
    """
    Check that the number of inputs provided by the user matches the number of inputs required.

    Parameters
    ----------
    xi[0], ..., xi[n] :
        The input samples, shape (number of samples, ).

    Raises
    ------
    ValueError
        If the number of inputs doesn't match what the Basis object requires.
    """
    n_input_dim = infer_input_dimensionality(bas)
    if len(xi) != n_input_dim:
        raise TypeError(
            f"This basis requires {n_input_dim} input(s) for evaluation, but {len(xi)} were provided."
        )


def _check_samples_consistency(*xi: NDArray) -> None:
    """
    Check that each input provided to the Basis object has the same number of time points.

    Parameters
    ----------
    xi[0], ..., xi[n] :
        The input samples, shape (number of samples, ).

    Raises
    ------
    ValueError
        If the time point number is inconsistent between inputs.
    """
    if not has_matching_axis_pytree(*xi, axis=0):
        raise ValueError(
            "Sample size mismatch. Input elements have inconsistent sample sizes."
        )


def _check_transform_input(
    bas: "BasisMixin | Basis",
    *xi: ArrayLike,
    conv_type: Literal["numpy", "jax", "none"] = "numpy",
) -> Tuple[NDArray]:
    # conversion type
    if conv_type == "numpy":
        at_least_1d = np.atleast_1d
    elif conv_type == "jax":
        at_least_1d = jnp.atleast_1d
    else:

        def at_least_1d(x):
            return x

    # check that the input is array-like (i.e., whether we can cast it to
    # numeric arrays)
    try:
        # make sure array is at least 1d (so that we succeed when only
        # passed a scalar)
        xi = tuple(at_least_1d(x).astype(float) for x in xi)
    # ValueError here surfaces the exception with e.g., `x=np.array["a", "b"])`
    except (TypeError, ValueError):
        raise TypeError("Input samples must be array-like of floats!")

    # check for non-empty samples
    _check_zero_samples(tuple(len(x) for x in xi))

    # checks on input and outputs
    _check_input_dimensionality(bas, xi)
    _check_samples_consistency(*xi)

    return xi
