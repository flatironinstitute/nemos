"""Module for calculating theoretical optimal defaults for SVRG and GLM configurations."""

import warnings
from functools import wraps
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from numpy.typing import NDArray

from .. import pytrees


def _convert_to_float(func):
    """
    Decorator to convert all inputs to float before passing them to the function.

    Ensures that calculations within the function are performed with floating-point precision.

    Parameters
    ----------
    func :
        The function to be wrapped by the decorator.

    Returns
    -------
    :
        Wrapped function with inputs converted to floats.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        args, kwargs = jax.tree_util.tree_map(float, (args, kwargs))
        return func(*args, **kwargs)

    return wrapper


def svrg_optimal_batch_and_stepsize(
    compute_smoothness_constants: Callable,
    *data: Any,
    batch_size: Optional[int] = None,
    stepsize: Optional[float] = None,
    strong_convexity: Optional[float] = None,
    n_power_iters: Optional[int] = 20,
    default_batch_size: int = 1,
    default_stepsize: float = 1e-3,
):
    """
    Calculate the optimal batch size and step size for SVRG optimization in GLMs.

    This function computes the optimal batch size and step size parameters for SVRG,
    based on the smoothness constants and strong convexity of the loss function.

    Parameters
    ----------
    compute_smoothness_constants :
        Function that computes the smoothness constants `l_smooth` and `l_smooth_max` for the problem.
        This is problem (loss function) specific.
    data :
        Input data, typically (X, y) for a GLM.
    batch_size :
        The batch size set by the user. If None, it will be calculated.
    stepsize :
        The step size set by the user. If None, it will be calculated.
    strong_convexity :
        The strong convexity constant. For L2-regularized losses, this should be the regularization strength.
    n_power_iters :
        Maximum number of iterations for the power method when finding the largest eigenvalue. If set to `None`,
        the lead eigenvalue is computed directly (no power method).
    default_batch_size :
        Default batch size to use if the optimal calculation fails.
    default_stepsize :
        Default step size to use if the optimal calculation fails.

    Returns
    -------
    dict
        Dictionary containing the optimal `batch_size` and `stepsize`.

    Raises
    ------
    ValueError
        If the data provided has inconsistent numbers of samples.

    Warnings
    --------
    UserWarning
        Warns the user if the calculation fails and defaults are used instead.

    Examples
    --------
    >>> import numpy as np
    >>> from nemos.solvers import svrg_optimal_batch_and_stepsize as compute_opt_params
    >>> from nemos.solvers import glm_softplus_poisson_l_max_and_l
    >>> np.random.seed(123)
    >>> X = np.random.normal(size=(500, 5))
    >>> y = np.random.poisson(np.exp(X.dot(np.ones(X.shape[1]))))
    >>> batch_size, stepsize = compute_opt_params(
    ...     glm_softplus_poisson_l_max_and_l, X, y, strong_convexity=0.08, batch_size=1
    ... )
    """
    # Get the number of samples, ensuring consistency across all inputs
    num_samples = {dd.shape[0] for dd in jax.tree_util.tree_leaves(data)}
    if len(num_samples) != 1:
        raise ValueError("Each array in data must have the same number of samples.")
    num_samples = num_samples.pop()

    # If both parameters are set by the user, return them directly
    if batch_size is not None and stepsize is not None:
        return {"batch_size": batch_size, "stepsize": stepsize}

    # Compute smoothness constants
    l_smooth_max, l_smooth = compute_smoothness_constants(
        *data, n_power_iters=n_power_iters, batch_size=batch_size
    )

    # Compute optimal batch size if not provided by the user
    if batch_size is None:
        batch_size = _calculate_optimal_batch_size_svrg(
            num_samples, l_smooth_max, l_smooth, strong_convexity=strong_convexity
        )

    # Fall back to defaults if batch size calculation fails
    if not jnp.isfinite(batch_size):
        batch_size = default_batch_size
        warnings.warn(
            "Could not determine batch size automatically. "
            f"Falling back on the default values of {default_batch_size}.",
            UserWarning,
        )

    # Compute optimal step size if not provided by the user
    if stepsize is None:
        stepsize = _calculate_stepsize_svrg(
            batch_size, num_samples, l_smooth_max, l_smooth
        )
        if stepsize < 0:
            stepsize = default_stepsize
            warnings.warn(
                "Could not determine step size automatically. "
                f"Falling back on the default value of {default_stepsize}.",
                UserWarning,
            )
    return {"batch_size": int(batch_size), "stepsize": stepsize}


def glm_softplus_poisson_l_max_and_l(
    *data: NDArray, n_power_iters: Optional[int] = 20, batch_size: Optional[int] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate smoothness constants for a Poisson GLM with a softplus inverse link function.

    Computes the smoothness constant (`l_smooth`) and the maximum smoothness constant
    (`l_smooth_max`) for SVRG, given the data and the GLM structure.

    Parameters
    ----------
    data :
        Input data, typically (X, y).
    n_power_iters :
        Number of power iterations to use when finding the largest eigenvalue. If None,
        the eigenvalue is calculated directly.
    batch_size:
        The batch size set by the user. If None, it will be n_samples.

    Returns
    -------
    :
        Maximum smoothness constant (`l_smooth_max`) and smoothness constant (`l_smooth`).
    """
    X, y = data

    if batch_size is None:
        batch_size = y.shape[0]

    # takes care of population glm (see bound found on overleaf)
    y = jnp.max(y, axis=tuple(range(1, y.ndim)))

    # concatenate all data (if X is FeaturePytree)
    if isinstance(X, (pytrees.FeaturePytree, dict)):
        X = jnp.hstack(jax.tree_util.tree_leaves(X))

    l_smooth = _glm_softplus_poisson_l_smooth(
        X, y, n_power_iters=n_power_iters, batch_size=batch_size
    )
    l_smooth_max = _glm_softplus_poisson_l_smooth_max(X, y, batch_size=batch_size)
    return l_smooth_max, l_smooth


def _glm_softplus_poisson_l_smooth_multiply(
    X: NDArray, y: NDArray, v: NDArray, batch_size: int
):
    """
    Multiply vector `v` with the matrix X.T @ D @ X without forming it explicitly.

    This method estimates the multiplication by calculating the Hessian of the loss.
    It is efficient for situations where X can fit in memory.
    If batch_size is provided, the computation will be done by slicing the array.

    Parameters
    ----------
    X :
        Input data matrix (N x d).
    y :
        Output data vector (N,).
    v :
        Vector to be multiplied (d,).

    Returns
    -------
    :
        Result of the multiplication (X.T @ D @ X) @ v.
    """
    N, K = X.shape
    out = jnp.zeros((K,))
    for i in range(0, N, batch_size):
        xb, yb = X[i : i + batch_size], y[i : i + batch_size]
        out = out + xb.T.dot((0.17 * yb + 0.25) * xb.dot(v))
    out = out / N
    return out


def _glm_softplus_poisson_l_smooth_with_power_iteration(
    X: NDArray, y: NDArray, n_power_iters: int = 20, batch_size: Optional[int] = None
):
    """
    Compute the largest eigenvalue of X.T @ D @ X using the power method.

    Instead of calculating the full matrix and its eigenvalue directly, this function
    uses the power iteration method, which is more memory efficient and scales better
    with large datasets.

    Parameters
    ----------
    X :
        Input data matrix (N x d).
    y :
        Output data vector (N,).
    n_power_iters :
        Maximum number of power iterations to use.
    batch_size :
        The batch size, if user provides one.

    Returns
    -------
    :
        The largest eigenvalue of X.T @ D @ X.
    """

    if batch_size is None:
        batch_size = X.shape[0]

    _, d = X.shape

    # Initialize a random d-dimensional vector for power iteration
    v = jnp.ones((d,))

    # Run power iteration to approximate the largest eigenvalue
    for _ in range(n_power_iters):
        v_prev = v.copy()
        v = _glm_softplus_poisson_l_smooth_multiply(X, y, v, batch_size)
        v /= v.max()

        # Check for convergence
        if jnp.allclose(v_prev, v):
            break

    # Final eigenvalue calculation
    v /= jnp.linalg.norm(v)
    return _glm_softplus_poisson_l_smooth_multiply(X, y, v, batch_size).dot(v)


def _glm_softplus_poisson_l_smooth(
    X: NDArray, y: NDArray, batch_size: int, n_power_iters: Optional[int] = 20
) -> jnp.ndarray:
    """
    Calculate the smoothness constant `L` for a Poisson GLM with softplus inverse link.

    Depending on whether `n_power_iters` is provided, this function either computes
    the largest eigenvalue directly or uses the power method.

    Parameters
    ----------
    X :
        Input data matrix (N x d).
    y :
        Output data vector (N,).
    batch_size :
        The batch size, if user provides one.
    n_power_iters :
        Number of power iterations to use when finding the largest eigenvalue. If None,
        the eigenvalue is calculated directly.

    Returns
    -------
    :
        Smoothness constant `L`.

    Raises
    ------
    TypeError:
        If `n_power_iters` is not an integer or None.
    RuntimeError:
        If the eigienvalue computation does not converge.
    ValueError:
        If a negative integer is provided.
    """
    if n_power_iters and not isinstance(n_power_iters, int):
        raise TypeError("`n_power_iters` must be an integer or None.")

    if n_power_iters is None:
        try:
            warnings.warn(
                "Direct computation of the eigenvalues requires a substantial amount of resources. "
                "Please, consider using the power method by setting the `n_power_iters` parameter "
                "(default behavior).",
                UserWarning,
            )
            # Calculate the Hessian directly and find the largest eigenvalue
            XDX = X.T.dot((0.17 * y.reshape(y.shape[0], 1) + 0.25) * X) / y.shape[0]
            return jnp.sort(jnp.linalg.eigvalsh(XDX))[-1]
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to calculate the largest eigenvalue: {e}. "
                "Consider using the power method by setting the `n_power_iters` parameter."
            )
    else:
        if n_power_iters <= 0:
            raise ValueError("`n_power_iters` must be positive.")
        # Use power iteration to find the largest eigenvalue
        return _glm_softplus_poisson_l_smooth_with_power_iteration(
            X, y, n_power_iters, batch_size=batch_size
        )


def _glm_softplus_poisson_l_smooth_max(
    X: NDArray, y: NDArray, batch_size: int
) -> NDArray:
    """
    Calculate the maximum smoothness constant `L_max` for individual observations.

    This function estimates the maximum smoothness constant among the individual
    components of the loss function.

    Parameters
    ----------
    X :
        Input data matrix (N x d).
    y :
        Output data vector (N,).
    batch_size:
        The batch size.

    Returns
    -------
    l_max :
        Maximum smoothness constant `L_max`.
    """
    N, _ = X.shape

    l_max = jnp.array([0])
    for nb in range(0, N, batch_size):
        xb, yb = X[nb : nb + batch_size], y[nb : nb + batch_size]
        l_max = jnp.maximum(
            l_max, jnp.max(jnp.linalg.norm(xb, axis=1) ** 2 * (0.17 * yb + 0.25))
        )

    return l_max[0]


@_convert_to_float
def _calculate_stepsize_svrg(
    batch_size: int, num_samples: int, l_smooth_max: float, l_smooth: float
):
    """
    Calculate optimal step size for SVRG according to [1].

    Parameters
    ----------
    batch_size :
        Mini-batch size.
    num_samples :
        Overall number of data points.
    l_smooth_max :
        Maximum smoothness constant among f_{i}.
    l_smooth :
        Smoothness constant.

    Returns
    -------
    :
        Optimal step size for the optimization.

    References
    ----------
    [1] Sebbouh, Othmane, et al. "Towards closing the gap between the theory and practice of SVRG."
    Advances in neural information processing systems 32 (2019).
    """
    numerator = 0.5 * batch_size * (num_samples - 1)
    denominator = (
        3 * (num_samples - batch_size) * l_smooth_max
        + num_samples * (batch_size - 1) * l_smooth
    )
    return numerator / denominator


def _calculate_optimal_batch_size_svrg(
    num_samples: int,
    l_smooth_max: float,
    l_smooth: float,
    strong_convexity: Optional[float] = None,
) -> int:
    r"""
    Calculate the optimal batch size for SVRG based on theoretical guidelines.

    The batch size is computed according to the smoothness constants, strong convexity,
    and number of samples, following the recommendations in Table 1 of [1].

    Parameters
    ----------
    num_samples:
        The number of samples.
    l_smooth_max:
        The $L_{\text{max}}$ smoothness constant.
    l_smooth:
        The $L$ smoothness constant.
    strong_convexity:
        The strong convexity constant.

    Returns
    -------
    batch_size:
        The optimal mini-batch size for SVRG.

    References
    ----------
    [1] Sebbouh, Othmane, et al. "Towards closing the gap between the theory and practice of SVRG."
    Advances in neural information processing systems 32 (2019).
    """
    if strong_convexity is None:
        # Assume that num_sample is large enough for mini-batching.
        # This is usually the case for neuroscience where num_sample
        # is typically very large.
        # If this assumption is not matched, convergence may be slow.
        batch_size = 1
    else:
        # Compute optimal batch size according to Table 1.
        if num_samples >= 3 * l_smooth_max / strong_convexity:
            batch_size = 1
        elif num_samples > l_smooth / strong_convexity:
            b_tilde = _calculate_b_tilde(
                num_samples, l_smooth_max, l_smooth, strong_convexity
            )
            if l_smooth_max < num_samples * l_smooth / 3:
                b_hat = _calculate_b_hat(num_samples, l_smooth_max, l_smooth)
                batch_size = int(jnp.floor(jnp.minimum(b_hat, b_tilde)))
            else:
                batch_size = int(jnp.floor(b_tilde))
        else:
            if l_smooth_max < num_samples * l_smooth / 3:
                batch_size = int(
                    jnp.floor(_calculate_b_hat(num_samples, l_smooth_max, l_smooth))
                )
            else:
                batch_size = int(num_samples)  # reset this to int
    return batch_size


@_convert_to_float
def _calculate_b_hat(num_samples: int, l_smooth_max: float, l_smooth: float):
    r"""
    Calculate the optimal `b_hat` batch size parameter for SVRG.

    This is a helper function to compute the theoretical batch size $\hat{b}$, as detailed
    in "Table 1" of [1].

    Parameters
    ----------
    num_samples :
        Total number of data points.
    l_smooth_max :
        Maximum smoothness constant $L_{\text{max}}$.
    l_smooth :
        Smoothness constant $L$.

    Returns
    -------
    float
        Optimal batch size parameter `b_hat`.

    References
    ----------
    [1] Sebbouh, Othmane, et al. "Towards closing the gap between the theory and practice of SVRG."
    Advances in neural information processing systems 32 (2019).
    """
    numerator = num_samples / 2 * (3 * l_smooth_max - l_smooth)
    denominator = num_samples * l_smooth - 3 * l_smooth_max
    return jnp.sqrt(numerator / denominator)


@_convert_to_float
def _calculate_b_tilde(num_samples, l_smooth_max, l_smooth, strong_convexity):
    r"""
    Calculate the optimal $\tilde{b}$ batch size parameter for SVRG.

    This is a helper function to compute the theoretical batch size  $\tilde{b}$, as detailed
    in "Table 1" of [1].

    Parameters
    ----------
    num_samples :
        Total number of data points.
    l_smooth_max :
        Maximum smoothness constant `L_max`.
    l_smooth :
        Smoothness constant `L`.
    strong_convexity :
        Strong convexity constant.

    Returns
    -------
    :
        Optimal batch size parameter `b_tilde`.

    References
    ----------
    [1] Sebbouh, Othmane, et al. "Towards closing the gap between the theory and practice of SVRG."
    Advances in neural information processing systems 32 (2019).
    """
    numerator = (3 * l_smooth_max - l_smooth) * num_samples
    denominator = (
        num_samples * (num_samples - 1) * strong_convexity
        - num_samples * l_smooth
        + 3 * l_smooth_max
    )
    return numerator / denominator
