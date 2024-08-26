"""Calculate theoretical optimal defaults if available."""

from functools import wraps
import warnings
from typing import Optional, Callable, Any, Tuple
import jax
import jax.numpy as jnp


def _convert_to_float(func):
    """Convert to float."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        args, kwargs = jax.tree_util.tree_map(float, (args, kwargs))
        return func(*args, **kwargs)

    return wrapper


# not using the previous one to avoid calculating L and L_max twice
def svrg_optimal_batch_and_stepsize(
    compute_smoothness_constants: Callable,
    *data: Any,
    batch_size: Optional[int] = None,
    stepsize: Optional[float] = None,
    strong_convexity: Optional[float] = None,
    n_power_iters: Optional[int] = None,
    default_batch_size: int = 1,
    default_stepsize: float = 1e-3,
):
    """
    Calculate the optimal batch size and step size to use for SVRG with a GLM
    that uses Poisson observations and softplus inverse link function.

    Parameters
    ----------
    compute_smoothness_constants:
        Function that computes l_smooth and l_smooth_max for the problem.
    data :
        The input data. For a GLM, X and y.
    n_power_iters: int, optional, default None
        If None, build the XDX matrix (which has a shape of n_features x n_features)
        and find its eigenvalues directly.
        If an integer, it is the max number of iterations to run the power
        iteration for when finding the largest eigenvalue.
    batch_size:
        The batch_size set by the user.
    stepsize:
        The stepsize set by the user.
    strong_convexity:
        The strong convexity constant. For penalized losses with an L2 component (Ridge, Elastic Net, etc.)
        the convexity constant should be equal to the penalization strength.
    default_batch_size : int
        Batch size to fall back on if the calculation fails.
    default_stepsize: float
        Step size to fall back on if the calculation fails.

    Returns
    -------
    batch_size : int
        Optimal batch size to use.
    stepsize : scalar jax array
        Optimal stepsize to use.

    Examples
    --------
    >>> import numpy as np
    >>> from nemos.solvers import svrg_optimal_batch_and_stepsize as compute_opt_params
    >>> from nemos.solvers import softplus_poisson_l_max_and_l
    >>> np.random.seed(123)
    >>> X = np.random.normal(size=(500, 5))
    >>> y = np.random.poisson(np.exp(X.dot(np.ones(X.shape[1]))))
    >>> batch_size, stepsize = compute_opt_params(softplus_poisson_l_max_and_l, X, y, strong_convexity=0.08)
    """
    # if both parameters are set by the user then just return them
    if batch_size is not None and stepsize is not None:
        return {"batch_size": batch_size, "stepsize": stepsize}

    data = jax.tree_util.tree_map(jnp.asarray, data)

    num_samples = {dd.shape[0] for dd in jax.tree_util.tree_leaves(data)}

    if len(num_samples) != 1:
        raise ValueError("Each array in data must have the same number of samples.")
    num_samples = num_samples.pop()

    l_smooth_max, l_smooth = compute_smoothness_constants(
        *data, n_power_iters=n_power_iters
    )

    if batch_size is None:
        batch_size = _calculate_optimal_batch_size_svrg(
            num_samples, l_smooth_max, l_smooth, strong_convexity=strong_convexity
        )

    if not jnp.isfinite(batch_size):
        batch_size = default_batch_size
        warnings.warn(
            "Could not determine batch and step size automatically. "
            f"Falling back on the default values of {batch_size} and {default_stepsize}."
        )

    if stepsize is None:
        stepsize = _calculate_stepsize_svrg(
            batch_size, num_samples, l_smooth_max, l_smooth
        )

    return {"batch_size": int(batch_size), "stepsize": stepsize}


def softplus_poisson_l_max_and_l(
    *data, n_power_iters: Optional[int] = 20
) -> Tuple[float, float]:
    """
    Calculate the smoothness constant and maximum smoothness constant for SVRG
    assuming that the optimized function is the log-likelihood of a Poisson GLM
    with a softplus inverse link function.

    Parameters
    ----------
    data:
        Tuple of X and y.
    n_power_iters :
        If None, calculate X.T @ D @ X and its largest eigenvalue directly.
        If an integer, the umber of power iterations to use to calculate the largest eigenvalue.

    Returns
    -------
    l_smooth_max, l_smooth :
        Maximum smoothness constant and smoothness constant.
    """
    X, y = data

    # takes care of population glm (see bound found on overleaf)
    y = jnp.max(y, axis=tuple(range(1, y.ndim)))

    # concatenate all data (if X is FeaturePytree)
    X = jnp.hstack(jax.tree_util.tree_leaves(X))

    l_smooth = _softplus_poisson_l_smooth(X, y, n_power_iters)
    l_smooth_max = _softplus_poisson_l_smooth_max(X, y)
    return l_smooth_max, l_smooth


def _softplus_poisson_l_smooth_multiply(X, y, v):
    """
    Perform the multiplication of v with X.T @ D @ X without forming the full X.T @ D @ X.

    This assumes that X fits in memory. This estimate is based on calculating the hessian of the loss.

    Parameters
    ----------
    X :
        Input data.
    y :
        Output data.
    v :
        d-dimensional vector.

    Returns
    -------
    :
        X.T @ D @ X @ v
    """
    N, _ = X.shape
    return X.T.dot((0.17 * y + 0.25) * X.dot(v)) / N


def _softplus_poisson_l_smooth_with_power_iteration(X, y, n_power_iters: int = 20):
    """
    Instead of calculating X.T @ D @ X and its largest eigenvalue directly,
    calculate it using the power method and by iterating through X and y,
    forming a small product at a time.

    Parameters
    ----------
    X :
        Input data.
    y :
        Output data.
    n_power_iters :
        Number of power iterations.

    Returns
    -------
    The largest eigenvalue of X.T @ D @ X
    """
    # key is fixed to random.key(0)
    _, d = X.shape

    # initialize a random d-dimensional vector
    v = jnp.ones((d,))

    # run the power iteration until convergence or the max steps
    for _ in range(n_power_iters):
        v_prev = v.copy()
        v = _softplus_poisson_l_smooth_multiply(X, y, v)
        v /= v.max()

        if jnp.allclose(v_prev, v):
            break

    # calculate the eigenvalue
    v /= jnp.linalg.norm(v)
    return _softplus_poisson_l_smooth_multiply(X, y, v).dot(v)


def _softplus_poisson_l_smooth(
    X: jnp.ndarray, y: jnp.ndarray, n_power_iters: Optional[int] = None
):
    """
    Calculate the smoothness constant from data, assuming that the optimized
    function is the log-likelihood of a Poisson GLM with a softplus inverse link function.

    Parameters
    ----------
    X :
        Input data.
    y :
        Output data.

    Returns
    -------
    L :
        Smoothness constant of f.
    """
    if n_power_iters is None:
        # calculate XDX/n and its largest eigenvalue directly
        XDX = X.T.dot((0.17 * y.reshape(y.shape[0], 1) + 0.25) * X) / y.shape[0]
        return jnp.sort(jnp.linalg.eigvalsh(XDX))[-1]
    else:
        # use the power iteration to calculate the largest eigenvalue
        return _softplus_poisson_l_smooth_with_power_iteration(X, y, n_power_iters)


def _softplus_poisson_l_smooth_max(X: jnp.ndarray, y: jnp.ndarray):
    """
    Calculate the maximum smoothness constant from data, assuming that
    the optimized function is the log-likelihood of a Poisson GLM with
    a softplus inverse link function.

    Parameters
    ----------
    X :
        Input data.
    y :
        Output data.

    Returns
    -------
    L_max :
        Maximum smoothness constant among f_{i}.
    """
    N, _ = X.shape

    def body_fun(i, current_max):
        return jnp.maximum(
            current_max, jnp.linalg.norm(X[i, :]) ** 2 * (0.17 * y[i] + 0.25)
        )

    L_max = jax.lax.fori_loop(0, N, body_fun, jnp.array([-jnp.inf]))

    return L_max[0]


@_convert_to_float
def _calculate_stepsize_svrg(
    batch_size: int, num_samples: int, l_smooth_max: float, l_smooth: float
):
    """
    Calculate optimal step size for SVRG$^{[1]}$.

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


@_convert_to_float
def _calculate_stepsize_saga(
    batch_size: int, num_samples: int, l_smooth_max: float, l_smooth: float
) -> float:
    """
    Calculate optimal step size for SAGA.

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
    [1] Gazagnadou, Nidham, Robert Gower, and Joseph Salmon.
    "Optimal mini-batch and step sizes for saga."
    International conference on machine learning. PMLR, 2019.
    """

    l_b = l_smooth * num_samples / batch_size * (batch_size - 1) / (
        num_samples - 1
    ) + l_smooth_max / batch_size * (num_samples - batch_size) / (num_samples - 1)

    return 0.25 / l_b


def _calculate_optimal_batch_size_svrg(
    num_samples: int,
    l_smooth_max: float,
    l_smooth: float,
    strong_convexity: Optional[float] = None,
) -> int:
    """
    Calculate the optimal batch size according to "Table 1" in [1].

    Parameters
    ----------
    num_samples:
        The number of samples.
    l_smooth_max:
        The Lmax smoothness constant.
    l_smooth:
        The L smoothness constant.
    strong_convexity:
        The strong convexity constant.

    Returns
    -------
    batch_size:
        The batch size.

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
    Helper function for calculating $\hat{b}$ in "Table 1" of [1].

    Parameters
    ----------
    num_samples :
        Overall number of data points.
    l_smooth_max :
        Maximum smoothness constant among f_{i}.
    l_smooth :
        Smoothness constant.

    Returns
    -------
    :
        Optimal batch size for the optimization.

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
    Helper function for calculating $\tilde{b}$ as in "Table 1" of [1].

    Parameters
    ----------
    num_samples :
        Overall number of data points.
    l_smooth_max :
        Maximum smoothness constant among f_{i}.
    l_smooth :
        Smoothness constant.
    strong_convexity :
        Strong convexity constant.

    Returns
    -------
    :
        Optimal batch size for the optimization.

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
