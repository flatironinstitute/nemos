"""Initialization of GLM parameters."""

from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp
from numpy.typing import ArrayLike
from scipy.optimize import root_scalar

from ..inverse_link_function_utils import (
    exp,
    identity,
    log_softmax,
    logistic,
    norm_cdf,
    softplus,
)
from ..typing import Pytree
from ..utils import one_over_x


def _log_softmax_inv(x):
    """Inverse of log_softmax with centering.

    For over-parameterized multinomial models, we center the log-probabilities
    by subtracting the mean. This ensures the intercepts sum to zero, matching
    sklearn's implicit constraint and making the parameters identifiable.
    """
    # Clipping is needed when initializing with a batch that do not contain
    # a category. In that case, the empirical frequency associated to the
    # category would be zero, and log(0) will be -inf.
    log_x = jnp.log(jnp.clip(x, jnp.finfo(float).eps, jnp.inf))
    return log_x - jnp.mean(log_x, axis=-1, keepdims=True)


def _softplus_inv(x):
    """Robust inverse of softplus, for small rates and above the exp overflow."""
    return x + jnp.log(-jnp.expm1(-x))


# dictionary of known inverse link functions.
INVERSE_FUNCS = {
    exp: jnp.log,
    softplus: _softplus_inv,
    logistic: jax.scipy.special.logit,
    norm_cdf: jax.scipy.stats.norm.ppf,
    one_over_x: one_over_x,
    identity: identity,
    log_softmax: _log_softmax_inv,
}

# Name-based lookup (for after pickling/copying)
INVERSE_FUNCS_BY_SIMPLE_NAME = {
    "exp": jnp.log,
    "softplus": _softplus_inv,
    "logistic": jax.scipy.special.logit,
    "norm_cdf": jax.scipy.stats.norm.ppf,
    "one_over_x": one_over_x,
    "identity": identity,
    "log_softmax": _log_softmax_inv,
}

non_finite_error = ValueError(
    "Failed to initialize the model intercept as the inverse of the firing rate for "
    "the provided link function. The inferred intercept has non-finite values. "
    "Please provide initial parameters instead."
)


def get_inverse_function(func: Callable):
    """Get the inverse function for a given link function."""
    # Strategy 1: Try identity lookup (fast path)
    if func in INVERSE_FUNCS:
        return INVERSE_FUNCS[func]

    # Strategy 2: Try name lookup (for copied/pickled functions)
    if hasattr(func, "__name__") and func.__name__ in INVERSE_FUNCS_BY_SIMPLE_NAME:
        return INVERSE_FUNCS_BY_SIMPLE_NAME[func.__name__]

    # No inverse function found
    return None


def scalar_root_find_elementwise(
    func: Callable, args: ArrayLike, x0: ArrayLike
) -> jnp.ndarray:
    """
    Find roots of a scalar function.

    This can be used as an attempt to find a numerical inverse of an unknown link function of a GLM; typically,
    this numerical inverse, is used to set the initial intercept to match the mean firing rate of the model.

    Parameters
    ----------
    func:
        A callable, which typically will be `inv_link_func(x) - jnp.mean(spikes)`.
    args:
        List of additional arguments passed to the function.
    x0:
        Initial values for the root-finding algorithm.

    Returns
    -------
    :
        An array containing the roots of each f(x) = func(x, args[k]), for k in 1,..., len(args).

    Raises
    ------
    ValueError:
        If any of the optimization is not successful.
    """
    opts = [root_scalar(func, arg, x0=x, method="secant") for arg, x in zip(args, x0)]

    if not all(jnp.abs(func(opt.root, args[i])) < 10**-4 for i, opt in enumerate(opts)):
        raise ValueError(
            "Could not set the initial intercept as the inverse of the firing rate for "
            "the provided link function. "
            "Please, provide initial parameters instead!"
        )

    return jnp.array([opt.root for opt in opts])


def compute_frozen_linear_predictor(X, frozen_coef, frozen_intercept):
    """Linear predictor from the frozen parameters.

    A frozen coefficient leaf contributes its ``X @ coef`` term; a free (``None``) leaf
    contributes a broadcastable ``[0.]``. ``frozen_intercept`` is added when present.
    """
    intercept = 0.0 if frozen_intercept is None else frozen_intercept
    return (
        jax.tree.reduce(
            jnp.add,
            jax.tree.map(
                lambda x1, x2: (
                    jnp.matmul(x1, x2) if x2 is not None else jnp.array([0.0])
                ),
                X,
                frozen_coef,
            ),
        )
        + intercept
    )


def initialize_intercept_matching_mean_rate(
    inverse_link_function: Callable,
    X: Union[Pytree, jnp.ndarray],
    y: jnp.ndarray,
    frozen_coef: Optional[Union[Pytree, jnp.ndarray]] = None,
    frozen_intercept: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Compute the initial intercept term for a regression models.

    This method compute an initial intercept term for a regression models such that the baseline activity
    matches the mean activity of each neuron, assuming that the model coefficients are initialized to zero.


    Parameters
    ----------
    inverse_link_function:
        The inverse link function of the model, linking the mean to the linear combination of the covariates in
        a GLM.
    X:
        The predictors.
    y:
        The neural activity, shape either (num_sample,) for single variable regressors as `GLM`
         or (n_sample, n_neurons) for multi-variable regressors, such as `PopulaitonGLM`.
    frozen_coef:
        The frozen parameters.

    Returns
    -------
    :
        The initial intercept term, shape (n_neurons,).

    """
    y = jnp.asarray(y, float)
    # expand tree to match structure
    if frozen_coef is None:
        frozen_coef = jax.tree_util.tree_map(lambda x: None, X)

    # return inverse if analytical solution is available
    analytical_inv = get_inverse_function(inverse_link_function)

    # compute the linear predictor from the frozen tree
    frozen_lin_pred = compute_frozen_linear_predictor(X, frozen_coef, frozen_intercept)
    mean_frozen = jnp.nanmean(frozen_lin_pred, axis=0)

    means = jnp.atleast_1d(jnp.nanmean(y, axis=0))
    if analytical_inv:
        out = analytical_inv(means) - mean_frozen
        if jnp.any(jnp.isnan(out)):
            raise ValueError(
                "Failed to initialize the model intercept as the inverse of the firing rate for "
                "the provided link function. The mean firing rate has some non-positive values."
            )
        if jnp.any(~jnp.isfinite(out)):
            raise non_finite_error

        return out

    def func(x, mean_x):
        return inverse_link_function(x) - mean_x

    try:
        out = scalar_root_find_elementwise(func, means, means) - mean_frozen
    except ValueError:
        raise ValueError(
            "Failed to initialize the model intercept as the inverse of the firing rate for the"
            " provided link function. Please, provide initial parameters instead!"
        )

    if jnp.any(~jnp.isfinite(out)):
        raise non_finite_error

    return out


def initialize_constant_coef_matching_mean_rate(
    inverse_link_function: Callable,
    X: Union[Pytree, jnp.ndarray],
    y: jnp.ndarray,
    empty_coef: Union[Pytree, jnp.ndarray],
    frozen_coef: Optional[Union[Pytree, jnp.ndarray]] = None,
    frozen_intercept: Optional[jnp.ndarray] = None,
    eps: Optional[float] = None,
) -> Union[Pytree, jnp.ndarray]:
    r"""
    Initialize coefficients as a constant matching the mean rate, with no intercept.

    When the intercept is held fixed at zero, coefficients initialized to zero would
    force the linear predictor to zero and, for example, imply an unreasonably high
    firing rate for a Poisson GLM with an exponential link. Instead, this sets every
    coefficient to a single constant ``c`` (per output), so that the linear predictor is

    .. math::
        \eta_t = \sum_j X_{tj}\, c = c\, s_t, \qquad s_t = \sum_j X_{tj},

    where :math:`s_t` is the row-sum of the design matrix. The constant is chosen as the
    least-squares projection of the uniform offset :math:`\eta^\star` (the value a free
    intercept would take, i.e. the inverse link of the mean rate) onto :math:`s` (see Notes):

    .. math::
        c = \eta^\star\, \frac{\sum_t s_t + \varepsilon}{\sum_t s_t^2 + \varepsilon},

    so the linear predictor stays close to :math:`\eta^\star` regardless of how large
    ``c`` grows. :math:`\varepsilon` is the numerical precision, introduced to get finite
    ``c`` even when :math:`s_t` is identically zero.

    Parameters
    ----------
    inverse_link_function :
        The inverse link function of the model.
    X :
        The design matrix, an array of shape ``(n_samples, n_features)`` or a pytree of
        such arrays. NaNs are ignored when forming the row-sum.
    y :
        The neural activity, shape ``(n_samples,)`` for single-output regressors such as
        ``GLM`` or ``(n_samples, n_neurons)`` for multi-output regressors such as
        ``PopulationGLM``.
    empty_coef :
        A pytree matching the target coefficient structure and shapes (e.g. as returned
        by the validator's ``get_empty_params``). Used only for its leaf shapes.
    frozen_coef :
        Pytree including the fixed valued model coefficients.
    eps :
        Stabilization added to both numerator and denominator. Defaults to the machine
        epsilon of the row-sum dtype, which only intervenes when the design row-sums are
        numerically zero and is otherwise negligible.

    Returns
    -------
    :
        The initialized coefficients, matching the structure and shapes of ``empty_coef``.

    Notes
    -----
    The constant :math:`c \in \mathbb{R}` is the minimizer of

    .. math::
        \mathcal{L}(c) = \sum_t \left(c\, s_t - \eta^\star\right)^2,

    where :math:`\eta^\star` is the linear-predictor value whose inverse link matches the
    mean rate (:math:`\text{inverse\_link}(\eta^\star) = \bar{y}`), and
    :math:`s_t = \sum_j X_{tj}` is the sum of the design over features. Setting
    :math:`\mathcal{L}'(c) = 0` gives the closed form above (with the
    :math:`\varepsilon` terms added for numerical stability). The problem is degenerate
    only when :math:`s \equiv 0`, in which case no constant coefficient can inject an
    offset and the :math:`\varepsilon` stabilization keeps :math:`c` finite.

    When some coefficients are frozen, their contribution is subtracted from the target
    per sample (:math:`\eta^\star_t = \eta^\star - \sum_j X_{tj}\,\beta^{\text{frozen}}_j`)
    and the frozen features are excluded from :math:`s_t`; for multi-output models the
    projection is computed independently for each output.
    """
    if frozen_coef is None:
        frozen_coef = jax.tree_util.tree_map(lambda x: None, X)
    # the linear-predictor target a free intercept would supply, shape (*out,)
    eta_target = initialize_intercept_matching_mean_rate(inverse_link_function, X, y)

    # coef is (n_features, *out): out is () for GLM, (n_out,) for population/classifier
    # GLM, (n_neurons, n_classes) for the population classifier. The frozen predictor is
    # therefore (n_samples, *out), or *out alone for a frozen intercept. The GLM matmul
    # drops its trailing size-1 output axis, so restore it before subtracting eta_target.
    frozen_lin_pred = compute_frozen_linear_predictor(X, frozen_coef, frozen_intercept)
    if eta_target.shape == (1,) and frozen_lin_pred.ndim == 1:
        frozen_lin_pred = frozen_lin_pred[:, None]
    residual = eta_target[None, ...] - frozen_lin_pred

    # row-sum of the design across all active features of all leaves, shape (n_samples,)
    row_sum = jax.tree.map(
        lambda x1, x2: jnp.nansum(x1, axis=1) if x2 is None else jnp.array(0.0),
        X,
        frozen_coef,
    )
    row_sum = jax.tree.reduce(jnp.add, row_sum)
    if eps is None:
        eps = jnp.finfo(row_sum.dtype).eps
    # per-output least-squares projection: weight each sample's residual by its row-sum
    # (broadcast across the output axes) and sum over samples. Result shape ``*out``.
    weight = row_sum.reshape(row_sum.shape + (1,) * eta_target.ndim)
    const = (jnp.sum(weight * residual, axis=0) + eps) / (jnp.sum(row_sum**2) + eps)
    return jax.tree_util.tree_map(
        lambda leaf1, leaf2: jnp.full_like(leaf1, const) if leaf2 is None else leaf2,
        empty_coef,
        frozen_coef,
        is_leaf=lambda x: x is None,
    )
