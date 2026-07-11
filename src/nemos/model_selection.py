"""Model-selection utilities for GLMs.

This module implements approximate leave-one-out cross-validation (LOO-CV) for
:class:`nemos.glm.GLM` and :class:`nemos.glm.PopulationGLM`. Exact LOO-CV requires
refitting the model ``n`` times (once per held-out observation), which is expensive.
The infinitesimal-jackknife / one-step-Newton approximation recovers per-observation
LOO predictions from a *single* full-data fit plus one ``O(p^2)`` correction per
observation, where ``p`` is the number of features.

References
----------
.. [1] Pregibon, D. (1981). Logistic regression diagnostics. *The Annals of
       Statistics*, 9(4), 705-724.
.. [2] Rad, K. R., & Maleki, A. (2020). A scalable estimate of the out-of-sample
       prediction error via approximate leave-one-out cross-validation. *Journal of
       the Royal Statistical Society: Series B*, 82(4), 965-996.
.. [3] Giordano, R., Stephenson, W., Liu, R., Jordan, M. I., & Broderick, T. (2019).
       A Swiss army infinitesimal jackknife. *AISTATS*.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .glm.glm import _var_func_of_mu
from .regularizer import ElasticNet, GroupLasso, Lasso
from .utils import _elementwise_derivative

__all__ = ["approximate_loo", "ApproximateLOO"]


class ApproximateLOO(NamedTuple):
    r"""Result of :func:`approximate_loo`.

    All arrays are indexed by observation (and by neuron, for
    :class:`~nemos.glm.PopulationGLM`, along a trailing axis).

    Attributes
    ----------
    predicted_mean :
        Approximate leave-one-out predicted mean :math:`\mu_i^{(-i)}` for each
        observation. Shape ``(n_samples,)`` for :class:`~nemos.glm.GLM`, or
        ``(n_samples, n_neurons)`` for :class:`~nemos.glm.PopulationGLM`.
    linear_predictor :
        Approximate leave-one-out linear predictor :math:`\eta_i^{(-i)}`, same shape
        as ``predicted_mean``.
    log_likelihood :
        Per-observation held-out log-likelihood evaluated at ``predicted_mean``,
        :math:`\log p(y_i \mid \mu_i^{(-i)})`. Summing (or averaging) over
        observations gives an approximate LOO log-likelihood.
    deviance :
        Per-observation held-out deviance evaluated at ``predicted_mean``.
    leverage :
        Diagnostic hat-matrix diagonal :math:`h_{ii} \in [0, 1)`. Values close to 1
        flag high-leverage points, where the approximation is least accurate.
    """

    predicted_mean: jnp.ndarray
    linear_predictor: jnp.ndarray
    log_likelihood: jnp.ndarray
    deviance: jnp.ndarray
    leverage: jnp.ndarray


def _alo_linear_predictor(X_aug, eta, w, s, hessian):
    r"""Core one-step-Newton LOO update on the linear-predictor scale (one GLM).

    Parameters
    ----------
    X_aug :
        Augmented design matrix ``[X, 1]`` of shape ``(n_samples, p + 1)`` (the trailing
        column of ones corresponds to the intercept).
    eta :
        Full-data linear predictor, shape ``(n_samples,)``.
    w :
        Fisher working weights :math:`w_i = g'(\eta_i)^2 / V(\mu_i)`, shape
        ``(n_samples,)``.
    s :
        Per-observation score :math:`s_i = \partial_{\eta_i}\,[-\log p(y_i\mid\mu_i)]`,
        shape ``(n_samples,)``.
    hessian :
        Curvature ``A = X_aug^T W X_aug + penalty`` in the *summed*-loss convention,
        shape ``(p + 1, p + 1)``.

    Returns
    -------
    eta_loo :
        Approximate LOO linear predictor, shape ``(n_samples,)``.
    leverage :
        Hat-matrix diagonal :math:`h_{ii}`, shape ``(n_samples,)``.
    """
    # generalized leverage g_i = x_i^T A^{-1} x_i (no working weight), then h_ii = w_i g_i
    g = jnp.sum(X_aug.T * jnp.linalg.solve(hessian, X_aug.T), axis=0)
    leverage = w * g
    delta_eta = s * g / (1.0 - leverage)
    return eta + delta_eta, leverage


def approximate_loo(
    model,
    X,
    y,
) -> ApproximateLOO:
    r"""Approximate leave-one-out cross-validation for a fitted GLM.

    Exact leave-one-out cross-validation (LOO-CV) refits the model ``n`` times, each
    time holding out a single observation. This function instead approximates every
    held-out fit with a single Newton step taken from the full-data solution, following
    the infinitesimal-jackknife / approximate-leave-one-out (ALO) method
    ([1]_, [2]_). It requires one full-data fit and an ``O(p^2)`` correction per
    observation (``p`` = number of features), rather than ``n`` refits.

    At the converged IRLS/Newton solution :math:`\hat\beta` with Fisher working
    weights :math:`w_i = g'(\eta_i)^2 / V(\mu_i)` (for the canonical-link Poisson model
    :math:`w_i = \mu_i`), let :math:`x_i` be the ``i``-th augmented design row
    :math:`[X_i, 1]`, let

    .. math::
        A = X^T W X + \text{penalty}, \qquad
        H = W^{1/2} X A^{-1} X^T W^{1/2},

    and let :math:`h_{ii}` be the ``i``-th diagonal of the hat matrix :math:`H`. The
    approximate leave-one-out parameter estimate for observation ``i`` is

    .. math::
        \hat\beta^{(-i)} \approx \hat\beta + A^{-1} x_i \, s_i / (1 - h_{ii}),

    where :math:`s_i = \partial_{\eta_i}\,[-\log p(y_i \mid \mu_i)]` is the working
    score contribution of observation ``i`` (for Poisson, :math:`s_i = \mu_i - y_i`).
    This yields the approximate LOO linear predictor and mean without refitting:

    .. math::
        \eta_i^{(-i)} \approx \eta_i + \frac{s_i \, x_i^T A^{-1} x_i}{1 - h_{ii}},
        \qquad \mu_i^{(-i)} = g^{-1}\!\left(\eta_i^{(-i)}\right).

    The approximation is exact to first order and degrades for high-leverage points
    (:math:`h_{ii}\to 1`); the returned :attr:`ApproximateLOO.leverage` flags them.

    Parameters
    ----------
    model :
        A **fitted** :class:`~nemos.glm.GLM` or :class:`~nemos.glm.PopulationGLM`.
    X :
        The design matrix used to fit ``model``, shape ``(n_samples, n_features)`` (or a
        pytree of such arrays). LOO-CV is defined on the training data, so ``X`` and
        ``y`` should be the same data passed to :meth:`~nemos.glm.GLM.fit`.
    y :
        The observations used to fit ``model``, shape ``(n_samples,)`` for
        :class:`~nemos.glm.GLM` or ``(n_samples, n_neurons)`` for
        :class:`~nemos.glm.PopulationGLM`.

    Returns
    -------
    :
        An :class:`ApproximateLOO` named tuple with the per-observation approximate LOO
        predicted mean, linear predictor, log-likelihood, deviance, and leverage.

    Raises
    ------
    NotImplementedError
        If ``model`` uses a non-smooth regularizer (:class:`~nemos.regularizer.Lasso`,
        :class:`~nemos.regularizer.ElasticNet`, or :class:`~nemos.regularizer.GroupLasso`),
        for which the infinitesimal-jackknife formula is not valid (the objective is not
        twice differentiable at the solution; see [2]_), if the observation model has no
        defined variance function (e.g. ``NegativeBinomial``), or if ``model`` is a
        :class:`~nemos.glm.PopulationGLM` with a ``feature_mask``.

    Notes
    -----
    Regularization is respected through the curvature ``A``: a :class:`~nemos.regularizer.Ridge`
    penalty contributes ``n * regularizer_strength`` to the diagonal of ``A`` (matching
    nemos's mean-loss + un-normalized-penalty objective), while the intercept is left
    unpenalized. Only smooth penalties are supported; non-smooth penalties raise.

    The curvature ``A`` reuses the model's own analytic Fisher-information Hessian
    (:meth:`~nemos.glm.GLM._get_hess_fn`), so results are consistent with the model's
    Newton solver.

    Examples
    --------
    >>> import numpy as np
    >>> import nemos as nmo
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(100, 3))
    >>> y = rng.poisson(np.exp(X @ np.array([0.2, -0.1, 0.3]) - 0.5)).astype(float)
    >>> model = nmo.glm.GLM().fit(X, y)
    >>> loo = nmo.model_selection.approximate_loo(model, X, y)
    >>> loo.predicted_mean.shape
    (100,)
    >>> approx_loo_log_likelihood = loo.log_likelihood.mean()

    References
    ----------
    .. [1] Pregibon, D. (1981). Logistic regression diagnostics. *The Annals of
           Statistics*, 9(4), 705-724.
    .. [2] Rad, K. R., & Maleki, A. (2020). A scalable estimate of the out-of-sample
           prediction error via approximate leave-one-out cross-validation. *Journal of
           the Royal Statistical Society: Series B*, 82(4), 965-996.
    """
    model._check_is_fit()

    # non-smooth penalties break the infinitesimal-jackknife assumptions (Rad & Maleki 2020)
    if isinstance(model.regularizer, (Lasso, ElasticNet, GroupLasso)):
        raise NotImplementedError(
            "`approximate_loo` is not available for non-smooth regularizers "
            f"({type(model.regularizer).__name__}). The infinitesimal-jackknife formula "
            "requires a twice-differentiable objective; see Rad & Maleki (2020). Use an "
            "`UnRegularized` or `Ridge` model, or run exact refit-based LOO-CV."
        )

    if getattr(model, "_feature_mask", None) is not None:
        raise NotImplementedError(
            "`approximate_loo` does not yet support `PopulationGLM` with a `feature_mask`."
        )

    # validate/preprocess exactly as `score` does (drops NaNs, checks consistency)
    params = model._get_model_params()
    model._validator.validate_inputs(X, y)
    X, y = model._preprocess_inputs(X, y, drop_nans=True)
    model._validator.validate_consistency(params, X, y)

    inv_link = model._inverse_link_function
    var_of_mu = _var_func_of_mu(
        model
    )  # raises for observation models w/o a variance fn
    gprime = _elementwise_derivative(inv_link)

    # --- flat augmented design [X, 1], ordered [coef features..., intercept] to match
    # the parameter ordering used by `_glm_hessian_block` / `_get_hess_fn`.
    design = jnp.concatenate(jax.tree_util.tree_leaves(X), axis=1)
    n_samples = design.shape[0]
    X_aug = jnp.concatenate([design, jnp.ones((n_samples, 1))], axis=1)

    # --- linear predictor eta = X_aug @ [coef; intercept] (matches `_predict`)
    coef_flat = jnp.concatenate(
        [
            jnp.reshape(leaf, (leaf.shape[0], -1))
            for leaf in jax.tree_util.tree_leaves(params.coef)
        ],
        axis=0,
    )  # (p, n_neurons) with n_neurons == 1 for a plain GLM
    intercept = jnp.atleast_1d(params.intercept)
    beta_aug = jnp.concatenate(
        [coef_flat, intercept[None, :]], axis=0
    )  # (p + 1, n_neurons)
    eta = X_aug @ beta_aug  # (n, n_neurons)
    is_population = jnp.ndim(y) == 2
    if not is_population:
        eta = eta[:, 0]
    mu = inv_link(eta)

    # --- Fisher working weights w_i = g'(eta_i)^2 / V(mu_i)
    w = gprime(eta) ** 2 / var_of_mu(mu)

    # --- exact per-observation score s_i = d/d eta_i [ -log p(y_i | mu_i) ].
    # The NLL is separable across observations, so grad of the summed NLL yields the
    # per-observation eta-derivatives directly (for Poisson this equals mu_i - y_i).
    def _summed_nll(linear_predictor):
        return model._observation_model._negative_log_likelihood(
            y, inv_link(linear_predictor), aggregate_sample_scores=jnp.sum
        )

    score = jax.grad(_summed_nll)(eta)

    # --- curvature A = X^T W X + penalty in the summed-loss convention.
    # `_get_hess_fn` returns the mean-loss Fisher Hessian ((1/n) X^T W X + penalty);
    # multiply by n to move to the summed convention used by the ALO formula above.
    hess = model._get_hess_fn(params, autodiff=False)(params, X)
    A = n_samples * hess

    if is_population:
        # independent per-neuron blocks: A has shape (n_neurons, p + 1, p + 1)
        eta_loo, leverage = jax.vmap(
            lambda e, ww, ss, a: _alo_linear_predictor(X_aug, e, ww, ss, a),
            in_axes=(1, 1, 1, 0),
            out_axes=1,
        )(eta, w, score, A)
    else:
        eta_loo, leverage = _alo_linear_predictor(X_aug, eta, w, score, A)

    mu_loo = inv_link(eta_loo)

    scale = model.scale_ if model.scale_ is not None else 1.0
    log_likelihood = model._observation_model.log_likelihood(
        y, mu_loo, scale, aggregate_sample_scores=lambda x: x
    )
    deviance = model._observation_model.deviance(y, mu_loo, scale)

    return ApproximateLOO(
        predicted_mean=mu_loo,
        linear_predictor=eta_loo,
        log_likelihood=log_likelihood,
        deviance=deviance,
        leverage=leverage,
    )
