import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nemos as nmo
from nemos.glm.initialize_parameters import (
    initialize_constant_coef_matching_mean_rate,
    initialize_intercept_matching_mean_rate,
)
from nemos.glm.params import GLMParams
from nemos.inverse_link_function_utils import exp as nmo_exp


@pytest.mark.parametrize(
    "non_linearity",
    [
        jnp.exp,
        jax.nn.softplus,
        lambda x: jnp.exp(x),
        jax.nn.sigmoid,
        jax.lax.logistic,
        lambda x: jax.lax.logistic(x),
        jax.scipy.special.expit,
        jax.scipy.stats.norm.cdf,
    ],
)
@pytest.mark.parametrize(
    "output_y",
    [np.random.uniform(0, 1, size=(10,)), np.random.uniform(0, 1, size=(10, 2))],
)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_invert_non_linearity(non_linearity, output_y):
    inv_y = initialize_intercept_matching_mean_rate(
        inverse_link_function=non_linearity, y=output_y
    )
    assert jnp.allclose(non_linearity(inv_y), jnp.mean(output_y, axis=0), rtol=10**-5)


@pytest.mark.parametrize(
    "non_linearity, expectation",
    [
        (jnp.exp, pytest.raises(ValueError, match=".+The mean firing rate has")),
        (
            jax.nn.softplus,
            pytest.raises(ValueError, match=".+The mean firing rate has"),
        ),
        (
            lambda x: jnp.exp(x),
            pytest.raises(
                ValueError, match=".+Please, provide initial parameters instead"
            ),
        ),
        (
            jax.nn.sigmoid,
            pytest.raises(
                ValueError, match=".+Please, provide initial parameters instead"
            ),
        ),
        (
            jax.lax.logistic,
            pytest.raises(ValueError, match=".+The mean firing rate has"),
        ),
        (
            lambda x: jax.lax.logistic(x),
            pytest.raises(
                ValueError, match=".+Please, provide initial parameters instead"
            ),
        ),
        (
            jax.scipy.stats.norm.cdf,
            pytest.raises(ValueError, match=".+Please, provide initial parameters"),
        ),
    ],
)
def test_initialization_error_nan_input(non_linearity, expectation):
    """Initialize invalid."""
    output_y = np.full((10, 2), np.nan)
    with expectation:
        initialize_intercept_matching_mean_rate(
            inverse_link_function=non_linearity, y=output_y
        )


def test_initialization_error_non_invertible():
    """Initialize invalid."""
    output_y = np.random.uniform(size=100)
    inv_link = lambda x: jax.nn.softplus(x) + 10
    with pytest.raises(
        ValueError, match="Failed to initialize the model intercept.+Please, provide"
    ):
        with warnings.catch_warnings():
            # ignore the warning raised by the root-finder (there is no root)
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message="Tolerance of"
            )
            initialize_intercept_matching_mean_rate(
                inverse_link_function=inv_link, y=output_y
            )


def test_initialization_error_logistic_all_one_output():
    output_y = np.ones(10)
    with pytest.raises(ValueError, match="has non-finite values"):
        initialize_intercept_matching_mean_rate(
            inverse_link_function=jax.lax.logistic, y=output_y
        )


# ---------------------------------------------------------------------------
# initialize_constant_coef_matching_mean_rate
# ---------------------------------------------------------------------------


def _row_sum(X):
    """Row-sum of the design over features across all leaves, matching the impl."""
    return sum(jnp.nansum(leaf, axis=1) for leaf in jax.tree_util.tree_leaves(X))


def _freeze_intercept_partition(params):
    """Split params into (trainable=coef, frozen=intercept).

    Local stand-in for the model-side freezing mechanism (not yet plugged into
    the model): the solver optimizes ``diff`` (intercept is ``None``), and the
    frozen intercept is recombined afterwards.
    """
    filter_spec = jax.tree_util.tree_map(lambda _: True, params)
    filter_spec = eqx.tree_at(lambda p: p.intercept, filter_spec, replace=False)
    return eqx.partition(params, filter_spec)


def _make_design(kind, n=4000, p=3, seed=0):
    """Corner-case design matrices, all shape (n, p)."""
    rng = np.random.RandomState(seed)
    if kind == "positive":
        X = np.abs(rng.randn(n, p))
    elif kind == "centered":
        # each column sums to ~0 over samples -> row-sums have ~zero mean
        X = np.abs(rng.randn(n, p))
        X = X - X.mean(axis=0, keepdims=True)
    elif kind == "outliers":
        X = 0.01 * rng.randn(n, p)
        X[:: n // 20] += 50.0  # a few large-magnitude rows
    elif kind == "with_nans":
        X = np.abs(rng.randn(n, p))
        X[::50, 0] = np.nan
    elif kind == "all_zero":
        X = np.zeros((n, p))
    else:  # pragma: no cover
        raise ValueError(kind)
    return jnp.asarray(X)


@pytest.fixture
def design(request):
    """A corner-case design matrix (n, p), selected by indirect parametrization."""
    return _make_design(request.param)


def _make_y(n=4000, n_neurons=None, seed=1):
    rng = np.random.RandomState(seed)
    if n_neurons is None:
        return jnp.asarray(rng.poisson(0.1, size=n).astype(float))
    rates = np.linspace(0.05, 0.5, n_neurons)
    return jnp.asarray(rng.poisson(rates, size=(n, n_neurons)).astype(float))


@pytest.mark.parametrize(
    "design", ["positive", "centered", "outliers", "with_nans"], indirect=True
)
@pytest.mark.parametrize("as_dict", [False, True])
@pytest.mark.parametrize("n_neurons", [None, 2])
def test_constant_coef_satisfies_normal_equation(design, as_dict, n_neurons):
    """The constant is the least-squares minimizer, i.e. it satisfies the
    (eps-regularized) normal equation ``c (Σs² + eps) = η* (Σs + eps)``."""
    X = design
    p = X.shape[1]
    y = _make_y(n_neurons=n_neurons)
    if as_dict:
        X = {"a": X[:, :1], "b": X[:, 1:]}
        empty_coef = {
            "a": jnp.empty((1,) if n_neurons is None else (1, n_neurons)),
            "b": jnp.empty((2,) if n_neurons is None else (2, n_neurons)),
        }
    else:
        empty_coef = jnp.empty((p,) if n_neurons is None else (p, n_neurons))

    coef = initialize_constant_coef_matching_mean_rate(nmo_exp, X, y, empty_coef)

    # structure and shapes preserved
    assert jax.tree_util.tree_structure(coef) == jax.tree_util.tree_structure(
        empty_coef
    )
    for c, e in zip(
        jax.tree_util.tree_leaves(coef), jax.tree_util.tree_leaves(empty_coef)
    ):
        assert c.shape == e.shape
        assert jnp.all(jnp.isfinite(c))

    # every leaf is a single constant per output
    const_leaves = [
        jnp.reshape(c, (-1,) if n_neurons is None else (-1, n_neurons))
        for c in jax.tree_util.tree_leaves(coef)
    ]
    const = const_leaves[0][0]  # shape () or (n_neurons,)
    for c in jax.tree_util.tree_leaves(coef):
        assert jnp.allclose(c, jnp.broadcast_to(const, c.shape), rtol=1e-5)

    # normal-equation identity, verified independently of the closed-form impl
    eta = initialize_intercept_matching_mean_rate(nmo_exp, y)  # (n_out,)
    s = _row_sum(X)
    eps = jnp.finfo(s.dtype).eps
    lhs = jnp.atleast_1d(const) * (jnp.sum(s**2) + eps)
    rhs = eta * (jnp.sum(s) + eps)
    assert jnp.allclose(lhs, rhs, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("design", ["all_zero"], indirect=True)
def test_constant_coef_all_zero_design_equals_target(design):
    """With an all-zero design no constant can inject an offset; the eps
    stabilization returns c = η* and stays finite (no NaN)."""
    y = _make_y()
    coef = initialize_constant_coef_matching_mean_rate(
        nmo_exp, design, y, jnp.empty((3,))
    )
    eta = initialize_intercept_matching_mean_rate(nmo_exp, y)
    assert jnp.all(jnp.isfinite(coef))
    assert jnp.allclose(coef, eta)


@pytest.mark.parametrize("design", ["centered"], indirect=True)
def test_constant_coef_centered_columns_degrades_to_zero(design):
    """On a zero-mean design the projection scale ~0, so the constant ~0 and the
    heuristic degrades gracefully (finite) rather than blowing up."""
    y = _make_y()
    coef = initialize_constant_coef_matching_mean_rate(
        nmo_exp, design, y, jnp.empty((3,))
    )
    assert jnp.all(jnp.isfinite(coef))
    assert jnp.all(jnp.abs(coef) < 1e-2)


@pytest.mark.requires_x64
@pytest.mark.parametrize("design", ["positive"], indirect=True)
def test_constant_coef_float_dtype_matches_empty_coef(design):
    """coef inherits the dtype of empty_coef (via ones_like)."""
    y = _make_y()
    empty_coef = jnp.empty((3,), dtype=jnp.float32)
    coef = initialize_constant_coef_matching_mean_rate(nmo_exp, design, y, empty_coef)
    assert coef.dtype == empty_coef.dtype


# ---------------------------------------------------------------------------
# no-intercept fit via partition around _compute_loss (mechanism precursor)
# ---------------------------------------------------------------------------


# obs model -> (single-neuron true coef, population true coef). Magnitudes are
# modest so exp(X @ coef) stays bounded on the centered design; Gamma's coef is
# positive so the default 1/x link gives a positive mean on the positive design.
_FIT_CASES = {
    "Poisson": ([-1.0, -0.8, -1.2], [[-1.0, -0.6], [-0.8, -0.4], [-1.2, -0.9]]),
    "Gaussian": ([0.5, -0.3, 0.2], [[0.5, 0.3], [-0.3, -0.1], [0.2, 0.4]]),
    "Gamma": ([0.5, 0.3, 0.8], [[0.5, 0.4], [0.3, 0.2], [0.8, 0.6]]),
}


@pytest.fixture
def fit_models(request, design):
    """Return ``(ground_truth, empty, X, n_neurons)`` for a fit case.

    ``ground_truth`` carries the true (intercept-free) params so ``simulate`` can
    generate ``y``; ``empty`` is an unfit model of the same config, used to fit.
    Parametrized indirectly by ``(obs_model, "single"|"population")``. Gamma's 1/x
    link needs a positive mean, which the centered design violates, so on that
    design Gamma uses an ``exp`` link instead.
    """
    obs, kind = request.param
    design_kind = request.node.callspec.params["design"]
    n_neurons = None if kind == "single" else 2
    model_cls = nmo.glm.GLM if n_neurons is None else nmo.glm.PopulationGLM
    true_coef = jnp.asarray(_FIT_CASES[obs][0 if n_neurons is None else 1])

    kwargs = dict(observation_model=obs)
    if obs == "Gamma" and design_kind == "centered":
        # default link is 1/x which doesn't enforce positivity
        # and fails with centered X.
        kwargs["inverse_link_function"] = nmo_exp

    ground_truth = model_cls(**kwargs)
    ground_truth.coef_ = true_coef
    ground_truth.intercept_ = jnp.zeros(1 if n_neurons is None else n_neurons)
    ground_truth.scale_ = jnp.ones_like(ground_truth.intercept_)

    empty = model_cls(**kwargs)
    return ground_truth, empty, design, n_neurons


@pytest.mark.requires_x64
@pytest.mark.parametrize("design", ["positive", "centered"], indirect=True)
@pytest.mark.parametrize(
    "fit_models",
    [
        (obs, kind)
        for obs in ("Poisson", "Gaussian", "Gamma")
        for kind in ("single", "population")
    ],
    indirect=True,
)
def test_fit_with_frozen_intercept(setup_solver, fit_models):
    """Freezing the intercept via eqx.partition around ``_compute_loss`` and
    fitting with the nemos solver leaves the intercept at 0, converges, and
    lowers the (unregularized) loss. ``y`` is generated via ``simulate`` from an
    intercept-free ground-truth model."""
    ground_truth, model, X, n_neurons = fit_models
    p = X.shape[1]

    y, _ = ground_truth.simulate(jax.random.key(0), X)

    intercept0 = jnp.zeros(1 if n_neurons is None else n_neurons)
    empty_coef = jnp.empty((p,) if n_neurons is None else (p, n_neurons))
    init_coef = initialize_constant_coef_matching_mean_rate(
        model._inverse_link_function, X, y, empty_coef
    )
    params0 = GLMParams(coef=init_coef, intercept=intercept0)
    diff0, frozen = _freeze_intercept_partition(params0)

    # partition correctness: intercept removed from the trainable subtree
    assert diff0.intercept is None
    assert jnp.all(frozen.intercept == 0.0)

    def loss_on_diff(diff, X, y):
        return model._compute_loss(eqx.combine(diff, frozen), X, y)

    loss_init = loss_on_diff(diff0, X, y)
    solver = setup_solver(loss_on_diff, init_params=diff0, tol=1e-10)
    diff_fit, state, _ = solver.run(diff0, X, y)
    fitted = eqx.combine(diff_fit, frozen)

    # intercept never moved
    assert jnp.all(fitted.intercept == 0.0)

    # solver reports convergence (mirrors the detection in GLM.fit)
    converged = getattr(
        getattr(state, "stats", None), "converged", getattr(state, "converged", None)
    )
    assert bool(converged)

    # loss decreased
    assert loss_on_diff(diff_fit, X, y) <= loss_init
